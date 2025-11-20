import { randf } from '@typegpu/noise';
import { useWindowDimensions } from 'react-native';
import { Canvas } from 'react-native-wgpu';
import tgpu, { type TgpuBufferMutable, type TgpuBufferReadonly } from 'typegpu';
import { fullScreenTriangle } from 'typegpu/common';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';

import { useWebGPU } from '../useWebGPU.ts';

const MAX_GRID_SIZE = 1024;

type GridData = typeof GridData;
/**
 * x - velocity.x
 * y - velocity.y
 * z - density
 * w - <unused>
 */
const GridData = d.arrayOf(d.vec4f, MAX_GRID_SIZE ** 2);

type BoxObstacle = typeof BoxObstacle;
const BoxObstacle = d.struct({
  center: d.vec2u,
  size: d.vec2u,
  enabled: d.u32,
});

const gridSize = 128;

export default function () {
  const ref = useWebGPU(({ context, device, presentationFormat }) => {
    const root = tgpu.initFromDevice({ device });

    const gridSizeBuffer = root.createBuffer(d.i32).$usage('uniform');
    const gridSizeUniform = gridSizeBuffer.as('uniform');

    const gridAlphaBuffer = root.createBuffer(GridData).$usage('storage');
    const gridBetaBuffer = root.createBuffer(GridData).$usage('storage');

    const inputGridSlot = tgpu.slot<
      TgpuBufferReadonly<GridData> | TgpuBufferMutable<GridData>
    >();
    const outputGridSlot = tgpu.slot<TgpuBufferMutable<GridData>>();

    const MAX_OBSTACLES = 4;

    const prevObstaclesBuffer = root
      .createBuffer(d.arrayOf(BoxObstacle, MAX_OBSTACLES))
      .$usage('storage');

    const prevObstacleReadonly = prevObstaclesBuffer.as('readonly');

    const obstaclesBuffer = root
      .createBuffer(d.arrayOf(BoxObstacle, MAX_OBSTACLES))
      .$usage('storage');

    const obstaclesReadonly = obstaclesBuffer.as('readonly');

    const isValidCoord = tgpu.fn(
      [d.i32, d.i32],
      d.bool,
    )(
      (x, y) =>
        x < gridSizeUniform.value &&
        x >= 0 &&
        y < gridSizeUniform.value &&
        y >= 0,
    );

    const coordsToIndex = tgpu.fn(
      [d.i32, d.i32],
      d.i32,
    )((x, y) => x + y * gridSizeUniform.value);

    const getCell = tgpu.fn(
      [d.i32, d.i32],
      d.vec4f,
    )((x, y) => inputGridSlot.value[coordsToIndex(x, y)]);

    const setCell = tgpu.fn([d.i32, d.i32, d.vec4f])((x, y, value) => {
      const index = coordsToIndex(x, y);
      outputGridSlot.value[index] = value;
    });

    const setVelocity = tgpu.fn([d.i32, d.i32, d.vec2f])((x, y, velocity) => {
      const index = coordsToIndex(x, y);
      outputGridSlot.value[index].x = velocity.x;
      outputGridSlot.value[index].y = velocity.y;
    });

    const addDensity = tgpu.fn([d.i32, d.i32, d.f32])((x, y, density) => {
      const index = coordsToIndex(x, y);
      outputGridSlot.value[index].z = inputGridSlot.value[index].z + density;
    });

    const flowFromCell = tgpu.fn(
      [d.i32, d.i32, d.i32, d.i32],
      d.f32,
    )((my_x, my_y, x, y) => {
      if (!isValidCoord(x, y)) {
        return 0;
      }
      const src = getCell(x, y);

      const destPos = d.vec2i(x + d.i32(src.x), y + d.i32(src.y));
      const dest = getCell(destPos.x, destPos.y);
      const diff = src.z - dest.z;
      let outFlow = std.min(std.max(0.01, 0.3 + diff * 0.1), src.z);

      if (std.length(src.xy) < 0.5) {
        outFlow = 0;
      }

      if (my_x === x && my_y === y) {
        // 'src.z - outFlow' is how much is left in the src
        return src.z - outFlow;
      }

      if (destPos.x === my_x && destPos.y === my_y) {
        return outFlow;
      }

      return 0;
    });

    const timeUniform = root.createUniform(d.f32);

    const isInsideObstacle = tgpu.fn(
      [d.i32, d.i32],
      d.bool,
    )((x, y) => {
      for (let obs_idx = 0; obs_idx < MAX_OBSTACLES; obs_idx += 1) {
        const obs = obstaclesReadonly.value[obs_idx];

        if (obs.enabled === 0) {
          continue;
        }

        const min_x = std.max(0, d.i32(obs.center.x) - d.i32(obs.size.x / 2));
        const max_x = std.min(
          d.i32(gridSize),
          d.i32(obs.center.x) + d.i32(obs.size.x / 2),
        );
        const min_y = std.max(0, d.i32(obs.center.y) - d.i32(obs.size.y / 2));
        const max_y = std.min(
          d.i32(gridSize),
          d.i32(obs.center.y) + d.i32(obs.size.y / 2),
        );

        if (x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
          return true;
        }
      }

      return false;
    });

    const isValidFlowOut = tgpu.fn(
      [d.i32, d.i32],
      d.bool,
    )((x, y) => {
      if (!isValidCoord(x, y)) {
        return false;
      }

      if (isInsideObstacle(x, y)) {
        return false;
      }

      return true;
    });

    const computeVelocity = tgpu.fn(
      [d.i32, d.i32],
      d.vec2f,
    )((x, y) => {
      const gravityCost = d.f32(0.5);
      const neighborOffsets = d.arrayOf(
        d.vec2i,
        4,
      )([d.vec2i(0, 1), d.vec2i(0, -1), d.vec2i(1, 0), d.vec2i(-1, 0)]);

      const cell = getCell(x, y);
      let leastCost = cell.z;

      const dirChoices = d.arrayOf(d.vec2f, 4)();
      let dirChoiceCount = d.u32(1);

      for (let i = 0; i < 4; i++) {
        const offset = neighborOffsets[i];
        const neighborDensity = getCell(x + offset.x, y + offset.y).z;
        const cost = neighborDensity + d.f32(offset.y) * gravityCost;
        const isValidFlowOutVal = isValidFlowOut(x + offset.x, y + offset.y);

        if (!isValidFlowOutVal) {
          continue;
        }

        if (cost === leastCost) {
          // another valid direction
          dirChoices[dirChoiceCount] = d.vec2f(offset);
          dirChoiceCount += 1;
        } else if (cost < leastCost) {
          // new best choice
          leastCost = cost;
          dirChoices[0] = d.vec2f(offset);
          dirChoiceCount = d.u32(1);
        }
      }

      const leastCostDir =
        dirChoices[d.u32(randf.sample() * d.f32(dirChoiceCount))];
      return leastCostDir;
    });

    const mainInitWorld = tgpu['~unstable'].computeFn({
      in: { gid: d.builtin.globalInvocationId },
      workgroupSize: [1],
    })((input) => {
      const x = d.i32(input.gid.x);
      const y = d.i32(input.gid.y);
      const index = coordsToIndex(x, y);

      let value = d.vec4f();

      if (!isValidFlowOut(x, y)) {
        value = d.vec4f(0, 0, 0, 0);
      } else {
        // Ocean
        if (y < d.i32(gridSizeUniform.value / 2)) {
          const depth = 1 - d.f32(y) / d.f32(gridSizeUniform.value / 2);
          value = d.vec4f(0, 0, 10 + depth * 10, 0);
        }
      }

      outputGridSlot.value[index] = value;
    });

    const mainMoveObstacles = tgpu['~unstable'].computeFn({
      workgroupSize: [1],
    })(() => {
      for (let obsIdx = 0; obsIdx < MAX_OBSTACLES; obsIdx += 1) {
        const obs = prevObstacleReadonly.value[obsIdx];
        const nextObs = obstaclesReadonly.value[obsIdx];

        if (obs.enabled === 0) {
          continue;
        }

        const diff = std.sub(
          d.vec2i(d.i32(nextObs.center.x), d.i32(nextObs.center.y)),
          d.vec2i(d.i32(obs.center.x), d.i32(obs.center.y)),
        );

        const min_x = std.max(0, d.i32(obs.center.x) - d.i32(obs.size.x / 2));
        const max_x = std.min(
          d.i32(gridSize),
          d.i32(obs.center.x) + d.i32(obs.size.x / 2),
        );
        const min_y = std.max(0, d.i32(obs.center.y) - d.i32(obs.size.y / 2));
        const max_y = std.min(
          d.i32(gridSize),
          d.i32(obs.center.y) + d.i32(obs.size.y / 2),
        );

        const nextMinX = std.max(
          0,
          d.i32(nextObs.center.x) - d.i32(obs.size.x / 2),
        );
        const nextMaxX = std.min(
          d.i32(gridSize),
          d.i32(nextObs.center.x) + d.i32(obs.size.x / 2),
        );
        const nextMinY = std.max(
          0,
          d.i32(nextObs.center.y) - d.i32(obs.size.y / 2),
        );
        const nextMaxY = std.min(
          d.i32(gridSize),
          d.i32(nextObs.center.y) + d.i32(obs.size.y / 2),
        );

        // does it move right
        if (diff.x > 0) {
          for (let y = min_y; y <= max_y; y += 1) {
            let rowDensity = d.f32(0);
            for (let x = max_x; x <= nextMaxX; x += 1) {
              const cell = getCell(x, y);
              rowDensity += cell.z;
              cell.z = 0;
              setCell(x, y, cell);
            }

            addDensity(nextMaxX + 1, y, rowDensity);
          }
        }

        // does it move left
        if (diff.x < 0) {
          for (let y = min_y; y <= max_y; y += 1) {
            let rowDensity = d.f32(0);
            for (let x = nextMinX; x < min_x; x += 1) {
              const cell = getCell(x, y);
              rowDensity += cell.z;
              cell.z = 0;
              setCell(x, y, cell);
            }

            addDensity(nextMinX - 1, y, rowDensity);
          }
        }

        // does it move up
        if (diff.y > 0) {
          for (let x = min_x; x <= max_x; x += 1) {
            let colDensity = d.f32(0);
            for (let y = max_y; y <= nextMaxY; y += 1) {
              const cell = getCell(x, y);
              colDensity += cell.z;
              cell.z = 0;
              setCell(x, y, cell);
            }

            addDensity(x, nextMaxY + 1, colDensity);
          }
        }

        // does it move down
        for (let x = min_x; x <= max_x; x += 1) {
          let colDensity = d.f32(0);
          for (let y = nextMinY; y < min_y; y += 1) {
            const cell = getCell(x, y);
            colDensity += cell.z;
            cell.z = 0;
            setCell(x, y, cell);
          }

          addDensity(x, nextMinY - 1, colDensity);
        }

        // Recompute velocity around the obstacle so that no cells end up inside it on the next tick.

        // left column
        for (let y = nextMinY; y <= nextMaxY; y += 1) {
          const newVel = computeVelocity(nextMinX - 1, y);
          setVelocity(nextMinX - 1, y, newVel);
        }

        // right column
        for (
          let y = std.max(1, nextMinY);
          y <= std.min(nextMaxY, gridSize - 2);
          y += 1
        ) {
          const newVel = computeVelocity(nextMaxX + 2, y);
          setVelocity(nextMaxX + 2, y, newVel);
        }
      }
    });

    const sourceIntensity = 0.2;
    const sourceRadius = 0.05;

    const sourceParamsBuffer = root
      .createBuffer(
        d.struct({
          center: d.vec2f,
          radius: d.f32,
          intensity: d.f32,
        }),
      )
      .$usage('uniform');
    const sourceParamsUniform = sourceParamsBuffer.as('uniform');

    const getMinimumInFlow = tgpu['~unstable'].fn(
      [d.i32, d.i32],
      d.f32,
    )((x, y) => {
      const gridSizeF = d.f32(gridSizeUniform.value);
      const sourceRadius = std.max(
        1,
        sourceParamsUniform.value.radius * gridSizeF,
      );
      const sourcePos = d.vec2f(
        sourceParamsUniform.value.center.x * gridSizeF,
        sourceParamsUniform.value.center.y * gridSizeF,
      );

      if (
        std.length(d.vec2f(d.f32(x) - sourcePos.x, d.f32(y) - sourcePos.y)) <
        sourceRadius
      ) {
        return sourceParamsUniform.value.intensity;
      }

      return 0;
    });

    const mainCompute = tgpu['~unstable'].computeFn({
      in: { gid: d.builtin.globalInvocationId },
      workgroupSize: [8, 8],
    })((input) => {
      const x = d.i32(input.gid.x);
      const y = d.i32(input.gid.y);
      const index = coordsToIndex(x, y);

      randf.seed2(d.vec2f(d.f32(index)).mul(timeUniform.value));

      const next = getCell(x, y);
      const nextVelocity = computeVelocity(x, y);
      next.x = nextVelocity.x;
      next.y = nextVelocity.y;

      // Processing in-flow

      next.z = flowFromCell(x, y, x, y);
      next.z += flowFromCell(x, y, x, y + 1);
      next.z += flowFromCell(x, y, x, y - 1);
      next.z += flowFromCell(x, y, x + 1, y);
      next.z += flowFromCell(x, y, x - 1, y);

      const min_inflow = getMinimumInFlow(x, y);
      next.z = std.max(min_inflow, next.z);

      outputGridSlot.value[index] = next;
    });

    const OBSTACLE_BOX = 0;
    const OBSTACLE_LEFT_WALL = 1;

    const obstacles: {
      x: number;
      y: number;
      width: number;
      height: number;
      enabled: boolean;
    }[] = [
      { x: 0.5, y: 0.5, width: 0.15, height: 0.5, enabled: true }, // box
      { x: 0, y: 0.5, width: 0.05, height: 1, enabled: true }, // left wall
      { x: 1, y: 0.5, width: 0.07, height: 1, enabled: true }, // right wall
      { x: 0.5, y: 0, width: 1, height: 0.1, enabled: true }, // floor
    ];

    function obstaclesToConcrete(): d.Infer<BoxObstacle>[] {
      return obstacles.map(({ x, y, width, height, enabled }) => ({
        center: d.vec2u(Math.round(x * gridSize), Math.round(y * gridSize)),
        size: d.vec2u(
          Math.round(width * gridSize),
          Math.round(height * gridSize),
        ),
        enabled: enabled ? 1 : 0,
      }));
    }

    const boxX = 0.5;
    const limitedBoxX = () => {
      const leftWallWidth = obstacles[OBSTACLE_LEFT_WALL].width;
      return Math.max(boxX, leftWallX + leftWallWidth / 2 + 0.15);
    };
    const leftWallX = 0;

    const fragmentMain = tgpu['~unstable'].fragmentFn({
      in: { uv: d.vec2f },
      out: d.vec4f,
    })((input) => {
      const x = d.i32(input.uv.x * d.f32(gridSizeUniform.value));
      const y = d.i32((1 - input.uv.y) * d.f32(gridSizeUniform.value));

      const index = coordsToIndex(x, y);
      const cell = inputGridSlot.value[index];
      const density = std.max(0, cell.z);

      const obstacleColor = d.vec4f(
        41 * 0.00390625,
        44 * 0.00390625,
        119 * 0.00390625,
        1,
      );

      const background = d.vec4f(0);
      const third_color = d.vec4f(
        82.0 * 0.00390625,
        89.0 * 0.00390625,
        238.0 * 0.00390625,
        1,
      );
      const second_color = d.vec4f(
        133.0 * 0.00390625,
        138.0 * 0.00390625,
        243.0 * 0.00390625,
        1,
      );
      const first_color = d.vec4f(
        185.0 * 0.00390625,
        188.0 * 0.00390625,
        248.0 * 0.00390625,
        1,
      );

      const firstThreshold = d.f32(2);
      const secondThreshold = d.f32(10);
      const thirdThreshold = d.f32(20);

      if (isInsideObstacle(x, y)) {
        return obstacleColor;
      }

      if (density <= 0) {
        return background;
      }

      if (density <= firstThreshold) {
        const t = 1 - std.pow(1 - density / firstThreshold, 2);
        return std.mix(background, first_color, t);
      }

      if (density <= secondThreshold) {
        return std.mix(
          first_color,
          second_color,
          (density - firstThreshold) / (secondThreshold - firstThreshold),
        );
      }

      return std.mix(
        second_color,
        third_color,
        std.min((density - secondThreshold) / thirdThreshold, 1),
      );
    });

    function makePipelines(
      inputGridReadonly: TgpuBufferReadonly<GridData>,
      outputGridMutable: TgpuBufferMutable<GridData>,
    ) {
      const initWorldPipeline = root['~unstable']
        .with(inputGridSlot, outputGridMutable)
        .with(outputGridSlot, outputGridMutable)
        .withCompute(mainInitWorld)
        .createPipeline();

      const computePipeline = root['~unstable']
        .with(inputGridSlot, inputGridReadonly)
        .with(outputGridSlot, outputGridMutable)
        .withCompute(mainCompute)
        .createPipeline();

      const moveObstaclesPipeline = root['~unstable']
        .with(inputGridSlot, outputGridMutable)
        .with(outputGridSlot, outputGridMutable)
        .withCompute(mainMoveObstacles)
        .createPipeline();

      const renderPipeline = root['~unstable']
        .with(inputGridSlot, inputGridReadonly)
        .withVertex(fullScreenTriangle, {})
        .withFragment(fragmentMain, { format: presentationFormat })
        .withPrimitive({ topology: 'triangle-strip' })
        .createPipeline();

      return {
        init() {
          initWorldPipeline.dispatchWorkgroups(gridSize, gridSize);
        },

        applyMovedObstacles(bufferData: d.Infer<BoxObstacle>[]) {
          obstaclesBuffer.write(bufferData);
          moveObstaclesPipeline.dispatchWorkgroups(1);

          prevObstaclesBuffer.write(bufferData);
        },

        compute() {
          computePipeline.dispatchWorkgroups(
            gridSize / mainCompute.shell.workgroupSize[0],
            gridSize / mainCompute.shell.workgroupSize[1],
          );
        },

        render() {
          const textureView = context.getCurrentTexture().createView();

          renderPipeline
            .withColorAttachment({
              view: textureView,
              clearValue: [1, 1, 1, 1],
              loadOp: 'clear',
              storeOp: 'store',
            })
            .draw(4);
        },
      };
    }

    const even = makePipelines(
      // in
      gridAlphaBuffer.as('readonly'),
      // out
      gridBetaBuffer.as('mutable'),
    );

    const odd = makePipelines(
      // in
      gridBetaBuffer.as('readonly'),
      // out
      gridAlphaBuffer.as('mutable'),
    );

    let primary = even;

    gridSizeBuffer.write(gridSize);
    obstaclesBuffer.write(obstaclesToConcrete());
    prevObstaclesBuffer.write(obstaclesToConcrete());
    primary.init();

    let frameNum = 1;

    const frame = () => {
      frameNum++;

      const time = Date.now() % 1000;
      timeUniform.write(time);
      obstacles[OBSTACLE_BOX].x =
        0.5 + 0.1 * Math.cos((2 * Math.PI * frameNum) / 200);
      primary.applyMovedObstacles(obstaclesToConcrete());

      sourceParamsBuffer.write({
        center: d.vec2f(0.5, 0.9),
        intensity: sourceIntensity,
        radius: sourceRadius,
      });

      primary = primary === even ? odd : even;
      primary.compute();
      primary.render();
    };

    return frame;
  });

  const { width, height } = useWindowDimensions();

  return (
    <Canvas
      ref={ref}
      style={{
        width: width > height ? undefined : '100%',
        height: width > height ? '100%' : undefined,
        aspectRatio: 1,
      }}
      transparent
    />
  );
}
