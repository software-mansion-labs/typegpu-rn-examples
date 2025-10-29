import { Canvas } from 'react-native-wgpu';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';

import { useWebGPU } from '../useWebGPU.ts';

const MAX_WATER_LEVEL_UNPRESSURIZED = tgpu['~unstable'].const(d.u32, 0xff);
const MAX_WATER_LEVEL = tgpu['~unstable'].const(d.u32, (1 << 24) - 1);
const MAX_PRESSURE = tgpu['~unstable'].const(d.u32, 12);

const options = {
  size: 32,
  timestep: 25,
  stepsPerTimestep: 1,
  workgroupSize: 1,
  viscosity: 1000,
  brushSize: 0,
  brushType: 'water',
};

export default function () {
  const ref = useWebGPU(({ context, device, presentationFormat }) => {
    const root = tgpu.initFromDevice({ device });

    const sizeBuffer = root
      .createBuffer(d.vec2u)
      .$name('size')
      .$usage('uniform');
    const sizeUniform = sizeBuffer.as('uniform');

    const viscosityBuffer = root
      .createBuffer(d.u32)
      .$name('viscosity')
      .$usage('uniform');
    const viscosityUniform = viscosityBuffer.as('uniform');

    const currentStateBuffer = root
      .createBuffer(d.arrayOf(d.u32, 1024 ** 2))
      .$name('current')
      .$usage('storage', 'vertex');
    const currentStateStorage = currentStateBuffer.as('readonly');

    const nextStateBuffer = root
      .createBuffer(d.arrayOf(d.atomic(d.u32), 1024 ** 2))
      .$name('next')
      .$usage('storage');
    const nextStateStorage = nextStateBuffer.as('mutable');

    const squareBuffer = root
      .createBuffer(d.arrayOf(d.vec2f, 4), [
        d.vec2f(0, 0),
        d.vec2f(0, 1),
        d.vec2f(1, 0),
        d.vec2f(1, 1),
      ])
      .$usage('vertex')
      .$name('square');

    const getIndex = tgpu['~unstable'].fn(
      [d.u32, d.u32],
      d.u32,
    )((x, y) => {
      const h = sizeUniform.value.y;
      const w = sizeUniform.value.x;
      return (y % h) * w + (x % w);
    });

    const getCell = tgpu['~unstable'].fn(
      [d.u32, d.u32],
      d.u32,
    )((x, y) => currentStateStorage.value[getIndex(x, y)]);

    const getCellNext = tgpu['~unstable']
      .fn(
        [d.u32, d.u32],
        d.u32,
      )(/* wgsl */ ` (x: u32, y: u32) -> u32 {
        return atomicLoad(&nextStateData[getIndex(x, y)]);
      }`)
      .$uses({ nextStateData: nextStateStorage, getIndex });

    const updateCell = tgpu['~unstable']
      .fn([d.u32, d.u32, d.u32])(/* wgsl */ ` (x: u32, y: u32, value: u32) {
        atomicStore(&nextStateData[getIndex(x, y)], value);
      }`)
      .$uses({ nextStateData: nextStateStorage, getIndex });

    const addToCell = tgpu['~unstable']
      .fn([d.u32, d.u32, d.u32])(/* wgsl */ `(x: u32, y: u32, value: u32) {
        let cell = getCellNext(x, y);
        let waterLevel = cell & MAX_WATER_LEVEL;
        let newWaterLevel = min(waterLevel + value, MAX_WATER_LEVEL);
        atomicAdd(&nextStateData[getIndex(x, y)], newWaterLevel - waterLevel);
      }`)
      .$uses({
        getCellNext,
        nextStateData: nextStateStorage,
        getIndex,
        MAX_WATER_LEVEL,
      });

    const subtractFromCell = tgpu['~unstable']
      .fn([d.u32, d.u32, d.u32])(/* wgsl */ `(x: u32, y: u32, value: u32) {
        let cell = getCellNext(x, y);
        let waterLevel = cell & MAX_WATER_LEVEL;
        let newWaterLevel = max(waterLevel - min(value, waterLevel), 0u);
        atomicSub(&nextStateData[getIndex(x, y)], waterLevel - newWaterLevel);
      }`)
      .$uses({
        getCellNext,
        nextStateData: nextStateStorage,
        getIndex,
        MAX_WATER_LEVEL,
      });

    const persistFlags = tgpu['~unstable'].fn([d.u32, d.u32])((x, y) => {
      const cell = getCell(x, y);
      const waterLevel = cell & MAX_WATER_LEVEL.value;
      const flags = cell >> 24;
      updateCell(x, y, (flags << 24) | waterLevel);
    });

    const getStableStateBelow = tgpu['~unstable'].fn(
      [d.u32, d.u32],
      d.u32,
    )((upper, lower) => {
      const totalMass = upper + lower;
      if (totalMass <= MAX_WATER_LEVEL_UNPRESSURIZED.value) {
        return totalMass;
      }
      if (
        totalMass >= MAX_WATER_LEVEL_UNPRESSURIZED.value * 2 &&
        upper > lower
      ) {
        return totalMass / 2 + MAX_PRESSURE.value;
      }
      return MAX_WATER_LEVEL_UNPRESSURIZED.value;
    });

    const isWall = tgpu['~unstable'].fn(
      [d.u32, d.u32],
      d.bool,
    )((x, y) => getCell(x, y) >> 24 === 1);

    const isWaterSource = tgpu['~unstable'].fn(
      [d.u32, d.u32],
      d.bool,
    )((x, y) => getCell(x, y) >> 24 === 2);

    const isWaterDrain = tgpu['~unstable'].fn(
      [d.u32, d.u32],
      d.bool,
    )((x, y) => getCell(x, y) >> 24 === 3);

    const isClearCell = tgpu['~unstable'].fn(
      [d.u32, d.u32],
      d.bool,
    )((x, y) => getCell(x, y) >> 24 === 4);

    const getWaterLevel = tgpu['~unstable'].fn(
      [d.u32, d.u32],
      d.u32,
    )((x, y) => getCell(x, y) & MAX_WATER_LEVEL.value);

    const checkForFlagsAndBounds = tgpu['~unstable'].fn(
      [d.u32, d.u32],
      d.bool,
    )((x, y) => {
      if (isClearCell(x, y)) {
        updateCell(x, y, 0);
        return true;
      }

      if (isWall(x, y)) {
        persistFlags(x, y);
        return true;
      }

      if (isWaterSource(x, y)) {
        persistFlags(x, y);
        addToCell(x, y, 20);
        return false;
      }

      if (isWaterDrain(x, y)) {
        persistFlags(x, y);
        updateCell(x, y, 3 << 24);
        return true;
      }

      if (
        y === 0 ||
        y === sizeUniform.value.y - 1 ||
        x === 0 ||
        x === sizeUniform.value.x - 1
      ) {
        subtractFromCell(x, y, getWaterLevel(x, y));
        return true;
      }

      return false;
    });

    const decideWaterLevel = tgpu['~unstable'].fn([d.u32, d.u32])((x, y) => {
      if (checkForFlagsAndBounds(x, y)) {
        return;
      }

      let remainingWater = getWaterLevel(x, y);

      if (remainingWater === 0) {
        return;
      }

      if (!isWall(x, y - 1)) {
        const waterLevelBelow = getWaterLevel(x, y - 1);
        const stable = getStableStateBelow(remainingWater, waterLevelBelow);
        if (waterLevelBelow < stable) {
          const change = stable - waterLevelBelow;
          const flow = std.min(change, viscosityUniform.value);
          subtractFromCell(x, y, flow);
          addToCell(x, y - 1, flow);
          remainingWater -= flow;
        }
      }

      if (remainingWater === 0) {
        return;
      }

      const waterLevelBefore = remainingWater;
      if (!isWall(x - 1, y)) {
        const flowRaw =
          d.i32(waterLevelBefore) - d.i32(getWaterLevel(x - 1, y));
        if (flowRaw > 0) {
          const change = std.max(
            std.min(4, remainingWater),
            d.u32(flowRaw) / 4,
          );
          const flow = std.min(change, viscosityUniform.value);
          subtractFromCell(x, y, flow);
          addToCell(x - 1, y, flow);
          remainingWater -= flow;
        }
      }

      if (remainingWater === 0) {
        return;
      }

      if (!isWall(x + 1, y)) {
        const flowRaw =
          d.i32(waterLevelBefore) - d.i32(getWaterLevel(x + 1, y));
        if (flowRaw > 0) {
          const change = std.max(
            std.min(4, remainingWater),
            d.u32(flowRaw) / 4,
          );
          const flow = std.min(change, viscosityUniform.value);
          subtractFromCell(x, y, flow);
          addToCell(x + 1, y, flow);
          remainingWater -= flow;
        }
      }

      if (remainingWater === 0) {
        return;
      }

      if (!isWall(x, y + 1)) {
        const stable = getStableStateBelow(
          getWaterLevel(x, y + 1),
          remainingWater,
        );
        if (stable < remainingWater) {
          const flow = std.min(remainingWater - stable, viscosityUniform.value);
          subtractFromCell(x, y, flow);
          addToCell(x, y + 1, flow);
          remainingWater -= flow;
        }
      }
    });

    const vertex = tgpu['~unstable'].vertexFn({
      in: {
        squareData: d.vec2f,
        currentStateData: d.u32,
        idx: d.builtin.instanceIndex,
      },
      out: { pos: d.builtin.position, cell: d.f32 },
    })((input) => {
      const w = sizeUniform.value.x;
      const h = sizeUniform.value.y;
      const x =
        (((d.f32(input.idx % w) + input.squareData.x) / d.f32(w) - 0.5) *
          2 *
          d.f32(w)) /
        d.f32(std.max(w, h));
      const y =
        ((d.f32((input.idx - (input.idx % w)) / w + d.u32(input.squareData.y)) /
          d.f32(h) -
          0.5) *
          2 *
          d.f32(h)) /
        d.f32(std.max(w, h));
      const cellFlags = input.currentStateData >> 24;
      let cell = d.f32(input.currentStateData & 0xffffff);
      if (cellFlags === 1) {
        cell = -1;
      }
      if (cellFlags === 2) {
        cell = -2;
      }
      if (cellFlags === 3) {
        cell = -3;
      }
      return { pos: d.vec4f(x, y, 0, 1), cell };
    });

    const fragment = tgpu['~unstable'].fragmentFn({
      in: { cell: d.f32 },
      out: d.location(0, d.vec4f),
    })((input) => {
      if (input.cell === -1) {
        return d.vec4f(41 * 0.00390625, 44 * 0.00390625, 119 * 0.00390625, 1);
      }
      if (input.cell === -2) {
        return d.vec4f(0, 1, 0, 1);
      }
      if (input.cell === -3) {
        return d.vec4f(1, 0, 0, 1);
      }

      const normalized = std.min(input.cell / d.f32(0xff), 1);

      if (normalized === 0) {
        return d.vec4f();
      }

      const res = 1 / (1 + std.exp(-(normalized - 0.2) * 10));
      return std.mul(res, d.vec4f(0.34309623431, 0.37238493723, 1, 1));
    });

    const vertexInstanceLayout = tgpu.vertexLayout(
      (n: number) => d.arrayOf(d.u32, n),
      'instance',
    );
    const vertexLayout = tgpu.vertexLayout(
      (n: number) => d.arrayOf(d.vec2f, n),
      'vertex',
    );

    let drawCanvasData = new Uint32Array(options.size * options.size);

    const createSampleScene = () => {
      let middlePoint = Math.floor(options.size / 3);
      let radius = Math.floor(options.size / 16);

      for (let i = -radius; i <= radius; i++) {
        for (let j = -radius; j <= radius; j++) {
          if (i * i + j * j <= radius * radius) {
            drawCanvasData[(middlePoint + j) * options.size + middlePoint + i] =
              1 << 24;
          }
        }
      }

      for (let i = -radius; i <= radius; i++) {
        for (let j = -radius; j <= radius; j++) {
          if (i * i + j * j <= radius * radius) {
            drawCanvasData[
              (middlePoint + j) * options.size + options.size - middlePoint + i
            ] = 1 << 24;
          }
        }
      }

      radius = Math.floor(options.size / 8);
      middlePoint = Math.floor(options.size / 2);
      for (let i = -radius; i <= radius; i++) {
        for (let j = -radius; j <= radius; j++) {
          if (i * i + j * j <= radius * radius) {
            drawCanvasData[(middlePoint + j) * options.size + middlePoint + i] =
              1 << 24;
          }
        }
      }

      const smallRadius = Math.min(Math.floor(radius / 8), 6);
      for (let i = -smallRadius; i <= smallRadius; i++) {
        for (let j = -smallRadius; j <= smallRadius; j++) {
          if (i * i + j * j <= smallRadius * smallRadius) {
            drawCanvasData[
              (middlePoint + j + options.size / 4) * options.size +
                middlePoint +
                i
            ] = 2 << 24;
          }
        }
      }

      for (let i = 0; i < options.size; i++) {
        drawCanvasData[i] = 1 << 24;
      }

      for (let i = 0; i < Math.floor(options.size / 2); i++) {
        drawCanvasData[i * options.size] = 1 << 24;
      }

      for (let i = 0; i < Math.floor(options.size / 2); i++) {
        drawCanvasData[i * options.size - 1 + options.size] = 1 << 24;
      }
    };

    drawCanvasData = new Uint32Array(options.size * options.size);

    const compute = tgpu['~unstable'].computeFn({
      in: { gid: d.builtin.globalInvocationId },
      workgroupSize: [options.workgroupSize, options.workgroupSize],
    })((input) => {
      decideWaterLevel(input.gid.x, input.gid.y);
    });

    const computePipeline = root['~unstable']
      .withCompute(compute)
      .createPipeline();
    const renderPipeline = root['~unstable']
      .withVertex(vertex, {
        squareData: vertexLayout.attrib,
        currentStateData: vertexInstanceLayout.attrib,
      })
      .withFragment(fragment, {
        format: presentationFormat,
      })
      .withPrimitive({
        topology: 'triangle-strip',
      })
      .createPipeline()
      .with(vertexLayout, squareBuffer)
      .with(vertexInstanceLayout, currentStateBuffer);

    const render = () => {
      // compute
      computePipeline.dispatchWorkgroups(
        options.size / options.workgroupSize,
        options.size / options.workgroupSize,
      );

      // render
      renderPipeline
        .withColorAttachment({
          view: context.getCurrentTexture().createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: 'clear' as const,
          storeOp: 'store' as const,
        })
        .draw(4, options.size ** 2);

      root['~unstable'].flush();

      currentStateBuffer.copyFrom(nextStateBuffer);
    };

    const applyDrawCanvas = () => {
      for (let i = 0; i < options.size; i++) {
        for (let j = 0; j < options.size; j++) {
          if (drawCanvasData[j * options.size + i] === 0) {
            continue;
          }

          const index = j * options.size + i;
          root.device.queue.writeBuffer(
            nextStateBuffer.buffer,
            index * Uint32Array.BYTES_PER_ELEMENT,
            drawCanvasData,
            index,
            1,
          );
        }
      }

      drawCanvasData.fill(0);
    };

    sizeBuffer.write(d.vec2u(options.size, options.size));
    viscosityBuffer.write(options.viscosity);

    createSampleScene();
    applyDrawCanvas();
    return render;
  });

  return (
    <Canvas
      ref={ref}
      style={{
        width: '100%',
        aspectRatio: 1,
      }}
      transparent
    />
  );
}
