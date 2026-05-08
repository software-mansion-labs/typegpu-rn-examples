import {
  useBindGroup,
  useBuffer,
  useConfigureContext,
  useFrame,
  useRoot,
  useUniform,
} from '@typegpu/react';
import { useMemo, useRef } from 'react';
import { useWindowDimensions } from 'react-native';
import { Canvas } from 'react-native-wgpu';
import tgpu, { d, std, type TgpuFragmentFn, type TgpuVertexFn } from 'typegpu';

const triangleAmount = 500;
const triangleSize = 0.08;

function rotate(v: d.v2f, angle: number) {
  'use gpu';
  return d.vec2f(
    v.x * std.cos(angle) - v.y * std.sin(angle),
    v.x * std.sin(angle) + v.y * std.cos(angle),
  );
}

function getRotationFromVelocity(velocity: d.v2f) {
  'use gpu';
  return -std.atan2(velocity.x, velocity.y);
}

const Boid = d.struct({
  position: d.vec2f,
  velocity: d.vec2f,
});

const renderLayout = tgpu.bindGroupLayout({
  boids: { storage: d.arrayOf(Boid), access: 'readonly' },
  colorPalette: { uniform: d.vec3f },
});

const triangleVertices = tgpu.const(d.arrayOf(d.vec2f), [
  d.vec2f(0.0, triangleSize),
  d.vec2f(-triangleSize / 2, -triangleSize / 2),
  d.vec2f(triangleSize / 2, -triangleSize / 2),
]);

function mainVert(input: TgpuVertexFn.AutoInEmpty) {
  'use gpu';

  const boid = renderLayout.$.boids[input.$instanceIndex];
  const localPos = triangleVertices.$[input.$vertexIndex];

  const angle = getRotationFromVelocity(boid.velocity);

  const pos = d.vec4f(boid.position + rotate(localPos, angle), 0, 1);
  const color = d.vec4f(
    std.sin(renderLayout.$.colorPalette + angle) * 0.45 + 0.45,
    1,
  );

  return {
    $position: pos,
    color,
  };
}

function mainFrag(input: TgpuFragmentFn.AutoIn<{ color: d.v4f }>) {
  'use gpu';
  return input.color;
}

const Params = d.struct({
  separationDistance: d.f32,
  separationStrength: d.f32,
  alignmentDistance: d.f32,
  alignmentStrength: d.f32,
  cohesionDistance: d.f32,
  cohesionStrength: d.f32,
});

const colorPresets = {
  plumTree: d.vec3f(1.0, 2.0, 1.0),
  jeans: d.vec3f(2.0, 1.5, 1.0),
  typegpu: d.vec3f(0, 0.345, 0.867),
  greyscale: d.vec3f(0, 0, 0),
  hotcold: d.vec3f(0, 3.14, 3.14),
};

const presets = {
  default: {
    separationDistance: 0.05,
    separationStrength: 0.001,
    alignmentDistance: 0.3,
    alignmentStrength: 0.01,
    cohesionDistance: 0.3,
    cohesionStrength: 0.001,
  },
  mosquitoes: {
    separationDistance: 0.02,
    separationStrength: 0.01,
    alignmentDistance: 0.0,
    alignmentStrength: 0.0,
    cohesionDistance: 0.177,
    cohesionStrength: 0.011,
  },
  blobs: {
    separationDistance: 0.033,
    separationStrength: 0.051,
    alignmentDistance: 0.047,
    alignmentStrength: 0.1,
    cohesionDistance: 0.3,
    cohesionStrength: 0.013,
  },
  particles: {
    separationDistance: 0.035,
    separationStrength: 1,
    alignmentDistance: 0.0,
    alignmentStrength: 0.0,
    cohesionDistance: 0.0,
    cohesionStrength: 0.0,
  },
  nanites: {
    separationDistance: 0.067,
    separationStrength: 0.01,
    alignmentDistance: 0.066,
    alignmentStrength: 0.021,
    cohesionDistance: 0.086,
    cohesionStrength: 0.094,
  },
} as const;

// compute

const computeLayout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  boids: { storage: d.arrayOf(Boid), access: 'readonly' },
  nextBoids: { storage: d.arrayOf(Boid), access: 'mutable' },
});

function mainCompute(boidIdx: number) {
  'use gpu';
  const params = computeLayout.$.params;
  const currentBoid = computeLayout.$.boids[boidIdx];
  const nextBoid = computeLayout.$.nextBoids[boidIdx];

  let separation = d.vec2f();
  let alignment = d.vec2f();
  let cohesion = d.vec2f();
  let alignmentCount = d.u32(0);
  let cohesionCount = d.u32(0);

  for (let i = d.u32(0); i < computeLayout.$.boids.length; i++) {
    if (i === boidIdx) {
      continue;
    }

    const other = computeLayout.$.boids[i];
    const dist = std.distance(currentBoid.position, other.position);
    if (dist < params.separationDistance) {
      separation += currentBoid.position - other.position;
    }
    if (dist < params.alignmentDistance) {
      alignment += other.velocity;
      alignmentCount++;
    }
    if (dist < params.cohesionDistance) {
      cohesion += other.position;
      cohesionCount++;
    }
  }
  if (alignmentCount > 0) {
    alignment = alignment / d.f32(alignmentCount);
  }
  if (cohesionCount > 0) {
    cohesion = cohesion / d.f32(cohesionCount) - currentBoid.position;
  }

  let newPosition = d.vec2f(currentBoid.position);
  let newVelocity = d.vec2f(currentBoid.velocity);

  newVelocity +=
    separation * params.separationStrength +
    alignment * params.alignmentStrength +
    cohesion * params.cohesionStrength;
  newVelocity =
    std.normalize(newVelocity) * std.clamp(std.length(newVelocity), 0, 0.01);

  if (newPosition[0] > 1.0 + triangleSize) {
    newPosition[0] = -1.0 - triangleSize;
  }
  if (newPosition[1] > 1.0 + triangleSize) {
    newPosition[1] = -1.0 - triangleSize;
  }
  if (newPosition[0] < -1.0 - triangleSize) {
    newPosition[0] = 1.0 + triangleSize;
  }
  if (newPosition[1] < -1.0 - triangleSize) {
    newPosition[1] = 1.0 + triangleSize;
  }
  newPosition += newVelocity;
  nextBoid.position = d.vec2f(newPosition);
  nextBoid.velocity = d.vec2f(newVelocity);
}

export default function Boids() {
  const root = useRoot();

  const paramsUniform = useUniform(Params, { initial: presets.default });
  const colorPaletteUniform = useUniform(d.vec3f, {
    initial: colorPresets.typegpu,
  });

  const initialData = useMemo(
    () =>
      Array.from({ length: triangleAmount }, () => ({
        position: [Math.random() * 2 - 1, Math.random() * 2 - 1] as [
          number,
          number,
        ],
        velocity: [Math.random() * 0.1 - 0.05, Math.random() * 0.1 - 0.05] as [
          number,
          number,
        ],
      })),
    [],
  );

  const boidBuffers = [0, 1].map(() =>
    // biome-ignore lint/correctness/useHookAtTopLevel: it's always 2 calls
    useBuffer(d.arrayOf(Boid, triangleAmount), {
      initial: initialData,
    }).$usage('storage'),
  );
  const computePipeline = useMemo(
    () => root.createGuardedComputePipeline(mainCompute),
    [root],
  );

  const renderPipeline = useMemo(
    () => root.createRenderPipeline({ vertex: mainVert, fragment: mainFrag }),
    [root],
  );

  const computeBindGroups = [0, 1].map((idx) =>
    // biome-ignore lint/correctness/useHookAtTopLevel: it's always 2 calls
    useBindGroup(computeLayout, {
      params: paramsUniform.buffer,
      boids: boidBuffers[idx],
      nextBoids: boidBuffers[1 - idx],
    }),
  );

  const renderBindGroups = [0, 1].map((idx) =>
    // biome-ignore lint/correctness/useHookAtTopLevel: it's always 2 calls
    useBindGroup(renderLayout, {
      boids: boidBuffers[idx],
      colorPalette: colorPaletteUniform.buffer,
    }),
  );

  const { ref, ctxRef } = useConfigureContext({ alphaMode: 'premultiplied' });

  const evenRef = useRef(false);
  useFrame(() => {
    if (!ctxRef.current) {
      return;
    }

    evenRef.current = !evenRef.current;

    computePipeline
      .with(computeBindGroups[evenRef.current ? 0 : 1])
      .dispatchThreads(triangleAmount);

    renderPipeline
      .withColorAttachment({ view: ctxRef.current })
      .with(renderBindGroups[evenRef.current ? 1 : 0])
      .draw(3, triangleAmount);

    ctxRef.current.present?.();
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
