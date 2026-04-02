import tgpu from 'typegpu';
import * as d from 'typegpu/data';

// schemas

export type Line3 = d.Infer<typeof Line3>;
export const Line3 = d.struct({
  /**
   * A point on the line
   */
  origin: d.vec3f,
  /**
   * Normalized direction along the line
   */
  dir: d.vec3f,
});

export const Camera = d.struct({
  position: d.vec4f,
  targetPos: d.vec4f,
  view: d.mat4x4f,
  projection: d.mat4x4f,
});

export const ModelData = d.struct({
  position: d.vec3f,
  direction: d.vec3f, // in case of the fish, this is also the velocity
  scale: d.f32,
  applySeaFog: d.u32, // bool
  applySeaDesaturation: d.u32, // bool
});

export const ModelDataArray = d.arrayOf(ModelData);

export const ModelVertexInput = {
  modelPosition: d.vec3f,
  modelNormal: d.vec3f,
  textureUV: d.vec2f,
} as const;

export const ModelVertexOutput = {
  worldPosition: d.vec3f,
  worldNormal: d.vec3f,
  canvasPosition: d.builtin.position,
  textureUV: d.vec2f,
  applySeaFog: d.interpolate('flat', d.u32), // bool
  applySeaDesaturation: d.interpolate('flat', d.u32), // bool
} as const;

// layouts

export const modelVertexLayout = tgpu.vertexLayout(
  d.arrayOf(d.struct(ModelVertexInput)),
);

export const renderInstanceLayout = tgpu.vertexLayout(
  ModelDataArray,
  'instance',
);

export const renderBindGroupLayout = tgpu.bindGroupLayout({
  modelData: { storage: ModelDataArray },
  modelTexture: { texture: d.texture2d() },
  camera: { uniform: Camera },
  sampler: { sampler: 'filtering' },
});

export const computeBindGroupLayout = tgpu.bindGroupLayout({
  currentFishData: { storage: ModelDataArray },
  nextFishData: {
    storage: ModelDataArray,
    access: 'mutable',
  },
  timePassed: { uniform: d.u32 },
});
