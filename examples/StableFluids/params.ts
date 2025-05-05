import * as d from 'typegpu/data';
import type { SimParams } from './types.ts';

export const [WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y] = [16, 16];
export const FORCE_SCALE = 0.6;
export const INK_AMOUNT = 0.02;
export const SIMULATION_QUALITY = 0.2;
export const DISPLACEMENT_SCALE = 0.005;

export const params: SimParams = {
  dt: 0.6,
  viscosity: 0.00001,
  jacobiIter: 10,
  showField: 'ink',
  enableBoundary: true,
};

export const Params = d.struct({
  dt: d.f32,
  viscosity: d.f32,
  enableBoundary: d.u32,
});

export const BrushParams = d.struct({
  pos: d.vec2i,
  delta: d.vec2f,
  radius: d.f32,
  forceScale: d.f32,
  inkAmount: d.f32,
});
