export type DisplayMode = 'ink' | 'velocity' | 'image';
export type BrushInfo = {
  pos: [number, number];
  delta: [number, number];
  isDown: boolean;
};
export type RenderEntries = {
  result: { texture: 'float' };
  background: { texture: 'float' };
  linSampler: { sampler: 'filtering' };
};
export type SimParams = {
  dt: number;
  viscosity: number;
  jacobiIter: number;
  showField: DisplayMode;
  enableBoundary: boolean;
};
