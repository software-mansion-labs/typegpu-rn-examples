import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import * as p from './params.ts';
import { computeBindGroupLayout as layout } from './schemas.ts';

export const computeShader = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [p.workGroupSize],
})((input) => {
  'use gpu';
  const fishIndex = input.gid.x;
  const fishData = layout.$.currentFishData[fishIndex];
  let separation = d.vec3f();
  let alignment = d.vec3f();
  let alignmentCount = 0;
  let cohesion = d.vec3f();
  let cohesionCount = 0;
  let wallRepulsion = d.vec3f();

  for (let i = 0; i < p.fishAmount; i += 1) {
    if (d.u32(i) === fishIndex) {
      continue;
    }

    const other = layout.$.currentFishData[i];
    const dist = std.distance(fishData.position, other.position);
    if (dist < p.fishSeparationDistance) {
      separation += fishData.position - other.position;
    }
    if (dist < p.fishAlignmentDistance) {
      alignment = alignment + other.direction;
      alignmentCount = alignmentCount + 1;
    }
    if (dist < p.fishCohesionDistance) {
      cohesion = cohesion + other.position;
      cohesionCount = cohesionCount + 1;
    }
  }
  if (alignmentCount > 0) {
    alignment = alignment / alignmentCount;
  }
  if (cohesionCount > 0) {
    cohesion = cohesion / cohesionCount - fishData.position;
  }
  for (const i of tgpu.unroll([0, 1, 2])) {
    const repulsion = d.vec3f();
    repulsion[i] = 1;

    const axisAquariumSize = p.aquariumSize[i] * 0.5;
    const axisPosition = fishData.position[i];
    const distance = p.fishWallRepulsionDistance;

    if (axisPosition > axisAquariumSize - distance) {
      const str = axisPosition - (axisAquariumSize - distance);
      wallRepulsion = wallRepulsion - repulsion * str;
    }

    if (axisPosition < -axisAquariumSize + distance) {
      const str = -axisAquariumSize + distance - axisPosition;
      wallRepulsion = wallRepulsion + repulsion * str;
    }
  }

  let direction = d.vec3f(fishData.direction);

  direction += separation * p.fishSeparationStrength;
  direction += alignment * p.fishAlignmentStrength;
  direction += cohesion * p.fishCohesionStrength;
  direction += wallRepulsion * p.fishWallRepulsionStrength;
  direction =
    std.normalize(direction) *
    std.clamp(std.length(fishData.direction), 0, 0.01);

  const translation = direction * (std.min(999, layout.$.timePassed) / 8);

  const nextFishData = layout.$.nextFishData[fishIndex];
  nextFishData.position = fishData.position + translation;
  nextFishData.direction = d.vec3f(direction);
});
