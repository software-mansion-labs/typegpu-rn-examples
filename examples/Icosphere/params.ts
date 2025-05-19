import * as d from 'typegpu/data';

export const rotationSpeed = 0.1;
export const smoothNormals = false;
export const subdivisions = 2;
export const materialProps = {
  shininess: 32,
  reflectivity: 0.7,
  ambient: d.vec3f(0.1, 0.1, 0.1),
  diffuse: d.vec3f(0.3, 0.3, 0.3),
  specular: d.vec3f(0.8, 0.8, 0.8),
};
