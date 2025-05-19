import { useWebGPU } from '../../useWebGPU';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { IcosphereGenerator } from './icosphere';
import {
  Camera,
  CubeVertex,
  DirectionalLight,
  Material,
  Vertex,
} from './dataTypes';
import { type CubemapNames, cubeVertices, loadCubemap } from './cubemap';
import * as m from 'wgpu-matrix';
import { Canvas } from 'react-native-wgpu';
import {
  rotationSpeed,
  materialProps,
  smoothNormals,
  subdivisions,
} from './params';

export default function () {
  const ref = useWebGPU(async ({ context, device, presentationFormat }) => {
    const root = tgpu.initFromDevice({ device });

    const canvas = context.canvas;
    context.configure({
      device: root.device,
      format: presentationFormat,
      alphaMode: 'premultiplied',
    });

    // Geometry & Material Setup

    const icosphereGenerator = new IcosphereGenerator(root);
    const vertexBuffer = icosphereGenerator.createIcosphere(
      subdivisions,
      smoothNormals,
    );
    const cubeVertexBuffer = root
      .createBuffer(d.arrayOf(CubeVertex, cubeVertices.length), cubeVertices)
      .$usage('vertex');

    // Camera Setup

    const cameraInitialPos = d.vec3f(0, 1, 5);
    const cameraBuffer = root
      .createBuffer(Camera, {
        view: m.mat4.lookAt(
          cameraInitialPos,
          [0, 0, 0],
          [0, 1, 0],
          d.mat4x4f(),
        ),
        projection: m.mat4.perspective(
          Math.PI / 4,
          canvas.width / canvas.height,
          0.1,
          10000,
          d.mat4x4f(),
        ),
        position: d.vec4f(cameraInitialPos, 1),
      })
      .$usage('uniform');

    // Light & Material Buffers

    const lightBuffer = root
      .createBuffer(DirectionalLight, {
        direction: d.vec3f(1, 1, 5),
        color: d.vec3f(1, 1, 1),
        intensity: 1,
      })
      .$usage('uniform');

    const materialBuffer = root
      .createBuffer(Material, materialProps)
      .$usage('uniform');

    const transformBuffer = root
      .createBuffer(d.mat4x4f, m.mat4.identity(d.mat4x4f()))
      .$usage('uniform');

    // Textures & Samplers

    const chosenCubemap: CubemapNames = 'campsite';
    const cubemapTexture = await loadCubemap(root, chosenCubemap);
    const cubemap = cubemapTexture.createView('sampled', { dimension: 'cube' });
    const sampler = tgpu['~unstable'].sampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });

    // Bind Groups & Layouts

    const renderLayout = tgpu.bindGroupLayout({
      camera: { uniform: Camera },
      light: { uniform: DirectionalLight },
      material: { uniform: Material },
      transform: { uniform: d.mat4x4f },
    });
    const { camera, light, material, transform } = renderLayout.bound;

    const renderBindGroup = root.createBindGroup(renderLayout, {
      camera: cameraBuffer,
      light: lightBuffer,
      material: materialBuffer,
      transform: transformBuffer,
    });

    const textureLayout = tgpu.bindGroupLayout({
      cubemap: { texture: 'float', viewDimension: 'cube' },
      texSampler: { sampler: 'filtering' },
    });
    const { cubemap: cubemapBinding, texSampler } = textureLayout.bound;

    const textureBindGroup = root.createBindGroup(textureLayout, {
      cubemap,
      texSampler: sampler,
    });

    const vertexLayout = tgpu.vertexLayout((n: number) =>
      d.disarrayOf(Vertex, n),
    );
    const cubeVertexLayout = tgpu.vertexLayout((n: number) =>
      d.arrayOf(CubeVertex, n),
    );

    // Shader Functions

    const vertexFn = tgpu['~unstable'].vertexFn({
      in: {
        position: d.vec4f,
        normal: d.vec4f,
      },
      out: {
        pos: d.builtin.position,
        normal: d.vec4f,
        worldPos: d.vec4f,
      },
    })((input) => ({
      pos: std.mul(
        camera.value.projection,
        std.mul(camera.value.view, std.mul(transform.value, input.position)),
      ),
      normal: std.mul(transform.value, input.normal),
      worldPos: std.mul(transform.value, input.position),
    }));

    const fragmentFn = tgpu['~unstable'].fragmentFn({
      in: {
        normal: d.vec4f,
        worldPos: d.vec4f,
      },
      out: d.vec4f,
    })((input) => {
      const normalizedNormal = std.normalize(input.normal.xyz);
      const normalizedLightDir = std.normalize(light.value.direction);

      const ambientLight = std.mul(
        material.value.ambient,
        std.mul(light.value.intensity, light.value.color),
      );

      const diffuseFactor = std.max(
        std.dot(normalizedNormal, normalizedLightDir),
        0.0,
      );
      const diffuseLight = std.mul(
        diffuseFactor,
        std.mul(
          material.value.diffuse,
          std.mul(light.value.intensity, light.value.color),
        ),
      );

      const viewDirection = std.normalize(
        std.sub(camera.value.position.xyz, input.worldPos.xyz),
      );
      const reflectionDirection = std.reflect(
        std.neg(normalizedLightDir),
        normalizedNormal,
      );

      const specularFactor = std.pow(
        std.max(std.dot(viewDirection, reflectionDirection), 0.0),
        material.value.shininess,
      );
      const specularLight = std.mul(
        specularFactor,
        std.mul(
          material.value.specular,
          std.mul(light.value.intensity, light.value.color),
        ),
      );

      const reflectionVector = std.reflect(
        std.neg(viewDirection),
        normalizedNormal,
      );
      const environmentColor = std.textureSample(
        cubemapBinding,
        texSampler,
        reflectionVector,
      );

      const directLighting = std.add(
        ambientLight,
        std.add(diffuseLight, specularLight),
      );

      const finalColor = std.mix(
        directLighting,
        environmentColor.xyz,
        material.value.reflectivity,
      );

      return d.vec4f(finalColor, 1.0);
    });

    const cubeVertexFn = tgpu['~unstable'].vertexFn({
      in: {
        position: d.vec3f,
        uv: d.vec2f,
      },
      out: {
        pos: d.builtin.position,
        texCoord: d.vec3f,
      },
    })((input) => {
      const viewPos = std.mul(
        camera.value.view,
        d.vec4f(input.position.xyz, 0),
      ).xyz;

      return {
        pos: std.mul(camera.value.projection, d.vec4f(viewPos, 1)),
        texCoord: input.position.xyz,
      };
    });

    const cubeFragmentFn = tgpu['~unstable'].fragmentFn({
      in: {
        texCoord: d.vec3f,
      },
      out: d.vec4f,
    })((input) => {
      return std.textureSample(
        cubemapBinding,
        texSampler,
        std.normalize(input.texCoord),
      );
    });

    // Pipeline Setup

    const cubePipeline = root['~unstable']
      .withVertex(cubeVertexFn, cubeVertexLayout.attrib)
      .withFragment(cubeFragmentFn, { format: presentationFormat })
      .withPrimitive({
        cullMode: 'front',
      })
      .createPipeline();

    const pipeline = root['~unstable']
      .withVertex(vertexFn, vertexLayout.attrib)
      .withFragment(fragmentFn, { format: presentationFormat })
      .withPrimitive({
        cullMode: 'back',
      })
      .createPipeline();

    // Render Functions
    let lastTimestamp: DOMHighResTimeStamp = 0;
    let modelTransform = m.mat4.identity(d.mat4x4f());

    function render(timestamp: DOMHighResTimeStamp) {
      const deltaTime = (timestamp - lastTimestamp) / 1000;
      lastTimestamp = timestamp;

      const rotation = rotationSpeed * deltaTime * Math.PI * 2;
      const rotationMatrix = m.mat4.rotateY(
        m.mat4.identity(),
        rotation,
        d.mat4x4f(),
      );
      modelTransform = m.mat4.multiply(
        modelTransform,
        rotationMatrix,
        d.mat4x4f(),
      );
      transformBuffer.write(modelTransform);

      cubePipeline
        .withColorAttachment({
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        })
        .with(cubeVertexLayout, cubeVertexBuffer)
        .with(renderLayout, renderBindGroup)
        .with(textureLayout, textureBindGroup)
        .draw(cubeVertices.length);

      pipeline
        .withColorAttachment({
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 },
          loadOp: 'load',
          storeOp: 'store',
        })
        .with(vertexLayout, vertexBuffer)
        .with(renderLayout, renderBindGroup)
        .with(textureLayout, textureBindGroup)
        .draw(vertexBuffer.dataType.elementCount);

      root['~unstable'].flush();
    }

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
