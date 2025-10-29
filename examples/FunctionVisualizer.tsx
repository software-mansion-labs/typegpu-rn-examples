import { PixelRatio } from 'react-native';
import { Canvas } from 'react-native-wgpu';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { mat4 } from 'wgpu-matrix';
import { useWebGPU } from '../useWebGPU.ts';

export default function () {
  const ref = useWebGPU(({ context, device, presentationFormat, canvas }) => {
    const root = tgpu.initFromDevice({ device });

    // Globals and init

    const initialFunctions: Array<{
      name: string;
      color: d.v4f;
      code: string;
    }> = [
      {
        name: 'blue function',
        color: fromHex('#1D72F0'),
        code: 'x',
      },
      {
        name: 'green function',
        color: fromHex('#3CB371'),
        code: 'cos(x*5)/3-x',
      },
      {
        name: 'purple function',
        color: fromHex('#C464FF'),
        code: 'x*sin(log(abs(x)))',
      },
    ];

    const Properties = d.struct({
      transformation: d.mat4x4f,
      inverseTransformation: d.mat4x4f,
      interpolationPoints: d.u32,
      lineWidth: d.f32,
    });

    const properties = Properties({
      transformation: mat4.identity(d.mat4x4f()),
      inverseTransformation: mat4.identity(d.mat4x4f()),
      interpolationPoints: 256,
      lineWidth: 0.04,
    });

    // Buffers

    const propertiesBuffer = root
      .createBuffer(Properties, properties)
      .$usage('uniform');

    // these buffers are recreated with a different size on interpolationPoints change
    function createLineVerticesBuffers() {
      const Schema = d.arrayOf(d.vec2f, properties.interpolationPoints);
      return initialFunctions.map(() =>
        root.createBuffer(Schema).$usage('storage'),
      );
    }
    const lineVerticesBuffers = createLineVerticesBuffers();

    const drawColorBuffers = initialFunctions.map((data) =>
      root.createBuffer(d.vec4f, data.color).$usage('uniform'),
    );

    // Compute shader

    const computeLayout = tgpu.bindGroupLayout({
      lineVertices: { storage: d.arrayOf(d.vec2f), access: 'mutable' },
      properties: { uniform: Properties },
    });

    const fnSlot = tgpu.slot<string>();
    const interpolatedFunction = tgpu.fn([d.f32], d.f32)`(x) {
      return fnSlot;
    }`.$uses({ fnSlot });

    const computePoints = (x: number) => {
      'use gpu';
      const properties = computeLayout.$.properties;
      const start = properties.transformation.mul(d.vec4f(-1, 0, 0, 1)).x;
      const end = properties.transformation.mul(d.vec4f(1, 0, 0, 1)).x;

      const pointX =
        start +
        ((end - start) / (d.f32(properties.interpolationPoints) - 1.0)) *
          d.f32(x);
      const pointY = interpolatedFunction(pointX);
      const result = properties.inverseTransformation.mul(
        d.vec4f(pointX, pointY, 0, 1),
      );
      computeLayout.$.lineVertices[x] = result.xy;
    };

    const computePipelines = initialFunctions.map(({ code }) =>
      root['~unstable']
        .with(fnSlot, code)
        .createGuardedComputePipeline(computePoints),
    );

    // Render background shader

    const renderBackgroundLayout = tgpu.bindGroupLayout({
      properties: { uniform: Properties },
    });

    const renderBackgroundVertex = tgpu['~unstable'].vertexFn({
      in: {
        vid: d.builtin.vertexIndex,
        iid: d.builtin.instanceIndex,
      },
      out: {
        position: d.builtin.position,
      },
    })(({ vid, iid }) => {
      const properties = renderBackgroundLayout.$.properties;

      const leftBot = properties.transformation.mul(d.vec4f(-1, -1, 0, 1));
      const rightTop = properties.transformation.mul(d.vec4f(1, 1, 0, 1));

      const transformedPoints = d.arrayOf(
        d.vec2f,
        4,
      )([
        d.vec2f(leftBot.x, 0.0),
        d.vec2f(rightTop.x, 0.0),
        d.vec2f(0.0, leftBot.y),
        d.vec2f(0.0, rightTop.y),
      ]);

      const currentPoint = properties.inverseTransformation.mul(
        d.vec4f(transformedPoints[iid * 2 + d.u32(vid / 2)].xy, 0, 1),
      );

      return {
        position: d.vec4f(
          currentPoint.x +
            d.f32(iid) *
              std.select(d.f32(-1), 1, d.u32(vid) % d.u32(2) === d.u32(0)) *
              0.005,
          currentPoint.y +
            d.f32(1 - iid) *
              std.select(d.f32(-1), 1, d.u32(vid) % d.u32(2) === d.u32(0)) *
              0.005,
          currentPoint.zw,
        ),
      };
    });

    const renderBackgroundFragment = tgpu['~unstable'].fragmentFn({
      out: d.vec4f,
    })(() => d.vec4f(0.9, 0.9, 0.9, 1.0));

    const renderBackgroundPipeline = root['~unstable']
      .withVertex(renderBackgroundVertex, {})
      .withFragment(renderBackgroundFragment, { format: presentationFormat })
      .withMultisample({ count: 4 })
      .withPrimitive({ topology: 'triangle-strip' })
      .createPipeline();

    // Render shader

    const renderLayout = tgpu.bindGroupLayout({
      lineVertices: { storage: (n: number) => d.arrayOf(d.vec2f, n) },
      properties: { uniform: Properties },
      color: { uniform: d.vec4f },
    });

    const orthoNormalForLine = (p1: d.v2f, p2: d.v2f) => {
      'use gpu';
      const line = p2.sub(p1);
      const ortho = d.vec2f(-line.y, line.x);
      return std.normalize(ortho);
    };

    const orthoNormalForVertex = (index: number) => {
      'use gpu';
      const properties = renderLayout.$.properties;
      if (index === 0 || index === properties.interpolationPoints - 1) {
        return d.vec2f(0.0, 1.0);
      }
      const previous = renderLayout.$.lineVertices[index - 1];
      const current = renderLayout.$.lineVertices[index];
      const next = renderLayout.$.lineVertices[index + 1];

      const n1 = orthoNormalForLine(previous, current);
      const n2 = orthoNormalForLine(current, next);

      const avg = n1.add(n2).div(2.0);

      return std.normalize(avg);
    };

    const renderVertex = tgpu['~unstable'].vertexFn({
      in: { vid: d.builtin.vertexIndex },
      out: { position: d.builtin.position },
    })(({ vid }) => {
      const currentVertex = d.u32(vid / 2);
      const orthonormal = orthoNormalForVertex(currentVertex);
      const properties = renderLayout.$.properties;
      const offset = orthonormal.mul(
        properties.lineWidth *
          std.select(d.f32(-1), d.f32(1), d.u32(vid) % d.u32(2) === d.u32(0)),
      );
      const pos = renderLayout.$.lineVertices[currentVertex].add(offset);
      return {
        position: d.vec4f(pos, 0.0, 1.0),
      };
    });

    const renderFragment = tgpu['~unstable'].fragmentFn({
      out: d.vec4f,
    })(() => renderLayout.$.color);

    const renderPipeline = root['~unstable']
      .withVertex(renderVertex, {})
      .withFragment(renderFragment, { format: presentationFormat })
      .withMultisample({ count: 4 })
      .withPrimitive({ topology: 'triangle-strip' })
      .createPipeline();

    // Draw

    function draw() {
      const scale = 0.99;

      mat4.scale(
        properties.transformation,
        [scale, scale, 1],
        properties.transformation,
      );

      queuePropertiesBufferUpdate();

      initialFunctions.forEach((_, i) => {
        runComputePass(i);
      });
      runRenderBackgroundPass();
      runRenderPass();
    }

    function runComputePass(functionNumber: number) {
      const computePipeline = computePipelines[functionNumber];

      const bindGroup = root.createBindGroup(computeLayout, {
        lineVertices: lineVerticesBuffers[functionNumber],
        properties: propertiesBuffer,
      });

      computePipeline
        .with(bindGroup)
        .dispatchThreads(properties.interpolationPoints);
    }

    const msTexture = root['~unstable']
      .createTexture({
        size: [
          canvas.clientWidth * PixelRatio.get(),
          canvas.clientHeight * PixelRatio.get(),
        ],
        sampleCount: 4,
        format: presentationFormat,
      })
      .$usage('render');

    function runRenderBackgroundPass() {
      const renderBindGroup = root.createBindGroup(renderBackgroundLayout, {
        properties: propertiesBuffer,
      });

      renderBackgroundPipeline
        .with(renderBindGroup)
        .withColorAttachment({
          view: root.unwrap(msTexture).createView(),
          resolveTarget: context.getCurrentTexture().createView(),
          clearValue: [0.6, 0.6, 0.6, 0.6],
          loadOp: 'clear',
          storeOp: 'store',
        })
        .draw(4, 2);
    }

    function runRenderPass() {
      const resolveTarget = context.getCurrentTexture();

      initialFunctions.forEach((_, functionNumber) => {
        const renderBindGroup = root.createBindGroup(renderLayout, {
          lineVertices: lineVerticesBuffers[functionNumber],
          properties: propertiesBuffer,
          color: drawColorBuffers[functionNumber],
        });

        renderPipeline
          .with(renderBindGroup)
          .withColorAttachment({
            view: root.unwrap(msTexture).createView(),
            resolveTarget: resolveTarget.createView(),
            loadOp: 'load',
            storeOp: 'store',
          })
          .draw(properties.interpolationPoints * 2);
      });
    }

    // Helper definitions

    function fromHex(hex: string) {
      const r = Number.parseInt(hex.slice(1, 3), 16);
      const g = Number.parseInt(hex.slice(3, 5), 16);
      const b = Number.parseInt(hex.slice(5, 7), 16);

      return d.vec4f(r / 255.0, g / 255.0, b / 255.0, 1.0);
    }

    function queuePropertiesBufferUpdate() {
      properties.inverseTransformation = mat4.inverse(
        properties.transformation,
        d.mat4x4f(),
      );
      propertiesBuffer.write(properties);
    }

    return draw;
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
