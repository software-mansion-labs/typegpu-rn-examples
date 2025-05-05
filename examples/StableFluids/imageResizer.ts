import * as std from 'typegpu/std';
import * as d from 'typegpu/data';
import tgpu, { type TgpuRoot } from 'typegpu';
import * as k from './kernels';

export const resampleImageLayout = tgpu.bindGroupLayout({
  src: { texture: 'float' },
  linSampler: { sampler: 'filtering' },
});

export const fragmentResampleFn = tgpu['~unstable'].fragmentFn({
  in: { uv: d.vec2f },
  out: d.vec4f,
})((inp) => {
  const color = std.textureSample(
    resampleImageLayout.$.src,
    resampleImageLayout.$.linSampler,
    inp.uv,
  );
  return d.vec4f(color.xyz, d.f32(1.0));
});

export async function resampleImageBitmapToTexture(
  root: TgpuRoot,
  imageBitmap: ImageBitmap,
  width: number,
  height: number,
) {
  const srcDims = [imageBitmap.width, imageBitmap.height] as const;
  const srcTexture = root['~unstable']
    .createTexture({ size: srcDims, format: 'rgba8unorm' })
    .$usage('sampled')
    .$name('resampleSrc');
  root.device.queue.copyExternalImageToTexture(
    { source: imageBitmap },
    { texture: root.unwrap(srcTexture) },
    srcDims,
  );

  const dstTexture = root['~unstable']
    .createTexture({ size: [width, height], format: 'rgba8unorm' })
    .$usage('render', 'sampled')
    .$name('resampled');

  const sampler = tgpu['~unstable'].sampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  const pipeline = root['~unstable']
    .withVertex(k.renderFn, k.renderFn.shell.attributes)
    .withFragment(fragmentResampleFn, { format: 'rgba8unorm' })
    .createPipeline();

  const bindGroup = root.createBindGroup(resampleImageLayout, {
    src: srcTexture.createView('sampled'),
    linSampler: sampler,
  });

  pipeline
    .withColorAttachment({
      view: dstTexture,
      loadOp: 'clear',
      storeOp: 'store',
    })
    .with(resampleImageLayout, bindGroup)
    .draw(6);
  root['~unstable'].flush();

  return dstTexture;
}
