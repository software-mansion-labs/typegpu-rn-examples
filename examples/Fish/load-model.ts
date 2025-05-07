import { load } from '@loaders.gl/core';
import { OBJLoader } from '@loaders.gl/obj';
import type { TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import { modelVertexLayout } from './schemas';

function base64ToArrayBuffer(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

export async function loadModel(
  root: TgpuRoot,
  modelBase64: string,
  textureBase64: string,
) {
  const modelData = base64ToArrayBuffer(modelBase64);
  const modelMesh = await load(modelData, OBJLoader);
  const polygonCount = modelMesh.attributes.POSITION.value.length / 3;

  const vertexBuffer = root
    .createBuffer(modelVertexLayout.schemaForCount(polygonCount))
    .$usage('vertex')
    .$name('model vertices');

  const modelVertices = [];
  for (let i = 0; i < polygonCount; i++) {
    modelVertices.push({
      modelPosition: d.vec3f(
        modelMesh.attributes.POSITION.value[3 * i],
        modelMesh.attributes.POSITION.value[3 * i + 1],
        modelMesh.attributes.POSITION.value[3 * i + 2],
      ),
      modelNormal: d.vec3f(
        modelMesh.attributes.NORMAL.value[3 * i],
        modelMesh.attributes.NORMAL.value[3 * i + 1],
        modelMesh.attributes.NORMAL.value[3 * i + 2],
      ),
      textureUV: d.vec2f(
        modelMesh.attributes.TEXCOORD_0.value[2 * i],
        1 - modelMesh.attributes.TEXCOORD_0.value[2 * i + 1],
      ),
    });
  }
  modelVertices.reverse();

  vertexBuffer.write(modelVertices);

  const textureResponse = await fetch(textureBase64);
  const imageBitmap = await createImageBitmap(await textureResponse.blob());
  const texture = root['~unstable']
    .createTexture({
      size: [imageBitmap.width, imageBitmap.height],
      format: 'rgba8unorm',
    })
    .$usage('sampled', 'render')
    .$name('model texture');

  root.device.queue.copyExternalImageToTexture(
    { source: imageBitmap },
    { texture: root.unwrap(texture) },
    [imageBitmap.width, imageBitmap.height],
  );

  return {
    vertexBuffer: vertexBuffer,
    polygonCount: polygonCount,
    texture: texture,
  };
}
