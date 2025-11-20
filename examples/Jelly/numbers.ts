import { Image } from 'react-native';
import type { SampledFlag, StorageFlag, TgpuRoot, TgpuTexture } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';

const PERCENTAGE_WIDTH = 256 * 2;
const PERCENTAGE_HEIGHT = 128 * 2;
const PERCENTAGE_COUNT = 101; // 0% to 100%

export class NumberProvider {
  #root: TgpuRoot;
  digitTextureAtlas: TgpuTexture<{
    size: [
      typeof PERCENTAGE_WIDTH,
      typeof PERCENTAGE_HEIGHT,
      typeof PERCENTAGE_COUNT,
    ];
    format: 'rgba8unorm';
  }> &
    SampledFlag &
    StorageFlag;

  constructor(root: TgpuRoot) {
    this.#root = root;
    this.digitTextureAtlas = root['~unstable']
      .createTexture({
        size: [PERCENTAGE_WIDTH, PERCENTAGE_HEIGHT, PERCENTAGE_COUNT],
        format: 'rgba8unorm',
      })
      .$usage('sampled', 'render', 'storage');
  }

  async fillAtlas() {
    const sources = [
      Image.resolveAssetSource(require(`../../assets/numbers/character-0.png`))
        .uri,
      Image.resolveAssetSource(require(`../../assets/numbers/character-1.png`))
        .uri,
      Image.resolveAssetSource(require(`../../assets/numbers/character-2.png`))
        .uri,
      Image.resolveAssetSource(require(`../../assets/numbers/character-3.png`))
        .uri,
      Image.resolveAssetSource(require(`../../assets/numbers/character-4.png`))
        .uri,
      Image.resolveAssetSource(require(`../../assets/numbers/character-5.png`))
        .uri,
      Image.resolveAssetSource(require(`../../assets/numbers/character-6.png`))
        .uri,
      Image.resolveAssetSource(require(`../../assets/numbers/character-7.png`))
        .uri,
      Image.resolveAssetSource(require(`../../assets/numbers/character-8.png`))
        .uri,
      Image.resolveAssetSource(require(`../../assets/numbers/character-9.png`))
        .uri,
      Image.resolveAssetSource(
        require(`../../assets/numbers/character-percent.png`),
      ).uri,
    ];

    const bitmaps = await Promise.all(
      sources.map(async (url) => {
        const response = await fetch(url);
        return await createImageBitmap(await response.blob());
      }),
    );

    const tempTexture = this.#root['~unstable']
      .createTexture({
        size: [128, 256, 11],
        format: 'rgba8unorm',
      })
      .$usage('storage');
    const tempTextureStorageView = tempTexture.createView(
      d.textureStorage2dArray('rgba8unorm', 'read-only'),
    );
    tempTexture.write(bitmaps);

    const atlasStorageView = this.digitTextureAtlas.createView(
      d.textureStorage2dArray('rgba8unorm'),
    );

    const fillAtlasCompute = this.#root[
      '~unstable'
    ].createGuardedComputePipeline((x, y, z) => {
      'use gpu';
      const hasTwoDigits = d.u32(z) >= d.u32(10);
      const hasThreeDigits = d.u32(z) >= d.u32(100);

      const quarterWidth = d.u32(PERCENTAGE_WIDTH / 4);
      const quarter = d.u32(x / quarterWidth);
      const quarterX = d.u32(x) % quarterWidth;

      let pixelColor = d.vec4f(0, 0, 0, 0);

      if (hasThreeDigits) {
        // Layout: [hundreds][tens][ones][%]
        const digit1 = d.u32(z / 100);
        const digit2 = d.u32(z / 10) % d.u32(10);
        const digit3 = d.u32(z) % d.u32(10);

        if (quarter === d.u32(0)) {
          pixelColor = std.textureLoad(
            tempTextureStorageView.$,
            d.vec2u(quarterX, y),
            digit1,
          );
        } else if (quarter === d.u32(1)) {
          pixelColor = std.textureLoad(
            tempTextureStorageView.$,
            d.vec2u(quarterX, y),
            digit2,
          );
        } else if (quarter === d.u32(2)) {
          pixelColor = std.textureLoad(
            tempTextureStorageView.$,
            d.vec2u(quarterX, y),
            digit3,
          );
        } else {
          pixelColor = std.textureLoad(
            tempTextureStorageView.$,
            d.vec2u(quarterX, y),
            d.u32(10),
          );
        }
      } else if (hasTwoDigits) {
        // Layout: [empty][tens][ones][%]
        const digit1 = d.u32(z / 10);
        const digit2 = d.u32(z) % d.u32(10);

        if (quarter === d.u32(1)) {
          pixelColor = std.textureLoad(
            tempTextureStorageView.$,
            d.vec2u(quarterX, y),
            digit1,
          );
        } else if (quarter === d.u32(2)) {
          pixelColor = std.textureLoad(
            tempTextureStorageView.$,
            d.vec2u(quarterX, y),
            digit2,
          );
        } else if (quarter === d.u32(3)) {
          pixelColor = std.textureLoad(
            tempTextureStorageView.$,
            d.vec2u(quarterX, y),
            d.u32(10),
          );
        }
      } else {
        // Layout: [empty][empty][ones][%]
        const digit1 = d.u32(z) % d.u32(10); // ones

        if (quarter === d.u32(2)) {
          pixelColor = std.textureLoad(
            tempTextureStorageView.$,
            d.vec2u(quarterX, y),
            digit1,
          );
        } else if (quarter === d.u32(3)) {
          pixelColor = std.textureLoad(
            tempTextureStorageView.$,
            d.vec2u(quarterX, y),
            d.u32(10),
          );
        }
      }

      std.textureStore(atlasStorageView.$, d.vec2u(x, y), d.u32(z), pixelColor);
    });

    fillAtlasCompute.dispatchThreads(
      PERCENTAGE_WIDTH,
      PERCENTAGE_HEIGHT,
      PERCENTAGE_COUNT,
    );
  }
}
