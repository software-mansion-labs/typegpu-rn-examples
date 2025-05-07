import { Canvas } from 'react-native-wgpu';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as m from 'wgpu-matrix';
import { computeShader } from './compute';
import { loadModel } from './load-model';
import * as p from './params';
import { fragmentShader, vertexShader } from './render';
import {
  Camera,
  computeBindGroupLayout,
  type ModelData,
  ModelDataArray,
  modelVertexLayout,
  MouseRay,
  renderBindGroupLayout,
  renderInstanceLayout,
} from './schemas';
import { Dimensions, Text, View } from 'react-native';
import { useWebGPU } from '../../useWebGPU';
import {
  fishModel as fishModelB64,
  fishTexture,
  floorModel,
  floorTexture,
} from './base64resources';

export default function () {
  const ref = useWebGPU(
    async ({ context, device, presentationFormat, canvas }) => {
      const root = tgpu.initFromDevice({ device });

      // models and textures

      // https://sketchfab.com/3d-models/animated-low-poly-fish-64adc2e5a4be471e8279532b9610c878
      const fishModel = await loadModel(root, fishModelB64, fishTexture);

      // https://www.cgtrader.com/free-3d-models/space/other/rainy-ocean
      // https://www.rawpixel.com/image/6032317/white-sand-texture-free-public-domain-cc0-photo
      const oceanFloorModel = await loadModel(root, floorModel, floorTexture);

      // buffers

      const fishDataBuffers = Array.from({ length: 2 }, (_, idx) =>
        root
          .createBuffer(ModelDataArray(p.fishAmount))
          .$usage('storage', 'vertex')
          .$name(`fish data buffer ${idx}`),
      );

      const randomizeFishPositions = () => {
        const positions: d.Infer<typeof ModelData>[] = Array.from(
          { length: p.fishAmount },
          () => ({
            position: d.vec3f(
              Math.random() * p.aquariumSize.x - p.aquariumSize.x / 2,
              Math.random() * p.aquariumSize.y - p.aquariumSize.y / 2,
              Math.random() * p.aquariumSize.z - p.aquariumSize.z / 2,
            ),
            direction: d.vec3f(
              Math.random() * 0.1 - 0.05,
              Math.random() * 0.1 - 0.05,
              Math.random() * 0.1 - 0.05,
            ),
            scale: p.fishModelScale * (1 + (Math.random() - 0.5) * 0.8),
            applySeaFog: 1,
            applySeaDesaturation: 1,
          }),
        );
        fishDataBuffers[0].write(positions);
        fishDataBuffers[1].write(positions);
      };
      randomizeFishPositions();

      const camera = {
        position: p.cameraInitialPosition,
        targetPos: p.cameraInitialTarget,
        view: m.mat4.lookAt(
          p.cameraInitialPosition,
          p.cameraInitialTarget,
          d.vec3f(0, 1, 0),
          d.mat4x4f(),
        ),
        projection: m.mat4.perspective(
          Math.PI / 4,
          canvas.clientWidth / canvas.clientHeight,
          0.1,
          1000,
          d.mat4x4f(),
        ),
      };

      const cameraBuffer = root
        .createBuffer(Camera, camera)
        .$usage('uniform')
        .$name('camera buffer');

      const mouseRay = MouseRay({
        activated: 0,
        pointX: d.vec3f(),
        pointY: d.vec3f(),
      });

      const mouseRayBuffer = root
        .createBuffer(MouseRay, mouseRay)
        .$usage('uniform')
        .$name('mouse buffer');

      const timePassedBuffer = root
        .createBuffer(d.u32)
        .$usage('uniform')
        .$name('time passed buffer');

      const oceanFloorDataBuffer = root
        .createBuffer(ModelDataArray(1), [
          {
            position: d.vec3f(0, -p.aquariumSize.y / 2 - 1, 0),
            direction: d.vec3f(1, 0, 0),
            scale: 1,
            applySeaFog: 1,
            applySeaDesaturation: 0,
          },
        ])
        .$usage('storage', 'vertex')
        .$name('ocean floor buffer');

      // pipelines

      const renderPipeline = root['~unstable']
        .withVertex(vertexShader, modelVertexLayout.attrib)
        .withFragment(fragmentShader, { format: presentationFormat })
        .withDepthStencil({
          format: 'depth24plus',
          depthWriteEnabled: true,
          depthCompare: 'less',
        })
        .withPrimitive({ topology: 'triangle-list' })
        .createPipeline()
        .$name('render pipeline');

      const depthTexture = root.device.createTexture({
        size: [canvas.width, canvas.height, 1],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });

      const computePipeline = root['~unstable']
        .withCompute(computeShader)
        .createPipeline()
        .$name('compute pipeline');

      // bind groups

      const sampler = root.device.createSampler({
        addressModeU: 'repeat',
        addressModeV: 'repeat',
        magFilter: 'linear',
        minFilter: 'linear',
      });

      const renderFishBindGroups = [0, 1].map((idx) =>
        root.createBindGroup(renderBindGroupLayout, {
          modelData: fishDataBuffers[idx],
          camera: cameraBuffer,
          modelTexture: fishModel.texture,
          sampler: sampler,
        }),
      );

      const renderOceanFloorBindGroup = root.createBindGroup(
        renderBindGroupLayout,
        {
          modelData: oceanFloorDataBuffer,
          camera: cameraBuffer,
          modelTexture: oceanFloorModel.texture,
          sampler: sampler,
        },
      );

      const computeBindGroups = [0, 1].map((idx) =>
        root.createBindGroup(computeBindGroupLayout, {
          currentFishData: fishDataBuffers[idx],
          nextFishData: fishDataBuffers[1 - idx],
          mouseRay: mouseRayBuffer,
          timePassed: timePassedBuffer,
        }),
      );

      // frame

      let odd = false;
      let lastTimestamp: DOMHighResTimeStamp = 0;

      function frame(timestamp: DOMHighResTimeStamp) {
        odd = !odd;

        timePassedBuffer.write(timestamp - lastTimestamp);
        lastTimestamp = timestamp;
        cameraBuffer.write(camera);
        mouseRayBuffer.write(mouseRay);

        computePipeline
          .with(computeBindGroupLayout, computeBindGroups[odd ? 1 : 0])
          .dispatchWorkgroups(p.fishAmount / p.workGroupSize);

        renderPipeline
          .withColorAttachment({
            view: context.getCurrentTexture().createView(),
            clearValue: [
              p.backgroundColor.x,
              p.backgroundColor.y,
              p.backgroundColor.z,
              1,
            ],
            loadOp: 'clear' as const,
            storeOp: 'store' as const,
          })
          .withDepthStencilAttachment({
            view: depthTexture.createView(),
            depthClearValue: 1,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
          })
          .with(modelVertexLayout, oceanFloorModel.vertexBuffer)
          .with(renderInstanceLayout, oceanFloorDataBuffer)
          .with(renderBindGroupLayout, renderOceanFloorBindGroup)
          .draw(oceanFloorModel.polygonCount, 1);

        renderPipeline
          .withColorAttachment({
            view: context.getCurrentTexture().createView(),
            clearValue: [
              p.backgroundColor.x,
              p.backgroundColor.y,
              p.backgroundColor.z,
              1,
            ],
            loadOp: 'load' as const,
            storeOp: 'store' as const,
          })
          .withDepthStencilAttachment({
            view: depthTexture.createView(),
            depthClearValue: 1,
            depthLoadOp: 'load',
            depthStoreOp: 'store',
          })
          .with(modelVertexLayout, fishModel.vertexBuffer)
          .with(renderInstanceLayout, fishDataBuffers[odd ? 1 : 0])
          .with(renderBindGroupLayout, renderFishBindGroups[odd ? 1 : 0])
          .draw(fishModel.polygonCount, p.fishAmount);

        root['~unstable'].flush();
      }

      return frame;
    },
  );

  return (
    <View style={{ position: 'static' }}>
      <View
        style={{
          zIndex: 30,
          position: 'static',
          padding: 30,
          gap: 20,
        }}
      >
        <Text
          style={{
            fontSize: 40,
            color: 'white',
            fontWeight: 900,
            zIndex: 80,
          }}
        >
          Aquarium
        </Text>

        <Text
          style={{
            fontSize: 20,
            color: 'white',
            fontWeight: 500,
            zIndex: 80,
          }}
        >
          A public aquarium (pl. aquaria) or public water zoo is the aquatic
          counterpart of a zoo, which houses living aquatic animal and plant
          specimens for public viewing. Most public aquariums feature tanks
          larger than those kept by home aquarists, as well as smaller tanks.
        </Text>

        <Text
          style={{
            fontSize: 20,
            color: 'white',
            fontWeight: 500,
            zIndex: 80,
          }}
        >
          Since the first public aquariums were built in the mid-19th century,
          they have become popular and their numbers have increased. Most modern
          accredited aquariums stress conservation issues and educating the
          public.
        </Text>
      </View>
      <Canvas
        ref={ref}
        style={{
          position: 'absolute',
          top: 0,
          right: 0,
          zIndex: 20,
          width: Dimensions.get('window').width,
          height: Dimensions.get('window').height,
        }}
      />
    </View>
  );
}
