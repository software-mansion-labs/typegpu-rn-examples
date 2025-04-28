import tgpu, { type TgpuRenderPipeline, type TgpuBindGroup } from 'typegpu';
import * as d from 'typegpu/data';
import * as k from './kernels';
import { useWebGPU } from '../../useWebGPU';
import { Canvas } from 'react-native-wgpu';
import {
  type GestureResponderEvent,
  Image,
  PixelRatio,
  View,
  Text,
  Switch,
} from 'react-native';
import {
  type MutableRefObject,
  type RefObject,
  useCallback,
  useRef,
  useState,
} from 'react';
import {
  INK_AMOUNT,
  WORKGROUP_SIZE_X,
  WORKGROUP_SIZE_Y,
  params,
  FORCE_SCALE,
  Params,
  BrushParams,
  SIMULATION_QUALITY,
} from './params';
import { resampleImageBitmapToTexture } from './imageResizer';
import type { BrushInfo, DisplayMode, RenderEntries } from './types';

const imageAsset = Image.resolveAssetSource(require('../../assets/plums.jpg'));

class DoubleBuffer<T> {
  buffers: [T, T];
  index: number;
  constructor(bufferA: T, bufferB: T, initialIndex = 0) {
    this.buffers = [bufferA, bufferB];
    this.index = initialIndex;
  }

  get current(): T {
    return this.buffers[this.index];
  }
  get currentIndex(): number {
    return this.index;
  }

  swap(): void {
    this.index ^= 1;
  }
  setCurrent(index: number): void {
    this.index = index;
  }
}

async function createScene({
  context,
  device,
  presentationFormat,
  brushInfo,
  showField,
  canvasSize,
  enableBoundary,
}: {
  context: GPUCanvasContext;
  device: GPUDevice;
  presentationFormat: GPUTextureFormat;
  brushInfo: RefObject<BrushInfo>;
  showField: RefObject<DisplayMode>;
  canvasSize: MutableRefObject<{ width: number; height: number } | null>;
  enableBoundary: RefObject<boolean>;
}) {
  const root = tgpu.initFromDevice({ device });

  const width = context.canvas.width;
  const height = context.canvas.height;
  canvasSize.current = { width, height };

  const simWidth = Math.max(1, Math.floor(width * SIMULATION_QUALITY));
  const simHeight = Math.max(1, Math.floor(height * SIMULATION_QUALITY));

  const dispatchX = Math.ceil(simWidth / WORKGROUP_SIZE_X);
  const dispatchY = Math.ceil(simHeight / WORKGROUP_SIZE_Y);

  const simParamBuffer = root
    .createBuffer(Params, {
      dt: params.dt,
      viscosity: params.viscosity,
      enableBoundary: params.enableBoundary ? 1 : 0,
    })
    .$usage('uniform');

  const brushParamBuffer = root
    .createBuffer(BrushParams, {
      pos: d.vec2i(0, 0),
      delta: d.vec2f(0, 0),
      radius: simWidth * 0.1,
      forceScale: FORCE_SCALE,
      inkAmount: INK_AMOUNT,
    })
    .$usage('uniform');

  function createField(name: string) {
    return root['~unstable']
      .createTexture({ size: [simWidth, simHeight], format: 'rgba16float' })
      .$usage('storage', 'sampled')
      .$name(name);
  }

  const plumsResponse = await fetch(imageAsset.uri);
  const plumsImage = await createImageBitmap(await plumsResponse.blob());
  const resized = await resampleImageBitmapToTexture(
    root,
    plumsImage,
    width,
    height,
  );

  const backgroundTexture = root['~unstable']
    .createTexture({ size: [width, height], format: 'rgba8unorm' })
    .$usage('sampled', 'render')
    .$name('background');

  const encoder = device.createCommandEncoder();
  encoder.copyTextureToTexture(
    { texture: root.unwrap(resized) },
    { texture: root.unwrap(backgroundTexture) },
    [width, height],
  );
  device.queue.submit([encoder.finish()]);

  const velTex = [createField('velocity0'), createField('velocity1')];
  const inkTex = [createField('density0'), createField('density1')];
  const pressureTex = [createField('pressure0'), createField('pressure1')];
  const newInkTex = createField('addedInk');
  const forceTex = createField('force');
  const divergenceTex = createField('divergence');

  const linSampler = tgpu['~unstable'].sampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  const brushPipeline = root['~unstable']
    .withCompute(k.brushFn)
    .createPipeline();
  const addForcePipeline = root['~unstable']
    .withCompute(k.addForcesFn)
    .createPipeline();
  const advectPipeline = root['~unstable']
    .withCompute(k.advectFn)
    .createPipeline();
  const diffusionPipeline = root['~unstable']
    .withCompute(k.diffusionFn)
    .createPipeline();
  const divergencePipeline = root['~unstable']
    .withCompute(k.divergenceFn)
    .createPipeline();
  const pressurePipeline = root['~unstable']
    .withCompute(k.pressureFn)
    .createPipeline();
  const projectPipeline = root['~unstable']
    .withCompute(k.projectFn)
    .createPipeline();
  const advectInkPipeline = root['~unstable']
    .withCompute(k.advectScalarFn)
    .createPipeline();
  const addInkPipeline = root['~unstable']
    .withCompute(k.addInkFn)
    .createPipeline();

  // Rendering
  const velBuffer = new DoubleBuffer(velTex[0], velTex[1]);
  const inkBuffer = new DoubleBuffer(inkTex[0], inkTex[1]);
  const pressureBuffer = new DoubleBuffer(pressureTex[0], pressureTex[1]);

  const renderPipelineImage = root['~unstable']
    .withVertex(k.renderFn, k.renderFn.shell.attributes)
    .withFragment(k.fragmentImageFn, { format: presentationFormat })
    .createPipeline();
  const renderPipelineInk = root['~unstable']
    .withVertex(k.renderFn, k.renderFn.shell.attributes)
    .withFragment(k.fragmentInkFn, { format: presentationFormat })
    .createPipeline();
  const renderPipelineVel = root['~unstable']
    .withVertex(k.renderFn, k.renderFn.shell.attributes)
    .withFragment(k.fragmentVelFn, { format: presentationFormat })
    .createPipeline();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

  const brushBindGroup = root.createBindGroup(k.brushLayout, {
    brushParams: brushParamBuffer,
    forceDst: forceTex.createView('writeonly'),
    inkDst: newInkTex.createView('writeonly'),
  });

  const addInkBindGroups = [0, 1].map((i) => {
    const srcIdx = i;
    const dstIdx = 1 - i;
    return root.createBindGroup(k.addInkLayout, {
      src: inkTex[srcIdx].createView('sampled'),
      add: newInkTex.createView('sampled'),
      dst: inkTex[dstIdx].createView('writeonly'),
    });
  });

  const addForceBindGroups = [0, 1].map((i) => {
    const srcIdx = i;
    const dstIdx = 1 - i;
    return root.createBindGroup(k.addForcesLayout, {
      src: velTex[srcIdx].createView('sampled'),
      force: forceTex.createView('sampled'),
      dst: velTex[dstIdx].createView('writeonly'),
      simParams: simParamBuffer,
    });
  });

  const advectBindGroups = [0, 1].map((i) => {
    const srcIdx = 1 - i;
    const dstIdx = i;
    return root.createBindGroup(k.advectLayout, {
      src: velTex[srcIdx].createView('sampled'),
      dst: velTex[dstIdx].createView('writeonly'),
      simParams: simParamBuffer,
      linSampler,
    });
  });

  const diffusionBindGroups = [0, 1].map((i) => {
    const srcIdx = i;
    const dstIdx = 1 - i;
    return root.createBindGroup(k.diffusionLayout, {
      in: velTex[srcIdx].createView('sampled'),
      out: velTex[dstIdx].createView('writeonly'),
      simParams: simParamBuffer,
    });
  });

  const divergenceBindGroups = [0, 1].map((i) =>
    root.createBindGroup(k.divergenceLayout, {
      vel: velTex[i].createView('sampled'),
      div: divergenceTex.createView('writeonly'),
    }),
  );

  const pressureBindGroups = [0, 1].map((i) => {
    const dstIdx = 1 - i;
    return root.createBindGroup(k.pressureLayout, {
      x: pressureTex[i].createView('sampled'),
      b: divergenceTex.createView('sampled'),
      out: pressureTex[dstIdx].createView('writeonly'),
    });
  });

  const projectBindGroups = [0, 1].map((velIdx) =>
    [0, 1].map((pIdx) =>
      root.createBindGroup(k.projectLayout, {
        vel: velTex[velIdx].createView('sampled'),
        p: pressureTex[pIdx].createView('sampled'),
        out: velTex[1 - velIdx].createView('writeonly'),
      }),
    ),
  );

  const advectInkBindGroups = [0, 1].map((velIdx) =>
    [0, 1].map((inkIdx) =>
      root.createBindGroup(k.advectInkLayout, {
        vel: velTex[velIdx].createView('sampled'),
        src: inkTex[inkIdx].createView('sampled'),
        dst: inkTex[1 - inkIdx].createView('writeonly'),
        simParams: simParamBuffer,
        linSampler,
      }),
    ),
  );

  const renderBindGroups = {
    image: [0, 1].map((idx) =>
      root.createBindGroup(k.renderLayout, {
        result: inkTex[idx].createView('sampled'),
        background: backgroundTexture.createView('sampled'),
        linSampler,
      }),
    ),
    ink: [0, 1].map((idx) =>
      root.createBindGroup(k.renderLayout, {
        result: inkTex[idx].createView('sampled'),
        background: backgroundTexture.createView('sampled'),
        linSampler,
      }),
    ),
    velocity: [0, 1].map((idx) =>
      root.createBindGroup(k.renderLayout, {
        result: velTex[idx].createView('sampled'),
        background: backgroundTexture.createView('sampled'),
        linSampler,
      }),
    ),
  };

  function loop() {
    simParamBuffer.write({
      dt: params.dt,
      viscosity: params.viscosity,
      enableBoundary: enableBoundary.current ? 1 : 0,
    });

    if (brushInfo.current?.isDown) {
      brushParamBuffer.write({
        pos: d.vec2i(...brushInfo.current.pos),
        delta: d.vec2f(...brushInfo.current.delta),
        radius: simWidth * 0.1,
        forceScale: FORCE_SCALE,
        inkAmount: INK_AMOUNT,
      });

      brushPipeline
        .with(k.brushLayout, brushBindGroup)
        .dispatchWorkgroups(dispatchX, dispatchY);

      addInkPipeline
        .with(k.addInkLayout, addInkBindGroups[inkBuffer.currentIndex])
        .dispatchWorkgroups(dispatchX, dispatchY);
      inkBuffer.swap();

      addForcePipeline
        .with(k.addForcesLayout, addForceBindGroups[velBuffer.currentIndex])
        .dispatchWorkgroups(dispatchX, dispatchY);
    } else {
      velBuffer.setCurrent(0);
    }

    advectPipeline
      .with(k.advectLayout, advectBindGroups[velBuffer.currentIndex])
      .dispatchWorkgroups(dispatchX, dispatchY);

    for (let i = 0; i < params.jacobiIter; i++) {
      diffusionPipeline
        .with(k.diffusionLayout, diffusionBindGroups[velBuffer.currentIndex])
        .dispatchWorkgroups(dispatchX, dispatchY);
      velBuffer.swap();
    }

    divergencePipeline
      .with(k.divergenceLayout, divergenceBindGroups[velBuffer.currentIndex])
      .dispatchWorkgroups(dispatchX, dispatchY);

    pressureBuffer.setCurrent(0);
    for (let i = 0; i < params.jacobiIter; i++) {
      pressurePipeline
        .with(k.pressureLayout, pressureBindGroups[pressureBuffer.currentIndex])
        .dispatchWorkgroups(dispatchX, dispatchY);
      pressureBuffer.swap();
    }

    projectPipeline
      .with(
        k.projectLayout,
        projectBindGroups[velBuffer.currentIndex][pressureBuffer.currentIndex],
      )
      .dispatchWorkgroups(dispatchX, dispatchY);
    velBuffer.swap();

    advectInkPipeline
      .with(
        k.advectInkLayout,
        advectInkBindGroups[velBuffer.currentIndex][inkBuffer.currentIndex],
      )
      .dispatchWorkgroups(dispatchX, dispatchY);
    inkBuffer.swap();

    let pipeline: TgpuRenderPipeline;
    let renderBindGroup: TgpuBindGroup<RenderEntries>[];

    if (showField.current === 'ink') {
      pipeline = renderPipelineInk;
      renderBindGroup = renderBindGroups.ink;
    } else if (showField.current === 'velocity') {
      pipeline = renderPipelineVel;
      renderBindGroup = renderBindGroups.velocity;
    } else {
      pipeline = renderPipelineImage;
      renderBindGroup = renderBindGroups.image;
    }

    pipeline
      .withColorAttachment({
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
      })
      .with(k.renderLayout, renderBindGroup[inkBuffer.currentIndex])
      .draw(6);

    root['~unstable'].flush();
  }

  return loop;
}

export default function () {
  const brushInfo = useRef<BrushInfo>({
    pos: [0, 0],
    delta: [0, 0],
    isDown: false,
  });
  const canvasSize = useRef<{ width: number; height: number } | null>(null);

  const [showField, setShowField] = useState<DisplayMode>(params.showField);
  const showFieldRef = useRef(showField);
  showFieldRef.current = showField;

  const [enableBoundary, setEnableBoundary] = useState(params.enableBoundary);
  const enableBoundaryRef = useRef(enableBoundary);
  enableBoundaryRef.current = enableBoundary;

  const sceneFunction = useCallback(
    async ({
      context,
      device,
      presentationFormat,
    }: {
      context: GPUCanvasContext;
      device: GPUDevice;
      presentationFormat: GPUTextureFormat;
    }) => {
      return await createScene({
        context,
        device,
        presentationFormat,
        brushInfo,
        showField: showFieldRef,
        canvasSize,
        enableBoundary: enableBoundaryRef,
      });
    },
    [],
  );

  const ref = useWebGPU(sceneFunction);

  const realToCanvas = useCallback((x: number, y: number): [number, number] => {
    const size = canvasSize.current;
    if (!size) {
      return [0, 0];
    }
    const dpr = PixelRatio.get();

    const gx = Math.floor(x * dpr * SIMULATION_QUALITY);
    const gy = Math.floor((size.height - y * dpr) * SIMULATION_QUALITY);
    const cx = Math.max(0, Math.min(size.width * SIMULATION_QUALITY - 1, gx));
    const cy = Math.max(0, Math.min(size.height * SIMULATION_QUALITY - 1, gy));

    return [cx, cy];
  }, []);

  const handleStart = useCallback(
    (e: GestureResponderEvent) => {
      const { locationX, locationY } = e.nativeEvent;
      brushInfo.current = {
        pos: realToCanvas(locationX, locationY),
        delta: [0, 0],
        isDown: true,
      };
    },
    [realToCanvas],
  );

  const handleMove = useCallback(
    (e: GestureResponderEvent) => {
      if (!brushInfo.current) {
        return;
      }
      const { locationX, locationY } = e.nativeEvent;
      const [gx, gy] = realToCanvas(locationX, locationY);
      const dx = gx - brushInfo.current.pos[0];
      const dy = gy - brushInfo.current.pos[1];

      brushInfo.current = {
        pos: [gx, gy],
        delta: [dx, dy],
        isDown: true,
      };
    },
    [realToCanvas],
  );

  const handleEnd = useCallback(() => {
    brushInfo.current = {
      pos: [0, 0],
      delta: [0, 0],
      isDown: false,
    };
  }, []);

  const handleShowField = useCallback((field: DisplayMode) => {
    setShowField(field);
    params.showField = field;
  }, []);

  const handleBoundaryToggle = useCallback((value: boolean) => {
    setEnableBoundary(value);
    params.enableBoundary = value;
  }, []);

  return (
    <View style={{ width: '100%', alignItems: 'center' }}>
      <Canvas
        ref={ref}
        style={{
          width: '100%',
          aspectRatio: 1,
        }}
        onTouchStart={(e) => handleStart(e)}
        onTouchMove={(e) => handleMove(e)}
        onTouchEnd={() => handleEnd()}
      />
      <View>
        <View
          style={{
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            gap: 10,
          }}
        >
          {(['ink', 'velocity', 'image'] as DisplayMode[]).map((field) => (
            <View
              key={field}
              style={{
                padding: 10,
                backgroundColor: showField === field ? '#ccc' : '#fff',
                borderRadius: 20,
                margin: 10,
              }}
              onStartShouldSetResponder={() => true}
              onResponderRelease={() => handleShowField(field)}
            >
              <Text>{field}</Text>
            </View>
          ))}
        </View>
        <View
          style={{
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            marginTop: 10,
          }}
        >
          <Text style={{ fontSize: 16, fontWeight: '600' }}>Boundary</Text>
          <Switch value={enableBoundary} onValueChange={handleBoundaryToggle} />
        </View>
      </View>
    </View>
  );
}
