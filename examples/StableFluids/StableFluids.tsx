import tgpu, { type TgpuRenderPipeline, type TgpuBindGroup } from "typegpu";
import * as d from "typegpu/data";
import * as k from "./kernels";
import { useWebGPU } from "../../useWebGPU";
import { Canvas } from "react-native-wgpu";
import {
  type GestureResponderEvent,
  Image,
  PixelRatio,
  View,
  Text,
} from "react-native";
import {
  type MutableRefObject,
  type RefObject,
  useCallback,
  useRef,
  useState,
} from "react";
import {
  N,
  SIM_N,
  INK_AMOUNT,
  WORKGROUP_SIZE_X,
  WORKGROUP_SIZE_Y,
  params,
  RADIUS,
  FORCE_SCALE,
  Params,
  BrushParams,
} from "./params";

type DisplayMode = "ink" | "velocity" | "image";
type BrushInfo = {
  pos: [number, number];
  delta: [number, number];
  isDown: boolean;
};
type RenderEntries = {
  result: { texture: "float" };
  background: { texture: "float" };
  linSampler: { sampler: "filtering" };
};

const imageAsset = Image.resolveAssetSource(require("../../assets/plums.jpg"));

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

const dispatchX = Math.ceil(SIM_N / WORKGROUP_SIZE_X);
const dispatchY = Math.ceil(SIM_N / WORKGROUP_SIZE_Y);

async function createScene({
  context,
  device,
  presentationFormat,
  brushInfo,
  showField,
  canvasSize,
}: {
  context: GPUCanvasContext;
  device: GPUDevice;
  presentationFormat: GPUTextureFormat;
  brushInfo: RefObject<BrushInfo>;
  showField: RefObject<DisplayMode>;
  canvasSize: MutableRefObject<{ width: number; height: number } | null>;
}) {
  const root = tgpu.initFromDevice({ device });
  canvasSize.current = {
    width: context.canvas.width,
    height: context.canvas.height,
  };

  const simParamBuffer = root
    .createBuffer(Params, {
      dt: params.dt,
      viscosity: params.viscosity,
    })
    .$usage("uniform");

  const brushParamBuffer = root
    .createBuffer(BrushParams, {
      pos: d.vec2i(0, 0),
      delta: d.vec2f(0, 0),
      radius: RADIUS,
      forceScale: FORCE_SCALE,
      inkAmount: INK_AMOUNT,
    })
    .$usage("uniform");

  function createField(name: string) {
    return root["~unstable"]
      .createTexture({ size: [SIM_N, SIM_N], format: "rgba16float" })
      .$usage("storage", "sampled") // Ensure storage usage for brush kernel
      .$name(name);
  }

  const plumsResponse = await fetch(imageAsset.uri);
  const plumsImage = await createImageBitmap(await plumsResponse.blob(), {
    resizeWidth: N,
    resizeHeight: N,
    resizeQuality: "high",
  });
  const backgroundTexture = root["~unstable"]
    .createTexture({ size: [N, N], format: "rgba8unorm" })
    .$usage("sampled", "render")
    .$name("background");

  root.device.queue.copyExternalImageToTexture(
    { source: plumsImage },
    { texture: root.unwrap(backgroundTexture) },
    [N, N],
  );

  const velTex = [createField("velocity0"), createField("velocity1")];
  const inkTex = [createField("density0"), createField("density1")];
  const pressureTex = [createField("pressure0"), createField("pressure1")];

  const newInkTex = createField("addedInk");
  const forceTex = createField("force");
  const divergenceTex = createField("divergence");

  const linSampler = tgpu["~unstable"].sampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  const brushPipeline = root["~unstable"]
    .withCompute(k.brushFn)
    .createPipeline();
  const addForcePipeline = root["~unstable"]
    .withCompute(k.addForcesFn)
    .createPipeline();
  const advectPipeline = root["~unstable"]
    .withCompute(k.advectFn)
    .createPipeline();
  const diffusionPipeline = root["~unstable"]
    .withCompute(k.diffusionFn)
    .createPipeline();
  const divergencePipeline = root["~unstable"]
    .withCompute(k.divergenceFn)
    .createPipeline();
  const pressurePipeline = root["~unstable"]
    .withCompute(k.pressureFn)
    .createPipeline();
  const projectPipeline = root["~unstable"]
    .withCompute(k.projectFn)
    .createPipeline();
  const advectInkPipeline = root["~unstable"]
    .withCompute(k.advectScalarFn)
    .createPipeline();
  const addInkPipeline = root["~unstable"]
    .withCompute(k.addInkFn)
    .createPipeline();

  // Rendering
  const velBuffer = new DoubleBuffer(velTex[0], velTex[1]);
  const inkBuffer = new DoubleBuffer(inkTex[0], inkTex[1]);
  const pressureBuffer = new DoubleBuffer(pressureTex[0], pressureTex[1]);

  const renderPipelineImage = root["~unstable"]
    .withVertex(k.renderFn, k.renderFn.shell.attributes)
    .withFragment(k.fragmentImageFn, { format: presentationFormat })
    .createPipeline();
  const renderPipelineInk = root["~unstable"]
    .withVertex(k.renderFn, k.renderFn.shell.attributes)
    .withFragment(k.fragmentInkFn, { format: presentationFormat })
    .createPipeline();
  const renderPipelineVel = root["~unstable"]
    .withVertex(k.renderFn, k.renderFn.shell.attributes)
    .withFragment(k.fragmentVelFn, { format: presentationFormat })
    .createPipeline();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: "premultiplied",
  });

  const brushBindGroup = root.createBindGroup(k.brushLayout, {
    brushParams: brushParamBuffer,
    forceDst: forceTex.createView("writeonly"),
    inkDst: newInkTex.createView("writeonly"),
  });

  const addInkBindGroups = [0, 1].map((i) => {
    const srcIdx = i;
    const dstIdx = 1 - i;
    return root.createBindGroup(k.addInkLayout, {
      src: inkTex[srcIdx].createView("sampled"),
      add: newInkTex.createView("sampled"),
      dst: inkTex[dstIdx].createView("writeonly"),
    });
  });

  const addForceBindGroups = [0, 1].map((i) => {
    const srcIdx = i;
    const dstIdx = 1 - i;
    return root.createBindGroup(k.addForcesLayout, {
      src: velTex[srcIdx].createView("sampled"),
      force: forceTex.createView("sampled"),
      dst: velTex[dstIdx].createView("writeonly"),
      simParams: simParamBuffer,
    });
  });

  const advectBindGroups = [0, 1].map((i) => {
    const srcIdx = 1 - i;
    const dstIdx = i;
    return root.createBindGroup(k.advectLayout, {
      src: velTex[srcIdx].createView("sampled"),
      dst: velTex[dstIdx].createView("writeonly"),
      simParams: simParamBuffer,
      linSampler,
    });
  });

  const diffusionBindGroups = [0, 1].map((i) => {
    const srcIdx = i;
    const dstIdx = 1 - i;
    return root.createBindGroup(k.diffusionLayout, {
      in: velTex[srcIdx].createView("sampled"),
      out: velTex[dstIdx].createView("writeonly"),
      simParams: simParamBuffer,
    });
  });

  const divergenceBindGroups = [0, 1].map((i) => {
    const srcIdx = i;
    return root.createBindGroup(k.divergenceLayout, {
      vel: velTex[srcIdx].createView("sampled"),
      div: divergenceTex.createView("writeonly"),
    });
  });

  const pressureBindGroups = [0, 1].map((i) => {
    const srcIdx = i;
    const dstIdx = 1 - i;
    return root.createBindGroup(k.pressureLayout, {
      x: pressureTex[srcIdx].createView("sampled"),
      b: divergenceTex.createView("sampled"),
      out: pressureTex[dstIdx].createView("writeonly"),
    });
  });

  const projectBindGroups = [0, 1].map((velIdx) =>
    [0, 1].map((pIdx) => {
      const srcVelIdx = velIdx;
      const dstVelIdx = 1 - velIdx;
      const srcPIdx = pIdx;
      return root.createBindGroup(k.projectLayout, {
        vel: velTex[srcVelIdx].createView("sampled"),
        p: pressureTex[srcPIdx].createView("sampled"),
        out: velTex[dstVelIdx].createView("writeonly"),
      });
    }),
  );

  const advectInkBindGroups = [0, 1].map((velIdx) =>
    [0, 1].map((inkIdx) => {
      const srcVelIdx = velIdx;
      const srcInkIdx = inkIdx;
      const dstInkIdx = 1 - inkIdx;
      return root.createBindGroup(k.advectInkLayout, {
        vel: velTex[srcVelIdx].createView("sampled"),
        src: inkTex[srcInkIdx].createView("sampled"),
        dst: inkTex[dstInkIdx].createView("writeonly"),
        simParams: simParamBuffer,
        linSampler,
      });
    }),
  );

  const renderBindGroups = {
    image: [0, 1].map((idx) =>
      root.createBindGroup(k.renderLayout, {
        result: inkTex[idx].createView("sampled"),
        background: backgroundTexture.createView("sampled"),
        linSampler,
      }),
    ),
    ink: [0, 1].map((idx) =>
      root.createBindGroup(k.renderLayout, {
        result: inkTex[idx].createView("sampled"),
        background: backgroundTexture.createView("sampled"),
        linSampler,
      }),
    ),
    velocity: [0, 1].map((idx) =>
      root.createBindGroup(k.renderLayout, {
        result: velTex[idx].createView("sampled"),
        background: backgroundTexture.createView("sampled"),
        linSampler,
      }),
    ),
  };

  function loop() {
    if (brushInfo.current?.isDown) {
      brushParamBuffer.write({
        pos: d.vec2i(...brushInfo.current.pos),
        delta: d.vec2f(...brushInfo.current.delta),
        radius: RADIUS,
        forceScale: FORCE_SCALE,
        inkAmount: INK_AMOUNT,
      });
    }

    if (brushInfo.current?.isDown) {
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

    if (showField.current === "ink") {
      pipeline = renderPipelineInk;
      renderBindGroup = renderBindGroups.ink;
    } else if (showField.current === "velocity") {
      pipeline = renderPipelineVel;
      renderBindGroup = renderBindGroups.velocity;
    } else {
      pipeline = renderPipelineImage;
      renderBindGroup = renderBindGroups.image;
    }

    pipeline
      .withColorAttachment({
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
      })
      .with(k.renderLayout, renderBindGroup[inkBuffer.currentIndex])
      .draw(6);

    root["~unstable"].flush();
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
      });
    },
    [],
  );

  const ref = useWebGPU(sceneFunction);

  const realToCanvas = useCallback((x: number, y: number): [number, number] => {
    const dpr = PixelRatio.get();
    const physicalWidth = canvasSize.current?.width ?? 0;
    const physicalHeight = canvasSize.current?.height ?? 0;

    const sx = Math.max(0, Math.min(x * dpr, physicalWidth - 1));
    const sy = Math.max(0, Math.min(y * dpr, physicalHeight - 1));

    const gx = Math.floor((sx / physicalWidth) * SIM_N);
    const gy = Math.floor(((physicalHeight - sy) / physicalHeight) * SIM_N);

    return [
      Math.max(0, Math.min(SIM_N - 1, gx)),
      Math.max(0, Math.min(SIM_N - 1, gy)),
    ];
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
      if (!brushInfo.current) return;
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

  return (
    <View style={{ width: "100%", alignItems: "center" }}>
      <Canvas
        ref={ref}
        style={{
          width: "100%",
          aspectRatio: 1,
        }}
        onTouchStart={(e) => handleStart(e)}
        onTouchMove={(e) => handleMove(e)}
        onTouchEnd={() => handleEnd()}
      />
      <View>
        <View style={{ flexDirection: "row", justifyContent: "center" }}>
          {(["ink", "velocity", "image"] as DisplayMode[]).map((field) => (
            <View
              key={field}
              style={{
                padding: 10,
                backgroundColor: showField === field ? "#ccc" : "#fff",
                borderRadius: 5,
                margin: 5,
              }}
              onStartShouldSetResponder={() => true}
              onResponderRelease={() => handleShowField(field)}
            >
              <Text>{field}</Text>
            </View>
          ))}
        </View>
      </View>
    </View>
  );
}
