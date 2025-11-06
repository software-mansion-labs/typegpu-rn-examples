import { useEffect, useRef } from "react";
import { PixelRatio } from "react-native";
import { type NativeCanvas, useCanvasRef, useDevice } from "react-native-wgpu";

interface SceneProps {
  context: GPUCanvasContext;
  device: GPUDevice;
  gpu: GPU;
  presentationFormat: GPUTextureFormat;
  canvas: NativeCanvas;
}

type RenderScene = (timestamp: number) => void;
type Scene = (props: SceneProps) => RenderScene | Promise<RenderScene>;

export const useWebGPU = (scene: Scene) => {
  const { device } = useDevice();
  const canvasRef = useCanvasRef();
  const animationFrameId = useRef<number | null>(null);
  useEffect(() => {
    (async () => {
      const ref = canvasRef.current;
      if (!ref || !device) {
        return;
      }

      const context = ref.getContext("webgpu");
      if (!context) {
        throw new Error("Failed to get WebGPU context from canvas.");
      }

      const canvas = context.canvas as HTMLCanvasElement;
      const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
      canvas.width = canvas.clientWidth * PixelRatio.get();
      canvas.height = canvas.clientHeight * PixelRatio.get();
      context.configure({
        device,
        format: presentationFormat,
        alphaMode: "premultiplied",
      });

      const sceneProps: SceneProps = {
        context,
        device,
        gpu: navigator.gpu,
        presentationFormat,
        canvas: context.canvas as unknown as NativeCanvas,
      };

      const r: RenderScene | Promise<RenderScene> = (
        scene instanceof Promise
          ? await withValidate(device, scene)(sceneProps)
          : withValidate(device, scene)(sceneProps)
      ) as RenderScene | Promise<RenderScene>;

      let renderScene: RenderScene;
      if (r instanceof Promise) {
        renderScene = await r;
      } else {
        renderScene = r as RenderScene;
      }
      if (typeof renderScene === "function") {
        const render = () => {
          const timestamp = Date.now();
          renderScene(timestamp);
          context.present();
          animationFrameId.current = requestAnimationFrame(render);
        };

        animationFrameId.current = requestAnimationFrame(render);
      }
    })();
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [canvasRef, device, scene]);
  return canvasRef;
};

/*
 * Dev utility to wrap GPU calls with error validation.
 * If not used the errors will not appear in console (unlike in web).
 */
export function withValidate<T extends unknown[], R>(
  device: GPUDevice,
  fn: (...args: T) => R,
) {
  return (...args: T): R => {
    const scopes: GPUErrorFilter[] = [
      "validation",
      "out-of-memory",
      "internal",
    ];
    for (const scope of scopes) {
      device.pushErrorScope(scope);
    }

    const result = fn(...args);

    for (const scope of scopes.reverse()) {
      device.popErrorScope().then((error) => {
        if (error) {
          console.error(`GPU Error [${scope}]:`, error.message);
        }
      });
    }

    return result;
  };
}
