import type { SharedValue } from 'react-native-reanimated';
import {
  MOUSE_MAX_X,
  MOUSE_MIN_X,
  MOUSE_RANGE_MAX,
  MOUSE_RANGE_MIN,
  MOUSE_SMOOTHING,
  TARGET_MAX,
  TARGET_MIN,
  TARGET_OFFSET,
} from './constants.ts';

export class EventHandler {
  private mouseX = 1.0;
  private targetMouseX = 1.0;
  private isPointerDown = false;

  constructor(canvasWidth: number) {
    this.canvasWidth = canvasWidth;
  }

  updateCanvasSize(width: number) {
    this.canvasWidth = width;
  }

  handleTouch(
    isDragging: SharedValue<boolean> | undefined,
    mousePos: SharedValue<{ x: number; y: number }> | undefined,
    canvasWidth: number,
    offsetLeft: number,
  ) {
    if (isDragging?.value && mousePos?.value.x !== undefined) {
      const normalizedX = (mousePos.value.x - offsetLeft) / canvasWidth;
      this.updateTargetMouseX(normalizedX);
      this.isPointerDown = true;
    } else {
      this.isPointerDown = false;
    }
  }

  private updateTargetMouseX(normalizedX: number) {
    const clampedX = Math.max(MOUSE_MIN_X, Math.min(MOUSE_MAX_X, normalizedX));
    this.targetMouseX =
      ((clampedX - MOUSE_RANGE_MIN) / (MOUSE_RANGE_MAX - MOUSE_RANGE_MIN)) *
        (TARGET_MAX - TARGET_MIN) +
      TARGET_OFFSET;
  }

  update() {
    if (this.isPointerDown) {
      this.mouseX += (this.targetMouseX - this.mouseX) * MOUSE_SMOOTHING;
    }
  }

  get currentMouseX() {
    return this.mouseX;
  }

  get isActive() {
    return this.isPointerDown;
  }
}
