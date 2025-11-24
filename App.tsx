import { useState } from 'react';
import { Pressable, Text, View } from 'react-native';
import {
  Gesture,
  GestureDetector,
  GestureHandlerRootView,
} from 'react-native-gesture-handler';
import Animated, { useSharedValue } from 'react-native-reanimated';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import Boids from './examples/Boids.tsx';
import Fish from './examples/Fish/Fish.tsx';
import FluidDoubleBuffering from './examples/FluidDoubleBuffering.tsx';
import FluidWithAtomics from './examples/FluidWithAtomics.tsx';
import FunctionVisualizer from './examples/FunctionVisualizer.tsx';
import GameOfLife from './examples/GameOfLife.tsx';
import Jelly from './examples/Jelly/Jelly.tsx';

const examples = ['🐠', '🚰', '🎮', '📈', '🛁', '🐥', '🪼'] as const;

export default function App() {
  const [currentExample, setCurrentExample] =
    useState<(typeof examples)[number]>('🪼');
  const isDragging = useSharedValue(false);
  const mousePos = useSharedValue({ x: 0, y: 0 });
  const gesture = Gesture.Pan()
    .onBegin(() => {
      isDragging.value = true;
    })
    .onUpdate((e) => {
      mousePos.value = { x: e.x, y: e.y };
    })
    .onEnd(() => {
      isDragging.value = false;
    });

  return (
    <SafeAreaProvider>
      <GestureHandlerRootView>
        <SafeAreaView
          style={{
            position: 'static',
            flex: 1,
            backgroundColor: 'rgb(239 239 249)',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 30,
          }}
        >
          <GestureDetector gesture={gesture}>
            <Animated.View
              style={{
                flex: 1,
                alignItems: 'center',
                justifyContent: 'center',
                position: 'static',
                width: '100%',
              }}
            >
              {currentExample === '🐥' ? (
                <Boids />
              ) : currentExample === '🛁' ? (
                <FluidDoubleBuffering />
              ) : currentExample === '🚰' ? (
                <FluidWithAtomics />
              ) : currentExample === '🎮' ? (
                <GameOfLife />
              ) : currentExample === '📈' ? (
                <FunctionVisualizer />
              ) : currentExample === '🐠' ? (
                <Fish />
              ) : currentExample === '🪼' ? (
                <Jelly isDragging={isDragging} mousePos={mousePos} />
              ) : null}
            </Animated.View>
          </GestureDetector>
          <View
            style={{
              flexDirection: 'row',
              gap: 20,
              paddingVertical: 40,
              alignItems: 'center',
              zIndex: 40,
            }}
          >
            {examples.map((example) => (
              <Pressable
                key={example}
                onPress={() => setCurrentExample(example)}
              >
                <Text
                  style={{ fontSize: currentExample === example ? 50 : 30 }}
                >
                  {example}
                </Text>
              </Pressable>
            ))}
          </View>
        </SafeAreaView>
      </GestureHandlerRootView>
    </SafeAreaProvider>
  );
}
