import { useState } from 'react';
import { Pressable, SafeAreaView, Text, View } from 'react-native';
import Boids from './examples/Boids';
import FluidDoubleBuffering from './examples/FluidDoubleBuffering';
import FluidWithAtomics from './examples/FluidWithAtomics';
import FunctionVisualizer from './examples/FunctionVisualizer';
import GameOfLife from './examples/GameOfLife';

const examples = ['🐥', '🛁', '🚰', '🎮', '📈'];

export default function App() {
  const [currentExample, setCurrentExample] =
    useState<(typeof examples)[number]>('📈');

  return (
    <SafeAreaView
      style={{
        flex: 1,
        backgroundColor: 'rgb(239 239 249)',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <View style={{ flex: 1, justifyContent: 'center' }}>
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
        ) : null}
      </View>
      <View
        style={{
          flexDirection: 'row',
          gap: 20,
          paddingVertical: 40,
          alignItems: 'center',
        }}
      >
        {examples.map((example) => (
          <Pressable key={example} onPress={() => setCurrentExample(example)}>
            <Text style={{ fontSize: currentExample === example ? 50 : 30 }}>
              {example}
            </Text>
          </Pressable>
        ))}
      </View>
    </SafeAreaView>
  );
}
