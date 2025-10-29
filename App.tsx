import { useState } from 'react';
import { Pressable, SafeAreaView, Text, View } from 'react-native';
import Boids from './examples/Boids.tsx';
import Fish from './examples/Fish/Fish.tsx';
import FluidDoubleBuffering from './examples/FluidDoubleBuffering.tsx';
import FluidWithAtomics from './examples/FluidWithAtomics.tsx';
import FunctionVisualizer from './examples/FunctionVisualizer.tsx';
import GameOfLife from './examples/GameOfLife.tsx';

const examples = ['🐠', '🚰', '🎮', '📈', '🛁', '🐥'];

export default function App() {
  const [currentExample, setCurrentExample] =
    useState<(typeof examples)[number]>('🐠');

  return (
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
      <View style={{ flex: 1, justifyContent: 'center', position: 'static' }}>
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
        ) : null}
      </View>
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
