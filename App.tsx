import { useState } from 'react';
import { Pressable, SafeAreaView, Text, View } from 'react-native';
import Boids from './examples/Boids';
import Fish from './examples/Fish/Fish';
import FluidDoubleBuffering from './examples/FluidDoubleBuffering';
import FluidWithAtomics from './examples/FluidWithAtomics';
import FunctionVisualizer from './examples/FunctionVisualizer';
import GameOfLife from './examples/GameOfLife';
import CubemapReflection from './examples/Icosphere/CubemapReflection';

const examples = ['🐠', '🚰', '🎮', '📈', '🛁', '🐥', '🧊'] as const;

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
        ) : currentExample === '🧊' ? (
          <CubemapReflection />
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
