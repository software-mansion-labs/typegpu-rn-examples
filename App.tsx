import { StyleSheet, View } from 'react-native';
// import Boids from './examples/Boids';
// import GameOfLife from './examples/GameOfLife';
import FluidsDoubleBuffering from './examples/FluidsDoubleBuffering';

export default function App() {
  return (
    <View style={styles.container}>
      {/* <Boids /> */}
      {/* <GameOfLife /> */}
      <FluidsDoubleBuffering />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'rgb(239 239 249)',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
