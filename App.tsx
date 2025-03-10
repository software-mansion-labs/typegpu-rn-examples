import { StyleSheet, View } from 'react-native';
// import Boids from './examples/Boids';
import GameOfLife from './examples/GameOfLife';

export default function App() {
  return (
    <View style={styles.container}>
      {/* <Boids /> */}
      <GameOfLife />
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
