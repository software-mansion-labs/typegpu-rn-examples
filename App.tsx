import { StyleSheet, View } from 'react-native';
import Boids from './examples/Boids';

export default function App() {
  return (
    <View style={styles.container}>
      <Boids />
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
