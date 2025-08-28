import { useRouter } from 'expo-router';
import { View, Text, Pressable } from 'react-native';

export default function TestRouter() {
  const router = useRouter();

  return (
    <View>
      <Text>Test Page</Text>
      <Pressable onPress={() => router.push('/')}>
        <Text>Go to Home</Text>
      </Pressable>
    </View>
  );
}