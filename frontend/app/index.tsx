import React, {useState} from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, TextInput, Button, Pressable } from 'react-native';
import { useRouter } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import { Alert } from "react-native";
import { Linking } from 'react-native';

export default function HomeScreen() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const router = useRouter();

  return (
    <View style={styles.container}>
      <Text style = {styles.title}>
        Skin Lesion Classifier
      </Text>
      <TextInput
        style = {styles.input}
        onChangeText = {setUsername}
        placeholder = "Username"
        placeholderTextColor = 'rgb(129, 129, 129)'
      />
      <TextInput
        style = {styles.input}
        onChangeText = {setPassword}
        placeholder = "Password"
        placeholderTextColor = 'rgb(129, 129, 129)'
      />
      <Pressable style={styles.loginButton} onPress={() => router.push('/home')}>
        <Text style={styles.loginButtonText}>Log In</Text>
      </Pressable>
      <Pressable onPress={() => Linking.openURL('app-settings:')}>
        <Text>App Settings</Text>
      </Pressable>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  input: {
    margin: 3,
    marginVertical: 8,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 13,
    padding: 10,
    fontSize: 16,
    width: 200,
  },
  title: {
    color: "black",
    top: -60,
    fontSize: 30,
  },
  loginButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 9,
    paddingHorizontal: 2,
    borderRadius: 8,
    marginTop: 16,
    marginBottom: 8,
    alignItems: 'center',
    width: '20%',
  },
  loginButtonText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
  },
});