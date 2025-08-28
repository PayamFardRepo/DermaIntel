import React, {useState} from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, TextInput, Button, Pressable } from 'react-native';

export default function App() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  return (
    <View style={styles.container}>
      <TextInput
        style = {styles.input}
        onChangeText = {setUsername}
        placeholder = "Username"
        placeholderTextColor = 'rgb(129, 129, 129)'
      />
      <TextInput
        style = {styles.input}
        onChangeText = {setPassword}
        placeholder = "Passwoasdrd"
        placeholderTextColor = 'rgb(129, 129, 129)'
      />
      <Pressable style={styles.loginButton}>
        <Text style={styles.loginButtonText}>Log In</Text>
      </Pressable>
      <Pressable>
        <Text>Create Account</Text>
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
    borderWidth: 1,
    padding: 5,

    marginVertical: 8,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 13,
    padding: 10,
    fontSize: 16,
    
  },
  loginButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginTop: 28,
    marginBottom: 8,
    width: '7%',
    alignItems: 'center',
  },
  loginButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
