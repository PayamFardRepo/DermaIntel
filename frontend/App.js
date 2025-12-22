import React, {useState} from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, TextInput, Button, Pressable } from 'react-native';

export default function App() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = () => {
    if (!username || !password) {
      alert('Please enter both username and password');
      return;
    }
    // TODO: Implement actual login logic
    console.log('Login attempt:', { username, password });
  };

  const handleCreateAccount = () => {
    // TODO: Navigate to registration screen
    console.log('Create account pressed');
  };

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
        placeholder = "Password"
        placeholderTextColor = 'rgb(129, 129, 129)'
        secureTextEntry = {true}
      />
      <Pressable style={styles.loginButton} onPress={handleLogin}>
        <Text style={styles.loginButtonText}>Log In</Text>
      </Pressable>
      <Pressable onPress={handleCreateAccount} style={styles.createAccountButton}>
        <Text style={styles.createAccountText}>Create Account</Text>
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
    marginVertical: 8,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 13,
    padding: 10,
    fontSize: 16,
    width: '80%',
  },
  loginButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginTop: 28,
    marginBottom: 8,
    width: '80%',
    alignItems: 'center',
  },
  loginButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  createAccountButton: {
    marginTop: 16,
    paddingVertical: 8,
  },
  createAccountText: {
    color: '#007AFF',
    fontSize: 16,
    textDecorationLine: 'underline',
  },
});
