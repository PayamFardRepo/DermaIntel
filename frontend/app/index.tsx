import React, { useState, useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, TextInput, Pressable, Alert, ActivityIndicator, ScrollView, Image } from 'react-native';
import { useRouter } from 'expo-router';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';

console.log('🔧 [LoginScreen] Module loaded');

export default function LoginScreen() {
  console.log('🔧 [LoginScreen] Component rendering');
  const { t } = useTranslation();
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { login, register, isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (isAuthenticated) {
      router.replace('/home');
    }
  }, [isAuthenticated]);

  const handleSubmit = async () => {
    if (!username || !password) {
      Alert.alert(t('common.error'), t('auth.fillAllFields'));
      return;
    }

    if (!isLogin && !email) {
      Alert.alert(t('common.error'), t('auth.emailRequired'));
      return;
    }

    setIsSubmitting(true);

    try {
      if (isLogin) {
        await login({ username, password });
      } else {
        await register({ username, email, password, full_name: fullName });
      }
      router.replace('/home');
    } catch (error: any) {
      Alert.alert(t('common.error'), error.message || `${isLogin ? t('auth.login') : t('auth.register')} failed`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setUsername('');
    setPassword('');
    setEmail('');
    setFullName('');
  };

  if (isLoading) {
    return (
      <View style={[styles.container, styles.centered]}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>{t('common.loading')}</Text>
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>
        {t('app.title')}
      </Text>

      <Text style={styles.subtitle}>
        {isLogin ? t('auth.welcomeBack') : t('auth.createAccount')}
      </Text>

      <TextInput
        style={styles.input}
        onChangeText={setUsername}
        value={username}
        placeholder={t('auth.username')}
        placeholderTextColor="rgb(129, 129, 129)"
        autoCapitalize="none"
      />

      {!isLogin && (
        <>
          <TextInput
            style={styles.input}
            onChangeText={setEmail}
            value={email}
            placeholder={t('auth.email')}
            placeholderTextColor="rgb(129, 129, 129)"
            keyboardType="email-address"
            autoCapitalize="none"
          />
          <TextInput
            style={styles.input}
            onChangeText={setFullName}
            value={fullName}
            placeholder={t('auth.fullName')}
            placeholderTextColor="rgb(129, 129, 129)"
          />
        </>
      )}

      <TextInput
        style={styles.input}
        onChangeText={setPassword}
        value={password}
        placeholder={t('auth.password')}
        placeholderTextColor="rgb(129, 129, 129)"
        secureTextEntry
      />

      <Pressable
        style={[styles.loginButton, isSubmitting && styles.disabledButton]}
        onPress={handleSubmit}
        disabled={isSubmitting}
      >
        {isSubmitting ? (
          <ActivityIndicator size="small" color="#fff" />
        ) : (
          <Text style={styles.loginButtonText}>
            {isLogin ? t('auth.login') : t('auth.register')}
          </Text>
        )}
      </Pressable>

      <Pressable style={styles.switchButton} onPress={toggleMode}>
        <Text style={styles.switchButtonText}>
          {isLogin ? t('auth.dontHaveAccount') : t('auth.alreadyHaveAccount')}
        </Text>
      </Pressable>

      <StatusBar style="auto" />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  centered: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  input: {
    margin: 3,
    marginVertical: 8,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 13,
    padding: 15,
    fontSize: 16,
    width: 280,
    backgroundColor: '#f8f8f8',
  },
  title: {
    color: "black",
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
  },
  subtitle: {
    color: '#666',
    fontSize: 16,
    marginBottom: 30,
    textAlign: 'center',
  },
  loginButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginTop: 20,
    marginBottom: 15,
    alignItems: 'center',
    width: 280,
  },
  disabledButton: {
    backgroundColor: '#ccc',
  },
  loginButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  switchButton: {
    marginTop: 10,
    paddingVertical: 10,
  },
  switchButtonText: {
    color: '#007AFF',
    fontSize: 14,
    textAlign: 'center',
  },
  loadingText: {
    marginTop: 10,
    color: '#666',
    fontSize: 16,
  },
});