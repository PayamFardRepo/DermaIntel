import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter, useLocalSearchParams } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import { API_BASE_URL } from '../config';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

interface ChatStatus {
  available: boolean;
  openai_installed: boolean;
  api_key_configured: boolean;
  model: string | null;
}

export default function AIChatScreen() {
  const router = useRouter();
  const params = useLocalSearchParams();
  const analysisId = params.analysisId ? parseInt(params.analysisId as string) : null;
  const diagnosis = params.diagnosis as string || null;
  const confidence = params.confidence as string || null;

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatStatus, setChatStatus] = useState<ChatStatus | null>(null);
  const [isCheckingStatus, setIsCheckingStatus] = useState(true);
  const scrollViewRef = useRef<ScrollView>(null);

  useEffect(() => {
    checkChatStatus();
    // Add initial context message if we have analysis data
    if (diagnosis) {
      const contextMessage: Message = {
        id: 'context-1',
        role: 'system',
        content: `Discussing analysis: ${diagnosis}${confidence ? ` (${confidence}% confidence)` : ''}`,
        timestamp: new Date(),
      };
      setMessages([contextMessage]);
    }
  }, []);

  const checkChatStatus = async () => {
    try {
      const token = await SecureStore.getItemAsync('auth_token');
      const response = await fetch(`${API_BASE_URL}/ai-chat/status`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const status = await response.json();
        setChatStatus(status);

        if (!status.available) {
          // Add helpful message about setup
          const setupMessage: Message = {
            id: 'setup-1',
            role: 'assistant',
            content: status.api_key_configured
              ? 'AI chat is not available. Please check the server configuration.'
              : 'AI chat requires an OpenAI API key. Please ask your administrator to configure the OPENAI_API_KEY environment variable on the server.\n\nIn the meantime, you can still ask common questions about skin conditions!',
            timestamp: new Date(),
          };
          setMessages(prev => [...prev, setupMessage]);
        } else {
          // Add welcome message
          const welcomeMessage: Message = {
            id: 'welcome-1',
            role: 'assistant',
            content: diagnosis
              ? `I can help you understand your skin analysis results for "${diagnosis}". Feel free to ask me:\n\n- What does this diagnosis mean?\n- Should I be concerned?\n- What are the next steps?\n- What treatments are available?\n\nHow can I help you today?`
              : 'Hello! I\'m your AI dermatology assistant. I can help you understand skin conditions, explain diagnoses, and answer questions about skin health.\n\nWhat would you like to know?',
            timestamp: new Date(),
          };
          setMessages(prev => [...prev, welcomeMessage]);
        }
      }
    } catch (error) {
      console.error('Error checking chat status:', error);
      setChatStatus({ available: false, openai_installed: false, api_key_configured: false, model: null });
    } finally {
      setIsCheckingStatus(false);
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: inputText.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    // Scroll to bottom
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);

    try {
      const token = await SecureStore.getItemAsync('auth_token');

      // Build conversation history (exclude system messages)
      const conversationHistory = messages
        .filter(m => m.role !== 'system')
        .map(m => ({ role: m.role, content: m.content }));

      const formData = new FormData();
      formData.append('message', userMessage.content);
      if (analysisId) {
        formData.append('analysis_id', analysisId.toString());
      }
      formData.append('conversation_history', JSON.stringify(conversationHistory));

      // Use quick endpoint for fallback support
      const endpoint = chatStatus?.available ? '/ai-chat' : '/ai-chat/quick';

      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();

        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: data.response,
          timestamp: new Date(),
        };

        setMessages(prev => [...prev, assistantMessage]);
      } else {
        const errorData = await response.json();
        Alert.alert('Error', errorData.detail || 'Failed to get response');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      Alert.alert('Error', 'Network error. Please check your connection.');
    } finally {
      setIsLoading(false);
      setTimeout(() => {
        scrollViewRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  };

  const suggestedQuestions = diagnosis
    ? [
        `What is ${diagnosis}?`,
        'Should I be worried?',
        'What are the treatment options?',
        'When should I see a doctor?',
      ]
    : [
        'What is melanoma?',
        'How do I check my moles?',
        'What causes skin cancer?',
        'How can I protect my skin?',
      ];

  const handleSuggestedQuestion = (question: string) => {
    setInputText(question);
  };

  const renderMessage = (message: Message) => {
    if (message.role === 'system') {
      return (
        <View key={message.id} style={styles.systemMessageContainer}>
          <View style={styles.systemMessage}>
            <Ionicons name="information-circle" size={16} color="#6b7280" />
            <Text style={styles.systemMessageText}>{message.content}</Text>
          </View>
        </View>
      );
    }

    const isUser = message.role === 'user';

    return (
      <View
        key={message.id}
        style={[
          styles.messageContainer,
          isUser ? styles.userMessageContainer : styles.assistantMessageContainer,
        ]}
      >
        {!isUser && (
          <View style={styles.avatarContainer}>
            <LinearGradient
              colors={['#0ea5e9', '#0284c7']}
              style={styles.avatar}
            >
              <Ionicons name="medical" size={16} color="white" />
            </LinearGradient>
          </View>
        )}
        <View
          style={[
            styles.messageBubble,
            isUser ? styles.userBubble : styles.assistantBubble,
          ]}
        >
          <Text style={[styles.messageText, isUser && styles.userMessageText]}>
            {message.content}
          </Text>
          <Text style={[styles.timestamp, isUser && styles.userTimestamp]}>
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </Text>
        </View>
        {isUser && (
          <View style={styles.avatarContainer}>
            <View style={[styles.avatar, styles.userAvatar]}>
              <Ionicons name="person" size={16} color="white" />
            </View>
          </View>
        )}
      </View>
    );
  };

  if (isCheckingStatus) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0ea5e9" />
        <Text style={styles.loadingText}>Initializing AI Chat...</Text>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <LinearGradient colors={['#0c4a6e', '#0369a1', '#0ea5e9']} style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="white" />
        </TouchableOpacity>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>AI Assistant</Text>
          <Text style={styles.headerSubtitle}>
            {chatStatus?.available ? `Powered by ${chatStatus.model || 'GPT-4'}` : 'Limited Mode'}
          </Text>
        </View>
        <View style={styles.statusIndicator}>
          <View style={[styles.statusDot, chatStatus?.available ? styles.statusOnline : styles.statusOffline]} />
        </View>
      </LinearGradient>

      <ScrollView
        ref={scrollViewRef}
        style={styles.messagesContainer}
        contentContainerStyle={styles.messagesContent}
        onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
      >
        {messages.map(renderMessage)}

        {isLoading && (
          <View style={styles.loadingMessage}>
            <View style={styles.avatarContainer}>
              <LinearGradient colors={['#0ea5e9', '#0284c7']} style={styles.avatar}>
                <Ionicons name="medical" size={16} color="white" />
              </LinearGradient>
            </View>
            <View style={styles.typingIndicator}>
              <ActivityIndicator size="small" color="#0ea5e9" />
              <Text style={styles.typingText}>Thinking...</Text>
            </View>
          </View>
        )}

        {messages.length <= 2 && (
          <View style={styles.suggestionsContainer}>
            <Text style={styles.suggestionsTitle}>Suggested questions:</Text>
            <View style={styles.suggestionsGrid}>
              {suggestedQuestions.map((question, index) => (
                <TouchableOpacity
                  key={index}
                  style={styles.suggestionChip}
                  onPress={() => handleSuggestedQuestion(question)}
                >
                  <Text style={styles.suggestionText}>{question}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}
      </ScrollView>

      <View style={styles.inputContainer}>
        <View style={styles.inputWrapper}>
          <TextInput
            style={styles.input}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Ask about your skin analysis..."
            placeholderTextColor="#9ca3af"
            multiline
            maxLength={500}
            editable={!isLoading}
          />
          <TouchableOpacity
            style={[styles.sendButton, (!inputText.trim() || isLoading) && styles.sendButtonDisabled]}
            onPress={sendMessage}
            disabled={!inputText.trim() || isLoading}
          >
            <Ionicons
              name="send"
              size={20}
              color={inputText.trim() && !isLoading ? 'white' : '#9ca3af'}
            />
          </TouchableOpacity>
        </View>
        <Text style={styles.disclaimer}>
          AI responses are for informational purposes only. Consult a doctor for medical advice.
        </Text>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f8fafc',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#64748b',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: Platform.OS === 'ios' ? 50 : 30,
    paddingBottom: 16,
    paddingHorizontal: 16,
  },
  backButton: {
    padding: 8,
  },
  headerContent: {
    flex: 1,
    marginLeft: 12,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: 'white',
  },
  headerSubtitle: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 2,
  },
  statusIndicator: {
    padding: 8,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  statusOnline: {
    backgroundColor: '#22c55e',
  },
  statusOffline: {
    backgroundColor: '#f59e0b',
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    padding: 16,
    paddingBottom: 24,
  },
  systemMessageContainer: {
    alignItems: 'center',
    marginBottom: 16,
  },
  systemMessage: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f1f5f9',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
  },
  systemMessageText: {
    fontSize: 12,
    color: '#6b7280',
    marginLeft: 6,
  },
  messageContainer: {
    flexDirection: 'row',
    marginBottom: 16,
    alignItems: 'flex-end',
  },
  userMessageContainer: {
    justifyContent: 'flex-end',
  },
  assistantMessageContainer: {
    justifyContent: 'flex-start',
  },
  avatarContainer: {
    marginHorizontal: 8,
  },
  avatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  userAvatar: {
    backgroundColor: '#6366f1',
  },
  messageBubble: {
    maxWidth: '70%',
    padding: 12,
    borderRadius: 16,
  },
  userBubble: {
    backgroundColor: '#0ea5e9',
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    backgroundColor: 'white',
    borderBottomLeftRadius: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  messageText: {
    fontSize: 15,
    lineHeight: 22,
    color: '#1f2937',
  },
  userMessageText: {
    color: 'white',
  },
  timestamp: {
    fontSize: 10,
    color: '#9ca3af',
    marginTop: 6,
    alignSelf: 'flex-end',
  },
  userTimestamp: {
    color: 'rgba(255,255,255,0.7)',
  },
  loadingMessage: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  typingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  typingText: {
    marginLeft: 8,
    fontSize: 14,
    color: '#64748b',
  },
  suggestionsContainer: {
    marginTop: 16,
    padding: 16,
    backgroundColor: 'white',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  suggestionsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 12,
  },
  suggestionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  suggestionChip: {
    backgroundColor: '#f0f9ff',
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#bae6fd',
  },
  suggestionText: {
    fontSize: 13,
    color: '#0369a1',
  },
  inputContainer: {
    padding: 16,
    backgroundColor: 'white',
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    backgroundColor: '#f1f5f9',
    borderRadius: 24,
    paddingLeft: 16,
    paddingRight: 4,
    paddingVertical: 4,
  },
  input: {
    flex: 1,
    fontSize: 15,
    color: '#1f2937',
    maxHeight: 100,
    paddingVertical: 10,
  },
  sendButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#0ea5e9',
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#e2e8f0',
  },
  disclaimer: {
    fontSize: 10,
    color: '#9ca3af',
    textAlign: 'center',
    marginTop: 8,
  },
});
