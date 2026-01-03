import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Dimensions,
  Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import { API_BASE_URL } from '../config';
import AuthService from '../services/AuthService';
import { useAuth } from '../contexts/AuthContext';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

interface ImageFeature {
  type: 'concern' | 'neutral' | 'positive';
  label: string;
  description: string;
  region: string;
}

interface ImageAnalysis {
  description: string;
  features: ImageFeature[];
  initial_impression: string;
  confidence_level: string;
  urgency: string;
  suggested_questions: string[];
}

interface ConsultationState {
  sessionId: string | null;
  imageUri: string | null;
  imageAnalysis: ImageAnalysis | null;
  messages: Message[];
  stage: 'upload' | 'analyzing' | 'conversation' | 'assessment';
  isLoading: boolean;
  questionsAsked: number;
}

export default function AIConsultationScreen() {
  const router = useRouter();
  const { user } = useAuth();
  const scrollViewRef = useRef<ScrollView>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;

  const [state, setState] = useState<ConsultationState>({
    sessionId: null,
    imageUri: null,
    imageAnalysis: null,
    messages: [],
    stage: 'upload',
    isLoading: false,
    questionsAsked: 0,
  });

  const [inputText, setInputText] = useState('');
  const [showFeatures, setShowFeatures] = useState(false);
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);

  // Pulse animation for the AI avatar
  useEffect(() => {
    if (state.isLoading) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.1,
            duration: 800,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 800,
            useNativeDriver: true,
          }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [state.isLoading]);

  // Check availability on mount
  useEffect(() => {
    checkAvailability();
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  }, [state.messages]);

  const checkAvailability = async () => {
    try {
      const token = AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/ai-consultation/status`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setIsAvailable(data.available);
      } else {
        setIsAvailable(false);
      }
    } catch (error) {
      console.error('Failed to check AI consultation status:', error);
      setIsAvailable(false);
    }
  };

  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      alert('Permission to access camera roll is required!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setState(prev => ({
        ...prev,
        imageUri: result.assets[0].uri,
      }));
    }
  };

  const takePhoto = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (!permissionResult.granted) {
      alert('Permission to access camera is required!');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setState(prev => ({
        ...prev,
        imageUri: result.assets[0].uri,
      }));
    }
  };

  const startConsultation = async () => {
    if (!state.imageUri) return;

    setState(prev => ({ ...prev, stage: 'analyzing', isLoading: true }));

    try {
      const token = AuthService.getToken();

      // Create form data with image
      const formData = new FormData();
      const imageUri = state.imageUri;
      const filename = imageUri.split('/').pop() || 'image.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : 'image/jpeg';

      formData.append('image', {
        uri: imageUri,
        name: filename,
        type,
      } as any);

      // Add clinical context if available
      if (user) {
        const clinicalContext = {
          age: user.age,
          skin_type: user.skin_type,
        };
        formData.append('clinical_context', JSON.stringify(clinicalContext));
      }

      const response = await fetch(`${API_BASE_URL}/ai-consultation/start`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to start consultation');
      }

      const data = await response.json();

      // Add the initial AI message
      const initialMessage: Message = {
        id: `ai-${Date.now()}`,
        role: 'assistant',
        content: data.initial_message,
        timestamp: new Date(),
      };

      setState(prev => ({
        ...prev,
        sessionId: data.session_id,
        imageAnalysis: data.image_analysis,
        messages: [initialMessage],
        stage: 'conversation',
        isLoading: false,
        questionsAsked: 1,
      }));

    } catch (error) {
      console.error('Failed to start consultation:', error);
      setState(prev => ({
        ...prev,
        stage: 'upload',
        isLoading: false,
        messages: [{
          id: 'error',
          role: 'system',
          content: 'Failed to start consultation. Please try again.',
          timestamp: new Date(),
        }],
      }));
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim() || !state.sessionId || state.isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: inputText.trim(),
      timestamp: new Date(),
    };

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      isLoading: true,
    }));
    setInputText('');

    try {
      const token = AuthService.getToken();

      const formData = new FormData();
      formData.append('session_id', state.sessionId);
      formData.append('message', userMessage.content);

      const response = await fetch(`${API_BASE_URL}/ai-consultation/respond`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const data = await response.json();

      const aiMessage: Message = {
        id: `ai-${Date.now()}`,
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
      };

      setState(prev => ({
        ...prev,
        messages: [...prev.messages, aiMessage],
        stage: data.stage === 'assessment' ? 'assessment' : 'conversation',
        isLoading: false,
        questionsAsked: data.questions_asked,
      }));

    } catch (error) {
      console.error('Failed to send message:', error);
      setState(prev => ({
        ...prev,
        isLoading: false,
        messages: [...prev.messages, {
          id: 'error',
          role: 'system',
          content: 'Failed to send message. Please try again.',
          timestamp: new Date(),
        }],
      }));
    }
  };

  const generateSummary = async () => {
    if (!state.sessionId) return;

    setState(prev => ({ ...prev, isLoading: true }));

    try {
      const token = AuthService.getToken();

      const formData = new FormData();
      formData.append('session_id', state.sessionId);

      const response = await fetch(`${API_BASE_URL}/ai-consultation/generate-summary`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to generate summary');
      }

      const data = await response.json();

      const summaryMessage: Message = {
        id: `summary-${Date.now()}`,
        role: 'assistant',
        content: `**Clinical Summary for Your Dermatologist**\n\n${data.summary}\n\n---\n*${data.disclaimer}*`,
        timestamp: new Date(),
      };

      setState(prev => ({
        ...prev,
        messages: [...prev.messages, summaryMessage],
        isLoading: false,
      }));

    } catch (error) {
      console.error('Failed to generate summary:', error);
      setState(prev => ({ ...prev, isLoading: false }));
    }
  };

  const resetConsultation = () => {
    setState({
      sessionId: null,
      imageUri: null,
      imageAnalysis: null,
      messages: [],
      stage: 'upload',
      isLoading: false,
      questionsAsked: 0,
    });
    setShowFeatures(false);
  };

  const getFeatureColor = (type: string) => {
    switch (type) {
      case 'concern': return '#ef4444';
      case 'positive': return '#22c55e';
      default: return '#f59e0b';
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'high': return '#ef4444';
      case 'moderate': return '#f59e0b';
      default: return '#22c55e';
    }
  };

  const renderUploadStage = () => (
    <View style={styles.uploadContainer}>
      <View style={styles.welcomeSection}>
        <Animated.View style={[styles.aiAvatarLarge, { transform: [{ scale: pulseAnim }] }]}>
          <LinearGradient
            colors={['#6366f1', '#8b5cf6']}
            style={styles.aiAvatarGradient}
          >
            <Ionicons name="medical" size={48} color="#fff" />
          </LinearGradient>
        </Animated.View>
        <Text style={styles.welcomeTitle}>AI Dermatology Consultation</Text>
        <Text style={styles.welcomeSubtitle}>
          Upload or take a photo of your skin concern, and I'll guide you through a personalized consultation
        </Text>
      </View>

      {state.imageUri ? (
        <View style={styles.imagePreviewContainer}>
          <Image source={{ uri: state.imageUri }} style={styles.imagePreview} />
          <TouchableOpacity
            style={styles.removeImageButton}
            onPress={() => setState(prev => ({ ...prev, imageUri: null }))}
          >
            <Ionicons name="close-circle" size={28} color="#ef4444" />
          </TouchableOpacity>
        </View>
      ) : (
        <View style={styles.uploadOptions}>
          <TouchableOpacity style={styles.uploadButton} onPress={takePhoto}>
            <LinearGradient
              colors={['#6366f1', '#8b5cf6']}
              style={styles.uploadButtonGradient}
            >
              <Ionicons name="camera" size={32} color="#fff" />
              <Text style={styles.uploadButtonText}>Take Photo</Text>
            </LinearGradient>
          </TouchableOpacity>

          <TouchableOpacity style={styles.uploadButton} onPress={pickImage}>
            <View style={styles.uploadButtonOutline}>
              <Ionicons name="images" size={32} color="#6366f1" />
              <Text style={styles.uploadButtonTextOutline}>Choose Photo</Text>
            </View>
          </TouchableOpacity>
        </View>
      )}

      {state.imageUri && (
        <TouchableOpacity
          style={styles.startButton}
          onPress={startConsultation}
        >
          <LinearGradient
            colors={['#6366f1', '#8b5cf6']}
            style={styles.startButtonGradient}
          >
            <Text style={styles.startButtonText}>Start Consultation</Text>
            <Ionicons name="arrow-forward" size={20} color="#fff" />
          </LinearGradient>
        </TouchableOpacity>
      )}

      <View style={styles.disclaimerBox}>
        <Ionicons name="information-circle" size={20} color="#6b7280" />
        <Text style={styles.disclaimerText}>
          This AI consultation is for informational purposes only and does not replace professional medical advice.
        </Text>
      </View>
    </View>
  );

  const renderAnalyzingStage = () => (
    <View style={styles.analyzingContainer}>
      <Image source={{ uri: state.imageUri! }} style={styles.analyzingImage} />
      <View style={styles.analyzingOverlay}>
        <ActivityIndicator size="large" color="#fff" />
        <Text style={styles.analyzingText}>Analyzing your image...</Text>
        <Text style={styles.analyzingSubtext}>
          I'm examining the details to provide you with the best guidance
        </Text>
      </View>
    </View>
  );

  const renderConversationStage = () => (
    <View style={styles.conversationContainer}>
      {/* Image with visual features toggle */}
      <View style={styles.imageSection}>
        <Image source={{ uri: state.imageUri! }} style={styles.conversationImage} />

        {/* Feature annotations overlay */}
        {showFeatures && state.imageAnalysis?.features && (
          <View style={styles.featuresOverlay}>
            {state.imageAnalysis.features.map((feature, index) => (
              <View
                key={index}
                style={[
                  styles.featureTag,
                  { backgroundColor: getFeatureColor(feature.type) + '20', borderColor: getFeatureColor(feature.type) }
                ]}
              >
                <View style={[styles.featureDot, { backgroundColor: getFeatureColor(feature.type) }]} />
                <Text style={[styles.featureLabel, { color: getFeatureColor(feature.type) }]}>
                  {feature.label}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Toggle features button */}
        <TouchableOpacity
          style={styles.toggleFeaturesButton}
          onPress={() => setShowFeatures(!showFeatures)}
        >
          <Ionicons
            name={showFeatures ? 'eye-off' : 'eye'}
            size={18}
            color="#fff"
          />
          <Text style={styles.toggleFeaturesText}>
            {showFeatures ? 'Hide' : 'Show'} Analysis
          </Text>
        </TouchableOpacity>

        {/* Urgency indicator */}
        {state.imageAnalysis && (
          <View style={[
            styles.urgencyBadge,
            { backgroundColor: getUrgencyColor(state.imageAnalysis.urgency) }
          ]}>
            <Text style={styles.urgencyText}>
              {state.imageAnalysis.urgency.charAt(0).toUpperCase() + state.imageAnalysis.urgency.slice(1)} Priority
            </Text>
          </View>
        )}
      </View>

      {/* Progress indicator */}
      <View style={styles.progressContainer}>
        <View style={styles.progressBar}>
          <View
            style={[
              styles.progressFill,
              { width: `${Math.min((state.questionsAsked / 4) * 100, 100)}%` }
            ]}
          />
        </View>
        <Text style={styles.progressText}>
          {state.stage === 'assessment' ? 'Assessment Complete' : `Question ${state.questionsAsked} of ~3-4`}
        </Text>
      </View>

      {/* Messages */}
      <ScrollView
        ref={scrollViewRef}
        style={styles.messagesContainer}
        contentContainerStyle={styles.messagesContent}
      >
        {state.messages.map((message) => (
          <View
            key={message.id}
            style={[
              styles.messageBubble,
              message.role === 'user' ? styles.userMessage : styles.aiMessage,
              message.role === 'system' && styles.systemMessage,
            ]}
          >
            {message.role === 'assistant' && (
              <View style={styles.aiAvatarSmall}>
                <LinearGradient
                  colors={['#6366f1', '#8b5cf6']}
                  style={styles.aiAvatarSmallGradient}
                >
                  <Ionicons name="medical" size={16} color="#fff" />
                </LinearGradient>
              </View>
            )}
            <View style={[
              styles.messageContent,
              message.role === 'user' && styles.userMessageContent,
            ]}>
              <Text style={[
                styles.messageText,
                message.role === 'user' && styles.userMessageText,
              ]}>
                {message.content}
              </Text>
            </View>
          </View>
        ))}

        {state.isLoading && (
          <View style={styles.typingIndicator}>
            <View style={styles.aiAvatarSmall}>
              <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
                <LinearGradient
                  colors={['#6366f1', '#8b5cf6']}
                  style={styles.aiAvatarSmallGradient}
                >
                  <Ionicons name="medical" size={16} color="#fff" />
                </LinearGradient>
              </Animated.View>
            </View>
            <View style={styles.typingDots}>
              <View style={styles.typingDot} />
              <View style={[styles.typingDot, styles.typingDotMiddle]} />
              <View style={styles.typingDot} />
            </View>
          </View>
        )}
      </ScrollView>

      {/* Input area */}
      {state.stage !== 'assessment' ? (
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.inputContainer}
        >
          <TextInput
            style={styles.textInput}
            placeholder="Type your response..."
            placeholderTextColor="#9ca3af"
            value={inputText}
            onChangeText={setInputText}
            multiline
            maxLength={500}
            editable={!state.isLoading}
          />
          <TouchableOpacity
            style={[styles.sendButton, (!inputText.trim() || state.isLoading) && styles.sendButtonDisabled]}
            onPress={sendMessage}
            disabled={!inputText.trim() || state.isLoading}
          >
            <Ionicons
              name="send"
              size={20}
              color={(!inputText.trim() || state.isLoading) ? '#9ca3af' : '#fff'}
            />
          </TouchableOpacity>
        </KeyboardAvoidingView>
      ) : (
        <View style={styles.assessmentActions}>
          <TouchableOpacity
            style={styles.summaryButton}
            onPress={generateSummary}
            disabled={state.isLoading}
          >
            <LinearGradient
              colors={['#6366f1', '#8b5cf6']}
              style={styles.summaryButtonGradient}
            >
              <Ionicons name="document-text" size={20} color="#fff" />
              <Text style={styles.summaryButtonText}>Generate Summary for Doctor</Text>
            </LinearGradient>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.newConsultButton}
            onPress={resetConsultation}
          >
            <Ionicons name="add-circle-outline" size={20} color="#6366f1" />
            <Text style={styles.newConsultButtonText}>New Consultation</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );

  // Unavailable state
  if (isAvailable === false) {
    return (
      <View style={styles.container}>
        <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.gradient}>
          <View style={styles.header}>
            <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
              <Ionicons name="arrow-back" size={24} color="#fff" />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>AI Consultation</Text>
            <View style={{ width: 40 }} />
          </View>

          <View style={styles.unavailableContainer}>
            <Ionicons name="cloud-offline" size={64} color="#6b7280" />
            <Text style={styles.unavailableTitle}>AI Consultation Unavailable</Text>
            <Text style={styles.unavailableText}>
              This feature requires OpenAI API configuration. Please contact your administrator.
            </Text>
            <TouchableOpacity style={styles.goBackButton} onPress={() => router.back()}>
              <Text style={styles.goBackButtonText}>Go Back</Text>
            </TouchableOpacity>
          </View>
        </LinearGradient>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.gradient}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => state.stage === 'upload' ? router.back() : resetConsultation()}
            style={styles.backButton}
          >
            <Ionicons
              name={state.stage === 'upload' ? 'arrow-back' : 'close'}
              size={24}
              color="#fff"
            />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>AI Consultation</Text>
          <View style={{ width: 40 }} />
        </View>

        {/* Main content based on stage */}
        {state.stage === 'upload' && renderUploadStage()}
        {state.stage === 'analyzing' && renderAnalyzingStage()}
        {(state.stage === 'conversation' || state.stage === 'assessment') && renderConversationStage()}
      </LinearGradient>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  gradient: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 16,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.1)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#fff',
  },

  // Upload Stage
  uploadContainer: {
    flex: 1,
    padding: 20,
  },
  welcomeSection: {
    alignItems: 'center',
    marginBottom: 32,
  },
  aiAvatarLarge: {
    marginBottom: 20,
  },
  aiAvatarGradient: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
  },
  welcomeTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#fff',
    marginBottom: 12,
    textAlign: 'center',
  },
  welcomeSubtitle: {
    fontSize: 16,
    color: '#9ca3af',
    textAlign: 'center',
    lineHeight: 24,
    paddingHorizontal: 20,
  },
  imagePreviewContainer: {
    alignItems: 'center',
    marginBottom: 24,
  },
  imagePreview: {
    width: SCREEN_WIDTH - 80,
    height: SCREEN_WIDTH - 80,
    borderRadius: 16,
  },
  removeImageButton: {
    position: 'absolute',
    top: -10,
    right: 30,
    backgroundColor: '#fff',
    borderRadius: 14,
  },
  uploadOptions: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
    marginBottom: 24,
  },
  uploadButton: {
    flex: 1,
    maxWidth: 160,
  },
  uploadButtonGradient: {
    paddingVertical: 24,
    borderRadius: 16,
    alignItems: 'center',
    gap: 8,
  },
  uploadButtonOutline: {
    paddingVertical: 24,
    borderRadius: 16,
    alignItems: 'center',
    gap: 8,
    borderWidth: 2,
    borderColor: '#6366f1',
    backgroundColor: 'rgba(99, 102, 241, 0.1)',
  },
  uploadButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  uploadButtonTextOutline: {
    color: '#6366f1',
    fontSize: 14,
    fontWeight: '600',
  },
  startButton: {
    marginBottom: 24,
  },
  startButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    borderRadius: 12,
    gap: 8,
  },
  startButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  disclaimerBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: 'rgba(107, 114, 128, 0.2)',
    borderRadius: 12,
    padding: 16,
    gap: 12,
  },
  disclaimerText: {
    flex: 1,
    color: '#9ca3af',
    fontSize: 13,
    lineHeight: 20,
  },

  // Analyzing Stage
  analyzingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  analyzingImage: {
    width: SCREEN_WIDTH - 80,
    height: SCREEN_WIDTH - 80,
    borderRadius: 16,
    opacity: 0.7,
  },
  analyzingOverlay: {
    position: 'absolute',
    alignItems: 'center',
  },
  analyzingText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '600',
    marginTop: 20,
  },
  analyzingSubtext: {
    color: '#9ca3af',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
    paddingHorizontal: 40,
  },

  // Conversation Stage
  conversationContainer: {
    flex: 1,
  },
  imageSection: {
    height: 160,
    marginHorizontal: 16,
    marginBottom: 8,
    borderRadius: 12,
    overflow: 'hidden',
    position: 'relative',
  },
  conversationImage: {
    width: '100%',
    height: '100%',
  },
  featuresOverlay: {
    position: 'absolute',
    bottom: 8,
    left: 8,
    right: 8,
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  featureTag: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
  },
  featureDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 6,
  },
  featureLabel: {
    fontSize: 11,
    fontWeight: '600',
  },
  toggleFeaturesButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 16,
    gap: 4,
  },
  toggleFeaturesText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
  },
  urgencyBadge: {
    position: 'absolute',
    top: 8,
    left: 8,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  urgencyText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
  },
  progressContainer: {
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  progressBar: {
    height: 4,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 2,
    marginBottom: 4,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#6366f1',
    borderRadius: 2,
  },
  progressText: {
    color: '#9ca3af',
    fontSize: 12,
    textAlign: 'center',
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    padding: 16,
    paddingBottom: 24,
  },
  messageBubble: {
    flexDirection: 'row',
    marginBottom: 16,
    maxWidth: '90%',
  },
  userMessage: {
    alignSelf: 'flex-end',
    flexDirection: 'row-reverse',
  },
  aiMessage: {
    alignSelf: 'flex-start',
  },
  systemMessage: {
    alignSelf: 'center',
  },
  aiAvatarSmall: {
    marginRight: 8,
  },
  aiAvatarSmallGradient: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  messageContent: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 16,
    padding: 12,
    maxWidth: '85%',
  },
  userMessageContent: {
    backgroundColor: '#6366f1',
  },
  messageText: {
    color: '#fff',
    fontSize: 15,
    lineHeight: 22,
  },
  userMessageText: {
    color: '#fff',
  },
  typingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
  },
  typingDots: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 16,
    padding: 12,
    gap: 4,
  },
  typingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#9ca3af',
  },
  typingDotMiddle: {
    opacity: 0.7,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    padding: 16,
    paddingBottom: Platform.OS === 'ios' ? 32 : 16,
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  textInput: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 24,
    paddingHorizontal: 16,
    paddingVertical: 12,
    color: '#fff',
    fontSize: 16,
    maxHeight: 100,
    marginRight: 12,
  },
  sendButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#6366f1',
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  assessmentActions: {
    padding: 16,
    paddingBottom: Platform.OS === 'ios' ? 32 : 16,
    gap: 12,
  },
  summaryButton: {
    overflow: 'hidden',
    borderRadius: 12,
  },
  summaryButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    gap: 8,
  },
  summaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  newConsultButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    gap: 6,
  },
  newConsultButtonText: {
    color: '#6366f1',
    fontSize: 15,
    fontWeight: '500',
  },

  // Unavailable
  unavailableContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
  },
  unavailableTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '600',
    marginTop: 20,
    marginBottom: 12,
  },
  unavailableText: {
    color: '#9ca3af',
    fontSize: 15,
    textAlign: 'center',
    lineHeight: 24,
  },
  goBackButton: {
    marginTop: 24,
    paddingHorizontal: 24,
    paddingVertical: 12,
    backgroundColor: '#6366f1',
    borderRadius: 8,
  },
  goBackButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
