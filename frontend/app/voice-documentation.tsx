/**
 * Voice-Controlled Clinical Documentation Screen
 *
 * Provides hands-free clinical documentation with:
 * - Voice dictation for symptom capture
 * - Hands-free image capture via voice commands
 * - Auto-generate SOAP notes from voice
 * - Extract structured data (duration, severity, location)
 * - Real-time transcription display
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Modal,
  Alert,
  ActivityIndicator,
  Platform,
  Animated,
  TextInput,
  Dimensions,
  Image,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import { Camera, CameraView } from 'expo-camera';
import * as Speech from 'expo-speech';
import * as ImagePicker from 'expo-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// Types
interface ExtractedData {
  type: string;
  value: string;
  normalized: string;
  unit: string | null;
  confidence: number;
}

interface SOAPNote {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
  structured_data: {
    chief_complaint: string | null;
    duration: string | null;
    severity: string | null;
    location: string | null;
    associated_symptoms: string[];
  };
  confidence_score: number;
}

interface VoiceCommand {
  type: string;
  action: string;
  confidence: number;
}

type DocumentationMode = 'idle' | 'dictating' | 'camera' | 'reviewing';

export default function VoiceDocumentationScreen() {
  const { t } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // State
  const [mode, setMode] = useState<DocumentationMode>('idle');
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [extractedData, setExtractedData] = useState<ExtractedData[]>([]);
  const [soapNote, setSoapNote] = useState<SOAPNote | null>(null);
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [showSOAPModal, setShowSOAPModal] = useState(false);
  const [voiceCommands, setVoiceCommands] = useState<any>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);
  const [isWebSpeechSupported, setIsWebSpeechSupported] = useState(false);
  const [manualInput, setManualInput] = useState('');
  const [showManualInput, setShowManualInput] = useState(false);

  // Refs
  const cameraRef = useRef<CameraView>(null);
  const recognitionRef = useRef<any>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const scrollViewRef = useRef<ScrollView>(null);

  // Auth headers
  const getAuthHeaders = async () => {
    const token = await AsyncStorage.getItem('accessToken');
    return {
      'Authorization': `Bearer ${token}`,
    };
  };

  // Initialize
  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
      return;
    }

    initializeSpeechRecognition();
    loadVoiceCommands();
    requestCameraPermission();

    return () => {
      stopListening();
    };
  }, [isAuthenticated]);

  // Pulse animation
  useEffect(() => {
    if (isListening) {
      const pulse = Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.3,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 600,
          useNativeDriver: true,
        }),
      ]);
      Animated.loop(pulse).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [isListening]);

  const initializeSpeechRecognition = () => {
    if (Platform.OS === 'web') {
      const SpeechRecognition =
        (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

      if (SpeechRecognition) {
        setIsWebSpeechSupported(true);
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = true;
        recognitionRef.current.interimResults = true;
        recognitionRef.current.lang = 'en-US';

        recognitionRef.current.onresult = handleSpeechResult;
        recognitionRef.current.onerror = handleSpeechError;
        recognitionRef.current.onend = handleSpeechEnd;
      }
    } else {
      // For native platforms, we'll use manual input or expo-av for audio recording
      // Speech recognition on native would require @react-native-voice/voice
      // For now, provide manual text input option on native
      setIsWebSpeechSupported(false);
    }
  };

  const requestCameraPermission = async () => {
    const { status } = await Camera.requestCameraPermissionsAsync();
    setHasCameraPermission(status === 'granted');
  };

  const loadVoiceCommands = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/voice/commands`);
      if (response.ok) {
        const data = await response.json();
        setVoiceCommands(data);
      }
    } catch (error) {
      console.error('Failed to load voice commands:', error);
    }
  };

  // Speech recognition handlers (web)
  const handleSpeechResult = (event: any) => {
    let interim = '';
    let final = '';

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      if (result.isFinal) {
        final += result[0].transcript;
      } else {
        interim += result[0].transcript;
      }
    }

    if (final) {
      const newTranscript = transcript + ' ' + final;
      setTranscript(newTranscript.trim());
      setInterimTranscript('');

      // Check for voice commands
      checkForVoiceCommand(final.trim());

      // Process the text
      processTranscriptSegment(final.trim());
    } else {
      setInterimTranscript(interim);
    }
  };

  const handleSpeechError = (event: any) => {
    console.error('Speech recognition error:', event.error);
    if (event.error === 'no-speech') {
      // Restart if still supposed to be listening
      if (isListening) {
        try {
          recognitionRef.current?.start();
        } catch (e) {
          // Already started
        }
      }
    } else {
      speakFeedback('Speech recognition error. Please try again.');
    }
  };

  const handleSpeechEnd = () => {
    if (isListening) {
      try {
        recognitionRef.current?.start();
      } catch (e) {
        // Already started
      }
    }
  };

  // Voice command detection
  const checkForVoiceCommand = async (text: string) => {
    const lowerText = text.toLowerCase().trim();

    // Camera commands
    if (lowerText.includes('take photo') || lowerText.includes('capture') ||
        lowerText.includes('take picture') || lowerText.includes('snap')) {
      await handleTakePhoto();
      return true;
    }

    // SOAP note commands
    if (lowerText.includes('generate soap') || lowerText.includes('create soap') ||
        lowerText.includes('soap note')) {
      await handleGenerateSOAP();
      return true;
    }

    // Stop commands
    if (lowerText.includes('stop dictation') || lowerText.includes('done dictating') ||
        lowerText.includes('end dictation')) {
      stopListening();
      return true;
    }

    // Help command
    if (lowerText.includes('help') || lowerText.includes('commands')) {
      setShowHelpModal(true);
      return true;
    }

    return false;
  };

  // Start/stop listening
  const startListening = async () => {
    if (!isWebSpeechSupported && Platform.OS !== 'web') {
      setShowManualInput(true);
      return;
    }

    setIsListening(true);
    setMode('dictating');

    try {
      // Start dictation session
      const headers = await getAuthHeaders();
      const formData = new FormData();

      const response = await fetch(`${API_BASE_URL}/voice/dictation/start`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setSessionId(data.session_id);
      }

      // Start speech recognition
      if (Platform.OS === 'web' && recognitionRef.current) {
        recognitionRef.current.start();
      }

      speakFeedback('Listening. Speak now.');
    } catch (error) {
      console.error('Failed to start listening:', error);
      setIsListening(false);
      setMode('idle');
    }
  };

  const stopListening = async () => {
    setIsListening(false);

    if (Platform.OS === 'web' && recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        // Already stopped
      }
    }

    speakFeedback('Dictation stopped.');
    setMode('reviewing');
  };

  // Process transcript segment
  const processTranscriptSegment = async (text: string) => {
    try {
      const headers = await getAuthHeaders();
      const formData = new FormData();
      formData.append('text', text);

      const response = await fetch(`${API_BASE_URL}/voice/process`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();

        // Update extracted data
        if (data.extracted_data && data.extracted_data.length > 0) {
          setExtractedData(prev => [...prev, ...data.extracted_data]);
        }

        // Handle voice command
        if (data.command) {
          handleVoiceCommandAction(data.command);
        }
      }
    } catch (error) {
      console.error('Failed to process transcript:', error);
    }
  };

  // Handle voice command action
  const handleVoiceCommandAction = (command: VoiceCommand) => {
    speakFeedback(`Executing ${command.action.replace(/_/g, ' ')}`);

    switch (command.action) {
      case 'take_photo':
        handleTakePhoto();
        break;
      case 'generate_soap':
        handleGenerateSOAP();
        break;
      case 'stop_dictation':
        stopListening();
        break;
      case 'go_back':
        router.back();
        break;
      case 'help':
        setShowHelpModal(true);
        break;
      default:
        console.log('Unknown command:', command.action);
    }
  };

  // Take photo (hands-free)
  const handleTakePhoto = async () => {
    if (mode !== 'camera') {
      setMode('camera');
      speakFeedback('Camera ready. Say take photo to capture.');
      return;
    }

    if (!cameraRef.current || !hasCameraPermission) {
      speakFeedback('Camera not available.');
      return;
    }

    try {
      speakFeedback('Taking photo.');
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
      });

      if (photo?.uri) {
        setCapturedImages(prev => [...prev, photo.uri]);
        speakFeedback('Photo captured successfully.');
      }
    } catch (error) {
      console.error('Failed to take photo:', error);
      speakFeedback('Failed to capture photo.');
    }
  };

  // Generate SOAP note
  const handleGenerateSOAP = async () => {
    if (!transcript.trim()) {
      speakFeedback('No dictation to generate SOAP note from.');
      return;
    }

    setIsProcessing(true);
    speakFeedback('Generating SOAP note.');

    try {
      const headers = await getAuthHeaders();
      const formData = new FormData();
      formData.append('dictation', transcript);

      const response = await fetch(`${API_BASE_URL}/voice/generate-soap`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setSoapNote(data);
        setShowSOAPModal(true);
        speakFeedback('SOAP note generated successfully.');
      } else {
        throw new Error('SOAP generation failed');
      }
    } catch (error) {
      console.error('Failed to generate SOAP:', error);
      speakFeedback('Failed to generate SOAP note.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Capture symptoms
  const handleCaptureSymptoms = async () => {
    if (!transcript.trim()) {
      Alert.alert('Error', 'No dictation to analyze');
      return;
    }

    setIsProcessing(true);

    try {
      const headers = await getAuthHeaders();
      const formData = new FormData();
      formData.append('text', transcript);

      const response = await fetch(`${API_BASE_URL}/voice/symptom-capture`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(
          'Symptoms Captured',
          `Duration: ${data.symptoms.duration || 'Not specified'}\n` +
          `Severity: ${data.symptoms.severity || 'Not specified'}\n` +
          `Location: ${data.symptoms.location || 'Not specified'}\n` +
          `Itching: ${data.symptoms.itching ? 'Yes' : 'No'}\n` +
          `Pain: ${data.symptoms.pain ? 'Yes' : 'No'}\n` +
          `Changes: ${data.symptoms.changes.join(', ') || 'None noted'}`
        );
      }
    } catch (error) {
      console.error('Failed to capture symptoms:', error);
      Alert.alert('Error', 'Failed to analyze symptoms');
    } finally {
      setIsProcessing(false);
    }
  };

  // Manual input submit
  const handleManualSubmit = () => {
    if (manualInput.trim()) {
      const newTranscript = transcript + ' ' + manualInput;
      setTranscript(newTranscript.trim());
      processTranscriptSegment(manualInput.trim());
      setManualInput('');
    }
    setShowManualInput(false);
  };

  // Clear all
  const handleClear = () => {
    Alert.alert(
      'Clear Documentation',
      'Are you sure you want to clear all dictation and captured images?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: () => {
            setTranscript('');
            setInterimTranscript('');
            setExtractedData([]);
            setSoapNote(null);
            setCapturedImages([]);
            setMode('idle');
          }
        }
      ]
    );
  };

  // Text-to-speech feedback
  const speakFeedback = (text: string) => {
    Speech.speak(text, {
      language: 'en-US',
      pitch: 1.0,
      rate: 1.0,
    });
  };

  // Get severity color
  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'mild': return '#10b981';
      case 'moderate': return '#f59e0b';
      case 'severe': return '#ef4444';
      default: return '#6b7280';
    }
  };

  // Get data type color
  const getDataTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      duration: '#2196F3',
      severity: '#FF9800',
      location: '#4CAF50',
      symptom: '#9C27B0',
      change: '#F44336',
    };
    return colors[type] || '#757575';
  };

  // Render camera view
  const renderCameraView = () => (
    <View style={styles.cameraContainer}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
      >
        <View style={styles.cameraOverlay}>
          <TouchableOpacity
            style={styles.cameraCloseButton}
            onPress={() => setMode('dictating')}
          >
            <Ionicons name="close" size={30} color="#fff" />
          </TouchableOpacity>

          <View style={styles.cameraInstructions}>
            <Text style={styles.cameraInstructionText}>
              Say "Take photo" or tap the button
            </Text>
          </View>

          <TouchableOpacity
            style={styles.captureButton}
            onPress={handleTakePhoto}
          >
            <View style={styles.captureButtonInner} />
          </TouchableOpacity>
        </View>
      </CameraView>
    </View>
  );

  // Render SOAP modal
  const renderSOAPModal = () => (
    <Modal
      visible={showSOAPModal}
      animationType="slide"
      transparent={true}
      onRequestClose={() => setShowSOAPModal(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.soapModalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>SOAP Note</Text>
            <TouchableOpacity onPress={() => setShowSOAPModal(false)}>
              <Ionicons name="close" size={24} color="#333" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.soapScrollView}>
            {soapNote && (
              <>
                <View style={styles.soapSection}>
                  <Text style={styles.soapSectionTitle}>Subjective</Text>
                  <Text style={styles.soapSectionContent}>{soapNote.subjective}</Text>
                </View>

                <View style={styles.soapSection}>
                  <Text style={styles.soapSectionTitle}>Objective</Text>
                  <Text style={styles.soapSectionContent}>{soapNote.objective}</Text>
                </View>

                <View style={styles.soapSection}>
                  <Text style={styles.soapSectionTitle}>Assessment</Text>
                  <Text style={styles.soapSectionContent}>{soapNote.assessment}</Text>
                </View>

                <View style={styles.soapSection}>
                  <Text style={styles.soapSectionTitle}>Plan</Text>
                  <Text style={styles.soapSectionContent}>{soapNote.plan}</Text>
                </View>

                {/* Structured Data */}
                <View style={styles.structuredDataSection}>
                  <Text style={styles.soapSectionTitle}>Extracted Data</Text>
                  {soapNote.structured_data.chief_complaint && (
                    <Text style={styles.structuredItem}>
                      Chief Complaint: {soapNote.structured_data.chief_complaint}
                    </Text>
                  )}
                  {soapNote.structured_data.duration && (
                    <Text style={styles.structuredItem}>
                      Duration: {soapNote.structured_data.duration}
                    </Text>
                  )}
                  {soapNote.structured_data.severity && (
                    <Text style={styles.structuredItem}>
                      Severity: {soapNote.structured_data.severity}
                    </Text>
                  )}
                  {soapNote.structured_data.location && (
                    <Text style={styles.structuredItem}>
                      Location: {soapNote.structured_data.location}
                    </Text>
                  )}
                  {soapNote.structured_data.associated_symptoms.length > 0 && (
                    <Text style={styles.structuredItem}>
                      Symptoms: {soapNote.structured_data.associated_symptoms.join(', ')}
                    </Text>
                  )}
                </View>

                <View style={styles.confidenceSection}>
                  <Text style={styles.confidenceText}>
                    Confidence Score: {(soapNote.confidence_score * 100).toFixed(0)}%
                  </Text>
                </View>
              </>
            )}
          </ScrollView>

          <TouchableOpacity
            style={styles.copyButton}
            onPress={() => {
              // Copy to clipboard would go here
              Alert.alert('Success', 'SOAP note ready for use');
              setShowSOAPModal(false);
            }}
          >
            <Text style={styles.copyButtonText}>Use This Note</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  // Render help modal
  const renderHelpModal = () => (
    <Modal
      visible={showHelpModal}
      animationType="slide"
      transparent={true}
      onRequestClose={() => setShowHelpModal(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.helpModalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Voice Commands</Text>
            <TouchableOpacity onPress={() => setShowHelpModal(false)}>
              <Ionicons name="close" size={24} color="#333" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.helpScrollView}>
            <View style={styles.helpCategory}>
              <Text style={styles.helpCategoryTitle}>CAPTURE COMMANDS</Text>
              <Text style={styles.helpItem}>"Take photo" - Capture an image</Text>
              <Text style={styles.helpItem}>"Capture" - Take a picture</Text>
              <Text style={styles.helpItem}>"Snap" - Quick photo capture</Text>
            </View>

            <View style={styles.helpCategory}>
              <Text style={styles.helpCategoryTitle}>DICTATION COMMANDS</Text>
              <Text style={styles.helpItem}>"Stop dictation" - End recording</Text>
              <Text style={styles.helpItem}>"Done dictating" - Finish dictation</Text>
              <Text style={styles.helpItem}>"Generate SOAP" - Create SOAP note</Text>
            </View>

            <View style={styles.helpCategory}>
              <Text style={styles.helpCategoryTitle}>EXAMPLE DICTATION</Text>
              <Text style={styles.helpExample}>
                "Patient reports itchy rash on the forearm for about 3 weeks.
                The rash is moderately severe and has been getting worse.
                No bleeding but some scaling noted."
              </Text>
            </View>

            <View style={styles.helpCategory}>
              <Text style={styles.helpCategoryTitle}>TIPS</Text>
              <Text style={styles.helpItem}>- Speak clearly and at a normal pace</Text>
              <Text style={styles.helpItem}>- Include duration (e.g., "3 weeks")</Text>
              <Text style={styles.helpItem}>- Mention body location</Text>
              <Text style={styles.helpItem}>- Describe severity (mild, moderate, severe)</Text>
            </View>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );

  // Render manual input modal
  const renderManualInputModal = () => (
    <Modal
      visible={showManualInput}
      animationType="slide"
      transparent={true}
      onRequestClose={() => setShowManualInput(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.manualInputContent}>
          <Text style={styles.modalTitle}>Enter Clinical Notes</Text>
          <Text style={styles.manualInputHint}>
            Voice input is not available on this device. Please type your notes.
          </Text>

          <TextInput
            style={styles.manualTextInput}
            multiline
            placeholder="Enter patient symptoms, observations, etc..."
            placeholderTextColor="#9ca3af"
            value={manualInput}
            onChangeText={setManualInput}
            autoFocus
          />

          <View style={styles.manualInputButtons}>
            <TouchableOpacity
              style={styles.manualCancelButton}
              onPress={() => setShowManualInput(false)}
            >
              <Text style={styles.manualCancelText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.manualSubmitButton}
              onPress={handleManualSubmit}
            >
              <Text style={styles.manualSubmitText}>Add Notes</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );

  // Main render
  if (mode === 'camera') {
    return renderCameraView();
  }

  return (
    <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#2563eb" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Voice Documentation</Text>
        <TouchableOpacity onPress={() => setShowHelpModal(true)}>
          <Ionicons name="help-circle-outline" size={28} color="#2563eb" />
        </TouchableOpacity>
      </View>

      <ScrollView
        ref={scrollViewRef}
        style={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* Voice Input Section */}
        <View style={styles.voiceSection}>
          <Animated.View style={[styles.micButtonContainer, { transform: [{ scale: pulseAnim }] }]}>
            <TouchableOpacity
              style={[
                styles.micButton,
                isListening && styles.micButtonActive,
                isProcessing && styles.micButtonProcessing,
              ]}
              onPress={isListening ? stopListening : startListening}
              disabled={isProcessing}
            >
              {isProcessing ? (
                <ActivityIndicator color="#fff" size="large" />
              ) : (
                <Ionicons
                  name={isListening ? 'stop' : 'mic'}
                  size={40}
                  color="#fff"
                />
              )}
            </TouchableOpacity>
          </Animated.View>

          <Text style={styles.statusText}>
            {isProcessing
              ? 'Processing...'
              : isListening
              ? 'Listening... Speak now'
              : 'Tap to start voice dictation'}
          </Text>

          {!isWebSpeechSupported && Platform.OS !== 'web' && (
            <TouchableOpacity
              style={styles.manualInputButton}
              onPress={() => setShowManualInput(true)}
            >
              <Ionicons name="create-outline" size={20} color="#2563eb" />
              <Text style={styles.manualInputButtonText}>Type instead</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Transcript Display */}
        {(transcript || interimTranscript) && (
          <View style={styles.transcriptSection}>
            <View style={styles.transcriptHeader}>
              <Text style={styles.transcriptLabel}>Transcript</Text>
              <TouchableOpacity onPress={handleClear}>
                <Ionicons name="trash-outline" size={20} color="#ef4444" />
              </TouchableOpacity>
            </View>
            <Text style={styles.transcriptText}>
              {transcript}
              {interimTranscript && (
                <Text style={styles.interimText}> {interimTranscript}</Text>
              )}
            </Text>
          </View>
        )}

        {/* Extracted Data */}
        {extractedData.length > 0 && (
          <View style={styles.extractedSection}>
            <Text style={styles.sectionTitle}>Extracted Clinical Data</Text>
            <View style={styles.extractedGrid}>
              {extractedData.map((item, index) => (
                <View
                  key={index}
                  style={[
                    styles.extractedItem,
                    { borderLeftColor: getDataTypeColor(item.type) }
                  ]}
                >
                  <Text style={styles.extractedType}>{item.type.toUpperCase()}</Text>
                  <Text style={styles.extractedValue}>{item.normalized || item.value}</Text>
                </View>
              ))}
            </View>
          </View>
        )}

        {/* Captured Images */}
        {capturedImages.length > 0 && (
          <View style={styles.imagesSection}>
            <Text style={styles.sectionTitle}>Captured Images</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {capturedImages.map((uri, index) => (
                <View key={index} style={styles.imageContainer}>
                  <Image source={{ uri }} style={styles.capturedImage} />
                  <TouchableOpacity
                    style={styles.removeImageButton}
                    onPress={() => {
                      setCapturedImages(prev => prev.filter((_, i) => i !== index));
                    }}
                  >
                    <Ionicons name="close-circle" size={24} color="#ef4444" />
                  </TouchableOpacity>
                </View>
              ))}
            </ScrollView>
          </View>
        )}

        {/* Action Buttons */}
        <View style={styles.actionsSection}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => setMode('camera')}
          >
            <Ionicons name="camera" size={24} color="#fff" />
            <Text style={styles.actionButtonText}>Capture Image</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, styles.soapButton]}
            onPress={handleGenerateSOAP}
            disabled={!transcript.trim() || isProcessing}
          >
            <Ionicons name="document-text" size={24} color="#fff" />
            <Text style={styles.actionButtonText}>Generate SOAP</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, styles.symptomsButton]}
            onPress={handleCaptureSymptoms}
            disabled={!transcript.trim() || isProcessing}
          >
            <Ionicons name="medical" size={24} color="#fff" />
            <Text style={styles.actionButtonText}>Analyze Symptoms</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.bottomSpacer} />
      </ScrollView>

      {renderHelpModal()}
      {renderSOAPModal()}
      {renderManualInputModal()}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 20,
    paddingBottom: 15,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  content: {
    flex: 1,
    padding: 20,
  },
  voiceSection: {
    alignItems: 'center',
    paddingVertical: 30,
  },
  micButtonContainer: {
    marginBottom: 16,
  },
  micButton: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: '#2563eb',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  micButtonActive: {
    backgroundColor: '#dc2626',
  },
  micButtonProcessing: {
    backgroundColor: '#f59e0b',
  },
  statusText: {
    fontSize: 16,
    color: '#6b7280',
    textAlign: 'center',
  },
  manualInputButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    padding: 10,
  },
  manualInputButtonText: {
    color: '#2563eb',
    marginLeft: 6,
    fontSize: 14,
  },
  transcriptSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  transcriptHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  transcriptLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
    textTransform: 'uppercase',
  },
  transcriptText: {
    fontSize: 16,
    color: '#1f2937',
    lineHeight: 24,
  },
  interimText: {
    color: '#9ca3af',
    fontStyle: 'italic',
  },
  extractedSection: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
    textTransform: 'uppercase',
    marginBottom: 12,
  },
  extractedGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  extractedItem: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    borderLeftWidth: 4,
    minWidth: '45%',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  extractedType: {
    fontSize: 10,
    fontWeight: '700',
    color: '#9ca3af',
    marginBottom: 4,
  },
  extractedValue: {
    fontSize: 14,
    color: '#1f2937',
    fontWeight: '500',
  },
  imagesSection: {
    marginBottom: 16,
  },
  imageContainer: {
    position: 'relative',
    marginRight: 12,
  },
  capturedImage: {
    width: 120,
    height: 120,
    borderRadius: 12,
  },
  removeImageButton: {
    position: 'absolute',
    top: -8,
    right: -8,
    backgroundColor: '#fff',
    borderRadius: 12,
  },
  actionsSection: {
    gap: 12,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    gap: 8,
  },
  soapButton: {
    backgroundColor: '#10b981',
  },
  symptomsButton: {
    backgroundColor: '#8b5cf6',
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  bottomSpacer: {
    height: 40,
  },
  // Camera styles
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    flex: 1,
    backgroundColor: 'transparent',
    justifyContent: 'space-between',
    padding: 20,
  },
  cameraCloseButton: {
    alignSelf: 'flex-start',
    padding: 12,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 25,
    marginTop: Platform.OS === 'ios' ? 40 : 20,
  },
  cameraInstructions: {
    alignItems: 'center',
  },
  cameraInstructionText: {
    color: '#fff',
    fontSize: 16,
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 12,
    borderRadius: 8,
  },
  captureButton: {
    alignSelf: 'center',
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 30,
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#fff',
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1f2937',
  },
  // SOAP Modal
  soapModalContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    width: '100%',
    maxHeight: '85%',
  },
  soapScrollView: {
    maxHeight: screenHeight * 0.5,
  },
  soapSection: {
    marginBottom: 16,
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 12,
  },
  soapSectionTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#2563eb',
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  soapSectionContent: {
    fontSize: 14,
    color: '#374151',
    lineHeight: 22,
  },
  structuredDataSection: {
    backgroundColor: '#f0f9ff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
  },
  structuredItem: {
    fontSize: 14,
    color: '#374151',
    marginBottom: 4,
  },
  confidenceSection: {
    alignItems: 'center',
    paddingVertical: 8,
  },
  confidenceText: {
    fontSize: 14,
    color: '#6b7280',
  },
  copyButton: {
    backgroundColor: '#2563eb',
    borderRadius: 10,
    padding: 14,
    alignItems: 'center',
    marginTop: 12,
  },
  copyButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  // Help Modal
  helpModalContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    width: '100%',
    maxHeight: '80%',
  },
  helpScrollView: {
    maxHeight: screenHeight * 0.55,
  },
  helpCategory: {
    marginBottom: 20,
  },
  helpCategoryTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: '#6b7280',
    marginBottom: 8,
    letterSpacing: 1,
  },
  helpItem: {
    fontSize: 14,
    color: '#374151',
    marginBottom: 6,
    paddingLeft: 8,
  },
  helpExample: {
    fontSize: 14,
    color: '#374151',
    fontStyle: 'italic',
    backgroundColor: '#f3f4f6',
    padding: 12,
    borderRadius: 8,
    lineHeight: 22,
  },
  // Manual Input Modal
  manualInputContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    width: '100%',
  },
  manualInputHint: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 16,
  },
  manualTextInput: {
    backgroundColor: '#f3f4f6',
    borderRadius: 10,
    padding: 16,
    fontSize: 16,
    minHeight: 150,
    textAlignVertical: 'top',
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  manualInputButtons: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 12,
  },
  manualCancelButton: {
    padding: 12,
  },
  manualCancelText: {
    color: '#6b7280',
    fontSize: 16,
  },
  manualSubmitButton: {
    backgroundColor: '#2563eb',
    borderRadius: 8,
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  manualSubmitText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
