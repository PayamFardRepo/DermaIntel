/**
 * VoiceInput Component
 *
 * Provides voice input capabilities for hands-free operation:
 * - Speech-to-text for clinical documentation
 * - Voice commands for app control
 * - Real-time transcription display
 * - Visual feedback during recording
 *
 * Uses device speech recognition (Web Speech API on web, native on mobile)
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  Modal,
  ScrollView,
  Platform,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Speech from 'expo-speech';
import VoiceDocumentationService, {
  ProcessedVoiceResult,
  ExtractedClinicalData,
  VoiceCommand,
} from '../services/VoiceDocumentationService';

// Props interface
interface VoiceInputProps {
  // Mode of operation
  mode?: 'command' | 'dictation' | 'auto';

  // Callbacks
  onTranscript?: (text: string) => void;
  onCommand?: (command: VoiceCommand) => void;
  onClinicalDataExtracted?: (data: ExtractedClinicalData[]) => void;
  onError?: (error: string) => void;

  // Optional analysis ID for linking dictation
  analysisId?: number;

  // UI customization
  showTranscript?: boolean;
  showExtractedData?: boolean;
  compact?: boolean;
  buttonColor?: string;

  // Auto-submit on silence
  autoSubmitDelay?: number; // ms, 0 to disable

  // Language
  language?: string;
}

// Recording state
type RecordingState = 'idle' | 'listening' | 'processing' | 'error';

const VoiceInput: React.FC<VoiceInputProps> = ({
  mode = 'auto',
  onTranscript,
  onCommand,
  onClinicalDataExtracted,
  onError,
  analysisId,
  showTranscript = true,
  showExtractedData = true,
  compact = false,
  buttonColor = '#4CAF50',
  autoSubmitDelay = 2000,
  language = 'en-US',
}) => {
  // State
  const [recordingState, setRecordingState] = useState<RecordingState>('idle');
  const [transcript, setTranscript] = useState<string>('');
  const [interimTranscript, setInterimTranscript] = useState<string>('');
  const [extractedData, setExtractedData] = useState<ExtractedClinicalData[]>([]);
  const [lastCommand, setLastCommand] = useState<VoiceCommand | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const [voiceCommands, setVoiceCommands] = useState<any>(null);
  const [isSupported, setIsSupported] = useState(true);

  // Animation
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const silenceTimer = useRef<NodeJS.Timeout | null>(null);

  // Web Speech API reference
  const recognitionRef = useRef<any>(null);

  // Check for speech recognition support
  useEffect(() => {
    if (Platform.OS === 'web') {
      const SpeechRecognition =
        (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      setIsSupported(!!SpeechRecognition);

      if (SpeechRecognition) {
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = true;
        recognitionRef.current.interimResults = true;
        recognitionRef.current.lang = language;

        recognitionRef.current.onresult = handleSpeechResult;
        recognitionRef.current.onerror = handleSpeechError;
        recognitionRef.current.onend = handleSpeechEnd;
      }
    } else {
      // For React Native, we'd need a native module like react-native-voice
      // For now, show a message about web-only support
      setIsSupported(Platform.OS === 'web');
    }

    // Load voice commands reference
    loadVoiceCommands();

    return () => {
      stopRecording();
    };
  }, [language]);

  // Pulse animation while recording
  useEffect(() => {
    if (recordingState === 'listening') {
      const pulse = Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.2,
          duration: 500,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 500,
          useNativeDriver: true,
        }),
      ]);

      Animated.loop(pulse).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [recordingState]);

  // Load voice commands
  const loadVoiceCommands = async () => {
    try {
      const commands = await VoiceDocumentationService.getVoiceCommands();
      setVoiceCommands(commands);
    } catch (error) {
      console.error('Failed to load voice commands:', error);
    }
  };

  // Handle speech recognition results (Web)
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
      setTranscript((prev) => prev + ' ' + final);
      setInterimTranscript('');
      resetSilenceTimer();

      // Process the final transcript
      processTranscript(final.trim());
    } else {
      setInterimTranscript(interim);
    }
  };

  // Handle speech recognition error
  const handleSpeechError = (event: any) => {
    console.error('Speech recognition error:', event.error);
    setRecordingState('error');

    const errorMessage = getErrorMessage(event.error);
    onError?.(errorMessage);

    // Auto-restart on some errors
    if (event.error === 'no-speech' && recordingState === 'listening') {
      startRecording();
    }
  };

  // Handle speech recognition end
  const handleSpeechEnd = () => {
    if (recordingState === 'listening') {
      // Restart if still supposed to be listening
      recognitionRef.current?.start();
    }
  };

  // Get user-friendly error message
  const getErrorMessage = (error: string): string => {
    switch (error) {
      case 'not-allowed':
        return 'Microphone access denied. Please enable microphone permissions.';
      case 'no-speech':
        return 'No speech detected. Please try again.';
      case 'network':
        return 'Network error. Please check your connection.';
      case 'audio-capture':
        return 'No microphone found. Please connect a microphone.';
      default:
        return `Speech recognition error: ${error}`;
    }
  };

  // Reset silence timer
  const resetSilenceTimer = () => {
    if (silenceTimer.current) {
      clearTimeout(silenceTimer.current);
    }

    if (autoSubmitDelay > 0) {
      silenceTimer.current = setTimeout(() => {
        if (transcript.trim()) {
          submitTranscript();
        }
      }, autoSubmitDelay);
    }
  };

  // Process transcript through backend
  const processTranscript = async (text: string) => {
    try {
      const result = await VoiceDocumentationService.processVoiceInput(text, mode, language);

      // Handle command
      if (result.command && result.mode === 'command') {
        setLastCommand(result.command);
        onCommand?.(result.command);

        // Provide audio feedback for commands
        speakFeedback(`Executing ${result.command.action.replace(/_/g, ' ')}`);
      }

      // Handle extracted clinical data
      if (result.extracted_data && result.extracted_data.length > 0) {
        setExtractedData((prev) => [...prev, ...result.extracted_data!]);
        onClinicalDataExtracted?.(result.extracted_data);
      }

      onTranscript?.(text);
    } catch (error) {
      console.error('Error processing transcript:', error);
    }
  };

  // Submit final transcript
  const submitTranscript = async () => {
    if (!transcript.trim()) return;

    setRecordingState('processing');
    stopRecording();

    try {
      // Final processing of complete transcript
      await processTranscript(transcript.trim());
    } finally {
      setRecordingState('idle');
    }
  };

  // Speak feedback using TTS
  const speakFeedback = (text: string) => {
    Speech.speak(text, {
      language,
      pitch: 1.0,
      rate: 1.0,
    });
  };

  // Start recording
  const startRecording = useCallback(async () => {
    if (!isSupported) {
      Alert.alert(
        'Not Supported',
        'Voice input is currently only supported on web browsers with Speech Recognition API.'
      );
      return;
    }

    setRecordingState('listening');
    setTranscript('');
    setInterimTranscript('');
    setExtractedData([]);
    setLastCommand(null);

    try {
      if (Platform.OS === 'web' && recognitionRef.current) {
        recognitionRef.current.start();
      }
      // For native, we would start the native speech recognition here
    } catch (error) {
      console.error('Error starting recording:', error);
      setRecordingState('error');
      onError?.('Failed to start voice recording');
    }
  }, [isSupported]);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (silenceTimer.current) {
      clearTimeout(silenceTimer.current);
      silenceTimer.current = null;
    }

    if (Platform.OS === 'web' && recognitionRef.current) {
      recognitionRef.current.stop();
    }

    setRecordingState('idle');
  }, []);

  // Toggle recording
  const toggleRecording = () => {
    if (recordingState === 'listening') {
      submitTranscript();
    } else if (recordingState === 'idle') {
      startRecording();
    }
  };

  // Clear transcript
  const clearTranscript = () => {
    setTranscript('');
    setInterimTranscript('');
    setExtractedData([]);
    setLastCommand(null);
  };

  // Render extracted data item
  const renderExtractedItem = (item: ExtractedClinicalData, index: number) => {
    const typeColors: Record<string, string> = {
      duration: '#2196F3',
      severity: '#FF9800',
      location: '#4CAF50',
      symptom: '#9C27B0',
      change: '#F44336',
    };

    return (
      <View
        key={index}
        style={[styles.extractedItem, { borderLeftColor: typeColors[item.type] || '#757575' }]}
      >
        <Text style={styles.extractedType}>{item.type.toUpperCase()}</Text>
        <Text style={styles.extractedValue}>{item.normalized || item.value}</Text>
      </View>
    );
  };

  // Render help modal
  const renderHelpModal = () => (
    <Modal visible={showHelp} animationType="slide" transparent={true}>
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Voice Commands</Text>
            <TouchableOpacity onPress={() => setShowHelp(false)}>
              <Ionicons name="close" size={24} color="#333" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.commandsList}>
            {voiceCommands?.commands &&
              Object.entries(voiceCommands.commands).map(([category, commands]: [string, any]) => (
                <View key={category} style={styles.commandCategory}>
                  <Text style={styles.categoryTitle}>
                    {category.replace(/_/g, ' ').toUpperCase()}
                  </Text>
                  {commands.map((cmd: any, idx: number) => (
                    <View key={idx} style={styles.commandItem}>
                      <Text style={styles.commandExample}>"{cmd.example}"</Text>
                      <Text style={styles.commandAction}>{cmd.action.replace(/_/g, ' ')}</Text>
                    </View>
                  ))}
                </View>
              ))}
          </ScrollView>

          <View style={styles.tipsList}>
            <Text style={styles.tipsTitle}>Tips:</Text>
            {voiceCommands?.usage?.tips?.map((tip: string, idx: number) => (
              <Text key={idx} style={styles.tipItem}>
                {'\u2022'} {tip}
              </Text>
            ))}
          </View>
        </View>
      </View>
    </Modal>
  );

  // Compact mode
  if (compact) {
    return (
      <View style={styles.compactContainer}>
        <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
          <TouchableOpacity
            style={[
              styles.compactButton,
              { backgroundColor: recordingState === 'listening' ? '#F44336' : buttonColor },
            ]}
            onPress={toggleRecording}
            disabled={recordingState === 'processing'}
          >
            {recordingState === 'processing' ? (
              <ActivityIndicator color="#fff" size="small" />
            ) : (
              <Ionicons
                name={recordingState === 'listening' ? 'stop' : 'mic'}
                size={20}
                color="#fff"
              />
            )}
          </TouchableOpacity>
        </Animated.View>
        {renderHelpModal()}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Main Recording Button */}
      <View style={styles.buttonRow}>
        <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
          <TouchableOpacity
            style={[
              styles.recordButton,
              recordingState === 'listening' && styles.recordingButton,
              recordingState === 'processing' && styles.processingButton,
              recordingState === 'error' && styles.errorButton,
            ]}
            onPress={toggleRecording}
            disabled={recordingState === 'processing' || !isSupported}
          >
            {recordingState === 'processing' ? (
              <ActivityIndicator color="#fff" size="large" />
            ) : (
              <Ionicons
                name={recordingState === 'listening' ? 'stop' : 'mic'}
                size={32}
                color="#fff"
              />
            )}
          </TouchableOpacity>
        </Animated.View>

        {/* Help Button */}
        <TouchableOpacity style={styles.helpButton} onPress={() => setShowHelp(true)}>
          <Ionicons name="help-circle-outline" size={28} color="#666" />
        </TouchableOpacity>
      </View>

      {/* Status Text */}
      <Text style={styles.statusText}>
        {!isSupported
          ? 'Voice input not supported on this device'
          : recordingState === 'listening'
          ? 'Listening... Speak now'
          : recordingState === 'processing'
          ? 'Processing...'
          : recordingState === 'error'
          ? 'Error - tap to retry'
          : 'Tap to start speaking'}
      </Text>

      {/* Transcript Display */}
      {showTranscript && (transcript || interimTranscript) && (
        <View style={styles.transcriptContainer}>
          <View style={styles.transcriptHeader}>
            <Text style={styles.transcriptLabel}>Transcript:</Text>
            <TouchableOpacity onPress={clearTranscript}>
              <Ionicons name="trash-outline" size={20} color="#666" />
            </TouchableOpacity>
          </View>
          <Text style={styles.transcriptText}>
            {transcript}
            <Text style={styles.interimText}>{interimTranscript}</Text>
          </Text>
        </View>
      )}

      {/* Extracted Clinical Data */}
      {showExtractedData && extractedData.length > 0 && (
        <View style={styles.extractedContainer}>
          <Text style={styles.extractedLabel}>Extracted Data:</Text>
          <View style={styles.extractedList}>
            {extractedData.map(renderExtractedItem)}
          </View>
        </View>
      )}

      {/* Last Command Feedback */}
      {lastCommand && (
        <View style={styles.commandFeedback}>
          <Ionicons name="checkmark-circle" size={20} color="#4CAF50" />
          <Text style={styles.commandFeedbackText}>
            Command: {lastCommand.action.replace(/_/g, ' ')}
          </Text>
        </View>
      )}

      {renderHelpModal()}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
    alignItems: 'center',
  },
  compactContainer: {
    alignItems: 'center',
  },
  buttonRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  recordButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#4CAF50',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  recordingButton: {
    backgroundColor: '#F44336',
  },
  processingButton: {
    backgroundColor: '#FF9800',
  },
  errorButton: {
    backgroundColor: '#9E9E9E',
  },
  compactButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 2,
  },
  helpButton: {
    padding: 8,
  },
  statusText: {
    marginTop: 12,
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  transcriptContainer: {
    marginTop: 16,
    width: '100%',
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
  },
  transcriptHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  transcriptLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
    textTransform: 'uppercase',
  },
  transcriptText: {
    fontSize: 16,
    color: '#333',
    lineHeight: 24,
  },
  interimText: {
    color: '#999',
    fontStyle: 'italic',
  },
  extractedContainer: {
    marginTop: 16,
    width: '100%',
  },
  extractedLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
    textTransform: 'uppercase',
    marginBottom: 8,
  },
  extractedList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  extractedItem: {
    backgroundColor: '#fff',
    borderRadius: 4,
    padding: 8,
    borderLeftWidth: 3,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  extractedType: {
    fontSize: 10,
    fontWeight: '600',
    color: '#999',
    marginBottom: 2,
  },
  extractedValue: {
    fontSize: 14,
    color: '#333',
  },
  commandFeedback: {
    marginTop: 12,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#E8F5E9',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
    gap: 8,
  },
  commandFeedbackText: {
    fontSize: 14,
    color: '#2E7D32',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '90%',
    maxHeight: '80%',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
  },
  commandsList: {
    maxHeight: 300,
  },
  commandCategory: {
    marginBottom: 16,
  },
  categoryTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: '#666',
    marginBottom: 8,
    letterSpacing: 1,
  },
  commandItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#f9f9f9',
    borderRadius: 6,
    marginBottom: 4,
  },
  commandExample: {
    fontSize: 14,
    color: '#333',
    fontStyle: 'italic',
  },
  commandAction: {
    fontSize: 12,
    color: '#666',
  },
  tipsList: {
    marginTop: 16,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  tipsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  tipItem: {
    fontSize: 13,
    color: '#666',
    marginBottom: 4,
    paddingLeft: 8,
  },
});

export default VoiceInput;
