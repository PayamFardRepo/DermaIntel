import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Modal,
  TextInput,
  Dimensions,
  RefreshControl,
  Animated,
  Platform,
  KeyboardAvoidingView,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import { API_BASE_URL } from '../config';
import AuthService from '../services/AuthService';
import { useAuth } from '../contexts/AuthContext';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface LesionInsight {
  lesion_id: number;
  lesion_name: string;
  status: string;
  status_emoji: string;
  observation: string;
  recommendation: string;
  urgency: string;
}

interface ActionItem {
  priority: number;
  action: string;
  reason: string;
  urgency: string;
}

interface LesionData {
  id: number;
  name: string;
  body_location: string;
  current_risk_level: string;
  requires_attention: boolean;
  total_analyses: number;
  days_since_last_check: number | null;
  overdue_for_check: boolean;
  analysis_history: any[];
  comparison_history: any[];
}

interface MonitoringInsights {
  overall_status: string;
  summary: string;
  lesion_insights: LesionInsight[];
  correlations: string[];
  action_items: ActionItem[];
  next_message: string;
  lesions_data: LesionData[];
}

interface ComparisonData {
  lesion_id: number;
  lesion_name: string;
  comparison_available: boolean;
  baseline: any;
  current: any;
  time_difference_days: number;
  comparison?: {
    change_heatmap?: string;
    change_severity?: string;
    size_change_percent?: number;
  };
}

export default function AIMonitoringScreen() {
  const router = useRouter();
  const { user } = useAuth();
  const pulseAnim = useRef(new Animated.Value(1)).current;

  const [insights, setInsights] = useState<MonitoringInsights | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);

  // Modal states
  const [showAddLesion, setShowAddLesion] = useState(false);
  const [showComparison, setShowComparison] = useState(false);
  const [showAskAgent, setShowAskAgent] = useState(false);
  const [showLesionDetail, setShowLesionDetail] = useState(false);

  // Selected lesion for modals
  const [selectedLesion, setSelectedLesion] = useState<LesionData | null>(null);
  const [selectedLesionAnalysis, setSelectedLesionAnalysis] = useState<any>(null);
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null);

  // Add lesion form
  const [newLesionName, setNewLesionName] = useState('');
  const [newLesionLocation, setNewLesionLocation] = useState('');
  const [newLesionImage, setNewLesionImage] = useState<string | null>(null);
  const [isAddingLesion, setIsAddingLesion] = useState(false);
  const [addLesionStep, setAddLesionStep] = useState<'image' | 'details' | 'analyzing'>('image');
  const [analysisProgress, setAnalysisProgress] = useState('');
  // Update existing lesion (for tracking over time)
  const [updatingLesion, setUpdatingLesion] = useState<LesionData | null>(null);

  // Ask agent
  const [agentQuestion, setAgentQuestion] = useState('');
  const [agentAnswer, setAgentAnswer] = useState<string | null>(null);
  const [isAskingAgent, setIsAskingAgent] = useState(false);

  // Pulse animation
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.05,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);

  useEffect(() => {
    checkAvailability();
    loadInsights();
  }, []);

  const checkAvailability = async () => {
    try {
      const token = AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/ai-monitoring/status`, {
        headers: { 'Authorization': `Bearer ${token}` },
      });

      if (response.ok) {
        const data = await response.json();
        setIsAvailable(data.available);
      } else {
        setIsAvailable(false);
      }
    } catch (error) {
      console.error('Failed to check AI monitoring status:', error);
      setIsAvailable(false);
    }
  };

  const loadInsights = async () => {
    try {
      setError(null);
      const token = AuthService.getToken();

      const response = await fetch(`${API_BASE_URL}/ai-monitoring/insights`, {
        headers: { 'Authorization': `Bearer ${token}` },
      });

      if (!response.ok) {
        throw new Error('Failed to load insights');
      }

      const data = await response.json();
      setInsights(data);
    } catch (error) {
      console.error('Failed to load insights:', error);
      setError('Failed to load monitoring insights');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  const onRefresh = () => {
    setIsRefreshing(true);
    loadInsights();
  };

  const pickImageForLesion = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert('Permission Required', 'Please allow access to your photo library.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setNewLesionImage(result.assets[0].uri);
      setAddLesionStep('details');
    }
  };

  const takePhotoForLesion = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert('Permission Required', 'Please allow access to your camera.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setNewLesionImage(result.assets[0].uri);
      setAddLesionStep('details');
    }
  };

  const resetAddLesionModal = () => {
    setShowAddLesion(false);
    setNewLesionName('');
    setNewLesionLocation('');
    setNewLesionImage(null);
    setAddLesionStep('image');
    setAnalysisProgress('');
    setUpdatingLesion(null);
  };

  const startUpdateLesion = (lesion: LesionData) => {
    setUpdatingLesion(lesion);
    setNewLesionName(lesion.name);
    setNewLesionLocation(lesion.body_location || '');
    setShowAddLesion(true);
  };

  const addLesion = async () => {
    if (!newLesionImage) return;
    // For new lesions, require a name. For updates, we already have the name.
    if (!updatingLesion && !newLesionName.trim()) return;

    setIsAddingLesion(true);
    setAddLesionStep('analyzing');

    try {
      const token = AuthService.getToken();

      // Step 1: Upload and analyze the image
      setAnalysisProgress('Uploading image...');
      const imageFormData = new FormData();
      const filename = newLesionImage.split('/').pop() || 'image.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : 'image/jpeg';

      imageFormData.append('file', {
        uri: newLesionImage,
        name: filename,
        type,
      } as any);

      // Add body location if provided
      if (newLesionLocation.trim()) {
        imageFormData.append('body_location', newLesionLocation.trim());
      }

      // If updating existing lesion, link to that group
      if (updatingLesion) {
        imageFormData.append('lesion_group_id', updatingLesion.id.toString());
      }

      setAnalysisProgress(updatingLesion ? 'Analyzing changes...' : 'Analyzing lesion...');
      const analysisResponse = await fetch(`${API_BASE_URL}/full_classify/`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: imageFormData,
      });

      if (!analysisResponse.ok) {
        throw new Error('Failed to analyze image');
      }

      const analysisResult = await analysisResponse.json();
      const analysisId = analysisResult.analysis_id;

      // Step 2: For new lesions, create lesion group. For updates, we're done.
      if (updatingLesion) {
        // Analysis is already linked to the group via lesion_group_id
        resetAddLesionModal();
        loadInsights();
        Alert.alert(
          'Lesion Updated',
          `New photo added to "${updatingLesion.name}". You can now compare changes over time.${analysisResult.risk_level ? ` Current risk level: ${analysisResult.risk_level}` : ''}`
        );
      } else {
        setAnalysisProgress('Creating lesion tracker...');
        const lesionFormData = new FormData();
        lesionFormData.append('lesion_name', newLesionName.trim());
        if (newLesionLocation.trim()) {
          lesionFormData.append('body_location', newLesionLocation.trim());
        }
        lesionFormData.append('monitoring_frequency', 'monthly');
        if (analysisId) {
          lesionFormData.append('analysis_id', analysisId.toString());
        }

        const lesionResponse = await fetch(`${API_BASE_URL}/lesion_groups/`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${token}` },
          body: lesionFormData,
        });

        if (lesionResponse.ok) {
          resetAddLesionModal();
          loadInsights();
          Alert.alert(
            'Lesion Added',
            `"${newLesionName}" has been analyzed and added to monitoring.${analysisResult.risk_level ? ` Risk level: ${analysisResult.risk_level}` : ''}`
          );
        } else {
          throw new Error('Failed to create lesion group');
        }
      }
    } catch (error) {
      console.error('Failed to add lesion:', error);
      Alert.alert('Error', `Failed to ${updatingLesion ? 'update' : 'add'} lesion. Please try again.`);
      setAddLesionStep('details');
    } finally {
      setIsAddingLesion(false);
    }
  };

  const loadLesionAnalysis = async (lesionId: number) => {
    try {
      const token = AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/ai-monitoring/lesion/${lesionId}/analysis`, {
        headers: { 'Authorization': `Bearer ${token}` },
      });

      if (response.ok) {
        const data = await response.json();
        setSelectedLesionAnalysis(data);
      }
    } catch (error) {
      console.error('Failed to load lesion analysis:', error);
    }
  };

  const loadComparison = async (lesionId: number) => {
    try {
      const token = AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/ai-monitoring/comparison/${lesionId}`, {
        headers: { 'Authorization': `Bearer ${token}` },
      });

      if (response.ok) {
        const data = await response.json();
        setComparisonData(data);
        setShowComparison(true);
      }
    } catch (error) {
      console.error('Failed to load comparison:', error);
    }
  };

  const askAgent = async () => {
    if (!agentQuestion.trim()) return;

    setIsAskingAgent(true);
    setAgentAnswer(null);

    try {
      const token = AuthService.getToken();
      const formData = new FormData();
      formData.append('question', agentQuestion.trim());
      if (selectedLesion) {
        formData.append('lesion_id', selectedLesion.id.toString());
      }

      const response = await fetch(`${API_BASE_URL}/ai-monitoring/ask`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setAgentAnswer(data.answer);
      }
    } catch (error) {
      console.error('Failed to ask agent:', error);
      setAgentAnswer('Sorry, I encountered an error. Please try again.');
    } finally {
      setIsAskingAgent(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'all_clear':
      case 'stable':
      case 'improving':
        return '#22c55e';
      case 'needs_attention':
      case 'monitoring':
        return '#f59e0b';
      case 'urgent_review':
      case 'concerning':
      case 'urgent':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk?.toLowerCase()) {
      case 'low': return '#22c55e';
      case 'medium': return '#f59e0b';
      case 'high': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'routine': return '#22c55e';
      case 'soon': return '#f59e0b';
      case 'urgent': return '#ef4444';
      default: return '#6b7280';
    }
  };

  // Unavailable state
  if (isAvailable === false) {
    return (
      <View style={styles.container}>
        <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.gradient}>
          <View style={styles.header}>
            <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
              <Ionicons name="arrow-back" size={24} color="#fff" />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>AI Monitoring</Text>
            <View style={{ width: 40 }} />
          </View>

          <View style={styles.unavailableContainer}>
            <Ionicons name="cloud-offline" size={64} color="#6b7280" />
            <Text style={styles.unavailableTitle}>AI Monitoring Unavailable</Text>
            <Text style={styles.unavailableText}>
              This feature requires OpenAI API configuration.
            </Text>
          </View>
        </LinearGradient>
      </View>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <View style={styles.container}>
        <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.gradient}>
          <View style={styles.loadingContainer}>
            <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
              <LinearGradient
                colors={['#6366f1', '#8b5cf6']}
                style={styles.loadingAvatar}
              >
                <Ionicons name="eye" size={40} color="#fff" />
              </LinearGradient>
            </Animated.View>
            <Text style={styles.loadingText}>Analyzing your lesions...</Text>
            <Text style={styles.loadingSubtext}>
              I'm reviewing your tracking data and generating personalized insights
            </Text>
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
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="#fff" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>AI Monitoring</Text>
          <TouchableOpacity
            onPress={() => setShowAskAgent(true)}
            style={styles.askButton}
          >
            <Ionicons name="chatbubble-ellipses" size={22} color="#fff" />
          </TouchableOpacity>
        </View>

        <ScrollView
          style={styles.content}
          contentContainerStyle={styles.contentContainer}
          refreshControl={
            <RefreshControl refreshing={isRefreshing} onRefresh={onRefresh} tintColor="#6366f1" />
          }
        >
          {/* AI Status Card */}
          {insights && (
            <View style={[
              styles.statusCard,
              { borderLeftColor: getStatusColor(insights.overall_status) }
            ]}>
              <View style={styles.statusHeader}>
                <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
                  <LinearGradient
                    colors={['#6366f1', '#8b5cf6']}
                    style={styles.aiAvatar}
                  >
                    <Ionicons name="eye" size={24} color="#fff" />
                  </LinearGradient>
                </Animated.View>
                <View style={styles.statusTextContainer}>
                  <Text style={styles.statusLabel}>
                    {insights.overall_status === 'all_clear' && '✅ All Clear'}
                    {insights.overall_status === 'needs_attention' && '⚠️ Needs Attention'}
                    {insights.overall_status === 'urgent_review' && '🔴 Urgent Review Needed'}
                  </Text>
                  <Text style={styles.statusSummary}>{insights.summary}</Text>
                </View>
              </View>

              {insights.next_message && (
                <Text style={styles.nextMessage}>{insights.next_message}</Text>
              )}
            </View>
          )}

          {/* Action Items */}
          {insights?.action_items && insights.action_items.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Recommended Actions</Text>
              {insights.action_items.map((item, index) => (
                <View
                  key={index}
                  style={[styles.actionItem, { borderLeftColor: getUrgencyColor(item.urgency) }]}
                >
                  <View style={styles.actionPriority}>
                    <Text style={styles.actionPriorityText}>{item.priority}</Text>
                  </View>
                  <View style={styles.actionContent}>
                    <Text style={styles.actionText}>{item.action}</Text>
                    <Text style={styles.actionReason}>{item.reason}</Text>
                  </View>
                </View>
              ))}
            </View>
          )}

          {/* Lesion Cards */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Tracked Lesions</Text>
              <TouchableOpacity
                style={styles.addButton}
                onPress={() => setShowAddLesion(true)}
              >
                <Ionicons name="add" size={20} color="#6366f1" />
                <Text style={styles.addButtonText}>Add</Text>
              </TouchableOpacity>
            </View>

            {insights?.lesions_data && insights.lesions_data.length > 0 ? (
              insights.lesions_data.map((lesion) => {
                const insight = insights.lesion_insights?.find(i => i.lesion_id === lesion.id);
                return (
                  <TouchableOpacity
                    key={lesion.id}
                    style={[
                      styles.lesionCard,
                      lesion.requires_attention && styles.lesionCardAttention
                    ]}
                    onPress={() => {
                      setSelectedLesion(lesion);
                      loadLesionAnalysis(lesion.id);
                      setShowLesionDetail(true);
                    }}
                  >
                    <View style={styles.lesionHeader}>
                      <View style={styles.lesionInfo}>
                        <Text style={styles.lesionEmoji}>
                          {insight?.status_emoji || '📍'}
                        </Text>
                        <View>
                          <Text style={styles.lesionName}>{lesion.name}</Text>
                          <Text style={styles.lesionLocation}>
                            {lesion.body_location || 'Location not set'}
                          </Text>
                        </View>
                      </View>
                      <View style={[
                        styles.riskBadge,
                        { backgroundColor: getRiskColor(lesion.current_risk_level) + '20' }
                      ]}>
                        <Text style={[
                          styles.riskText,
                          { color: getRiskColor(lesion.current_risk_level) }
                        ]}>
                          {lesion.current_risk_level || 'Unknown'}
                        </Text>
                      </View>
                    </View>

                    {insight && (
                      <Text style={styles.lesionObservation}>{insight.observation}</Text>
                    )}

                    <View style={styles.lesionMeta}>
                      <Text style={styles.lesionMetaText}>
                        {lesion.total_analyses} {lesion.total_analyses === 1 ? 'analysis' : 'analyses'}
                      </Text>
                      {lesion.days_since_last_check !== null && (
                        <Text style={[
                          styles.lesionMetaText,
                          lesion.overdue_for_check && styles.overdueText
                        ]}>
                          {lesion.overdue_for_check ? '⚠️ ' : ''}
                          Last check: {lesion.days_since_last_check} days ago
                        </Text>
                      )}
                    </View>

                    <View style={styles.lesionActions}>
                      <TouchableOpacity
                        style={[styles.lesionActionButton, styles.lesionActionButtonPrimary]}
                        onPress={(e) => {
                          e.stopPropagation();
                          startUpdateLesion(lesion);
                        }}
                      >
                        <Ionicons name="camera" size={16} color="#fff" />
                        <Text style={styles.lesionActionTextPrimary}>Update</Text>
                      </TouchableOpacity>
                      <TouchableOpacity
                        style={styles.lesionActionButton}
                        onPress={(e) => {
                          e.stopPropagation();
                          loadComparison(lesion.id);
                        }}
                      >
                        <Ionicons name="git-compare" size={16} color="#6366f1" />
                        <Text style={styles.lesionActionText}>Compare</Text>
                      </TouchableOpacity>
                      <TouchableOpacity
                        style={styles.lesionActionButton}
                        onPress={(e) => {
                          e.stopPropagation();
                          setSelectedLesion(lesion);
                          setShowAskAgent(true);
                        }}
                      >
                        <Ionicons name="chatbubble" size={16} color="#6366f1" />
                        <Text style={styles.lesionActionText}>Ask AI</Text>
                      </TouchableOpacity>
                    </View>
                  </TouchableOpacity>
                );
              })
            ) : (
              <View style={styles.emptyState}>
                <Ionicons name="body" size={48} color="#4b5563" />
                <Text style={styles.emptyText}>No lesions tracked yet</Text>
                <Text style={styles.emptySubtext}>
                  Add your first lesion to start AI-powered monitoring
                </Text>
                <TouchableOpacity
                  style={styles.emptyButton}
                  onPress={() => setShowAddLesion(true)}
                >
                  <Text style={styles.emptyButtonText}>Add First Lesion</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>

          {/* Correlations */}
          {insights?.correlations && insights.correlations.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Patterns & Correlations</Text>
              {insights.correlations.map((correlation, index) => (
                <View key={index} style={styles.correlationItem}>
                  <Ionicons name="analytics" size={18} color="#8b5cf6" />
                  <Text style={styles.correlationText}>{correlation}</Text>
                </View>
              ))}
            </View>
          )}

          <View style={styles.disclaimer}>
            <Ionicons name="information-circle" size={16} color="#6b7280" />
            <Text style={styles.disclaimerText}>
              AI monitoring supplements but does not replace professional dermatologist care.
            </Text>
          </View>
        </ScrollView>

        {/* Add Lesion Modal */}
        <Modal
          visible={showAddLesion}
          transparent
          animationType="slide"
          onRequestClose={resetAddLesionModal}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>
                  {addLesionStep === 'image' && (updatingLesion ? `Update: ${updatingLesion.name}` : 'Capture Lesion Image')}
                  {addLesionStep === 'details' && (updatingLesion ? 'Confirm Update' : 'Lesion Details')}
                  {addLesionStep === 'analyzing' && (updatingLesion ? 'Comparing...' : 'Analyzing...')}
                </Text>
                <TouchableOpacity onPress={resetAddLesionModal}>
                  <Ionicons name="close" size={24} color="#fff" />
                </TouchableOpacity>
              </View>

              {/* Step 1: Capture Image */}
              {addLesionStep === 'image' && (
                <View style={styles.imageStepContainer}>
                  <View style={styles.imageStepIcon}>
                    <Ionicons name="camera" size={48} color="#6366f1" />
                  </View>
                  <Text style={styles.imageStepTitle}>
                    {updatingLesion ? 'Take a new photo for comparison' : 'Take a photo of the lesion'}
                  </Text>
                  <Text style={styles.imageStepSubtitle}>
                    {updatingLesion
                      ? 'Take a photo of the same lesion to track changes over time'
                      : 'A clear, well-lit photo helps the AI provide accurate analysis'}
                  </Text>

                  <TouchableOpacity
                    style={styles.imageCaptureButton}
                    onPress={takePhotoForLesion}
                  >
                    <Ionicons name="camera" size={24} color="#fff" />
                    <Text style={styles.imageCaptureButtonText}>Take Photo</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={styles.imagePickerButton}
                    onPress={pickImageForLesion}
                  >
                    <Ionicons name="images" size={20} color="#6366f1" />
                    <Text style={styles.imagePickerButtonText}>Choose from Gallery</Text>
                  </TouchableOpacity>
                </View>
              )}

              {/* Step 2: Enter Details */}
              {addLesionStep === 'details' && (
                <View>
                  {/* Image Preview */}
                  {newLesionImage && (
                    <View style={styles.imagePreviewContainer}>
                      <Image
                        source={{ uri: newLesionImage }}
                        style={styles.imagePreview}
                        resizeMode="cover"
                      />
                      <TouchableOpacity
                        style={styles.changeImageButton}
                        onPress={() => setAddLesionStep('image')}
                      >
                        <Ionicons name="refresh" size={16} color="#6366f1" />
                        <Text style={styles.changeImageText}>Change</Text>
                      </TouchableOpacity>
                    </View>
                  )}

                  {updatingLesion ? (
                    <View style={styles.updateInfoBox}>
                      <Ionicons name="information-circle" size={20} color="#6366f1" />
                      <Text style={styles.updateInfoText}>
                        This photo will be added to "{updatingLesion.name}" for tracking changes over time.
                      </Text>
                    </View>
                  ) : (
                    <>
                      <Text style={styles.inputLabel}>
                        Lesion Name <Text style={styles.requiredStar}>*</Text>
                      </Text>
                      <TextInput
                        style={styles.input}
                        placeholder="e.g., 'Mole on left shoulder'"
                        placeholderTextColor="#6b7280"
                        value={newLesionName}
                        onChangeText={setNewLesionName}
                        autoFocus
                      />

                      <TextInput
                        style={styles.input}
                        placeholder="Body location (optional)"
                        placeholderTextColor="#6b7280"
                        value={newLesionLocation}
                        onChangeText={setNewLesionLocation}
                      />
                    </>
                  )}

                  <TouchableOpacity
                    style={[styles.modalButton, (!updatingLesion && !newLesionName.trim()) && styles.modalButtonDisabled]}
                    onPress={addLesion}
                    disabled={(!updatingLesion && !newLesionName.trim()) || isAddingLesion}
                  >
                    <Text style={styles.modalButtonText}>
                      {updatingLesion ? 'Analyze & Compare' : 'Analyze & Add Lesion'}
                    </Text>
                  </TouchableOpacity>
                </View>
              )}

              {/* Step 3: Analyzing */}
              {addLesionStep === 'analyzing' && (
                <View style={styles.analyzingContainer}>
                  <ActivityIndicator size="large" color="#6366f1" />
                  <Text style={styles.analyzingText}>{analysisProgress}</Text>
                  <Text style={styles.analyzingSubtext}>
                    This may take a few seconds...
                  </Text>
                </View>
              )}
            </View>
          </View>
        </Modal>

        {/* Comparison Modal */}
        <Modal
          visible={showComparison}
          transparent
          animationType="slide"
          onRequestClose={() => setShowComparison(false)}
        >
          <View style={styles.modalOverlay}>
            <View style={[styles.modalContent, styles.comparisonModal]}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>
                  {comparisonData?.lesion_name || 'Comparison'}
                </Text>
                <TouchableOpacity onPress={() => setShowComparison(false)}>
                  <Ionicons name="close" size={24} color="#fff" />
                </TouchableOpacity>
              </View>

              {comparisonData?.comparison_available ? (
                <ScrollView style={styles.comparisonContent}>
                  <Text style={styles.comparisonSubtitle}>
                    Comparing changes in this lesion over time
                  </Text>
                  <Text style={styles.comparisonPeriod}>
                    {comparisonData.time_difference_days} days between analyses
                  </Text>

                  {/* Side by side images would go here */}
                  <View style={styles.comparisonImages}>
                    <View style={styles.comparisonImageContainer}>
                      <Text style={styles.comparisonLabel}>Baseline</Text>
                      <View style={styles.imagePlaceholder}>
                        <Ionicons name="image" size={40} color="#4b5563" />
                        <Text style={styles.imagePlaceholderText}>
                          {comparisonData.baseline?.date?.split('T')[0]}
                        </Text>
                      </View>
                    </View>
                    <View style={styles.comparisonImageContainer}>
                      <Text style={styles.comparisonLabel}>Current</Text>
                      <View style={styles.imagePlaceholder}>
                        <Ionicons name="image" size={40} color="#4b5563" />
                        <Text style={styles.imagePlaceholderText}>
                          {comparisonData.current?.date?.split('T')[0]}
                        </Text>
                      </View>
                    </View>
                  </View>

                  {/* Comparison metrics */}
                  {comparisonData.comparison && (
                    <View style={styles.comparisonMetrics}>
                      <View style={styles.metricRow}>
                        <Text style={styles.metricLabel}>Change Severity</Text>
                        <Text style={[
                          styles.metricValue,
                          { color: getStatusColor(comparisonData.comparison.change_severity || '') }
                        ]}>
                          {comparisonData.comparison.change_severity || 'Unknown'}
                        </Text>
                      </View>
                      {comparisonData.comparison.size_change_percent !== undefined && (
                        <View style={styles.metricRow}>
                          <Text style={styles.metricLabel}>Size Change</Text>
                          <Text style={styles.metricValue}>
                            {comparisonData.comparison.size_change_percent > 0 ? '+' : ''}
                            {comparisonData.comparison.size_change_percent?.toFixed(1)}%
                          </Text>
                        </View>
                      )}
                    </View>
                  )}

                  {/* Change heatmap */}
                  {comparisonData.comparison?.change_heatmap && (
                    <View style={styles.heatmapContainer}>
                      <Text style={styles.heatmapTitle}>Change Heatmap</Text>
                      <Image
                        source={{ uri: `data:image/png;base64,${comparisonData.comparison.change_heatmap}` }}
                        style={styles.heatmapImage}
                        resizeMode="contain"
                      />
                      <Text style={styles.heatmapLegend}>
                        Red areas indicate the most change
                      </Text>
                    </View>
                  )}
                </ScrollView>
              ) : (
                <View style={styles.noComparisonContainer}>
                  <Ionicons name="time-outline" size={48} color="#4b5563" />
                  <Text style={styles.noComparisonTitle}>Track Changes Over Time</Text>
                  <Text style={styles.noComparisonText}>
                    {comparisonData?.message || 'Need at least 2 analyses of this lesion to compare changes over time.'}
                  </Text>
                  <Text style={styles.noComparisonHint}>
                    Tip: Analyze this same lesion again in a few weeks to see how it's changing.
                  </Text>
                </View>
              )}
            </View>
          </View>
        </Modal>

        {/* Ask Agent Modal */}
        <Modal
          visible={showAskAgent}
          transparent
          animationType="slide"
          onRequestClose={() => {
            setShowAskAgent(false);
            setAgentAnswer(null);
            setAgentQuestion('');
          }}
        >
          <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={styles.modalOverlay}
          >
            <View style={[styles.modalContent, styles.askModal]}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>
                  Ask AI {selectedLesion ? `about ${selectedLesion.name}` : 'Monitoring Agent'}
                </Text>
                <TouchableOpacity onPress={() => {
                  setShowAskAgent(false);
                  setAgentAnswer(null);
                  setAgentQuestion('');
                  setSelectedLesion(null);
                }}>
                  <Ionicons name="close" size={24} color="#fff" />
                </TouchableOpacity>
              </View>

              <ScrollView style={styles.askContent}>
                {agentAnswer && (
                  <View style={styles.answerContainer}>
                    <View style={styles.answerHeader}>
                      <LinearGradient
                        colors={['#6366f1', '#8b5cf6']}
                        style={styles.answerAvatar}
                      >
                        <Ionicons name="eye" size={16} color="#fff" />
                      </LinearGradient>
                      <Text style={styles.answerLabel}>AI Response</Text>
                    </View>
                    <Text style={styles.answerText}>{agentAnswer}</Text>
                  </View>
                )}

                <View style={styles.suggestedQuestions}>
                  <Text style={styles.suggestedTitle}>Try asking:</Text>
                  {[
                    "How has this lesion changed over time?",
                    "Should I be concerned about the growth rate?",
                    "When should I see a dermatologist?",
                    "What's the connection with my sun exposure?"
                  ].map((q, i) => (
                    <TouchableOpacity
                      key={i}
                      style={styles.suggestedQuestion}
                      onPress={() => setAgentQuestion(q)}
                    >
                      <Text style={styles.suggestedQuestionText}>{q}</Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </ScrollView>

              <View style={styles.askInputContainer}>
                <TextInput
                  style={styles.askInput}
                  placeholder="Type your question..."
                  placeholderTextColor="#6b7280"
                  value={agentQuestion}
                  onChangeText={setAgentQuestion}
                  multiline
                />
                <TouchableOpacity
                  style={[styles.askSendButton, (!agentQuestion.trim() || isAskingAgent) && styles.askSendButtonDisabled]}
                  onPress={askAgent}
                  disabled={!agentQuestion.trim() || isAskingAgent}
                >
                  {isAskingAgent ? (
                    <ActivityIndicator color="#fff" size="small" />
                  ) : (
                    <Ionicons name="send" size={20} color="#fff" />
                  )}
                </TouchableOpacity>
              </View>
            </View>
          </KeyboardAvoidingView>
        </Modal>

        {/* Lesion Detail Modal */}
        <Modal
          visible={showLesionDetail}
          transparent
          animationType="slide"
          onRequestClose={() => {
            setShowLesionDetail(false);
            setSelectedLesionAnalysis(null);
          }}
        >
          <View style={styles.modalOverlay}>
            <View style={[styles.modalContent, styles.detailModal]}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>{selectedLesion?.name}</Text>
                <TouchableOpacity onPress={() => {
                  setShowLesionDetail(false);
                  setSelectedLesionAnalysis(null);
                }}>
                  <Ionicons name="close" size={24} color="#fff" />
                </TouchableOpacity>
              </View>

              <ScrollView style={styles.detailContent}>
                {selectedLesionAnalysis ? (
                  <>
                    {/* Current Status */}
                    <View style={styles.detailSection}>
                      <Text style={styles.detailSectionTitle}>Current Status</Text>
                      <View style={[
                        styles.statusBadgeLarge,
                        { backgroundColor: getStatusColor(selectedLesionAnalysis.current_status?.status) + '20' }
                      ]}>
                        <Text style={[
                          styles.statusBadgeText,
                          { color: getStatusColor(selectedLesionAnalysis.current_status?.status) }
                        ]}>
                          {selectedLesionAnalysis.current_status?.status}
                        </Text>
                      </View>
                      <Text style={styles.detailText}>
                        {selectedLesionAnalysis.current_status?.explanation}
                      </Text>
                    </View>

                    {/* History Summary */}
                    <View style={styles.detailSection}>
                      <Text style={styles.detailSectionTitle}>History</Text>
                      <Text style={styles.detailText}>
                        {selectedLesionAnalysis.history_summary}
                      </Text>
                    </View>

                    {/* Trend Analysis */}
                    {selectedLesionAnalysis.trend_analysis && (
                      <View style={styles.detailSection}>
                        <Text style={styles.detailSectionTitle}>Trends</Text>
                        <View style={styles.trendItem}>
                          <Text style={styles.trendLabel}>Overall:</Text>
                          <Text style={styles.trendValue}>
                            {selectedLesionAnalysis.trend_analysis.overall_trend}
                          </Text>
                        </View>
                        {selectedLesionAnalysis.trend_analysis.size_trend && (
                          <View style={styles.trendItem}>
                            <Text style={styles.trendLabel}>Size:</Text>
                            <Text style={styles.trendValue}>
                              {selectedLesionAnalysis.trend_analysis.size_trend}
                            </Text>
                          </View>
                        )}
                      </View>
                    )}

                    {/* Key Observations */}
                    {selectedLesionAnalysis.key_observations && (
                      <View style={styles.detailSection}>
                        <Text style={styles.detailSectionTitle}>Key Observations</Text>
                        {selectedLesionAnalysis.key_observations.map((obs: string, i: number) => (
                          <View key={i} style={styles.observationItem}>
                            <Text style={styles.observationBullet}>•</Text>
                            <Text style={styles.observationText}>{obs}</Text>
                          </View>
                        ))}
                      </View>
                    )}

                    {/* Recommendations */}
                    {selectedLesionAnalysis.recommendations && (
                      <View style={styles.detailSection}>
                        <Text style={styles.detailSectionTitle}>Recommendations</Text>
                        {selectedLesionAnalysis.recommendations.map((rec: any, i: number) => (
                          <View key={i} style={styles.recommendationItem}>
                            <Text style={styles.recommendationAction}>{rec.action}</Text>
                            <Text style={styles.recommendationMeta}>
                              {rec.timeframe} — {rec.reason}
                            </Text>
                          </View>
                        ))}
                      </View>
                    )}

                    {/* Watch For */}
                    {selectedLesionAnalysis.watch_for && (
                      <View style={styles.detailSection}>
                        <Text style={styles.detailSectionTitle}>Watch For</Text>
                        {selectedLesionAnalysis.watch_for.map((item: string, i: number) => (
                          <View key={i} style={styles.watchItem}>
                            <Ionicons name="eye" size={14} color="#f59e0b" />
                            <Text style={styles.watchText}>{item}</Text>
                          </View>
                        ))}
                      </View>
                    )}
                  </>
                ) : (
                  <View style={styles.loadingDetail}>
                    <ActivityIndicator color="#6366f1" />
                    <Text style={styles.loadingDetailText}>Loading analysis...</Text>
                  </View>
                )}
              </ScrollView>
            </View>
          </View>
        </Modal>
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
  askButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(99, 102, 241, 0.3)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 16,
    paddingBottom: 32,
  },

  // Loading
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
  },
  loadingAvatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  loadingText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
  },
  loadingSubtext: {
    color: '#9ca3af',
    fontSize: 14,
    textAlign: 'center',
  },

  // Status Card
  statusCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
  },
  statusHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  aiAvatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  statusTextContainer: {
    flex: 1,
  },
  statusLabel: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  statusSummary: {
    color: '#d1d5db',
    fontSize: 14,
    lineHeight: 20,
  },
  nextMessage: {
    color: '#9ca3af',
    fontSize: 13,
    fontStyle: 'italic',
    marginTop: 8,
  },

  // Sections
  section: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  addButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: 'rgba(99, 102, 241, 0.2)',
    borderRadius: 16,
  },
  addButtonText: {
    color: '#6366f1',
    fontSize: 14,
    fontWeight: '500',
  },

  // Action Items
  actionItem: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
    borderLeftWidth: 3,
  },
  actionPriority: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: 'rgba(99, 102, 241, 0.3)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  actionPriorityText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '700',
  },
  actionContent: {
    flex: 1,
  },
  actionText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 2,
  },
  actionReason: {
    color: '#9ca3af',
    fontSize: 12,
  },

  // Lesion Cards
  lesionCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
  },
  lesionCardAttention: {
    borderWidth: 1,
    borderColor: 'rgba(245, 158, 11, 0.5)',
  },
  lesionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  lesionInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  lesionEmoji: {
    fontSize: 24,
    marginRight: 10,
  },
  lesionName: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
  lesionLocation: {
    color: '#9ca3af',
    fontSize: 12,
  },
  riskBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  riskText: {
    fontSize: 11,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  lesionObservation: {
    color: '#d1d5db',
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 8,
  },
  lesionMeta: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  lesionMetaText: {
    color: '#6b7280',
    fontSize: 11,
  },
  overdueText: {
    color: '#f59e0b',
  },
  lesionActions: {
    flexDirection: 'row',
    gap: 12,
  },
  lesionActionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingVertical: 6,
    paddingHorizontal: 10,
    backgroundColor: 'rgba(99, 102, 241, 0.15)',
    borderRadius: 8,
  },
  lesionActionText: {
    color: '#6366f1',
    fontSize: 12,
    fontWeight: '500',
  },
  lesionActionButtonPrimary: {
    backgroundColor: '#6366f1',
  },
  lesionActionTextPrimary: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
  },

  // Empty State
  emptyState: {
    alignItems: 'center',
    padding: 32,
  },
  emptyText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginTop: 16,
  },
  emptySubtext: {
    color: '#6b7280',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 8,
  },
  emptyButton: {
    marginTop: 20,
    backgroundColor: '#6366f1',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  emptyButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },

  // Correlations
  correlationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: 'rgba(139, 92, 246, 0.1)',
    padding: 12,
    borderRadius: 10,
    marginBottom: 8,
    gap: 10,
  },
  correlationText: {
    flex: 1,
    color: '#d1d5db',
    fontSize: 13,
    lineHeight: 18,
  },

  // Disclaimer
  disclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    marginTop: 8,
  },
  disclaimerText: {
    flex: 1,
    color: '#6b7280',
    fontSize: 11,
    lineHeight: 16,
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
  },
  unavailableText: {
    color: '#9ca3af',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 8,
  },

  // Modals
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.8)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#1a1a2e',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    padding: 20,
    maxHeight: '80%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  modalTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
  },
  input: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 12,
    padding: 14,
    color: '#fff',
    fontSize: 15,
    marginBottom: 12,
  },
  inputLabel: {
    color: '#d1d5db',
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 6,
  },
  requiredStar: {
    color: '#ef4444',
  },
  modalButton: {
    backgroundColor: '#6366f1',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  modalButtonDisabled: {
    opacity: 0.5,
  },
  modalButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },

  // Add Lesion Multi-Step
  imageStepContainer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  imageStepIcon: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'rgba(99, 102, 241, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  imageStepTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
    textAlign: 'center',
  },
  imageStepSubtitle: {
    color: '#9ca3af',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 24,
    paddingHorizontal: 16,
  },
  imageCaptureButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    backgroundColor: '#6366f1',
    borderRadius: 12,
    paddingVertical: 16,
    paddingHorizontal: 32,
    width: '100%',
    marginBottom: 12,
  },
  imageCaptureButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  imagePickerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: 'rgba(99, 102, 241, 0.15)',
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 24,
    width: '100%',
  },
  imagePickerButtonText: {
    color: '#6366f1',
    fontSize: 15,
    fontWeight: '500',
  },
  imagePreviewContainer: {
    position: 'relative',
    marginBottom: 16,
    borderRadius: 12,
    overflow: 'hidden',
  },
  imagePreview: {
    width: '100%',
    height: 200,
    borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.05)',
  },
  changeImageButton: {
    position: 'absolute',
    top: 10,
    right: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 16,
  },
  changeImageText: {
    color: '#6366f1',
    fontSize: 12,
    fontWeight: '500',
  },
  analyzingContainer: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  analyzingText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginTop: 20,
  },
  analyzingSubtext: {
    color: '#6b7280',
    fontSize: 14,
    marginTop: 8,
  },
  updateInfoBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: 'rgba(99, 102, 241, 0.1)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
    gap: 10,
  },
  updateInfoText: {
    flex: 1,
    color: '#d1d5db',
    fontSize: 14,
    lineHeight: 20,
  },

  // Comparison Modal
  comparisonModal: {
    maxHeight: '90%',
  },
  comparisonContent: {
    flex: 1,
  },
  comparisonSubtitle: {
    color: '#6366f1',
    fontSize: 13,
    textAlign: 'center',
    marginBottom: 4,
  },
  comparisonPeriod: {
    color: '#9ca3af',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 16,
  },
  comparisonImages: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  comparisonImageContainer: {
    flex: 1,
    marginHorizontal: 4,
  },
  comparisonLabel: {
    color: '#9ca3af',
    fontSize: 12,
    textAlign: 'center',
    marginBottom: 8,
  },
  imagePlaceholder: {
    height: 140,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  imagePlaceholderText: {
    color: '#6b7280',
    fontSize: 11,
    marginTop: 8,
  },
  comparisonMetrics: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 6,
  },
  metricLabel: {
    color: '#9ca3af',
    fontSize: 14,
  },
  metricValue: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  heatmapContainer: {
    marginTop: 8,
  },
  heatmapTitle: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  heatmapImage: {
    width: '100%',
    height: 200,
    borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.05)',
  },
  heatmapLegend: {
    color: '#6b7280',
    fontSize: 11,
    textAlign: 'center',
    marginTop: 8,
  },
  noComparisonContainer: {
    alignItems: 'center',
    padding: 32,
  },
  noComparisonTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginTop: 16,
  },
  noComparisonText: {
    color: '#9ca3af',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 8,
    lineHeight: 20,
  },
  noComparisonHint: {
    color: '#6366f1',
    fontSize: 13,
    textAlign: 'center',
    marginTop: 16,
    fontStyle: 'italic',
  },

  // Ask Modal
  askModal: {
    maxHeight: '85%',
  },
  askContent: {
    flex: 1,
    marginBottom: 16,
  },
  answerContainer: {
    backgroundColor: 'rgba(99, 102, 241, 0.1)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
  },
  answerHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  answerAvatar: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 8,
  },
  answerLabel: {
    color: '#8b5cf6',
    fontSize: 12,
    fontWeight: '600',
  },
  answerText: {
    color: '#d1d5db',
    fontSize: 14,
    lineHeight: 20,
  },
  suggestedQuestions: {
    marginTop: 8,
  },
  suggestedTitle: {
    color: '#6b7280',
    fontSize: 12,
    marginBottom: 8,
  },
  suggestedQuestion: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 6,
  },
  suggestedQuestionText: {
    color: '#9ca3af',
    fontSize: 13,
  },
  askInputContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 10,
  },
  askInput: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    color: '#fff',
    fontSize: 14,
    maxHeight: 80,
  },
  askSendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#6366f1',
    alignItems: 'center',
    justifyContent: 'center',
  },
  askSendButtonDisabled: {
    opacity: 0.5,
  },

  // Detail Modal
  detailModal: {
    maxHeight: '90%',
  },
  detailContent: {
    flex: 1,
  },
  detailSection: {
    marginBottom: 20,
  },
  detailSectionTitle: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  detailText: {
    color: '#d1d5db',
    fontSize: 14,
    lineHeight: 20,
  },
  statusBadgeLarge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 16,
    marginBottom: 8,
  },
  statusBadgeText: {
    fontSize: 13,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  trendItem: {
    flexDirection: 'row',
    marginBottom: 4,
  },
  trendLabel: {
    color: '#6b7280',
    fontSize: 13,
    width: 60,
  },
  trendValue: {
    color: '#d1d5db',
    fontSize: 13,
    flex: 1,
  },
  observationItem: {
    flexDirection: 'row',
    marginBottom: 4,
  },
  observationBullet: {
    color: '#6366f1',
    fontSize: 14,
    marginRight: 8,
  },
  observationText: {
    color: '#d1d5db',
    fontSize: 13,
    flex: 1,
    lineHeight: 18,
  },
  recommendationItem: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
  },
  recommendationAction: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '500',
    marginBottom: 4,
  },
  recommendationMeta: {
    color: '#6b7280',
    fontSize: 11,
  },
  watchItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
  },
  watchText: {
    color: '#d1d5db',
    fontSize: 13,
    flex: 1,
  },
  loadingDetail: {
    alignItems: 'center',
    padding: 32,
  },
  loadingDetailText: {
    color: '#6b7280',
    fontSize: 14,
    marginTop: 12,
  },
});
