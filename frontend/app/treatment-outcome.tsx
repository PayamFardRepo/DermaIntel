/**
 * Treatment Outcome Prediction Screen
 *
 * Features:
 * - AI-powered treatment outcome prediction with before/after images
 * - Treatment tracking and dose logging
 * - Effectiveness assessments over time
 * - Progressive timeline visualization
 * - Confidence intervals (best/typical/worst case scenarios)
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Platform,
  RefreshControl,
  Image,
  Modal,
  Dimensions,
  TextInput,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as ImagePicker from 'expo-image-picker';
import { API_BASE_URL } from '../config';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Treatment types configuration
const TREATMENT_TYPES = [
  { id: 'topical-steroid', label: 'Topical Steroid', icon: 'bandage-outline', improvement: 70 },
  { id: 'laser-therapy', label: 'Laser Therapy', icon: 'flash-outline', improvement: 85 },
  { id: 'cryotherapy', label: 'Cryotherapy', icon: 'snow-outline', improvement: 90 },
  { id: 'prescription-cream', label: 'Prescription Cream', icon: 'medical-outline', improvement: 75 },
  { id: 'oral-isotretinoin', label: 'Oral Isotretinoin', icon: 'fitness-outline', improvement: 85 },
  { id: 'phototherapy', label: 'Phototherapy', icon: 'sunny-outline', improvement: 80 },
];

const TIMEFRAMES = [
  { id: '6months', label: '6 Months', weeks: 24 },
  { id: '1year', label: '1 Year', weeks: 52 },
  { id: '2years', label: '2 Years', weeks: 104 },
];

interface Treatment {
  id: number;
  treatment_name: string;
  treatment_type: string;
  active_ingredient: string | null;
  brand_name: string | null;
  dosage: string | null;
  frequency: string | null;
  start_date: string;
  planned_end_date: string | null;
  is_active: boolean;
  target_condition: string | null;
  log_count: number;
  latest_effectiveness: {
    improvement_percentage: number;
    overall_effectiveness: string;
    assessment_date: string;
  } | null;
}

interface PredictionResult {
  treatmentId: string;
  projectedImprovement: number;
  baseImprovement: number;
  beforeImage: string;
  afterImage: string;
  timeline: {
    weeks: number;
    improvement: number;
    image_url: string;
    description: string;
  }[];
  timeframe: string;
  recommendations: string[];
  severity: string;
  confidenceIntervals: {
    best_case: number;
    typical: number;
    worst_case: number;
  };
  metadata: any;
  disclaimer: string;
}

interface TreatmentLog {
  id: number;
  administered_date: string;
  dose_amount: number | null;
  taken_as_prescribed: boolean;
  missed_dose: boolean;
  notes: string | null;
}

export default function TreatmentOutcomeScreen() {
  const { t } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // State
  const [activeTab, setActiveTab] = useState<'predict' | 'treatments' | 'history'>('predict');
  const [isLoading, setIsLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // Prediction state
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedTreatment, setSelectedTreatment] = useState<string>('topical-steroid');
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('6months');
  const [diagnosis, setDiagnosis] = useState<string>('');
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);

  // Treatments state
  const [treatments, setTreatments] = useState<Treatment[]>([]);
  const [selectedTreatmentDetail, setSelectedTreatmentDetail] = useState<Treatment | null>(null);
  const [treatmentLogs, setTreatmentLogs] = useState<TreatmentLog[]>([]);

  // Modal state
  const [resultModalVisible, setResultModalVisible] = useState(false);
  const [timelineModalVisible, setTimelineModalVisible] = useState(false);
  const [addTreatmentModalVisible, setAddTreatmentModalVisible] = useState(false);
  const [logDoseModalVisible, setLogDoseModalVisible] = useState(false);

  // New treatment form state
  const [newTreatmentName, setNewTreatmentName] = useState('');
  const [newTreatmentType, setNewTreatmentType] = useState('topical-steroid');
  const [newTargetCondition, setNewTargetCondition] = useState('');
  const [newDosage, setNewDosage] = useState('');
  const [newFrequency, setNewFrequency] = useState('');

  const getAuthHeaders = async () => {
    const token = await AsyncStorage.getItem('accessToken');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  };

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
      return;
    }
    loadTreatments();
  }, [isAuthenticated]);

  const loadTreatments = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/treatments`, { headers });

      if (response.ok) {
        const data = await response.json();
        setTreatments(data || []);
      }
    } catch (error) {
      console.error('Error loading treatments:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const selectImage = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Required', 'Please grant camera roll permissions');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        setSelectedImage(result.assets[0].uri);
        setPredictionResult(null);
      }
    } catch (error) {
      console.error('Error selecting image:', error);
      Alert.alert('Error', 'Failed to select image');
    }
  };

  const takePhoto = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Required', 'Please grant camera permissions');
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        setSelectedImage(result.assets[0].uri);
        setPredictionResult(null);
      }
    } catch (error) {
      console.error('Error taking photo:', error);
      Alert.alert('Error', 'Failed to take photo');
    }
  };

  const predictOutcome = async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select or take a photo of your skin condition');
      return;
    }

    if (!diagnosis.trim()) {
      Alert.alert('Missing Diagnosis', 'Please enter a diagnosis for more accurate predictions');
      return;
    }

    try {
      setIsPredicting(true);

      const token = await AsyncStorage.getItem('accessToken');
      const formData = new FormData();

      const filename = selectedImage.split('/').pop() || 'image.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : 'image/jpeg';

      formData.append('image', {
        uri: selectedImage,
        name: filename,
        type,
      } as any);
      formData.append('treatment_type', selectedTreatment);
      formData.append('timeframe', selectedTimeframe);
      formData.append('diagnosis', diagnosis);

      const response = await fetch(`${API_BASE_URL}/predict-treatment-outcome`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setPredictionResult(data);
        setResultModalVisible(true);
      } else {
        throw new Error('Prediction failed');
      }
    } catch (error) {
      console.error('Error predicting outcome:', error);
      Alert.alert('Error', 'Failed to predict treatment outcome');
    } finally {
      setIsPredicting(false);
    }
  };

  const createTreatment = async () => {
    if (!newTreatmentName.trim()) {
      Alert.alert('Missing Name', 'Please enter a treatment name');
      return;
    }

    try {
      setIsLoading(true);
      const token = await AsyncStorage.getItem('accessToken');
      const formData = new FormData();

      formData.append('treatment_name', newTreatmentName);
      formData.append('treatment_type', newTreatmentType);
      formData.append('start_date', new Date().toISOString());
      if (newTargetCondition) formData.append('target_condition', newTargetCondition);
      if (newDosage) formData.append('dosage', newDosage);
      if (newFrequency) formData.append('frequency', newFrequency);

      const response = await fetch(`${API_BASE_URL}/treatments`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        Alert.alert('Success', 'Treatment added successfully');
        setAddTreatmentModalVisible(false);
        resetNewTreatmentForm();
        loadTreatments();
      } else {
        throw new Error('Failed to create treatment');
      }
    } catch (error) {
      console.error('Error creating treatment:', error);
      Alert.alert('Error', 'Failed to add treatment');
    } finally {
      setIsLoading(false);
    }
  };

  const resetNewTreatmentForm = () => {
    setNewTreatmentName('');
    setNewTreatmentType('topical-steroid');
    setNewTargetCondition('');
    setNewDosage('');
    setNewFrequency('');
  };

  const loadTreatmentLogs = async (treatmentId: number) => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/treatment-logs/${treatmentId}`, { headers });

      if (response.ok) {
        const data = await response.json();
        setTreatmentLogs(data || []);
      }
    } catch (error) {
      console.error('Error loading treatment logs:', error);
    }
  };

  const logDose = async (treatmentId: number) => {
    try {
      setIsLoading(true);
      const token = await AsyncStorage.getItem('accessToken');
      const formData = new FormData();

      formData.append('treatment_id', treatmentId.toString());
      formData.append('administered_date', new Date().toISOString());
      formData.append('taken_as_prescribed', 'true');

      const response = await fetch(`${API_BASE_URL}/treatment-logs`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        Alert.alert('Success', 'Dose logged successfully');
        loadTreatments();
        if (selectedTreatmentDetail) {
          loadTreatmentLogs(selectedTreatmentDetail.id);
        }
      } else {
        throw new Error('Failed to log dose');
      }
    } catch (error) {
      console.error('Error logging dose:', error);
      Alert.alert('Error', 'Failed to log dose');
    } finally {
      setIsLoading(false);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadTreatments();
  }, []);

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'severe': return '#ef4444';
      case 'moderate': return '#f59e0b';
      case 'mild': return '#10b981';
      default: return '#6b7280';
    }
  };

  const getEffectivenessColor = (effectiveness: string) => {
    switch (effectiveness?.toLowerCase()) {
      case 'excellent': return '#10b981';
      case 'good': return '#22c55e';
      case 'moderate': return '#f59e0b';
      case 'poor': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  // Render Predict Tab
  const renderPredictTab = () => (
    <ScrollView style={styles.tabContent}>
      {/* Image Selection */}
      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Upload Condition Image</Text>
        <Text style={styles.sectionDescription}>
          Take a clear photo of the affected area for accurate prediction
        </Text>

        {selectedImage ? (
          <View style={styles.imagePreviewContainer}>
            <Image source={{ uri: selectedImage }} style={styles.imagePreview} />
            <TouchableOpacity
              style={styles.removeImageButton}
              onPress={() => setSelectedImage(null)}
            >
              <Ionicons name="close-circle" size={28} color="#ef4444" />
            </TouchableOpacity>
          </View>
        ) : (
          <View style={styles.imageSelectionButtons}>
            <TouchableOpacity style={styles.imageButton} onPress={takePhoto}>
              <Ionicons name="camera-outline" size={32} color="#2563eb" />
              <Text style={styles.imageButtonText}>Take Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.imageButton} onPress={selectImage}>
              <Ionicons name="images-outline" size={32} color="#2563eb" />
              <Text style={styles.imageButtonText}>Choose Photo</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>

      {/* Diagnosis Input */}
      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Diagnosis</Text>
        <TextInput
          style={styles.textInput}
          placeholder="e.g., Eczema, Psoriasis, Acne..."
          value={diagnosis}
          onChangeText={setDiagnosis}
          placeholderTextColor="#9ca3af"
        />
      </View>

      {/* Treatment Selection */}
      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Treatment Type</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <View style={styles.treatmentOptions}>
            {TREATMENT_TYPES.map(treatment => (
              <TouchableOpacity
                key={treatment.id}
                style={[
                  styles.treatmentOption,
                  selectedTreatment === treatment.id && styles.treatmentOptionSelected,
                ]}
                onPress={() => setSelectedTreatment(treatment.id)}
              >
                <Ionicons
                  name={treatment.icon as any}
                  size={24}
                  color={selectedTreatment === treatment.id ? '#fff' : '#2563eb'}
                />
                <Text style={[
                  styles.treatmentOptionText,
                  selectedTreatment === treatment.id && styles.treatmentOptionTextSelected,
                ]}>
                  {treatment.label}
                </Text>
                <Text style={[
                  styles.treatmentImprovement,
                  selectedTreatment === treatment.id && styles.treatmentImprovementSelected,
                ]}>
                  ~{treatment.improvement}%
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </ScrollView>
      </View>

      {/* Timeframe Selection */}
      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Prediction Timeframe</Text>
        <View style={styles.timeframeOptions}>
          {TIMEFRAMES.map(timeframe => (
            <TouchableOpacity
              key={timeframe.id}
              style={[
                styles.timeframeOption,
                selectedTimeframe === timeframe.id && styles.timeframeOptionSelected,
              ]}
              onPress={() => setSelectedTimeframe(timeframe.id)}
            >
              <Text style={[
                styles.timeframeText,
                selectedTimeframe === timeframe.id && styles.timeframeTextSelected,
              ]}>
                {timeframe.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Predict Button */}
      <TouchableOpacity
        style={[styles.predictButton, !selectedImage && styles.predictButtonDisabled]}
        onPress={predictOutcome}
        disabled={isPredicting || !selectedImage}
      >
        {isPredicting ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="sparkles" size={24} color="#fff" />
            <Text style={styles.predictButtonText}>Predict Outcome</Text>
          </>
        )}
      </TouchableOpacity>

      {/* Disclaimer */}
      <View style={styles.disclaimerCard}>
        <Ionicons name="information-circle-outline" size={20} color="#6b7280" />
        <Text style={styles.disclaimerText}>
          Predictions are AI-generated simulations for educational purposes only.
          Actual results vary. Always consult a dermatologist.
        </Text>
      </View>

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Render Treatments Tab
  const renderTreatmentsTab = () => (
    <ScrollView
      style={styles.tabContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Add Treatment Button */}
      <TouchableOpacity
        style={styles.addTreatmentButton}
        onPress={() => setAddTreatmentModalVisible(true)}
      >
        <Ionicons name="add-circle-outline" size={24} color="#2563eb" />
        <Text style={styles.addTreatmentText}>Add New Treatment</Text>
      </TouchableOpacity>

      {/* Active Treatments */}
      <Text style={styles.listTitle}>Active Treatments</Text>

      {treatments.filter(t => t.is_active).length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="medical-outline" size={48} color="#9ca3af" />
          <Text style={styles.emptyTitle}>No Active Treatments</Text>
          <Text style={styles.emptyText}>Add a treatment to start tracking</Text>
        </View>
      ) : (
        treatments.filter(t => t.is_active).map(treatment => (
          <TouchableOpacity
            key={treatment.id}
            style={styles.treatmentCard}
            onPress={() => {
              setSelectedTreatmentDetail(treatment);
              loadTreatmentLogs(treatment.id);
            }}
          >
            <View style={styles.treatmentCardHeader}>
              <View style={styles.treatmentInfo}>
                <Text style={styles.treatmentName}>{treatment.treatment_name}</Text>
                <Text style={styles.treatmentType}>{treatment.treatment_type}</Text>
              </View>
              <View style={styles.treatmentStatus}>
                <View style={styles.activeBadge}>
                  <View style={styles.activeDot} />
                  <Text style={styles.activeText}>Active</Text>
                </View>
              </View>
            </View>

            {treatment.target_condition && (
              <Text style={styles.treatmentCondition}>
                For: {treatment.target_condition}
              </Text>
            )}

            <View style={styles.treatmentStats}>
              <View style={styles.treatmentStat}>
                <Ionicons name="calendar-outline" size={16} color="#6b7280" />
                <Text style={styles.treatmentStatText}>
                  Started {formatDate(treatment.start_date)}
                </Text>
              </View>
              <View style={styles.treatmentStat}>
                <Ionicons name="document-text-outline" size={16} color="#6b7280" />
                <Text style={styles.treatmentStatText}>
                  {treatment.log_count} doses logged
                </Text>
              </View>
            </View>

            {treatment.latest_effectiveness && (
              <View style={styles.effectivenessRow}>
                <Text style={styles.effectivenessLabel}>Latest Assessment:</Text>
                <View style={[
                  styles.effectivenessBadge,
                  { backgroundColor: `${getEffectivenessColor(treatment.latest_effectiveness.overall_effectiveness)}20` }
                ]}>
                  <Text style={[
                    styles.effectivenessText,
                    { color: getEffectivenessColor(treatment.latest_effectiveness.overall_effectiveness) }
                  ]}>
                    {treatment.latest_effectiveness.improvement_percentage}% improved
                  </Text>
                </View>
              </View>
            )}

            <TouchableOpacity
              style={styles.logDoseButton}
              onPress={(e) => {
                e.stopPropagation();
                logDose(treatment.id);
              }}
            >
              <Ionicons name="checkmark-circle-outline" size={20} color="#10b981" />
              <Text style={styles.logDoseText}>Log Dose</Text>
            </TouchableOpacity>
          </TouchableOpacity>
        ))
      )}

      {/* Past Treatments */}
      {treatments.filter(t => !t.is_active).length > 0 && (
        <>
          <Text style={styles.listTitle}>Past Treatments</Text>
          {treatments.filter(t => !t.is_active).map(treatment => (
            <View key={treatment.id} style={[styles.treatmentCard, styles.pastTreatmentCard]}>
              <Text style={styles.treatmentName}>{treatment.treatment_name}</Text>
              <Text style={styles.treatmentType}>{treatment.treatment_type}</Text>
              <Text style={styles.treatmentDates}>
                {formatDate(treatment.start_date)} - {treatment.planned_end_date ? formatDate(treatment.planned_end_date) : 'Ongoing'}
              </Text>
            </View>
          ))}
        </>
      )}

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Render History Tab (Prediction History)
  const renderHistoryTab = () => (
    <ScrollView style={styles.tabContent}>
      <View style={styles.infoCard}>
        <Ionicons name="time-outline" size={24} color="#2563eb" />
        <View style={styles.infoContent}>
          <Text style={styles.infoTitle}>Prediction History</Text>
          <Text style={styles.infoText}>
            Your past treatment outcome predictions will appear here. Start by making a prediction in the Predict tab.
          </Text>
        </View>
      </View>

      {/* Placeholder for history - in a real app, this would fetch from backend */}
      <View style={styles.emptyState}>
        <Ionicons name="analytics-outline" size={64} color="#9ca3af" />
        <Text style={styles.emptyTitle}>No Predictions Yet</Text>
        <Text style={styles.emptyText}>
          Your treatment outcome predictions will be saved here for future reference
        </Text>
        <TouchableOpacity
          style={styles.goToPredictButton}
          onPress={() => setActiveTab('predict')}
        >
          <Text style={styles.goToPredictText}>Make a Prediction</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Result Modal
  const renderResultModal = () => (
    <Modal
      visible={resultModalVisible}
      animationType="slide"
      onRequestClose={() => setResultModalVisible(false)}
    >
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setResultModalVisible(false)}>
            <Ionicons name="close" size={28} color="#1e3a5f" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Prediction Results</Text>
          <TouchableOpacity onPress={() => setTimelineModalVisible(true)}>
            <Ionicons name="time-outline" size={28} color="#2563eb" />
          </TouchableOpacity>
        </View>

        {predictionResult && (
          <ScrollView style={styles.resultContent}>
            {/* Before/After Comparison */}
            <View style={styles.comparisonCard}>
              <Text style={styles.comparisonTitle}>Before & After Prediction</Text>
              <View style={styles.imagesRow}>
                <View style={styles.imageContainer}>
                  <Image
                    source={{ uri: `${API_BASE_URL}${predictionResult.beforeImage}` }}
                    style={styles.comparisonImage}
                  />
                  <Text style={styles.imageLabel}>Before</Text>
                </View>
                <Ionicons name="arrow-forward" size={24} color="#2563eb" />
                <View style={styles.imageContainer}>
                  <Image
                    source={{ uri: `${API_BASE_URL}${predictionResult.afterImage}` }}
                    style={styles.comparisonImage}
                  />
                  <Text style={styles.imageLabel}>After ({selectedTimeframe === '6months' ? '6 mo' : selectedTimeframe === '1year' ? '1 yr' : '2 yr'})</Text>
                </View>
              </View>
            </View>

            {/* Improvement Stats */}
            <View style={styles.statsCard}>
              <View style={styles.mainStat}>
                <Text style={styles.mainStatValue}>{predictionResult.projectedImprovement}%</Text>
                <Text style={styles.mainStatLabel}>Projected Improvement</Text>
              </View>

              <View style={[
                styles.severityBadge,
                { backgroundColor: `${getSeverityColor(predictionResult.severity)}20` }
              ]}>
                <Text style={[styles.severityText, { color: getSeverityColor(predictionResult.severity) }]}>
                  {predictionResult.severity.charAt(0).toUpperCase() + predictionResult.severity.slice(1)} Severity
                </Text>
              </View>
            </View>

            {/* Confidence Intervals */}
            {predictionResult.confidenceIntervals && (
              <View style={styles.confidenceCard}>
                <Text style={styles.confidenceTitle}>Outcome Scenarios</Text>
                <View style={styles.scenariosRow}>
                  <View style={styles.scenarioItem}>
                    <Ionicons name="trending-down" size={20} color="#ef4444" />
                    <Text style={styles.scenarioValue}>{Math.round(predictionResult.confidenceIntervals.worst_case)}%</Text>
                    <Text style={styles.scenarioLabel}>Worst Case</Text>
                  </View>
                  <View style={[styles.scenarioItem, styles.scenarioItemMain]}>
                    <Ionicons name="remove" size={20} color="#2563eb" />
                    <Text style={[styles.scenarioValue, styles.scenarioValueMain]}>{Math.round(predictionResult.confidenceIntervals.typical)}%</Text>
                    <Text style={styles.scenarioLabel}>Typical</Text>
                  </View>
                  <View style={styles.scenarioItem}>
                    <Ionicons name="trending-up" size={20} color="#10b981" />
                    <Text style={styles.scenarioValue}>{Math.round(predictionResult.confidenceIntervals.best_case)}%</Text>
                    <Text style={styles.scenarioLabel}>Best Case</Text>
                  </View>
                </View>
              </View>
            )}

            {/* Recommendations */}
            <View style={styles.recommendationsCard}>
              <Text style={styles.recommendationsTitle}>Recommendations</Text>
              {predictionResult.recommendations.map((rec, index) => (
                <View key={index} style={styles.recommendationItem}>
                  <Ionicons
                    name={rec.includes('⚠️') ? 'warning' : rec.includes('✓') ? 'checkmark-circle' : rec.includes('💊') ? 'medical' : 'chevron-forward'}
                    size={16}
                    color={rec.includes('⚠️') ? '#f59e0b' : rec.includes('✓') ? '#10b981' : '#2563eb'}
                  />
                  <Text style={styles.recommendationText}>{rec.replace(/[⚠️✓💊]/g, '').trim()}</Text>
                </View>
              ))}
            </View>

            {/* View Timeline Button */}
            <TouchableOpacity
              style={styles.viewTimelineButton}
              onPress={() => setTimelineModalVisible(true)}
            >
              <Ionicons name="git-branch-outline" size={20} color="#2563eb" />
              <Text style={styles.viewTimelineText}>View Progressive Timeline</Text>
            </TouchableOpacity>

            {/* Disclaimer */}
            <View style={styles.resultDisclaimer}>
              <Text style={styles.resultDisclaimerText}>{predictionResult.disclaimer}</Text>
            </View>

            <View style={styles.bottomSpacer} />
          </ScrollView>
        )}
      </LinearGradient>
    </Modal>
  );

  // Timeline Modal
  const renderTimelineModal = () => (
    <Modal
      visible={timelineModalVisible}
      animationType="slide"
      onRequestClose={() => setTimelineModalVisible(false)}
    >
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setTimelineModalVisible(false)}>
            <Ionicons name="arrow-back" size={28} color="#1e3a5f" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Treatment Timeline</Text>
          <View style={{ width: 28 }} />
        </View>

        {predictionResult && (
          <ScrollView style={styles.timelineContent}>
            <Text style={styles.timelineSubtitle}>Progressive improvement over time</Text>

            {predictionResult.timeline.map((point, index) => (
              <View key={index} style={styles.timelineItem}>
                <View style={styles.timelineConnector}>
                  <View style={[
                    styles.timelineDot,
                    index === predictionResult.timeline.length - 1 && styles.timelineDotFinal
                  ]} />
                  {index < predictionResult.timeline.length - 1 && <View style={styles.timelineLine} />}
                </View>
                <View style={styles.timelineCard}>
                  <View style={styles.timelineHeader}>
                    <Text style={styles.timelineWeek}>Week {point.weeks}</Text>
                    <View style={styles.timelineImprovement}>
                      <Text style={styles.timelineImprovementText}>{point.improvement}%</Text>
                    </View>
                  </View>
                  <Image
                    source={{ uri: `${API_BASE_URL}${point.image_url}` }}
                    style={styles.timelineImage}
                  />
                  <Text style={styles.timelineDescription}>{point.description}</Text>
                </View>
              </View>
            ))}

            <View style={styles.bottomSpacer} />
          </ScrollView>
        )}
      </LinearGradient>
    </Modal>
  );

  // Add Treatment Modal
  const renderAddTreatmentModal = () => (
    <Modal
      visible={addTreatmentModalVisible}
      transparent
      animationType="slide"
      onRequestClose={() => setAddTreatmentModalVisible(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.formModalContent}>
          <View style={styles.formModalHeader}>
            <Text style={styles.formModalTitle}>Add Treatment</Text>
            <TouchableOpacity onPress={() => setAddTreatmentModalVisible(false)}>
              <Ionicons name="close" size={24} color="#6b7280" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.formContent}>
            <Text style={styles.formLabel}>Treatment Name *</Text>
            <TextInput
              style={styles.formInput}
              placeholder="e.g., Hydrocortisone Cream"
              value={newTreatmentName}
              onChangeText={setNewTreatmentName}
              placeholderTextColor="#9ca3af"
            />

            <Text style={styles.formLabel}>Treatment Type</Text>
            <View style={styles.typeSelector}>
              {TREATMENT_TYPES.slice(0, 4).map(type => (
                <TouchableOpacity
                  key={type.id}
                  style={[
                    styles.typeOption,
                    newTreatmentType === type.id && styles.typeOptionSelected,
                  ]}
                  onPress={() => setNewTreatmentType(type.id)}
                >
                  <Ionicons
                    name={type.icon as any}
                    size={20}
                    color={newTreatmentType === type.id ? '#fff' : '#6b7280'}
                  />
                  <Text style={[
                    styles.typeOptionText,
                    newTreatmentType === type.id && styles.typeOptionTextSelected,
                  ]}>
                    {type.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.formLabel}>Target Condition</Text>
            <TextInput
              style={styles.formInput}
              placeholder="e.g., Eczema"
              value={newTargetCondition}
              onChangeText={setNewTargetCondition}
              placeholderTextColor="#9ca3af"
            />

            <Text style={styles.formLabel}>Dosage</Text>
            <TextInput
              style={styles.formInput}
              placeholder="e.g., 1% cream"
              value={newDosage}
              onChangeText={setNewDosage}
              placeholderTextColor="#9ca3af"
            />

            <Text style={styles.formLabel}>Frequency</Text>
            <TextInput
              style={styles.formInput}
              placeholder="e.g., Twice daily"
              value={newFrequency}
              onChangeText={setNewFrequency}
              placeholderTextColor="#9ca3af"
            />

            <TouchableOpacity
              style={styles.submitButton}
              onPress={createTreatment}
              disabled={isLoading}
            >
              {isLoading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.submitButtonText}>Add Treatment</Text>
              )}
            </TouchableOpacity>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );

  return (
    <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#2563eb" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Treatment Prediction</Text>
        <View style={{ width: 40 }} />
      </View>

      {/* Tabs */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'predict' && styles.activeTab]}
          onPress={() => setActiveTab('predict')}
        >
          <Ionicons
            name="sparkles-outline"
            size={18}
            color={activeTab === 'predict' ? '#2563eb' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'predict' && styles.activeTabText]}>
            Predict
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'treatments' && styles.activeTab]}
          onPress={() => setActiveTab('treatments')}
        >
          <Ionicons
            name="medical-outline"
            size={18}
            color={activeTab === 'treatments' ? '#2563eb' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'treatments' && styles.activeTabText]}>
            Treatments
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'history' && styles.activeTab]}
          onPress={() => setActiveTab('history')}
        >
          <Ionicons
            name="time-outline"
            size={18}
            color={activeTab === 'history' ? '#2563eb' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'history' && styles.activeTabText]}>
            History
          </Text>
        </TouchableOpacity>
      </View>

      {/* Tab Content */}
      {activeTab === 'predict' && renderPredictTab()}
      {activeTab === 'treatments' && renderTreatmentsTab()}
      {activeTab === 'history' && renderHistoryTab()}

      {/* Modals */}
      {renderResultModal()}
      {renderTimelineModal()}
      {renderAddTreatmentModal()}
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
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    gap: 6,
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: '#2563eb',
  },
  tabText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
  },
  activeTabText: {
    color: '#2563eb',
  },
  tabContent: {
    flex: 1,
    padding: 16,
  },
  sectionCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 8,
  },
  sectionDescription: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 16,
  },
  imageSelectionButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  imageButton: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    padding: 24,
    borderWidth: 2,
    borderColor: '#bfdbfe',
    borderStyle: 'dashed',
  },
  imageButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2563eb',
    marginTop: 8,
  },
  imagePreviewContainer: {
    position: 'relative',
  },
  imagePreview: {
    width: '100%',
    height: 200,
    borderRadius: 12,
    resizeMode: 'cover',
  },
  removeImageButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: '#fff',
    borderRadius: 14,
  },
  textInput: {
    backgroundColor: '#f9fafb',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 10,
    padding: 14,
    fontSize: 15,
    color: '#1e3a5f',
  },
  treatmentOptions: {
    flexDirection: 'row',
    gap: 10,
    paddingVertical: 4,
  },
  treatmentOption: {
    alignItems: 'center',
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    padding: 16,
    minWidth: 110,
    borderWidth: 2,
    borderColor: '#bfdbfe',
  },
  treatmentOptionSelected: {
    backgroundColor: '#2563eb',
    borderColor: '#2563eb',
  },
  treatmentOptionText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#1e3a5f',
    marginTop: 8,
    textAlign: 'center',
  },
  treatmentOptionTextSelected: {
    color: '#fff',
  },
  treatmentImprovement: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 4,
  },
  treatmentImprovementSelected: {
    color: '#bfdbfe',
  },
  timeframeOptions: {
    flexDirection: 'row',
    gap: 10,
  },
  timeframeOption: {
    flex: 1,
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
    borderRadius: 10,
    paddingVertical: 14,
  },
  timeframeOptionSelected: {
    backgroundColor: '#2563eb',
  },
  timeframeText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
  },
  timeframeTextSelected: {
    color: '#fff',
  },
  predictButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 18,
    gap: 10,
  },
  predictButtonDisabled: {
    backgroundColor: '#9ca3af',
  },
  predictButtonText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '700',
  },
  disclaimerCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#f9fafb',
    borderRadius: 10,
    padding: 14,
    marginTop: 16,
    gap: 10,
  },
  disclaimerText: {
    flex: 1,
    fontSize: 12,
    color: '#6b7280',
    lineHeight: 18,
  },
  addTreatmentButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    gap: 10,
  },
  addTreatmentText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2563eb',
  },
  listTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 40,
    backgroundColor: '#fff',
    borderRadius: 16,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  treatmentCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
  },
  pastTreatmentCard: {
    opacity: 0.7,
  },
  treatmentCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  treatmentInfo: {
    flex: 1,
  },
  treatmentName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  treatmentType: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  treatmentStatus: {},
  activeBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ecfdf5',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 6,
  },
  activeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#10b981',
  },
  activeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#10b981',
  },
  treatmentCondition: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 8,
    fontStyle: 'italic',
  },
  treatmentStats: {
    flexDirection: 'row',
    marginTop: 12,
    gap: 16,
  },
  treatmentStat: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  treatmentStatText: {
    fontSize: 12,
    color: '#6b7280',
  },
  effectivenessRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    gap: 8,
  },
  effectivenessLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
  effectivenessBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  effectivenessText: {
    fontSize: 12,
    fontWeight: '600',
  },
  logDoseButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ecfdf5',
    borderRadius: 8,
    padding: 10,
    marginTop: 12,
    gap: 6,
  },
  logDoseText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#10b981',
  },
  treatmentDates: {
    fontSize: 12,
    color: '#9ca3af',
    marginTop: 4,
  },
  infoCard: {
    flexDirection: 'row',
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    gap: 12,
  },
  infoContent: {
    flex: 1,
  },
  infoTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  infoText: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 4,
    lineHeight: 20,
  },
  goToPredictButton: {
    backgroundColor: '#2563eb',
    borderRadius: 8,
    paddingHorizontal: 20,
    paddingVertical: 12,
    marginTop: 16,
  },
  goToPredictText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  modalContainer: {
    flex: 1,
  },
  modalHeader: {
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
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  resultContent: {
    flex: 1,
    padding: 16,
  },
  comparisonCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  comparisonTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 16,
    textAlign: 'center',
  },
  imagesRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  imageContainer: {
    alignItems: 'center',
  },
  comparisonImage: {
    width: (SCREEN_WIDTH - 100) / 2,
    height: (SCREEN_WIDTH - 100) / 2,
    borderRadius: 12,
    resizeMode: 'cover',
  },
  imageLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
    marginTop: 8,
  },
  statsCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    alignItems: 'center',
  },
  mainStat: {
    alignItems: 'center',
    marginBottom: 16,
  },
  mainStatValue: {
    fontSize: 48,
    fontWeight: '700',
    color: '#10b981',
  },
  mainStatLabel: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 4,
  },
  severityBadge: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  severityText: {
    fontSize: 14,
    fontWeight: '600',
  },
  confidenceCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  confidenceTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 16,
    textAlign: 'center',
  },
  scenariosRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  scenarioItem: {
    alignItems: 'center',
  },
  scenarioItemMain: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
  },
  scenarioValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1e3a5f',
    marginTop: 8,
  },
  scenarioValueMain: {
    fontSize: 28,
    color: '#2563eb',
  },
  scenarioLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  recommendationsCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  recommendationsTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 16,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
    gap: 10,
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
  },
  viewTimelineButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    gap: 10,
    marginBottom: 16,
  },
  viewTimelineText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#2563eb',
  },
  resultDisclaimer: {
    backgroundColor: '#fef3c7',
    borderRadius: 12,
    padding: 16,
  },
  resultDisclaimerText: {
    fontSize: 12,
    color: '#92400e',
    lineHeight: 18,
  },
  timelineContent: {
    flex: 1,
    padding: 16,
  },
  timelineSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 24,
  },
  timelineItem: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  timelineConnector: {
    alignItems: 'center',
    width: 24,
    marginRight: 12,
  },
  timelineDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#2563eb',
  },
  timelineDotFinal: {
    backgroundColor: '#10b981',
  },
  timelineLine: {
    flex: 1,
    width: 2,
    backgroundColor: '#e5e7eb',
  },
  timelineCard: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  timelineHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  timelineWeek: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  timelineImprovement: {
    backgroundColor: '#ecfdf5',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  timelineImprovementText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#10b981',
  },
  timelineImage: {
    width: '100%',
    height: 180,
    borderRadius: 10,
    resizeMode: 'cover',
    marginBottom: 12,
  },
  timelineDescription: {
    fontSize: 13,
    color: '#6b7280',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  formModalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '80%',
  },
  formModalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  formModalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  formContent: {
    padding: 20,
  },
  formLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
    marginTop: 16,
  },
  formInput: {
    backgroundColor: '#f9fafb',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 10,
    padding: 14,
    fontSize: 15,
    color: '#1e3a5f',
  },
  typeSelector: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  typeOption: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 6,
  },
  typeOptionSelected: {
    backgroundColor: '#2563eb',
  },
  typeOptionText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
  },
  typeOptionTextSelected: {
    color: '#fff',
  },
  submitButton: {
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 24,
    marginBottom: Platform.OS === 'ios' ? 40 : 20,
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  bottomSpacer: {
    height: 30,
  },
});
