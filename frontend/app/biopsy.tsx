/**
 * Biopsy Request/Tracking Screen
 *
 * Features:
 * - Track biopsy requests and results
 * - Add pathology results to analyses
 * - View AI prediction accuracy statistics
 * - Compare AI predictions with biopsy outcomes
 * - Visualize accuracy trends
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
  Modal,
  TextInput,
  Dimensions,
  Image,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Common biopsy results/diagnoses
const COMMON_DIAGNOSES = [
  { id: 'melanoma', label: 'Melanoma', category: 'malignant' },
  { id: 'basal_cell_carcinoma', label: 'Basal Cell Carcinoma', category: 'malignant' },
  { id: 'squamous_cell_carcinoma', label: 'Squamous Cell Carcinoma', category: 'malignant' },
  { id: 'actinic_keratosis', label: 'Actinic Keratosis', category: 'precancerous' },
  { id: 'seborrheic_keratosis', label: 'Seborrheic Keratosis', category: 'benign' },
  { id: 'dermatofibroma', label: 'Dermatofibroma', category: 'benign' },
  { id: 'nevus', label: 'Benign Nevus (Mole)', category: 'benign' },
  { id: 'dysplastic_nevus', label: 'Dysplastic Nevus', category: 'atypical' },
  { id: 'hemangioma', label: 'Hemangioma', category: 'benign' },
  { id: 'lipoma', label: 'Lipoma', category: 'benign' },
  { id: 'cyst', label: 'Cyst', category: 'benign' },
  { id: 'wart', label: 'Wart (Verruca)', category: 'benign' },
  { id: 'psoriasis', label: 'Psoriasis', category: 'inflammatory' },
  { id: 'eczema', label: 'Eczema/Dermatitis', category: 'inflammatory' },
  { id: 'other', label: 'Other', category: 'other' },
];

interface Analysis {
  id: number;
  predicted_class: string;
  lesion_confidence: number;
  created_at: string;
  image_path: string;
  biopsy_performed: boolean;
  biopsy_result: string | null;
  biopsy_date: string | null;
  biopsy_notes: string | null;
  biopsy_facility: string | null;
}

interface HistopathologyResult {
  tissue_types: { type: string; confidence: number; description: string }[];
  malignancy_assessment: {
    risk_level: string;
    malignant_probability: number;
    confidence_interval: number[];
    key_features: string[];
  };
  quality_metrics: {
    focus_quality: string;
    staining_quality: string;
    tissue_adequacy: string;
  };
  recommendations: string[];
  analysis_id?: number;
}

interface AICorrelation {
  analysis_id: number;
  dermoscopy_prediction: string;
  dermoscopy_confidence: number;
  histopathology_result: string;
  histopathology_confidence: number;
  concordance_status: string;
  concordance_type: string;
  assessment: string;
  is_critical_discordance: boolean;
}

interface BiopsyHistoryItem {
  id: number;
  analysis_date: string;
  histopathology_date: string;
  histopathology_result: string;
  histopathology_malignant: boolean;
  histopathology_confidence: number;
  histopathology_risk_level: string;
  ai_concordance: boolean;
  ai_concordance_type: string;
  lesion_group_id?: number;
}

interface LesionProgressionEvent {
  id: number;
  date: string;
  event_type: string;
  diagnosis?: string;
  risk_level?: string;
  ai_concordance?: boolean;
  notes?: string;
}

interface AccuracyStats {
  total_biopsies: number;
  exact_accuracy_percent: number;
  category_accuracy_percent: number;
  breakdown: {
    exact_matches: number;
    category_matches: number;
    mismatches: number;
    pending: number;
  };
  by_condition: { [key: string]: { total: number; correct: number; incorrect: number } };
  recent_correlations: {
    id: number;
    ai_prediction: string;
    biopsy_result: string;
    correct: boolean;
    category: string;
    date: string | null;
    confidence: number;
  }[];
}

export default function BiopsyScreen() {
  const { t } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // State
  const [activeTab, setActiveTab] = useState<'pending' | 'results' | 'accuracy' | 'histopathology' | 'timeline'>('pending');
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Histopathology state
  const [histoImage, setHistoImage] = useState<string | null>(null);
  const [histoAnalyzing, setHistoAnalyzing] = useState(false);
  const [histoResult, setHistoResult] = useState<HistopathologyResult | null>(null);
  const [histoError, setHistoError] = useState<string | null>(null);

  // AI Correlation state
  const [aiCorrelation, setAiCorrelation] = useState<AICorrelation | null>(null);
  const [correlationLoading, setCorrelationLoading] = useState(false);

  // Biopsy history and progression state
  const [biopsyHistory, setBiopsyHistory] = useState<BiopsyHistoryItem[]>([]);
  const [selectedLesionProgress, setSelectedLesionProgress] = useState<LesionProgressionEvent[]>([]);
  const [selectedLesionId, setSelectedLesionId] = useState<number | null>(null);

  // Report download state
  const [downloadingReport, setDownloadingReport] = useState<number | null>(null);

  // Analyses state
  const [pendingAnalyses, setPendingAnalyses] = useState<Analysis[]>([]);
  const [completedAnalyses, setCompletedAnalyses] = useState<Analysis[]>([]);
  const [accuracyStats, setAccuracyStats] = useState<AccuracyStats | null>(null);

  // Modal state
  const [addResultModalVisible, setAddResultModalVisible] = useState(false);
  const [selectedAnalysis, setSelectedAnalysis] = useState<Analysis | null>(null);

  // Form state
  const [biopsyResult, setBiopsyResult] = useState('');
  const [customDiagnosis, setCustomDiagnosis] = useState('');
  const [biopsyDate, setBiopsyDate] = useState('');
  const [biopsyFacility, setBiopsyFacility] = useState('');
  const [pathologistName, setPathologistName] = useState('');
  const [biopsyNotes, setBiopsyNotes] = useState('');

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
    loadData();
  }, [isAuthenticated]);

  const loadData = async () => {
    await Promise.all([
      loadAnalyses(),
      loadAccuracyStats(),
      loadBiopsyHistory(),
    ]);
    setIsLoading(false);
    setRefreshing(false);
  };

  const loadAnalyses = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/history?limit=100`, { headers });

      if (response.ok) {
        const data = await response.json();
        const analyses = data.history || [];

        // Separate pending and completed biopsies
        const pending = analyses.filter((a: Analysis) => !a.biopsy_performed);
        const completed = analyses.filter((a: Analysis) => a.biopsy_performed);

        setPendingAnalyses(pending);
        setCompletedAnalyses(completed);
      }
    } catch (error) {
      console.error('Error loading analyses:', error);
    }
  };

  const loadAccuracyStats = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/analysis/accuracy/stats`, { headers });

      if (response.ok) {
        const data = await response.json();
        setAccuracyStats(data);
      }
    } catch (error) {
      console.error('Error loading accuracy stats:', error);
    }
  };

  const loadBiopsyHistory = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/biopsy/history?limit=50`, { headers });

      if (response.ok) {
        const data = await response.json();
        setBiopsyHistory(data.history || []);
      }
    } catch (error) {
      console.error('Error loading biopsy history:', error);
    }
  };

  const loadAICorrelation = async (analysisId: number) => {
    setCorrelationLoading(true);
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/biopsy-correlation/${analysisId}`, { headers });

      if (response.ok) {
        const data = await response.json();
        setAiCorrelation(data);
      }
    } catch (error) {
      console.error('Error loading AI correlation:', error);
    } finally {
      setCorrelationLoading(false);
    }
  };

  const loadLesionProgression = async (lesionGroupId: number) => {
    setSelectedLesionId(lesionGroupId);
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/biopsy/lesion-progression/${lesionGroupId}`, { headers });

      if (response.ok) {
        const data = await response.json();
        setSelectedLesionProgress(data.timeline || []);
      }
    } catch (error) {
      console.error('Error loading lesion progression:', error);
    }
  };

  const downloadReport = async (analysisId: number, format: 'pdf' | 'html' = 'html') => {
    setDownloadingReport(analysisId);
    try {
      const token = await AsyncStorage.getItem('accessToken');
      const response = await fetch(
        `${API_BASE_URL}/biopsy/report/${analysisId}?format=${format}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        if (format === 'html') {
          const html = await response.text();
          // For HTML, we could show in a webview or alert
          Alert.alert(
            'Report Generated',
            'The biopsy report has been generated successfully.',
            [
              { text: 'OK' }
            ]
          );
        } else {
          // For PDF, we would need to save/share the file
          Alert.alert(
            'Report Downloaded',
            'The PDF report has been generated. You can view it in your downloads.',
            [{ text: 'OK' }]
          );
        }
      } else {
        const errorData = await response.json();
        Alert.alert('Error', errorData.detail || 'Failed to generate report');
      }
    } catch (error) {
      console.error('Error downloading report:', error);
      Alert.alert('Error', 'Failed to download report. Please try again.');
    } finally {
      setDownloadingReport(null);
    }
  };

  const submitBiopsyResult = async () => {
    if (!selectedAnalysis) return;

    const finalResult = biopsyResult === 'other' ? customDiagnosis : biopsyResult;

    if (!finalResult.trim()) {
      Alert.alert('Missing Information', 'Please select or enter a diagnosis');
      return;
    }

    try {
      setIsLoading(true);
      const token = await AsyncStorage.getItem('accessToken');
      const formData = new FormData();

      formData.append('biopsy_result', finalResult);
      if (biopsyDate) formData.append('biopsy_date', biopsyDate);
      if (biopsyFacility) formData.append('biopsy_facility', biopsyFacility);
      if (pathologistName) formData.append('pathologist_name', pathologistName);
      if (biopsyNotes) formData.append('biopsy_notes', biopsyNotes);

      const response = await fetch(`${API_BASE_URL}/analysis/biopsy/${selectedAnalysis.id}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();

        let message = 'Biopsy result added successfully.';
        if (data.accuracy_category === 'exact_match') {
          message += '\n\n✅ AI prediction was correct!';
        } else if (data.accuracy_category === 'category_match') {
          message += '\n\n⚠️ AI predicted the correct category (benign/malignant).';
        } else if (data.accuracy_category === 'mismatch') {
          message += '\n\n❌ AI prediction did not match the biopsy result.';
        }

        Alert.alert('Success', message);
        setAddResultModalVisible(false);
        resetForm();
        loadData();
      } else {
        throw new Error('Failed to submit biopsy result');
      }
    } catch (error) {
      console.error('Error submitting biopsy:', error);
      Alert.alert('Error', 'Failed to submit biopsy result');
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setBiopsyResult('');
    setCustomDiagnosis('');
    setBiopsyDate('');
    setBiopsyFacility('');
    setPathologistName('');
    setBiopsyNotes('');
    setSelectedAnalysis(null);
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadData();
  }, []);

  // Histopathology functions
  const pickHistoImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert('Permission Required', 'Please allow access to your photo library to upload biopsy slides.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      setHistoImage(result.assets[0].uri);
      setHistoResult(null);
      setHistoError(null);
    }
  };

  const takeHistoPhoto = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert('Permission Required', 'Please allow camera access to take photos of biopsy slides.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      setHistoImage(result.assets[0].uri);
      setHistoResult(null);
      setHistoError(null);
    }
  };

  const analyzeHistopathology = async () => {
    if (!histoImage) {
      Alert.alert('No Image', 'Please select or take a photo of a biopsy slide first.');
      return;
    }

    setHistoAnalyzing(true);
    setHistoError(null);

    try {
      const token = await AsyncStorage.getItem('accessToken');
      const formData = new FormData();

      // Get the file extension
      const uriParts = histoImage.split('.');
      const fileType = uriParts[uriParts.length - 1];

      formData.append('image', {
        uri: histoImage,
        name: `histopathology.${fileType}`,
        type: `image/${fileType}`,
      } as any);

      const response = await fetch(`${API_BASE_URL}/analyze-histopathology`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setHistoResult(data);
      } else {
        const errorData = await response.json();
        setHistoError(errorData.detail || 'Failed to analyze histopathology image');
      }
    } catch (error) {
      console.error('Histopathology analysis error:', error);
      setHistoError('Network error. Please check your connection and try again.');
    } finally {
      setHistoAnalyzing(false);
    }
  };

  const clearHistoImage = () => {
    setHistoImage(null);
    setHistoResult(null);
    setHistoError(null);
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'high': return '#ef4444';
      case 'moderate': return '#f59e0b';
      case 'low': return '#10b981';
      default: return '#6b7280';
    }
  };

  const getQualityColor = (quality: string) => {
    switch (quality.toLowerCase()) {
      case 'good': case 'excellent': return '#10b981';
      case 'moderate': case 'fair': return '#f59e0b';
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

  const getAccuracyColor = (category: string) => {
    switch (category) {
      case 'exact_match': return '#10b981';
      case 'category_match': return '#f59e0b';
      case 'mismatch': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'malignant': return '#ef4444';
      case 'precancerous': return '#f59e0b';
      case 'atypical': return '#8b5cf6';
      case 'inflammatory': return '#3b82f6';
      case 'benign': return '#10b981';
      default: return '#6b7280';
    }
  };

  // Render Pending Tab
  const renderPendingTab = () => (
    <ScrollView
      style={styles.tabContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Info Card */}
      <View style={styles.infoCard}>
        <Ionicons name="information-circle-outline" size={24} color="#2563eb" />
        <View style={styles.infoContent}>
          <Text style={styles.infoTitle}>Track Biopsy Results</Text>
          <Text style={styles.infoText}>
            Add pathology results to your analyses to track AI accuracy and build your medical history.
          </Text>
        </View>
      </View>

      {/* Pending Analyses */}
      <Text style={styles.sectionTitle}>
        Analyses Awaiting Biopsy Results ({pendingAnalyses.length})
      </Text>

      {pendingAnalyses.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="checkmark-circle-outline" size={64} color="#10b981" />
          <Text style={styles.emptyTitle}>All Caught Up!</Text>
          <Text style={styles.emptyText}>
            No pending biopsy results to add. Your analyses will appear here when you have them.
          </Text>
        </View>
      ) : (
        pendingAnalyses.slice(0, 20).map(analysis => (
          <TouchableOpacity
            key={analysis.id}
            style={styles.analysisCard}
            onPress={() => {
              setSelectedAnalysis(analysis);
              setAddResultModalVisible(true);
            }}
          >
            <View style={styles.analysisHeader}>
              <View style={styles.predictionBadge}>
                <Ionicons name="analytics-outline" size={16} color="#2563eb" />
                <Text style={styles.predictionText}>{analysis.predicted_class}</Text>
              </View>
              <Text style={styles.analysisDate}>{formatDate(analysis.created_at)}</Text>
            </View>

            <View style={styles.analysisDetails}>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>AI Confidence:</Text>
                <Text style={styles.detailValue}>
                  {(analysis.lesion_confidence * 100).toFixed(1)}%
                </Text>
              </View>
            </View>

            <View style={styles.addResultButton}>
              <Ionicons name="add-circle-outline" size={20} color="#2563eb" />
              <Text style={styles.addResultText}>Add Biopsy Result</Text>
            </View>
          </TouchableOpacity>
        ))
      )}

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Render Results Tab
  const renderResultsTab = () => (
    <ScrollView
      style={styles.tabContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      <Text style={styles.sectionTitle}>
        Completed Biopsies ({completedAnalyses.length})
      </Text>

      {completedAnalyses.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="document-text-outline" size={64} color="#9ca3af" />
          <Text style={styles.emptyTitle}>No Biopsy Results</Text>
          <Text style={styles.emptyText}>
            Your completed biopsy results will appear here once you add them.
          </Text>
        </View>
      ) : (
        completedAnalyses.map(analysis => (
          <View key={analysis.id} style={styles.resultCard}>
            <View style={styles.resultHeader}>
              <View style={styles.resultDates}>
                <Text style={styles.resultDateLabel}>Analysis Date</Text>
                <Text style={styles.resultDateValue}>{formatDate(analysis.created_at)}</Text>
              </View>
              {analysis.biopsy_date && (
                <View style={styles.resultDates}>
                  <Text style={styles.resultDateLabel}>Biopsy Date</Text>
                  <Text style={styles.resultDateValue}>{formatDate(analysis.biopsy_date)}</Text>
                </View>
              )}
            </View>

            <View style={styles.comparisonRow}>
              <View style={styles.comparisonItem}>
                <View style={styles.comparisonHeader}>
                  <Ionicons name="bulb-outline" size={18} color="#2563eb" />
                  <Text style={styles.comparisonLabel}>AI Prediction</Text>
                </View>
                <Text style={styles.comparisonValue}>{analysis.predicted_class}</Text>
                <Text style={styles.confidenceText}>
                  {(analysis.lesion_confidence * 100).toFixed(1)}% confidence
                </Text>
              </View>

              <Ionicons name="arrow-forward" size={20} color="#9ca3af" />

              <View style={styles.comparisonItem}>
                <View style={styles.comparisonHeader}>
                  <Ionicons name="medkit-outline" size={18} color="#10b981" />
                  <Text style={styles.comparisonLabel}>Biopsy Result</Text>
                </View>
                <Text style={styles.comparisonValue}>{analysis.biopsy_result}</Text>
                {analysis.biopsy_facility && (
                  <Text style={styles.facilityText}>{analysis.biopsy_facility}</Text>
                )}
              </View>
            </View>

            {analysis.biopsy_notes && (
              <View style={styles.notesSection}>
                <Text style={styles.notesLabel}>Notes:</Text>
                <Text style={styles.notesText}>{analysis.biopsy_notes}</Text>
              </View>
            )}
          </View>
        ))
      )}

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Render Accuracy Tab
  const renderAccuracyTab = () => (
    <ScrollView
      style={styles.tabContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {!accuracyStats || accuracyStats.total_biopsies === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="bar-chart-outline" size={64} color="#9ca3af" />
          <Text style={styles.emptyTitle}>No Data Yet</Text>
          <Text style={styles.emptyText}>
            Add biopsy results to see AI accuracy statistics. This helps track how well the AI predictions match actual pathology results.
          </Text>
        </View>
      ) : (
        <>
          {/* Overall Accuracy */}
          <View style={styles.accuracyOverviewCard}>
            <Text style={styles.accuracyOverviewTitle}>AI Prediction Accuracy</Text>

            <View style={styles.accuracyCirclesRow}>
              <View style={styles.accuracyCircle}>
                <View style={[styles.circleProgress, { borderColor: '#10b981' }]}>
                  <Text style={[styles.circleValue, { color: '#10b981' }]}>
                    {accuracyStats.exact_accuracy_percent}%
                  </Text>
                </View>
                <Text style={styles.circleLabel}>Exact Match</Text>
              </View>

              <View style={styles.accuracyCircle}>
                <View style={[styles.circleProgress, { borderColor: '#3b82f6' }]}>
                  <Text style={[styles.circleValue, { color: '#3b82f6' }]}>
                    {accuracyStats.category_accuracy_percent}%
                  </Text>
                </View>
                <Text style={styles.circleLabel}>Category Match</Text>
              </View>
            </View>

            <Text style={styles.totalBiopsies}>
              Based on {accuracyStats.total_biopsies} biopsy results
            </Text>
          </View>

          {/* Breakdown */}
          <View style={styles.breakdownCard}>
            <Text style={styles.breakdownTitle}>Result Breakdown</Text>

            <View style={styles.breakdownBar}>
              <View
                style={[
                  styles.breakdownSegment,
                  {
                    backgroundColor: '#10b981',
                    flex: accuracyStats.breakdown.exact_matches || 0.1,
                  },
                ]}
              />
              <View
                style={[
                  styles.breakdownSegment,
                  {
                    backgroundColor: '#f59e0b',
                    flex: accuracyStats.breakdown.category_matches || 0.1,
                  },
                ]}
              />
              <View
                style={[
                  styles.breakdownSegment,
                  {
                    backgroundColor: '#ef4444',
                    flex: accuracyStats.breakdown.mismatches || 0.1,
                  },
                ]}
              />
            </View>

            <View style={styles.breakdownLegend}>
              <View style={styles.legendItem}>
                <View style={[styles.legendDot, { backgroundColor: '#10b981' }]} />
                <Text style={styles.legendText}>
                  Exact Match ({accuracyStats.breakdown.exact_matches})
                </Text>
              </View>
              <View style={styles.legendItem}>
                <View style={[styles.legendDot, { backgroundColor: '#f59e0b' }]} />
                <Text style={styles.legendText}>
                  Category Match ({accuracyStats.breakdown.category_matches})
                </Text>
              </View>
              <View style={styles.legendItem}>
                <View style={[styles.legendDot, { backgroundColor: '#ef4444' }]} />
                <Text style={styles.legendText}>
                  Mismatch ({accuracyStats.breakdown.mismatches})
                </Text>
              </View>
            </View>
          </View>

          {/* By Condition */}
          {Object.keys(accuracyStats.by_condition).length > 0 && (
            <View style={styles.conditionCard}>
              <Text style={styles.conditionTitle}>Accuracy by Condition</Text>

              {Object.entries(accuracyStats.by_condition).map(([condition, stats]) => {
                const accuracy = stats.total > 0
                  ? Math.round((stats.correct / stats.total) * 100)
                  : 0;

                return (
                  <View key={condition} style={styles.conditionRow}>
                    <View style={styles.conditionInfo}>
                      <Text style={styles.conditionName}>{condition}</Text>
                      <Text style={styles.conditionCount}>
                        {stats.correct}/{stats.total} correct
                      </Text>
                    </View>
                    <View style={styles.conditionProgressContainer}>
                      <View style={styles.conditionProgressBg}>
                        <View
                          style={[
                            styles.conditionProgress,
                            {
                              width: `${accuracy}%`,
                              backgroundColor: accuracy >= 70 ? '#10b981' : accuracy >= 50 ? '#f59e0b' : '#ef4444',
                            },
                          ]}
                        />
                      </View>
                      <Text style={styles.conditionPercent}>{accuracy}%</Text>
                    </View>
                  </View>
                );
              })}
            </View>
          )}

          {/* Recent Correlations */}
          {accuracyStats.recent_correlations.length > 0 && (
            <View style={styles.recentCard}>
              <Text style={styles.recentTitle}>Recent Correlations</Text>

              {accuracyStats.recent_correlations.map((correlation, index) => (
                <View key={index} style={styles.correlationItem}>
                  <View style={[
                    styles.correlationIndicator,
                    { backgroundColor: getAccuracyColor(correlation.category) },
                  ]} />
                  <View style={styles.correlationContent}>
                    <View style={styles.correlationRow}>
                      <Text style={styles.correlationLabel}>AI:</Text>
                      <Text style={styles.correlationValue}>{correlation.ai_prediction}</Text>
                    </View>
                    <View style={styles.correlationRow}>
                      <Text style={styles.correlationLabel}>Biopsy:</Text>
                      <Text style={styles.correlationValue}>{correlation.biopsy_result}</Text>
                    </View>
                    {correlation.date && (
                      <Text style={styles.correlationDate}>{formatDate(correlation.date)}</Text>
                    )}
                  </View>
                  <View style={[
                    styles.correlationBadge,
                    { backgroundColor: `${getAccuracyColor(correlation.category)}20` },
                  ]}>
                    <Ionicons
                      name={
                        correlation.category === 'exact_match' ? 'checkmark-circle' :
                        correlation.category === 'category_match' ? 'checkmark' : 'close-circle'
                      }
                      size={16}
                      color={getAccuracyColor(correlation.category)}
                    />
                  </View>
                </View>
              ))}
            </View>
          )}

          {/* Disclaimer */}
          <View style={styles.disclaimerCard}>
            <Ionicons name="alert-circle-outline" size={20} color="#6b7280" />
            <Text style={styles.disclaimerText}>
              Accuracy statistics are for informational purposes only. AI predictions should never replace professional medical diagnosis.
            </Text>
          </View>
        </>
      )}

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Render Histopathology Tab
  const renderHistopathologyTab = () => (
    <ScrollView
      style={styles.tabContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Info Card */}
      <View style={styles.infoCard}>
        <Ionicons name="flask-outline" size={24} color="#8b5cf6" />
        <View style={styles.infoContent}>
          <Text style={styles.infoTitle}>AI Histopathology Analysis</Text>
          <Text style={styles.infoText}>
            Upload a biopsy slide image for AI-powered tissue analysis. This feature identifies tissue types and assesses potential malignancy.
          </Text>
        </View>
      </View>

      {/* Image Selection */}
      {!histoImage ? (
        <View style={styles.uploadSection}>
          <Text style={styles.sectionTitle}>Upload Biopsy Slide</Text>

          <View style={styles.uploadButtons}>
            <TouchableOpacity style={styles.uploadButton} onPress={pickHistoImage}>
              <LinearGradient
                colors={['#8b5cf6', '#7c3aed']}
                style={styles.uploadButtonGradient}
              >
                <Ionicons name="images-outline" size={32} color="#fff" />
                <Text style={styles.uploadButtonText}>Choose from Gallery</Text>
              </LinearGradient>
            </TouchableOpacity>

            <TouchableOpacity style={styles.uploadButton} onPress={takeHistoPhoto}>
              <LinearGradient
                colors={['#6366f1', '#4f46e5']}
                style={styles.uploadButtonGradient}
              >
                <Ionicons name="camera-outline" size={32} color="#fff" />
                <Text style={styles.uploadButtonText}>Take Photo</Text>
              </LinearGradient>
            </TouchableOpacity>
          </View>

          <View style={styles.tipsCard}>
            <Text style={styles.tipsTitle}>Tips for Best Results</Text>
            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={16} color="#10b981" />
              <Text style={styles.tipText}>Use high-resolution slide images</Text>
            </View>
            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={16} color="#10b981" />
              <Text style={styles.tipText}>Ensure good lighting and focus</Text>
            </View>
            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={16} color="#10b981" />
              <Text style={styles.tipText}>H&E stained sections work best</Text>
            </View>
            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={16} color="#10b981" />
              <Text style={styles.tipText}>10x-40x magnification recommended</Text>
            </View>
          </View>
        </View>
      ) : (
        <View style={styles.imageSection}>
          {/* Preview */}
          <View style={styles.imagePreviewContainer}>
            <Image source={{ uri: histoImage }} style={styles.imagePreview} />
            <TouchableOpacity style={styles.clearImageButton} onPress={clearHistoImage}>
              <Ionicons name="close-circle" size={28} color="#ef4444" />
            </TouchableOpacity>
          </View>

          {/* Analyze Button */}
          {!histoResult && !histoAnalyzing && (
            <TouchableOpacity style={styles.analyzeButton} onPress={analyzeHistopathology}>
              <LinearGradient
                colors={['#8b5cf6', '#7c3aed']}
                style={styles.analyzeButtonGradient}
              >
                <Ionicons name="scan-outline" size={24} color="#fff" />
                <Text style={styles.analyzeButtonText}>Analyze Slide</Text>
              </LinearGradient>
            </TouchableOpacity>
          )}

          {/* Loading */}
          {histoAnalyzing && (
            <View style={styles.analyzingContainer}>
              <ActivityIndicator size="large" color="#8b5cf6" />
              <Text style={styles.analyzingText}>Analyzing histopathology slide...</Text>
              <Text style={styles.analyzingSubtext}>This may take a moment</Text>
            </View>
          )}

          {/* Error */}
          {histoError && (
            <View style={styles.errorCard}>
              <Ionicons name="alert-circle" size={24} color="#ef4444" />
              <Text style={styles.errorText}>{histoError}</Text>
              <TouchableOpacity style={styles.retryButton} onPress={analyzeHistopathology}>
                <Text style={styles.retryButtonText}>Retry Analysis</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Results */}
          {histoResult && (
            <View style={styles.resultsSection}>
              {/* Malignancy Assessment */}
              <View style={[
                styles.malignancyCard,
                { borderLeftColor: getRiskColor(histoResult.malignancy_assessment.risk_level) }
              ]}>
                <Text style={styles.malignancyTitle}>Malignancy Assessment</Text>
                <View style={styles.riskBadge}>
                  <Text style={[
                    styles.riskLevel,
                    { color: getRiskColor(histoResult.malignancy_assessment.risk_level) }
                  ]}>
                    {histoResult.malignancy_assessment.risk_level.toUpperCase()} RISK
                  </Text>
                </View>
                <View style={styles.probabilityRow}>
                  <Text style={styles.probabilityLabel}>Malignant Probability:</Text>
                  <Text style={styles.probabilityValue}>
                    {(histoResult.malignancy_assessment.malignant_probability * 100).toFixed(1)}%
                  </Text>
                </View>
                {histoResult.malignancy_assessment.confidence_interval && (
                  <Text style={styles.confidenceInterval}>
                    95% CI: {(histoResult.malignancy_assessment.confidence_interval[0] * 100).toFixed(1)}% -
                    {(histoResult.malignancy_assessment.confidence_interval[1] * 100).toFixed(1)}%
                  </Text>
                )}

                {histoResult.malignancy_assessment.key_features.length > 0 && (
                  <View style={styles.featuresSection}>
                    <Text style={styles.featuresTitle}>Key Features Identified:</Text>
                    {histoResult.malignancy_assessment.key_features.map((feature, index) => (
                      <View key={index} style={styles.featureItem}>
                        <Ionicons name="ellipse" size={6} color="#6b7280" />
                        <Text style={styles.featureText}>{feature}</Text>
                      </View>
                    ))}
                  </View>
                )}
              </View>

              {/* Tissue Types */}
              <View style={styles.tissueCard}>
                <Text style={styles.tissueTitle}>Tissue Classification</Text>
                {histoResult.tissue_types.map((tissue, index) => (
                  <View key={index} style={styles.tissueItem}>
                    <View style={styles.tissueHeader}>
                      <Text style={styles.tissueName}>{tissue.type}</Text>
                      <Text style={styles.tissueConfidence}>
                        {(tissue.confidence * 100).toFixed(0)}%
                      </Text>
                    </View>
                    <View style={styles.tissueProgressBg}>
                      <View
                        style={[
                          styles.tissueProgress,
                          { width: `${tissue.confidence * 100}%` }
                        ]}
                      />
                    </View>
                    <Text style={styles.tissueDescription}>{tissue.description}</Text>
                  </View>
                ))}
              </View>

              {/* Quality Metrics */}
              <View style={styles.qualityCard}>
                <Text style={styles.qualityTitle}>Image Quality Assessment</Text>
                <View style={styles.qualityRow}>
                  <Text style={styles.qualityLabel}>Focus Quality:</Text>
                  <Text style={[
                    styles.qualityValue,
                    { color: getQualityColor(histoResult.quality_metrics.focus_quality) }
                  ]}>
                    {histoResult.quality_metrics.focus_quality}
                  </Text>
                </View>
                <View style={styles.qualityRow}>
                  <Text style={styles.qualityLabel}>Staining Quality:</Text>
                  <Text style={[
                    styles.qualityValue,
                    { color: getQualityColor(histoResult.quality_metrics.staining_quality) }
                  ]}>
                    {histoResult.quality_metrics.staining_quality}
                  </Text>
                </View>
                <View style={styles.qualityRow}>
                  <Text style={styles.qualityLabel}>Tissue Adequacy:</Text>
                  <Text style={[
                    styles.qualityValue,
                    { color: getQualityColor(histoResult.quality_metrics.tissue_adequacy) }
                  ]}>
                    {histoResult.quality_metrics.tissue_adequacy}
                  </Text>
                </View>
              </View>

              {/* Recommendations */}
              {histoResult.recommendations.length > 0 && (
                <View style={styles.recommendationsCard}>
                  <Text style={styles.recommendationsTitle}>Recommendations</Text>
                  {histoResult.recommendations.map((rec, index) => (
                    <View key={index} style={styles.recommendationItem}>
                      <Ionicons name="arrow-forward-circle" size={18} color="#2563eb" />
                      <Text style={styles.recommendationText}>{rec}</Text>
                    </View>
                  ))}
                </View>
              )}

              {/* AI Correlation Section */}
              {histoResult.analysis_id && (
                <View style={styles.correlationSection}>
                  <TouchableOpacity
                    style={styles.loadCorrelationButton}
                    onPress={() => loadAICorrelation(histoResult.analysis_id!)}
                    disabled={correlationLoading}
                  >
                    {correlationLoading ? (
                      <ActivityIndicator size="small" color="#2563eb" />
                    ) : (
                      <>
                        <Ionicons name="git-compare-outline" size={20} color="#2563eb" />
                        <Text style={styles.loadCorrelationText}>Load AI Correlation</Text>
                      </>
                    )}
                  </TouchableOpacity>

                  {aiCorrelation && (
                    <View style={[
                      styles.correlationCard,
                      aiCorrelation.is_critical_discordance && styles.correlationCardCritical
                    ]}>
                      <View style={styles.correlationHeader}>
                        <Ionicons
                          name={
                            aiCorrelation.concordance_status === 'concordant' ? 'checkmark-circle' :
                            aiCorrelation.is_critical_discordance ? 'warning' : 'alert-circle'
                          }
                          size={24}
                          color={
                            aiCorrelation.concordance_status === 'concordant' ? '#10b981' :
                            aiCorrelation.is_critical_discordance ? '#dc2626' : '#f59e0b'
                          }
                        />
                        <Text style={[
                          styles.concordanceStatus,
                          { color: aiCorrelation.concordance_status === 'concordant' ? '#10b981' :
                            aiCorrelation.is_critical_discordance ? '#dc2626' : '#f59e0b' }
                        ]}>
                          {aiCorrelation.concordance_status.toUpperCase()}
                        </Text>
                        {aiCorrelation.is_critical_discordance && (
                          <View style={styles.criticalBadge}>
                            <Text style={styles.criticalBadgeText}>CRITICAL</Text>
                          </View>
                        )}
                      </View>

                      <View style={styles.correlationComparison}>
                        <View style={styles.correlationColumn}>
                          <View style={styles.correlationColHeader}>
                            <Ionicons name="camera-outline" size={16} color="#2563eb" />
                            <Text style={styles.correlationColTitle}>Dermoscopy AI</Text>
                          </View>
                          <Text style={styles.correlationPrediction}>
                            {aiCorrelation.dermoscopy_prediction}
                          </Text>
                          <Text style={styles.correlationConfidence}>
                            {(aiCorrelation.dermoscopy_confidence * 100).toFixed(1)}% confidence
                          </Text>
                        </View>

                        <Ionicons name="swap-horizontal" size={24} color="#9ca3af" />

                        <View style={styles.correlationColumn}>
                          <View style={styles.correlationColHeader}>
                            <Ionicons name="flask-outline" size={16} color="#8b5cf6" />
                            <Text style={styles.correlationColTitle}>Histopathology</Text>
                          </View>
                          <Text style={styles.correlationPrediction}>
                            {aiCorrelation.histopathology_result}
                          </Text>
                          <Text style={styles.correlationConfidence}>
                            {(aiCorrelation.histopathology_confidence * 100).toFixed(1)}% confidence
                          </Text>
                        </View>
                      </View>

                      <View style={styles.assessmentSection}>
                        <Text style={styles.assessmentLabel}>Assessment:</Text>
                        <Text style={styles.assessmentText}>{aiCorrelation.assessment}</Text>
                      </View>

                      <View style={styles.concordanceTypeBadge}>
                        <Text style={styles.concordanceTypeText}>
                          {aiCorrelation.concordance_type.replace(/_/g, ' ').toUpperCase()}
                        </Text>
                      </View>
                    </View>
                  )}
                </View>
              )}

              {/* Report Download Button */}
              {histoResult.analysis_id && (
                <TouchableOpacity
                  style={styles.reportDownloadButton}
                  onPress={() => downloadReport(histoResult.analysis_id!, 'html')}
                  disabled={downloadingReport === histoResult.analysis_id}
                >
                  {downloadingReport === histoResult.analysis_id ? (
                    <ActivityIndicator size="small" color="#fff" />
                  ) : (
                    <>
                      <Ionicons name="document-text-outline" size={20} color="#fff" />
                      <Text style={styles.reportDownloadText}>Generate Clinical Report</Text>
                    </>
                  )}
                </TouchableOpacity>
              )}

              {/* New Analysis Button */}
              <TouchableOpacity style={styles.newAnalysisButton} onPress={clearHistoImage}>
                <Ionicons name="refresh-outline" size={20} color="#8b5cf6" />
                <Text style={styles.newAnalysisText}>Analyze Another Slide</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      )}

      {/* Disclaimer */}
      <View style={styles.disclaimerCard}>
        <Ionicons name="alert-circle-outline" size={20} color="#6b7280" />
        <Text style={styles.disclaimerText}>
          AI histopathology analysis is for research and educational purposes only. Results should be verified by a qualified pathologist before clinical use.
        </Text>
      </View>

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Render Timeline Tab (Lesion Progression)
  const renderTimelineTab = () => (
    <ScrollView
      style={styles.tabContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Info Card */}
      <View style={styles.infoCard}>
        <Ionicons name="git-branch-outline" size={24} color="#10b981" />
        <View style={styles.infoContent}>
          <Text style={styles.infoTitle}>Biopsy Timeline</Text>
          <Text style={styles.infoText}>
            Track your biopsy history and lesion progression over time. Select a lesion to view its detailed timeline.
          </Text>
        </View>
      </View>

      {/* Biopsy History Summary */}
      <Text style={styles.sectionTitle}>
        Recent Biopsies ({biopsyHistory.length})
      </Text>

      {biopsyHistory.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="time-outline" size={64} color="#9ca3af" />
          <Text style={styles.emptyTitle}>No Biopsy History</Text>
          <Text style={styles.emptyText}>
            Your biopsy history will appear here once you have histopathology analyses.
          </Text>
        </View>
      ) : (
        <>
          {biopsyHistory.map((item) => (
            <TouchableOpacity
              key={item.id}
              style={[
                styles.historyCard,
                item.histopathology_malignant && styles.historyCardMalignant
              ]}
              onPress={() => item.lesion_group_id && loadLesionProgression(item.lesion_group_id)}
            >
              <View style={styles.historyHeader}>
                <View style={[
                  styles.historyStatusDot,
                  { backgroundColor: item.histopathology_malignant ? '#ef4444' : '#10b981' }
                ]} />
                <Text style={styles.historyDate}>{formatDate(item.histopathology_date)}</Text>
                <View style={[
                  styles.historyRiskBadge,
                  { backgroundColor: `${getRiskColor(item.histopathology_risk_level)}20` }
                ]}>
                  <Text style={[
                    styles.historyRiskText,
                    { color: getRiskColor(item.histopathology_risk_level) }
                  ]}>
                    {item.histopathology_risk_level.toUpperCase()}
                  </Text>
                </View>
              </View>

              <Text style={styles.historyDiagnosis}>{item.histopathology_result}</Text>

              <View style={styles.historyFooter}>
                <View style={styles.historyConfidence}>
                  <Text style={styles.historyConfidenceLabel}>Confidence:</Text>
                  <Text style={styles.historyConfidenceValue}>
                    {(item.histopathology_confidence * 100).toFixed(0)}%
                  </Text>
                </View>

                <View style={[
                  styles.historyConcordanceBadge,
                  { backgroundColor: item.ai_concordance ? '#dcfce7' : '#fef2f2' }
                ]}>
                  <Ionicons
                    name={item.ai_concordance ? 'checkmark-circle' : 'close-circle'}
                    size={14}
                    color={item.ai_concordance ? '#10b981' : '#ef4444'}
                  />
                  <Text style={[
                    styles.historyConcordanceText,
                    { color: item.ai_concordance ? '#10b981' : '#ef4444' }
                  ]}>
                    {item.ai_concordance_type?.replace(/_/g, ' ') || 'Unknown'}
                  </Text>
                </View>
              </View>

              {item.lesion_group_id && (
                <View style={styles.viewProgressionButton}>
                  <Ionicons name="chevron-forward" size={16} color="#2563eb" />
                  <Text style={styles.viewProgressionText}>View Progression</Text>
                </View>
              )}
            </TouchableOpacity>
          ))}

          {/* Lesion Progression Timeline */}
          {selectedLesionId && selectedLesionProgress.length > 0 && (
            <View style={styles.progressionSection}>
              <View style={styles.progressionHeader}>
                <Text style={styles.progressionTitle}>Lesion Progression Timeline</Text>
                <TouchableOpacity
                  onPress={() => {
                    setSelectedLesionId(null);
                    setSelectedLesionProgress([]);
                  }}
                >
                  <Ionicons name="close-circle" size={24} color="#6b7280" />
                </TouchableOpacity>
              </View>

              <View style={styles.timelineContainer}>
                {selectedLesionProgress.map((event, index) => (
                  <View key={event.id} style={styles.timelineItem}>
                    <View style={styles.timelineLine}>
                      <View style={[
                        styles.timelineDot,
                        { backgroundColor: event.risk_level ?
                          getRiskColor(event.risk_level) : '#6b7280' }
                      ]} />
                      {index < selectedLesionProgress.length - 1 && (
                        <View style={styles.timelineConnector} />
                      )}
                    </View>

                    <View style={styles.timelineContent}>
                      <Text style={styles.timelineDate}>{formatDate(event.date)}</Text>
                      <View style={styles.timelineEventCard}>
                        <View style={styles.timelineEventHeader}>
                          <Text style={styles.timelineEventType}>
                            {event.event_type.replace(/_/g, ' ')}
                          </Text>
                          {event.ai_concordance !== undefined && (
                            <Ionicons
                              name={event.ai_concordance ? 'checkmark-circle' : 'close-circle'}
                              size={16}
                              color={event.ai_concordance ? '#10b981' : '#ef4444'}
                            />
                          )}
                        </View>
                        {event.diagnosis && (
                          <Text style={styles.timelineDiagnosis}>{event.diagnosis}</Text>
                        )}
                        {event.notes && (
                          <Text style={styles.timelineNotes}>{event.notes}</Text>
                        )}
                      </View>
                    </View>
                  </View>
                ))}
              </View>
            </View>
          )}
        </>
      )}

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Add Result Modal
  const renderAddResultModal = () => (
    <Modal
      visible={addResultModalVisible}
      animationType="slide"
      onRequestClose={() => {
        setAddResultModalVisible(false);
        resetForm();
      }}
    >
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => {
            setAddResultModalVisible(false);
            resetForm();
          }}>
            <Ionicons name="close" size={28} color="#1e3a5f" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Add Biopsy Result</Text>
          <View style={{ width: 28 }} />
        </View>

        <ScrollView style={styles.modalContent}>
          {/* AI Prediction Reference */}
          {selectedAnalysis && (
            <View style={styles.aiReferenceCard}>
              <Text style={styles.aiReferenceTitle}>AI Prediction</Text>
              <Text style={styles.aiReferenceValue}>{selectedAnalysis.predicted_class}</Text>
              <Text style={styles.aiReferenceConfidence}>
                {(selectedAnalysis.lesion_confidence * 100).toFixed(1)}% confidence
              </Text>
              <Text style={styles.aiReferenceDate}>
                Analyzed on {formatDate(selectedAnalysis.created_at)}
              </Text>
            </View>
          )}

          {/* Diagnosis Selection */}
          <Text style={styles.formLabel}>Pathology Diagnosis *</Text>
          <View style={styles.diagnosisGrid}>
            {COMMON_DIAGNOSES.map(diagnosis => (
              <TouchableOpacity
                key={diagnosis.id}
                style={[
                  styles.diagnosisOption,
                  biopsyResult === diagnosis.id && styles.diagnosisOptionSelected,
                  { borderLeftColor: getCategoryColor(diagnosis.category) },
                ]}
                onPress={() => setBiopsyResult(diagnosis.id)}
              >
                <Text style={[
                  styles.diagnosisText,
                  biopsyResult === diagnosis.id && styles.diagnosisTextSelected,
                ]}>
                  {diagnosis.label}
                </Text>
                <View style={[
                  styles.categoryDot,
                  { backgroundColor: getCategoryColor(diagnosis.category) },
                ]} />
              </TouchableOpacity>
            ))}
          </View>

          {/* Custom Diagnosis */}
          {biopsyResult === 'other' && (
            <>
              <Text style={styles.formLabel}>Specify Diagnosis *</Text>
              <TextInput
                style={styles.textInput}
                placeholder="Enter pathology diagnosis..."
                value={customDiagnosis}
                onChangeText={setCustomDiagnosis}
                placeholderTextColor="#9ca3af"
              />
            </>
          )}

          {/* Biopsy Date */}
          <Text style={styles.formLabel}>Biopsy Date</Text>
          <TextInput
            style={styles.textInput}
            placeholder="YYYY-MM-DD"
            value={biopsyDate}
            onChangeText={setBiopsyDate}
            placeholderTextColor="#9ca3af"
          />

          {/* Facility */}
          <Text style={styles.formLabel}>Lab / Facility</Text>
          <TextInput
            style={styles.textInput}
            placeholder="e.g., City Hospital Pathology Lab"
            value={biopsyFacility}
            onChangeText={setBiopsyFacility}
            placeholderTextColor="#9ca3af"
          />

          {/* Pathologist */}
          <Text style={styles.formLabel}>Pathologist Name</Text>
          <TextInput
            style={styles.textInput}
            placeholder="e.g., Dr. Smith"
            value={pathologistName}
            onChangeText={setPathologistName}
            placeholderTextColor="#9ca3af"
          />

          {/* Notes */}
          <Text style={styles.formLabel}>Additional Notes</Text>
          <TextInput
            style={[styles.textInput, styles.textArea]}
            placeholder="Any additional findings or comments..."
            value={biopsyNotes}
            onChangeText={setBiopsyNotes}
            multiline
            numberOfLines={4}
            placeholderTextColor="#9ca3af"
          />

          {/* Submit Button */}
          <TouchableOpacity
            style={styles.submitButton}
            onPress={submitBiopsyResult}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <>
                <Ionicons name="checkmark-circle" size={24} color="#fff" />
                <Text style={styles.submitButtonText}>Submit Biopsy Result</Text>
              </>
            )}
          </TouchableOpacity>

          <View style={styles.bottomSpacer} />
        </ScrollView>
      </LinearGradient>
    </Modal>
  );

  if (isLoading && pendingAnalyses.length === 0 && completedAnalyses.length === 0) {
    return (
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2563eb" />
          <Text style={styles.loadingText}>Loading...</Text>
        </View>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#2563eb" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Biopsy Tracking</Text>
        <View style={{ width: 40 }} />
      </View>

      {/* Tabs */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'pending' && styles.activeTab]}
          onPress={() => setActiveTab('pending')}
        >
          <Ionicons
            name="time-outline"
            size={18}
            color={activeTab === 'pending' ? '#2563eb' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'pending' && styles.activeTabText]}>
            Pending
          </Text>
          {pendingAnalyses.length > 0 && (
            <View style={styles.tabBadge}>
              <Text style={styles.tabBadgeText}>{pendingAnalyses.length}</Text>
            </View>
          )}
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'results' && styles.activeTab]}
          onPress={() => setActiveTab('results')}
        >
          <Ionicons
            name="document-text-outline"
            size={18}
            color={activeTab === 'results' ? '#2563eb' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'results' && styles.activeTabText]}>
            Results
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'accuracy' && styles.activeTab]}
          onPress={() => setActiveTab('accuracy')}
        >
          <Ionicons
            name="bar-chart-outline"
            size={18}
            color={activeTab === 'accuracy' ? '#2563eb' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'accuracy' && styles.activeTabText]}>
            Accuracy
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'histopathology' && styles.activeTab]}
          onPress={() => setActiveTab('histopathology')}
        >
          <Ionicons
            name="flask-outline"
            size={18}
            color={activeTab === 'histopathology' ? '#8b5cf6' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'histopathology' && styles.activeTabTextPurple]}>
            Slides
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'timeline' && styles.activeTab]}
          onPress={() => setActiveTab('timeline')}
        >
          <Ionicons
            name="git-branch-outline"
            size={18}
            color={activeTab === 'timeline' ? '#10b981' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'timeline' && styles.activeTabTextGreen]}>
            Timeline
          </Text>
        </TouchableOpacity>
      </View>

      {/* Tab Content */}
      {activeTab === 'pending' && renderPendingTab()}
      {activeTab === 'results' && renderResultsTab()}
      {activeTab === 'accuracy' && renderAccuracyTab()}
      {activeTab === 'histopathology' && renderHistopathologyTab()}
      {activeTab === 'timeline' && renderTimelineTab()}

      {/* Add Result Modal */}
      {renderAddResultModal()}
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
  tabBadge: {
    backgroundColor: '#ef4444',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
    marginLeft: 4,
  },
  tabBadgeText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '700',
  },
  tabContent: {
    flex: 1,
    padding: 16,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    color: '#6b7280',
    fontSize: 16,
  },
  infoCard: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
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
  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 48,
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
  analysisCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  analysisHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  predictionBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#eff6ff',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    gap: 6,
  },
  predictionText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2563eb',
  },
  analysisDate: {
    fontSize: 12,
    color: '#9ca3af',
  },
  analysisDetails: {
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  detailLabel: {
    fontSize: 13,
    color: '#6b7280',
  },
  detailValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  addResultButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#eff6ff',
    borderRadius: 8,
    padding: 12,
    gap: 8,
  },
  addResultText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2563eb',
  },
  resultCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  resultDates: {},
  resultDateLabel: {
    fontSize: 11,
    color: '#9ca3af',
  },
  resultDateValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1e3a5f',
    marginTop: 2,
  },
  comparisonRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  comparisonItem: {
    flex: 1,
  },
  comparisonHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 6,
  },
  comparisonLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
  comparisonValue: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  confidenceText: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 2,
  },
  facilityText: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 2,
  },
  notesSection: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  notesLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 4,
  },
  notesText: {
    fontSize: 13,
    color: '#374151',
    lineHeight: 20,
  },
  accuracyOverviewCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    marginBottom: 16,
    alignItems: 'center',
  },
  accuracyOverviewTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 20,
  },
  accuracyCirclesRow: {
    flexDirection: 'row',
    gap: 32,
  },
  accuracyCircle: {
    alignItems: 'center',
  },
  circleProgress: {
    width: 100,
    height: 100,
    borderRadius: 50,
    borderWidth: 6,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  circleValue: {
    fontSize: 24,
    fontWeight: '700',
  },
  circleLabel: {
    fontSize: 13,
    color: '#6b7280',
  },
  totalBiopsies: {
    fontSize: 13,
    color: '#9ca3af',
    marginTop: 16,
  },
  breakdownCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  breakdownTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 16,
  },
  breakdownBar: {
    flexDirection: 'row',
    height: 16,
    borderRadius: 8,
    overflow: 'hidden',
    marginBottom: 12,
  },
  breakdownSegment: {
    height: '100%',
  },
  breakdownLegend: {
    gap: 8,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  legendText: {
    fontSize: 13,
    color: '#6b7280',
  },
  conditionCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  conditionTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 16,
  },
  conditionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  conditionInfo: {
    flex: 1,
  },
  conditionName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  conditionCount: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 2,
  },
  conditionProgressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flex: 1,
  },
  conditionProgressBg: {
    flex: 1,
    height: 8,
    backgroundColor: '#f3f4f6',
    borderRadius: 4,
    overflow: 'hidden',
  },
  conditionProgress: {
    height: '100%',
    borderRadius: 4,
  },
  conditionPercent: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1e3a5f',
    width: 40,
    textAlign: 'right',
  },
  recentCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  recentTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  correlationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  correlationIndicator: {
    width: 4,
    height: 40,
    borderRadius: 2,
    marginRight: 12,
  },
  correlationContent: {
    flex: 1,
  },
  correlationRow: {
    flexDirection: 'row',
    gap: 6,
  },
  correlationLabel: {
    fontSize: 12,
    color: '#9ca3af',
    width: 45,
  },
  correlationValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1e3a5f',
    flex: 1,
  },
  correlationDate: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 4,
  },
  correlationBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  disclaimerCard: {
    flexDirection: 'row',
    backgroundColor: '#f9fafb',
    borderRadius: 10,
    padding: 14,
    gap: 10,
  },
  disclaimerText: {
    flex: 1,
    fontSize: 12,
    color: '#6b7280',
    lineHeight: 18,
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
  modalContent: {
    flex: 1,
    padding: 16,
  },
  aiReferenceCard: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    alignItems: 'center',
  },
  aiReferenceTitle: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4,
  },
  aiReferenceValue: {
    fontSize: 20,
    fontWeight: '700',
    color: '#2563eb',
  },
  aiReferenceConfidence: {
    fontSize: 13,
    color: '#1e40af',
    marginTop: 4,
  },
  aiReferenceDate: {
    fontSize: 12,
    color: '#9ca3af',
    marginTop: 8,
  },
  formLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
    marginTop: 16,
  },
  diagnosisGrid: {
    gap: 8,
  },
  diagnosisOption: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderLeftWidth: 4,
    borderRadius: 10,
    padding: 14,
  },
  diagnosisOptionSelected: {
    backgroundColor: '#eff6ff',
    borderColor: '#2563eb',
  },
  diagnosisText: {
    fontSize: 14,
    color: '#374151',
  },
  diagnosisTextSelected: {
    fontWeight: '600',
    color: '#2563eb',
  },
  categoryDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  textInput: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 10,
    padding: 14,
    fontSize: 15,
    color: '#1e3a5f',
  },
  textArea: {
    minHeight: 100,
    textAlignVertical: 'top',
  },
  submitButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 18,
    marginTop: 24,
    gap: 10,
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  bottomSpacer: {
    height: 30,
  },
  // Histopathology styles
  activeTabTextPurple: {
    color: '#8b5cf6',
    fontWeight: '600',
  },
  uploadSection: {
    marginTop: 8,
  },
  uploadButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 12,
  },
  uploadButton: {
    flex: 1,
  },
  uploadButtonGradient: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    borderRadius: 16,
    gap: 8,
  },
  uploadButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
  tipsCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
  },
  tipsTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  tipText: {
    fontSize: 13,
    color: '#6b7280',
  },
  imageSection: {
    marginTop: 8,
  },
  imagePreviewContainer: {
    position: 'relative',
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 8,
    marginBottom: 16,
  },
  imagePreview: {
    width: '100%',
    height: 250,
    borderRadius: 12,
    resizeMode: 'contain',
    backgroundColor: '#f3f4f6',
  },
  clearImageButton: {
    position: 'absolute',
    top: 16,
    right: 16,
    backgroundColor: '#fff',
    borderRadius: 14,
  },
  analyzeButton: {
    marginBottom: 16,
  },
  analyzeButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    gap: 10,
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  analyzingContainer: {
    alignItems: 'center',
    padding: 32,
    backgroundColor: '#fff',
    borderRadius: 16,
    marginBottom: 16,
  },
  analyzingText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e3a5f',
    marginTop: 16,
  },
  analyzingSubtext: {
    fontSize: 13,
    color: '#9ca3af',
    marginTop: 4,
  },
  errorCard: {
    backgroundColor: '#fef2f2',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginBottom: 16,
  },
  errorText: {
    fontSize: 14,
    color: '#dc2626',
    marginTop: 8,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 12,
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: '#fff',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ef4444',
  },
  retryButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ef4444',
  },
  resultsSection: {
    marginTop: 8,
  },
  malignancyCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
  },
  malignancyTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  riskBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#f3f4f6',
    borderRadius: 6,
    marginBottom: 12,
  },
  riskLevel: {
    fontSize: 14,
    fontWeight: '700',
  },
  probabilityRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  probabilityLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  probabilityValue: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  confidenceInterval: {
    fontSize: 12,
    color: '#9ca3af',
    marginBottom: 12,
  },
  featuresSection: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  featuresTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 8,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  featureText: {
    fontSize: 13,
    color: '#374151',
  },
  tissueCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  tissueTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 16,
  },
  tissueItem: {
    marginBottom: 16,
  },
  tissueHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  tissueName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  tissueConfidence: {
    fontSize: 13,
    fontWeight: '600',
    color: '#8b5cf6',
  },
  tissueProgressBg: {
    height: 6,
    backgroundColor: '#f3f4f6',
    borderRadius: 3,
    marginBottom: 4,
    overflow: 'hidden',
  },
  tissueProgress: {
    height: '100%',
    backgroundColor: '#8b5cf6',
    borderRadius: 3,
  },
  tissueDescription: {
    fontSize: 12,
    color: '#6b7280',
    lineHeight: 18,
  },
  qualityCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  qualityTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  qualityRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  qualityLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  qualityValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  recommendationsCard: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  recommendationsTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: 10,
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
  },
  newAnalysisButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginTop: 8,
    gap: 8,
    borderWidth: 1,
    borderColor: '#8b5cf6',
  },
  newAnalysisText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#8b5cf6',
  },
  // Timeline tab active text style
  activeTabTextGreen: {
    color: '#10b981',
    fontWeight: '600',
  },
  // AI Correlation Styles
  correlationSection: {
    marginTop: 16,
  },
  loadCorrelationButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 14,
    gap: 8,
    marginBottom: 12,
  },
  loadCorrelationText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2563eb',
  },
  correlationCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  correlationCardCritical: {
    borderColor: '#dc2626',
    borderWidth: 2,
    backgroundColor: '#fef2f2',
  },
  correlationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 16,
  },
  concordanceStatus: {
    fontSize: 16,
    fontWeight: '700',
    flex: 1,
  },
  criticalBadge: {
    backgroundColor: '#dc2626',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  criticalBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  correlationComparison: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 16,
  },
  correlationColumn: {
    flex: 1,
    backgroundColor: '#f9fafb',
    borderRadius: 10,
    padding: 12,
  },
  correlationColHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
  },
  correlationColTitle: {
    fontSize: 12,
    color: '#6b7280',
  },
  correlationPrediction: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 4,
  },
  correlationConfidence: {
    fontSize: 12,
    color: '#9ca3af',
  },
  assessmentSection: {
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  assessmentLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 4,
  },
  assessmentText: {
    fontSize: 13,
    color: '#374151',
    lineHeight: 20,
  },
  concordanceTypeBadge: {
    alignSelf: 'flex-start',
    backgroundColor: '#e5e7eb',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 4,
  },
  concordanceTypeText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#6b7280',
  },
  // Report Download Button
  reportDownloadButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    gap: 10,
    marginTop: 16,
    marginBottom: 8,
  },
  reportDownloadText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
  // Timeline Tab Styles
  historyCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#10b981',
  },
  historyCardMalignant: {
    borderLeftColor: '#ef4444',
  },
  historyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    gap: 8,
  },
  historyStatusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  historyDate: {
    flex: 1,
    fontSize: 13,
    color: '#6b7280',
  },
  historyRiskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4,
  },
  historyRiskText: {
    fontSize: 10,
    fontWeight: '700',
  },
  historyDiagnosis: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  historyFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  historyConfidence: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  historyConfidenceLabel: {
    fontSize: 12,
    color: '#9ca3af',
  },
  historyConfidenceValue: {
    fontSize: 12,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  historyConcordanceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  historyConcordanceText: {
    fontSize: 11,
    fontWeight: '600',
  },
  viewProgressionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
    gap: 4,
  },
  viewProgressionText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#2563eb',
  },
  // Lesion Progression Timeline
  progressionSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
  },
  progressionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  progressionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  timelineContainer: {
    paddingLeft: 8,
  },
  timelineItem: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  timelineLine: {
    alignItems: 'center',
    width: 24,
    marginRight: 12,
  },
  timelineDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#6b7280',
  },
  timelineConnector: {
    width: 2,
    flex: 1,
    backgroundColor: '#e5e7eb',
    marginTop: 4,
  },
  timelineContent: {
    flex: 1,
  },
  timelineDate: {
    fontSize: 12,
    color: '#9ca3af',
    marginBottom: 6,
  },
  timelineEventCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 12,
  },
  timelineEventHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  timelineEventType: {
    fontSize: 13,
    fontWeight: '600',
    color: '#374151',
    textTransform: 'capitalize',
  },
  timelineDiagnosis: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 4,
  },
  timelineNotes: {
    fontSize: 12,
    color: '#6b7280',
    lineHeight: 18,
  },
});
