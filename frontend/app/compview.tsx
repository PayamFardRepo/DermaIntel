import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  Alert,
  ActivityIndicator,
  Dimensions,
  Platform
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';
import { Ionicons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');
const imageWidth = (width - 48) / 2;

interface ComparisonData {
  id: number;
  lesion_group_id: number;
  baseline_analysis: {
    id: number;
    image_url: string;
    predicted_class: string;
    confidence: number;
    risk_level: string;
    created_at: string;
  };
  current_analysis: {
    id: number;
    image_url: string;
    predicted_class: string;
    confidence: number;
    risk_level: string;
    created_at: string;
  };
  time_difference_days: number;
  change_detected: boolean;
  change_severity: string;
  change_score: number;
  size_changed: boolean;
  size_change_percent: number | null;
  size_change_mm: number | null;
  size_trend: string | null;
  color_changed: boolean;
  color_change_score: number | null;
  color_description: string | null;
  new_colors_appeared: boolean;
  shape_changed: boolean;
  asymmetry_increased: boolean;
  border_irregularity_increased: boolean;
  shape_change_score: number | null;
  new_symptoms: boolean;
  symptom_worsening: boolean;
  symptom_changes_list: string[] | null;
  diagnosis_changed: boolean;
  baseline_diagnosis: string;
  current_diagnosis: string;
  risk_escalated: boolean;
  baseline_risk: string;
  current_risk: string;
  abcde_worsening: boolean;
  abcde_comparison: any;
  visual_similarity_score: number | null;
  change_heatmap: string | null;
  action_required: boolean;
  recommendation: string;
  urgency_level: string;
  alert_triggered: boolean;
  alert_reasons: string[] | null;
  created_at: string;
}

export default function ComparisonViewScreen() {
  console.log('ComparisonViewScreen mounted');
  const router = useRouter();
  const { t } = useTranslation();
  const [comparison, setComparison] = useState<ComparisonData | null>(null);
  const [loading, setLoading] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);

  useEffect(() => {
    console.log('ComparisonViewScreen useEffect triggered');
    loadComparison();
  }, []);

  const loadComparison = async () => {
    try {
      console.log('loadComparison: Starting to load...');
      const storedData = await AsyncStorage.getItem('tempComparisonData');
      console.log('loadComparison: Retrieved data, length:', storedData?.length || 0);

      if (storedData) {
        const data = JSON.parse(storedData);
        console.log('loadComparison: Parsed data successfully');
        // Transform the data to match ComparisonData interface
        setComparison({
          ...data.comparison,
          id: data.comparison_id,
          baseline_analysis: data.baseline,
          current_analysis: data.current,
          time_difference_days: data.time_difference_days
        });
        console.log('loadComparison: Comparison data set');
        // Clear the temp data after loading
        await AsyncStorage.removeItem('tempComparisonData');
        console.log('loadComparison: Temp data cleared');
      } else {
        console.log('loadComparison: No stored data found');
        Alert.alert('Error', 'No comparison data available');
      }
    } catch (error) {
      console.error('loadComparison: Error occurred:', error);
      Alert.alert('Error', 'Failed to load comparison data: ' + error.message);
    } finally {
      console.log('loadComparison: Setting loading to false');
      setLoading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'concerning': return '#dc2626';
      case 'significant': return '#f59e0b';
      case 'moderate': return '#eab308';
      case 'minimal': return '#10b981';
      case 'none': return '#6b7280';
      default: return '#6b7280';
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'emergency': return '#dc2626';
      case 'urgent': return '#f59e0b';
      case 'soon': return '#eab308';
      case 'routine': return '#10b981';
      default: return '#6b7280';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return '#dc2626';
      case 'medium': return '#f59e0b';
      case 'low': return '#10b981';
      default: return '#6b7280';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#3b82f6" />
        <Text style={styles.loadingText}>Loading comparison...</Text>
      </View>
    );
  }

  if (!comparison) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>Comparison not found</Text>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => router.back()}
        >
          <Text style={styles.backButtonText}>{t('common.goBack')}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => router.back()}
          style={styles.headerButton}
        >
          <Ionicons name="arrow-back" size={24} color="#1f2937" />
        </TouchableOpacity>
        <Text style={styles.title}>Lesion Comparison</Text>
        <View style={styles.headerButton} />
      </View>

      {/* Alert Banner */}
      {comparison.alert_triggered && comparison.alert_reasons && (
        <View style={styles.alertBanner}>
          <Ionicons name="warning" size={24} color="#dc2626" />
          <View style={styles.alertContent}>
            <Text style={styles.alertTitle}>Alerts Triggered</Text>
            {comparison.alert_reasons.map((reason, index) => (
              <Text key={index} style={styles.alertText}>• {reason}</Text>
            ))}
          </View>
        </View>
      )}

      {/* Overall Assessment */}
      <View style={styles.assessmentCard}>
        <Text style={styles.cardTitle}>Overall Assessment</Text>

        <View style={styles.severityRow}>
          <View style={[
            styles.severityBadge,
            { backgroundColor: getSeverityColor(comparison.change_severity) }
          ]}>
            <Text style={styles.severityText}>
              {comparison.change_severity.toUpperCase()}
            </Text>
          </View>
          <Text style={styles.severitySubtext}>
            Change Score: {(comparison.change_score * 100).toFixed(1)}%
          </Text>
        </View>

        <View style={styles.timeRow}>
          <Ionicons name="time" size={20} color="#6b7280" />
          <Text style={styles.timeText}>
            {comparison.time_difference_days} days between analyses
          </Text>
        </View>

        {comparison.action_required && (
          <View style={styles.actionRequiredBanner}>
            <Ionicons name="alert-circle" size={20} color="#dc2626" />
            <Text style={styles.actionRequiredText}>Action Required</Text>
          </View>
        )}
      </View>

      {/* Side-by-Side Images */}
      <View style={styles.imagesCard}>
        <Text style={styles.cardTitle}>Visual Comparison</Text>

        <View style={styles.imagesRow}>
          <View style={styles.imageContainer}>
            <Text style={styles.imageLabel}>Baseline</Text>
            <Text style={styles.imageDate}>
              {formatDate(comparison.baseline_analysis.created_at)}
            </Text>
            <Image
              source={{ uri: `${API_BASE_URL}${comparison.baseline_analysis.image_url}` }}
              style={styles.comparisonImage}
            />
            <View style={[
              styles.imageRiskBadge,
              { backgroundColor: getRiskColor(comparison.baseline_risk) }
            ]}>
              <Text style={styles.imageRiskText}>{comparison.baseline_risk}</Text>
            </View>
          </View>

          <View style={styles.imageContainer}>
            <Text style={styles.imageLabel}>Current</Text>
            <Text style={styles.imageDate}>
              {formatDate(comparison.current_analysis.created_at)}
            </Text>
            <Image
              source={{ uri: `${API_BASE_URL}${comparison.current_analysis.image_url}` }}
              style={styles.comparisonImage}
            />
            <View style={[
              styles.imageRiskBadge,
              { backgroundColor: getRiskColor(comparison.current_risk) }
            ]}>
              <Text style={styles.imageRiskText}>{comparison.current_risk}</Text>
            </View>
          </View>
        </View>

        {/* Visual Similarity */}
        {comparison.visual_similarity_score !== null && (
          <View style={styles.similarityRow}>
            <Text style={styles.similarityLabel}>Visual Similarity:</Text>
            <View style={styles.similarityBar}>
              <View
                style={[
                  styles.similarityFill,
                  { width: `${comparison.visual_similarity_score * 100}%` }
                ]}
              />
            </View>
            <Text style={styles.similarityValue}>
              {(comparison.visual_similarity_score * 100).toFixed(1)}%
            </Text>
          </View>
        )}

        {/* Heatmap Toggle */}
        {comparison.change_heatmap && (
          <>
            <TouchableOpacity
              style={styles.heatmapButton}
              onPress={() => setShowHeatmap(!showHeatmap)}
            >
              <Ionicons
                name={showHeatmap ? "eye-off" : "eye"}
                size={20}
                color="#3b82f6"
              />
              <Text style={styles.heatmapButtonText}>
                {showHeatmap ? 'Hide' : 'Show'} Change Heatmap
              </Text>
            </TouchableOpacity>

            {showHeatmap && (
              <View style={styles.heatmapContainer}>
                <Image
                  source={{ uri: `data:image/png;base64,${comparison.change_heatmap}` }}
                  style={styles.heatmapImage}
                />
                <Text style={styles.heatmapCaption}>
                  Red areas show where the AI detected the most change
                </Text>
              </View>
            )}
          </>
        )}
      </View>

      {/* Detailed Changes */}
      <View style={styles.changesCard}>
        <Text style={styles.cardTitle}>Detailed Changes</Text>

        {/* Size Changes */}
        {comparison.size_changed && (
          <View style={styles.changeSection}>
            <View style={styles.changeSectionHeader}>
              <Ionicons name="resize" size={20} color="#f59e0b" />
              <Text style={styles.changeSectionTitle}>Size Change</Text>
            </View>
            {comparison.size_change_percent !== null && (
              <Text style={styles.changeText}>
                {comparison.size_change_percent > 0 ? '+' : ''}
                {comparison.size_change_percent.toFixed(1)}% change
              </Text>
            )}
            {comparison.size_trend && (
              <Text style={styles.changeSubtext}>
                Trend: {comparison.size_trend}
              </Text>
            )}
          </View>
        )}

        {/* Color Changes */}
        {comparison.color_changed && (
          <View style={styles.changeSection}>
            <View style={styles.changeSectionHeader}>
              <Ionicons name="color-palette" size={20} color="#f59e0b" />
              <Text style={styles.changeSectionTitle}>Color Change</Text>
            </View>
            {comparison.color_description && (
              <Text style={styles.changeText}>{comparison.color_description}</Text>
            )}
            {comparison.new_colors_appeared && (
              <Text style={styles.warningText}>⚠️ New colors detected</Text>
            )}
          </View>
        )}

        {/* Shape Changes */}
        {comparison.shape_changed && (
          <View style={styles.changeSection}>
            <View style={styles.changeSectionHeader}>
              <Ionicons name="shapes" size={20} color="#f59e0b" />
              <Text style={styles.changeSectionTitle}>Shape Change</Text>
            </View>
            {comparison.asymmetry_increased && (
              <Text style={styles.warningText}>• Asymmetry increased</Text>
            )}
            {comparison.border_irregularity_increased && (
              <Text style={styles.warningText}>• Border irregularity increased</Text>
            )}
          </View>
        )}

        {/* Symptom Changes */}
        {(comparison.new_symptoms || comparison.symptom_worsening) && (
          <View style={styles.changeSection}>
            <View style={styles.changeSectionHeader}>
              <Ionicons name="medkit" size={20} color="#dc2626" />
              <Text style={styles.changeSectionTitle}>Symptom Changes</Text>
            </View>
            {comparison.symptom_changes_list && comparison.symptom_changes_list.length > 0 && (
              comparison.symptom_changes_list.map((symptom, index) => (
                <Text key={index} style={styles.warningText}>• {symptom}</Text>
              ))
            )}
          </View>
        )}

        {/* Diagnosis */}
        <View style={styles.diagnosisSection}>
          <View style={styles.diagnosisRow}>
            <View style={styles.diagnosisColumn}>
              <Text style={styles.diagnosisLabel}>Baseline</Text>
              <Text style={styles.diagnosisValue}>{comparison.baseline_diagnosis}</Text>
            </View>
            <Ionicons
              name={comparison.diagnosis_changed ? "arrow-forward" : "checkmark"}
              size={24}
              color={comparison.diagnosis_changed ? "#f59e0b" : "#10b981"}
            />
            <View style={styles.diagnosisColumn}>
              <Text style={styles.diagnosisLabel}>Current</Text>
              <Text style={styles.diagnosisValue}>{comparison.current_diagnosis}</Text>
            </View>
          </View>
          {comparison.diagnosis_changed && (
            <Text style={styles.warningText}>⚠️ Diagnosis changed</Text>
          )}
        </View>

        {/* Risk Escalation */}
        {comparison.risk_escalated && (
          <View style={styles.riskEscalationBanner}>
            <Ionicons name="trending-up" size={20} color="#dc2626" />
            <Text style={styles.riskEscalationText}>
              Risk escalated from {comparison.baseline_risk} to {comparison.current_risk}
            </Text>
          </View>
        )}
      </View>

      {/* ABCDE Comparison */}
      {comparison.abcde_worsening && comparison.abcde_comparison && (
        <View style={styles.abcdeCard}>
          <Text style={styles.cardTitle}>ABCDE Criteria</Text>
          <View style={styles.abcdeWarning}>
            <Ionicons name="warning" size={20} color="#dc2626" />
            <Text style={styles.abcdeWarningText}>ABCDE criteria worsened</Text>
          </View>
          {/* Add detailed ABCDE comparison rendering here */}
        </View>
      )}

      {/* Recommendation */}
      <View style={[
        styles.recommendationCard,
        { borderLeftColor: getUrgencyColor(comparison.urgency_level) }
      ]}>
        <View style={styles.recommendationHeader}>
          <Ionicons name="medical" size={24} color={getUrgencyColor(comparison.urgency_level)} />
          <View style={styles.recommendationHeaderText}>
            <Text style={styles.recommendationTitle}>Clinical Recommendation</Text>
            <View style={[
              styles.urgencyBadge,
              { backgroundColor: getUrgencyColor(comparison.urgency_level) }
            ]}>
              <Text style={styles.urgencyText}>{comparison.urgency_level.toUpperCase()}</Text>
            </View>
          </View>
        </View>
        <Text style={styles.recommendationText}>{comparison.recommendation}</Text>
      </View>

      {/* Action Button */}
      {comparison.action_required && (
        <View style={styles.actionSection}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => {
              Alert.alert(
                'Recommended Action',
                'Consider scheduling an appointment with a dermatologist to review these changes.',
                [
                  { text: 'Later', style: 'cancel' },
                  { text: 'Schedule', onPress: () => {
                    // TODO: Navigate to appointment booking or share with dermatologist
                    Alert.alert('Feature Coming Soon', 'Telemedicine integration is in development');
                  }}
                ]
              );
            }}
          >
            <Text style={styles.actionButtonText}>Schedule Dermatologist Review</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb'
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9fafb'
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#6b7280'
  },
  errorText: {
    fontSize: 18,
    color: '#dc2626',
    marginBottom: 16
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 60 : 16,
    paddingBottom: 16,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb'
  },
  headerButton: {
    padding: 4,
    width: 32
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    flex: 1,
    textAlign: 'center'
  },
  backButton: {
    backgroundColor: '#3b82f6',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8
  },
  backButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600'
  },
  alertBanner: {
    flexDirection: 'row',
    backgroundColor: '#fee2e2',
    padding: 16,
    margin: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#dc2626'
  },
  alertContent: {
    flex: 1,
    marginLeft: 12
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 8
  },
  alertText: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
    marginBottom: 4
  },
  assessmentCard: {
    backgroundColor: 'white',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16
  },
  severityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12
  },
  severityBadge: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8
  },
  severityText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold'
  },
  severitySubtext: {
    fontSize: 14,
    color: '#6b7280'
  },
  timeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12
  },
  timeText: {
    fontSize: 14,
    color: '#6b7280'
  },
  actionRequiredBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#fee2e2',
    padding: 12,
    borderRadius: 8
  },
  actionRequiredText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#dc2626'
  },
  imagesCard: {
    backgroundColor: 'white',
    margin: 16,
    marginTop: 0,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  imagesRow: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 16
  },
  imageContainer: {
    flex: 1,
    alignItems: 'center'
  },
  imageLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 4,
    textTransform: 'uppercase'
  },
  imageDate: {
    fontSize: 11,
    color: '#9ca3af',
    marginBottom: 8
  },
  comparisonImage: {
    width: imageWidth,
    height: imageWidth,
    borderRadius: 8,
    marginBottom: 8
  },
  imageRiskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 6
  },
  imageRiskText: {
    color: 'white',
    fontSize: 11,
    fontWeight: 'bold',
    textTransform: 'uppercase'
  },
  similarityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 16
  },
  similarityLabel: {
    fontSize: 13,
    color: '#6b7280',
    width: 100
  },
  similarityBar: {
    flex: 1,
    height: 8,
    backgroundColor: '#e5e7eb',
    borderRadius: 4,
    overflow: 'hidden'
  },
  similarityFill: {
    height: '100%',
    backgroundColor: '#3b82f6'
  },
  similarityValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1f2937',
    width: 50,
    textAlign: 'right'
  },
  heatmapButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 12,
    backgroundColor: '#eff6ff',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#3b82f6'
  },
  heatmapButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#3b82f6'
  },
  heatmapContainer: {
    marginTop: 16,
    alignItems: 'center'
  },
  heatmapImage: {
    width: width - 64,
    height: width - 64,
    borderRadius: 12
  },
  heatmapCaption: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center'
  },
  changesCard: {
    backgroundColor: 'white',
    margin: 16,
    marginTop: 0,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  changeSection: {
    marginBottom: 16,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6'
  },
  changeSectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8
  },
  changeSectionTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937'
  },
  changeText: {
    fontSize: 14,
    color: '#4b5563',
    marginLeft: 28,
    lineHeight: 20
  },
  changeSubtext: {
    fontSize: 13,
    color: '#6b7280',
    marginLeft: 28,
    marginTop: 4
  },
  warningText: {
    fontSize: 14,
    color: '#dc2626',
    marginLeft: 28,
    marginTop: 4,
    fontWeight: '500'
  },
  diagnosisSection: {
    marginTop: 8
  },
  diagnosisRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8
  },
  diagnosisColumn: {
    flex: 1,
    alignItems: 'center'
  },
  diagnosisLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4,
    textTransform: 'uppercase',
    fontWeight: '600'
  },
  diagnosisValue: {
    fontSize: 14,
    color: '#1f2937',
    fontWeight: '600',
    textAlign: 'center'
  },
  riskEscalationBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#fee2e2',
    padding: 12,
    borderRadius: 8,
    marginTop: 8
  },
  riskEscalationText: {
    fontSize: 14,
    color: '#dc2626',
    fontWeight: '600',
    flex: 1
  },
  abcdeCard: {
    backgroundColor: 'white',
    margin: 16,
    marginTop: 0,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  abcdeWarning: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#fee2e2',
    padding: 12,
    borderRadius: 8
  },
  abcdeWarningText: {
    fontSize: 14,
    color: '#dc2626',
    fontWeight: '600'
  },
  recommendationCard: {
    backgroundColor: 'white',
    margin: 16,
    marginTop: 0,
    padding: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  recommendationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12
  },
  recommendationHeaderText: {
    flex: 1
  },
  recommendationTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 6
  },
  urgencyBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 6
  },
  urgencyText: {
    color: 'white',
    fontSize: 11,
    fontWeight: 'bold'
  },
  recommendationText: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 22
  },
  actionSection: {
    padding: 16,
    paddingTop: 0
  },
  actionButton: {
    backgroundColor: '#3b82f6',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#3b82f6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4
  },
  actionButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold'
  }
});
