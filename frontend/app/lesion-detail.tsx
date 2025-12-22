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
  Platform,
  Modal
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
// Charts temporarily removed due to Victory library issues
// import SizeTimelineChart from '../components/SizeTimelineChart';
// import RiskTrendChart from '../components/RiskTrendChart';
// import ConfidenceTimelineChart from '../components/ConfidenceTimelineChart';

const { width } = Dimensions.get('window');
const imageWidth = (width - 64) / 2;

interface Analysis {
  id: number;
  image_url: string;
  predicted_class: string;
  lesion_confidence: number;
  risk_level: string;
  created_at: string;
}

interface Comparison {
  id: number;
  baseline_analysis_id: number;
  current_analysis_id: number;
  time_difference_days: number;
  change_detected: boolean;
  change_severity: string;
  change_score: number;
  created_at: string;
}

interface LesionDetail {
  id: number;
  lesion_name: string;
  lesion_description: string;
  body_location: string;
  body_sublocation: string;
  body_side: string;
  first_noticed_date: string;
  monitoring_frequency: string;
  next_check_date: string;
  current_risk_level: string;
  requires_attention: boolean;
  attention_reason: string;
  total_analyses: number;
  change_detected: boolean;
  change_summary: any;
  growth_rate: number | null;
  is_active: boolean;
  archived: boolean;
  archive_reason: string;
  last_analyzed_at: string;
  created_at: string;
  analyses: Analysis[];
  comparisons: Comparison[];
}

export default function LesionDetailScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const { id } = useLocalSearchParams();
  const [lesion, setLesion] = useState<LesionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedAnalysis, setSelectedAnalysis] = useState<number | null>(null);
  const [comparisonMode, setComparisonMode] = useState(false);
  const [selectedForComparison, setSelectedForComparison] = useState<number[]>([]);
  const [comparing, setComparing] = useState(false);
  const [comparisonResult, setComparisonResult] = useState<any>(null);
  const [showComparisonModal, setShowComparisonModal] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);

  useEffect(() => {
    fetchLesionDetail();
  }, [id]);

  const fetchLesionDetail = async () => {
    try {
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert(t('lesionDetail.alerts.loginAgain'), t('lesionDetail.alerts.loginAgain'));
        router.replace('/');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/lesion_groups/${id}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setLesion(data);
      } else {
        Alert.alert(t('lesionTracking.common.error'), t('lesionDetail.alerts.fetchFailed'));
      }
    } catch (error) {
      console.error('Error fetching lesion detail:', error);
      Alert.alert(t('lesionTracking.common.error'), t('lesionDetail.alerts.networkError'));
    } finally {
      setLoading(false);
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

  const getChangeSeverityColor = (severity: string) => {
    switch (severity) {
      case 'concerning': return '#dc2626';
      case 'significant': return '#f59e0b';
      case 'moderate': return '#eab308';
      case 'minimal': return '#10b981';
      default: return '#6b7280';
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

  const formatDate = (dateString: string) => {
    if (!dateString) return t('lesionTracking.card.notSet');
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const getDaysSince = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const days = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));
    return days;
  };

  const isOverdue = (nextCheckDate: string) => {
    if (!nextCheckDate) return false;
    return new Date(nextCheckDate) < new Date();
  };

  const handleArchive = async () => {
    Alert.alert(
      t('lesionDetail.alerts.archiveTitle'),
      t('lesionDetail.alerts.archiveMessage'),
      [
        { text: t('lesionDetail.actions.cancel'), style: 'cancel' },
        {
          text: t('lesionDetail.alerts.archiveButton'),
          style: 'destructive',
          onPress: async () => {
            try {
              const token = AuthService.getToken();
              const response = await fetch(`${API_BASE_URL}/lesion_groups/${id}`, {
                method: 'DELETE',
                headers: {
                  'Authorization': `Bearer ${token}`
                }
              });

              if (response.ok) {
                Alert.alert(t('lesionTracking.common.error'), t('lesionDetail.alerts.archiveSuccess'));
                router.back();
              } else {
                Alert.alert(t('lesionTracking.common.error'), t('lesionDetail.alerts.archiveFailed'));
              }
            } catch (error) {
              Alert.alert(t('lesionTracking.common.error'), t('lesionDetail.alerts.networkError'));
            }
          }
        }
      ]
    );
  };

  const viewComparison = (comparisonId: number) => {
    router.push(`/comparison-view?id=${comparisonId}` as any);
  };

  const toggleComparisonMode = () => {
    setComparisonMode(!comparisonMode);
    setSelectedForComparison([]);
  };

  const toggleAnalysisSelection = (analysisId: number) => {
    if (selectedForComparison.includes(analysisId)) {
      setSelectedForComparison(selectedForComparison.filter(id => id !== analysisId));
    } else if (selectedForComparison.length < 2) {
      setSelectedForComparison([...selectedForComparison, analysisId]);
    } else {
      // Replace the first selection with the new one
      setSelectedForComparison([selectedForComparison[1], analysisId]);
    }
  };

  const compareSelectedAnalyses = async () => {
    if (selectedForComparison.length !== 2) {
      Alert.alert(t('lesionDetail.comparison.error'), t('lesionDetail.comparison.selectTwo'));
      return;
    }

    setComparing(true);

    try {
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert(t('lesionDetail.alerts.loginAgain'), t('lesionDetail.alerts.loginAgain'));
        return;
      }

      // Determine baseline (earlier) and current (later) based on created_at
      const analysis1 = lesion?.analyses.find(a => a.id === selectedForComparison[0]);
      const analysis2 = lesion?.analyses.find(a => a.id === selectedForComparison[1]);

      if (!analysis1 || !analysis2) {
        Alert.alert(t('lesionDetail.comparison.error'), 'Could not find selected analyses');
        return;
      }

      const baseline = new Date(analysis1.created_at) < new Date(analysis2.created_at)
        ? selectedForComparison[0]
        : selectedForComparison[1];
      const current = baseline === selectedForComparison[0]
        ? selectedForComparison[1]
        : selectedForComparison[0];

      const response = await fetch(
        `${API_BASE_URL}/lesion_groups/${id}/compare?baseline_analysis_id=${baseline}&current_analysis_id=${current}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.ok) {
        const result = await response.json();
        console.log('[COMPARISON SUCCESS] Received result keys:', Object.keys(result));
        console.log('[COMPARISON SUCCESS] Has comparison?', !!result.comparison);
        console.log('[COMPARISON SUCCESS] Has baseline?', !!result.baseline);
        console.log('[COMPARISON SUCCESS] Has current?', !!result.current);
        console.log('[COMPARISON SUCCESS] Setting modal state...');

        // Store comparison result for modal display
        setComparisonResult(result);
        setShowComparisonModal(true);
        setComparisonMode(false);
        setSelectedForComparison([]);

        console.log('[COMPARISON SUCCESS] Modal should be visible now');
      } else {
        const error = await response.json();
        console.error('[COMPARISON ERROR] API error:', error);
        Alert.alert(t('lesionDetail.comparison.error'), error.detail || 'Comparison failed');
      }
    } catch (error) {
      console.error('Error comparing analyses:', error);
      Alert.alert(t('lesionDetail.comparison.error'), t('lesionDetail.comparison.networkError'));
    } finally {
      setComparing(false);
    }
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#3b82f6" />
        <Text style={styles.loadingText}>{t('lesionDetail.loading')}</Text>
      </View>
    );
  }

  if (!lesion) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>{t('lesionDetail.notFound')}</Text>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => router.back()}
        >
          <Text style={styles.backButtonText}>{t('lesionDetail.goBack')}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const overdue = isOverdue(lesion.next_check_date);

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
        <Text style={styles.title} numberOfLines={1}>{lesion.lesion_name}</Text>
        <TouchableOpacity
          onPress={handleArchive}
          style={styles.headerButton}
        >
          <Ionicons name="archive" size={24} color="#dc2626" />
        </TouchableOpacity>
      </View>

      {/* Alert Banner */}
      {lesion.requires_attention && (
        <View style={styles.alertBanner}>
          <Ionicons name="warning" size={24} color="#dc2626" />
          <View style={styles.alertContent}>
            <Text style={styles.alertTitle}>{t('lesionDetail.alerts.attentionRequired')}</Text>
            <Text style={styles.alertText}>{lesion.attention_reason}</Text>
          </View>
        </View>
      )}

      {/* Overdue Banner */}
      {overdue && (
        <View style={styles.overdueBanner}>
          <Ionicons name="time" size={24} color="#dc2626" />
          <View style={styles.alertContent}>
            <Text style={styles.alertTitle}>{t('lesionDetail.alerts.checkOverdue')}</Text>
            <Text style={styles.alertText}>
              {t('lesionDetail.alerts.nextCheckDue', { date: formatDate(lesion.next_check_date) })}
            </Text>
          </View>
        </View>
      )}

      {/* Info Card */}
      <View style={styles.infoCard}>
        <View style={styles.infoRow}>
          <View style={styles.infoItem}>
            <Text style={styles.infoLabel}>{t('lesionDetail.info.location')}</Text>
            <Text style={styles.infoValue}>
              {lesion.body_location || t('lesionDetail.info.notSpecified')}
              {lesion.body_side && ` (${lesion.body_side})`}
            </Text>
          </View>
          <View style={[
            styles.riskBadge,
            { backgroundColor: getRiskColor(lesion.current_risk_level) }
          ]}>
            <Text style={styles.riskText}>
              {lesion.current_risk_level?.toUpperCase() || 'N/A'}
            </Text>
          </View>
        </View>

        {lesion.lesion_description && (
          <View style={styles.descriptionSection}>
            <Text style={styles.infoLabel}>{t('lesionDetail.info.description')}</Text>
            <Text style={styles.description}>{lesion.lesion_description}</Text>
          </View>
        )}

        <View style={styles.statsGrid}>
          <View style={styles.statCard}>
            <Ionicons name="images" size={24} color="#3b82f6" />
            <Text style={styles.statNumber}>{lesion.total_analyses}</Text>
            <Text style={styles.statLabel}>{t('lesionDetail.info.analyses')}</Text>
          </View>
          <View style={styles.statCard}>
            <Ionicons name="calendar" size={24} color="#3b82f6" />
            <Text style={styles.statNumber}>{lesion.monitoring_frequency}</Text>
            <Text style={styles.statLabel}>{t('lesionDetail.info.frequency')}</Text>
          </View>
          <View style={styles.statCard}>
            <Ionicons name="time" size={24} color="#3b82f6" />
            <Text style={styles.statNumber}>
              {lesion.last_analyzed_at ? getDaysSince(lesion.last_analyzed_at) : 'N/A'}
            </Text>
            <Text style={styles.statLabel}>{t('lesionDetail.info.daysSince')}</Text>
          </View>
        </View>

        {lesion.growth_rate !== null && lesion.growth_rate > 0 && (
          <View style={styles.growthCard}>
            <Ionicons name="trending-up" size={20} color="#dc2626" />
            <Text style={styles.growthText}>
              {t('lesionDetail.info.growingAt', { rate: lesion.growth_rate.toFixed(2) })}
            </Text>
          </View>
        )}

        <View style={styles.nextCheckCard}>
          <Text style={styles.nextCheckLabel}>{t('lesionDetail.info.nextCheckup')}</Text>
          <Text style={[
            styles.nextCheckDate,
            overdue && styles.overdueText
          ]}>
            {formatDate(lesion.next_check_date)}
          </Text>
        </View>
      </View>

      {/* Action Buttons */}
      <View style={styles.actionButtons}>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => {
            Alert.alert(
              t('lesionDetail.actions.addAnalysisTitle'),
              t('lesionDetail.actions.addAnalysisMessage'),
              [
                { text: t('lesionDetail.actions.cancel'), style: 'cancel' },
                {
                  text: t('lesionDetail.actions.takePhoto'),
                  onPress: () => router.push(`/home?lesion_group_id=${id}` as any)
                }
              ]
            );
          }}
        >
          <Ionicons name="camera" size={20} color="white" />
          <Text style={styles.primaryButtonText}>{t('lesionDetail.actions.addNewAnalysis')}</Text>
        </TouchableOpacity>

        {lesion.analyses.length >= 2 && (
          <TouchableOpacity
            style={[styles.secondaryButton, comparisonMode && styles.activeButton]}
            onPress={toggleComparisonMode}
          >
            <Ionicons name="git-compare" size={20} color={comparisonMode ? "white" : "#3b82f6"} />
            <Text style={[styles.secondaryButtonText, comparisonMode && styles.activeButtonText]}>
              {comparisonMode ? 'Cancel' : 'Compare Analyses'}
            </Text>
          </TouchableOpacity>
        )}

        {/* AR Treatment Simulator Button */}
        {lesion.analyses.length > 0 && (
          <TouchableOpacity
            style={styles.arTreatmentButton}
            onPress={() => {
              const latestAnalysis = lesion.analyses[0];
              const diagnosis = latestAnalysis.predicted_class || 'Unknown';
              const imageUrl = latestAnalysis.image_url || '';
              router.push(`/ar-treatment-simulator?lesionId=${id}&diagnosis=${encodeURIComponent(diagnosis)}&imageUrl=${encodeURIComponent(imageUrl)}`);
            }}
          >
            <View style={styles.arTreatmentIconContainer}>
              <Ionicons name="fitness" size={20} color="white" />
            </View>
            <View style={styles.arTreatmentTextContainer}>
              <Text style={styles.arTreatmentTitle}>Treatment Outcome Simulator</Text>
              <Text style={styles.arTreatmentSubtitle}>AI before & after predictions</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="white" />
          </TouchableOpacity>
        )}
      </View>

      {/* Comparison Mode Action Bar */}
      {comparisonMode && (
        <View style={styles.comparisonBar}>
          <View style={styles.comparisonBarContent}>
            <Ionicons name="information-circle" size={20} color="#3b82f6" />
            <Text style={styles.comparisonBarText}>
              Select 2 analyses to compare ({selectedForComparison.length}/2)
            </Text>
          </View>
          {selectedForComparison.length === 2 && (
            <TouchableOpacity
              style={styles.compareButton}
              onPress={compareSelectedAnalyses}
              disabled={comparing}
            >
              {comparing ? (
                <ActivityIndicator size="small" color="white" />
              ) : (
                <>
                  <Ionicons name="checkmark-circle" size={20} color="white" />
                  <Text style={styles.compareButtonText}>Compare</Text>
                </>
              )}
            </TouchableOpacity>
          )}
        </View>
      )}

      {/* Charts Section */}
      {/* Charts temporarily disabled for debugging */}
      {false && lesion.analyses.length >= 2 && (
        <View style={styles.chartsSection}>
          <Text style={styles.sectionTitle}>{t('lesionDetail.charts.title')}</Text>

          {/* Risk Trend Chart */}
          <RiskTrendChart
            data={lesion.analyses.map(a => ({
              date: new Date(a.created_at),
              riskLevel: a.risk_level as 'low' | 'medium' | 'high',
              diagnosis: a.predicted_class
            }))}
          />

          {/* Confidence Timeline Chart */}
          <ConfidenceTimelineChart
            data={lesion.analyses.map(a => ({
              date: new Date(a.created_at),
              confidence: a.lesion_confidence,
              diagnosis: a.predicted_class
            }))}
          />

          {/* Size Timeline Chart - only if size data available */}
          {lesion.analyses.some((a: any) => a.estimated_size_mm2) && (
            <SizeTimelineChart
              data={lesion.analyses
                .filter((a: any) => a.estimated_size_mm2)
                .map((a: any) => ({
                  date: new Date(a.created_at),
                  size: a.estimated_size_mm2,
                  label: a.predicted_class
                }))}
            />
          )}
        </View>
      )}

      {/* Timeline Section */}
      <View style={styles.timelineSection}>
        <Text style={styles.sectionTitle}>{t('lesionDetail.timeline.title', { count: lesion.analyses.length })}</Text>

        {lesion.analyses.length === 0 ? (
          <View style={styles.emptyTimeline}>
            <Ionicons name="time-outline" size={48} color="#9ca3af" />
            <Text style={styles.emptyText}>{t('lesionDetail.timeline.emptyTitle')}</Text>
          </View>
        ) : (
          <View style={styles.timeline}>
            {lesion.analyses.map((analysis, index) => {
              const comparison = lesion.comparisons.find(
                c => c.current_analysis_id === analysis.id
              );

              return (
                <View key={analysis.id} style={styles.timelineItem}>
                  {/* Timeline Line */}
                  {index < lesion.analyses.length - 1 && (
                    <View style={styles.timelineLine} />
                  )}

                  {/* Timeline Dot */}
                  <View style={[
                    styles.timelineDot,
                    { backgroundColor: getRiskColor(analysis.risk_level) }
                  ]} />

                  {/* Analysis Card */}
                  <TouchableOpacity
                    style={[
                      styles.analysisCard,
                      comparisonMode && selectedForComparison.includes(analysis.id) && styles.selectedAnalysisCard
                    ]}
                    onPress={() => {
                      if (comparisonMode) {
                        toggleAnalysisSelection(analysis.id);
                      } else {
                        setSelectedAnalysis(selectedAnalysis === analysis.id ? null : analysis.id);
                      }
                    }}
                  >
                    {comparisonMode && (
                      <View style={styles.selectionCheckbox}>
                        {selectedForComparison.includes(analysis.id) ? (
                          <Ionicons name="checkmark-circle" size={24} color="#3b82f6" />
                        ) : (
                          <Ionicons name="ellipse-outline" size={24} color="#9ca3af" />
                        )}
                      </View>
                    )}
                    <Image
                      source={{ uri: `${API_BASE_URL}${analysis.image_url}` }}
                      style={styles.analysisImage}
                    />
                    <View style={styles.analysisInfo}>
                      <Text style={styles.analysisDate}>
                        {formatDate(analysis.created_at)}
                      </Text>
                      <Text style={styles.analysisClass} numberOfLines={1}>
                        {analysis.predicted_class}
                      </Text>
                      <Text style={styles.analysisConfidence}>
                        {t('lesionDetail.timeline.confidence', { value: (analysis.lesion_confidence * 100).toFixed(1) })}
                      </Text>
                      <View style={[
                        styles.analysisRiskBadge,
                        { backgroundColor: getRiskColor(analysis.risk_level) }
                      ]}>
                        <Text style={styles.analysisRiskText}>
                          {analysis.risk_level}
                        </Text>
                      </View>
                    </View>

                    {comparison && (
                      <TouchableOpacity
                        style={styles.comparisonButton}
                        onPress={() => viewComparison(comparison.id)}
                      >
                        <Ionicons name="git-compare" size={20} color="#3b82f6" />
                      </TouchableOpacity>
                    )}
                  </TouchableOpacity>

                  {/* Comparison Badge */}
                  {comparison && comparison.change_detected && (
                    <View style={[
                      styles.changeBadge,
                      { backgroundColor: getChangeSeverityColor(comparison.change_severity) }
                    ]}>
                      <Ionicons name="alert-circle" size={16} color="white" />
                      <Text style={styles.changeBadgeText}>
                        {t('lesionDetail.timeline.change', { severity: comparison.change_severity })}
                      </Text>
                      <Text style={styles.changeBadgeSubtext}>
                        {t('lesionDetail.timeline.daysLater', { days: comparison.time_difference_days })}
                      </Text>
                    </View>
                  )}
                </View>
              );
            })}
          </View>
        )}
      </View>

      {/* History Info */}
      {lesion.first_noticed_date && (
        <View style={styles.historyCard}>
          <Text style={styles.historyLabel}>{t('lesionDetail.history.firstNoticed')}</Text>
          <Text style={styles.historyValue}>{formatDate(lesion.first_noticed_date)}</Text>
          <Text style={styles.historySubtext}>
            {t('lesionDetail.history.daysAgo', { days: getDaysSince(lesion.first_noticed_date) })}
          </Text>
        </View>
      )}

      {/* Comparison Modal */}
      <Modal
        visible={showComparisonModal}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => {
          console.log('[MODAL] Closing modal');
          setShowComparisonModal(false);
          setShowHeatmap(false);
          fetchLesionDetail(); // Refresh to show new comparison
        }}
      >
        {(() => {
          console.log('[MODAL] Rendering modal content');
          console.log('[MODAL] showComparisonModal:', showComparisonModal);
          console.log('[MODAL] comparisonResult exists:', !!comparisonResult);
          console.log('[MODAL] comparisonResult.comparison exists:', !!(comparisonResult && comparisonResult.comparison));

          if (!comparisonResult || !comparisonResult.comparison) {
            console.log('[MODAL] Showing fallback - missing data');
            return (
              <View style={styles.modalContainer}>
                <View style={styles.modalHeader}>
                  <Text style={styles.modalTitle}>Loading Comparison...</Text>
                  <TouchableOpacity
                    onPress={() => {
                      console.log('[MODAL] Force closing modal - no data');
                      setShowComparisonModal(false);
                    }}
                    style={styles.modalCloseButton}
                  >
                    <Ionicons name="close" size={28} color="#1f2937" />
                  </TouchableOpacity>
                </View>
                <View style={styles.centerContainer}>
                  <Text style={styles.errorText}>
                    No comparison data available
                  </Text>
                </View>
              </View>
            );
          }

          console.log('[MODAL] Rendering full comparison view');

          // Get image URLs from the lesion's analyses array
          const baselineAnalysis = lesion?.analyses.find(a => a.id === comparisonResult.baseline.id);
          const currentAnalysis = lesion?.analyses.find(a => a.id === comparisonResult.current.id);

          console.log('[MODAL] Baseline analysis found:', !!baselineAnalysis);
          console.log('[MODAL] Current analysis found:', !!currentAnalysis);
          console.log('[MODAL] Baseline image URL:', baselineAnalysis?.image_url);
          console.log('[MODAL] Current image URL:', currentAnalysis?.image_url);

          if (!baselineAnalysis || !currentAnalysis) {
            console.error('[MODAL] Cannot find analysis images in lesion data');
            return (
              <View style={styles.modalContainer}>
                <View style={styles.modalHeader}>
                  <Text style={styles.modalTitle}>Error</Text>
                  <TouchableOpacity
                    onPress={() => setShowComparisonModal(false)}
                    style={styles.modalCloseButton}
                  >
                    <Ionicons name="close" size={28} color="#1f2937" />
                  </TouchableOpacity>
                </View>
                <View style={styles.centerContainer}>
                  <Text style={styles.errorText}>
                    Cannot find image data for comparison
                  </Text>
                </View>
              </View>
            );
          }

          return (
            <View style={styles.modalContainer}>
            {/* Modal Header */}
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Lesion Comparison</Text>
              <TouchableOpacity
                onPress={() => {
                  setShowComparisonModal(false);
                  setShowHeatmap(false);
                  fetchLesionDetail();
                }}
                style={styles.modalCloseButton}
              >
                <Ionicons name="close" size={28} color="#1f2937" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalContent}>
                {/* Alert Banner */}
                {comparisonResult.comparison.alert_triggered && comparisonResult.comparison.alert_reasons && (
                <View style={styles.alertBanner}>
                  <Ionicons name="warning" size={24} color="#dc2626" />
                  <View style={styles.alertContent}>
                    <Text style={styles.alertTitle}>Alerts Triggered</Text>
                    {comparisonResult.comparison.alert_reasons.map((reason: string, index: number) => (
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
                    { backgroundColor: getSeverityColor(comparisonResult.comparison.change_severity) }
                  ]}>
                    <Text style={styles.severityText}>
                      {comparisonResult.comparison.change_severity.toUpperCase()}
                    </Text>
                  </View>
                  <Text style={styles.severitySubtext}>
                    Change Score: {(comparisonResult.comparison.change_score * 100).toFixed(1)}%
                  </Text>
                </View>

                <View style={styles.timeRow}>
                  <Ionicons name="time" size={20} color="#6b7280" />
                  <Text style={styles.timeText}>
                    {comparisonResult.time_difference_days} days between analyses
                  </Text>
                </View>

                {comparisonResult.comparison.action_required && (
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
                      {formatDate(comparisonResult.baseline.date)}
                    </Text>
                    <Image
                      source={{ uri: `${API_BASE_URL}${baselineAnalysis.image_url}` }}
                      style={styles.comparisonImage}
                      onError={(e) => console.error('[MODAL] Baseline image error:', e.nativeEvent.error)}
                    />
                    <View style={[
                      styles.imageRiskBadge,
                      { backgroundColor: getRiskColor(comparisonResult.comparison.baseline_risk) }
                    ]}>
                      <Text style={styles.imageRiskText}>{comparisonResult.comparison.baseline_risk}</Text>
                    </View>
                  </View>

                  <View style={styles.imageContainer}>
                    <Text style={styles.imageLabel}>Current</Text>
                    <Text style={styles.imageDate}>
                      {formatDate(comparisonResult.current.date)}
                    </Text>
                    <Image
                      source={{ uri: `${API_BASE_URL}${currentAnalysis.image_url}` }}
                      style={styles.comparisonImage}
                      onError={(e) => console.error('[MODAL] Current image error:', e.nativeEvent.error)}
                    />
                    <View style={[
                      styles.imageRiskBadge,
                      { backgroundColor: getRiskColor(comparisonResult.comparison.current_risk) }
                    ]}>
                      <Text style={styles.imageRiskText}>{comparisonResult.comparison.current_risk}</Text>
                    </View>
                  </View>
                </View>

                {/* Visual Similarity */}
                {comparisonResult.comparison.visual_similarity_score !== null && (
                  <View style={styles.similarityRow}>
                    <Text style={styles.similarityLabel}>Visual Similarity:</Text>
                    <View style={styles.similarityBar}>
                      <View
                        style={[
                          styles.similarityFill,
                          { width: `${comparisonResult.comparison.visual_similarity_score * 100}%` }
                        ]}
                      />
                    </View>
                    <Text style={styles.similarityValue}>
                      {(comparisonResult.comparison.visual_similarity_score * 100).toFixed(1)}%
                    </Text>
                  </View>
                )}

                {/* Heatmap Toggle */}
                {comparisonResult.comparison.change_heatmap && (
                  <>
                    <TouchableOpacity
                      style={styles.heatmapButton}
                      onPress={() => {
                        console.log('[MODAL] Toggling heatmap, current:', showHeatmap);
                        setShowHeatmap(!showHeatmap);
                      }}
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
                          source={{ uri: `data:image/png;base64,${comparisonResult.comparison.change_heatmap}` }}
                          style={styles.heatmapImage}
                          onError={(e) => {
                            console.error('[MODAL] Heatmap image error:', e.nativeEvent.error);
                            console.error('[MODAL] Heatmap base64 length:', comparisonResult.comparison.change_heatmap?.length);
                          }}
                          onLoad={() => console.log('[MODAL] Heatmap loaded successfully')}
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
                {comparisonResult.comparison.size_changed && (
                  <View style={styles.changeSection}>
                    <View style={styles.changeSectionHeader}>
                      <Ionicons name="resize" size={20} color="#f59e0b" />
                      <Text style={styles.changeSectionTitle}>Size Change</Text>
                    </View>
                    {comparisonResult.comparison.size_change_percent !== null && (
                      <Text style={styles.changeText}>
                        {comparisonResult.comparison.size_change_percent > 0 ? '+' : ''}
                        {comparisonResult.comparison.size_change_percent.toFixed(1)}% change
                      </Text>
                    )}
                    {comparisonResult.comparison.size_trend && (
                      <Text style={styles.changeSubtext}>
                        Trend: {comparisonResult.comparison.size_trend}
                      </Text>
                    )}
                  </View>
                )}

                {/* Color Changes */}
                {comparisonResult.comparison.color_changed && (
                  <View style={styles.changeSection}>
                    <View style={styles.changeSectionHeader}>
                      <Ionicons name="color-palette" size={20} color="#f59e0b" />
                      <Text style={styles.changeSectionTitle}>Color Change</Text>
                    </View>
                    {comparisonResult.comparison.color_description && (
                      <Text style={styles.changeText}>{comparisonResult.comparison.color_description}</Text>
                    )}
                    {comparisonResult.comparison.new_colors_appeared && (
                      <Text style={styles.warningText}>⚠️ New colors detected</Text>
                    )}
                  </View>
                )}

                {/* Shape Changes */}
                {comparisonResult.comparison.shape_changed && (
                  <View style={styles.changeSection}>
                    <View style={styles.changeSectionHeader}>
                      <Ionicons name="shapes" size={20} color="#f59e0b" />
                      <Text style={styles.changeSectionTitle}>Shape Change</Text>
                    </View>
                    {comparisonResult.comparison.asymmetry_increased && (
                      <Text style={styles.warningText}>• Asymmetry increased</Text>
                    )}
                    {comparisonResult.comparison.border_irregularity_increased && (
                      <Text style={styles.warningText}>• Border irregularity increased</Text>
                    )}
                  </View>
                )}

                {/* Symptom Changes */}
                {(comparisonResult.comparison.new_symptoms || comparisonResult.comparison.symptom_worsening) && (
                  <View style={styles.changeSection}>
                    <View style={styles.changeSectionHeader}>
                      <Ionicons name="medkit" size={20} color="#dc2626" />
                      <Text style={styles.changeSectionTitle}>Symptom Changes</Text>
                    </View>
                    {comparisonResult.comparison.symptom_changes_list && comparisonResult.comparison.symptom_changes_list.length > 0 && (
                      comparisonResult.comparison.symptom_changes_list.map((symptom: string, index: number) => (
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
                      <Text style={styles.diagnosisValue}>{comparisonResult.comparison.baseline_diagnosis}</Text>
                    </View>
                    <Ionicons
                      name={comparisonResult.comparison.diagnosis_changed ? "arrow-forward" : "checkmark"}
                      size={24}
                      color={comparisonResult.comparison.diagnosis_changed ? "#f59e0b" : "#10b981"}
                    />
                    <View style={styles.diagnosisColumn}>
                      <Text style={styles.diagnosisLabel}>Current</Text>
                      <Text style={styles.diagnosisValue}>{comparisonResult.comparison.current_diagnosis}</Text>
                    </View>
                  </View>
                  {comparisonResult.comparison.diagnosis_changed && (
                    <Text style={styles.warningText}>⚠️ Diagnosis changed</Text>
                  )}
                </View>

                {/* Risk Escalation */}
                {comparisonResult.comparison.risk_escalated && (
                  <View style={styles.riskEscalationBanner}>
                    <Ionicons name="trending-up" size={20} color="#dc2626" />
                    <Text style={styles.riskEscalationText}>
                      Risk escalated from {comparisonResult.comparison.baseline_risk} to {comparisonResult.comparison.current_risk}
                    </Text>
                  </View>
                )}
              </View>

              {/* Recommendation */}
              <View style={[
                styles.recommendationCard,
                { borderLeftColor: getUrgencyColor(comparisonResult.comparison.urgency_level) }
              ]}>
                <View style={styles.recommendationHeader}>
                  <Ionicons name="medical" size={24} color={getUrgencyColor(comparisonResult.comparison.urgency_level)} />
                  <View style={styles.recommendationHeaderText}>
                    <Text style={styles.recommendationTitle}>Clinical Recommendation</Text>
                    <View style={[
                      styles.urgencyBadge,
                      { backgroundColor: getUrgencyColor(comparisonResult.comparison.urgency_level) }
                    ]}>
                      <Text style={styles.urgencyText}>{comparisonResult.comparison.urgency_level.toUpperCase()}</Text>
                    </View>
                  </View>
                </View>
                <Text style={styles.recommendationText}>{comparisonResult.comparison.recommendation}</Text>
              </View>

              {/* Action Button */}
              {comparisonResult.comparison.action_required && (
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
          </View>
          );
        })()}
      </Modal>
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
    padding: 4
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    flex: 1,
    textAlign: 'center',
    marginHorizontal: 16
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
    marginBottom: 8,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#dc2626'
  },
  overdueBanner: {
    flexDirection: 'row',
    backgroundColor: '#fef3c7',
    padding: 16,
    margin: 16,
    marginTop: 8,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b'
  },
  alertContent: {
    flex: 1,
    marginLeft: 12
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4
  },
  alertText: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20
  },
  infoCard: {
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
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 16
  },
  infoItem: {
    flex: 1
  },
  infoLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4,
    fontWeight: '600',
    textTransform: 'uppercase'
  },
  infoValue: {
    fontSize: 16,
    color: '#1f2937',
    fontWeight: '600'
  },
  riskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6
  },
  riskText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold'
  },
  descriptionSection: {
    marginBottom: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6'
  },
  description: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20
  },
  statsGrid: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16
  },
  statCard: {
    flex: 1,
    backgroundColor: '#f9fafb',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center'
  },
  statNumber: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginTop: 8,
    marginBottom: 4
  },
  statLabel: {
    fontSize: 11,
    color: '#6b7280',
    textAlign: 'center'
  },
  growthCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 12,
    backgroundColor: '#fee2e2',
    borderRadius: 8,
    marginBottom: 16
  },
  growthText: {
    fontSize: 14,
    color: '#dc2626',
    fontWeight: '600'
  },
  nextCheckCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6'
  },
  nextCheckLabel: {
    fontSize: 14,
    color: '#6b7280',
    fontWeight: '600'
  },
  nextCheckDate: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#1f2937'
  },
  overdueText: {
    color: '#dc2626'
  },
  actionButtons: {
    paddingHorizontal: 16,
    marginBottom: 16
  },
  primaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#3b82f6',
    paddingVertical: 14,
    borderRadius: 12,
    shadowColor: '#3b82f6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold'
  },
  chartsSection: {
    paddingHorizontal: 16,
    marginBottom: 16
  },
  timelineSection: {
    marginTop: 8
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    paddingHorizontal: 16,
    marginBottom: 16
  },
  emptyTimeline: {
    alignItems: 'center',
    paddingVertical: 40
  },
  emptyText: {
    fontSize: 16,
    color: '#6b7280',
    marginTop: 12
  },
  timeline: {
    paddingHorizontal: 16
  },
  timelineItem: {
    position: 'relative',
    marginBottom: 24
  },
  timelineLine: {
    position: 'absolute',
    left: 12,
    top: 28,
    bottom: -24,
    width: 2,
    backgroundColor: '#e5e7eb'
  },
  timelineDot: {
    position: 'absolute',
    left: 6,
    top: 20,
    width: 14,
    height: 14,
    borderRadius: 7,
    borderWidth: 3,
    borderColor: 'white'
  },
  analysisCard: {
    flexDirection: 'row',
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 12,
    marginLeft: 32,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  analysisImage: {
    width: 80,
    height: 80,
    borderRadius: 8
  },
  analysisInfo: {
    flex: 1,
    marginLeft: 12
  },
  analysisDate: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4
  },
  analysisClass: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4
  },
  analysisConfidence: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 6
  },
  analysisRiskBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4
  },
  analysisRiskText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: 'white',
    textTransform: 'uppercase'
  },
  comparisonButton: {
    padding: 8
  },
  changeBadge: {
    marginLeft: 32,
    marginTop: 8,
    padding: 12,
    borderRadius: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8
  },
  changeBadgeText: {
    color: 'white',
    fontSize: 13,
    fontWeight: '600',
    textTransform: 'capitalize'
  },
  changeBadgeSubtext: {
    color: 'white',
    fontSize: 11,
    marginLeft: 'auto'
  },
  historyCard: {
    backgroundColor: 'white',
    margin: 16,
    marginTop: 8,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center'
  },
  historyLabel: {
    fontSize: 12,
    color: '#6b7280',
    fontWeight: '600',
    textTransform: 'uppercase',
    marginBottom: 8
  },
  historyValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4
  },
  historySubtext: {
    fontSize: 13,
    color: '#6b7280'
  },
  secondaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'white',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#3b82f6',
    marginTop: 12
  },
  secondaryButtonText: {
    color: '#3b82f6',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8
  },
  activeButton: {
    backgroundColor: '#3b82f6'
  },
  activeButtonText: {
    color: 'white'
  },
  comparisonBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#eff6ff',
    padding: 16,
    margin: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6'
  },
  comparisonBarContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1
  },
  comparisonBarText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginLeft: 8
  },
  compareButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#3b82f6',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8
  },
  compareButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 6
  },
  selectedAnalysisCard: {
    borderWidth: 3,
    borderColor: '#3b82f6',
    backgroundColor: '#eff6ff'
  },
  selectionCheckbox: {
    position: 'absolute',
    top: 8,
    right: 8,
    zIndex: 10,
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 2
  },
  // Modal styles
  modalContainer: {
    flex: 1,
    backgroundColor: '#f9fafb'
  },
  modalHeader: {
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
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    flex: 1,
    textAlign: 'center'
  },
  modalCloseButton: {
    padding: 4
  },
  modalContent: {
    flex: 1
  },
  // Comparison card styles
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
  },
  // AR Treatment Simulator Button Styles
  arTreatmentButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#667eea',
    padding: 16,
    borderRadius: 12,
    marginTop: 12,
    shadowColor: '#667eea',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4
  },
  arTreatmentIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12
  },
  arTreatmentIcon: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white'
  },
  arTreatmentTextContainer: {
    flex: 1
  },
  arTreatmentTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 2
  },
  arTreatmentSubtitle: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.9)'
  }
});
