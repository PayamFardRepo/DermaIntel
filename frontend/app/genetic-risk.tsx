import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Platform
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';

interface GeneticRiskProfile {
  overall_genetic_risk_score: number;
  overall_risk_level: string;
  melanoma_risk_score: number;
  melanoma_risk_level: string;
  basal_cell_carcinoma_risk_score: number;
  squamous_cell_carcinoma_risk_score: number;
  family_history_score: number;
  first_degree_relatives_affected: number;
  second_degree_relatives_affected: number;
  total_relatives_with_skin_cancer: number;
  has_multiple_family_melanomas: boolean;
  has_early_onset_melanoma: boolean;
  familial_melanoma_syndrome_suspected: boolean;
  personal_risk_score: number;
  high_risk_phenotype: boolean;
  atypical_mole_syndrome: boolean;
  previous_skin_cancers_count: number;
  genetic_counseling_recommended: boolean;
  recommended_screening_frequency: string;
  recommended_professional_frequency: string;
  high_priority_monitoring: boolean;
  risk_reduction_recommendations: string[];
  inheritance_pattern: string;
  affected_lineages: {
    maternal: boolean;
    paternal: boolean;
  };
  generation_pattern: {
    grandparents: number;
    parents: number;
    siblings: number;
    aunts_uncles: number;
  };
  last_calculated: string;
  confidence_level: number;
}

export default function GeneticRiskScreen() {
  const { t } = useTranslation();
  const { isAuthenticated, logout } = useAuth();
  const router = useRouter();
  const [riskProfile, setRiskProfile] = useState<GeneticRiskProfile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    } else {
      loadGeneticRisk();
    }
  }, [isAuthenticated]);

  const loadGeneticRisk = async () => {
    try {
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert(t('geneticRisk.authError'), t('geneticRisk.pleaseLoginAgain'));
        logout();
        return;
      }

      const response = await fetch(`${API_BASE_URL}/genetic-risk`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setRiskProfile(data);
      } else if (response.status === 401) {
        Alert.alert(t('geneticRisk.sessionExpired'), t('geneticRisk.pleaseLoginAgain'));
        logout();
      } else {
        Alert.alert(t('geneticRisk.error'), t('geneticRisk.loadError'));
      }
    } catch (error) {
      console.error('Error loading genetic risk:', error);
      Alert.alert(t('geneticRisk.error'), t('geneticRisk.networkError'));
    } finally {
      setLoading(false);
    }
  };

  const recalculateRisk = async () => {
    setLoading(true);
    try {
      const token = AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/genetic-risk/recalculate`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        Alert.alert(t('geneticRisk.success'), t('geneticRisk.recalculated'));
        loadGeneticRisk();
      } else {
        Alert.alert(t('geneticRisk.error'), t('geneticRisk.recalculateError'));
      }
    } catch (error) {
      Alert.alert(t('geneticRisk.error'), t('geneticRisk.networkError'));
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'very_high':
        return '#dc2626';
      case 'high':
        return '#f59e0b';
      case 'moderate':
        return '#eab308';
      case 'low':
        return '#10b981';
      default:
        return '#6b7280';
    }
  };

  const getRiskLevelLabel = (level: string) => {
    switch (level) {
      case 'very_high':
        return t('geneticRisk.riskLevels.veryHigh');
      case 'high':
        return t('geneticRisk.riskLevels.high');
      case 'moderate':
        return t('geneticRisk.riskLevels.moderate');
      case 'low':
        return t('geneticRisk.riskLevels.low');
      default:
        return level.replace('_', ' ').toUpperCase();
    }
  };

  const getInheritancePatternLabel = (pattern: string) => {
    switch (pattern) {
      case 'likely_hereditary':
        return t('geneticRisk.inheritancePatterns.likelyHereditary');
      case 'possible_hereditary':
        return t('geneticRisk.inheritancePatterns.possiblyHereditary');
      case 'confirmed_hereditary':
        return t('geneticRisk.inheritancePatterns.confirmedHereditary');
      default:
        return t('geneticRisk.inheritancePatterns.sporadic');
    }
  };

  const renderRiskScore = (label: string, score: number, level: string) => {
    const color = getRiskLevelColor(level);
    const percentage = score;

    return (
      <View style={styles.riskScoreCard}>
        <Text style={styles.riskScoreLabel}>{label}</Text>
        <View style={styles.scoreContainer}>
          <Text style={[styles.scoreNumber, { color }]}>{score.toFixed(0)}</Text>
          <Text style={styles.scoreMax}>/100</Text>
        </View>
        <View style={styles.progressBarContainer}>
          <View style={[styles.progressBar, { width: `${percentage}%`, backgroundColor: color }]} />
        </View>
        <Text style={[styles.riskLevel, { color }]}>{getRiskLevelLabel(level)}</Text>
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#8b5cf6" />
        <Text style={styles.loadingText}>{t('geneticRisk.calculating')}</Text>
      </View>
    );
  }

  if (!riskProfile) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>{t('geneticRisk.noData')}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={loadGeneticRisk}>
          <Text style={styles.retryButtonText}>{t('geneticRisk.retry')}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <LinearGradient
      colors={['#7c3aed', '#8b5cf6', '#a78bfa']}
      style={styles.container}
    >
      <ScrollView style={styles.scrollView}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.title}>{t('geneticRisk.title')}</Text>
          <TouchableOpacity onPress={recalculateRisk} style={styles.refreshButton}>
            <Ionicons name="refresh" size={24} color="white" />
          </TouchableOpacity>
        </View>

        {/* Overall Risk Card */}
        <View style={styles.overallRiskCard}>
          <Text style={styles.overallRiskTitle}>{t('geneticRisk.overallRisk')}</Text>
          <View style={styles.overallRiskContent}>
            <View style={styles.overallScoreCircle}>
              <Text style={[styles.overallScore, { color: getRiskLevelColor(riskProfile.overall_risk_level) }]}>
                {riskProfile.overall_genetic_risk_score.toFixed(0)}
              </Text>
              <Text style={styles.overallScoreLabel}>/100</Text>
            </View>
            <View style={styles.overallRiskInfo}>
              <Text style={[styles.overallRiskLevel, { color: getRiskLevelColor(riskProfile.overall_risk_level) }]}>
                {getRiskLevelLabel(riskProfile.overall_risk_level)}
              </Text>
              <Text style={styles.overallRiskSubtext}>
                {t('geneticRisk.basedOn', {
                  count: riskProfile.total_relatives_with_skin_cancer
                })}
              </Text>
            </View>
          </View>

          {/* Confidence Level */}
          <View style={styles.confidenceBar}>
            <Text style={styles.confidenceLabel}>
              {t('geneticRisk.assessmentConfidence')} {(riskProfile.confidence_level * 100).toFixed(0)}%
            </Text>
            <View style={styles.confidenceProgress}>
              <View style={[styles.confidenceProgressFill, { width: `${riskProfile.confidence_level * 100}%` }]} />
            </View>
          </View>
        </View>

        {/* Specific Risk Scores */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>{t('geneticRisk.riskBreakdown')}</Text>
          {renderRiskScore(t('geneticRisk.melanomaRisk'), riskProfile.melanoma_risk_score, riskProfile.melanoma_risk_level)}
          {renderRiskScore(t('geneticRisk.basalCellCarcinoma'), riskProfile.basal_cell_carcinoma_risk_score, riskProfile.overall_risk_level)}
          {renderRiskScore(t('geneticRisk.squamousCellCarcinoma'), riskProfile.squamous_cell_carcinoma_risk_score, riskProfile.overall_risk_level)}
        </View>

        {/* Family History Summary */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>{t('geneticRisk.familyHistorySummary')}</Text>
          <View style={styles.summaryCard}>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>{t('geneticRisk.familyHistoryScore')}</Text>
              <Text style={styles.summaryValue}>{riskProfile.family_history_score.toFixed(0)}/100</Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>{t('geneticRisk.firstDegreeRelatives')}</Text>
              <Text style={styles.summaryValue}>{riskProfile.first_degree_relatives_affected}</Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>{t('geneticRisk.secondDegreeRelatives')}</Text>
              <Text style={styles.summaryValue}>{riskProfile.second_degree_relatives_affected}</Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>{t('geneticRisk.inheritancePattern')}</Text>
              <Text style={[styles.summaryValue, { color: '#8b5cf6' }]}>
                {getInheritancePatternLabel(riskProfile.inheritance_pattern)}
              </Text>
            </View>

            {/* Lineage Affected */}
            <View style={styles.lineageContainer}>
              <Text style={styles.summaryLabel}>{t('geneticRisk.affectedLineages')}</Text>
              <View style={styles.lineageBadges}>
                {riskProfile.affected_lineages.maternal && (
                  <View style={styles.lineageBadge}>
                    <Text style={styles.lineageBadgeText}>{t('geneticRisk.maternal')}</Text>
                  </View>
                )}
                {riskProfile.affected_lineages.paternal && (
                  <View style={styles.lineageBadge}>
                    <Text style={styles.lineageBadgeText}>{t('geneticRisk.paternal')}</Text>
                  </View>
                )}
                {!riskProfile.affected_lineages.maternal && !riskProfile.affected_lineages.paternal && (
                  <Text style={styles.summaryValue}>{t('geneticRisk.none')}</Text>
                )}
              </View>
            </View>
          </View>
        </View>

        {/* High Risk Indicators */}
        {(riskProfile.familial_melanoma_syndrome_suspected ||
          riskProfile.has_early_onset_melanoma ||
          riskProfile.has_multiple_family_melanomas) && (
          <View style={styles.alertSection}>
            <View style={styles.alertHeader}>
              <Ionicons name="warning" size={24} color="#dc2626" />
              <Text style={styles.alertTitle}>{t('geneticRisk.highRiskIndicators')}</Text>
            </View>
            {riskProfile.familial_melanoma_syndrome_suspected && (
              <View style={styles.alertItem}>
                <Ionicons name="alert-circle" size={16} color="#dc2626" />
                <Text style={styles.alertText}>{t('geneticRisk.familialMelanomaSyndrome')}</Text>
              </View>
            )}
            {riskProfile.has_early_onset_melanoma && (
              <View style={styles.alertItem}>
                <Ionicons name="alert-circle" size={16} color="#dc2626" />
                <Text style={styles.alertText}>{t('geneticRisk.earlyOnsetMelanoma')}</Text>
              </View>
            )}
            {riskProfile.has_multiple_family_melanomas && (
              <View style={styles.alertItem}>
                <Ionicons name="alert-circle" size={16} color="#dc2626" />
                <Text style={styles.alertText}>{t('geneticRisk.multipleFamilyMelanomas')}</Text>
              </View>
            )}
          </View>
        )}

        {/* Screening Recommendations */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>{t('geneticRisk.screeningRecommendations')}</Text>
          <View style={styles.recommendationCard}>
            <View style={styles.recommendationRow}>
              <Ionicons name="calendar" size={20} color="#8b5cf6" />
              <View style={styles.recommendationContent}>
                <Text style={styles.recommendationLabel}>{t('geneticRisk.selfExamination')}</Text>
                <Text style={styles.recommendationValue}>
                  {riskProfile.recommended_screening_frequency.charAt(0).toUpperCase() +
                   riskProfile.recommended_screening_frequency.slice(1)}
                </Text>
              </View>
            </View>
            <View style={styles.recommendationRow}>
              <Ionicons name="medkit" size={20} color="#8b5cf6" />
              <View style={styles.recommendationContent}>
                <Text style={styles.recommendationLabel}>{t('geneticRisk.professionalScreening')}</Text>
                <Text style={styles.recommendationValue}>
                  {riskProfile.recommended_professional_frequency.charAt(0).toUpperCase() +
                   riskProfile.recommended_professional_frequency.slice(1)}
                </Text>
              </View>
            </View>
            {riskProfile.genetic_counseling_recommended && (
              <View style={styles.counselingAlert}>
                <Ionicons name="information-circle" size={20} color="#3b82f6" />
                <Text style={styles.counselingText}>
                  {t('geneticRisk.geneticCounselingRecommended')}
                </Text>
              </View>
            )}
          </View>
        </View>

        {/* Risk Reduction Recommendations */}
        {riskProfile.risk_reduction_recommendations.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>{t('geneticRisk.personalizedRecommendations')}</Text>
            <View style={styles.recommendationsCard}>
              {riskProfile.risk_reduction_recommendations.map((recommendation, index) => (
                <View key={index} style={styles.recommendationItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#10b981" />
                  <Text style={styles.recommendationItemText}>{recommendation}</Text>
                </View>
              ))}
            </View>
          </View>
        )}

        {/* Generation Pattern */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>{t('geneticRisk.generationalPattern')}</Text>
          <View style={styles.generationCard}>
            {Object.entries(riskProfile.generation_pattern).map(([generation, count]) => (
              <View key={generation} style={styles.generationRow}>
                <Text style={styles.generationLabel}>
                  {generation === 'grandparents' ? t('geneticRisk.grandparents') :
                   generation === 'parents' ? t('geneticRisk.parents') :
                   generation === 'siblings' ? t('geneticRisk.siblings') :
                   generation === 'aunts_uncles' ? t('geneticRisk.auntsUncles') :
                   generation.charAt(0).toUpperCase() + generation.slice(1).replace('_', '/')}
                </Text>
                <Text style={styles.generationValue}>
                  {count} {t('geneticRisk.affected')}
                </Text>
              </View>
            ))}
          </View>
        </View>

        {/* Last Updated */}
        <Text style={styles.lastUpdated}>
          {t('geneticRisk.lastCalculated')} {new Date(riskProfile.last_calculated).toLocaleDateString()}
        </Text>

        {/* Action Buttons */}
        <TouchableOpacity
          style={styles.exportPdfButton}
          onPress={() => exportRiskProfilePDF()}
        >
          <Ionicons name="document-text" size={20} color="white" />
          <Text style={styles.exportPdfButtonText}>{t('geneticRisk.exportPdfReport')}</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.geneticTestingButton}
          onPress={() => router.push('/genetic-testing' as any)}
        >
          <Ionicons name="flask" size={20} color="white" />
          <Text style={styles.geneticTestingButtonText}>{t('geneticRisk.uploadGeneticTest')}</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.updateButton}
          onPress={() => router.push('/family-history' as any)}
        >
          <Ionicons name="people" size={20} color="white" />
          <Text style={styles.updateButtonText}>{t('geneticRisk.updateFamilyHistory')}</Text>
        </TouchableOpacity>
      </ScrollView>
    </LinearGradient>
  );

  async function exportRiskProfilePDF() {
    if (!riskProfile) return;

    Alert.alert(
      t('geneticRisk.exportPdfTitle'),
      t('geneticRisk.exportPdfMessage'),
      [
        { text: t('geneticRisk.cancel'), style: 'cancel' },
        {
          text: t('geneticRisk.generatePdf'),
          onPress: async () => {
            try {
              // In a real implementation, you would call a PDF generation service
              // For now, we'll show the structure
              Alert.alert(
                t('geneticRisk.pdfReportTitle'),
                t('geneticRisk.pdfReportContent', {
                  riskLevel: riskProfile.overall_risk_level,
                  riskScore: riskProfile.overall_genetic_risk_score.toFixed(0)
                }),
                [{ text: t('geneticRisk.ok') }]
              );
            } catch (error) {
              Alert.alert(t('geneticRisk.error'), 'Failed to generate PDF report');
            }
          }
        }
      ]
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#7c3aed',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: 'white',
  },
  errorText: {
    fontSize: 18,
    color: 'white',
    marginBottom: 20,
  },
  retryButton: {
    padding: 12,
    backgroundColor: 'white',
    borderRadius: 8,
  },
  retryButtonText: {
    color: '#7c3aed',
    fontSize: 16,
    fontWeight: 'bold',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 16,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.2)',
  },
  backButton: {
    padding: 8,
  },
  refreshButton: {
    padding: 8,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    flex: 1,
    textAlign: 'center',
  },
  overallRiskCard: {
    margin: 20,
    padding: 24,
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  overallRiskTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
    textAlign: 'center',
  },
  overallRiskContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  overallScoreCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#f9fafb',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#e5e7eb',
  },
  overallScore: {
    fontSize: 48,
    fontWeight: 'bold',
  },
  overallScoreLabel: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: -8,
  },
  overallRiskInfo: {
    flex: 1,
    marginLeft: 20,
  },
  overallRiskLevel: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  overallRiskSubtext: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
  },
  confidenceBar: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  confidenceLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 8,
  },
  confidenceProgress: {
    height: 6,
    backgroundColor: '#e5e7eb',
    borderRadius: 3,
    overflow: 'hidden',
  },
  confidenceProgressFill: {
    height: '100%',
    backgroundColor: '#8b5cf6',
  },
  section: {
    marginHorizontal: 20,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 12,
  },
  riskScoreCard: {
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  riskScoreLabel: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 8,
  },
  scoreContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 12,
  },
  scoreNumber: {
    fontSize: 36,
    fontWeight: 'bold',
  },
  scoreMax: {
    fontSize: 18,
    color: '#9ca3af',
    marginLeft: 4,
  },
  progressBarContainer: {
    height: 8,
    backgroundColor: '#e5e7eb',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressBar: {
    height: '100%',
    borderRadius: 4,
  },
  riskLevel: {
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'right',
  },
  summaryCard: {
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 12,
    padding: 16,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  summaryLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  summaryValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  lineageContainer: {
    paddingTop: 10,
  },
  lineageBadges: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 8,
  },
  lineageBadge: {
    backgroundColor: '#ede9fe',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  lineageBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#7c3aed',
  },
  alertSection: {
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 16,
    backgroundColor: '#fee2e2',
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#dc2626',
  },
  alertHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
    gap: 8,
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#991b1b',
  },
  alertItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    gap: 8,
  },
  alertText: {
    fontSize: 14,
    color: '#991b1b',
    flex: 1,
  },
  recommendationCard: {
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 12,
    padding: 16,
  },
  recommendationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
    gap: 12,
  },
  recommendationContent: {
    flex: 1,
  },
  recommendationLabel: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 4,
  },
  recommendationValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  counselingAlert: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    padding: 12,
    backgroundColor: '#dbeafe',
    borderRadius: 8,
    gap: 8,
  },
  counselingText: {
    flex: 1,
    fontSize: 14,
    fontWeight: '500',
    color: '#1e40af',
  },
  recommendationsCard: {
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 12,
    padding: 16,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
    gap: 10,
  },
  recommendationItemText: {
    flex: 1,
    fontSize: 14,
    color: '#1f2937',
    lineHeight: 20,
  },
  generationCard: {
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 12,
    padding: 16,
  },
  generationRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  generationLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  generationValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  lastUpdated: {
    textAlign: 'center',
    fontSize: 12,
    color: 'rgba(255,255,255,0.8)',
    marginBottom: 16,
  },
  exportPdfButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 20,
    marginBottom: 12,
    padding: 16,
    backgroundColor: '#dc2626',
    borderRadius: 12,
    gap: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 5,
  },
  exportPdfButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  geneticTestingButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 20,
    marginBottom: 12,
    padding: 16,
    backgroundColor: '#8b5cf6',
    borderRadius: 12,
    gap: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 5,
  },
  geneticTestingButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  updateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 20,
    marginBottom: 30,
    padding: 16,
    backgroundColor: '#10b981',
    borderRadius: 12,
    gap: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 5,
  },
  updateButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
