import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  Pressable,
  Alert,
  ActivityIndicator,
  Image,
  Dimensions,
  Platform,
  Modal
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { useTranslation } from 'react-i18next';
import * as SecureStore from 'expo-secure-store';
import AnalysisHistoryService from '../services/AnalysisHistoryService';
import { API_BASE_URL } from '../config';

const screenWidth = Dimensions.get('window').width;

export default function HistoryScreen() {
  const { t } = useTranslation();
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [showMenu, setShowMenu] = useState(false);

  // AI Explanation state - tracks which analysis item is being explained
  const [expandedAiExplanation, setExpandedAiExplanation] = useState<number | null>(null);
  const [aiExplanations, setAiExplanations] = useState<{[key: number]: string}>({});
  const [loadingAiExplanation, setLoadingAiExplanation] = useState<number | null>(null);
  const [aiExplanationErrors, setAiExplanationErrors] = useState<{[key: number]: string}>({});

  const { user, isAuthenticated, logout } = useAuth();
  const router = useRouter();

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    }
  }, [isAuthenticated]);

  const loadData = async (showRefreshing = false) => {
    try {
      if (showRefreshing) {
        setIsRefreshing(true);
      } else {
        setIsLoading(true);
      }
      setError(null);

      // Load analysis history and statistics in parallel
      const [historyData, statsData] = await Promise.all([
        AnalysisHistoryService.getAnalysisHistory(0, 50),
        AnalysisHistoryService.getAnalysisStatistics()
      ]);

      setAnalysisHistory(historyData);
      setStatistics(statsData);

    } catch (error) {
      console.error('Failed to load history data:', error);
      setError(error.message);

      if (error.message.includes('Authentication') || error.message.includes('401')) {
        Alert.alert(
          t('history.alerts.sessionExpired.title'),
          t('history.alerts.sessionExpired.message'),
          [{ text: t('common.ok'), onPress: () => logout() }]
        );
      }
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    if (isAuthenticated) {
      loadData();
    }
  }, [isAuthenticated]);

  const handleLogout = async () => {
    Alert.alert(
      t('history.alerts.logout.title'),
      t('history.alerts.logout.message', { username: user?.username || 'User' }),
      [
        { text: t('history.alerts.logout.cancel'), style: "cancel" },
        {
          text: t('history.alerts.logout.confirm'),
          style: "destructive",
          onPress: async () => {
            try {
              await logout();
              router.replace('/');
            } catch (error) {
              console.error('Logout error:', error);
            }
          }
        }
      ]
    );
  };

  // Function to fetch AI explanation for a condition
  const fetchAIExplanation = async (analysisId: number, condition: string, severity?: string) => {
    if (!condition || condition === 'Unknown') {
      setAiExplanationErrors(prev => ({ ...prev, [analysisId]: 'No condition to explain' }));
      return;
    }

    setLoadingAiExplanation(analysisId);
    setAiExplanationErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors[analysisId];
      return newErrors;
    });
    setExpandedAiExplanation(analysisId);

    try {
      const token = await SecureStore.getItemAsync('auth_token');

      const formData = new FormData();
      formData.append('condition', condition);
      if (severity) {
        formData.append('severity', severity);
      }

      const response = await fetch(`${API_BASE_URL}/ai-chat/explain-condition`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get AI explanation');
      }

      const data = await response.json();
      setAiExplanations(prev => ({ ...prev, [analysisId]: data.explanation }));
    } catch (error: any) {
      console.error('AI explanation error:', error);
      setAiExplanationErrors(prev => ({
        ...prev,
        [analysisId]: error.message || 'Failed to load explanation. Please try again.'
      }));
    } finally {
      setLoadingAiExplanation(null);
    }
  };

  const formatAnalysisForDisplay = (analysis) => {
    return AnalysisHistoryService.formatAnalysisForDisplay(analysis);
  };

  const getRiskBadgeStyle = (riskLevel) => {
    const baseStyle = { ...styles.riskBadge };
    switch (riskLevel?.toLowerCase()) {
      case 'high':
        return { ...baseStyle, backgroundColor: '#dc3545' };
      case 'medium':
        return { ...baseStyle, backgroundColor: '#ffc107', color: '#000' };
      case 'low':
        return { ...baseStyle, backgroundColor: '#28a745' };
      default:
        return { ...baseStyle, backgroundColor: '#6c757d' };
    }
  };

  if (isLoading) {
    return (
      <View style={styles.container}>
        <LinearGradient
          colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.backgroundContainer}
        />
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4299e1" />
          <Text style={styles.loadingText}>{t('history.loading.message')}</Text>
        </View>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.container}>
        <LinearGradient
          colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.backgroundContainer}
        />
        <View style={styles.errorContainer}>
          <Text style={styles.errorTitle}>{t('history.error.title')}</Text>
          <Text style={styles.errorText}>{error}</Text>
          <Pressable style={styles.retryButton} onPress={() => loadData()}>
            <Text style={styles.retryButtonText}>{t('history.error.retryButton')}</Text>
          </Pressable>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.backgroundContainer}
      />

      {/* Header */}
      <View style={styles.header}>
        <Pressable
          style={styles.backButton}
          onPress={() => router.push('/home')}
          android_ripple={{ color: 'rgba(255,255,255,0.3)' }}
        >
          <Text style={styles.backButtonText}>{t('history.header.backButton')}</Text>
        </Pressable>

        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>{t('history.header.title')}</Text>
          <Text style={styles.headerSubtitle}>{t('history.header.subtitle')}</Text>
        </View>

        <Pressable
          style={styles.menuButton}
          onPress={() => setShowMenu(true)}
          android_ripple={{ color: 'rgba(255,255,255,0.3)' }}
        >
          <Text style={styles.menuButtonText}>☰</Text>
        </Pressable>
      </View>

      <ScrollView
        style={styles.scrollContainer}
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl
            refreshing={isRefreshing}
            onRefresh={() => loadData(true)}
            colors={['#4299e1']}
            tintColor="#4299e1"
          />
        }
      >
        {/* Statistics Card */}
        {statistics && (
          <View style={styles.statsCard}>
            <Text style={styles.statsTitle}>{t('history.statistics.title')}</Text>
            <View style={styles.statsGrid}>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{statistics.total_analyses}</Text>
                <Text style={styles.statLabel}>{t('history.statistics.totalAnalyses')}</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{statistics.lesion_detections}</Text>
                <Text style={styles.statLabel}>{t('history.statistics.lesionsDetected')}</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{Math.round(statistics.average_confidence * 100)}%</Text>
                <Text style={styles.statLabel}>{t('history.statistics.avgConfidence')}</Text>
              </View>
            </View>
          </View>
        )}

        {/* Analysis History */}
        <View style={styles.historySection}>
          <Text style={styles.sectionTitle}>{t('history.recentAnalyses.title')}</Text>

          {analysisHistory.length === 0 ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyStateText}>{t('history.emptyState.title')}</Text>
              <Text style={styles.emptyStateSubtext}>
                {t('history.emptyState.subtitle')}
              </Text>
              <Pressable
                style={styles.startAnalysisButton}
                onPress={() => router.push('/home')}
              >
                <Text style={styles.startAnalysisButtonText}>{t('history.emptyState.button')}</Text>
              </Pressable>
            </View>
          ) : (
            analysisHistory.map((analysis) => {
              const formatted = formatAnalysisForDisplay(analysis);
              return (
                <Pressable
                  key={analysis.id}
                  style={styles.analysisCard}
                  onPress={() => router.push(`/analysis/${analysis.id}`)}
                >
                  <View style={styles.analysisHeader}>
                    <View style={styles.analysisInfo}>
                      <Text style={styles.analysisDate}>{formatted.createdAt.date}</Text>
                      <Text style={styles.analysisTime}>{formatted.createdAt.time}</Text>
                    </View>
                    <View style={getRiskBadgeStyle(formatted.riskLevel)}>
                      <Text style={styles.riskBadgeText}>{formatted.riskLevel?.toUpperCase()}</Text>
                    </View>
                  </View>

                  {/* Image Display */}
                  {analysis.image_url && (
                    <View style={styles.imageContainer}>
                      <Image
                        source={{ uri: `${API_BASE_URL}${analysis.image_url}` }}
                        style={styles.analysisImage}
                        resizeMode="cover"
                      />
                    </View>
                  )}

                  <View style={styles.analysisBody}>
                    <Text style={styles.analysisResult}>
                      {formatted.isLesion ? t('history.analysisCard.lesionDetected') : t('history.analysisCard.noLesion')}
                    </Text>
                    {formatted.predictedClass && (
                      <Text style={styles.diagnosisText}>{formatted.predictedClass}</Text>
                    )}

                    {/* Inflammatory Condition */}
                    {analysis.inflammatory_condition && (
                      <Text style={styles.secondaryDiagnosisText}>
                        🔥 {analysis.inflammatory_condition}
                      </Text>
                    )}

                    {/* Infectious Disease */}
                    {analysis.infectious_disease && (
                      <Text style={styles.secondaryDiagnosisText}>
                        🦠 {analysis.infectious_disease}
                        {analysis.contagious && <Text style={styles.warningText}> {t('history.analysisCard.contagious')}</Text>}
                      </Text>
                    )}

                    <View style={styles.confidenceRow}>
                      <Text style={styles.confidenceLabel}>{t('history.analysisCard.confidence')}</Text>
                      <Text style={[styles.confidenceValue, { color: formatted.confidenceLevel.color }]}>
                        {Math.round((formatted.confidence || 0) * 100)}% ({formatted.confidenceLevel.level})
                      </Text>
                    </View>

                    {formatted.recommendation && (
                      <Text style={styles.recommendationText}>{formatted.recommendation}</Text>
                    )}

                    {/* AI Explanation Button */}
                    {formatted.predictedClass && (
                      <Pressable
                        style={[
                          styles.learnMoreButton,
                          expandedAiExplanation === analysis.id && styles.learnMoreButtonActive
                        ]}
                        onPress={(e) => {
                          e.stopPropagation();
                          if (expandedAiExplanation === analysis.id) {
                            setExpandedAiExplanation(null);
                          } else {
                            const condition = formatted.predictedClass || analysis.inflammatory_condition || analysis.infectious_disease;
                            fetchAIExplanation(analysis.id, condition, formatted.riskLevel);
                          }
                        }}
                      >
                        <Ionicons
                          name={expandedAiExplanation === analysis.id ? "chevron-up-circle" : "information-circle"}
                          size={18}
                          color="#667eea"
                          style={{ marginRight: 6 }}
                        />
                        <Text style={styles.learnMoreButtonText}>
                          {expandedAiExplanation === analysis.id ? 'Hide' : 'Learn More'}
                        </Text>
                        {loadingAiExplanation === analysis.id && (
                          <ActivityIndicator size="small" color="#667eea" style={{ marginLeft: 6 }} />
                        )}
                      </Pressable>
                    )}

                    {/* AI Explanation Content */}
                    {expandedAiExplanation === analysis.id && (
                      <View style={styles.aiExplanationContent}>
                        {loadingAiExplanation === analysis.id ? (
                          <View style={styles.aiExplanationLoading}>
                            <ActivityIndicator size="small" color="#667eea" />
                            <Text style={styles.aiExplanationLoadingText}>Getting AI explanation...</Text>
                          </View>
                        ) : aiExplanationErrors[analysis.id] ? (
                          <View style={styles.aiExplanationError}>
                            <Ionicons name="alert-circle" size={20} color="#e53e3e" />
                            <Text style={styles.aiExplanationErrorText}>{aiExplanationErrors[analysis.id]}</Text>
                            <Pressable
                              style={styles.retryButton}
                              onPress={(e) => {
                                e.stopPropagation();
                                const condition = formatted.predictedClass || analysis.inflammatory_condition || analysis.infectious_disease;
                                fetchAIExplanation(analysis.id, condition, formatted.riskLevel);
                              }}
                            >
                              <Text style={styles.retryButtonText}>Try Again</Text>
                            </Pressable>
                          </View>
                        ) : aiExplanations[analysis.id] ? (
                          <View style={styles.aiExplanationText}>
                            <View style={styles.aiExplanationHeader}>
                              <Ionicons name="sparkles" size={16} color="#667eea" />
                              <Text style={styles.aiExplanationTitle}>About {formatted.predictedClass}</Text>
                            </View>
                            <Text style={styles.aiExplanationBody}>{aiExplanations[analysis.id]}</Text>
                            <View style={styles.aiExplanationDisclaimer}>
                              <Ionicons name="information-circle-outline" size={12} color="#718096" />
                              <Text style={styles.aiExplanationDisclaimerText}>
                                For educational purposes only. Consult a healthcare provider for medical advice.
                              </Text>
                            </View>
                          </View>
                        ) : null}
                      </View>
                    )}
                  </View>

                  <View style={styles.analysisFooter}>
                    <Text style={styles.analysisMetadata}>
                      {t('history.analysisCard.processing')} {formatted.processingTime?.toFixed(2) || 0}s • {formatted.modelVersion}
                    </Text>
                    <View style={styles.analysisActions}>
                      <Pressable
                        style={styles.trackButton}
                        onPress={(e) => {
                          e.stopPropagation();
                          router.push(`/create-lesion-group?analysis_id=${analysis.id}` as any);
                        }}
                      >
                        <Text style={styles.trackButtonText}>{t('history.analysisCard.trackButton')}</Text>
                      </Pressable>
                      <Pressable
                        style={styles.viewDetailsButton}
                        onPress={(e) => {
                          e.stopPropagation();
                          router.push(`/analysis/${analysis.id}`);
                        }}
                      >
                        <Text style={styles.viewDetailsButtonText}>{t('history.analysisCard.detailsButton')}</Text>
                      </Pressable>
                    </View>
                  </View>
                </Pressable>
              );
            })
          )}
        </View>

        {/* Navigation Button */}
        <View style={styles.navigationSection}>
          <Pressable
            style={styles.navigationButton}
            onPress={() => router.push('/home')}
          >
            <Text style={styles.navigationButtonText}>{t('history.navigation.newAnalysis')}</Text>
          </Pressable>
        </View>
      </ScrollView>

      {/* Navigation Menu Modal */}
      <Modal
        visible={showMenu}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowMenu(false)}
      >
        <Pressable
          style={styles.menuOverlay}
          onPress={() => setShowMenu(false)}
        >
          <View style={styles.menuContainer}>
            <View style={styles.menuHeader}>
              <Text style={styles.menuTitle}>{t('history.menu.title')}</Text>
              <Pressable onPress={() => setShowMenu(false)}>
                <Text style={styles.menuCloseButton}>✕</Text>
              </Pressable>
            </View>

            <ScrollView style={styles.menuContent}>
              {/* Home */}
              <Pressable
                style={styles.menuItem}
                onPress={() => {
                  setShowMenu(false);
                  router.push('/home');
                }}
              >
                <Text style={styles.menuItemIcon}>🏠</Text>
                <View style={styles.menuItemTextContainer}>
                  <Text style={styles.menuItemText}>{t('history.menu.home.title')}</Text>
                  <Text style={styles.menuItemSubtext}>{t('history.menu.home.subtitle')}</Text>
                </View>
              </Pressable>

              {/* Track Lesions */}
              <Pressable
                style={styles.menuItem}
                onPress={() => {
                  setShowMenu(false);
                  router.push('/lesion-tracking');
                }}
              >
                <Text style={styles.menuItemIcon}>🔍</Text>
                <View style={styles.menuItemTextContainer}>
                  <Text style={styles.menuItemText}>{t('history.menu.trackLesions.title')}</Text>
                  <Text style={styles.menuItemSubtext}>{t('history.menu.trackLesions.subtitle')}</Text>
                </View>
              </Pressable>

              {/* Family History */}
              <Pressable
                style={styles.menuItem}
                onPress={() => {
                  setShowMenu(false);
                  router.push('/family-history');
                }}
              >
                <Text style={styles.menuItemIcon}>🧬</Text>
                <View style={styles.menuItemTextContainer}>
                  <Text style={styles.menuItemText}>{t('history.menu.familyHistory.title')}</Text>
                  <Text style={styles.menuItemSubtext}>{t('history.menu.familyHistory.subtitle')}</Text>
                </View>
              </Pressable>

              {/* Predictive Analytics */}
              <Pressable
                style={styles.menuItem}
                onPress={() => {
                  setShowMenu(false);
                  router.push('/analytics');
                }}
              >
                <Text style={styles.menuItemIcon}>📊</Text>
                <View style={styles.menuItemTextContainer}>
                  <Text style={styles.menuItemText}>{t('history.menu.analytics.title')}</Text>
                  <Text style={styles.menuItemSubtext}>{t('history.menu.analytics.subtitle')}</Text>
                </View>
              </Pressable>

              {/* Sun Exposure Tracking */}
              <Pressable
                style={styles.menuItem}
                onPress={() => {
                  setShowMenu(false);
                  router.push('/sun-exposure');
                }}
              >
                <Text style={styles.menuItemIcon}>☀️</Text>
                <View style={styles.menuItemTextContainer}>
                  <Text style={styles.menuItemText}>{t('history.menu.sunExposure.title')}</Text>
                  <Text style={styles.menuItemSubtext}>{t('history.menu.sunExposure.subtitle')}</Text>
                </View>
              </Pressable>

              {/* Treatment Monitoring */}
              <Pressable
                style={styles.menuItem}
                onPress={() => {
                  setShowMenu(false);
                  router.push('/treatment-monitoring');
                }}
              >
                <Text style={styles.menuItemIcon}>💊</Text>
                <View style={styles.menuItemTextContainer}>
                  <Text style={styles.menuItemText}>{t('history.menu.treatmentMonitoring.title')}</Text>
                  <Text style={styles.menuItemSubtext}>{t('history.menu.treatmentMonitoring.subtitle')}</Text>
                </View>
              </Pressable>

              {/* Dermatologist Integration */}
              <Pressable
                style={styles.menuItem}
                onPress={() => {
                  setShowMenu(false);
                  router.push('/dermatologist-integration');
                }}
              >
                <Text style={styles.menuItemIcon}>👨‍⚕️</Text>
                <View style={styles.menuItemTextContainer}>
                  <Text style={styles.menuItemText}>{t('history.menu.dermatologist.title')}</Text>
                  <Text style={styles.menuItemSubtext}>{t('history.menu.dermatologist.subtitle')}</Text>
                </View>
              </Pressable>

              {/* Help & Guide */}
              <Pressable
                style={styles.menuItem}
                onPress={() => {
                  setShowMenu(false);
                  router.push('/help');
                }}
              >
                <Text style={styles.menuItemIcon}>📖</Text>
                <View style={styles.menuItemTextContainer}>
                  <Text style={styles.menuItemText}>{t('history.menu.help.title')}</Text>
                  <Text style={styles.menuItemSubtext}>{t('history.menu.help.subtitle')}</Text>
                </View>
              </Pressable>

              {/* Divider before Logout */}
              <View style={styles.menuDivider} />

              {/* Logout */}
              <Pressable
                style={[styles.menuItem, styles.menuItemLogout]}
                onPress={() => {
                  setShowMenu(false);
                  handleLogout();
                }}
              >
                <Text style={styles.menuItemIcon}>🚪</Text>
                <View style={styles.menuItemTextContainer}>
                  <Text style={[styles.menuItemText, styles.menuItemLogoutText]}>{t('history.menu.logout.title')}</Text>
                  <Text style={styles.menuItemSubtext}>{t('history.menu.logout.subtitle')}</Text>
                </View>
              </Pressable>
            </ScrollView>
          </View>
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  backgroundContainer: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#4a5568',
    textAlign: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#e53e3e',
    marginBottom: 12,
    textAlign: 'center',
  },
  errorText: {
    fontSize: 16,
    color: '#4a5568',
    textAlign: 'center',
    marginBottom: 20,
    lineHeight: 22,
  },
  retryButton: {
    backgroundColor: '#4299e1',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 16,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.2)',
  },
  backButton: {
    backgroundColor: 'rgba(66, 153, 225, 0.9)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#4299e1',
    marginRight: 15,
  },
  backButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  headerContent: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#4a5568',
    marginTop: 4,
  },
  scrollContainer: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  statsCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 16,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#4299e1',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#4a5568',
    textAlign: 'center',
  },
  historySection: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 16,
  },
  emptyState: {
    alignItems: 'center',
    padding: 40,
    backgroundColor: 'white',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  emptyStateText: {
    fontSize: 24,
    color: '#4a5568',
    marginBottom: 8,
  },
  emptyStateSubtext: {
    fontSize: 16,
    color: '#718096',
    textAlign: 'center',
    marginBottom: 24,
  },
  startAnalysisButton: {
    backgroundColor: '#4299e1',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  startAnalysisButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  analysisCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 3,
  },
  analysisHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  analysisInfo: {
    flex: 1,
  },
  analysisDate: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2d3748',
  },
  analysisTime: {
    fontSize: 12,
    color: '#718096',
    marginTop: 2,
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  riskBadgeText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
  },
  imageContainer: {
    marginVertical: 12,
    alignItems: 'center',
    backgroundColor: '#f7fafc',
    borderRadius: 8,
    padding: 8,
  },
  analysisImage: {
    width: screenWidth - 80,
    height: 200,
    borderRadius: 8,
    backgroundColor: '#e2e8f0',
  },
  analysisBody: {
    marginBottom: 12,
  },
  analysisResult: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2d3748',
    marginBottom: 6,
  },
  diagnosisText: {
    fontSize: 14,
    color: '#4a5568',
    marginBottom: 8,
  },
  secondaryDiagnosisText: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 6,
    fontStyle: 'italic',
  },
  warningText: {
    color: '#dc3545',
    fontWeight: 'bold',
    fontStyle: 'normal',
  },
  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  confidenceLabel: {
    fontSize: 14,
    color: '#4a5568',
    marginRight: 8,
  },
  confidenceValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  recommendationText: {
    fontSize: 13,
    color: '#4a5568',
    fontStyle: 'italic',
    lineHeight: 18,
  },
  analysisFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    paddingTop: 12,
  },
  analysisMetadata: {
    fontSize: 11,
    color: '#718096',
    flex: 1,
  },
  analysisActions: {
    flexDirection: 'row',
    gap: 8,
  },
  trackButton: {
    backgroundColor: '#10b981',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  trackButtonText: {
    color: 'white',
    fontSize: 11,
    fontWeight: '600',
  },
  viewDetailsButton: {
    backgroundColor: '#4299e1',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  viewDetailsButtonText: {
    color: 'white',
    fontSize: 11,
    fontWeight: '600',
  },
  navigationSection: {
    marginTop: 20,
    alignItems: 'center',
  },
  navigationButton: {
    backgroundColor: '#38a169',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  navigationButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  menuButton: {
    backgroundColor: 'rgba(59, 130, 246, 0.9)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#3b82f6',
    marginLeft: 15,
  },
  menuButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  menuOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  menuContainer: {
    backgroundColor: '#ffffff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '80%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.25,
    shadowRadius: 12,
    elevation: 10,
  },
  menuHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  menuTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  menuCloseButton: {
    fontSize: 28,
    color: '#6b7280',
    fontWeight: '300',
  },
  menuContent: {
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  menuItemIcon: {
    fontSize: 28,
    marginRight: 16,
  },
  menuItemTextContainer: {
    flex: 1,
  },
  menuItemText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  menuItemSubtext: {
    fontSize: 13,
    color: '#6b7280',
  },
  menuDivider: {
    height: 1,
    backgroundColor: '#e5e7eb',
    marginVertical: 8,
  },
  menuItemLogout: {
    backgroundColor: '#fee2e2',
    borderWidth: 1,
    borderColor: '#fecaca',
  },
  menuItemLogoutText: {
    color: '#dc2626',
  },

  // AI Explanation Styles
  learnMoreButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    paddingHorizontal: 14,
    marginTop: 12,
    backgroundColor: 'rgba(102, 126, 234, 0.08)',
    borderRadius: 8,
  },
  learnMoreButtonActive: {
    backgroundColor: 'rgba(102, 126, 234, 0.15)',
  },
  learnMoreButtonText: {
    color: '#667eea',
    fontSize: 14,
    fontWeight: '600',
  },
  aiExplanationContent: {
    marginTop: 12,
    padding: 12,
    backgroundColor: 'rgba(102, 126, 234, 0.05)',
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#667eea',
  },
  aiExplanationLoading: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
  },
  aiExplanationLoadingText: {
    marginLeft: 8,
    color: '#667eea',
    fontSize: 13,
  },
  aiExplanationError: {
    alignItems: 'center',
    paddingVertical: 12,
  },
  aiExplanationErrorText: {
    color: '#e53e3e',
    fontSize: 13,
    marginTop: 6,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 10,
    paddingVertical: 6,
    paddingHorizontal: 16,
    backgroundColor: '#667eea',
    borderRadius: 6,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 13,
    fontWeight: '600',
  },
  aiExplanationText: {
    // Container for explanation text
  },
  aiExplanationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  aiExplanationTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#2d3748',
    marginLeft: 6,
  },
  aiExplanationBody: {
    fontSize: 13,
    lineHeight: 20,
    color: '#4a5568',
  },
  aiExplanationDisclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginTop: 12,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0, 0, 0, 0.06)',
  },
  aiExplanationDisclaimerText: {
    flex: 1,
    marginLeft: 4,
    fontSize: 10,
    color: '#718096',
    lineHeight: 14,
  },
});