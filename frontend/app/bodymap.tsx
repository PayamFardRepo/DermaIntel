import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  Pressable,
  Alert
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { useTranslation } from 'react-i18next';
import AnalysisHistoryService from '../services/AnalysisHistoryService';

export default function BodyMapOverviewScreen() {
  const { t } = useTranslation();
  const [analyses, setAnalyses] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [groupBy, setGroupBy] = useState<'location' | 'date'>('location');
  const { user, isAuthenticated, logout } = useAuth();
  const router = useRouter();

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    }
  }, [isAuthenticated]);

  useEffect(() => {
    if (isAuthenticated) {
      loadBodyMapData();
    }
  }, [isAuthenticated]);

  const loadBodyMapData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const historyData = await AnalysisHistoryService.getAnalysisHistory();

      // Filter only analyses with body location data
      const analysesWithLocation = historyData.filter(
        (a: any) => a.body_location || a.body_map_coordinates
      );

      setAnalyses(analysesWithLocation);
    } catch (error: any) {
      console.error('Failed to load body map data:', error);
      setError(error.message);

      if (error.message.includes('Authentication') || error.message.includes('401')) {
        Alert.alert(
          t('bodymap.alerts.sessionExpired.title'),
          t('bodymap.alerts.sessionExpired.message'),
          [{ text: t('common.ok'), onPress: () => logout() }]
        );
      }
    } finally {
      setIsLoading(false);
    }
  };

  const groupAnalysesByLocation = () => {
    const grouped: { [key: string]: any[] } = {};

    analyses.forEach((analysis) => {
      const location = analysis.body_location || t('bodymap.location.unknown');
      if (!grouped[location]) {
        grouped[location] = [];
      }
      grouped[location].push(analysis);
    });

    return grouped;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const handleAnalysisPress = (analysisId: number) => {
    router.push(`/analysis/${analysisId}`);
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
          <Text style={styles.loadingText}>{t('bodymap.loading.message')}</Text>
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
          <Text style={styles.errorText}>{t('bodymap.error.title')}</Text>
          <Text style={styles.errorMessage}>{error}</Text>
          <Pressable style={styles.retryButton} onPress={loadBodyMapData}>
            <Text style={styles.retryButtonText}>{t('bodymap.error.retryButton')}</Text>
          </Pressable>
          <Pressable style={styles.backButton} onPress={() => router.back()}>
            <Text style={styles.backButtonText}>{t('bodymap.error.backButton')}</Text>
          </Pressable>
        </View>
      </View>
    );
  }

  const groupedAnalyses = groupBy === 'location' ? groupAnalysesByLocation() : {};

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
        <Pressable style={styles.backHeaderButton} onPress={() => router.back()}>
          <Text style={styles.backHeaderButtonText}>{t('bodymap.header.backButton')}</Text>
        </Pressable>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>{t('bodymap.header.title')}</Text>
          <Text style={styles.headerSubtitle}>{t('bodymap.header.subtitle')}</Text>
        </View>
      </View>

      <ScrollView style={styles.scrollContainer} contentContainerStyle={styles.scrollContent}>
        {/* Summary Card */}
        <View style={styles.summaryCard}>
          <Text style={styles.summaryTitle}>{t('bodymap.summary.title')}</Text>
          <Text style={styles.summaryCount}>{analyses.length}</Text>
          <Text style={styles.summarySubtitle}>
            {Object.keys(groupedAnalyses).length} {t('bodymap.summary.regionsLabel')}
          </Text>
        </View>

        {analyses.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyStateIcon}>📍</Text>
            <Text style={styles.emptyStateTitle}>{t('bodymap.emptyState.title')}</Text>
            <Text style={styles.emptyStateText}>
              {t('bodymap.emptyState.message')}
            </Text>
            <Pressable style={styles.goHomeButton} onPress={() => router.push('/home')}>
              <Text style={styles.goHomeButtonText}>{t('bodymap.emptyState.button')}</Text>
            </Pressable>
          </View>
        ) : (
          <>
            {/* Group by toggle */}
            <View style={styles.toggleContainer}>
              <Text style={styles.toggleLabel}>{t('bodymap.groupBy.label')}</Text>
              <View style={styles.toggleButtons}>
                <Pressable
                  style={[styles.toggleButton, groupBy === 'location' && styles.toggleButtonActive]}
                  onPress={() => setGroupBy('location')}
                >
                  <Text style={[styles.toggleButtonText, groupBy === 'location' && styles.toggleButtonTextActive]}>
                    {t('bodymap.groupBy.location')}
                  </Text>
                </Pressable>
                <Pressable
                  style={[styles.toggleButton, groupBy === 'date' && styles.toggleButtonActive]}
                  onPress={() => setGroupBy('date')}
                >
                  <Text style={[styles.toggleButtonText, groupBy === 'date' && styles.toggleButtonTextActive]}>
                    {t('bodymap.groupBy.date')}
                  </Text>
                </Pressable>
              </View>
            </View>

            {/* Grouped Analyses */}
            {groupBy === 'location' ? (
              Object.entries(groupedAnalyses).map(([location, locationAnalyses]) => (
                <View key={location} style={styles.locationGroup}>
                  <Text style={styles.locationGroupTitle}>
                    📍 {location.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                  </Text>
                  <Text style={styles.locationGroupCount}>
                    {locationAnalyses.length} {locationAnalyses.length === 1 ? t('bodymap.analysisCount.singular') : t('bodymap.analysisCount.plural')}
                  </Text>

                  {locationAnalyses.map((analysis: any) => (
                    <Pressable
                      key={analysis.id}
                      style={styles.analysisCard}
                      onPress={() => handleAnalysisPress(analysis.id)}
                    >
                      <View style={styles.analysisCardHeader}>
                        <Text style={styles.analysisDate}>{formatDate(analysis.created_at)}</Text>
                        <View style={[styles.riskBadge, styles[`risk${analysis.risk_level?.replace(/\b\w/g, (l: string) => l.toUpperCase())}`]]}>
                          <Text style={styles.riskBadgeText}>
                            {analysis.risk_level?.toUpperCase() || t('bodymap.riskLevel.notAvailable')}
                          </Text>
                        </View>
                      </View>

                      {analysis.predicted_class && (
                        <Text style={styles.analysisDiagnosis}>
                          {analysis.predicted_class}
                        </Text>
                      )}

                      {analysis.body_sublocation && (
                        <Text style={styles.analysisSublocation}>
                          {analysis.body_sublocation.replace(/_/g, ' ')} • {analysis.body_side || 'center'}
                        </Text>
                      )}

                      <Text style={styles.viewDetailsText}>{t('bodymap.analysisCard.tapToView')}</Text>
                    </Pressable>
                  ))}
                </View>
              ))
            ) : (
              // Date view
              <View style={styles.locationGroup}>
                {analyses.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()).map((analysis: any) => (
                  <Pressable
                    key={analysis.id}
                    style={styles.analysisCard}
                    onPress={() => handleAnalysisPress(analysis.id)}
                  >
                    <View style={styles.analysisCardHeader}>
                      <Text style={styles.analysisDate}>{formatDate(analysis.created_at)}</Text>
                      <View style={[styles.riskBadge, styles[`risk${analysis.risk_level?.replace(/\b\w/g, (l: string) => l.toUpperCase())}`]]}>
                        <Text style={styles.riskBadgeText}>
                          {analysis.risk_level?.toUpperCase() || t('bodymap.riskLevel.notAvailable')}
                        </Text>
                      </View>
                    </View>

                    <Text style={styles.analysisLocation}>
                      📍 {analysis.body_location?.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()) || t('bodymap.location.unknown')}
                    </Text>

                    {analysis.predicted_class && (
                      <Text style={styles.analysisDiagnosis}>
                        {analysis.predicted_class}
                      </Text>
                    )}

                    <Text style={styles.viewDetailsText}>{t('bodymap.analysisCard.tapToView')}</Text>
                  </Pressable>
                ))}
              </View>
            )}
          </>
        )}
      </ScrollView>
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
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: 50,
    paddingHorizontal: 16,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
  },
  backHeaderButton: {
    padding: 8,
    marginRight: 12,
  },
  backHeaderButtonText: {
    fontSize: 16,
    color: '#4299e1',
    fontWeight: '600',
  },
  headerContent: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2d3748',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#718096',
  },
  scrollContainer: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#64748b',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#dc2626',
    marginBottom: 8,
  },
  errorMessage: {
    fontSize: 14,
    color: '#64748b',
    textAlign: 'center',
    marginBottom: 20,
  },
  retryButton: {
    backgroundColor: '#4299e1',
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 8,
    marginBottom: 12,
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  backButton: {
    backgroundColor: '#f1f5f9',
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 8,
  },
  backButtonText: {
    color: '#64748b',
    fontSize: 16,
    fontWeight: '600',
  },
  summaryCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    marginBottom: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  summaryTitle: {
    fontSize: 16,
    color: '#64748b',
    marginBottom: 8,
  },
  summaryCount: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#4299e1',
    marginBottom: 4,
  },
  summarySubtitle: {
    fontSize: 14,
    color: '#94a3b8',
  },
  emptyState: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 40,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  emptyStateIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyStateTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2d3748',
    marginBottom: 8,
  },
  emptyStateText: {
    fontSize: 14,
    color: '#64748b',
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 20,
  },
  goHomeButton: {
    backgroundColor: '#4299e1',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  goHomeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  toggleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    backgroundColor: '#fff',
    padding: 12,
    borderRadius: 12,
  },
  toggleLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#475569',
    marginRight: 12,
  },
  toggleButtons: {
    flexDirection: 'row',
    flex: 1,
    gap: 8,
  },
  toggleButton: {
    flex: 1,
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
    backgroundColor: '#f1f5f9',
    alignItems: 'center',
  },
  toggleButtonActive: {
    backgroundColor: '#4299e1',
  },
  toggleButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#64748b',
  },
  toggleButtonTextActive: {
    color: '#fff',
  },
  locationGroup: {
    marginBottom: 24,
  },
  locationGroupTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2d3748',
    marginBottom: 4,
  },
  locationGroupCount: {
    fontSize: 14,
    color: '#64748b',
    marginBottom: 12,
  },
  analysisCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  analysisCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  analysisDate: {
    fontSize: 14,
    color: '#64748b',
    fontWeight: '600',
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  riskHigh: {
    backgroundColor: '#fecaca',
  },
  riskMedium: {
    backgroundColor: '#fed7aa',
  },
  riskLow: {
    backgroundColor: '#bbf7d0',
  },
  riskBadgeText: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  analysisLocation: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2d3748',
    marginBottom: 4,
  },
  analysisDiagnosis: {
    fontSize: 15,
    color: '#4b5563',
    marginBottom: 4,
  },
  analysisSublocation: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 8,
  },
  viewDetailsText: {
    fontSize: 13,
    color: '#4299e1',
    fontWeight: '600',
  },
});
