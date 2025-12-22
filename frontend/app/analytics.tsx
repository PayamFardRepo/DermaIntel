import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  Alert,
  ActivityIndicator,
  Platform,
  RefreshControl
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { router } from 'expo-router';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';

interface Forecast {
  id: number;
  lesion_name: string;
  growth_trend: string;
  growth_rate_mm_per_month: number;
  current_size_mm: number;
  predicted_size_90d: number;
  current_risk_level: string;
  predicted_risk_level_90d: string;
  risk_escalation_probability: number;
  change_probability: number;
  recommended_action: string;
  next_check_date: string;
  monitoring_frequency: string;
  confidence_score: number;
  primary_risk_factors: string[];
  forecast_data: any;
}

interface ScheduleItem {
  id: number;
  schedule_type: string;
  priority: number;
  title: string;
  description: string;
  recommended_date: string;
  is_completed: boolean;
  is_overdue: boolean;
  is_recurring: boolean;
  recurrence_frequency: string;
  based_on_risk_level: string;
  based_on_genetic_risk: boolean;
  based_on_lesion_changes: boolean;
  related_entity_id: number;
}

interface RiskTrend {
  snapshot_date: string;
  overall_risk_score: number;
  overall_risk_level: string;
  melanoma_risk_score: number;
  total_lesions_tracked: number;
  high_risk_lesions_count: number;
  genetic_risk_score: number;
  risk_trend: string;
  predicted_future_risk: number;
}

interface AnalysisStats {
  total_analyses: number;
  lesion_detections: number;
  non_lesion_detections: number;
  most_common_diagnosis: string | null;
  average_confidence: number;
  monthly_analysis_counts: { [key: string]: number };
}

interface AccuracyStats {
  total_with_biopsy: number;
  exact_matches: number;
  category_matches: number;
  mismatches: number;
  overall_accuracy: number;
  category_accuracy: number;
  breakdown_by_prediction: { [key: string]: { correct: number; total: number } };
}

interface LesionTrends {
  has_history: boolean;
  total_checks: number;
  current: {
    check_id: string;
    date: string;
    total_lesions: number;
    high_risk_count: number;
  };
  change: {
    absolute: number;
    percent: number;
    trend: string;
  };
  history: Array<{
    date: string;
    total_lesions: number;
    high_risk: number;
  }>;
  recommendation: string;
  message?: string;
}

type TabType = 'stats' | 'accuracy' | 'lesions' | 'forecasts' | 'schedule' | 'trends';

export default function PredictiveAnalyticsDashboard() {
  const { t } = useTranslation();
  const [forecasts, setForecasts] = useState<Forecast[]>([]);
  const [schedule, setSchedule] = useState<ScheduleItem[]>([]);
  const [riskTrends, setRiskTrends] = useState<RiskTrend[]>([]);
  const [analysisStats, setAnalysisStats] = useState<AnalysisStats | null>(null);
  const [accuracyStats, setAccuracyStats] = useState<AccuracyStats | null>(null);
  const [lesionTrends, setLesionTrends] = useState<LesionTrends | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState<TabType>('stats');

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setIsLoading(true);
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert(t('analytics.common.error'), t('analytics.common.loginAgain'));
        router.push('/login');
        return;
      }

      // Load all dashboard data in parallel
      const [forecastsRes, scheduleRes, trendsRes, statsRes, accuracyRes, lesionTrendsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/analytics/forecasts`, {
          headers: { 'Authorization': `Bearer ${token}` }
        }),
        fetch(`${API_BASE_URL}/analytics/schedule`, {
          headers: { 'Authorization': `Bearer ${token}` }
        }),
        fetch(`${API_BASE_URL}/analytics/risk-trends`, {
          headers: { 'Authorization': `Bearer ${token}` }
        }),
        fetch(`${API_BASE_URL}/analysis/stats`, {
          headers: { 'Authorization': `Bearer ${token}` }
        }),
        fetch(`${API_BASE_URL}/analysis/accuracy/stats`, {
          headers: { 'Authorization': `Bearer ${token}` }
        }),
        fetch(`${API_BASE_URL}/batch/lesion-trends`, {
          headers: { 'Authorization': `Bearer ${token}` }
        })
      ]);

      if (forecastsRes.ok) {
        const data = await forecastsRes.json();
        setForecasts(data);
      }

      if (scheduleRes.ok) {
        const data = await scheduleRes.json();
        setSchedule(data);
      }

      if (trendsRes.ok) {
        const data = await trendsRes.json();
        setRiskTrends(data);
      }

      if (statsRes.ok) {
        const data = await statsRes.json();
        setAnalysisStats(data);
      }

      if (accuracyRes.ok) {
        const data = await accuracyRes.json();
        setAccuracyStats(data);
      }

      if (lesionTrendsRes.ok) {
        const data = await lesionTrendsRes.json();
        setLesionTrends(data);
      }
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      Alert.alert(t('analytics.common.error'), t('analytics.common.loadFailed'));
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadDashboardData();
  };

  const generateSchedule = async () => {
    try {
      const token = AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/analytics/schedule/generate`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(t('analytics.common.success'), t('analytics.schedule.generateSuccess', { count: data.schedules_created }));
        loadDashboardData();
      } else {
        Alert.alert(t('analytics.common.error'), t('analytics.schedule.generateFailed'));
      }
    } catch (error) {
      Alert.alert(t('analytics.common.error'), t('analytics.schedule.generateFailed'));
    }
  };

  const completeScheduleItem = async (scheduleId: number) => {
    Alert.alert(
      t('analytics.schedule.completeTitle'),
      t('analytics.schedule.completeConfirm'),
      [
        { text: t('analytics.common.cancel'), style: 'cancel' },
        {
          text: t('analytics.schedule.markComplete'),
          onPress: async () => {
            try {
              const token = AuthService.getToken();
              const formData = new FormData();
              formData.append('completion_result', 'normal');

              const response = await fetch(`${API_BASE_URL}/analytics/schedule/${scheduleId}/complete`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: formData
              });

              if (response.ok) {
                Alert.alert(t('analytics.common.success'), t('analytics.schedule.completeSuccess'));
                loadDashboardData();
              }
            } catch (error) {
              Alert.alert(t('analytics.common.error'), t('analytics.schedule.completeFailed'));
            }
          }
        }
      ]
    );
  };

  const getGrowthTrendColor = (trend: string) => {
    switch (trend) {
      case 'rapid_growth': return '#dc2626';
      case 'moderate_growth': return '#f59e0b';
      case 'slow_growth': return '#eab308';
      case 'stable': return '#10b981';
      case 'shrinking': return '#3b82f6';
      default: return '#6b7280';
    }
  };

  const getGrowthTrendLabel = (trend: string) => {
    return trend.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return '#dc2626';
      case 'medium': return '#f59e0b';
      case 'low': return '#10b981';
      default: return '#6b7280';
    }
  };

  const getActionColor = (action: string) => {
    switch (action) {
      case 'urgent_consultation': return '#dc2626';
      case 'increase_frequency': return '#f59e0b';
      case 'continue_monitoring': return '#10b981';
      default: return '#6b7280';
    }
  };

  const getScheduleTypeIcon = (type: string) => {
    switch (type) {
      case 'self_exam': return '🔍';
      case 'dermatologist_visit': return '👨‍⚕️';
      case 'lesion_check': return '📸';
      case 'genetic_counseling': return '🧬';
      default: return '📋';
    }
  };

  const renderForecasts = () => {
    if (forecasts.length === 0) {
      return (
        <View style={styles.emptyState}>
          <Text style={styles.emptyIcon}>📊</Text>
          <Text style={styles.emptyText}>{t('analytics.forecasts.emptyTitle')}</Text>
          <Text style={styles.emptySubtext}>
            {t('analytics.forecasts.emptySubtext')}
          </Text>
          <Pressable
            style={styles.emptyButton}
            onPress={() => router.push('/lesion-tracking' as any)}
          >
            <Text style={styles.emptyButtonText}>{t('analytics.forecasts.emptyButton')}</Text>
          </Pressable>
        </View>
      );
    }

    return (
      <View style={styles.forecastsContainer}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>{t('analytics.forecasts.title')}</Text>
          <Text style={styles.sectionSubtitle}>{t('analytics.forecasts.analyzed', { count: forecasts.length })}</Text>
        </View>

        {forecasts.map((forecast) => (
          <View key={forecast.id} style={styles.forecastCard}>
            <View style={styles.forecastHeader}>
              <Text style={styles.forecastLesionName}>{forecast.lesion_name}</Text>
              <View style={[styles.trendBadge, { backgroundColor: getGrowthTrendColor(forecast.growth_trend) }]}>
                <Text style={styles.trendBadgeText}>{getGrowthTrendLabel(forecast.growth_trend)}</Text>
              </View>
            </View>

            {forecast.current_size_mm && (
              <View style={styles.forecastMetric}>
                <Text style={styles.metricLabel}>{t('analytics.forecasts.currentSize')}</Text>
                <Text style={styles.metricValue}>{forecast.current_size_mm.toFixed(1)} mm</Text>
              </View>
            )}

            {forecast.growth_rate_mm_per_month !== null && (
              <View style={styles.forecastMetric}>
                <Text style={styles.metricLabel}>{t('analytics.forecasts.growthRate')}</Text>
                <Text style={[styles.metricValue, { color: forecast.growth_rate_mm_per_month > 0.5 ? '#dc2626' : '#10b981' }]}>
                  {forecast.growth_rate_mm_per_month > 0 ? '+' : ''}{forecast.growth_rate_mm_per_month.toFixed(2)} mm/month
                </Text>
              </View>
            )}

            {forecast.predicted_size_90d && (
              <View style={styles.forecastMetric}>
                <Text style={styles.metricLabel}>{t('analytics.forecasts.predictedSize')}</Text>
                <Text style={styles.metricValue}>{forecast.predicted_size_90d.toFixed(1)} mm</Text>
              </View>
            )}

            <View style={styles.riskComparison}>
              <View style={styles.riskItem}>
                <Text style={styles.riskLabel}>{t('analytics.forecasts.currentRisk')}</Text>
                <View style={[styles.riskBadge, { backgroundColor: getRiskColor(forecast.current_risk_level) }]}>
                  <Text style={styles.riskBadgeText}>{forecast.current_risk_level?.toUpperCase()}</Text>
                </View>
              </View>
              <Text style={styles.riskArrow}>→</Text>
              <View style={styles.riskItem}>
                <Text style={styles.riskLabel}>{t('analytics.forecasts.forecastRisk')}</Text>
                <View style={[styles.riskBadge, { backgroundColor: getRiskColor(forecast.predicted_risk_level_90d) }]}>
                  <Text style={styles.riskBadgeText}>{forecast.predicted_risk_level_90d?.toUpperCase()}</Text>
                </View>
              </View>
            </View>

            {forecast.risk_escalation_probability > 0.3 && (
              <View style={styles.warningBanner}>
                <Text style={styles.warningIcon}>⚠️</Text>
                <Text style={styles.warningText}>
                  {t('analytics.forecasts.escalationWarning', { probability: (forecast.risk_escalation_probability * 100).toFixed(0) })}
                </Text>
              </View>
            )}

            {forecast.primary_risk_factors && forecast.primary_risk_factors.length > 0 && (
              <View style={styles.riskFactors}>
                <Text style={styles.riskFactorsTitle}>{t('analytics.forecasts.riskFactors')}</Text>
                {forecast.primary_risk_factors.map((factor, index) => (
                  <Text key={index} style={styles.riskFactorItem}>• {factor}</Text>
                ))}
              </View>
            )}

            <View style={styles.forecastActions}>
              <View style={[styles.actionBadge, { backgroundColor: getActionColor(forecast.recommended_action) }]}>
                <Text style={styles.actionBadgeText}>
                  {forecast.recommended_action.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                </Text>
              </View>
              <Text style={styles.nextCheckText}>
                {t('analytics.forecasts.nextCheck', { date: new Date(forecast.next_check_date).toLocaleDateString() })}
              </Text>
            </View>

            <View style={styles.confidenceBar}>
              <Text style={styles.confidenceLabel}>
                {t('analytics.forecasts.confidence', { score: (forecast.confidence_score * 100).toFixed(0) })}
              </Text>
              <View style={styles.confidenceProgress}>
                <View style={[styles.confidenceProgressFill, { width: `${forecast.confidence_score * 100}%` }]} />
              </View>
            </View>
          </View>
        ))}
      </View>
    );
  };

  const renderSchedule = () => {
    const upcomingSchedules = schedule.filter(s => !s.is_completed);
    const overdueSchedules = upcomingSchedules.filter(s => s.is_overdue);

    return (
      <View style={styles.scheduleContainer}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>{t('analytics.schedule.title')}</Text>
          <Pressable style={styles.generateButton} onPress={generateSchedule}>
            <Text style={styles.generateButtonText}>{t('analytics.schedule.regenerate')}</Text>
          </Pressable>
        </View>

        {upcomingSchedules.length === 0 && (
          <View style={styles.emptyState}>
            <Text style={styles.emptyIcon}>📅</Text>
            <Text style={styles.emptyText}>{t('analytics.schedule.emptyTitle')}</Text>
            <Text style={styles.emptySubtext}>
              {t('analytics.schedule.emptySubtext')}
            </Text>
            <Pressable style={styles.emptyButton} onPress={generateSchedule}>
              <Text style={styles.emptyButtonText}>{t('analytics.schedule.emptyButton')}</Text>
            </Pressable>
          </View>
        )}

        {overdueSchedules.length > 0 && (
          <View style={styles.overdueSection}>
            <Text style={styles.overdueSectionTitle}>{t('analytics.schedule.overdue', { count: overdueSchedules.length })}</Text>
            {overdueSchedules.map((item) => renderScheduleItem(item))}
          </View>
        )}

        {upcomingSchedules.filter(s => !s.is_overdue).map((item) => renderScheduleItem(item))}
      </View>
    );
  };

  const renderScheduleItem = (item: ScheduleItem) => {
    return (
      <View
        key={item.id}
        style={[
          styles.scheduleCard,
          item.is_overdue && styles.scheduleCardOverdue,
          item.priority >= 8 && styles.scheduleCardHighPriority
        ]}
      >
        <View style={styles.scheduleHeader}>
          <View style={styles.scheduleIconContainer}>
            <Text style={styles.scheduleIcon}>{getScheduleTypeIcon(item.schedule_type)}</Text>
            {item.priority >= 8 && <View style={styles.priorityDot} />}
          </View>
          <View style={styles.scheduleTitleContainer}>
            <Text style={styles.scheduleTitle}>{item.title}</Text>
            <Text style={styles.scheduleDate}>
              {new Date(item.recommended_date).toLocaleDateString('en-US', {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
                year: 'numeric'
              })}
            </Text>
          </View>
        </View>

        <Text style={styles.scheduleDescription}>{item.description}</Text>

        <View style={styles.scheduleMetadata}>
          {item.is_recurring && (
            <View style={styles.metadataBadge}>
              <Text style={styles.metadataBadgeText}>🔄 {item.recurrence_frequency}</Text>
            </View>
          )}
          {item.based_on_genetic_risk && (
            <View style={styles.metadataBadge}>
              <Text style={styles.metadataBadgeText}>{t('analytics.schedule.geneticRisk')}</Text>
            </View>
          )}
          {item.based_on_lesion_changes && (
            <View style={styles.metadataBadge}>
              <Text style={styles.metadataBadgeText}>{t('analytics.schedule.lesionChanges')}</Text>
            </View>
          )}
        </View>

        <Pressable
          style={styles.completeButton}
          onPress={() => completeScheduleItem(item.id)}
        >
          <Text style={styles.completeButtonText}>{t('analytics.schedule.markComplete')}</Text>
        </Pressable>
      </View>
    );
  };

  const renderTrends = () => {
    if (riskTrends.length === 0) {
      return (
        <View style={styles.emptyState}>
          <Text style={styles.emptyIcon}>📈</Text>
          <Text style={styles.emptyText}>{t('analytics.trends.emptyTitle')}</Text>
          <Text style={styles.emptySubtext}>
            {t('analytics.trends.emptySubtext')}
          </Text>
        </View>
      );
    }

    const latestTrend = riskTrends[riskTrends.length - 1];

    return (
      <View style={styles.trendsContainer}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>{t('analytics.trends.title')}</Text>
          <Text style={styles.sectionSubtitle}>{t('analytics.trends.snapshots', { count: riskTrends.length })}</Text>
        </View>

        <View style={styles.trendSummaryCard}>
          <Text style={styles.trendSummaryTitle}>{t('analytics.trends.currentStatus')}</Text>

          <View style={styles.trendMetricRow}>
            <View style={styles.trendMetric}>
              <Text style={styles.trendMetricLabel}>{t('analytics.trends.overallRisk')}</Text>
              <Text style={[styles.trendMetricValue, { color: getRiskColor(latestTrend.overall_risk_level) }]}>
                {latestTrend.overall_risk_score?.toFixed(0) || 'N/A'}
              </Text>
            </View>
            <View style={styles.trendMetric}>
              <Text style={styles.trendMetricLabel}>{t('analytics.trends.riskLevel')}</Text>
              <View style={[styles.riskBadge, { backgroundColor: getRiskColor(latestTrend.overall_risk_level) }]}>
                <Text style={styles.riskBadgeText}>{latestTrend.overall_risk_level?.toUpperCase()}</Text>
              </View>
            </View>
          </View>

          <View style={styles.trendMetricRow}>
            <View style={styles.trendMetric}>
              <Text style={styles.trendMetricLabel}>{t('analytics.trends.lesionsTracked')}</Text>
              <Text style={styles.trendMetricValue}>{latestTrend.total_lesions_tracked || 0}</Text>
            </View>
            <View style={styles.trendMetric}>
              <Text style={styles.trendMetricLabel}>{t('analytics.trends.highRiskLesions')}</Text>
              <Text style={[styles.trendMetricValue, { color: '#dc2626' }]}>
                {latestTrend.high_risk_lesions_count || 0}
              </Text>
            </View>
          </View>

          <View style={styles.trendMetricRow}>
            <View style={styles.trendMetric}>
              <Text style={styles.trendMetricLabel}>{t('analytics.trends.melanomaRisk')}</Text>
              <Text style={styles.trendMetricValue}>{latestTrend.melanoma_risk_score?.toFixed(0) || 'N/A'}</Text>
            </View>
            <View style={styles.trendMetric}>
              <Text style={styles.trendMetricLabel}>{t('analytics.trends.geneticRisk')}</Text>
              <Text style={styles.trendMetricValue}>{latestTrend.genetic_risk_score?.toFixed(0) || 'N/A'}</Text>
            </View>
          </View>
        </View>

        {riskTrends.length > 1 && (
          <View style={styles.trendHistoryCard}>
            <Text style={styles.trendHistoryTitle}>{t('analytics.trends.historicalTrend')}</Text>
            <View style={styles.trendChart}>
              {riskTrends.map((trend, index) => {
                const maxScore = 100;
                const height = ((trend.overall_risk_score || 0) / maxScore) * 100;
                return (
                  <View key={index} style={styles.trendBar}>
                    <View style={[styles.trendBarFill, { height: `${height}%`, backgroundColor: getRiskColor(trend.overall_risk_level) }]} />
                    <Text style={styles.trendBarLabel}>
                      {new Date(trend.snapshot_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                    </Text>
                  </View>
                );
              })}
            </View>
          </View>
        )}
      </View>
    );
  };

  const renderStats = () => {
    if (!analysisStats) {
      return (
        <View style={styles.emptyState}>
          <Text style={styles.emptyIcon}>📊</Text>
          <Text style={styles.emptyText}>No Analysis Data</Text>
          <Text style={styles.emptySubtext}>
            Start analyzing skin conditions to see your statistics here.
          </Text>
          <Pressable
            style={styles.emptyButton}
            onPress={() => router.push('/home' as any)}
          >
            <Text style={styles.emptyButtonText}>Start Analysis</Text>
          </Pressable>
        </View>
      );
    }

    const monthlyData = Object.entries(analysisStats.monthly_analysis_counts)
      .sort(([a], [b]) => a.localeCompare(b))
      .slice(-6);

    const maxMonthlyCount = Math.max(...monthlyData.map(([, count]) => count), 1);

    return (
      <View style={styles.statsContainer}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Analysis Overview</Text>
          <Text style={styles.sectionSubtitle}>{analysisStats.total_analyses} total analyses</Text>
        </View>

        {/* Summary Cards */}
        <View style={styles.statsGrid}>
          <View style={styles.statCard}>
            <Text style={styles.statIcon}>📸</Text>
            <Text style={styles.statValue}>{analysisStats.total_analyses}</Text>
            <Text style={styles.statLabel}>Total Analyses</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statIcon}>🔍</Text>
            <Text style={styles.statValue}>{analysisStats.lesion_detections}</Text>
            <Text style={styles.statLabel}>Lesions Detected</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statIcon}>✓</Text>
            <Text style={styles.statValue}>{analysisStats.non_lesion_detections}</Text>
            <Text style={styles.statLabel}>Non-Lesion</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statIcon}>📈</Text>
            <Text style={styles.statValue}>{(analysisStats.average_confidence * 100).toFixed(0)}%</Text>
            <Text style={styles.statLabel}>Avg Confidence</Text>
          </View>
        </View>

        {/* Most Common Diagnosis */}
        {analysisStats.most_common_diagnosis && (
          <View style={styles.diagnosisCard}>
            <Text style={styles.diagnosisTitle}>Most Common Finding</Text>
            <Text style={styles.diagnosisValue}>{analysisStats.most_common_diagnosis}</Text>
          </View>
        )}

        {/* Monthly Chart */}
        {monthlyData.length > 0 && (
          <View style={styles.monthlyChartCard}>
            <Text style={styles.monthlyChartTitle}>Monthly Activity</Text>
            <View style={styles.monthlyChart}>
              {monthlyData.map(([month, count]) => {
                const height = (count / maxMonthlyCount) * 100;
                const [year, monthNum] = month.split('-');
                const monthName = new Date(parseInt(year), parseInt(monthNum) - 1).toLocaleDateString('en-US', { month: 'short' });
                return (
                  <View key={month} style={styles.monthlyBar}>
                    <Text style={styles.monthlyBarCount}>{count}</Text>
                    <View style={[styles.monthlyBarFill, { height: `${Math.max(height, 5)}%` }]} />
                    <Text style={styles.monthlyBarLabel}>{monthName}</Text>
                  </View>
                );
              })}
            </View>
          </View>
        )}

        {/* Detection Breakdown */}
        <View style={styles.breakdownCard}>
          <Text style={styles.breakdownTitle}>Detection Breakdown</Text>
          <View style={styles.breakdownBar}>
            <View
              style={[
                styles.breakdownSegment,
                styles.breakdownLesion,
                { flex: analysisStats.lesion_detections || 1 }
              ]}
            />
            <View
              style={[
                styles.breakdownSegment,
                styles.breakdownNonLesion,
                { flex: analysisStats.non_lesion_detections || 1 }
              ]}
            />
          </View>
          <View style={styles.breakdownLegend}>
            <View style={styles.legendItem}>
              <View style={[styles.legendDot, styles.breakdownLesion]} />
              <Text style={styles.legendText}>Lesions ({analysisStats.lesion_detections})</Text>
            </View>
            <View style={styles.legendItem}>
              <View style={[styles.legendDot, styles.breakdownNonLesion]} />
              <Text style={styles.legendText}>Non-Lesion ({analysisStats.non_lesion_detections})</Text>
            </View>
          </View>
        </View>
      </View>
    );
  };

  const renderAccuracy = () => {
    if (!accuracyStats || accuracyStats.total_with_biopsy === 0) {
      return (
        <View style={styles.emptyState}>
          <Text style={styles.emptyIcon}>🎯</Text>
          <Text style={styles.emptyText}>No Accuracy Data Yet</Text>
          <Text style={styles.emptySubtext}>
            Add biopsy results to your analyses to track AI prediction accuracy.
          </Text>
          <Pressable
            style={styles.emptyButton}
            onPress={() => router.push('/history' as any)}
          >
            <Text style={styles.emptyButtonText}>View History</Text>
          </Pressable>
        </View>
      );
    }

    const getAccuracyColor = (accuracy: number) => {
      if (accuracy >= 0.9) return '#10b981';
      if (accuracy >= 0.7) return '#f59e0b';
      return '#dc2626';
    };

    return (
      <View style={styles.accuracyContainer}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>AI Prediction Accuracy</Text>
          <Text style={styles.sectionSubtitle}>Based on {accuracyStats.total_with_biopsy} biopsy results</Text>
        </View>

        {/* Main Accuracy Score */}
        <View style={styles.accuracyMainCard}>
          <View style={styles.accuracyCircle}>
            <Text style={[styles.accuracyPercent, { color: getAccuracyColor(accuracyStats.overall_accuracy) }]}>
              {(accuracyStats.overall_accuracy * 100).toFixed(0)}%
            </Text>
            <Text style={styles.accuracyCircleLabel}>Overall Accuracy</Text>
          </View>
          <View style={styles.accuracySecondary}>
            <View style={styles.accuracyMetric}>
              <Text style={styles.accuracyMetricValue}>
                {(accuracyStats.category_accuracy * 100).toFixed(0)}%
              </Text>
              <Text style={styles.accuracyMetricLabel}>Category Match</Text>
            </View>
          </View>
        </View>

        {/* Breakdown */}
        <View style={styles.accuracyBreakdownCard}>
          <Text style={styles.accuracyBreakdownTitle}>Result Breakdown</Text>

          <View style={styles.accuracyRow}>
            <View style={styles.accuracyRowIcon}>
              <Text>✓</Text>
            </View>
            <View style={styles.accuracyRowContent}>
              <Text style={styles.accuracyRowLabel}>Exact Matches</Text>
              <Text style={styles.accuracyRowDesc}>AI prediction matched biopsy exactly</Text>
            </View>
            <Text style={[styles.accuracyRowValue, { color: '#10b981' }]}>
              {accuracyStats.exact_matches}
            </Text>
          </View>

          <View style={styles.accuracyRow}>
            <View style={[styles.accuracyRowIcon, { backgroundColor: '#fef3c7' }]}>
              <Text>~</Text>
            </View>
            <View style={styles.accuracyRowContent}>
              <Text style={styles.accuracyRowLabel}>Category Matches</Text>
              <Text style={styles.accuracyRowDesc}>Correct benign/malignant category</Text>
            </View>
            <Text style={[styles.accuracyRowValue, { color: '#f59e0b' }]}>
              {accuracyStats.category_matches}
            </Text>
          </View>

          <View style={styles.accuracyRow}>
            <View style={[styles.accuracyRowIcon, { backgroundColor: '#fee2e2' }]}>
              <Text>✗</Text>
            </View>
            <View style={styles.accuracyRowContent}>
              <Text style={styles.accuracyRowLabel}>Mismatches</Text>
              <Text style={styles.accuracyRowDesc}>Prediction did not match biopsy</Text>
            </View>
            <Text style={[styles.accuracyRowValue, { color: '#dc2626' }]}>
              {accuracyStats.mismatches}
            </Text>
          </View>
        </View>

        {/* Info Card */}
        <View style={styles.infoCard}>
          <Text style={styles.infoIcon}>ℹ️</Text>
          <Text style={styles.infoText}>
            Accuracy improves as more biopsy results are added. This data helps calibrate AI predictions for better results.
          </Text>
        </View>
      </View>
    );
  };

  const renderLesionTrends = () => {
    if (!lesionTrends || !lesionTrends.has_history) {
      return (
        <View style={styles.emptyState}>
          <Text style={styles.emptyIcon}>📈</Text>
          <Text style={styles.emptyText}>No Lesion History</Text>
          <Text style={styles.emptySubtext}>
            {lesionTrends?.message || 'Complete a full-body skin check to start tracking lesion changes over time.'}
          </Text>
          <Pressable
            style={styles.emptyButton}
            onPress={() => router.push('/full-body-check' as any)}
          >
            <Text style={styles.emptyButtonText}>Start Skin Check</Text>
          </Pressable>
        </View>
      );
    }

    const getTrendIcon = (trend: string) => {
      switch (trend) {
        case 'increasing': return '📈';
        case 'decreasing': return '📉';
        case 'stable': return '➡️';
        default: return '📊';
      }
    };

    const getTrendColor = (trend: string) => {
      switch (trend) {
        case 'increasing': return '#f59e0b';
        case 'decreasing': return '#10b981';
        case 'stable': return '#3b82f6';
        default: return '#6b7280';
      }
    };

    const maxLesions = Math.max(...lesionTrends.history.map(h => h.total_lesions), 1);

    return (
      <View style={styles.lesionTrendsContainer}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Lesion Count Trends</Text>
          <Text style={styles.sectionSubtitle}>{lesionTrends.total_checks} skin checks</Text>
        </View>

        {/* Current Status */}
        <View style={styles.lesionCurrentCard}>
          <View style={styles.lesionCurrentHeader}>
            <Text style={styles.lesionCurrentTitle}>Current Status</Text>
            <View style={[styles.trendBadge, { backgroundColor: getTrendColor(lesionTrends.change.trend) }]}>
              <Text style={styles.trendBadgeText}>
                {getTrendIcon(lesionTrends.change.trend)} {lesionTrends.change.trend.toUpperCase()}
              </Text>
            </View>
          </View>

          <View style={styles.lesionStatsRow}>
            <View style={styles.lesionStat}>
              <Text style={styles.lesionStatValue}>{lesionTrends.current.total_lesions}</Text>
              <Text style={styles.lesionStatLabel}>Total Lesions</Text>
            </View>
            <View style={styles.lesionStat}>
              <Text style={[styles.lesionStatValue, { color: '#dc2626' }]}>
                {lesionTrends.current.high_risk_count}
              </Text>
              <Text style={styles.lesionStatLabel}>High Risk</Text>
            </View>
            <View style={styles.lesionStat}>
              <Text style={[
                styles.lesionStatValue,
                { color: lesionTrends.change.absolute > 0 ? '#f59e0b' : '#10b981' }
              ]}>
                {lesionTrends.change.absolute > 0 ? '+' : ''}{lesionTrends.change.absolute}
              </Text>
              <Text style={styles.lesionStatLabel}>Change</Text>
            </View>
          </View>

          {lesionTrends.change.percent !== 0 && (
            <Text style={styles.lesionChangePercent}>
              {lesionTrends.change.percent > 0 ? '+' : ''}{lesionTrends.change.percent.toFixed(1)}% from last check
            </Text>
          )}
        </View>

        {/* History Chart */}
        {lesionTrends.history.length > 1 && (
          <View style={styles.lesionHistoryCard}>
            <Text style={styles.lesionHistoryTitle}>History</Text>
            <View style={styles.lesionChart}>
              {lesionTrends.history.slice(-8).map((record, index) => {
                const height = (record.total_lesions / maxLesions) * 100;
                const highRiskHeight = (record.high_risk / maxLesions) * 100;
                return (
                  <View key={index} style={styles.lesionBar}>
                    <View style={styles.lesionBarStack}>
                      <View style={[styles.lesionBarNormal, { height: `${height}%` }]}>
                        <View style={[styles.lesionBarHighRisk, { height: `${(record.high_risk / record.total_lesions) * 100}%` }]} />
                      </View>
                    </View>
                    <Text style={styles.lesionBarLabel}>
                      {new Date(record.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                    </Text>
                  </View>
                );
              })}
            </View>
            <View style={styles.lesionLegend}>
              <View style={styles.legendItem}>
                <View style={[styles.legendDot, { backgroundColor: '#8b5cf6' }]} />
                <Text style={styles.legendText}>Total</Text>
              </View>
              <View style={styles.legendItem}>
                <View style={[styles.legendDot, { backgroundColor: '#dc2626' }]} />
                <Text style={styles.legendText}>High Risk</Text>
              </View>
            </View>
          </View>
        )}

        {/* Recommendation */}
        {lesionTrends.recommendation && (
          <View style={styles.recommendationCard}>
            <Text style={styles.recommendationIcon}>💡</Text>
            <Text style={styles.recommendationText}>{lesionTrends.recommendation}</Text>
          </View>
        )}
      </View>
    );
  };

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#8b5cf6" />
        <Text style={styles.loadingText}>{t('analytics.loading')}</Text>
      </View>
    );
  }

  return (
    <LinearGradient
      colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
      style={styles.container}
    >
      {/* Header */}
      <View style={styles.header}>
        <Pressable onPress={() => router.back()} style={styles.backButton}>
          <Text style={styles.backButtonText}>{t('analytics.common.back')}</Text>
        </Pressable>
        <Text style={styles.headerTitle}>{t('analytics.title')}</Text>
        <View style={{ width: 60 }} />
      </View>

      {/* Tabs */}
      <View style={styles.tabs}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.tabScroll}>
          <Pressable
            style={[styles.tab, activeTab === 'stats' && styles.tabActive]}
            onPress={() => setActiveTab('stats')}
          >
            <Text style={[styles.tabText, activeTab === 'stats' && styles.tabTextActive]}>
              Stats
            </Text>
          </Pressable>
          <Pressable
            style={[styles.tab, activeTab === 'accuracy' && styles.tabActive]}
            onPress={() => setActiveTab('accuracy')}
          >
            <Text style={[styles.tabText, activeTab === 'accuracy' && styles.tabTextActive]}>
              Accuracy
            </Text>
          </Pressable>
          <Pressable
            style={[styles.tab, activeTab === 'lesions' && styles.tabActive]}
            onPress={() => setActiveTab('lesions')}
          >
            <Text style={[styles.tabText, activeTab === 'lesions' && styles.tabTextActive]}>
              Lesions
            </Text>
          </Pressable>
          <Pressable
            style={[styles.tab, activeTab === 'forecasts' && styles.tabActive]}
            onPress={() => setActiveTab('forecasts')}
          >
            <Text style={[styles.tabText, activeTab === 'forecasts' && styles.tabTextActive]}>
              {t('analytics.tabs.forecasts')}
            </Text>
          </Pressable>
          <Pressable
            style={[styles.tab, activeTab === 'schedule' && styles.tabActive]}
            onPress={() => setActiveTab('schedule')}
          >
            <Text style={[styles.tabText, activeTab === 'schedule' && styles.tabTextActive]}>
              {t('analytics.tabs.schedule')}
            </Text>
          </Pressable>
          <Pressable
            style={[styles.tab, activeTab === 'trends' && styles.tabActive]}
            onPress={() => setActiveTab('trends')}
          >
            <Text style={[styles.tabText, activeTab === 'trends' && styles.tabTextActive]}>
              {t('analytics.tabs.trends')}
            </Text>
          </Pressable>
        </ScrollView>
      </View>

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {activeTab === 'stats' && renderStats()}
        {activeTab === 'accuracy' && renderAccuracy()}
        {activeTab === 'lesions' && renderLesionTrends()}
        {activeTab === 'forecasts' && renderForecasts()}
        {activeTab === 'schedule' && renderSchedule()}
        {activeTab === 'trends' && renderTrends()}
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
    color: '#6b7280',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 16,
    paddingBottom: 16,
    paddingHorizontal: 20,
    backgroundColor: '#8b5cf6',
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '600',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  tabs: {
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  tab: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    alignItems: 'center',
  },
  tabActive: {
    borderBottomWidth: 3,
    borderBottomColor: '#8b5cf6',
  },
  tabText: {
    fontSize: 14,
    color: '#6b7280',
    fontWeight: '500',
  },
  tabTextActive: {
    color: '#8b5cf6',
    fontWeight: '600',
  },
  content: {
    flex: 1,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  sectionSubtitle: {
    fontSize: 12,
    color: '#6b7280',
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 80,
    paddingHorizontal: 40,
  },
  emptyIcon: {
    fontSize: 48,
    marginBottom: 16,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#9ca3af',
    textAlign: 'center',
    marginBottom: 24,
  },
  emptyButton: {
    paddingHorizontal: 24,
    paddingVertical: 12,
    backgroundColor: '#8b5cf6',
    borderRadius: 8,
  },
  emptyButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  forecastsContainer: {
    paddingBottom: 20,
  },
  forecastCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginBottom: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  forecastHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  forecastLesionName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    flex: 1,
  },
  trendBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  trendBadgeText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#fff',
  },
  forecastMetric: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  metricLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  metricValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  riskComparison: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginVertical: 12,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: '#e5e7eb',
  },
  riskItem: {
    alignItems: 'center',
  },
  riskLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginBottom: 6,
  },
  riskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 8,
  },
  riskBadgeText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#fff',
  },
  riskArrow: {
    fontSize: 20,
    color: '#9ca3af',
  },
  warningBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fef3c7',
    padding: 12,
    borderRadius: 8,
    marginVertical: 12,
  },
  warningIcon: {
    fontSize: 16,
    marginRight: 8,
  },
  warningText: {
    fontSize: 13,
    color: '#92400e',
    flex: 1,
  },
  riskFactors: {
    marginTop: 12,
    padding: 12,
    backgroundColor: '#f9fafb',
    borderRadius: 8,
  },
  riskFactorsTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  riskFactorItem: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4,
  },
  forecastActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 12,
  },
  actionBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  actionBadgeText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#fff',
  },
  nextCheckText: {
    fontSize: 12,
    color: '#6b7280',
  },
  confidenceBar: {
    marginTop: 12,
  },
  confidenceLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginBottom: 6,
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
  scheduleContainer: {
    paddingBottom: 20,
  },
  generateButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#8b5cf6',
    borderRadius: 6,
  },
  generateButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#fff',
  },
  overdueSection: {
    marginHorizontal: 20,
    marginBottom: 16,
  },
  overdueSectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#dc2626',
    marginBottom: 12,
  },
  scheduleCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginBottom: 12,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  scheduleCardOverdue: {
    borderColor: '#dc2626',
    borderWidth: 2,
    backgroundColor: '#fef2f2',
  },
  scheduleCardHighPriority: {
    borderColor: '#f59e0b',
  },
  scheduleHeader: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  scheduleIconContainer: {
    position: 'relative',
    marginRight: 12,
  },
  scheduleIcon: {
    fontSize: 24,
  },
  priorityDot: {
    position: 'absolute',
    top: -2,
    right: -2,
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#dc2626',
  },
  scheduleTitleContainer: {
    flex: 1,
  },
  scheduleTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  scheduleDate: {
    fontSize: 12,
    color: '#6b7280',
  },
  scheduleDescription: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 12,
    lineHeight: 18,
  },
  scheduleMetadata: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginBottom: 12,
  },
  metadataBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    backgroundColor: '#ede9fe',
    borderRadius: 6,
  },
  metadataBadgeText: {
    fontSize: 10,
    color: '#7c3aed',
  },
  completeButton: {
    paddingVertical: 10,
    backgroundColor: '#10b981',
    borderRadius: 8,
    alignItems: 'center',
  },
  completeButtonText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#fff',
  },
  trendsContainer: {
    paddingBottom: 20,
  },
  trendSummaryCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginBottom: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  trendSummaryTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 16,
  },
  trendMetricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  trendMetric: {
    flex: 1,
  },
  trendMetricLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 6,
  },
  trendMetricValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  trendHistoryCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  trendHistoryTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 16,
  },
  trendChart: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 150,
    gap: 8,
  },
  trendBar: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  trendBarFill: {
    width: '100%',
    borderTopLeftRadius: 4,
    borderTopRightRadius: 4,
    minHeight: 4,
  },
  trendBarLabel: {
    fontSize: 9,
    color: '#9ca3af',
    marginTop: 6,
    transform: [{ rotate: '-45deg' }],
  },
  tabScroll: {
    flexGrow: 0,
  },
  // Stats Tab Styles
  statsContainer: {
    paddingBottom: 20,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 16,
    gap: 12,
  },
  statCard: {
    width: '47%',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  statIcon: {
    fontSize: 24,
    marginBottom: 8,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  diagnosisCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginTop: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  diagnosisTitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 8,
  },
  diagnosisValue: {
    fontSize: 18,
    fontWeight: '600',
    color: '#8b5cf6',
  },
  monthlyChartCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginTop: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  monthlyChartTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 16,
  },
  monthlyChart: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 120,
    gap: 8,
  },
  monthlyBar: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  monthlyBarCount: {
    fontSize: 12,
    fontWeight: '600',
    color: '#8b5cf6',
    marginBottom: 4,
  },
  monthlyBarFill: {
    width: '100%',
    backgroundColor: '#8b5cf6',
    borderTopLeftRadius: 4,
    borderTopRightRadius: 4,
    minHeight: 4,
  },
  monthlyBarLabel: {
    fontSize: 10,
    color: '#9ca3af',
    marginTop: 6,
  },
  breakdownCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginTop: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  breakdownTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 16,
  },
  breakdownBar: {
    flexDirection: 'row',
    height: 24,
    borderRadius: 12,
    overflow: 'hidden',
  },
  breakdownSegment: {
    height: '100%',
  },
  breakdownLesion: {
    backgroundColor: '#8b5cf6',
  },
  breakdownNonLesion: {
    backgroundColor: '#10b981',
  },
  breakdownLegend: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 12,
    gap: 24,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 6,
  },
  legendText: {
    fontSize: 12,
    color: '#6b7280',
  },
  // Accuracy Tab Styles
  accuracyContainer: {
    paddingBottom: 20,
  },
  accuracyMainCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    padding: 20,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    alignItems: 'center',
  },
  accuracyCircle: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: '#f3f4f6',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  accuracyPercent: {
    fontSize: 36,
    fontWeight: 'bold',
  },
  accuracyCircleLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  accuracySecondary: {
    flexDirection: 'row',
    justifyContent: 'center',
  },
  accuracyMetric: {
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  accuracyMetricValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  accuracyMetricLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  accuracyBreakdownCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginTop: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  accuracyBreakdownTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 16,
  },
  accuracyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  accuracyRowIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#dcfce7',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  accuracyRowContent: {
    flex: 1,
  },
  accuracyRowLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  accuracyRowDesc: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  accuracyRowValue: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  infoCard: {
    flexDirection: 'row',
    backgroundColor: '#eff6ff',
    marginHorizontal: 20,
    marginTop: 16,
    padding: 16,
    borderRadius: 12,
  },
  infoIcon: {
    fontSize: 16,
    marginRight: 12,
  },
  infoText: {
    flex: 1,
    fontSize: 13,
    color: '#1e40af',
    lineHeight: 18,
  },
  // Lesion Trends Tab Styles
  lesionTrendsContainer: {
    paddingBottom: 20,
  },
  lesionCurrentCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  lesionCurrentHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  lesionCurrentTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  lesionStatsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  lesionStat: {
    alignItems: 'center',
  },
  lesionStatValue: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  lesionStatLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  lesionChangePercent: {
    textAlign: 'center',
    fontSize: 12,
    color: '#6b7280',
    marginTop: 12,
  },
  lesionHistoryCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginTop: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  lesionHistoryTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 16,
  },
  lesionChart: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 120,
    gap: 8,
  },
  lesionBar: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  lesionBarStack: {
    width: '100%',
    height: 100,
    justifyContent: 'flex-end',
  },
  lesionBarNormal: {
    width: '100%',
    backgroundColor: '#8b5cf6',
    borderTopLeftRadius: 4,
    borderTopRightRadius: 4,
    minHeight: 4,
    justifyContent: 'flex-end',
  },
  lesionBarHighRisk: {
    width: '100%',
    backgroundColor: '#dc2626',
    borderTopLeftRadius: 4,
    borderTopRightRadius: 4,
  },
  lesionBarLabel: {
    fontSize: 9,
    color: '#9ca3af',
    marginTop: 6,
    textAlign: 'center',
  },
  lesionLegend: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 12,
    gap: 24,
  },
  recommendationCard: {
    flexDirection: 'row',
    backgroundColor: '#fef3c7',
    marginHorizontal: 20,
    marginTop: 16,
    padding: 16,
    borderRadius: 12,
  },
  recommendationIcon: {
    fontSize: 16,
    marginRight: 12,
  },
  recommendationText: {
    flex: 1,
    fontSize: 13,
    color: '#92400e',
    lineHeight: 18,
  },
});
