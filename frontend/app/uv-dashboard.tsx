import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  Dimensions,
  ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import config from '../config';

const { width: screenWidth } = Dimensions.get('window');

interface DailySummary {
  date: string;
  outdoor_minutes: number;
  uv_dose: number;
  avg_uv_index: number | null;
  max_uv_index: number | null;
  risk_score: number | null;
  risk_category: string;
}

interface UVHistory {
  period_days: number;
  days_with_data: number;
  period_stats: {
    total_outdoor_hours: number;
    total_uv_dose: number;
    average_daily_outdoor_minutes: number;
    high_risk_days: number;
    high_risk_percentage: number;
  };
  daily_summaries: DailySummary[];
}

interface UVTrends {
  has_sufficient_data: boolean;
  message?: string;
  weekly_trends?: Array<{
    week: number;
    avg_outdoor_minutes: number;
    avg_uv_index: number;
    avg_risk_score: number;
  }>;
  time_of_day_breakdown?: {
    morning_percentage: number;
    midday_percentage: number;
    afternoon_percentage: number;
    recommendation: string;
  };
  risk_trend?: string;
  insights?: Array<{
    type: string;
    title: string;
    message: string;
    priority: string;
  }>;
}

interface LesionCorrelation {
  id: number;
  lesion_group_id: number;
  body_location: string;
  days_analyzed: number;
  correlation_strength: string;
  uv_contribution_score: number;
  cumulative_uv_dose: number;
  recommendations: string[];
  analyzed_at: string;
}

export default function UVDashboardScreen() {
  const router = useRouter();
  const { token } = useAuth();
  const [history, setHistory] = useState<UVHistory | null>(null);
  const [trends, setTrends] = useState<UVTrends | null>(null);
  const [correlations, setCorrelations] = useState<LesionCorrelation[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedPeriod, setSelectedPeriod] = useState(30);

  const fetchData = useCallback(async () => {
    if (!token) return;

    try {
      const [historyRes, trendsRes, correlationsRes] = await Promise.all([
        fetch(`${config.API_URL}/wearables/uv-exposure/history?days=${selectedPeriod}`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch(`${config.API_URL}/wearables/uv-exposure/trends`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch(`${config.API_URL}/wearables/correlations`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ]);

      if (historyRes.ok) {
        const data = await historyRes.json();
        setHistory(data);
      }

      if (trendsRes.ok) {
        const data = await trendsRes.json();
        setTrends(data);
      }

      if (correlationsRes.ok) {
        const data = await correlationsRes.json();
        setCorrelations(data.correlations);
      }
    } catch (error) {
      console.error('Error fetching UV data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [token, selectedPeriod]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const onRefresh = () => {
    setRefreshing(true);
    fetchData();
  };

  const getRiskColor = (category: string) => {
    switch (category) {
      case 'low':
        return '#10b981';
      case 'moderate':
        return '#f59e0b';
      case 'high':
        return '#f97316';
      case 'very_high':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const getCorrelationColor = (strength: string) => {
    switch (strength) {
      case 'strong':
        return '#ef4444';
      case 'moderate':
        return '#f59e0b';
      case 'weak':
        return '#10b981';
      default:
        return '#6b7280';
    }
  };

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'warning':
        return 'warning';
      case 'alert':
        return 'alert-circle';
      case 'success':
        return 'checkmark-circle';
      default:
        return 'information-circle';
    }
  };

  const getInsightColor = (type: string) => {
    switch (type) {
      case 'warning':
        return '#f59e0b';
      case 'alert':
        return '#ef4444';
      case 'success':
        return '#10b981';
      default:
        return '#3b82f6';
    }
  };

  const renderMiniChart = (summaries: DailySummary[]) => {
    if (!summaries || summaries.length === 0) return null;

    const maxRisk = Math.max(...summaries.map(s => s.risk_score || 0), 1);
    const chartWidth = screenWidth - 64;
    const barWidth = Math.max((chartWidth / summaries.length) - 2, 4);

    return (
      <View style={styles.miniChart}>
        <View style={styles.chartBars}>
          {summaries.slice(-14).map((summary, index) => {
            const height = ((summary.risk_score || 0) / maxRisk) * 60;
            return (
              <View
                key={index}
                style={[
                  styles.chartBar,
                  {
                    width: barWidth,
                    height: Math.max(height, 4),
                    backgroundColor: getRiskColor(summary.risk_category),
                  },
                ]}
              />
            );
          })}
        </View>
        <View style={styles.chartLabels}>
          <Text style={styles.chartLabel}>14 days ago</Text>
          <Text style={styles.chartLabel}>Today</Text>
        </View>
      </View>
    );
  };

  const renderTimeOfDayChart = () => {
    if (!trends?.time_of_day_breakdown) return null;

    const { morning_percentage, midday_percentage, afternoon_percentage } = trends.time_of_day_breakdown;

    return (
      <View style={styles.timeChart}>
        <View style={styles.timeBarContainer}>
          <View style={[styles.timeBar, styles.morningBar, { flex: morning_percentage }]} />
          <View style={[styles.timeBar, styles.middayBar, { flex: midday_percentage }]} />
          <View style={[styles.timeBar, styles.afternoonBar, { flex: afternoon_percentage }]} />
        </View>
        <View style={styles.timeLabels}>
          <View style={styles.timeLabelItem}>
            <View style={[styles.timeDot, { backgroundColor: '#10b981' }]} />
            <Text style={styles.timeLabelText}>Morning {morning_percentage.toFixed(0)}%</Text>
          </View>
          <View style={styles.timeLabelItem}>
            <View style={[styles.timeDot, { backgroundColor: '#ef4444' }]} />
            <Text style={styles.timeLabelText}>Midday {midday_percentage.toFixed(0)}%</Text>
          </View>
          <View style={styles.timeLabelItem}>
            <View style={[styles.timeDot, { backgroundColor: '#f59e0b' }]} />
            <Text style={styles.timeLabelText}>Afternoon {afternoon_percentage.toFixed(0)}%</Text>
          </View>
        </View>
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#8b5cf6" />
        <Text style={styles.loadingText}>Loading UV data...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#1f2937" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>UV Exposure Dashboard</Text>
        <TouchableOpacity onPress={() => router.push('/wearables')} style={styles.deviceButton}>
          <Ionicons name="watch-outline" size={24} color="#8b5cf6" />
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={['#8b5cf6']} />
        }
      >
        {/* Period Selector */}
        <View style={styles.periodSelector}>
          {[7, 30, 90].map((days) => (
            <TouchableOpacity
              key={days}
              style={[styles.periodButton, selectedPeriod === days && styles.periodButtonActive]}
              onPress={() => setSelectedPeriod(days)}
            >
              <Text
                style={[styles.periodButtonText, selectedPeriod === days && styles.periodButtonTextActive]}
              >
                {days} Days
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Summary Stats */}
        {history && (
          <View style={styles.statsCard}>
            <View style={styles.statsHeader}>
              <Ionicons name="sunny" size={24} color="#f59e0b" />
              <Text style={styles.statsTitle}>Period Summary</Text>
            </View>

            <View style={styles.statsGrid}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{history.period_stats.total_outdoor_hours.toFixed(1)}</Text>
                <Text style={styles.statLabel}>Hours Outdoors</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{history.period_stats.average_daily_outdoor_minutes.toFixed(0)}</Text>
                <Text style={styles.statLabel}>Avg Min/Day</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={[styles.statValue, { color: history.period_stats.high_risk_percentage > 30 ? '#ef4444' : '#10b981' }]}>
                  {history.period_stats.high_risk_days}
                </Text>
                <Text style={styles.statLabel}>High Risk Days</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{history.days_with_data}</Text>
                <Text style={styles.statLabel}>Days Tracked</Text>
              </View>
            </View>

            {renderMiniChart(history.daily_summaries)}
          </View>
        )}

        {/* Time of Day Analysis */}
        {trends?.has_sufficient_data && trends.time_of_day_breakdown && (
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Ionicons name="time-outline" size={20} color="#8b5cf6" />
              <Text style={styles.cardTitle}>Exposure by Time of Day</Text>
            </View>

            {renderTimeOfDayChart()}

            <View style={styles.recommendationBox}>
              <Ionicons
                name={trends.time_of_day_breakdown.midday_percentage > 40 ? 'warning' : 'checkmark-circle'}
                size={16}
                color={trends.time_of_day_breakdown.midday_percentage > 40 ? '#f59e0b' : '#10b981'}
              />
              <Text style={styles.recommendationText}>
                {trends.time_of_day_breakdown.recommendation}
              </Text>
            </View>
          </View>
        )}

        {/* Risk Trend */}
        {trends?.has_sufficient_data && trends.risk_trend && (
          <View style={styles.trendCard}>
            <View style={styles.trendHeader}>
              <Ionicons
                name={
                  trends.risk_trend === 'decreasing'
                    ? 'trending-down'
                    : trends.risk_trend === 'increasing'
                    ? 'trending-up'
                    : 'remove'
                }
                size={24}
                color={
                  trends.risk_trend === 'decreasing'
                    ? '#10b981'
                    : trends.risk_trend === 'increasing'
                    ? '#ef4444'
                    : '#6b7280'
                }
              />
              <View style={styles.trendInfo}>
                <Text style={styles.trendTitle}>Risk Trend</Text>
                <Text
                  style={[
                    styles.trendValue,
                    {
                      color:
                        trends.risk_trend === 'decreasing'
                          ? '#10b981'
                          : trends.risk_trend === 'increasing'
                          ? '#ef4444'
                          : '#6b7280',
                    },
                  ]}
                >
                  {trends.risk_trend.charAt(0).toUpperCase() + trends.risk_trend.slice(1)}
                </Text>
              </View>
            </View>
          </View>
        )}

        {/* Insights */}
        {trends?.insights && trends.insights.length > 0 && (
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Ionicons name="bulb-outline" size={20} color="#8b5cf6" />
              <Text style={styles.cardTitle}>Insights</Text>
            </View>

            {trends.insights.map((insight, index) => (
              <View
                key={index}
                style={[styles.insightItem, { borderLeftColor: getInsightColor(insight.type) }]}
              >
                <Ionicons
                  name={getInsightIcon(insight.type) as any}
                  size={20}
                  color={getInsightColor(insight.type)}
                />
                <View style={styles.insightContent}>
                  <Text style={styles.insightTitle}>{insight.title}</Text>
                  <Text style={styles.insightMessage}>{insight.message}</Text>
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Lesion Correlations */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="git-compare-outline" size={20} color="#8b5cf6" />
            <Text style={styles.cardTitle}>Lesion UV Correlations</Text>
          </View>

          {correlations.length > 0 ? (
            correlations.map((correlation) => (
              <TouchableOpacity
                key={correlation.id}
                style={styles.correlationCard}
                onPress={() => router.push(`/lesion-detail?id=${correlation.lesion_group_id}`)}
              >
                <View style={styles.correlationHeader}>
                  <View style={styles.correlationLocation}>
                    <Ionicons name="body-outline" size={16} color="#6b7280" />
                    <Text style={styles.correlationLocationText}>
                      {correlation.body_location || 'Unknown location'}
                    </Text>
                  </View>
                  <View
                    style={[
                      styles.correlationStrengthBadge,
                      { backgroundColor: getCorrelationColor(correlation.correlation_strength) + '20' },
                    ]}
                  >
                    <Text
                      style={[
                        styles.correlationStrengthText,
                        { color: getCorrelationColor(correlation.correlation_strength) },
                      ]}
                    >
                      {correlation.correlation_strength} correlation
                    </Text>
                  </View>
                </View>

                <View style={styles.correlationStats}>
                  <View style={styles.correlationStatItem}>
                    <Text style={styles.correlationStatValue}>
                      {correlation.uv_contribution_score.toFixed(0)}%
                    </Text>
                    <Text style={styles.correlationStatLabel}>UV Contribution</Text>
                  </View>
                  <View style={styles.correlationStatItem}>
                    <Text style={styles.correlationStatValue}>
                      {correlation.cumulative_uv_dose.toFixed(0)}
                    </Text>
                    <Text style={styles.correlationStatLabel}>UV Dose</Text>
                  </View>
                  <View style={styles.correlationStatItem}>
                    <Text style={styles.correlationStatValue}>{correlation.days_analyzed}</Text>
                    <Text style={styles.correlationStatLabel}>Days Analyzed</Text>
                  </View>
                </View>

                {correlation.recommendations && correlation.recommendations.length > 0 && (
                  <View style={styles.correlationRecommendations}>
                    <Ionicons name="shield-checkmark-outline" size={14} color="#10b981" />
                    <Text style={styles.correlationRecommendationText} numberOfLines={2}>
                      {correlation.recommendations[0]}
                    </Text>
                  </View>
                )}
              </TouchableOpacity>
            ))
          ) : (
            <View style={styles.emptyCorrelations}>
              <Ionicons name="analytics-outline" size={40} color="#d1d5db" />
              <Text style={styles.emptyTitle}>No Correlations Yet</Text>
              <Text style={styles.emptyText}>
                Track lesions and sync UV data to see how sun exposure affects your skin
              </Text>
            </View>
          )}
        </View>

        {/* UV Protection Tips */}
        <View style={styles.tipsCard}>
          <Text style={styles.tipsTitle}>UV Protection Tips</Text>

          <View style={styles.tipItem}>
            <View style={[styles.tipIcon, { backgroundColor: '#fef3c7' }]}>
              <Ionicons name="sunny" size={16} color="#f59e0b" />
            </View>
            <View style={styles.tipContent}>
              <Text style={styles.tipHeading}>Avoid Peak Hours</Text>
              <Text style={styles.tipText}>Stay indoors or in shade from 10am to 4pm when UV is strongest</Text>
            </View>
          </View>

          <View style={styles.tipItem}>
            <View style={[styles.tipIcon, { backgroundColor: '#dbeafe' }]}>
              <Ionicons name="water" size={16} color="#3b82f6" />
            </View>
            <View style={styles.tipContent}>
              <Text style={styles.tipHeading}>Apply Sunscreen</Text>
              <Text style={styles.tipText}>Use SPF 30+ and reapply every 2 hours when outdoors</Text>
            </View>
          </View>

          <View style={styles.tipItem}>
            <View style={[styles.tipIcon, { backgroundColor: '#dcfce7' }]}>
              <Ionicons name="shirt" size={16} color="#10b981" />
            </View>
            <View style={styles.tipContent}>
              <Text style={styles.tipHeading}>Wear Protective Clothing</Text>
              <Text style={styles.tipText}>Long sleeves, wide-brimmed hat, and UV-blocking sunglasses</Text>
            </View>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#6b7280',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  backButton: {
    padding: 4,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  deviceButton: {
    padding: 4,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  periodSelector: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 4,
    marginBottom: 16,
  },
  periodButton: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
  },
  periodButtonActive: {
    backgroundColor: '#8b5cf6',
  },
  periodButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#6b7280',
  },
  periodButtonTextActive: {
    color: '#fff',
  },
  statsCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  statsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 8,
  },
  statsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 16,
  },
  statItem: {
    width: '50%',
    paddingVertical: 12,
    alignItems: 'center',
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
  miniChart: {
    marginTop: 8,
  },
  chartBars: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    height: 60,
  },
  chartBar: {
    borderRadius: 2,
  },
  chartLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  chartLabel: {
    fontSize: 10,
    color: '#9ca3af',
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 8,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  timeChart: {
    marginBottom: 12,
  },
  timeBarContainer: {
    flexDirection: 'row',
    height: 24,
    borderRadius: 12,
    overflow: 'hidden',
  },
  timeBar: {
    height: '100%',
  },
  morningBar: {
    backgroundColor: '#10b981',
  },
  middayBar: {
    backgroundColor: '#ef4444',
  },
  afternoonBar: {
    backgroundColor: '#f59e0b',
  },
  timeLabels: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 12,
  },
  timeLabelItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  timeDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  timeLabelText: {
    fontSize: 12,
    color: '#6b7280',
  },
  recommendationBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    padding: 12,
    borderRadius: 8,
    gap: 8,
  },
  recommendationText: {
    flex: 1,
    fontSize: 13,
    color: '#4b5563',
  },
  trendCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  trendHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  trendInfo: {
    flex: 1,
  },
  trendTitle: {
    fontSize: 14,
    color: '#6b7280',
  },
  trendValue: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  insightItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#f9fafb',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderLeftWidth: 3,
    gap: 10,
  },
  insightContent: {
    flex: 1,
  },
  insightTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  insightMessage: {
    fontSize: 13,
    color: '#6b7280',
    lineHeight: 18,
  },
  correlationCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  correlationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  correlationLocation: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  correlationLocationText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#1f2937',
  },
  correlationStrengthBadge: {
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 6,
  },
  correlationStrengthText: {
    fontSize: 12,
    fontWeight: '500',
    textTransform: 'capitalize',
  },
  correlationStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  correlationStatItem: {
    alignItems: 'center',
  },
  correlationStatValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  correlationStatLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 2,
  },
  correlationRecommendations: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    gap: 6,
  },
  correlationRecommendationText: {
    flex: 1,
    fontSize: 12,
    color: '#10b981',
  },
  emptyCorrelations: {
    alignItems: 'center',
    paddingVertical: 24,
  },
  emptyTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#4b5563',
    marginTop: 12,
  },
  emptyText: {
    fontSize: 13,
    color: '#9ca3af',
    textAlign: 'center',
    marginTop: 4,
  },
  tipsCard: {
    backgroundColor: '#fefce8',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#fef08a',
  },
  tipsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#854d0e',
    marginBottom: 16,
  },
  tipItem: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  tipIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  tipContent: {
    flex: 1,
    marginLeft: 12,
  },
  tipHeading: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  tipText: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
});
