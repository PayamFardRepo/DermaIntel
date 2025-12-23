import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { API_BASE_URL } from '../config';

const { width } = Dimensions.get('window');

type TabType = 'overview' | 'conditions' | 'skin_types' | 'feedback';

interface TimelinePoint {
  period: string;
  label: string;
  accuracy: number;
  accuracy_pct: string;
  samples: number;
}

interface ConditionData {
  condition: string;
  condition_id: string;
  current_accuracy: string;
  current_sensitivity: string;
  current_specificity: string;
  accuracy_improvement: string;
  timeline: any[];
}

interface SkinTypeData {
  skin_type: string;
  current_accuracy: string;
  sample_count: number;
  accuracy_improvement: string;
  sample_growth: string;
}

export default function AIAccuracyScreen() {
  const router = useRouter();
  const { token } = useAuth();

  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Data states
  const [overviewData, setOverviewData] = useState<any>(null);
  const [conditionsData, setConditionsData] = useState<any>(null);
  const [skinTypesData, setSkinTypesData] = useState<any>(null);
  const [feedbackData, setFeedbackData] = useState<any>(null);
  const [projectionsData, setProjectionsData] = useState<any>(null);
  const [versionsData, setVersionsData] = useState<any>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const headers = { 'Authorization': `Bearer ${token}` };

      // Load all data in parallel
      const [overview, conditions, skinTypes, feedback, projections, versions] = await Promise.all([
        fetch(`${API_BASE_URL}/clinical/ai-accuracy/overview`, { headers }).then(r => r.json()),
        fetch(`${API_BASE_URL}/clinical/ai-accuracy/by-condition`, { headers }).then(r => r.json()),
        fetch(`${API_BASE_URL}/clinical/ai-accuracy/by-skin-type`, { headers }).then(r => r.json()),
        fetch(`${API_BASE_URL}/clinical/ai-accuracy/feedback-impact`, { headers }).then(r => r.json()),
        fetch(`${API_BASE_URL}/clinical/ai-accuracy/projections`, { headers }).then(r => r.json()),
        fetch(`${API_BASE_URL}/clinical/ai-accuracy/model-versions`, { headers }).then(r => r.json()),
      ]);

      setOverviewData(overview);
      setConditionsData(conditions);
      setSkinTypesData(skinTypes);
      setFeedbackData(feedback);
      setProjectionsData(projections);
      setVersionsData(versions);
    } catch (err: any) {
      console.error('Error loading AI accuracy data:', err);
      setError(err.message || 'Failed to load data');
    } finally {
      setIsLoading(false);
    }
  };

  const renderAccuracyGraph = (timeline: TimelinePoint[], projections?: any[]) => {
    const chartHeight = 160;
    const chartWidth = width - 80;
    const minAccuracy = 0.8;
    const maxAccuracy = 1.0;
    const range = maxAccuracy - minAccuracy;

    const allPoints = [...timeline];
    if (projections) {
      projections.forEach(p => {
        allPoints.push({
          period: p.period,
          label: p.label,
          accuracy: p.projected_accuracy,
          accuracy_pct: p.projected_accuracy_pct,
          samples: 0,
        });
      });
    }

    return (
      <View style={styles.graphContainer}>
        <View style={styles.graphWrapper}>
          {/* Y-axis labels */}
          <View style={styles.yAxisLabels}>
            <Text style={styles.axisLabel}>100%</Text>
            <Text style={styles.axisLabel}>95%</Text>
            <Text style={styles.axisLabel}>90%</Text>
            <Text style={styles.axisLabel}>85%</Text>
            <Text style={styles.axisLabel}>80%</Text>
          </View>

          {/* Chart area */}
          <View style={[styles.chartArea, { height: chartHeight, width: chartWidth }]}>
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => (
              <View
                key={i}
                style={[styles.gridLine, { bottom: pct * chartHeight }]}
              />
            ))}

            {/* Accuracy line and points */}
            {timeline.map((point, i) => {
              const x = (i / (allPoints.length - 1)) * chartWidth;
              const y = ((point.accuracy - minAccuracy) / range) * chartHeight;

              return (
                <View key={point.period}>
                  {/* Line to next point */}
                  {i < timeline.length - 1 && (
                    <View
                      style={[
                        styles.graphLine,
                        {
                          left: x,
                          bottom: y,
                          width: chartWidth / (allPoints.length - 1),
                          transform: [
                            {
                              rotate: `${Math.atan2(
                                ((timeline[i + 1].accuracy - minAccuracy) / range) * chartHeight - y,
                                chartWidth / (allPoints.length - 1)
                              ) * (180 / Math.PI)}deg`,
                            },
                          ],
                        },
                      ]}
                    />
                  )}
                  {/* Point */}
                  <View
                    style={[
                      styles.graphPoint,
                      {
                        left: x - 6,
                        bottom: y - 6,
                        backgroundColor: '#10b981',
                      },
                    ]}
                  />
                </View>
              );
            })}

            {/* Projection points (if available) */}
            {projections && projections.map((point, i) => {
              const idx = timeline.length + i;
              const x = (idx / (allPoints.length - 1)) * chartWidth;
              const y = ((point.projected_accuracy - minAccuracy) / range) * chartHeight;

              return (
                <View
                  key={point.period}
                  style={[
                    styles.graphPoint,
                    styles.projectionPoint,
                    {
                      left: x - 6,
                      bottom: y - 6,
                    },
                  ]}
                />
              );
            })}
          </View>
        </View>

        {/* X-axis labels */}
        <View style={styles.xAxisLabels}>
          {allPoints.filter((_, i) => i % 2 === 0 || i === allPoints.length - 1).map((point) => (
            <Text key={point.period} style={styles.axisLabel}>
              {point.label.split(' ')[0]}
            </Text>
          ))}
        </View>

        {/* Legend */}
        <View style={styles.graphLegend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#10b981' }]} />
            <Text style={styles.legendText}>Actual</Text>
          </View>
          {projections && (
            <View style={styles.legendItem}>
              <View style={[styles.legendDot, styles.projectionDot]} />
              <Text style={styles.legendText}>Projected</Text>
            </View>
          )}
        </View>
      </View>
    );
  };

  const renderOverviewTab = () => {
    if (!overviewData) return null;

    return (
      <View>
        {/* Current Accuracy Hero */}
        <View style={styles.heroCard}>
          <Text style={styles.heroLabel}>Current Diagnostic Accuracy</Text>
          <Text style={styles.heroValue}>{overviewData.current_accuracy}</Text>
          <View style={styles.heroMetrics}>
            <View style={styles.heroMetric}>
              <Text style={styles.heroMetricValue}>
                {(overviewData.current_metrics.precision * 100).toFixed(1)}%
              </Text>
              <Text style={styles.heroMetricLabel}>Precision</Text>
            </View>
            <View style={styles.heroMetric}>
              <Text style={styles.heroMetricValue}>
                {(overviewData.current_metrics.recall * 100).toFixed(1)}%
              </Text>
              <Text style={styles.heroMetricLabel}>Recall</Text>
            </View>
            <View style={styles.heroMetric}>
              <Text style={styles.heroMetricValue}>
                {overviewData.current_metrics.total_samples.toLocaleString()}
              </Text>
              <Text style={styles.heroMetricLabel}>Samples</Text>
            </View>
          </View>
        </View>

        {/* Improvement Summary */}
        <View style={styles.improvementCard}>
          <Text style={styles.sectionTitle}>
            <Ionicons name="trending-up" size={18} color="#10b981" /> Improvement Over Time
          </Text>
          <View style={styles.improvementGrid}>
            <View style={styles.improvementItem}>
              <Text style={styles.improvementValue}>
                +{overviewData.improvement_summary.accuracy_gain}%
              </Text>
              <Text style={styles.improvementLabel}>Accuracy Gain</Text>
            </View>
            <View style={styles.improvementItem}>
              <Text style={styles.improvementValue}>
                +{overviewData.improvement_summary.sample_growth.toLocaleString()}
              </Text>
              <Text style={styles.improvementLabel}>New Samples</Text>
            </View>
            <View style={styles.improvementItem}>
              <Text style={styles.improvementValue}>
                +{overviewData.improvement_summary.recall_gain}%
              </Text>
              <Text style={styles.improvementLabel}>Recall Gain</Text>
            </View>
          </View>
        </View>

        {/* Accuracy Graph */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            <Ionicons name="analytics-outline" size={18} color="#0ea5e9" /> Accuracy Trend
          </Text>
          {renderAccuracyGraph(overviewData.timeline, projectionsData?.projections)}
        </View>

        {/* Model Versions */}
        {versionsData && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>
              <Ionicons name="git-branch-outline" size={18} color="#0ea5e9" /> Model Versions
            </Text>
            <Text style={styles.currentVersion}>
              Current: v{versionsData.current_version}
            </Text>
            {versionsData.versions.map((version: any) => (
              <View key={version.version} style={styles.versionCard}>
                <View style={styles.versionHeader}>
                  <Text style={styles.versionNumber}>v{version.version}</Text>
                  <Text style={styles.versionAccuracy}>
                    {(version.accuracy * 100).toFixed(1)}%
                  </Text>
                </View>
                <Text style={styles.versionDate}>{version.date}</Text>
                <Text style={styles.versionNotes}>{version.notes}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Improvement Drivers */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            <Ionicons name="rocket-outline" size={18} color="#0ea5e9" /> What's Driving Improvement
          </Text>
          {overviewData.improvement_drivers.map((driver: string, i: number) => (
            <View key={i} style={styles.driverItem}>
              <Ionicons name="checkmark-circle" size={16} color="#10b981" />
              <Text style={styles.driverText}>{driver}</Text>
            </View>
          ))}
        </View>
      </View>
    );
  };

  const renderConditionsTab = () => {
    if (!conditionsData) return null;

    return (
      <View>
        <View style={styles.summaryRow}>
          <View style={styles.summaryItem}>
            <Text style={styles.summaryLabel}>Best Performing</Text>
            <Text style={styles.summaryValue}>{conditionsData.best_performing}</Text>
          </View>
          <View style={styles.summaryItem}>
            <Text style={styles.summaryLabel}>Most Improved</Text>
            <Text style={styles.summaryValue}>{conditionsData.most_improved}</Text>
          </View>
        </View>

        {conditionsData.conditions.map((condition: ConditionData) => (
          <View key={condition.condition_id} style={styles.conditionCard}>
            <View style={styles.conditionHeader}>
              <Text style={styles.conditionName}>{condition.condition}</Text>
              <View style={styles.accuracyBadge}>
                <Text style={styles.accuracyBadgeText}>{condition.current_accuracy}</Text>
              </View>
            </View>

            <View style={styles.conditionMetrics}>
              <View style={styles.metricItem}>
                <Text style={styles.metricValue}>{condition.current_sensitivity}</Text>
                <Text style={styles.metricLabel}>Sensitivity</Text>
              </View>
              <View style={styles.metricItem}>
                <Text style={styles.metricValue}>{condition.current_specificity}</Text>
                <Text style={styles.metricLabel}>Specificity</Text>
              </View>
              <View style={styles.metricItem}>
                <Text style={[styles.metricValue, { color: '#10b981' }]}>
                  {condition.accuracy_improvement}
                </Text>
                <Text style={styles.metricLabel}>Improvement</Text>
              </View>
            </View>

            {/* Mini timeline */}
            <View style={styles.miniTimeline}>
              {condition.timeline.map((point: any, i: number) => (
                <View
                  key={point.period}
                  style={[
                    styles.miniTimelinePoint,
                    {
                      height: (point.accuracy - 0.8) * 200,
                      backgroundColor: i === condition.timeline.length - 1 ? '#10b981' : '#475569',
                    },
                  ]}
                />
              ))}
            </View>
          </View>
        ))}
      </View>
    );
  };

  const renderSkinTypesTab = () => {
    if (!skinTypesData) return null;

    return (
      <View>
        {/* Equity Metrics */}
        <View style={styles.equityCard}>
          <Text style={styles.equityTitle}>Equity Metrics</Text>
          <View style={styles.equityGrid}>
            <View style={styles.equityItem}>
              <Text style={styles.equityValue}>{skinTypesData.equity_metrics.accuracy_gap}</Text>
              <Text style={styles.equityLabel}>Accuracy Gap</Text>
            </View>
            <View style={styles.equityItem}>
              <Text style={[styles.equityValue, { color: '#10b981' }]}>
                {skinTypesData.equity_metrics.gap_trend}
              </Text>
              <Text style={styles.equityLabel}>Trend</Text>
            </View>
          </View>
          <Text style={styles.equityNote}>{skinTypesData.equity_metrics.note}</Text>
        </View>

        {skinTypesData.skin_types.map((skinType: SkinTypeData) => (
          <View key={skinType.skin_type} style={styles.skinTypeCard}>
            <View style={styles.skinTypeHeader}>
              <Text style={styles.skinTypeName}>{skinType.skin_type}</Text>
              <Text style={styles.skinTypeAccuracy}>{skinType.current_accuracy}</Text>
            </View>

            <View style={styles.skinTypeDetails}>
              <View style={styles.skinTypeDetail}>
                <Ionicons name="people" size={14} color="#64748b" />
                <Text style={styles.skinTypeDetailText}>
                  {skinType.sample_count.toLocaleString()} samples
                </Text>
              </View>
              <View style={styles.skinTypeDetail}>
                <Ionicons name="trending-up" size={14} color="#10b981" />
                <Text style={[styles.skinTypeDetailText, { color: '#10b981' }]}>
                  {skinType.accuracy_improvement}
                </Text>
              </View>
              <View style={styles.skinTypeDetail}>
                <Ionicons name="add-circle" size={14} color="#0ea5e9" />
                <Text style={[styles.skinTypeDetailText, { color: '#0ea5e9' }]}>
                  {skinType.sample_growth} samples
                </Text>
              </View>
            </View>

            {/* Progress bar */}
            <View style={styles.progressBarContainer}>
              <View
                style={[
                  styles.progressBar,
                  { width: `${parseFloat(skinType.current_accuracy)}%` },
                ]}
              />
            </View>
          </View>
        ))}
      </View>
    );
  };

  const renderFeedbackTab = () => {
    if (!feedbackData) return null;

    return (
      <View>
        {/* Feedback Summary */}
        <View style={styles.feedbackHero}>
          <Text style={styles.feedbackHeroLabel}>Your Feedback Matters</Text>
          <Text style={styles.feedbackHeroValue}>
            {feedbackData.summary.accuracy_improvement}
          </Text>
          <Text style={styles.feedbackHeroSubtext}>
            accuracy improvement from user feedback
          </Text>
        </View>

        <View style={styles.feedbackStats}>
          <View style={styles.feedbackStat}>
            <Text style={styles.feedbackStatValue}>
              {feedbackData.summary.total_feedback.toLocaleString()}
            </Text>
            <Text style={styles.feedbackStatLabel}>Total Feedback</Text>
          </View>
          <View style={styles.feedbackStat}>
            <Text style={styles.feedbackStatValue}>
              {feedbackData.summary.incorporated.toLocaleString()}
            </Text>
            <Text style={styles.feedbackStatLabel}>Incorporated</Text>
          </View>
          <View style={styles.feedbackStat}>
            <Text style={styles.feedbackStatValue}>
              {feedbackData.summary.incorporation_rate}
            </Text>
            <Text style={styles.feedbackStatLabel}>Rate</Text>
          </View>
        </View>

        {/* Top Correction Categories */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            <Ionicons name="git-compare-outline" size={18} color="#0ea5e9" /> Top Correction Categories
          </Text>
          {feedbackData.top_categories.map((cat: any, i: number) => (
            <View key={cat.category} style={styles.categoryCard}>
              <View style={styles.categoryHeader}>
                <Text style={styles.categoryRank}>#{i + 1}</Text>
                <Text style={styles.categoryName}>
                  {cat.category.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                </Text>
              </View>
              <View style={styles.categoryStats}>
                <Text style={styles.categoryCorrections}>
                  {cat.corrections.toLocaleString()} corrections
                </Text>
                <Text style={styles.categoryImpact}>{cat.impact}</Text>
              </View>
            </View>
          ))}
        </View>

        {/* How Feedback Helps */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            <Ionicons name="help-circle-outline" size={18} color="#0ea5e9" /> How Your Feedback Helps
          </Text>
          {feedbackData.how_feedback_helps.map((item: string, i: number) => (
            <View key={i} style={styles.helpItem}>
              <Text style={styles.helpNumber}>{i + 1}</Text>
              <Text style={styles.helpText}>{item}</Text>
            </View>
          ))}
        </View>

        {/* Call to Action */}
        <View style={styles.ctaCard}>
          <Ionicons name="chatbubble-ellipses" size={32} color="#0ea5e9" />
          <Text style={styles.ctaText}>{feedbackData.call_to_action}</Text>
        </View>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>AI Accuracy</Text>
          <Text style={styles.headerSubtitle}>
            Diagnostic accuracy improving over time with more data
          </Text>
        </View>
      </LinearGradient>

      {/* Tabs */}
      <View style={styles.tabBar}>
        {[
          { id: 'overview' as TabType, label: 'Overview', icon: 'analytics-outline' },
          { id: 'conditions' as TabType, label: 'Conditions', icon: 'medkit-outline' },
          { id: 'skin_types' as TabType, label: 'Skin Types', icon: 'people-outline' },
          { id: 'feedback' as TabType, label: 'Feedback', icon: 'chatbubble-outline' },
        ].map((tab) => (
          <TouchableOpacity
            key={tab.id}
            style={[styles.tab, activeTab === tab.id && styles.tabActive]}
            onPress={() => setActiveTab(tab.id)}
          >
            <Ionicons
              name={tab.icon as any}
              size={16}
              color={activeTab === tab.id ? '#0ea5e9' : '#64748b'}
            />
            <Text style={[styles.tabText, activeTab === tab.id && styles.tabTextActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {isLoading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#0ea5e9" />
            <Text style={styles.loadingText}>Loading accuracy data...</Text>
          </View>
        ) : error ? (
          <View style={styles.errorContainer}>
            <Ionicons name="alert-circle" size={48} color="#ef4444" />
            <Text style={styles.errorText}>{error}</Text>
            <TouchableOpacity style={styles.retryButton} onPress={loadData}>
              <Text style={styles.retryButtonText}>Retry</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <>
            {activeTab === 'overview' && renderOverviewTab()}
            {activeTab === 'conditions' && renderConditionsTab()}
            {activeTab === 'skin_types' && renderSkinTypesTab()}
            {activeTab === 'feedback' && renderFeedbackTab()}
          </>
        )}
        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  header: {
    paddingTop: 50,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  backButton: {
    marginBottom: 15,
  },
  headerContent: {},
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#94a3b8',
    marginTop: 4,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.05)',
    marginHorizontal: 20,
    marginTop: -10,
    borderRadius: 12,
    padding: 4,
  },
  tab: {
    flex: 1,
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    borderRadius: 8,
    gap: 2,
  },
  tabActive: {
    backgroundColor: 'rgba(14,165,233,0.2)',
  },
  tabText: {
    color: '#64748b',
    fontSize: 10,
    fontWeight: '500',
  },
  tabTextActive: {
    color: '#0ea5e9',
  },
  content: {
    flex: 1,
    padding: 20,
  },
  loadingContainer: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  loadingText: {
    color: '#94a3b8',
    marginTop: 16,
  },
  errorContainer: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  errorText: {
    color: '#ef4444',
    marginTop: 16,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 16,
    backgroundColor: '#0ea5e9',
    paddingHorizontal: 24,
    paddingVertical: 10,
    borderRadius: 8,
  },
  retryButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  heroCard: {
    backgroundColor: 'rgba(16,185,129,0.1)',
    borderRadius: 20,
    padding: 24,
    alignItems: 'center',
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(16,185,129,0.3)',
  },
  heroLabel: {
    color: '#94a3b8',
    fontSize: 14,
  },
  heroValue: {
    color: '#10b981',
    fontSize: 56,
    fontWeight: 'bold',
    marginVertical: 8,
  },
  heroMetrics: {
    flexDirection: 'row',
    gap: 24,
    marginTop: 12,
  },
  heroMetric: {
    alignItems: 'center',
  },
  heroMetricValue: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  heroMetricLabel: {
    color: '#64748b',
    fontSize: 11,
    marginTop: 2,
  },
  improvementCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 16,
  },
  improvementGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  improvementItem: {
    alignItems: 'center',
    flex: 1,
  },
  improvementValue: {
    color: '#10b981',
    fontSize: 20,
    fontWeight: 'bold',
  },
  improvementLabel: {
    color: '#64748b',
    fontSize: 11,
    marginTop: 4,
  },
  section: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  graphContainer: {
    marginTop: 8,
  },
  graphWrapper: {
    flexDirection: 'row',
  },
  yAxisLabels: {
    width: 35,
    justifyContent: 'space-between',
    paddingRight: 4,
  },
  chartArea: {
    position: 'relative',
    backgroundColor: 'rgba(0,0,0,0.2)',
    borderRadius: 8,
  },
  gridLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  graphLine: {
    position: 'absolute',
    height: 2,
    backgroundColor: '#10b981',
    transformOrigin: 'left center',
  },
  graphPoint: {
    position: 'absolute',
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: '#fff',
  },
  projectionPoint: {
    backgroundColor: '#f59e0b',
    borderStyle: 'dashed',
  },
  xAxisLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingLeft: 35,
    paddingTop: 8,
  },
  axisLabel: {
    color: '#64748b',
    fontSize: 10,
  },
  graphLegend: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 20,
    marginTop: 12,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  projectionDot: {
    backgroundColor: '#f59e0b',
  },
  legendText: {
    color: '#94a3b8',
    fontSize: 11,
  },
  currentVersion: {
    color: '#0ea5e9',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 12,
  },
  versionCard: {
    backgroundColor: 'rgba(0,0,0,0.2)',
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
  },
  versionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  versionNumber: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  versionAccuracy: {
    color: '#10b981',
    fontSize: 14,
    fontWeight: '600',
  },
  versionDate: {
    color: '#64748b',
    fontSize: 11,
    marginTop: 4,
  },
  versionNotes: {
    color: '#94a3b8',
    fontSize: 12,
    marginTop: 6,
  },
  driverItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 10,
  },
  driverText: {
    color: '#94a3b8',
    fontSize: 13,
    flex: 1,
  },
  summaryRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  summaryItem: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 14,
    alignItems: 'center',
  },
  summaryLabel: {
    color: '#64748b',
    fontSize: 11,
  },
  summaryValue: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    marginTop: 4,
  },
  conditionCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
  },
  conditionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  conditionName: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '500',
  },
  accuracyBadge: {
    backgroundColor: 'rgba(16,185,129,0.2)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  accuracyBadgeText: {
    color: '#10b981',
    fontSize: 13,
    fontWeight: '600',
  },
  conditionMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metricItem: {
    alignItems: 'center',
  },
  metricValue: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },
  metricLabel: {
    color: '#64748b',
    fontSize: 10,
    marginTop: 2,
  },
  miniTimeline: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    height: 30,
    marginTop: 12,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.1)',
  },
  miniTimelinePoint: {
    width: 16,
    borderRadius: 4,
    minHeight: 4,
  },
  equityCard: {
    backgroundColor: 'rgba(14,165,233,0.1)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(14,165,233,0.3)',
  },
  equityTitle: {
    color: '#0ea5e9',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  equityGrid: {
    flexDirection: 'row',
    gap: 24,
    marginBottom: 12,
  },
  equityItem: {
    alignItems: 'center',
  },
  equityValue: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  equityLabel: {
    color: '#64748b',
    fontSize: 11,
    marginTop: 2,
  },
  equityNote: {
    color: '#94a3b8',
    fontSize: 12,
    lineHeight: 18,
  },
  skinTypeCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
  },
  skinTypeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  skinTypeName: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '500',
  },
  skinTypeAccuracy: {
    color: '#10b981',
    fontSize: 18,
    fontWeight: 'bold',
  },
  skinTypeDetails: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 10,
  },
  skinTypeDetail: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  skinTypeDetailText: {
    color: '#94a3b8',
    fontSize: 12,
  },
  progressBarContainer: {
    height: 6,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#10b981',
    borderRadius: 3,
  },
  feedbackHero: {
    backgroundColor: 'rgba(14,165,233,0.1)',
    borderRadius: 20,
    padding: 24,
    alignItems: 'center',
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(14,165,233,0.3)',
  },
  feedbackHeroLabel: {
    color: '#94a3b8',
    fontSize: 14,
  },
  feedbackHeroValue: {
    color: '#0ea5e9',
    fontSize: 48,
    fontWeight: 'bold',
    marginVertical: 8,
  },
  feedbackHeroSubtext: {
    color: '#64748b',
    fontSize: 13,
  },
  feedbackStats: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  feedbackStat: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 14,
    alignItems: 'center',
  },
  feedbackStatValue: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  feedbackStatLabel: {
    color: '#64748b',
    fontSize: 11,
    marginTop: 4,
  },
  categoryCard: {
    backgroundColor: 'rgba(0,0,0,0.2)',
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
  },
  categoryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  categoryRank: {
    color: '#0ea5e9',
    fontSize: 14,
    fontWeight: 'bold',
  },
  categoryName: {
    color: '#fff',
    fontSize: 13,
    flex: 1,
  },
  categoryStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  categoryCorrections: {
    color: '#64748b',
    fontSize: 12,
  },
  categoryImpact: {
    color: '#10b981',
    fontSize: 12,
    fontWeight: '600',
  },
  helpItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
    marginBottom: 12,
  },
  helpNumber: {
    color: '#0ea5e9',
    fontSize: 14,
    fontWeight: 'bold',
    width: 20,
  },
  helpText: {
    color: '#94a3b8',
    fontSize: 13,
    flex: 1,
    lineHeight: 20,
  },
  ctaCard: {
    backgroundColor: 'rgba(14,165,233,0.1)',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(14,165,233,0.3)',
  },
  ctaText: {
    color: '#0ea5e9',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 12,
    lineHeight: 20,
  },
});
