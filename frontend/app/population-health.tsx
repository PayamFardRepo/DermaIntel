import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');

interface PopulationStats {
  total_analyses: number;
  total_users: number;
  total_high_risk: number;
  average_age?: number;
  gender_distribution: {
    male: number;
    female: number;
    other: number;
  };
}

interface ConditionPrevalence {
  condition: string;
  count: number;
  percentage: number;
  trend: 'up' | 'down' | 'stable';
}

interface RiskDistribution {
  risk_level: string;
  count: number;
  percentage: number;
}

interface GeographicData {
  region: string;
  analyses_count: number;
  high_risk_percentage: number;
  top_conditions: string[];
}

interface DemographicInsight {
  age_group: string;
  common_conditions: string[];
  risk_score: number;
}

export default function PopulationHealthScreen() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const { t } = useTranslation();

  const [isLoading, setIsLoading] = useState(true);
  const [stats, setStats] = useState<PopulationStats | null>(null);
  const [prevalence, setPrevalence] = useState<ConditionPrevalence[]>([]);
  const [riskDistribution, setRiskDistribution] = useState<RiskDistribution[]>([]);
  const [geographicData, setGeographicData] = useState<GeographicData[]>([]);
  const [demographics, setDemographics] = useState<DemographicInsight[]>([]);
  const [timeRange, setTimeRange] = useState<string>('30days');

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    } else {
      loadPopulationData();
    }
  }, [isAuthenticated, timeRange]);

  const loadPopulationData = async () => {
    setIsLoading(true);

    if (!user?.token) {
      console.error('No authentication token available');
      setIsLoading(false);
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/population-health/dashboard?time_range=${timeRange}`,
        {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${user.token}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setStats(data.stats || null);
        setPrevalence(data.condition_prevalence || []);
        setRiskDistribution(data.risk_distribution || []);
        setGeographicData(data.geographic_data || []);
        setDemographics(data.demographic_insights || []);
      } else {
        console.log('Failed to load population health data. Status:', response.status);
      }
    } catch (error) {
      console.error('Error loading population health data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return 'trending-up';
      case 'down':
        return 'trending-down';
      default:
        return 'remove';
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'up':
        return '#dc3545';
      case 'down':
        return '#28a745';
      default:
        return '#6c757d';
    }
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'high':
      case 'very_high':
        return '#dc3545';
      case 'medium':
      case 'moderate':
        return '#ffc107';
      default:
        return '#28a745';
    }
  };

  const renderOverviewStats = () => {
    if (!stats) return null;

    return (
      <View style={styles.statsContainer}>
        <Text style={styles.sectionTitle}>{t('populationHealth.overview')}</Text>
        <View style={styles.statsGrid}>
          <View style={styles.statCard}>
            <Ionicons name="people" size={32} color="#4299e1" />
            <Text style={styles.statNumber}>{stats.total_users.toLocaleString()}</Text>
            <Text style={styles.statLabel}>{t('populationHealth.totalUsers')}</Text>
          </View>

          <View style={styles.statCard}>
            <Ionicons name="analytics" size={32} color="#9333ea" />
            <Text style={styles.statNumber}>{stats.total_analyses.toLocaleString()}</Text>
            <Text style={styles.statLabel}>{t('populationHealth.totalAnalyses')}</Text>
          </View>

          <View style={[styles.statCard, styles.warningCard]}>
            <Ionicons name="warning" size={32} color="#dc3545" />
            <Text style={[styles.statNumber, styles.warningText]}>
              {stats.total_high_risk.toLocaleString()}
            </Text>
            <Text style={styles.statLabel}>{t('populationHealth.highRiskCases')}</Text>
          </View>

          {stats.average_age && (
            <View style={styles.statCard}>
              <Ionicons name="calendar" size={32} color="#16a34a" />
              <Text style={styles.statNumber}>{Math.round(stats.average_age)}</Text>
              <Text style={styles.statLabel}>{t('populationHealth.averageAge')}</Text>
            </View>
          )}
        </View>

        {/* Gender Distribution */}
        {stats.gender_distribution && (
          <View style={styles.genderContainer}>
            <Text style={styles.subsectionTitle}>{t('populationHealth.genderDistribution')}</Text>
            <View style={styles.genderBars}>
              <View style={styles.genderRow}>
                <Ionicons name="male" size={20} color="#4299e1" />
                <View style={styles.genderBarContainer}>
                  <View
                    style={[
                      styles.genderBar,
                      {
                        width: `${stats.gender_distribution.male}%`,
                        backgroundColor: '#4299e1',
                      },
                    ]}
                  />
                </View>
                <Text style={styles.genderPercent}>{stats.gender_distribution.male}%</Text>
              </View>

              <View style={styles.genderRow}>
                <Ionicons name="female" size={20} color="#ec4899" />
                <View style={styles.genderBarContainer}>
                  <View
                    style={[
                      styles.genderBar,
                      {
                        width: `${stats.gender_distribution.female}%`,
                        backgroundColor: '#ec4899',
                      },
                    ]}
                  />
                </View>
                <Text style={styles.genderPercent}>{stats.gender_distribution.female}%</Text>
              </View>

              {stats.gender_distribution.other > 0 && (
                <View style={styles.genderRow}>
                  <Ionicons name="person" size={20} color="#6c757d" />
                  <View style={styles.genderBarContainer}>
                    <View
                      style={[
                        styles.genderBar,
                        {
                          width: `${stats.gender_distribution.other}%`,
                          backgroundColor: '#6c757d',
                        },
                      ]}
                    />
                  </View>
                  <Text style={styles.genderPercent}>{stats.gender_distribution.other}%</Text>
                </View>
              )}
            </View>
          </View>
        )}
      </View>
    );
  };

  const renderConditionPrevalence = () => {
    if (prevalence.length === 0) return null;

    return (
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('populationHealth.conditionPrevalence')}</Text>
        <Text style={styles.sectionSubtitle}>{t('populationHealth.mostCommonConditions')}</Text>

        {prevalence.map((item, index) => (
          <View key={index} style={styles.prevalenceCard}>
            <View style={styles.prevalenceHeader}>
              <Text style={styles.prevalenceRank}>#{index + 1}</Text>
              <Text style={styles.prevalenceName}>{item.condition}</Text>
              <View style={styles.trendBadge}>
                <Ionicons
                  name={getTrendIcon(item.trend) as any}
                  size={16}
                  color={getTrendColor(item.trend)}
                />
              </View>
            </View>

            <View style={styles.prevalenceStats}>
              <Text style={styles.prevalenceCount}>
                {item.count.toLocaleString()} {t('populationHealth.cases')}
              </Text>
              <Text style={styles.prevalencePercent}>{item.percentage.toFixed(1)}%</Text>
            </View>

            <View style={styles.prevalenceBarContainer}>
              <View
                style={[
                  styles.prevalenceBar,
                  { width: `${item.percentage}%` },
                ]}
              />
            </View>
          </View>
        ))}
      </View>
    );
  };

  const renderRiskDistribution = () => {
    if (riskDistribution.length === 0) return null;

    return (
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('populationHealth.riskDistribution')}</Text>
        <Text style={styles.sectionSubtitle}>{t('populationHealth.riskLevelBreakdown')}</Text>

        <View style={styles.riskPieContainer}>
          {riskDistribution.map((item, index) => (
            <View key={index} style={styles.riskSegment}>
              <View
                style={[
                  styles.riskIndicator,
                  { backgroundColor: getRiskColor(item.risk_level) },
                ]}
              />
              <View style={styles.riskInfo}>
                <Text style={styles.riskLevel}>{item.risk_level.toUpperCase()}</Text>
                <Text style={styles.riskCount}>
                  {item.count.toLocaleString()} ({item.percentage.toFixed(1)}%)
                </Text>
              </View>
              <View style={styles.riskBarWrapper}>
                <View
                  style={[
                    styles.riskBar,
                    {
                      width: `${item.percentage}%`,
                      backgroundColor: getRiskColor(item.risk_level),
                    },
                  ]}
                />
              </View>
            </View>
          ))}
        </View>
      </View>
    );
  };

  const renderGeographicData = () => {
    if (geographicData.length === 0) return null;

    return (
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('populationHealth.geographicInsights')}</Text>
        <Text style={styles.sectionSubtitle}>{t('populationHealth.regionalData')}</Text>

        {geographicData.slice(0, 5).map((region, index) => (
          <View key={index} style={styles.regionCard}>
            <View style={styles.regionHeader}>
              <Ionicons name="location" size={20} color="#4299e1" />
              <Text style={styles.regionName}>{region.region}</Text>
            </View>

            <View style={styles.regionStats}>
              <View style={styles.regionStat}>
                <Text style={styles.regionStatLabel}>{t('populationHealth.analyses')}</Text>
                <Text style={styles.regionStatValue}>{region.analyses_count}</Text>
              </View>

              <View style={styles.regionStat}>
                <Text style={styles.regionStatLabel}>{t('populationHealth.highRiskRate')}</Text>
                <Text style={[styles.regionStatValue, styles.riskValue]}>
                  {region.high_risk_percentage.toFixed(1)}%
                </Text>
              </View>
            </View>

            {region.top_conditions && region.top_conditions.length > 0 && (
              <View style={styles.topConditions}>
                <Text style={styles.topConditionsLabel}>
                  {t('populationHealth.topConditions')}:
                </Text>
                <Text style={styles.topConditionsList}>
                  {region.top_conditions.slice(0, 3).join(', ')}
                </Text>
              </View>
            )}
          </View>
        ))}
      </View>
    );
  };

  const renderDemographics = () => {
    if (demographics.length === 0) return null;

    return (
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('populationHealth.demographicInsights')}</Text>
        <Text style={styles.sectionSubtitle}>{t('populationHealth.ageGroupAnalysis')}</Text>

        {demographics.map((demo, index) => (
          <View key={index} style={styles.demoCard}>
            <View style={styles.demoHeader}>
              <Text style={styles.demoAgeGroup}>{demo.age_group}</Text>
              <View
                style={[
                  styles.riskScoreBadge,
                  { backgroundColor: demo.risk_score > 60 ? '#dc3545' : demo.risk_score > 30 ? '#ffc107' : '#28a745' },
                ]}
              >
                <Text style={styles.riskScoreText}>{t('populationHealth.riskScore')}: {demo.risk_score}</Text>
              </View>
            </View>

            {demo.common_conditions && demo.common_conditions.length > 0 && (
              <View style={styles.demoConditions}>
                <Text style={styles.demoConditionsLabel}>
                  {t('populationHealth.commonConditions')}:
                </Text>
                {demo.common_conditions.map((condition, idx) => (
                  <View key={idx} style={styles.conditionTag}>
                    <Text style={styles.conditionTagText}>{condition}</Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        ))}
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.background}
      />

      {/* Header */}
      <View style={styles.header}>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Text style={styles.backButtonText}>← {t('common.back')}</Text>
        </Pressable>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>📊 {t('populationHealth.title')}</Text>
          <Text style={styles.headerSubtitle}>{t('populationHealth.subtitle')}</Text>
        </View>
      </View>

      {/* Time Range Selector */}
      <View style={styles.timeRangeContainer}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {['7days', '30days', '90days', '1year', 'all'].map((range) => (
            <Pressable
              key={range}
              style={[
                styles.timeButton,
                timeRange === range && styles.timeButtonActive,
              ]}
              onPress={() => setTimeRange(range)}
            >
              <Text
                style={[
                  styles.timeButtonText,
                  timeRange === range && styles.timeButtonTextActive,
                ]}
              >
                {t(`populationHealth.${range}`)}
              </Text>
            </Pressable>
          ))}
        </ScrollView>
      </View>

      <ScrollView style={styles.content}>
        {isLoading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#4299e1" />
            <Text style={styles.loadingText}>{t('populationHealth.loading')}</Text>
          </View>
        ) : (
          <>
            {renderOverviewStats()}
            {renderConditionPrevalence()}
            {renderRiskDistribution()}
            {renderGeographicData()}
            {renderDemographics()}
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
  background: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.2)',
  },
  backButton: {
    backgroundColor: 'rgba(66, 153, 225, 0.9)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
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
  timeRangeContainer: {
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  timeButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  timeButtonActive: {
    backgroundColor: '#4299e1',
    borderColor: '#4299e1',
  },
  timeButtonText: {
    fontSize: 13,
    color: '#4a5568',
    fontWeight: '600',
  },
  timeButtonTextActive: {
    color: '#fff',
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
  },
  loadingContainer: {
    paddingVertical: 60,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#4a5568',
  },
  statsContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: '#718096',
    marginBottom: 16,
  },
  subsectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c5282',
    marginTop: 16,
    marginBottom: 12,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  statCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: '#f7fafc',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  warningCard: {
    backgroundColor: '#fff3cd',
  },
  statNumber: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c5282',
    marginTop: 8,
  },
  warningText: {
    color: '#856404',
  },
  statLabel: {
    fontSize: 12,
    color: '#4a5568',
    textAlign: 'center',
    marginTop: 4,
  },
  genderContainer: {
    marginTop: 16,
  },
  genderBars: {
    marginTop: 8,
  },
  genderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  genderBarContainer: {
    flex: 1,
    height: 24,
    backgroundColor: '#e2e8f0',
    borderRadius: 12,
    marginHorizontal: 12,
    overflow: 'hidden',
  },
  genderBar: {
    height: '100%',
    borderRadius: 12,
  },
  genderPercent: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2c5282',
    minWidth: 45,
    textAlign: 'right',
  },
  section: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  prevalenceCard: {
    marginBottom: 16,
    padding: 16,
    backgroundColor: '#f7fafc',
    borderRadius: 12,
  },
  prevalenceHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  prevalenceRank: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#4299e1',
    marginRight: 12,
  },
  prevalenceName: {
    flex: 1,
    fontSize: 16,
    fontWeight: '600',
    color: '#2c5282',
  },
  trendBadge: {
    padding: 4,
  },
  prevalenceStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  prevalenceCount: {
    fontSize: 14,
    color: '#4a5568',
  },
  prevalencePercent: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  prevalenceBarContainer: {
    height: 8,
    backgroundColor: '#e2e8f0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  prevalenceBar: {
    height: '100%',
    backgroundColor: '#4299e1',
    borderRadius: 4,
  },
  riskPieContainer: {
    marginTop: 8,
  },
  riskSegment: {
    marginBottom: 16,
  },
  riskIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginBottom: 8,
  },
  riskInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  riskLevel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2c5282',
  },
  riskCount: {
    fontSize: 14,
    color: '#4a5568',
  },
  riskBarWrapper: {
    height: 24,
    backgroundColor: '#e2e8f0',
    borderRadius: 12,
    overflow: 'hidden',
  },
  riskBar: {
    height: '100%',
    borderRadius: 12,
  },
  regionCard: {
    marginBottom: 16,
    padding: 16,
    backgroundColor: '#f7fafc',
    borderRadius: 12,
  },
  regionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  regionName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c5282',
    marginLeft: 8,
  },
  regionStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 12,
  },
  regionStat: {
    alignItems: 'center',
  },
  regionStatLabel: {
    fontSize: 12,
    color: '#718096',
    marginBottom: 4,
  },
  regionStatValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  riskValue: {
    color: '#dc3545',
  },
  topConditions: {
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    paddingTop: 12,
  },
  topConditionsLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#4a5568',
    marginBottom: 4,
  },
  topConditionsList: {
    fontSize: 13,
    color: '#2c5282',
  },
  demoCard: {
    marginBottom: 16,
    padding: 16,
    backgroundColor: '#f7fafc',
    borderRadius: 12,
  },
  demoHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  demoAgeGroup: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  riskScoreBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  riskScoreText: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#fff',
  },
  demoConditions: {
    marginTop: 8,
  },
  demoConditionsLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#4a5568',
    marginBottom: 8,
  },
  conditionTag: {
    backgroundColor: '#e6f2ff',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    marginRight: 8,
    marginBottom: 6,
    alignSelf: 'flex-start',
  },
  conditionTagText: {
    fontSize: 12,
    color: '#2c5282',
    fontWeight: '600',
  },
});
