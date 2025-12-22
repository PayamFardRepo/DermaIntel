import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  ActivityIndicator,
  Dimensions,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');

interface ProgressionEvent {
  id: number;
  lesion_group_id: number;
  lesion_name: string;
  body_location: string;
  event_type: string; // 'analysis', 'change_detected', 'risk_increase', 'alert'
  event_date: string;
  analysis_id?: number;
  predicted_class?: string;
  risk_level?: string;
  change_severity?: string;
  growth_rate?: number;
  description: string;
}

interface TimelineStats {
  total_lesions: number;
  total_analyses: number;
  changes_detected: number;
  high_risk_lesions: number;
  lesions_requiring_attention: number;
}

export default function ProgressionTimelineScreen() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const { t } = useTranslation();

  const [events, setEvents] = useState<ProgressionEvent[]>([]);
  const [stats, setStats] = useState<TimelineStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [filterType, setFilterType] = useState<string>('all'); // 'all', 'changes', 'alerts', 'analyses'
  const [timeRange, setTimeRange] = useState<number>(90); // days

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    } else {
      loadProgressionData();
    }
  }, [isAuthenticated, filterType, timeRange]);

  const loadProgressionData = async () => {
    setIsLoading(true);

    if (!user?.token) {
      console.error('No authentication token available');
      setIsLoading(false);
      return;
    }

    try {
      console.log('Loading progression data with token:', user.token.substring(0, 20) + '...');
      const response = await fetch(
        `${API_BASE_URL}/progression/timeline?days=${timeRange}&filter=${filterType}`,
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
        setEvents(data.events || []);
        setStats(data.stats || null);
      } else {
        console.log('Failed to load progression data. Status:', response.status);
        const errorText = await response.text();
        console.log('Error response:', errorText);
        setEvents([]);
        setStats(null);
      }
    } catch (error) {
      console.error('Error loading progression data:', error);
      setEvents([]);
      setStats(null);
    } finally {
      setIsLoading(false);
    }
  };

  const getEventIcon = (eventType: string) => {
    switch (eventType) {
      case 'analysis':
        return 'camera-outline';
      case 'change_detected':
        return 'trending-up-outline';
      case 'risk_increase':
        return 'warning-outline';
      case 'alert':
        return 'notifications-outline';
      case 'lesion_created':
        return 'add-circle-outline';
      default:
        return 'ellipse-outline';
    }
  };

  const getEventColor = (eventType: string, severity?: string) => {
    if (eventType === 'alert' || eventType === 'risk_increase') {
      return '#dc3545';
    }
    if (eventType === 'change_detected') {
      if (severity === 'concerning' || severity === 'significant') {
        return '#ffc107';
      }
      return '#17a2b8';
    }
    if (eventType === 'lesion_created') {
      return '#28a745';
    }
    return '#6c757d';
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return t('progressionTimeline.today');
    if (diffDays === 1) return t('progressionTimeline.yesterday');
    if (diffDays < 7) return t('progressionTimeline.daysAgo', { count: diffDays });
    if (diffDays < 30) return t('progressionTimeline.weeksAgo', { count: Math.floor(diffDays / 7) });
    if (diffDays < 365) return t('progressionTimeline.monthsAgo', { count: Math.floor(diffDays / 30) });
    return date.toLocaleDateString();
  };

  const renderStats = () => {
    if (!stats) return null;

    return (
      <View style={styles.statsContainer}>
        <Text style={styles.statsTitle}>{t('progressionTimeline.overview')}</Text>
        <View style={styles.statsGrid}>
          <View style={styles.statCard}>
            <Text style={styles.statNumber}>{stats.total_lesions}</Text>
            <Text style={styles.statLabel}>{t('progressionTimeline.trackedLesions')}</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statNumber}>{stats.total_analyses}</Text>
            <Text style={styles.statLabel}>{t('progressionTimeline.totalAnalyses')}</Text>
          </View>
          <View style={[styles.statCard, styles.warningCard]}>
            <Text style={[styles.statNumber, styles.warningText]}>{stats.changes_detected}</Text>
            <Text style={styles.statLabel}>{t('progressionTimeline.changesDetected')}</Text>
          </View>
          <View style={[styles.statCard, styles.dangerCard]}>
            <Text style={[styles.statNumber, styles.dangerText]}>{stats.lesions_requiring_attention}</Text>
            <Text style={styles.statLabel}>{t('progressionTimeline.needsAttention')}</Text>
          </View>
        </View>
      </View>
    );
  };

  const renderFilters = () => {
    return (
      <View style={styles.filtersContainer}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <Pressable
            style={[styles.filterButton, filterType === 'all' && styles.filterButtonActive]}
            onPress={() => setFilterType('all')}
          >
            <Text style={[styles.filterButtonText, filterType === 'all' && styles.filterButtonTextActive]}>
              {t('progressionTimeline.all')}
            </Text>
          </Pressable>
          <Pressable
            style={[styles.filterButton, filterType === 'changes' && styles.filterButtonActive]}
            onPress={() => setFilterType('changes')}
          >
            <Text style={[styles.filterButtonText, filterType === 'changes' && styles.filterButtonTextActive]}>
              {t('progressionTimeline.changes')}
            </Text>
          </Pressable>
          <Pressable
            style={[styles.filterButton, filterType === 'alerts' && styles.filterButtonActive]}
            onPress={() => setFilterType('alerts')}
          >
            <Text style={[styles.filterButtonText, filterType === 'alerts' && styles.filterButtonTextActive]}>
              {t('progressionTimeline.alerts')}
            </Text>
          </Pressable>
          <Pressable
            style={[styles.filterButton, filterType === 'analyses' && styles.filterButtonActive]}
            onPress={() => setFilterType('analyses')}
          >
            <Text style={[styles.filterButtonText, filterType === 'analyses' && styles.filterButtonTextActive]}>
              {t('progressionTimeline.analyses')}
            </Text>
          </Pressable>
        </ScrollView>

        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.timeRangeContainer}>
          {[30, 90, 180, 365].map((days) => (
            <Pressable
              key={days}
              style={[styles.timeButton, timeRange === days && styles.timeButtonActive]}
              onPress={() => setTimeRange(days)}
            >
              <Text style={[styles.timeButtonText, timeRange === days && styles.timeButtonTextActive]}>
                {days === 30 && t('progressionTimeline.30days')}
                {days === 90 && t('progressionTimeline.90days')}
                {days === 180 && t('progressionTimeline.6months')}
                {days === 365 && t('progressionTimeline.1year')}
              </Text>
            </Pressable>
          ))}
        </ScrollView>
      </View>
    );
  };

  const renderTimeline = () => {
    if (events.length === 0) {
      return (
        <View style={styles.emptyState}>
          <Ionicons name="time-outline" size={64} color="#cbd5e0" />
          <Text style={styles.emptyStateTitle}>{t('progressionTimeline.noEvents')}</Text>
          <Text style={styles.emptyStateSubtext}>{t('progressionTimeline.noEventsSubtext')}</Text>
        </View>
      );
    }

    return (
      <View style={styles.timelineContainer}>
        {events.map((event, index) => (
          <View key={event.id} style={styles.timelineEvent}>
            <View style={styles.timelineLeft}>
              <View
                style={[
                  styles.timelineIcon,
                  { backgroundColor: getEventColor(event.event_type, event.change_severity) },
                ]}
              >
                <Ionicons
                  name={getEventIcon(event.event_type) as any}
                  size={20}
                  color="#fff"
                />
              </View>
              {index < events.length - 1 && <View style={styles.timelineLine} />}
            </View>

            <Pressable
              style={styles.timelineCard}
              onPress={() => router.push(`/lesion-detail?id=${event.lesion_group_id}`)}
            >
              <View style={styles.timelineCardHeader}>
                <Text style={styles.lesionName}>{event.lesion_name}</Text>
                <Text style={styles.eventDate}>{formatDate(event.event_date)}</Text>
              </View>
              <Text style={styles.bodyLocation}>📍 {event.body_location}</Text>
              <Text style={styles.eventDescription}>{event.description}</Text>

              {event.predicted_class && (
                <View style={styles.eventDetails}>
                  <Text style={styles.detailLabel}>{t('progressionTimeline.diagnosis')}:</Text>
                  <Text style={styles.detailValue}>{event.predicted_class}</Text>
                </View>
              )}

              {event.risk_level && (
                <View style={styles.riskBadge}>
                  <Text style={styles.riskBadgeText}>
                    {t('progressionTimeline.risk')}: {event.risk_level.toUpperCase()}
                  </Text>
                </View>
              )}

              {event.growth_rate !== null && event.growth_rate !== undefined && (
                <View style={styles.growthInfo}>
                  <Ionicons name="trending-up" size={16} color="#dc3545" />
                  <Text style={styles.growthText}>
                    {t('progressionTimeline.growthRate')}: {event.growth_rate.toFixed(2)} mm²/month
                  </Text>
                </View>
              )}
            </Pressable>
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
          <Text style={styles.headerTitle}>📊 {t('progressionTimeline.title')}</Text>
          <Text style={styles.headerSubtitle}>{t('progressionTimeline.subtitle')}</Text>
        </View>
      </View>

      <ScrollView style={styles.content}>
        {isLoading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#4299e1" />
            <Text style={styles.loadingText}>{t('progressionTimeline.loading')}</Text>
          </View>
        ) : (
          <>
            {renderStats()}
            {renderFilters()}
            {renderTimeline()}
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
  content: {
    flex: 1,
    paddingHorizontal: 20,
    paddingVertical: 20,
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
  statsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 16,
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
  dangerCard: {
    backgroundColor: '#f8d7da',
  },
  statNumber: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#4299e1',
  },
  warningText: {
    color: '#856404',
  },
  dangerText: {
    color: '#721c24',
  },
  statLabel: {
    fontSize: 12,
    color: '#4a5568',
    textAlign: 'center',
    marginTop: 4,
  },
  filtersContainer: {
    marginBottom: 20,
  },
  filterButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    marginRight: 10,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  filterButtonActive: {
    backgroundColor: '#4299e1',
    borderColor: '#4299e1',
  },
  filterButtonText: {
    fontSize: 14,
    color: '#4a5568',
    fontWeight: '600',
  },
  filterButtonTextActive: {
    color: '#fff',
  },
  timeRangeContainer: {
    marginTop: 10,
  },
  timeButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 16,
    marginRight: 8,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  timeButtonActive: {
    backgroundColor: '#2d3748',
    borderColor: '#2d3748',
  },
  timeButtonText: {
    fontSize: 13,
    color: '#4a5568',
  },
  timeButtonTextActive: {
    color: '#fff',
  },
  timelineContainer: {
    paddingTop: 10,
  },
  timelineEvent: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  timelineLeft: {
    alignItems: 'center',
    marginRight: 16,
  },
  timelineIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  timelineLine: {
    width: 2,
    flex: 1,
    backgroundColor: '#e2e8f0',
    marginTop: 8,
  },
  timelineCard: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  timelineCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  lesionName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2d3748',
    flex: 1,
  },
  eventDate: {
    fontSize: 12,
    color: '#718096',
  },
  bodyLocation: {
    fontSize: 13,
    color: '#4a5568',
    marginBottom: 8,
  },
  eventDescription: {
    fontSize: 14,
    color: '#2d3748',
    lineHeight: 20,
    marginBottom: 12,
  },
  eventDetails: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  detailLabel: {
    fontSize: 13,
    color: '#718096',
    marginRight: 8,
  },
  detailValue: {
    fontSize: 13,
    color: '#2d3748',
    fontWeight: '600',
  },
  riskBadge: {
    backgroundColor: '#fed7d7',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    alignSelf: 'flex-start',
    marginTop: 8,
  },
  riskBadgeText: {
    fontSize: 11,
    color: '#9b2c2c',
    fontWeight: 'bold',
  },
  growthInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  growthText: {
    fontSize: 12,
    color: '#dc3545',
    marginLeft: 6,
    fontWeight: '600',
  },
  emptyState: {
    paddingVertical: 80,
    alignItems: 'center',
  },
  emptyStateTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#4a5568',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#718096',
    textAlign: 'center',
    paddingHorizontal: 40,
  },
});
