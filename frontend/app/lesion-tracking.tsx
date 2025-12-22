import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  Alert,
  ActivityIndicator,
  Platform,
  StatusBar
} from 'react-native';
import { useRouter } from 'expo-router';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';

interface LesionGroup {
  id: number;
  lesion_name: string;
  lesion_description: string;
  body_location: string;
  body_sublocation: string;
  body_side: string;
  monitoring_frequency: string;
  next_check_date: string;
  current_risk_level: string;
  requires_attention: boolean;
  attention_reason: string;
  total_analyses: number;
  change_detected: boolean;
  growth_rate: number | null;
  is_active: boolean;
  archived: boolean;
  first_noticed_date: string;
  last_analyzed_at: string;
  created_at: string;
}

export default function LesionTrackingScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const [lesionGroups, setLesionGroups] = useState<LesionGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [showArchived, setShowArchived] = useState(false);

  useEffect(() => {
    fetchLesionGroups();
  }, [showArchived]);

  const fetchLesionGroups = async () => {
    try {
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert(t('lesionTracking.common.error'), t('lesionTracking.common.loginAgain'));
        router.replace('/');
        return;
      }

      const url = `${API_BASE_URL}/lesion_groups/?include_archived=${showArchived}`;
      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setLesionGroups(data);
      } else {
        Alert.alert(t('lesionTracking.common.error'), t('lesionTracking.common.fetchFailed'));
      }
    } catch (error) {
      console.error('Error fetching lesion groups:', error);
      Alert.alert(t('lesionTracking.common.error'), t('lesionTracking.common.networkError'));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchLesionGroups();
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return '#dc2626';
      case 'medium': return '#f59e0b';
      case 'low': return '#10b981';
      default: return '#6b7280';
    }
  };

  const getFrequencyIcon = (frequency: string) => {
    switch (frequency) {
      case 'weekly': return 'calendar';
      case 'monthly': return 'calendar-outline';
      case 'quarterly': return 'calendar-number';
      case 'biannual': return 'calendar-sharp';
      case 'annual': return 'calendar-clear';
      default: return 'calendar';
    }
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return t('lesionTracking.card.notSet');
    const date = new Date(dateString);
    return date.toLocaleDateString();
  };

  const isOverdue = (nextCheckDate: string) => {
    if (!nextCheckDate) return false;
    return new Date(nextCheckDate) < new Date();
  };

  const renderLesionCard = (lesion: LesionGroup) => {
    const overdue = isOverdue(lesion.next_check_date);

    return (
      <TouchableOpacity
        key={lesion.id}
        style={[
          styles.lesionCard,
          lesion.requires_attention && styles.attentionCard,
          overdue && styles.overdueCard
        ]}
        onPress={() => router.push(`/lesion-detail?id=${lesion.id}` as any)}
      >
        {/* Alert Badge */}
        {lesion.requires_attention && (
          <View style={styles.alertBadge}>
            <Ionicons name="warning" size={16} color="white" />
            <Text style={styles.alertBadgeText}>{t('lesionTracking.card.attentionRequired')}</Text>
          </View>
        )}

        {/* Overdue Badge */}
        {overdue && (
          <View style={styles.overdueBadge}>
            <Ionicons name="time" size={16} color="white" />
            <Text style={styles.overdueBadgeText}>{t('lesionTracking.card.checkOverdue')}</Text>
          </View>
        )}

        {/* Header */}
        <View style={styles.cardHeader}>
          <View style={styles.headerLeft}>
            <Text style={styles.lesionName}>{lesion.lesion_name}</Text>
            {lesion.body_location && (
              <Text style={styles.bodyLocation}>
                {lesion.body_location}
                {lesion.body_side && ` (${lesion.body_side})`}
              </Text>
            )}
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

        {/* Description */}
        {lesion.lesion_description && (
          <Text style={styles.description} numberOfLines={2}>
            {lesion.lesion_description}
          </Text>
        )}

        {/* Stats Row */}
        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Ionicons name="images" size={16} color="#6b7280" />
            <Text style={styles.statText}>{lesion.total_analyses} {t('lesionTracking.card.analyses')}</Text>
          </View>

          <View style={styles.statItem}>
            <Ionicons name={getFrequencyIcon(lesion.monitoring_frequency)} size={16} color="#6b7280" />
            <Text style={styles.statText}>{lesion.monitoring_frequency}</Text>
          </View>

          {lesion.change_detected && (
            <View style={styles.statItem}>
              <Ionicons name="alert-circle" size={16} color="#f59e0b" />
              <Text style={[styles.statText, { color: '#f59e0b' }]}>{t('lesionTracking.card.changeDetected')}</Text>
            </View>
          )}
        </View>

        {/* Growth Rate */}
        {lesion.growth_rate !== null && lesion.growth_rate > 0 && (
          <View style={styles.growthRow}>
            <Ionicons name="trending-up" size={16} color="#dc2626" />
            <Text style={styles.growthText}>
              {t('lesionTracking.card.growingAt', { rate: lesion.growth_rate.toFixed(2) })}
            </Text>
          </View>
        )}

        {/* Attention Reason */}
        {lesion.requires_attention && lesion.attention_reason && (
          <View style={styles.attentionReason}>
            <Text style={styles.attentionReasonText}>{lesion.attention_reason}</Text>
          </View>
        )}

        {/* Next Check */}
        <View style={styles.nextCheckRow}>
          <Text style={styles.nextCheckLabel}>{t('lesionTracking.card.nextCheck')}</Text>
          <Text style={[
            styles.nextCheckDate,
            overdue && styles.overdueText
          ]}>
            {formatDate(lesion.next_check_date)}
            {overdue && ` (${t('lesionTracking.card.overdue')})`}
          </Text>
        </View>

        {/* Last Analysis */}
        {lesion.last_analyzed_at && (
          <Text style={styles.lastAnalysis}>
            {t('lesionTracking.card.lastChecked', { date: formatDate(lesion.last_analyzed_at) })}
          </Text>
        )}
      </TouchableOpacity>
    );
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#3b82f6" />
        <Text style={styles.loadingText}>{t('lesionTracking.loading')}</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => router.back()}
          style={styles.backButton}
        >
          <Ionicons name="arrow-back" size={24} color="#1f2937" />
        </TouchableOpacity>
        <Text style={styles.title}>{t('lesionTracking.title')}</Text>
        <TouchableOpacity
          onPress={() => router.push('/create-lesion-group' as any)}
          style={styles.addButton}
        >
          <Ionicons name="add-circle" size={28} color="#3b82f6" />
        </TouchableOpacity>
      </View>

      {/* Info Card */}
      <View style={styles.infoCard}>
        <Ionicons name="information-circle" size={24} color="#3b82f6" />
        <Text style={styles.infoText}>
          {t('lesionTracking.infoText')}
        </Text>
      </View>

      {/* Filter Toggle */}
      <View style={styles.filterRow}>
        <TouchableOpacity
          style={[styles.filterButton, !showArchived && styles.filterButtonActive]}
          onPress={() => setShowArchived(false)}
        >
          <Text style={[styles.filterButtonText, !showArchived && styles.filterButtonTextActive]}>
            {t('lesionTracking.filters.active', { count: lesionGroups.filter(l => !l.archived).length })}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.filterButton, showArchived && styles.filterButtonActive]}
          onPress={() => setShowArchived(true)}
        >
          <Text style={[styles.filterButtonText, showArchived && styles.filterButtonTextActive]}>
            {t('lesionTracking.filters.archived')}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Summary Stats */}
      {!showArchived && lesionGroups.length > 0 && (
        <View style={styles.summaryRow}>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryNumber}>
              {lesionGroups.filter(l => l.requires_attention).length}
            </Text>
            <Text style={styles.summaryLabel}>{t('lesionTracking.summary.needAttention')}</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryNumber}>
              {lesionGroups.filter(l => isOverdue(l.next_check_date)).length}
            </Text>
            <Text style={styles.summaryLabel}>{t('lesionTracking.summary.overdue')}</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryNumber}>
              {lesionGroups.filter(l => l.change_detected).length}
            </Text>
            <Text style={styles.summaryLabel}>{t('lesionTracking.summary.changesDetected')}</Text>
          </View>
        </View>
      )}

      {/* Lesion Cards */}
      {lesionGroups.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="eye-off-outline" size={64} color="#9ca3af" />
          <Text style={styles.emptyTitle}>{t('lesionTracking.emptyState.title')}</Text>
          <Text style={styles.emptyText}>
            {t('lesionTracking.emptyState.subtitle')}
          </Text>
          <TouchableOpacity
            style={styles.emptyButton}
            onPress={() => router.push('/create-lesion-group' as any)}
          >
            <Text style={styles.emptyButtonText}>{t('lesionTracking.emptyState.button')}</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <View style={styles.lesionList}>
          {lesionGroups.map(renderLesionCard)}
        </View>
      )}

      {/* Help Section */}
      <View style={styles.helpSection}>
        <Text style={styles.helpTitle}>{t('lesionTracking.help.title')}</Text>
        <View style={styles.helpItem}>
          <Ionicons name="checkmark-circle" size={20} color="#10b981" />
          <Text style={styles.helpText}>
            {t('lesionTracking.help.step1')}
          </Text>
        </View>
        <View style={styles.helpItem}>
          <Ionicons name="checkmark-circle" size={20} color="#10b981" />
          <Text style={styles.helpText}>
            {t('lesionTracking.help.step2')}
          </Text>
        </View>
        <View style={styles.helpItem}>
          <Ionicons name="checkmark-circle" size={20} color="#10b981" />
          <Text style={styles.helpText}>
            {t('lesionTracking.help.step3')}
          </Text>
        </View>
        <View style={styles.helpItem}>
          <Ionicons name="checkmark-circle" size={20} color="#10b981" />
          <Text style={styles.helpText}>
            {t('lesionTracking.help.step4')}
          </Text>
        </View>
      </View>
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
  backButton: {
    padding: 4
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
    flex: 1,
    textAlign: 'center',
    marginHorizontal: 16
  },
  addButton: {
    padding: 4
  },
  infoCard: {
    margin: 16,
    padding: 16,
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'flex-start'
  },
  infoText: {
    flex: 1,
    marginLeft: 12,
    fontSize: 14,
    color: '#1e40af',
    lineHeight: 20
  },
  filterRow: {
    flexDirection: 'row',
    marginHorizontal: 16,
    marginBottom: 16,
    gap: 12
  },
  filterButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    backgroundColor: 'white',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    alignItems: 'center'
  },
  filterButtonActive: {
    backgroundColor: '#3b82f6',
    borderColor: '#3b82f6'
  },
  filterButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280'
  },
  filterButtonTextActive: {
    color: 'white'
  },
  summaryRow: {
    flexDirection: 'row',
    marginHorizontal: 16,
    marginBottom: 16,
    gap: 12
  },
  summaryCard: {
    flex: 1,
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2
  },
  summaryNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4
  },
  summaryLabel: {
    fontSize: 12,
    color: '#6b7280',
    textAlign: 'center'
  },
  lesionList: {
    paddingHorizontal: 16
  },
  lesionCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  attentionCard: {
    borderWidth: 2,
    borderColor: '#f59e0b'
  },
  overdueCard: {
    borderWidth: 2,
    borderColor: '#dc2626'
  },
  alertBadge: {
    position: 'absolute',
    top: 12,
    right: 12,
    backgroundColor: '#f59e0b',
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    gap: 4
  },
  alertBadgeText: {
    color: 'white',
    fontSize: 11,
    fontWeight: '600'
  },
  overdueBadge: {
    position: 'absolute',
    top: 12,
    left: 12,
    backgroundColor: '#dc2626',
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    gap: 4
  },
  overdueBadgeText: {
    color: 'white',
    fontSize: 11,
    fontWeight: '600'
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8
  },
  headerLeft: {
    flex: 1,
    marginRight: 12
  },
  lesionName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4
  },
  bodyLocation: {
    fontSize: 14,
    color: '#6b7280'
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
  description: {
    fontSize: 14,
    color: '#4b5563',
    marginBottom: 12,
    lineHeight: 20
  },
  statsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
    marginBottom: 8
  },
  statItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6
  },
  statText: {
    fontSize: 13,
    color: '#6b7280'
  },
  growthRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
    padding: 8,
    backgroundColor: '#fee2e2',
    borderRadius: 6
  },
  growthText: {
    fontSize: 13,
    color: '#dc2626',
    fontWeight: '600'
  },
  attentionReason: {
    padding: 10,
    backgroundColor: '#fef3c7',
    borderRadius: 6,
    marginBottom: 8
  },
  attentionReasonText: {
    fontSize: 13,
    color: '#92400e',
    fontWeight: '500'
  },
  nextCheckRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6'
  },
  nextCheckLabel: {
    fontSize: 13,
    color: '#6b7280'
  },
  nextCheckDate: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1f2937'
  },
  overdueText: {
    color: '#dc2626'
  },
  lastAnalysis: {
    fontSize: 12,
    color: '#9ca3af',
    marginTop: 4
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
    paddingHorizontal: 32
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
    marginTop: 16,
    marginBottom: 8
  },
  emptyText: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 20
  },
  emptyButton: {
    backgroundColor: '#3b82f6',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8
  },
  emptyButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600'
  },
  helpSection: {
    margin: 16,
    padding: 16,
    backgroundColor: 'white',
    borderRadius: 12
  },
  helpTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 12
  },
  helpItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
    gap: 8
  },
  helpText: {
    flex: 1,
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20
  }
});
