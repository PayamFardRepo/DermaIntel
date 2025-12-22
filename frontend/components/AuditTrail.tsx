import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  Pressable,
  RefreshControl,
  Modal
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useTranslation } from 'react-i18next';
import { API_BASE_URL } from '../config';
import authService from '../services/AuthService';

interface AuditLog {
  id: number;
  event_type: string;
  event_category: string;
  severity: string;
  action: string;
  model_name: string;
  model_version: string;
  prediction_result: string;
  confidence_score: number;
  analysis_id: number;
  processing_time_ms: number;
  quality_passed: boolean;
  reliability_score: number;
  flags: string[];
  endpoint: string;
  http_method: string;
  response_status: number;
  error_occurred: boolean;
  error_message: string;
  created_at: string;
  ip_address: string;
}

interface AuditStats {
  total_logs: number;
  event_types: Record<string, number>;
  severities: Record<string, number>;
  average_confidence: number;
  average_processing_time_ms: number;
  error_count: number;
}

/**
 * Audit Trail Component
 * Displays AI prediction logs for quality assurance and legal documentation
 */
export default function AuditTrail() {
  const { t } = useTranslation();
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [stats, setStats] = useState<AuditStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedLog, setSelectedLog] = useState<any>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [filter, setFilter] = useState({
    severity: null as string | null,
    event_type: null as string | null
  });

  useEffect(() => {
    fetchAuditData();
  }, [filter]);

  const fetchAuditData = async () => {
    try {
      const token = await authService.getToken();
      if (!token) return;

      // Fetch logs
      let logsUrl = `${API_BASE_URL}/audit/logs?limit=50`;
      if (filter.severity) logsUrl += `&severity=${filter.severity}`;
      if (filter.event_type) logsUrl += `&event_type=${filter.event_type}`;

      const logsResponse = await fetch(logsUrl, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (logsResponse.ok) {
        const logsData = await logsResponse.json();
        setLogs(logsData.logs || []);
      }

      // Fetch stats
      const statsResponse = await fetch(`${API_BASE_URL}/audit/stats`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        setStats(statsData);
      }
    } catch (error) {
      console.error('Error fetching audit data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchAuditData();
  };

  const viewLogDetail = async (logId: number) => {
    try {
      const token = await authService.getToken();
      if (!token) return;

      const response = await fetch(`${API_BASE_URL}/audit/logs/${logId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        setSelectedLog(data);
        setDetailModalVisible(true);
      }
    } catch (error) {
      console.error('Error fetching log detail:', error);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'info': return '#3b82f6';
      case 'warning': return '#f59e0b';
      case 'error': return '#ef4444';
      case 'critical': return '#dc2626';
      default: return '#6b7280';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'info': return 'information-circle';
      case 'warning': return 'warning';
      case 'error': return 'close-circle';
      case 'critical': return 'alert-circle';
      default: return 'help-circle';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const exportToCSV = async () => {
    try {
      const token = await authService.getToken();
      if (!token) return;

      const url = `${API_BASE_URL}/audit/export/csv`;
      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const blob = await response.blob();
        // Note: In a real mobile app, you would use react-native-fs or expo-file-system
        // to save the file. For now, we'll just show an alert.
        alert(t('audit.exportSuccess'));
      } else {
        alert(t('audit.exportFailed'));
      }
    } catch (error) {
      console.error('Error exporting audit logs:', error);
      alert(t('audit.exportError'));
    }
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#3b82f6" />
        <Text style={styles.loadingText}>{t('audit.loadingText')}</Text>
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
      {/* Statistics Cards */}
      {stats && (
        <View style={styles.statsContainer}>
          <Text style={styles.sectionTitle}>{t('audit.sectionTitle')}</Text>

          <View style={styles.statsGrid}>
            <View style={styles.statCard}>
              <Text style={styles.statValue}>{stats.total_logs}</Text>
              <Text style={styles.statLabel}>{t('audit.totalLogs')}</Text>
            </View>

            <View style={styles.statCard}>
              <Text style={styles.statValue}>{stats.error_count}</Text>
              <Text style={styles.statLabel}>{t('audit.errors')}</Text>
            </View>

            <View style={styles.statCard}>
              <Text style={styles.statValue}>
                {stats.average_confidence ? (stats.average_confidence * 100).toFixed(1) + '%' : 'N/A'}
              </Text>
              <Text style={styles.statLabel}>{t('audit.avgConfidence')}</Text>
            </View>

            <View style={styles.statCard}>
              <Text style={styles.statValue}>
                {stats.average_processing_time_ms ? (stats.average_processing_time_ms / 1000).toFixed(1) + 's' : 'N/A'}
              </Text>
              <Text style={styles.statLabel}>{t('audit.avgTime')}</Text>
            </View>
          </View>
        </View>
      )}

      {/* Filters */}
      <View style={styles.filtersContainer}>
        <Text style={styles.sectionTitle}>{t('audit.filtersTitle')}</Text>

        <View style={styles.filterRow}>
          <Pressable
            style={[styles.filterButton, !filter.severity && styles.filterButtonActive]}
            onPress={() => setFilter({ ...filter, severity: null })}
          >
            <Text style={[styles.filterButtonText, !filter.severity && styles.filterButtonTextActive]}>
              {t('audit.all')}
            </Text>
          </Pressable>

          <Pressable
            style={[styles.filterButton, filter.severity === 'info' && styles.filterButtonActive]}
            onPress={() => setFilter({ ...filter, severity: 'info' })}
          >
            <Text style={[styles.filterButtonText, filter.severity === 'info' && styles.filterButtonTextActive]}>
              {t('audit.info')}
            </Text>
          </Pressable>

          <Pressable
            style={[styles.filterButton, filter.severity === 'warning' && styles.filterButtonActive]}
            onPress={() => setFilter({ ...filter, severity: 'warning' })}
          >
            <Text style={[styles.filterButtonText, filter.severity === 'warning' && styles.filterButtonTextActive]}>
              {t('audit.warning')}
            </Text>
          </Pressable>

          <Pressable
            style={[styles.filterButton, filter.severity === 'error' && styles.filterButtonActive]}
            onPress={() => setFilter({ ...filter, severity: 'error' })}
          >
            <Text style={[styles.filterButtonText, filter.severity === 'error' && styles.filterButtonTextActive]}>
              {t('audit.error')}
            </Text>
          </Pressable>
        </View>
      </View>

      {/* Audit Logs List */}
      <View style={styles.logsContainer}>
        <View style={styles.logsHeader}>
          <Text style={styles.sectionTitle}>{t('audit.recentLogs')}</Text>
          <Pressable style={styles.exportButton} onPress={exportToCSV}>
            <Ionicons name="download-outline" size={20} color="#fff" />
            <Text style={styles.exportButtonText}>{t('audit.exportCSV')}</Text>
          </Pressable>
        </View>

        {logs.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons name="document-text-outline" size={64} color="#9ca3af" />
            <Text style={styles.emptyStateText}>{t('audit.noLogsFound')}</Text>
          </View>
        ) : (
          logs.map((log) => (
            <Pressable
              key={log.id}
              style={styles.logCard}
              onPress={() => viewLogDetail(log.id)}
            >
              <View style={styles.logHeader}>
                <View style={styles.logHeaderLeft}>
                  <Ionicons
                    name={getSeverityIcon(log.severity)}
                    size={20}
                    color={getSeverityColor(log.severity)}
                  />
                  <Text style={styles.logEventType}>{log.event_type}</Text>
                  {log.flags && log.flags.length > 0 && (
                    <View style={styles.flagBadge}>
                      <Text style={styles.flagBadgeText}>{log.flags.length}</Text>
                    </View>
                  )}
                </View>
                <Text style={styles.logDate}>{formatDate(log.created_at)}</Text>
              </View>

              <View style={styles.logBody}>
                {log.model_name && (
                  <Text style={styles.logDetail}>
                    {t('audit.model')}: <Text style={styles.logDetailValue}>{log.model_name}</Text>
                  </Text>
                )}

                {log.prediction_result && (
                  <Text style={styles.logDetail}>
                    {t('audit.prediction')}: <Text style={styles.logDetailValue}>{log.prediction_result}</Text>
                  </Text>
                )}

                {log.confidence_score !== null && (
                  <Text style={styles.logDetail}>
                    {t('audit.confidence')}: <Text style={styles.logDetailValue}>{(log.confidence_score * 100).toFixed(1)}%</Text>
                  </Text>
                )}

                {log.processing_time_ms && (
                  <Text style={styles.logDetail}>
                    {t('audit.processingTime')}: <Text style={styles.logDetailValue}>{(log.processing_time_ms / 1000).toFixed(2)}s</Text>
                  </Text>
                )}

                {log.error_occurred && log.error_message && (
                  <View style={styles.errorBox}>
                    <Text style={styles.errorText}>{log.error_message}</Text>
                  </View>
                )}
              </View>

              <View style={styles.logFooter}>
                <Text style={styles.logEndpoint}>{log.http_method} {log.endpoint}</Text>
                <Ionicons name="chevron-forward" size={16} color="#9ca3af" />
              </View>
            </Pressable>
          ))
        )}
      </View>

      {/* Detail Modal */}
      <Modal
        visible={detailModalVisible}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setDetailModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>{t('audit.logDetails')}</Text>
              <Pressable onPress={() => setDetailModalVisible(false)}>
                <Ionicons name="close" size={24} color="#6b7280" />
              </Pressable>
            </View>

            <ScrollView style={styles.modalBody}>
              {selectedLog && (
                <>
                  <View style={styles.detailSection}>
                    <Text style={styles.detailSectionTitle}>{t('audit.eventInformation')}</Text>
                    <Text style={styles.detailRow}>{t('audit.type')}: {selectedLog.event_type}</Text>
                    <Text style={styles.detailRow}>{t('audit.category')}: {selectedLog.event_category}</Text>
                    <Text style={styles.detailRow}>{t('audit.severity')}: {selectedLog.severity}</Text>
                    <Text style={styles.detailRow}>{t('audit.timestamp')}: {formatDate(selectedLog.created_at)}</Text>
                  </View>

                  {selectedLog.model_name && (
                    <View style={styles.detailSection}>
                      <Text style={styles.detailSectionTitle}>{t('audit.modelInformation')}</Text>
                      <Text style={styles.detailRow}>{t('audit.model')}: {selectedLog.model_name}</Text>
                      <Text style={styles.detailRow}>{t('audit.version')}: {selectedLog.model_version}</Text>
                      <Text style={styles.detailRow}>{t('audit.prediction')}: {selectedLog.prediction_result}</Text>
                      <Text style={styles.detailRow}>{t('audit.confidence')}: {(selectedLog.confidence_score * 100).toFixed(2)}%</Text>
                      {selectedLog.reliability_score && (
                        <Text style={styles.detailRow}>{t('audit.reliability')}: {(selectedLog.reliability_score * 100).toFixed(2)}%</Text>
                      )}
                    </View>
                  )}

                  {selectedLog.input_metadata && (
                    <View style={styles.detailSection}>
                      <Text style={styles.detailSectionTitle}>{t('audit.inputMetadata')}</Text>
                      <Text style={styles.detailRow}>{t('audit.filename')}: {selectedLog.input_metadata.filename}</Text>
                      <Text style={styles.detailRow}>{t('audit.size')}: {(selectedLog.input_metadata.file_size / 1024).toFixed(0)} KB</Text>
                      <Text style={styles.detailRow}>
                        {t('audit.dimensions')}: {selectedLog.input_metadata.width} x {selectedLog.input_metadata.height}
                      </Text>
                      {selectedLog.input_data_hash && (
                        <Text style={styles.detailRow}>{t('audit.hash')}: {selectedLog.input_data_hash.substring(0, 16)}...</Text>
                      )}
                    </View>
                  )}

                  {selectedLog.flags && selectedLog.flags.length > 0 && (
                    <View style={styles.detailSection}>
                      <Text style={styles.detailSectionTitle}>{t('audit.flags')}</Text>
                      {selectedLog.flags.map((flag: string, index: number) => (
                        <View key={index} style={styles.flagItem}>
                          <Ionicons name="flag" size={16} color="#ef4444" />
                          <Text style={styles.flagText}>{flag}</Text>
                        </View>
                      ))}
                    </View>
                  )}

                  <View style={styles.detailSection}>
                    <Text style={styles.detailSectionTitle}>{t('audit.technicalDetails')}</Text>
                    <Text style={styles.detailRow}>{t('audit.processingTime')}: {selectedLog.processing_time_ms}ms</Text>
                    <Text style={styles.detailRow}>{t('audit.gpuUsed')}: {selectedLog.gpu_used ? t('audit.yes') : t('audit.no')}</Text>
                    <Text style={styles.detailRow}>{t('audit.qualityPassed')}: {selectedLog.quality_passed ? t('audit.yes') : t('audit.no')}</Text>
                    <Text style={styles.detailRow}>{t('audit.endpoint')}: {selectedLog.endpoint}</Text>
                    <Text style={styles.detailRow}>{t('audit.httpStatus')}: {selectedLog.response_status}</Text>
                    <Text style={styles.detailRow}>{t('audit.ipAddress')}: {selectedLog.ip_address || 'N/A'}</Text>
                  </View>

                  {selectedLog.error_occurred && (
                    <View style={styles.detailSection}>
                      <Text style={styles.detailSectionTitle}>{t('audit.errorInformation')}</Text>
                      <Text style={styles.errorText}>{selectedLog.error_message}</Text>
                    </View>
                  )}
                </>
              )}
            </ScrollView>

            <View style={styles.modalFooter}>
              <Pressable
                style={styles.modalCloseButton}
                onPress={() => setDetailModalVisible(false)}
              >
                <Text style={styles.modalCloseButtonText}>{t('audit.close')}</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
    </ScrollView>
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
  statsContainer: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
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
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#3b82f6',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
  filtersContainer: {
    padding: 16,
    paddingTop: 0,
  },
  filterRow: {
    flexDirection: 'row',
    gap: 8,
    flexWrap: 'wrap',
  },
  filterButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  filterButtonActive: {
    backgroundColor: '#3b82f6',
    borderColor: '#3b82f6',
  },
  filterButtonText: {
    fontSize: 14,
    color: '#6b7280',
  },
  filterButtonTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  logsContainer: {
    padding: 16,
    paddingTop: 0,
  },
  logsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  exportButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#3b82f6',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  exportButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 48,
  },
  emptyStateText: {
    marginTop: 12,
    fontSize: 16,
    color: '#9ca3af',
  },
  logCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  logHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  logHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  logEventType: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  flagBadge: {
    backgroundColor: '#ef4444',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  flagBadgeText: {
    fontSize: 12,
    color: '#fff',
    fontWeight: '600',
  },
  logDate: {
    fontSize: 12,
    color: '#9ca3af',
  },
  logBody: {
    gap: 6,
    marginBottom: 12,
  },
  logDetail: {
    fontSize: 14,
    color: '#6b7280',
  },
  logDetailValue: {
    fontWeight: '600',
    color: '#1f2937',
  },
  errorBox: {
    backgroundColor: '#fef2f2',
    padding: 8,
    borderRadius: 6,
    borderLeftWidth: 3,
    borderLeftColor: '#ef4444',
    marginTop: 8,
  },
  errorText: {
    fontSize: 13,
    color: '#dc2626',
  },
  logFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  logEndpoint: {
    fontSize: 12,
    color: '#9ca3af',
    fontFamily: 'monospace',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '90%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1f2937',
  },
  modalBody: {
    padding: 20,
  },
  detailSection: {
    marginBottom: 20,
  },
  detailSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  detailRow: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 6,
  },
  flagItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
  },
  flagText: {
    fontSize: 14,
    color: '#dc2626',
  },
  modalFooter: {
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  modalCloseButton: {
    backgroundColor: '#3b82f6',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  modalCloseButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
