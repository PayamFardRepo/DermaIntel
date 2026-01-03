import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  Alert,
  ActivityIndicator,
  RefreshControl,
  Platform,
  Modal
} from 'react-native';
import { router } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { API_BASE_URL } from '../config';

interface Patient {
  id: number;
  username: string;
  full_name: string;
  email: string;
}

interface Dermatologist {
  id: number;
  full_name: string;
  credentials: string;
  specializations?: string[];
  average_rating?: number;
  total_reviews?: number;
  years_experience?: number;
  video_platform?: string;
  timezone?: string;
  workload?: {
    scheduled_consultations: number;
    completed_consultations: number;
  };
}

interface Consultation {
  id: number;
  patient: Patient;
  dermatologist: Dermatologist | null;
  consultation_type: string;
  consultation_reason: string;
  scheduled_datetime: string;
  duration_minutes: number;
  status: string;
  created_at: string;
}

interface Stats {
  pending_assignment: number;
  scheduled: number;
  completed: number;
}

export default function OpsDashboardScreen() {
  const { user } = useAuth();
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [consultations, setConsultations] = useState<Consultation[]>([]);
  const [stats, setStats] = useState<Stats>({ pending_assignment: 0, scheduled: 0, completed: 0 });
  const [statusFilter, setStatusFilter] = useState<string>('pending_assignment');

  // Assignment modal state
  const [showAssignModal, setShowAssignModal] = useState(false);
  const [selectedConsultation, setSelectedConsultation] = useState<Consultation | null>(null);
  const [availableDermatologists, setAvailableDermatologists] = useState<Dermatologist[]>([]);
  const [isAssigning, setIsAssigning] = useState(false);

  useEffect(() => {
    loadConsultations();
  }, [statusFilter]);

  const loadConsultations = async () => {
    try {
      setIsLoading(true);
      const url = statusFilter
        ? `${API_BASE_URL}/ops/consultations?status=${statusFilter}`
        : `${API_BASE_URL}/ops/consultations`;

      const response = await fetch(url, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setConsultations(data.consultations || []);
        setStats(data.stats || { pending_assignment: 0, scheduled: 0, completed: 0 });
      } else {
        console.log('Failed to load consultations');
        setConsultations([]);
      }
    } catch (error) {
      console.log('Error loading consultations:', error);
      setConsultations([]);
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const loadAvailableDermatologists = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/ops/dermatologists/available`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setAvailableDermatologists(data.dermatologists || []);
      }
    } catch (error) {
      console.log('Error loading dermatologists:', error);
      setAvailableDermatologists([]);
    }
  };

  const openAssignModal = async (consultation: Consultation) => {
    setSelectedConsultation(consultation);
    setShowAssignModal(true);
    await loadAvailableDermatologists();
  };

  const assignDermatologist = async (dermatologistId: number) => {
    if (!selectedConsultation) return;

    try {
      setIsAssigning(true);
      const formData = new FormData();
      formData.append('dermatologist_id', dermatologistId.toString());

      const response = await fetch(
        `${API_BASE_URL}/ops/consultations/${selectedConsultation.id}/assign`,
        {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${user?.token}` },
          body: formData
        }
      );

      if (response.ok) {
        const data = await response.json();
        Alert.alert(
          'Success',
          `Assigned ${data.dermatologist.full_name} to consultation with ${selectedConsultation.patient.full_name}`
        );
        setShowAssignModal(false);
        setSelectedConsultation(null);
        loadConsultations(); // Refresh the list
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to assign dermatologist');
      }
    } catch (error) {
      Alert.alert('Error', 'Network error. Please try again.');
    } finally {
      setIsAssigning(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending_assignment': return '#f59e0b';
      case 'scheduled': return '#3b82f6';
      case 'confirmed': return '#10b981';
      case 'completed': return '#6b7280';
      case 'cancelled': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return 'Not scheduled';
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  };

  const formatTimeAgo = (dateString: string) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable onPress={() => router.back()} style={styles.backButton}>
          <Text style={styles.backButtonText}>{'<'}</Text>
        </Pressable>
        <Text style={styles.headerTitle}>Operations Dashboard</Text>
      </View>

      {/* Stats Cards */}
      <View style={styles.statsContainer}>
        <Pressable
          style={[styles.statCard, statusFilter === 'pending_assignment' && styles.statCardActive]}
          onPress={() => setStatusFilter('pending_assignment')}
        >
          <Text style={styles.statNumber}>{stats.pending_assignment}</Text>
          <Text style={styles.statLabel}>Pending</Text>
        </Pressable>
        <Pressable
          style={[styles.statCard, styles.statCardBlue, statusFilter === 'scheduled' && styles.statCardActive]}
          onPress={() => setStatusFilter('scheduled')}
        >
          <Text style={styles.statNumber}>{stats.scheduled}</Text>
          <Text style={styles.statLabel}>Scheduled</Text>
        </Pressable>
        <Pressable
          style={[styles.statCard, styles.statCardGreen, statusFilter === 'completed' && styles.statCardActive]}
          onPress={() => setStatusFilter('completed')}
        >
          <Text style={styles.statNumber}>{stats.completed}</Text>
          <Text style={styles.statLabel}>Completed</Text>
        </Pressable>
        <Pressable
          style={[styles.statCard, styles.statCardGray, statusFilter === '' && styles.statCardActive]}
          onPress={() => setStatusFilter('')}
        >
          <Text style={styles.statNumber}>{stats.pending_assignment + stats.scheduled + stats.completed}</Text>
          <Text style={styles.statLabel}>All</Text>
        </Pressable>
      </View>

      {/* Consultations List */}
      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={() => { setRefreshing(true); loadConsultations(); }} />
        }
      >
        <Text style={styles.sectionTitle}>
          {statusFilter === 'pending_assignment' ? 'Pending Assignment' :
           statusFilter === 'scheduled' ? 'Scheduled Consultations' :
           statusFilter === 'completed' ? 'Completed Consultations' : 'All Consultations'}
        </Text>

        {isLoading ? (
          <ActivityIndicator size="large" color="#0284c7" style={{ marginTop: 40 }} />
        ) : consultations.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyStateIcon}>
              {statusFilter === 'pending_assignment' ? '✅' : '📋'}
            </Text>
            <Text style={styles.emptyStateText}>
              {statusFilter === 'pending_assignment'
                ? 'No pending consultations'
                : 'No consultations found'}
            </Text>
          </View>
        ) : (
          consultations.map((consultation) => (
            <View key={consultation.id} style={styles.consultationCard}>
              <View style={styles.cardHeader}>
                <View style={styles.patientInfo}>
                  <Text style={styles.patientName}>{consultation.patient.full_name}</Text>
                  <Text style={styles.patientEmail}>{consultation.patient.email}</Text>
                </View>
                <View style={[styles.statusBadge, { backgroundColor: getStatusColor(consultation.status) + '20' }]}>
                  <Text style={[styles.statusText, { color: getStatusColor(consultation.status) }]}>
                    {consultation.status.replace('_', ' ')}
                  </Text>
                </View>
              </View>

              <View style={styles.cardBody}>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Type:</Text>
                  <Text style={styles.infoValue}>{consultation.consultation_type.replace('_', ' ')}</Text>
                </View>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Reason:</Text>
                  <Text style={styles.infoValue} numberOfLines={2}>{consultation.consultation_reason}</Text>
                </View>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Scheduled:</Text>
                  <Text style={styles.infoValue}>{formatDate(consultation.scheduled_datetime)}</Text>
                </View>
                {consultation.dermatologist ? (
                  <View style={styles.infoRow}>
                    <Text style={styles.infoLabel}>Assigned to:</Text>
                    <Text style={styles.infoValueHighlight}>
                      {consultation.dermatologist.full_name} {consultation.dermatologist.credentials}
                    </Text>
                  </View>
                ) : (
                  <View style={styles.infoRow}>
                    <Text style={styles.infoLabel}>Assigned to:</Text>
                    <Text style={styles.infoValueWarning}>Not assigned</Text>
                  </View>
                )}
                <Text style={styles.timeAgo}>Requested {formatTimeAgo(consultation.created_at)}</Text>
              </View>

              {consultation.status === 'pending_assignment' && (
                <Pressable
                  style={styles.assignButton}
                  onPress={() => openAssignModal(consultation)}
                >
                  <Text style={styles.assignButtonText}>Assign Dermatologist</Text>
                </Pressable>
              )}

              {consultation.status === 'scheduled' && (
                <Pressable
                  style={[styles.assignButton, styles.reassignButton]}
                  onPress={() => openAssignModal(consultation)}
                >
                  <Text style={[styles.assignButtonText, styles.reassignButtonText]}>Reassign</Text>
                </Pressable>
              )}
            </View>
          ))
        )}
      </ScrollView>

      {/* Assignment Modal */}
      <Modal
        visible={showAssignModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowAssignModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Assign Dermatologist</Text>
              <Pressable onPress={() => setShowAssignModal(false)}>
                <Text style={styles.modalClose}>X</Text>
              </Pressable>
            </View>

            {selectedConsultation && (
              <View style={styles.modalPatientInfo}>
                <Text style={styles.modalPatientName}>
                  Patient: {selectedConsultation.patient.full_name}
                </Text>
                <Text style={styles.modalPatientReason}>
                  Reason: {selectedConsultation.consultation_reason}
                </Text>
              </View>
            )}

            <Text style={styles.modalSubtitle}>Available Dermatologists</Text>
            <Text style={styles.modalHint}>Sorted by lowest workload first</Text>

            <ScrollView style={styles.dermList}>
              {availableDermatologists.length === 0 ? (
                <Text style={styles.noDermsText}>No dermatologists available</Text>
              ) : (
                availableDermatologists.map((derm) => (
                  <Pressable
                    key={derm.id}
                    style={styles.dermCard}
                    onPress={() => assignDermatologist(derm.id)}
                    disabled={isAssigning}
                  >
                    <View style={styles.dermInfo}>
                      <Text style={styles.dermName}>{derm.full_name}</Text>
                      <Text style={styles.dermCredentials}>{derm.credentials}</Text>
                      {derm.specializations && derm.specializations.length > 0 && (
                        <Text style={styles.dermSpecializations}>
                          {derm.specializations.join(', ')}
                        </Text>
                      )}
                      <View style={styles.dermStats}>
                        {derm.average_rating && (
                          <Text style={styles.dermRating}>⭐ {derm.average_rating.toFixed(1)}</Text>
                        )}
                        {derm.years_experience && (
                          <Text style={styles.dermExperience}>{derm.years_experience} yrs exp</Text>
                        )}
                      </View>
                    </View>
                    <View style={styles.dermWorkload}>
                      <Text style={styles.workloadNumber}>
                        {derm.workload?.scheduled_consultations || 0}
                      </Text>
                      <Text style={styles.workloadLabel}>scheduled</Text>
                    </View>
                  </Pressable>
                ))
              )}
            </ScrollView>

            {isAssigning && (
              <View style={styles.assigningOverlay}>
                <ActivityIndicator size="large" color="#0284c7" />
                <Text style={styles.assigningText}>Assigning...</Text>
              </View>
            )}
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f1f5f9',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: Platform.OS === 'ios' ? 50 : 16,
    paddingBottom: 16,
    paddingHorizontal: 16,
    backgroundColor: '#0f172a',
  },
  backButton: {
    padding: 8,
    marginRight: 8,
    minWidth: 44,
    minHeight: 44,
    justifyContent: 'center',
    alignItems: 'center',
  },
  backButtonText: {
    fontSize: 24,
    color: '#fff',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  statsContainer: {
    flexDirection: 'row',
    padding: 16,
    gap: 8,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#fef3c7',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  statCardBlue: {
    backgroundColor: '#dbeafe',
  },
  statCardGreen: {
    backgroundColor: '#d1fae5',
  },
  statCardGray: {
    backgroundColor: '#e5e7eb',
  },
  statCardActive: {
    borderColor: '#0284c7',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  statLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 2,
  },
  content: {
    flex: 1,
    paddingHorizontal: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 12,
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
  },
  emptyStateIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  emptyStateText: {
    fontSize: 16,
    color: '#6b7280',
  },
  consultationCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  patientInfo: {
    flex: 1,
  },
  patientName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  patientEmail: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    fontSize: 11,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  cardBody: {
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
    paddingTop: 12,
  },
  infoRow: {
    flexDirection: 'row',
    marginBottom: 6,
  },
  infoLabel: {
    fontSize: 13,
    color: '#6b7280',
    width: 80,
  },
  infoValue: {
    fontSize: 13,
    color: '#1f2937',
    flex: 1,
  },
  infoValueHighlight: {
    fontSize: 13,
    color: '#059669',
    fontWeight: '500',
    flex: 1,
  },
  infoValueWarning: {
    fontSize: 13,
    color: '#f59e0b',
    fontWeight: '500',
    flex: 1,
  },
  timeAgo: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 8,
  },
  assignButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 12,
  },
  assignButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  reassignButton: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#0284c7',
  },
  reassignButtonText: {
    color: '#0284c7',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '80%',
    padding: 20,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  modalClose: {
    fontSize: 20,
    color: '#6b7280',
    padding: 8,
  },
  modalPatientInfo: {
    backgroundColor: '#f3f4f6',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  modalPatientName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  modalPatientReason: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 4,
  },
  modalSubtitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  modalHint: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 12,
  },
  dermList: {
    maxHeight: 300,
  },
  noDermsText: {
    textAlign: 'center',
    color: '#6b7280',
    paddingVertical: 20,
  },
  dermCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  dermInfo: {
    flex: 1,
  },
  dermName: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937',
  },
  dermCredentials: {
    fontSize: 13,
    color: '#6b7280',
  },
  dermSpecializations: {
    fontSize: 12,
    color: '#0284c7',
    marginTop: 2,
  },
  dermStats: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 4,
  },
  dermRating: {
    fontSize: 12,
    color: '#f59e0b',
  },
  dermExperience: {
    fontSize: 12,
    color: '#6b7280',
  },
  dermWorkload: {
    alignItems: 'center',
    backgroundColor: '#dbeafe',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    marginLeft: 12,
  },
  workloadNumber: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1e40af',
  },
  workloadLabel: {
    fontSize: 10,
    color: '#3b82f6',
  },
  assigningOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(255,255,255,0.9)',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 20,
  },
  assigningText: {
    marginTop: 12,
    fontSize: 16,
    color: '#0284c7',
  },
});
