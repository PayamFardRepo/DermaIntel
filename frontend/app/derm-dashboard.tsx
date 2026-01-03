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
  full_name: string;
  email: string;
  age?: number;
  skin_type?: string;
}

interface Consultation {
  id: number;
  patient: Patient;
  consultation_type: string;
  consultation_reason: string;
  patient_notes?: string;
  patient_questions?: string;
  scheduled_datetime: string | null;
  duration_minutes: number;
  status: string;
  video_platform?: string;
  meeting_link?: string;
  created_at: string;
}

interface Stats {
  total: number;
  scheduled: number;
  confirmed: number;
  in_progress: number;
  completed: number;
  cancelled: number;
}

interface DermatologistInfo {
  id: number;
  full_name: string;
  credentials: string;
}

export default function DermDashboardScreen() {
  const { user } = useAuth();
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [consultations, setConsultations] = useState<Consultation[]>([]);
  const [stats, setStats] = useState<Stats>({ total: 0, scheduled: 0, confirmed: 0, in_progress: 0, completed: 0, cancelled: 0 });
  const [statusFilter, setStatusFilter] = useState<string>('scheduled');
  const [dermInfo, setDermInfo] = useState<DermatologistInfo | null>(null);

  // Detail modal state
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [selectedConsultation, setSelectedConsultation] = useState<Consultation | null>(null);
  const [consultationDetails, setConsultationDetails] = useState<any>(null);
  const [isStarting, setIsStarting] = useState(false);

  useEffect(() => {
    loadConsultations();
  }, [statusFilter]);

  const loadConsultations = async () => {
    try {
      setIsLoading(true);
      const url = statusFilter
        ? `${API_BASE_URL}/my/consultations?status=${statusFilter}`
        : `${API_BASE_URL}/my/consultations`;

      const response = await fetch(url, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setConsultations(data.consultations || []);
        setStats(data.stats || { total: 0, scheduled: 0, confirmed: 0, in_progress: 0, completed: 0, cancelled: 0 });
        setDermInfo(data.dermatologist || null);
      } else {
        const errorText = await response.text();
        console.log('Failed to load consultations:', errorText);
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

  const onRefresh = () => {
    setRefreshing(true);
    loadConsultations();
  };

  const viewConsultationDetails = async (consultation: Consultation) => {
    setSelectedConsultation(consultation);
    setShowDetailModal(true);

    try {
      const response = await fetch(`${API_BASE_URL}/my/consultations/${consultation.id}`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setConsultationDetails(data);
      }
    } catch (error) {
      console.log('Error loading consultation details:', error);
    }
  };

  const startConsultation = async (consultationId: number) => {
    setIsStarting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/my/consultations/${consultationId}/start`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });

      if (response.ok) {
        Alert.alert('Success', 'Consultation started. You can now begin the video call.');
        setShowDetailModal(false);
        loadConsultations();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to start consultation');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to start consultation');
    } finally {
      setIsStarting(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'scheduled': return '#2196F3';
      case 'confirmed': return '#4CAF50';
      case 'in_progress': return '#FF9800';
      case 'completed': return '#9E9E9E';
      case 'cancelled': return '#F44336';
      default: return '#757575';
    }
  };

  const formatDateTime = (dateString: string | null) => {
    if (!dateString) return 'Not scheduled';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const StatusFilterButton = ({ status, label, count }: { status: string; label: string; count: number }) => (
    <Pressable
      style={[
        styles.filterButton,
        statusFilter === status && styles.filterButtonActive
      ]}
      onPress={() => setStatusFilter(status)}
    >
      <Text style={[
        styles.filterButtonText,
        statusFilter === status && styles.filterButtonTextActive
      ]}>
        {label} ({count})
      </Text>
    </Pressable>
  );

  if (isLoading && !refreshing) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#4FACFE" />
        <Text style={styles.loadingText}>Loading your consultations...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>My Consultations</Text>
          {dermInfo && (
            <Text style={styles.headerSubtitle}>{dermInfo.full_name}, {dermInfo.credentials}</Text>
          )}
        </View>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Text style={styles.backButtonText}>Back</Text>
        </Pressable>
      </View>

      {/* Stats Cards */}
      <View style={styles.statsContainer}>
        <View style={[styles.statCard, { backgroundColor: '#E3F2FD' }]}>
          <Text style={styles.statNumber}>{stats.scheduled + stats.confirmed}</Text>
          <Text style={styles.statLabel}>Upcoming</Text>
        </View>
        <View style={[styles.statCard, { backgroundColor: '#FFF3E0' }]}>
          <Text style={styles.statNumber}>{stats.in_progress}</Text>
          <Text style={styles.statLabel}>In Progress</Text>
        </View>
        <View style={[styles.statCard, { backgroundColor: '#E8F5E9' }]}>
          <Text style={styles.statNumber}>{stats.completed}</Text>
          <Text style={styles.statLabel}>Completed</Text>
        </View>
      </View>

      {/* Filter Buttons */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterContainer}>
        <StatusFilterButton status="scheduled" label="Scheduled" count={stats.scheduled} />
        <StatusFilterButton status="confirmed" label="Confirmed" count={stats.confirmed} />
        <StatusFilterButton status="in_progress" label="In Progress" count={stats.in_progress} />
        <StatusFilterButton status="completed" label="Completed" count={stats.completed} />
        <StatusFilterButton status="" label="All" count={stats.total} />
      </ScrollView>

      {/* Consultations List */}
      <ScrollView
        style={styles.listContainer}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {consultations.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyIcon}>📋</Text>
            <Text style={styles.emptyText}>No consultations found</Text>
            <Text style={styles.emptySubtext}>
              {statusFilter ? `No ${statusFilter} consultations` : 'You have no assigned consultations yet'}
            </Text>
          </View>
        ) : (
          consultations.map((consultation) => (
            <Pressable
              key={consultation.id}
              style={styles.consultationCard}
              onPress={() => viewConsultationDetails(consultation)}
            >
              <View style={styles.cardHeader}>
                <View style={[styles.statusBadge, { backgroundColor: getStatusColor(consultation.status) }]}>
                  <Text style={styles.statusText}>{consultation.status.replace('_', ' ')}</Text>
                </View>
                <Text style={styles.consultationId}>#{consultation.id}</Text>
              </View>

              <View style={styles.patientInfo}>
                <Text style={styles.patientName}>{consultation.patient.full_name}</Text>
                {consultation.patient.age && (
                  <Text style={styles.patientDetail}>Age: {consultation.patient.age}</Text>
                )}
                {consultation.patient.skin_type && (
                  <Text style={styles.patientDetail}>Skin Type: {consultation.patient.skin_type}</Text>
                )}
              </View>

              <Text style={styles.consultationReason} numberOfLines={2}>
                {consultation.consultation_reason || 'No reason provided'}
              </Text>

              <View style={styles.cardFooter}>
                <Text style={styles.dateTime}>
                  {formatDateTime(consultation.scheduled_datetime)}
                </Text>
                <Text style={styles.viewDetails}>View Details →</Text>
              </View>
            </Pressable>
          ))
        )}
      </ScrollView>

      {/* Consultation Detail Modal */}
      <Modal
        visible={showDetailModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowDetailModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Consultation Details</Text>
              <Pressable onPress={() => setShowDetailModal(false)}>
                <Text style={styles.closeButton}>✕</Text>
              </Pressable>
            </View>

            <ScrollView style={styles.modalScroll}>
              {consultationDetails ? (
                <>
                  {/* Patient Info */}
                  <View style={styles.detailSection}>
                    <Text style={styles.sectionTitle}>Patient Information</Text>
                    <Text style={styles.detailText}>Name: {consultationDetails.patient?.full_name}</Text>
                    <Text style={styles.detailText}>Email: {consultationDetails.patient?.email}</Text>
                    {consultationDetails.patient?.age && (
                      <Text style={styles.detailText}>Age: {consultationDetails.patient.age}</Text>
                    )}
                    {consultationDetails.patient?.skin_type && (
                      <Text style={styles.detailText}>Skin Type: {consultationDetails.patient.skin_type}</Text>
                    )}
                    {consultationDetails.patient?.medical_conditions && (
                      <Text style={styles.detailText}>Medical Conditions: {consultationDetails.patient.medical_conditions}</Text>
                    )}
                    {consultationDetails.patient?.allergies && (
                      <Text style={styles.detailText}>Allergies: {consultationDetails.patient.allergies}</Text>
                    )}
                  </View>

                  {/* Consultation Info */}
                  <View style={styles.detailSection}>
                    <Text style={styles.sectionTitle}>Consultation Details</Text>
                    <Text style={styles.detailText}>Type: {consultationDetails.consultation?.consultation_type}</Text>
                    <Text style={styles.detailText}>Reason: {consultationDetails.consultation?.consultation_reason}</Text>
                    <Text style={styles.detailText}>Status: {consultationDetails.consultation?.status}</Text>
                    <Text style={styles.detailText}>
                      Scheduled: {formatDateTime(consultationDetails.consultation?.scheduled_datetime)}
                    </Text>
                    {consultationDetails.consultation?.patient_notes && (
                      <>
                        <Text style={styles.notesLabel}>Patient Notes:</Text>
                        <Text style={styles.notesText}>{consultationDetails.consultation.patient_notes}</Text>
                      </>
                    )}
                    {consultationDetails.consultation?.patient_questions && (
                      <>
                        <Text style={styles.notesLabel}>Patient Questions:</Text>
                        <Text style={styles.notesText}>{consultationDetails.consultation.patient_questions}</Text>
                      </>
                    )}
                  </View>

                  {/* Recent Analyses */}
                  {consultationDetails.recent_analyses && consultationDetails.recent_analyses.length > 0 && (
                    <View style={styles.detailSection}>
                      <Text style={styles.sectionTitle}>Recent Skin Analyses</Text>
                      {consultationDetails.recent_analyses.map((analysis: any, index: number) => (
                        <View key={index} style={styles.analysisItem}>
                          <Text style={styles.analysisText}>
                            {analysis.predicted_class} ({(analysis.confidence * 100).toFixed(1)}%)
                          </Text>
                          <Text style={styles.analysisDate}>
                            {new Date(analysis.created_at).toLocaleDateString()}
                          </Text>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* Action Buttons */}
                  <View style={styles.actionButtons}>
                    {selectedConsultation?.status === 'scheduled' || selectedConsultation?.status === 'confirmed' ? (
                      <Pressable
                        style={[styles.actionButton, styles.startButton]}
                        onPress={() => startConsultation(selectedConsultation.id)}
                        disabled={isStarting}
                      >
                        {isStarting ? (
                          <ActivityIndicator color="#fff" />
                        ) : (
                          <Text style={styles.actionButtonText}>Start Consultation</Text>
                        )}
                      </Pressable>
                    ) : null}

                    {selectedConsultation?.status === 'in_progress' && (
                      <Pressable
                        style={[styles.actionButton, styles.completeButton]}
                        onPress={() => {
                          setShowDetailModal(false);
                          router.push(`/complete-consultation?id=${selectedConsultation.id}` as any);
                        }}
                      >
                        <Text style={styles.actionButtonText}>Complete & Add Notes</Text>
                      </Pressable>
                    )}

                    {consultationDetails.consultation?.meeting_link && (
                      <Pressable
                        style={[styles.actionButton, styles.joinButton]}
                        onPress={() => {
                          Alert.alert('Join Video Call', `Platform: ${consultationDetails.consultation.video_platform}\nLink: ${consultationDetails.consultation.meeting_link}`);
                        }}
                      >
                        <Text style={styles.actionButtonText}>Join Video Call</Text>
                      </Pressable>
                    )}
                  </View>
                </>
              ) : (
                <ActivityIndicator size="large" color="#4FACFE" />
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5F7FA',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1a1a2e',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  backButton: {
    padding: 10,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
  },
  backButtonText: {
    color: '#333',
    fontWeight: '600',
  },
  statsContainer: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
  },
  statCard: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1a1a2e',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  filterContainer: {
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#E0E0E0',
    marginRight: 8,
  },
  filterButtonActive: {
    backgroundColor: '#4FACFE',
  },
  filterButtonText: {
    color: '#666',
    fontWeight: '500',
  },
  filterButtonTextActive: {
    color: '#fff',
  },
  listContainer: {
    flex: 1,
    padding: 16,
  },
  emptyContainer: {
    alignItems: 'center',
    padding: 40,
  },
  emptyIcon: {
    fontSize: 48,
    marginBottom: 16,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  emptySubtext: {
    fontSize: 14,
    color: '#666',
    marginTop: 8,
  },
  consultationCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  consultationId: {
    fontSize: 12,
    color: '#999',
  },
  patientInfo: {
    marginBottom: 8,
  },
  patientName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1a1a2e',
  },
  patientDetail: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  consultationReason: {
    fontSize: 14,
    color: '#666',
    marginBottom: 12,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
    paddingTop: 12,
  },
  dateTime: {
    fontSize: 13,
    color: '#666',
  },
  viewDetails: {
    fontSize: 14,
    color: '#4FACFE',
    fontWeight: '600',
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
    maxHeight: '90%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1a1a2e',
  },
  closeButton: {
    fontSize: 24,
    color: '#666',
    padding: 4,
  },
  modalScroll: {
    padding: 20,
  },
  detailSection: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1a1a2e',
    marginBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
    paddingBottom: 8,
  },
  detailText: {
    fontSize: 14,
    color: '#333',
    marginBottom: 6,
  },
  notesLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginTop: 8,
  },
  notesText: {
    fontSize: 14,
    color: '#666',
    backgroundColor: '#f9f9f9',
    padding: 12,
    borderRadius: 8,
    marginTop: 4,
  },
  analysisItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  analysisText: {
    fontSize: 14,
    color: '#333',
  },
  analysisDate: {
    fontSize: 12,
    color: '#999',
  },
  actionButtons: {
    marginTop: 16,
    marginBottom: 32,
    gap: 12,
  },
  actionButton: {
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  startButton: {
    backgroundColor: '#4CAF50',
  },
  completeButton: {
    backgroundColor: '#2196F3',
  },
  joinButton: {
    backgroundColor: '#9C27B0',
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
