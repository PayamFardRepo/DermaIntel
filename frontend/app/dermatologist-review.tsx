import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  Pressable,
  Alert,
  ActivityIndicator,
  RefreshControl,
  Platform,
  StatusBar,
  Modal,
  Switch
} from 'react-native';
import { router } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import { API_BASE_URL } from '../config';

interface QueueItem {
  id: number;
  patient_name: string;
  original_diagnosis: string;
  reason: string;
  urgency: string;
  status: string;
  created_at: string;
  has_images: boolean;
  sla_status?: {
    deadline: string;
    hours_remaining: number;
    is_urgent: boolean;
    is_breached: boolean;
  };
}

interface SecondOpinionDetail {
  id: number;
  user_id: number;
  original_diagnosis: string;
  original_provider_name?: string;
  original_diagnosis_date?: string;
  original_treatment_plan?: string;
  reason_for_second_opinion: string;
  specific_questions?: string[];
  concerns?: string;
  analysis_id?: number;
  lesion_group_id?: number;
  urgency: string;
  status: string;
  created_at: string;
}

interface DermatologistStats {
  role: string;
  total_reviews: number;
  completed_reviews: number;
  pending_reviews: number;
  urgent_pending: number;
  average_rating?: number;
  dermatologist_name: string;
}

export default function DermatologistReviewScreen() {
  const { user } = useAuth();
  const { t } = useTranslation();

  // State
  const [isLoading, setIsLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [stats, setStats] = useState<DermatologistStats | null>(null);
  const [statusFilter, setStatusFilter] = useState<'assigned' | 'in_review' | 'completed' | 'all'>('assigned');

  // Detail/Review Modal
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [selectedCase, setSelectedCase] = useState<SecondOpinionDetail | null>(null);
  const [loadingCase, setLoadingCase] = useState(false);

  // Review Form State
  const [diagnosis, setDiagnosis] = useState('');
  const [agreesWithOriginal, setAgreesWithOriginal] = useState(true);
  const [confidenceLevel, setConfidenceLevel] = useState<'high' | 'medium' | 'low'>('high');
  const [notes, setNotes] = useState('');
  const [treatmentPlan, setTreatmentPlan] = useState('');
  const [differencesFromOriginal, setDifferencesFromOriginal] = useState('');
  const [recommendedAction, setRecommendedAction] = useState('');
  const [recommendedNextSteps, setRecommendedNextSteps] = useState('');
  const [additionalTestsNeeded, setAdditionalTestsNeeded] = useState('');
  const [biopsyRecommended, setBiopsyRecommended] = useState(false);
  const [submittingReview, setSubmittingReview] = useState(false);

  useEffect(() => {
    loadStats();
    loadQueue();
  }, [statusFilter]);

  const loadStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/second-opinions/stats/dashboard`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (response.ok) {
        const data = await response.json();
        if (data.role === 'dermatologist') {
          setStats(data);
        } else {
          Alert.alert('Access Denied', 'This page is only for registered dermatologists');
          router.back();
        }
      }
    } catch (error) {
      console.log('Error loading stats:', error);
    }
  };

  const loadQueue = async () => {
    try {
      setIsLoading(true);
      const statusParam = statusFilter === 'all' ? '' : `status=${statusFilter}`;
      const response = await fetch(`${API_BASE_URL}/second-opinions/dermatologist/queue?${statusParam}&limit=50`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setQueue(data.queue || []);
      }
    } catch (error) {
      console.log('Error loading queue:', error);
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const loadCaseDetails = async (caseId: number) => {
    try {
      setLoadingCase(true);
      const response = await fetch(`${API_BASE_URL}/second-opinions/${caseId}`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setSelectedCase(data);
        setShowReviewModal(true);

        // Pre-fill diagnosis if there was an original
        setDiagnosis('');
        setAgreesWithOriginal(true);
        setConfidenceLevel('high');
        setNotes('');
        setTreatmentPlan('');
        setDifferencesFromOriginal('');
        setRecommendedAction('');
        setRecommendedNextSteps('');
        setAdditionalTestsNeeded('');
        setBiopsyRecommended(false);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to load case details');
    } finally {
      setLoadingCase(false);
    }
  };

  const submitReview = async () => {
    if (!selectedCase) return;

    if (!diagnosis.trim()) {
      Alert.alert('Required', 'Please enter your diagnosis');
      return;
    }

    try {
      setSubmittingReview(true);
      const formData = new FormData();
      formData.append('diagnosis', diagnosis);
      formData.append('agrees_with_original', agreesWithOriginal.toString());
      formData.append('confidence_level', confidenceLevel);
      formData.append('biopsy_recommended', biopsyRecommended.toString());

      if (notes) formData.append('notes', notes);
      if (treatmentPlan) formData.append('treatment_plan', treatmentPlan);
      if (differencesFromOriginal) formData.append('differences_from_original', differencesFromOriginal);
      if (recommendedAction) formData.append('recommended_action', recommendedAction);
      if (recommendedNextSteps) formData.append('recommended_next_steps', recommendedNextSteps);
      if (additionalTestsNeeded) formData.append('additional_tests_needed', additionalTestsNeeded);

      const response = await fetch(`${API_BASE_URL}/second-opinions/${selectedCase.id}/review`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user?.token}` },
        body: formData
      });

      const data = await response.json();

      if (response.ok && data.success) {
        Alert.alert('Success', 'Review submitted successfully. The patient will be notified.');
        setShowReviewModal(false);
        setSelectedCase(null);
        loadQueue();
        loadStats();
      } else {
        Alert.alert('Error', data.detail || 'Failed to submit review');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to submit review. Please try again.');
    } finally {
      setSubmittingReview(false);
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'emergency': return '#D32F2F';
      case 'urgent': return '#EF6C00';
      default: return '#388E3C';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return '#4CAF50';
      case 'in_review': return '#2196F3';
      case 'assigned': return '#FF9800';
      default: return '#9E9E9E';
    }
  };

  const renderStats = () => {
    if (!stats) return null;

    return (
      <View style={styles.statsContainer}>
        <View style={styles.welcomeRow}>
          <Text style={styles.welcomeText}>Welcome, {stats.dermatologist_name}</Text>
          {stats.average_rating && (
            <View style={styles.ratingBadge}>
              <Text style={styles.ratingText}>★ {stats.average_rating.toFixed(1)}</Text>
            </View>
          )}
        </View>
        <View style={styles.statsRow}>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>{stats.pending_reviews}</Text>
            <Text style={styles.statLabel}>Pending</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={[styles.statValue, { color: '#FF5722' }]}>{stats.urgent_pending}</Text>
            <Text style={styles.statLabel}>Urgent</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={[styles.statValue, { color: '#4CAF50' }]}>{stats.completed_reviews}</Text>
            <Text style={styles.statLabel}>Completed</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>{stats.total_reviews}</Text>
            <Text style={styles.statLabel}>Total</Text>
          </View>
        </View>
      </View>
    );
  };

  const renderFilterTabs = () => (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterContainer}>
      {[
        { value: 'assigned', label: 'Assigned' },
        { value: 'in_review', label: 'In Review' },
        { value: 'completed', label: 'Completed' },
        { value: 'all', label: 'All Cases' }
      ].map(filter => (
        <Pressable
          key={filter.value}
          style={[
            styles.filterButton,
            statusFilter === filter.value && styles.filterButtonActive
          ]}
          onPress={() => setStatusFilter(filter.value as any)}
        >
          <Text style={[
            styles.filterButtonText,
            statusFilter === filter.value && styles.filterButtonTextActive
          ]}>
            {filter.label}
          </Text>
        </Pressable>
      ))}
    </ScrollView>
  );

  const renderQueue = () => (
    <ScrollView
      style={styles.queueContainer}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={() => {
            setRefreshing(true);
            loadQueue();
          }}
        />
      }
    >
      {queue.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateText}>No cases in queue</Text>
        </View>
      ) : (
        queue.map(item => (
          <Pressable
            key={item.id}
            style={[
              styles.caseCard,
              item.sla_status?.is_breached && styles.caseCardBreached,
              item.sla_status?.is_urgent && !item.sla_status?.is_breached && styles.caseCardUrgent
            ]}
            onPress={() => loadCaseDetails(item.id)}
          >
            {item.sla_status?.is_breached && (
              <View style={styles.breachedBanner}>
                <Text style={styles.breachedBannerText}>SLA BREACHED</Text>
              </View>
            )}

            <View style={styles.caseHeader}>
              <View style={styles.caseInfo}>
                <Text style={styles.caseDiagnosis}>{item.original_diagnosis}</Text>
                <Text style={styles.casePatient}>Patient: {item.patient_name}</Text>
              </View>
              <View style={[styles.urgencyBadge, { backgroundColor: getUrgencyColor(item.urgency) }]}>
                <Text style={styles.urgencyBadgeText}>{item.urgency.toUpperCase()}</Text>
              </View>
            </View>

            <Text style={styles.caseReason} numberOfLines={2}>{item.reason}</Text>

            <View style={styles.caseFooter}>
              <View style={styles.caseMetadata}>
                <Text style={styles.caseDate}>
                  {new Date(item.created_at).toLocaleDateString()}
                </Text>
                {item.has_images && (
                  <View style={styles.hasImagesBadge}>
                    <Text style={styles.hasImagesBadgeText}>Has Images</Text>
                  </View>
                )}
              </View>

              {item.sla_status && (
                <View style={styles.slaInfo}>
                  <Text style={[
                    styles.slaText,
                    item.sla_status.hours_remaining < 4 && { color: '#D32F2F' }
                  ]}>
                    {item.sla_status.hours_remaining > 0
                      ? `${Math.round(item.sla_status.hours_remaining)}h remaining`
                      : 'Overdue'}
                  </Text>
                </View>
              )}
            </View>

            <View style={[styles.statusIndicator, { backgroundColor: getStatusColor(item.status) }]}>
              <Text style={styles.statusIndicatorText}>{item.status.replace('_', ' ')}</Text>
            </View>
          </Pressable>
        ))
      )}
    </ScrollView>
  );

  const renderReviewModal = () => (
    <Modal
      visible={showReviewModal}
      animationType="slide"
      onRequestClose={() => setShowReviewModal(false)}
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <Pressable onPress={() => setShowReviewModal(false)}>
            <Text style={styles.closeButton}>← Back</Text>
          </Pressable>
          <Text style={styles.modalTitle}>Review Case #{selectedCase?.id}</Text>
          <View style={{ width: 60 }} />
        </View>

        {loadingCase ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#4A90A4" />
          </View>
        ) : selectedCase ? (
          <ScrollView style={styles.modalContent}>
            {/* Case Information */}
            <View style={styles.sectionCard}>
              <Text style={styles.sectionTitle}>Patient's Original Diagnosis</Text>
              <Text style={styles.diagnosisText}>{selectedCase.original_diagnosis}</Text>
              {selectedCase.original_provider_name && (
                <Text style={styles.providerText}>
                  By: {selectedCase.original_provider_name}
                </Text>
              )}
            </View>

            {selectedCase.original_treatment_plan && (
              <View style={styles.sectionCard}>
                <Text style={styles.sectionTitle}>Current Treatment Plan</Text>
                <Text style={styles.contentText}>{selectedCase.original_treatment_plan}</Text>
              </View>
            )}

            <View style={styles.sectionCard}>
              <Text style={styles.sectionTitle}>Reason for Second Opinion</Text>
              <Text style={styles.contentText}>{selectedCase.reason_for_second_opinion}</Text>
            </View>

            {selectedCase.specific_questions && selectedCase.specific_questions.length > 0 && (
              <View style={styles.sectionCard}>
                <Text style={styles.sectionTitle}>Patient's Questions</Text>
                {selectedCase.specific_questions.map((q, i) => (
                  <Text key={i} style={styles.questionItem}>• {q}</Text>
                ))}
              </View>
            )}

            {selectedCase.concerns && (
              <View style={styles.sectionCard}>
                <Text style={styles.sectionTitle}>Additional Concerns</Text>
                <Text style={styles.contentText}>{selectedCase.concerns}</Text>
              </View>
            )}

            {selectedCase.analysis_id && (
              <Pressable style={styles.viewImagesButton}>
                <Text style={styles.viewImagesButtonText}>View Associated Images</Text>
              </Pressable>
            )}

            {/* Review Form */}
            <View style={styles.divider} />
            <Text style={styles.formSectionTitle}>Your Professional Opinion</Text>

            <View style={styles.sectionCard}>
              <Text style={styles.label}>Your Diagnosis *</Text>
              <TextInput
                style={[styles.input, styles.multilineInput]}
                value={diagnosis}
                onChangeText={setDiagnosis}
                placeholder="Enter your diagnosis"
                placeholderTextColor="#999"
                multiline
              />
            </View>

            <View style={styles.sectionCard}>
              <Text style={styles.label}>Agreement with Original</Text>
              <View style={styles.agreementToggle}>
                <Pressable
                  style={[
                    styles.agreementButton,
                    agreesWithOriginal && styles.agreementButtonActive
                  ]}
                  onPress={() => setAgreesWithOriginal(true)}
                >
                  <Text style={[
                    styles.agreementButtonText,
                    agreesWithOriginal && styles.agreementButtonTextActive
                  ]}>
                    Agrees
                  </Text>
                </Pressable>
                <Pressable
                  style={[
                    styles.agreementButton,
                    !agreesWithOriginal && styles.agreementButtonDifferent
                  ]}
                  onPress={() => setAgreesWithOriginal(false)}
                >
                  <Text style={[
                    styles.agreementButtonText,
                    !agreesWithOriginal && styles.agreementButtonTextActive
                  ]}>
                    Differs
                  </Text>
                </Pressable>
              </View>
            </View>

            <View style={styles.sectionCard}>
              <Text style={styles.label}>Confidence Level</Text>
              <View style={styles.confidenceContainer}>
                {(['high', 'medium', 'low'] as const).map(level => (
                  <Pressable
                    key={level}
                    style={[
                      styles.confidenceButton,
                      confidenceLevel === level && styles.confidenceButtonActive
                    ]}
                    onPress={() => setConfidenceLevel(level)}
                  >
                    <Text style={[
                      styles.confidenceButtonText,
                      confidenceLevel === level && styles.confidenceButtonTextActive
                    ]}>
                      {level.charAt(0).toUpperCase() + level.slice(1)}
                    </Text>
                  </Pressable>
                ))}
              </View>
            </View>

            {!agreesWithOriginal && (
              <View style={styles.sectionCard}>
                <Text style={styles.label}>Key Differences</Text>
                <TextInput
                  style={[styles.input, styles.multilineInput]}
                  value={differencesFromOriginal}
                  onChangeText={setDifferencesFromOriginal}
                  placeholder="Explain how your diagnosis differs"
                  placeholderTextColor="#999"
                  multiline
                />
              </View>
            )}

            <View style={styles.sectionCard}>
              <Text style={styles.label}>Clinical Notes</Text>
              <TextInput
                style={[styles.input, styles.multilineInput]}
                value={notes}
                onChangeText={setNotes}
                placeholder="Detailed clinical observations and reasoning"
                placeholderTextColor="#999"
                multiline
              />
            </View>

            <View style={styles.sectionCard}>
              <Text style={styles.label}>Recommended Treatment Plan</Text>
              <TextInput
                style={[styles.input, styles.multilineInput]}
                value={treatmentPlan}
                onChangeText={setTreatmentPlan}
                placeholder="Your recommended treatment approach"
                placeholderTextColor="#999"
                multiline
              />
            </View>

            <View style={styles.sectionCard}>
              <Text style={styles.label}>Recommended Action</Text>
              <View style={styles.actionsContainer}>
                {[
                  { value: 'monitor', label: 'Monitor' },
                  { value: 'treat', label: 'Treat' },
                  { value: 'refer', label: 'Refer' },
                  { value: 'biopsy', label: 'Biopsy' },
                  { value: 'urgent_care', label: 'Urgent Care' }
                ].map(action => (
                  <Pressable
                    key={action.value}
                    style={[
                      styles.actionButton,
                      recommendedAction === action.value && styles.actionButtonActive
                    ]}
                    onPress={() => setRecommendedAction(action.value)}
                  >
                    <Text style={[
                      styles.actionButtonText,
                      recommendedAction === action.value && styles.actionButtonTextActive
                    ]}>
                      {action.label}
                    </Text>
                  </Pressable>
                ))}
              </View>
            </View>

            <View style={styles.sectionCard}>
              <Text style={styles.label}>Recommended Next Steps</Text>
              <TextInput
                style={[styles.input, styles.multilineInput]}
                value={recommendedNextSteps}
                onChangeText={setRecommendedNextSteps}
                placeholder="What should the patient do next?"
                placeholderTextColor="#999"
                multiline
              />
            </View>

            <View style={styles.sectionCard}>
              <Text style={styles.label}>Additional Tests Needed</Text>
              <TextInput
                style={styles.input}
                value={additionalTestsNeeded}
                onChangeText={setAdditionalTestsNeeded}
                placeholder="List any recommended tests"
                placeholderTextColor="#999"
              />
            </View>

            <View style={styles.sectionCard}>
              <View style={styles.biopsyToggle}>
                <View>
                  <Text style={styles.label}>Biopsy Recommended</Text>
                  <Text style={styles.sublabel}>This will alert the patient</Text>
                </View>
                <Switch
                  value={biopsyRecommended}
                  onValueChange={setBiopsyRecommended}
                  trackColor={{ false: '#E0E0E0', true: '#81C784' }}
                  thumbColor={biopsyRecommended ? '#4CAF50' : '#f4f3f4'}
                />
              </View>
            </View>

            <Pressable
              style={[styles.submitButton, submittingReview && styles.submitButtonDisabled]}
              onPress={submitReview}
              disabled={submittingReview}
            >
              {submittingReview ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.submitButtonText}>Submit Review</Text>
              )}
            </Pressable>

            <View style={{ height: 40 }} />
          </ScrollView>
        ) : null}
      </View>
    </Modal>
  );

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" />

      <View style={styles.header}>
        <Pressable onPress={() => router.back()} style={styles.backButton}>
          <Text style={styles.backButtonText}>←</Text>
        </Pressable>
        <Text style={styles.headerTitle}>Review Queue</Text>
        <Pressable onPress={() => { loadStats(); loadQueue(); }}>
          <Text style={styles.refreshButton}>↻</Text>
        </Pressable>
      </View>

      {renderStats()}
      {renderFilterTabs()}

      {isLoading && !refreshing ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4A90A4" />
        </View>
      ) : (
        renderQueue()
      )}

      {renderReviewModal()}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    fontSize: 24,
    color: '#4A90A4',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  refreshButton: {
    fontSize: 24,
    color: '#4A90A4',
    padding: 8,
  },
  statsContainer: {
    backgroundColor: '#fff',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  welcomeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  welcomeText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  ratingBadge: {
    backgroundColor: '#FFF8E1',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  ratingText: {
    color: '#F57C00',
    fontWeight: '600',
  },
  statsRow: {
    flexDirection: 'row',
  },
  statCard: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#333',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  filterContainer: {
    backgroundColor: '#fff',
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#F5F5F5',
    marginHorizontal: 4,
  },
  filterButtonActive: {
    backgroundColor: '#4A90A4',
  },
  filterButtonText: {
    fontSize: 14,
    color: '#666',
  },
  filterButtonTextActive: {
    color: '#fff',
    fontWeight: '500',
  },
  queueContainer: {
    flex: 1,
    padding: 16,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyStateText: {
    fontSize: 16,
    color: '#666',
  },
  caseCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
    borderLeftWidth: 4,
    borderLeftColor: '#4A90A4',
  },
  caseCardBreached: {
    borderLeftColor: '#D32F2F',
    backgroundColor: '#FFF5F5',
  },
  caseCardUrgent: {
    borderLeftColor: '#FF9800',
  },
  breachedBanner: {
    backgroundColor: '#D32F2F',
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 4,
    alignSelf: 'flex-start',
    marginBottom: 8,
  },
  breachedBannerText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  caseHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  caseInfo: {
    flex: 1,
    marginRight: 12,
  },
  caseDiagnosis: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  casePatient: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  urgencyBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  urgencyBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  caseReason: {
    fontSize: 14,
    color: '#555',
    marginBottom: 12,
    lineHeight: 20,
  },
  caseFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  caseMetadata: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  caseDate: {
    fontSize: 12,
    color: '#999',
  },
  hasImagesBadge: {
    backgroundColor: '#E3F2FD',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  hasImagesBadgeText: {
    fontSize: 10,
    color: '#1976D2',
    fontWeight: '500',
  },
  slaInfo: {
    alignItems: 'flex-end',
  },
  slaText: {
    fontSize: 12,
    color: '#666',
    fontWeight: '500',
  },
  statusIndicator: {
    position: 'absolute',
    top: 16,
    right: 16,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  statusIndicatorText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '500',
    textTransform: 'capitalize',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#F5F7FA',
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 44,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  closeButton: {
    fontSize: 16,
    color: '#4A90A4',
    fontWeight: '500',
  },
  modalContent: {
    flex: 1,
    padding: 16,
  },
  sectionCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 12,
    color: '#666',
    textTransform: 'uppercase',
    marginBottom: 8,
    fontWeight: '500',
  },
  diagnosisText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  providerText: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  contentText: {
    fontSize: 15,
    color: '#333',
    lineHeight: 22,
  },
  questionItem: {
    fontSize: 15,
    color: '#333',
    lineHeight: 24,
  },
  viewImagesButton: {
    backgroundColor: '#E3F2FD',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 12,
  },
  viewImagesButtonText: {
    color: '#1976D2',
    fontSize: 14,
    fontWeight: '500',
  },
  divider: {
    height: 1,
    backgroundColor: '#E0E0E0',
    marginVertical: 20,
  },
  formSectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    color: '#555',
    marginBottom: 8,
    fontWeight: '500',
  },
  sublabel: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
  },
  input: {
    backgroundColor: '#F5F5F5',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    color: '#333',
  },
  multilineInput: {
    minHeight: 100,
    textAlignVertical: 'top',
  },
  agreementToggle: {
    flexDirection: 'row',
    gap: 12,
  },
  agreementButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: '#F5F5F5',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  agreementButtonActive: {
    backgroundColor: '#E8F5E9',
    borderColor: '#4CAF50',
  },
  agreementButtonDifferent: {
    backgroundColor: '#FFF3E0',
    borderColor: '#FF9800',
  },
  agreementButtonText: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  agreementButtonTextActive: {
    color: '#333',
  },
  confidenceContainer: {
    flexDirection: 'row',
    gap: 8,
  },
  confidenceButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: '#F5F5F5',
    alignItems: 'center',
  },
  confidenceButtonActive: {
    backgroundColor: '#4A90A4',
  },
  confidenceButtonText: {
    fontSize: 14,
    color: '#666',
  },
  confidenceButtonTextActive: {
    color: '#fff',
    fontWeight: '500',
  },
  actionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  actionButton: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#F5F5F5',
  },
  actionButtonActive: {
    backgroundColor: '#4A90A4',
  },
  actionButtonText: {
    fontSize: 13,
    color: '#666',
  },
  actionButtonTextActive: {
    color: '#fff',
    fontWeight: '500',
  },
  biopsyToggle: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  submitButton: {
    backgroundColor: '#4A90A4',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 16,
  },
  submitButtonDisabled: {
    backgroundColor: '#B0BEC5',
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
