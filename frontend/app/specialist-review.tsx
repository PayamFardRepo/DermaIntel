import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  Alert,
  Platform,
  ActivityIndicator,
  TextInput,
  Modal,
  Image,
  RefreshControl
} from 'react-native';
import { router } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';

interface AssignedCase {
  case_id: string;
  db_id: number;
  status: string;
  case_summary: string;
  urgency: string;
  deadline: string;
  created_at: string;
  assignment_status: string;
  opinion_submitted: boolean;
  images: string[];
}

interface SpecialistInfo {
  specialist_id: number;
  specialist_name: string;
  assigned_cases: AssignedCase[];
  pending_count: number;
  completed_count: number;
}

export default function SpecialistReviewScreen() {
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [specialistInfo, setSpecialistInfo] = useState<SpecialistInfo | null>(null);
  const [selectedCase, setSelectedCase] = useState<AssignedCase | null>(null);
  const [showOpinionModal, setShowOpinionModal] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Opinion form state
  const [diagnosis, setDiagnosis] = useState('');
  const [confidence, setConfidence] = useState(0.8);
  const [differentialDiagnoses, setDifferentialDiagnoses] = useState('');
  const [recommendedActions, setRecommendedActions] = useState('');
  const [clinicalNotes, setClinicalNotes] = useState('');
  const [agreesWithAI, setAgreesWithAI] = useState<boolean | null>(null);

  useEffect(() => {
    fetchAssignedCases();
  }, []);

  const fetchAssignedCases = async () => {
    try {
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert('Error', 'Please login to view assigned cases');
        router.push('/');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/teledermatology/advanced/consensus/specialist/assigned`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setSpecialistInfo(data);
      } else {
        const error = await response.json().catch(() => ({ detail: 'Failed to load cases' }));
        if (response.status === 403) {
          Alert.alert('Access Denied', 'You need to be registered as a specialist to view this screen.');
          router.back();
        } else {
          Alert.alert('Error', error.detail);
        }
      }
    } catch (error) {
      console.error('Error fetching cases:', error);
      Alert.alert('Error', 'Failed to connect to server');
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchAssignedCases();
  };

  const openCaseReview = (caseItem: AssignedCase) => {
    setSelectedCase(caseItem);
    // Reset form
    setDiagnosis('');
    setConfidence(0.8);
    setDifferentialDiagnoses('');
    setRecommendedActions('');
    setClinicalNotes('');
    setAgreesWithAI(null);
    setShowOpinionModal(true);
  };

  const submitOpinion = async () => {
    if (!selectedCase || !diagnosis.trim()) {
      Alert.alert('Error', 'Please provide a diagnosis');
      return;
    }

    try {
      setIsSubmitting(true);
      const token = AuthService.getToken();

      const response = await fetch(`${API_BASE_URL}/teledermatology/advanced/consensus/opinion`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          case_id: selectedCase.case_id,
          specialist_id: specialistInfo?.specialist_id,
          diagnosis: diagnosis,
          confidence: confidence,
          differential_diagnoses: differentialDiagnoses.split(',').map(d => d.trim()).filter(d => d),
          recommended_actions: recommendedActions.split(',').map(a => a.trim()).filter(a => a),
          notes: clinicalNotes,
          agrees_with_ai: agreesWithAI
        }),
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(
          'Opinion Submitted',
          `Your opinion has been recorded.\n\nOpinions received: ${data.opinions_received}/${data.opinions_expected}\nStatus: ${data.current_status}`,
          [{ text: 'OK', onPress: () => {
            setShowOpinionModal(false);
            fetchAssignedCases();
          }}]
        );
      } else {
        const error = await response.json().catch(() => ({ detail: 'Failed to submit opinion' }));
        Alert.alert('Error', error.detail);
      }
    } catch (error) {
      console.error('Error submitting opinion:', error);
      Alert.alert('Error', 'Failed to submit opinion');
    } finally {
      setIsSubmitting(false);
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'urgent': return '#dc2626';
      case 'emergency': return '#7f1d1d';
      default: return '#3b82f6';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return '#f59e0b';
      case 'in_review': return '#3b82f6';
      case 'consensus_reached': return '#10b981';
      case 'disagreement': return '#ef4444';
      default: return '#6b7280';
    }
  };

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#8b5cf6" />
        <Text style={styles.loadingText}>Loading assigned cases...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <LinearGradient colors={['#8b5cf6', '#7c3aed']} style={styles.header}>
        <Pressable onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </Pressable>
        <Text style={styles.title}>Specialist Review</Text>
        <Pressable onPress={onRefresh} style={styles.refreshButton}>
          <Ionicons name="refresh" size={24} color="#fff" />
        </Pressable>
      </LinearGradient>

      {specialistInfo && (
        <View style={styles.statsBar}>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{specialistInfo.pending_count}</Text>
            <Text style={styles.statLabel}>Pending</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{specialistInfo.completed_count}</Text>
            <Text style={styles.statLabel}>Completed</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{specialistInfo.assigned_cases.length}</Text>
            <Text style={styles.statLabel}>Total</Text>
          </View>
        </View>
      )}

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {specialistInfo?.assigned_cases.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons name="checkmark-circle" size={64} color="#10b981" />
            <Text style={styles.emptyTitle}>No Pending Cases</Text>
            <Text style={styles.emptySubtitle}>
              You have no cases assigned for review at this time.
            </Text>
          </View>
        ) : (
          specialistInfo?.assigned_cases.map((caseItem) => (
            <Pressable
              key={caseItem.case_id}
              style={[styles.caseCard, caseItem.opinion_submitted && styles.caseCardCompleted]}
              onPress={() => !caseItem.opinion_submitted && openCaseReview(caseItem)}
            >
              <View style={styles.caseHeader}>
                <View style={styles.caseIdContainer}>
                  <Text style={styles.caseId}>{caseItem.case_id}</Text>
                  <View style={[styles.urgencyBadge, { backgroundColor: getUrgencyColor(caseItem.urgency) }]}>
                    <Text style={styles.urgencyText}>{caseItem.urgency.toUpperCase()}</Text>
                  </View>
                </View>
                {caseItem.opinion_submitted ? (
                  <View style={styles.completedBadge}>
                    <Ionicons name="checkmark-circle" size={20} color="#10b981" />
                    <Text style={styles.completedText}>Submitted</Text>
                  </View>
                ) : (
                  <View style={[styles.statusBadge, { backgroundColor: getStatusColor(caseItem.status) }]}>
                    <Text style={styles.statusText}>{caseItem.status.replace('_', ' ')}</Text>
                  </View>
                )}
              </View>

              <Text style={styles.caseSummary} numberOfLines={3}>
                {caseItem.case_summary}
              </Text>

              <View style={styles.caseFooter}>
                <View style={styles.footerItem}>
                  <Ionicons name="calendar" size={14} color="#6b7280" />
                  <Text style={styles.footerText}>
                    Due: {new Date(caseItem.deadline).toLocaleDateString()}
                  </Text>
                </View>
                {caseItem.images?.length > 0 && (
                  <View style={styles.footerItem}>
                    <Ionicons name="images" size={14} color="#6b7280" />
                    <Text style={styles.footerText}>{caseItem.images.length} images</Text>
                  </View>
                )}
              </View>

              {!caseItem.opinion_submitted && (
                <View style={styles.reviewButton}>
                  <Text style={styles.reviewButtonText}>Review & Submit Opinion</Text>
                  <Ionicons name="arrow-forward" size={16} color="#8b5cf6" />
                </View>
              )}
            </Pressable>
          ))
        )}
        <View style={{ height: 40 }} />
      </ScrollView>

      {/* Opinion Modal */}
      <Modal
        visible={showOpinionModal}
        animationType="slide"
        onRequestClose={() => setShowOpinionModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Submit Opinion</Text>
            <Pressable onPress={() => setShowOpinionModal(false)}>
              <Ionicons name="close" size={24} color="#6b7280" />
            </Pressable>
          </View>

          <ScrollView style={styles.modalContent}>
            {selectedCase && (
              <>
                <View style={styles.caseInfo}>
                  <Text style={styles.caseInfoTitle}>{selectedCase.case_id}</Text>
                  <Text style={styles.caseInfoSummary}>{selectedCase.case_summary}</Text>
                </View>

                <View style={styles.formGroup}>
                  <Text style={styles.formLabel}>Primary Diagnosis *</Text>
                  <TextInput
                    style={styles.textInput}
                    value={diagnosis}
                    onChangeText={setDiagnosis}
                    placeholder="Enter your diagnosis..."
                    placeholderTextColor="#9ca3af"
                  />
                </View>

                <View style={styles.formGroup}>
                  <Text style={styles.formLabel}>Confidence Level: {Math.round(confidence * 100)}%</Text>
                  <View style={styles.confidenceSlider}>
                    {[0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((level) => (
                      <Pressable
                        key={level}
                        style={[styles.confidenceDot, confidence >= level && styles.confidenceDotActive]}
                        onPress={() => setConfidence(level)}
                      >
                        <Text style={styles.confidenceDotText}>{Math.round(level * 100)}</Text>
                      </Pressable>
                    ))}
                  </View>
                </View>

                <View style={styles.formGroup}>
                  <Text style={styles.formLabel}>Differential Diagnoses</Text>
                  <TextInput
                    style={styles.textInput}
                    value={differentialDiagnoses}
                    onChangeText={setDifferentialDiagnoses}
                    placeholder="Comma-separated list..."
                    placeholderTextColor="#9ca3af"
                  />
                </View>

                <View style={styles.formGroup}>
                  <Text style={styles.formLabel}>Recommended Actions</Text>
                  <TextInput
                    style={styles.textInput}
                    value={recommendedActions}
                    onChangeText={setRecommendedActions}
                    placeholder="Comma-separated list (e.g., biopsy, follow-up)..."
                    placeholderTextColor="#9ca3af"
                  />
                </View>

                <View style={styles.formGroup}>
                  <Text style={styles.formLabel}>Clinical Notes</Text>
                  <TextInput
                    style={[styles.textInput, styles.textArea]}
                    value={clinicalNotes}
                    onChangeText={setClinicalNotes}
                    placeholder="Additional observations and reasoning..."
                    placeholderTextColor="#9ca3af"
                    multiline
                    numberOfLines={4}
                  />
                </View>

                <View style={styles.formGroup}>
                  <Text style={styles.formLabel}>Do you agree with AI analysis?</Text>
                  <View style={styles.aiAgreementRow}>
                    <Pressable
                      style={[styles.aiButton, agreesWithAI === true && styles.aiButtonActive]}
                      onPress={() => setAgreesWithAI(true)}
                    >
                      <Ionicons name="thumbs-up" size={20} color={agreesWithAI === true ? '#fff' : '#6b7280'} />
                      <Text style={[styles.aiButtonText, agreesWithAI === true && styles.aiButtonTextActive]}>Yes</Text>
                    </Pressable>
                    <Pressable
                      style={[styles.aiButton, agreesWithAI === false && styles.aiButtonActiveNo]}
                      onPress={() => setAgreesWithAI(false)}
                    >
                      <Ionicons name="thumbs-down" size={20} color={agreesWithAI === false ? '#fff' : '#6b7280'} />
                      <Text style={[styles.aiButtonText, agreesWithAI === false && styles.aiButtonTextActive]}>No</Text>
                    </Pressable>
                    <Pressable
                      style={[styles.aiButton, agreesWithAI === null && styles.aiButtonActiveNeutral]}
                      onPress={() => setAgreesWithAI(null)}
                    >
                      <Ionicons name="help" size={20} color={agreesWithAI === null ? '#fff' : '#6b7280'} />
                      <Text style={[styles.aiButtonText, agreesWithAI === null && styles.aiButtonTextActive]}>N/A</Text>
                    </Pressable>
                  </View>
                </View>

                <View style={{ height: 100 }} />
              </>
            )}
          </ScrollView>

          <View style={styles.modalFooter}>
            <Pressable
              style={styles.cancelButton}
              onPress={() => setShowOpinionModal(false)}
            >
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </Pressable>
            <Pressable
              style={[styles.submitButton, isSubmitting && styles.submitButtonDisabled]}
              onPress={submitOpinion}
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <>
                  <Ionicons name="send" size={18} color="#fff" />
                  <Text style={styles.submitButtonText}>Submit Opinion</Text>
                </>
              )}
            </Pressable>
          </View>
        </View>
      </Modal>
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
    fontSize: 14,
    color: '#6b7280',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 16,
    paddingHorizontal: 16,
  },
  backButton: {
    padding: 8,
  },
  refreshButton: {
    padding: 8,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
  },
  statsBar: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1f2937',
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  statDivider: {
    width: 1,
    backgroundColor: '#e5e7eb',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  emptyState: {
    alignItems: 'center',
    paddingTop: 60,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1f2937',
    marginTop: 16,
  },
  emptySubtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center',
  },
  caseCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  caseCardCompleted: {
    opacity: 0.7,
    backgroundColor: '#f0fdf4',
  },
  caseHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  caseIdContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  caseId: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
  },
  urgencyBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  urgencyText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#fff',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    fontSize: 11,
    fontWeight: '500',
    color: '#fff',
    textTransform: 'capitalize',
  },
  completedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  completedText: {
    fontSize: 12,
    color: '#10b981',
    fontWeight: '500',
  },
  caseSummary: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
    marginBottom: 12,
  },
  caseFooter: {
    flexDirection: 'row',
    gap: 16,
  },
  footerItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  footerText: {
    fontSize: 12,
    color: '#6b7280',
  },
  reviewButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  reviewButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#8b5cf6',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
  },
  modalContent: {
    flex: 1,
    padding: 16,
  },
  caseInfo: {
    backgroundColor: '#f3f4f6',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
  },
  caseInfoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  caseInfoSummary: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
  },
  formGroup: {
    marginBottom: 20,
  },
  formLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    color: '#1f2937',
  },
  textArea: {
    minHeight: 100,
    textAlignVertical: 'top',
  },
  confidenceSlider: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  confidenceDot: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  confidenceDotActive: {
    backgroundColor: '#8b5cf6',
  },
  confidenceDotText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
  },
  aiAgreementRow: {
    flexDirection: 'row',
    gap: 12,
  },
  aiButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  aiButtonActive: {
    backgroundColor: '#10b981',
    borderColor: '#10b981',
  },
  aiButtonActiveNo: {
    backgroundColor: '#ef4444',
    borderColor: '#ef4444',
  },
  aiButtonActiveNeutral: {
    backgroundColor: '#6b7280',
    borderColor: '#6b7280',
  },
  aiButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#6b7280',
  },
  aiButtonTextActive: {
    color: '#fff',
  },
  modalFooter: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
    paddingBottom: Platform.OS === 'ios' ? 32 : 16,
  },
  cancelButton: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    alignItems: 'center',
  },
  cancelButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6b7280',
  },
  submitButton: {
    flex: 2,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#8b5cf6',
  },
  submitButtonDisabled: {
    backgroundColor: '#d1d5db',
  },
  submitButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
});
