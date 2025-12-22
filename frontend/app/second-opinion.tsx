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
  Modal
} from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import { API_BASE_URL } from '../config';

interface SecondOpinionRequest {
  id: number;
  original_diagnosis: string;
  reason_for_second_opinion: string;
  urgency: string;
  status: string;
  dermatologist_id?: number;
  dermatologist_name?: string;
  second_opinion_diagnosis?: string;
  agrees_with_original?: boolean;
  created_at: string;
  sla_status?: {
    deadline: string;
    hours_remaining: number;
    is_urgent: boolean;
    is_breached: boolean;
  };
}

interface Pricing {
  base_price: number;
  specialty_fee: number;
  platform_fee: number;
  total: number;
  currency: string;
}

interface CreditPackage {
  id: string;
  name: string;
  credits: number;
  price: number;
  price_per_credit: number;
  savings: number;
  popular?: boolean;
  best_value?: boolean;
}

interface DashboardStats {
  role: string;
  total_requests?: number;
  completed?: number;
  pending?: number;
  // Dermatologist specific
  total_reviews?: number;
  completed_reviews?: number;
  pending_reviews?: number;
  urgent_pending?: number;
  average_rating?: number;
}

interface Dermatologist {
  id: number;
  full_name: string;
  specialties: string[];
  rating: number;
  accepts_second_opinions: boolean;
}

export default function SecondOpinionScreen() {
  const { user } = useAuth();
  const { t } = useTranslation();
  const params = useLocalSearchParams();

  // Tab state
  const [activeTab, setActiveTab] = useState<'request' | 'history' | 'credits'>('request');
  const [isLoading, setIsLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // Request form state
  const [originalDiagnosis, setOriginalDiagnosis] = useState('');
  const [originalProvider, setOriginalProvider] = useState('');
  const [originalTreatmentPlan, setOriginalTreatmentPlan] = useState('');
  const [reasonForSecondOpinion, setReasonForSecondOpinion] = useState('uncertainty');
  const [specificQuestions, setSpecificQuestions] = useState<string[]>(['']);
  const [concerns, setConcerns] = useState('');
  const [urgency, setUrgency] = useState<'routine' | 'urgent' | 'emergency'>('routine');
  const [specialtyPreference, setSpecialtyPreference] = useState('');
  const [preferredDermatologist, setPreferredDermatologist] = useState<number | null>(null);
  const [useCredits, setUseCredits] = useState(false);

  // Linked analysis
  const [linkedAnalysisId, setLinkedAnalysisId] = useState<number | null>(
    params.analysisId ? Number(params.analysisId) : null
  );

  // Pricing state
  const [pricing, setPricing] = useState<Pricing | null>(null);
  const [loadingPricing, setLoadingPricing] = useState(false);

  // History state
  const [requests, setRequests] = useState<SecondOpinionRequest[]>([]);
  const [selectedRequest, setSelectedRequest] = useState<SecondOpinionRequest | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  // Credits state
  const [creditPackages, setCreditPackages] = useState<CreditPackage[]>([]);
  const [userCredits, setUserCredits] = useState(0);

  // Dashboard stats
  const [stats, setStats] = useState<DashboardStats | null>(null);

  // Dermatologists
  const [dermatologists, setDermatologists] = useState<Dermatologist[]>([]);

  // Rating modal
  const [showRatingModal, setShowRatingModal] = useState(false);
  const [ratingValue, setRatingValue] = useState(5);
  const [ratingFeedback, setRatingFeedback] = useState('');
  const [requestToRate, setRequestToRate] = useState<number | null>(null);

  useEffect(() => {
    loadStats();
    if (activeTab === 'history') {
      loadRequests();
    } else if (activeTab === 'credits') {
      loadCreditPackages();
    } else if (activeTab === 'request') {
      loadPricing();
      loadDermatologists();
    }
  }, [activeTab, urgency, specialtyPreference]);

  const loadStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/second-opinions/stats/dashboard`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.log('Error loading stats:', error);
    }
  };

  const loadPricing = async () => {
    try {
      setLoadingPricing(true);
      const params = new URLSearchParams({ urgency });
      if (specialtyPreference) params.append('specialty', specialtyPreference);

      const response = await fetch(`${API_BASE_URL}/second-opinions/pricing?${params}`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setPricing(data);
      }
    } catch (error) {
      console.log('Error loading pricing:', error);
    } finally {
      setLoadingPricing(false);
    }
  };

  const loadRequests = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/second-opinions?limit=50`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setRequests(data.second_opinions || []);
      }
    } catch (error) {
      console.log('Error loading requests:', error);
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const loadCreditPackages = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/second-opinions/credits/packages`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setCreditPackages(data.packages || []);
      }
    } catch (error) {
      console.log('Error loading credit packages:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadDermatologists = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/dermatologists?accepts_second_opinions=true&limit=20`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setDermatologists(data.dermatologists || []);
      }
    } catch (error) {
      console.log('Error loading dermatologists:', error);
    }
  };

  const addQuestion = () => {
    setSpecificQuestions([...specificQuestions, '']);
  };

  const updateQuestion = (index: number, value: string) => {
    const updated = [...specificQuestions];
    updated[index] = value;
    setSpecificQuestions(updated);
  };

  const removeQuestion = (index: number) => {
    if (specificQuestions.length > 1) {
      setSpecificQuestions(specificQuestions.filter((_, i) => i !== index));
    }
  };

  const submitRequest = async () => {
    if (!originalDiagnosis.trim()) {
      Alert.alert('Required', 'Please enter the original diagnosis');
      return;
    }

    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append('original_diagnosis', originalDiagnosis);
      formData.append('reason_for_second_opinion', reasonForSecondOpinion);
      formData.append('urgency', urgency);
      formData.append('use_credits', useCredits.toString());

      if (originalProvider) formData.append('original_provider_name', originalProvider);
      if (originalTreatmentPlan) formData.append('original_treatment_plan', originalTreatmentPlan);
      if (concerns) formData.append('concerns', concerns);
      if (linkedAnalysisId) formData.append('analysis_id', linkedAnalysisId.toString());
      if (specialtyPreference) formData.append('specialty_preference', specialtyPreference);
      if (preferredDermatologist) formData.append('preferred_dermatologist_id', preferredDermatologist.toString());

      const filteredQuestions = specificQuestions.filter(q => q.trim());
      if (filteredQuestions.length > 0) {
        formData.append('specific_questions', JSON.stringify(filteredQuestions));
      }

      const response = await fetch(`${API_BASE_URL}/second-opinions/workflow/submit`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user?.token}` },
        body: formData
      });

      const data = await response.json();

      if (response.ok && data.success) {
        if (data.payment_required) {
          Alert.alert(
            'Payment Required',
            `Your request has been created. Total: $${data.pricing?.total?.toFixed(2) || pricing?.total?.toFixed(2)}. Please complete payment to proceed.`,
            [
              { text: 'Pay Later', style: 'cancel', onPress: () => {
                resetForm();
                setActiveTab('history');
              }},
              { text: 'Pay Now', onPress: () => handlePayment(data.second_opinion_id) }
            ]
          );
        } else {
          Alert.alert(
            'Success',
            `Your second opinion request has been submitted${data.dermatologist_id ? ' and assigned to a dermatologist' : ''}. Expected response: ${getExpectedResponseTime(urgency)}`,
            [{ text: 'OK', onPress: () => {
              resetForm();
              setActiveTab('history');
              loadRequests();
            }}]
          );
        }
      } else {
        Alert.alert('Error', data.message || 'Failed to submit request');
      }
    } catch (error) {
      console.error('Error submitting request:', error);
      Alert.alert('Error', 'Failed to submit request. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePayment = async (opinionId: number) => {
    // In a real app, this would integrate with Stripe Elements or similar
    Alert.alert(
      'Payment',
      'Payment integration would open here. For demo purposes, simulating successful payment.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Simulate Payment', onPress: async () => {
          try {
            setIsLoading(true);
            const formData = new FormData();
            formData.append('payment_method_id', 'pm_demo_card');

            const response = await fetch(`${API_BASE_URL}/second-opinions/${opinionId}/process-payment`, {
              method: 'POST',
              headers: { 'Authorization': `Bearer ${user?.token}` },
              body: formData
            });

            const data = await response.json();
            if (data.success) {
              Alert.alert('Payment Successful', 'Your payment has been processed and a dermatologist will be assigned.');
              resetForm();
              setActiveTab('history');
              loadRequests();
            } else {
              Alert.alert('Payment Failed', data.error || 'Please try again');
            }
          } catch (error) {
            Alert.alert('Error', 'Payment processing failed');
          } finally {
            setIsLoading(false);
          }
        }}
      ]
    );
  };

  const resetForm = () => {
    setOriginalDiagnosis('');
    setOriginalProvider('');
    setOriginalTreatmentPlan('');
    setReasonForSecondOpinion('uncertainty');
    setSpecificQuestions(['']);
    setConcerns('');
    setUrgency('routine');
    setSpecialtyPreference('');
    setPreferredDermatologist(null);
    setUseCredits(false);
    setLinkedAnalysisId(null);
  };

  const getExpectedResponseTime = (urg: string) => {
    switch (urg) {
      case 'emergency': return 'within 4 hours';
      case 'urgent': return 'within 24 hours';
      default: return 'within 7 days';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return '#4CAF50';
      case 'assigned':
      case 'in_review': return '#2196F3';
      case 'pending_payment': return '#FF9800';
      case 'cancelled': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  const viewRequestDetails = async (requestId: number) => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/second-opinions/${requestId}`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (response.ok) {
        const data = await response.json();

        // Also get SLA status
        const slaResponse = await fetch(`${API_BASE_URL}/second-opinions/${requestId}/sla-status`, {
          headers: { 'Authorization': `Bearer ${user?.token}` }
        });
        if (slaResponse.ok) {
          const slaData = await slaResponse.json();
          data.sla_info = slaData;
        }

        setSelectedRequest(data);
        setShowDetailModal(true);
      }
    } catch (error) {
      console.log('Error loading request details:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const submitRating = async () => {
    if (!requestToRate) return;

    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append('rating', ratingValue.toString());
      formData.append('satisfied', (ratingValue >= 4).toString());
      if (ratingFeedback) formData.append('feedback', ratingFeedback);

      const response = await fetch(`${API_BASE_URL}/second-opinions/${requestToRate}/rate`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user?.token}` },
        body: formData
      });

      if (response.ok) {
        Alert.alert('Thank You', 'Your feedback has been submitted');
        setShowRatingModal(false);
        setRatingValue(5);
        setRatingFeedback('');
        setRequestToRate(null);
        loadRequests();
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to submit rating');
    } finally {
      setIsLoading(false);
    }
  };

  const renderStats = () => {
    if (!stats) return null;

    return (
      <View style={styles.statsContainer}>
        <View style={styles.statCard}>
          <Text style={styles.statValue}>{stats.total_requests || stats.total_reviews || 0}</Text>
          <Text style={styles.statLabel}>Total</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={[styles.statValue, { color: '#4CAF50' }]}>{stats.completed || stats.completed_reviews || 0}</Text>
          <Text style={styles.statLabel}>Completed</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={[styles.statValue, { color: '#2196F3' }]}>{stats.pending || stats.pending_reviews || 0}</Text>
          <Text style={styles.statLabel}>Pending</Text>
        </View>
        {stats.urgent_pending !== undefined && (
          <View style={styles.statCard}>
            <Text style={[styles.statValue, { color: '#FF5722' }]}>{stats.urgent_pending}</Text>
            <Text style={styles.statLabel}>Urgent</Text>
          </View>
        )}
      </View>
    );
  };

  const renderRequestForm = () => (
    <ScrollView style={styles.formContainer}>
      <Text style={styles.sectionTitle}>Original Diagnosis Information</Text>

      <Text style={styles.label}>Original Diagnosis *</Text>
      <TextInput
        style={styles.input}
        value={originalDiagnosis}
        onChangeText={setOriginalDiagnosis}
        placeholder="e.g., Seborrheic Keratosis"
        placeholderTextColor="#999"
      />

      <Text style={styles.label}>Original Provider</Text>
      <TextInput
        style={styles.input}
        value={originalProvider}
        onChangeText={setOriginalProvider}
        placeholder="Name of diagnosing provider"
        placeholderTextColor="#999"
      />

      <Text style={styles.label}>Original Treatment Plan</Text>
      <TextInput
        style={[styles.input, styles.multilineInput]}
        value={originalTreatmentPlan}
        onChangeText={setOriginalTreatmentPlan}
        placeholder="Current treatment recommendations"
        placeholderTextColor="#999"
        multiline
      />

      <Text style={styles.sectionTitle}>Second Opinion Request</Text>

      <Text style={styles.label}>Reason for Second Opinion</Text>
      <View style={styles.optionsRow}>
        {[
          { value: 'uncertainty', label: 'Uncertainty' },
          { value: 'treatment_options', label: 'Treatment Options' },
          { value: 'confirmation', label: 'Confirmation' },
          { value: 'other', label: 'Other' }
        ].map(option => (
          <Pressable
            key={option.value}
            style={[
              styles.optionButton,
              reasonForSecondOpinion === option.value && styles.optionButtonSelected
            ]}
            onPress={() => setReasonForSecondOpinion(option.value)}
          >
            <Text style={[
              styles.optionButtonText,
              reasonForSecondOpinion === option.value && styles.optionButtonTextSelected
            ]}>
              {option.label}
            </Text>
          </Pressable>
        ))}
      </View>

      <Text style={styles.label}>Urgency Level</Text>
      <View style={styles.urgencyContainer}>
        {[
          { value: 'routine', label: 'Routine', time: '7 days', color: '#4CAF50' },
          { value: 'urgent', label: 'Urgent', time: '24 hours', color: '#FF9800' },
          { value: 'emergency', label: 'Emergency', time: '4 hours', color: '#F44336' }
        ].map(option => (
          <Pressable
            key={option.value}
            style={[
              styles.urgencyButton,
              urgency === option.value && { backgroundColor: option.color }
            ]}
            onPress={() => setUrgency(option.value as any)}
          >
            <Text style={[
              styles.urgencyButtonText,
              urgency === option.value && styles.urgencyButtonTextSelected
            ]}>
              {option.label}
            </Text>
            <Text style={[
              styles.urgencyTimeText,
              urgency === option.value && styles.urgencyButtonTextSelected
            ]}>
              {option.time}
            </Text>
          </Pressable>
        ))}
      </View>

      <Text style={styles.label}>Specific Questions</Text>
      {specificQuestions.map((question, index) => (
        <View key={index} style={styles.questionRow}>
          <TextInput
            style={[styles.input, { flex: 1 }]}
            value={question}
            onChangeText={(text) => updateQuestion(index, text)}
            placeholder={`Question ${index + 1}`}
            placeholderTextColor="#999"
          />
          {specificQuestions.length > 1 && (
            <Pressable
              style={styles.removeButton}
              onPress={() => removeQuestion(index)}
            >
              <Text style={styles.removeButtonText}>×</Text>
            </Pressable>
          )}
        </View>
      ))}
      <Pressable style={styles.addButton} onPress={addQuestion}>
        <Text style={styles.addButtonText}>+ Add Question</Text>
      </Pressable>

      <Text style={styles.label}>Additional Concerns</Text>
      <TextInput
        style={[styles.input, styles.multilineInput]}
        value={concerns}
        onChangeText={setConcerns}
        placeholder="Any other concerns or information"
        placeholderTextColor="#999"
        multiline
      />

      <Text style={styles.sectionTitle}>Dermatologist Preference (Optional)</Text>

      <Text style={styles.label}>Specialty</Text>
      <View style={styles.optionsRow}>
        {[
          { value: '', label: 'Any' },
          { value: 'dermoscopy', label: 'Dermoscopy' },
          { value: 'skin_cancer', label: 'Skin Cancer' },
          { value: 'pediatric', label: 'Pediatric' }
        ].map(option => (
          <Pressable
            key={option.value}
            style={[
              styles.optionButton,
              specialtyPreference === option.value && styles.optionButtonSelected
            ]}
            onPress={() => setSpecialtyPreference(option.value)}
          >
            <Text style={[
              styles.optionButtonText,
              specialtyPreference === option.value && styles.optionButtonTextSelected
            ]}>
              {option.label}
            </Text>
          </Pressable>
        ))}
      </View>

      {dermatologists.length > 0 && (
        <>
          <Text style={styles.label}>Preferred Dermatologist</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.dermatologistsScroll}>
            <Pressable
              style={[
                styles.dermatologistCard,
                !preferredDermatologist && styles.dermatologistCardSelected
              ]}
              onPress={() => setPreferredDermatologist(null)}
            >
              <Text style={styles.dermatologistName}>Auto-Assign</Text>
              <Text style={styles.dermatologistInfo}>Best match</Text>
            </Pressable>
            {dermatologists.map(derm => (
              <Pressable
                key={derm.id}
                style={[
                  styles.dermatologistCard,
                  preferredDermatologist === derm.id && styles.dermatologistCardSelected
                ]}
                onPress={() => setPreferredDermatologist(derm.id)}
              >
                <Text style={styles.dermatologistName}>{derm.full_name}</Text>
                <Text style={styles.dermatologistInfo}>
                  {'★'.repeat(Math.round(derm.rating || 0))} {derm.rating?.toFixed(1) || 'N/A'}
                </Text>
              </Pressable>
            ))}
          </ScrollView>
        </>
      )}

      {linkedAnalysisId && (
        <View style={styles.linkedAnalysisBox}>
          <Text style={styles.linkedAnalysisText}>
            Linked to Analysis #{linkedAnalysisId}
          </Text>
        </View>
      )}

      <Text style={styles.sectionTitle}>Payment</Text>

      {loadingPricing ? (
        <ActivityIndicator size="small" color="#4A90A4" />
      ) : pricing && (
        <View style={styles.pricingBox}>
          <View style={styles.pricingRow}>
            <Text style={styles.pricingLabel}>Base Price ({urgency})</Text>
            <Text style={styles.pricingValue}>${pricing.base_price.toFixed(2)}</Text>
          </View>
          {pricing.specialty_fee > 0 && (
            <View style={styles.pricingRow}>
              <Text style={styles.pricingLabel}>Specialty Fee</Text>
              <Text style={styles.pricingValue}>${pricing.specialty_fee.toFixed(2)}</Text>
            </View>
          )}
          <View style={styles.pricingRow}>
            <Text style={styles.pricingLabel}>Platform Fee</Text>
            <Text style={styles.pricingValue}>${pricing.platform_fee.toFixed(2)}</Text>
          </View>
          <View style={[styles.pricingRow, styles.pricingTotal]}>
            <Text style={styles.pricingTotalLabel}>Total</Text>
            <Text style={styles.pricingTotalValue}>${pricing.total.toFixed(2)}</Text>
          </View>
        </View>
      )}

      {userCredits > 0 && (
        <Pressable
          style={[styles.creditToggle, useCredits && styles.creditToggleActive]}
          onPress={() => setUseCredits(!useCredits)}
        >
          <View style={styles.checkbox}>
            {useCredits && <Text style={styles.checkmark}>✓</Text>}
          </View>
          <Text style={styles.creditToggleText}>
            Use credit ({userCredits} available)
          </Text>
        </Pressable>
      )}

      <Pressable
        style={[styles.submitButton, isLoading && styles.submitButtonDisabled]}
        onPress={submitRequest}
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.submitButtonText}>
            {useCredits ? 'Submit Request (Using Credit)' : `Submit Request - $${pricing?.total?.toFixed(2) || '...'}`}
          </Text>
        )}
      </Pressable>
    </ScrollView>
  );

  const renderHistory = () => (
    <ScrollView
      style={styles.historyContainer}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={() => {
          setRefreshing(true);
          loadRequests();
        }} />
      }
    >
      {requests.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateText}>No second opinion requests yet</Text>
          <Pressable
            style={styles.emptyStateButton}
            onPress={() => setActiveTab('request')}
          >
            <Text style={styles.emptyStateButtonText}>Request Second Opinion</Text>
          </Pressable>
        </View>
      ) : (
        requests.map(request => (
          <Pressable
            key={request.id}
            style={styles.requestCard}
            onPress={() => viewRequestDetails(request.id)}
          >
            <View style={styles.requestHeader}>
              <Text style={styles.requestDiagnosis}>{request.original_diagnosis}</Text>
              <View style={[styles.statusBadge, { backgroundColor: getStatusColor(request.status) }]}>
                <Text style={styles.statusText}>{request.status.replace('_', ' ')}</Text>
              </View>
            </View>
            <Text style={styles.requestReason}>{request.reason_for_second_opinion}</Text>
            <View style={styles.requestFooter}>
              <Text style={styles.requestDate}>
                {new Date(request.created_at).toLocaleDateString()}
              </Text>
              <View style={[styles.urgencyBadge, {
                backgroundColor: request.urgency === 'emergency' ? '#FFE0E0' :
                                 request.urgency === 'urgent' ? '#FFF3E0' : '#E8F5E9'
              }]}>
                <Text style={[styles.urgencyBadgeText, {
                  color: request.urgency === 'emergency' ? '#D32F2F' :
                         request.urgency === 'urgent' ? '#EF6C00' : '#388E3C'
                }]}>
                  {request.urgency}
                </Text>
              </View>
            </View>
            {request.status === 'completed' && !request.patient_rating && (
              <Pressable
                style={styles.rateButton}
                onPress={() => {
                  setRequestToRate(request.id);
                  setShowRatingModal(true);
                }}
              >
                <Text style={styles.rateButtonText}>Rate this review</Text>
              </Pressable>
            )}
          </Pressable>
        ))
      )}
    </ScrollView>
  );

  const renderCredits = () => (
    <ScrollView style={styles.creditsContainer}>
      <View style={styles.currentCreditsBox}>
        <Text style={styles.currentCreditsLabel}>Your Credits</Text>
        <Text style={styles.currentCreditsValue}>{userCredits}</Text>
      </View>

      <Text style={styles.sectionTitle}>Purchase Credits</Text>

      {creditPackages.map(pkg => (
        <Pressable
          key={pkg.id}
          style={[styles.packageCard, pkg.popular && styles.packageCardPopular]}
        >
          {pkg.popular && (
            <View style={styles.popularBadge}>
              <Text style={styles.popularBadgeText}>POPULAR</Text>
            </View>
          )}
          {pkg.best_value && (
            <View style={[styles.popularBadge, { backgroundColor: '#4CAF50' }]}>
              <Text style={styles.popularBadgeText}>BEST VALUE</Text>
            </View>
          )}
          <View style={styles.packageHeader}>
            <Text style={styles.packageName}>{pkg.name}</Text>
            <Text style={styles.packagePrice}>${pkg.price.toFixed(2)}</Text>
          </View>
          <View style={styles.packageDetails}>
            <Text style={styles.packageCredits}>{pkg.credits} credit{pkg.credits > 1 ? 's' : ''}</Text>
            <Text style={styles.packagePerCredit}>${pkg.price_per_credit.toFixed(2)}/credit</Text>
            {pkg.savings > 0 && (
              <Text style={styles.packageSavings}>Save ${pkg.savings.toFixed(2)}</Text>
            )}
          </View>
          <Pressable style={styles.buyButton}>
            <Text style={styles.buyButtonText}>Purchase</Text>
          </Pressable>
        </Pressable>
      ))}
    </ScrollView>
  );

  const renderDetailModal = () => (
    <Modal
      visible={showDetailModal}
      animationType="slide"
      onRequestClose={() => setShowDetailModal(false)}
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <Text style={styles.modalTitle}>Second Opinion Details</Text>
          <Pressable onPress={() => setShowDetailModal(false)}>
            <Text style={styles.closeButton}>×</Text>
          </Pressable>
        </View>

        {selectedRequest && (
          <ScrollView style={styles.modalContent}>
            <View style={styles.detailSection}>
              <Text style={styles.detailSectionTitle}>Original Diagnosis</Text>
              <Text style={styles.detailText}>{selectedRequest.original_diagnosis}</Text>
            </View>

            <View style={styles.detailSection}>
              <Text style={styles.detailSectionTitle}>Status</Text>
              <View style={[styles.statusBadgeLarge, { backgroundColor: getStatusColor(selectedRequest.status) }]}>
                <Text style={styles.statusTextLarge}>{selectedRequest.status.replace('_', ' ')}</Text>
              </View>
            </View>

            {selectedRequest.sla_info && (
              <View style={styles.detailSection}>
                <Text style={styles.detailSectionTitle}>Response Timeline</Text>
                <View style={styles.slaBox}>
                  <Text style={styles.slaDeadline}>
                    Due: {new Date(selectedRequest.sla_info.sla_deadline).toLocaleString()}
                  </Text>
                  <Text style={[
                    styles.slaRemaining,
                    selectedRequest.sla_info.is_breached && { color: '#F44336' }
                  ]}>
                    {selectedRequest.sla_info.is_breached
                      ? 'SLA Breached'
                      : `${Math.round(selectedRequest.sla_info.time_remaining_hours)} hours remaining`}
                  </Text>
                </View>
              </View>
            )}

            {selectedRequest.second_opinion_diagnosis && (
              <>
                <View style={styles.detailSection}>
                  <Text style={styles.detailSectionTitle}>Second Opinion</Text>
                  <Text style={styles.detailText}>{selectedRequest.second_opinion_diagnosis}</Text>
                </View>

                <View style={styles.detailSection}>
                  <Text style={styles.detailSectionTitle}>Agreement</Text>
                  <Text style={[styles.agreementText, {
                    color: selectedRequest.agrees_with_original_diagnosis ? '#4CAF50' : '#FF9800'
                  }]}>
                    {selectedRequest.agrees_with_original_diagnosis
                      ? '✓ Agrees with original diagnosis'
                      : '⚠ Different opinion from original'}
                  </Text>
                </View>

                {selectedRequest.second_opinion_notes && (
                  <View style={styles.detailSection}>
                    <Text style={styles.detailSectionTitle}>Notes</Text>
                    <Text style={styles.detailText}>{selectedRequest.second_opinion_notes}</Text>
                  </View>
                )}

                {selectedRequest.recommended_next_steps && (
                  <View style={styles.detailSection}>
                    <Text style={styles.detailSectionTitle}>Recommended Next Steps</Text>
                    <Text style={styles.detailText}>{selectedRequest.recommended_next_steps}</Text>
                  </View>
                )}

                {selectedRequest.biopsy_recommended && (
                  <View style={styles.warningBox}>
                    <Text style={styles.warningText}>⚠️ Biopsy Recommended</Text>
                  </View>
                )}
              </>
            )}
          </ScrollView>
        )}
      </View>
    </Modal>
  );

  const renderRatingModal = () => (
    <Modal
      visible={showRatingModal}
      transparent
      animationType="fade"
      onRequestClose={() => setShowRatingModal(false)}
    >
      <View style={styles.ratingModalOverlay}>
        <View style={styles.ratingModalContent}>
          <Text style={styles.ratingModalTitle}>Rate Your Experience</Text>

          <View style={styles.starsContainer}>
            {[1, 2, 3, 4, 5].map(star => (
              <Pressable key={star} onPress={() => setRatingValue(star)}>
                <Text style={[styles.star, ratingValue >= star && styles.starSelected]}>
                  ★
                </Text>
              </Pressable>
            ))}
          </View>

          <TextInput
            style={[styles.input, styles.feedbackInput]}
            value={ratingFeedback}
            onChangeText={setRatingFeedback}
            placeholder="Additional feedback (optional)"
            placeholderTextColor="#999"
            multiline
          />

          <View style={styles.ratingModalButtons}>
            <Pressable
              style={[styles.ratingButton, styles.ratingButtonCancel]}
              onPress={() => setShowRatingModal(false)}
            >
              <Text style={styles.ratingButtonCancelText}>Cancel</Text>
            </Pressable>
            <Pressable
              style={[styles.ratingButton, styles.ratingButtonSubmit]}
              onPress={submitRating}
            >
              <Text style={styles.ratingButtonSubmitText}>Submit</Text>
            </Pressable>
          </View>
        </View>
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
        <Text style={styles.headerTitle}>Second Opinion</Text>
        <View style={{ width: 40 }} />
      </View>

      {renderStats()}

      <View style={styles.tabContainer}>
        <Pressable
          style={[styles.tab, activeTab === 'request' && styles.tabActive]}
          onPress={() => setActiveTab('request')}
        >
          <Text style={[styles.tabText, activeTab === 'request' && styles.tabTextActive]}>
            Request
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tab, activeTab === 'history' && styles.tabActive]}
          onPress={() => setActiveTab('history')}
        >
          <Text style={[styles.tabText, activeTab === 'history' && styles.tabTextActive]}>
            History
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tab, activeTab === 'credits' && styles.tabActive]}
          onPress={() => setActiveTab('credits')}
        >
          <Text style={[styles.tabText, activeTab === 'credits' && styles.tabTextActive]}>
            Credits
          </Text>
        </Pressable>
      </View>

      {isLoading && !refreshing && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#4A90A4" />
        </View>
      )}

      {activeTab === 'request' && renderRequestForm()}
      {activeTab === 'history' && renderHistory()}
      {activeTab === 'credits' && renderCredits()}

      {renderDetailModal()}
      {renderRatingModal()}
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
  statsContainer: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
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
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  tab: {
    flex: 1,
    paddingVertical: 14,
    alignItems: 'center',
  },
  tabActive: {
    borderBottomWidth: 2,
    borderBottomColor: '#4A90A4',
  },
  tabText: {
    fontSize: 14,
    color: '#666',
  },
  tabTextActive: {
    color: '#4A90A4',
    fontWeight: '600',
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(255,255,255,0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 100,
  },
  formContainer: {
    flex: 1,
    padding: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginTop: 20,
    marginBottom: 12,
  },
  label: {
    fontSize: 14,
    color: '#555',
    marginBottom: 6,
    marginTop: 12,
  },
  input: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    color: '#333',
  },
  multilineInput: {
    minHeight: 80,
    textAlignVertical: 'top',
  },
  optionsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  optionButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  optionButtonSelected: {
    backgroundColor: '#4A90A4',
    borderColor: '#4A90A4',
  },
  optionButtonText: {
    fontSize: 14,
    color: '#666',
  },
  optionButtonTextSelected: {
    color: '#fff',
  },
  urgencyContainer: {
    flexDirection: 'row',
    gap: 8,
  },
  urgencyButton: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderRadius: 8,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#E0E0E0',
    alignItems: 'center',
  },
  urgencyButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  urgencyTimeText: {
    fontSize: 11,
    color: '#666',
    marginTop: 2,
  },
  urgencyButtonTextSelected: {
    color: '#fff',
  },
  questionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    gap: 8,
  },
  removeButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#FFE0E0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  removeButtonText: {
    fontSize: 20,
    color: '#D32F2F',
  },
  addButton: {
    padding: 12,
    alignItems: 'center',
  },
  addButtonText: {
    color: '#4A90A4',
    fontSize: 14,
    fontWeight: '500',
  },
  dermatologistsScroll: {
    marginTop: 8,
  },
  dermatologistCard: {
    width: 120,
    padding: 12,
    backgroundColor: '#fff',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    marginRight: 8,
  },
  dermatologistCardSelected: {
    borderColor: '#4A90A4',
    backgroundColor: '#E3F2FD',
  },
  dermatologistName: {
    fontSize: 12,
    fontWeight: '600',
    color: '#333',
  },
  dermatologistInfo: {
    fontSize: 11,
    color: '#666',
    marginTop: 4,
  },
  linkedAnalysisBox: {
    backgroundColor: '#E3F2FD',
    padding: 12,
    borderRadius: 8,
    marginTop: 12,
  },
  linkedAnalysisText: {
    color: '#1976D2',
    fontSize: 14,
  },
  pricingBox: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 16,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  pricingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  pricingLabel: {
    fontSize: 14,
    color: '#666',
  },
  pricingValue: {
    fontSize: 14,
    color: '#333',
  },
  pricingTotal: {
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0',
    paddingTop: 8,
    marginTop: 8,
  },
  pricingTotalLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  pricingTotalValue: {
    fontSize: 16,
    fontWeight: '700',
    color: '#4A90A4',
  },
  creditToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#fff',
    borderRadius: 8,
    marginTop: 12,
  },
  creditToggleActive: {
    backgroundColor: '#E8F5E9',
  },
  checkbox: {
    width: 24,
    height: 24,
    borderRadius: 4,
    borderWidth: 2,
    borderColor: '#4A90A4',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  checkmark: {
    color: '#4A90A4',
    fontSize: 16,
    fontWeight: '700',
  },
  creditToggleText: {
    fontSize: 14,
    color: '#333',
  },
  submitButton: {
    backgroundColor: '#4A90A4',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 24,
    marginBottom: 40,
  },
  submitButtonDisabled: {
    backgroundColor: '#B0BEC5',
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  historyContainer: {
    flex: 1,
    padding: 16,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyStateText: {
    fontSize: 16,
    color: '#666',
    marginBottom: 16,
  },
  emptyStateButton: {
    backgroundColor: '#4A90A4',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  emptyStateButtonText: {
    color: '#fff',
    fontWeight: '500',
  },
  requestCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  requestHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  requestDiagnosis: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    flex: 1,
    marginRight: 8,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
    textTransform: 'capitalize',
  },
  requestReason: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
  requestFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  requestDate: {
    fontSize: 12,
    color: '#999',
  },
  urgencyBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  urgencyBadgeText: {
    fontSize: 11,
    fontWeight: '500',
    textTransform: 'uppercase',
  },
  rateButton: {
    marginTop: 12,
    paddingVertical: 8,
    backgroundColor: '#FFF8E1',
    borderRadius: 6,
    alignItems: 'center',
  },
  rateButtonText: {
    color: '#F57C00',
    fontSize: 14,
    fontWeight: '500',
  },
  creditsContainer: {
    flex: 1,
    padding: 16,
  },
  currentCreditsBox: {
    backgroundColor: '#4A90A4',
    borderRadius: 12,
    padding: 24,
    alignItems: 'center',
    marginBottom: 24,
  },
  currentCreditsLabel: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 14,
  },
  currentCreditsValue: {
    color: '#fff',
    fontSize: 48,
    fontWeight: '700',
    marginTop: 4,
  },
  packageCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  packageCardPopular: {
    borderColor: '#4A90A4',
    borderWidth: 2,
  },
  popularBadge: {
    position: 'absolute',
    top: -10,
    right: 16,
    backgroundColor: '#4A90A4',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 10,
  },
  popularBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  packageHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  packageName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  packagePrice: {
    fontSize: 20,
    fontWeight: '700',
    color: '#4A90A4',
  },
  packageDetails: {
    marginBottom: 12,
  },
  packageCredits: {
    fontSize: 14,
    color: '#666',
  },
  packagePerCredit: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
  },
  packageSavings: {
    fontSize: 12,
    color: '#4CAF50',
    fontWeight: '500',
    marginTop: 4,
  },
  buyButton: {
    backgroundColor: '#4A90A4',
    paddingVertical: 10,
    borderRadius: 6,
    alignItems: 'center',
  },
  buyButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
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
    fontSize: 28,
    color: '#666',
    padding: 4,
  },
  modalContent: {
    flex: 1,
    padding: 16,
  },
  detailSection: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
  },
  detailSectionTitle: {
    fontSize: 12,
    color: '#666',
    textTransform: 'uppercase',
    marginBottom: 8,
  },
  detailText: {
    fontSize: 16,
    color: '#333',
    lineHeight: 24,
  },
  statusBadgeLarge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 16,
  },
  statusTextLarge: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  slaBox: {
    backgroundColor: '#F5F5F5',
    padding: 12,
    borderRadius: 6,
  },
  slaDeadline: {
    fontSize: 14,
    color: '#333',
  },
  slaRemaining: {
    fontSize: 14,
    color: '#4A90A4',
    fontWeight: '500',
    marginTop: 4,
  },
  agreementText: {
    fontSize: 16,
    fontWeight: '500',
  },
  warningBox: {
    backgroundColor: '#FFF3E0',
    padding: 16,
    borderRadius: 8,
    marginTop: 8,
  },
  warningText: {
    color: '#E65100',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  ratingModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  ratingModalContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    width: '85%',
    maxWidth: 400,
  },
  ratingModalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    textAlign: 'center',
    marginBottom: 20,
  },
  starsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 20,
  },
  star: {
    fontSize: 40,
    color: '#E0E0E0',
    marginHorizontal: 4,
  },
  starSelected: {
    color: '#FFB400',
  },
  feedbackInput: {
    minHeight: 100,
    marginBottom: 20,
  },
  ratingModalButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  ratingButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  ratingButtonCancel: {
    backgroundColor: '#F5F5F5',
  },
  ratingButtonCancelText: {
    color: '#666',
    fontSize: 14,
    fontWeight: '500',
  },
  ratingButtonSubmit: {
    backgroundColor: '#4A90A4',
  },
  ratingButtonSubmitText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
});
