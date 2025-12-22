/**
 * Complete Teledermatology Consultation Screen
 *
 * Features:
 * - Book video consultations with dermatologists
 * - View and manage upcoming/past consultations
 * - Video call integration (Zoom, Twilio, Agora)
 * - Payment processing for consultations
 * - Consultation notes viewing
 * - Real-time status updates
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Modal,
  Alert,
  ActivityIndicator,
  Platform,
  TextInput,
  Linking,
  RefreshControl,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

interface Dermatologist {
  id: number;
  full_name: string;
  credentials: string;
  practice_name: string;
  city: string;
  state: string;
  specializations: string[];
  accepts_video_consultations: boolean;
  accepts_referrals: boolean;
  availability_status: string;
  typical_wait_time_days: number;
  average_rating: number;
  consultation_fee: number;
  years_experience: number;
  is_verified: boolean;
}

interface Consultation {
  id: number;
  dermatologist_id: number;
  dermatologist_name: string;
  consultation_type: string;
  consultation_reason: string;
  scheduled_datetime: string;
  duration_minutes: number;
  status: string;
  video_platform: string;
  video_meeting_url: string | null;
  video_meeting_id: string | null;
  consultation_fee: number;
  payment_status: string;
  patient_notes: string | null;
  dermatologist_notes: string | null;
  diagnosis: string | null;
  treatment_plan: string | null;
  follow_up_needed: boolean;
  follow_up_timeframe: string | null;
  rating: number | null;
  created_at: string;
}

interface PaymentConfig {
  stripe_configured: boolean;
  stripe_publishable_key: string | null;
  demo_mode: boolean;
}

type TabType = 'upcoming' | 'past' | 'book';

export default function ConsultationsScreen() {
  const { t } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // State
  const [activeTab, setActiveTab] = useState<TabType>('upcoming');
  const [consultations, setConsultations] = useState<Consultation[]>([]);
  const [dermatologists, setDermatologists] = useState<Dermatologist[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [paymentConfig, setPaymentConfig] = useState<PaymentConfig | null>(null);

  // Booking state
  const [selectedDermatologist, setSelectedDermatologist] = useState<Dermatologist | null>(null);
  const [showDermatologistModal, setShowDermatologistModal] = useState(false);
  const [showBookingModal, setShowBookingModal] = useState(false);
  const [bookingType, setBookingType] = useState<'initial' | 'follow_up' | 'urgent'>('initial');
  const [bookingReason, setBookingReason] = useState('');
  const [bookingDate, setBookingDate] = useState('');
  const [bookingTime, setBookingTime] = useState('');
  const [bookingNotes, setBookingNotes] = useState('');
  const [isBooking, setIsBooking] = useState(false);

  // Payment state
  const [showPaymentModal, setShowPaymentModal] = useState(false);
  const [payingConsultation, setPayingConsultation] = useState<Consultation | null>(null);
  const [isProcessingPayment, setIsProcessingPayment] = useState(false);

  // Consultation detail state
  const [selectedConsultation, setSelectedConsultation] = useState<Consultation | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  // Search state
  const [searchCity, setSearchCity] = useState('');
  const [searchSpecialization, setSearchSpecialization] = useState('');

  // Auth headers
  const getAuthHeaders = async () => {
    const token = await AsyncStorage.getItem('accessToken');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  };

  // Initialize
  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
      return;
    }
    loadData();
    loadPaymentConfig();
  }, [isAuthenticated]);

  // Load data based on active tab
  useEffect(() => {
    if (activeTab === 'book') {
      loadDermatologists();
    } else {
      loadConsultations();
    }
  }, [activeTab]);

  const loadData = async () => {
    setIsLoading(true);
    await Promise.all([loadConsultations(), loadDermatologists()]);
    setIsLoading(false);
  };

  const loadConsultations = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/consultations?limit=50`, {
        headers,
      });

      if (response.ok) {
        const data = await response.json();
        setConsultations(data.consultations || []);
      }
    } catch (error) {
      console.error('Error loading consultations:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const loadDermatologists = async () => {
    try {
      const headers = await getAuthHeaders();
      let url = `${API_BASE_URL}/dermatologists?accepts_video=true&limit=50`;
      if (searchCity) url += `&city=${encodeURIComponent(searchCity)}`;
      if (searchSpecialization) url += `&specialization=${encodeURIComponent(searchSpecialization)}`;

      const response = await fetch(url, { headers });

      if (response.ok) {
        const data = await response.json();
        setDermatologists(data.dermatologists || []);
      }
    } catch (error) {
      console.error('Error loading dermatologists:', error);
    }
  };

  const loadPaymentConfig = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/payments/config`);
      if (response.ok) {
        const data = await response.json();
        setPaymentConfig(data);
      }
    } catch (error) {
      console.error('Error loading payment config:', error);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadConsultations();
  }, []);

  // Filter consultations
  const upcomingConsultations = consultations.filter(c =>
    ['scheduled', 'confirmed', 'in_progress'].includes(c.status) &&
    new Date(c.scheduled_datetime) >= new Date()
  );

  const pastConsultations = consultations.filter(c =>
    ['completed', 'cancelled', 'no_show'].includes(c.status) ||
    new Date(c.scheduled_datetime) < new Date()
  );

  // Book consultation
  const handleBookConsultation = async () => {
    if (!selectedDermatologist) {
      Alert.alert('Error', 'Please select a dermatologist');
      return;
    }

    if (!bookingReason.trim()) {
      Alert.alert('Error', 'Please enter a reason for the consultation');
      return;
    }

    if (!bookingDate || !bookingTime) {
      Alert.alert('Error', 'Please select a date and time');
      return;
    }

    setIsBooking(true);

    try {
      const headers = await getAuthHeaders();
      const formData = new FormData();
      formData.append('dermatologist_id', selectedDermatologist.id.toString());
      formData.append('consultation_type', bookingType);
      formData.append('consultation_reason', bookingReason);
      formData.append('scheduled_datetime', `${bookingDate}T${bookingTime}:00`);
      formData.append('duration_minutes', '30');
      if (bookingNotes) formData.append('patient_notes', bookingNotes);

      const response = await fetch(`${API_BASE_URL}/consultations/book`, {
        method: 'POST',
        headers: { 'Authorization': (await getAuthHeaders()).Authorization },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(
          'Consultation Booked',
          `Your consultation with ${data.dermatologist_name} has been scheduled for ${new Date(data.scheduled_datetime).toLocaleString()}`,
          [{ text: 'OK', onPress: () => {
            setShowBookingModal(false);
            resetBookingForm();
            setActiveTab('upcoming');
            loadConsultations();
          }}]
        );
      } else {
        const error = await response.json();
        Alert.alert('Booking Failed', error.detail || 'Unable to book consultation');
      }
    } catch (error) {
      console.error('Booking error:', error);
      Alert.alert('Error', 'Failed to book consultation');
    } finally {
      setIsBooking(false);
    }
  };

  const resetBookingForm = () => {
    setSelectedDermatologist(null);
    setBookingType('initial');
    setBookingReason('');
    setBookingDate('');
    setBookingTime('');
    setBookingNotes('');
  };

  // Cancel consultation
  const handleCancelConsultation = async (consultation: Consultation) => {
    Alert.alert(
      'Cancel Consultation',
      'Are you sure you want to cancel this consultation?',
      [
        { text: 'No', style: 'cancel' },
        {
          text: 'Yes, Cancel',
          style: 'destructive',
          onPress: async () => {
            try {
              const headers = await getAuthHeaders();
              const response = await fetch(
                `${API_BASE_URL}/consultations/${consultation.id}/cancel`,
                { method: 'PUT', headers }
              );

              if (response.ok) {
                Alert.alert('Success', 'Consultation cancelled');
                loadConsultations();
              } else {
                Alert.alert('Error', 'Failed to cancel consultation');
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to cancel consultation');
            }
          }
        }
      ]
    );
  };

  // Join video call
  const handleJoinVideoCall = async (consultation: Consultation) => {
    if (consultation.video_meeting_url) {
      Linking.openURL(consultation.video_meeting_url);
    } else {
      // Generate video link first
      try {
        const headers = await getAuthHeaders();
        const formData = new FormData();
        formData.append('platform', consultation.video_platform || 'zoom');

        const response = await fetch(
          `${API_BASE_URL}/consultations/${consultation.id}/generate-video-link`,
          {
            method: 'POST',
            headers: { 'Authorization': headers.Authorization },
            body: formData,
          }
        );

        if (response.ok) {
          const data = await response.json();
          Linking.openURL(data.meeting_url);
          loadConsultations();
        } else {
          Alert.alert('Error', 'Failed to generate video link');
        }
      } catch (error) {
        Alert.alert('Error', 'Failed to join video call');
      }
    }
  };

  // Process payment
  const handlePayment = async (consultation: Consultation) => {
    setPayingConsultation(consultation);

    if (paymentConfig?.demo_mode) {
      // Demo mode - simulate payment
      Alert.alert(
        'Demo Payment',
        `This would charge $${consultation.consultation_fee || 100} for the consultation.\n\nIn production, Stripe payment would be processed here.`,
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Simulate Payment',
            onPress: async () => {
              setIsProcessingPayment(true);
              try {
                const headers = await getAuthHeaders();
                const formData = new FormData();
                formData.append('consultation_id', consultation.id.toString());
                formData.append('amount', ((consultation.consultation_fee || 100) * 100).toString());

                const response = await fetch(`${API_BASE_URL}/payments/create-intent`, {
                  method: 'POST',
                  headers: { 'Authorization': headers.Authorization },
                  body: formData,
                });

                if (response.ok) {
                  const data = await response.json();

                  // Confirm payment
                  const confirmForm = new FormData();
                  confirmForm.append('consultation_id', consultation.id.toString());
                  confirmForm.append('payment_intent_id', data.payment_intent_id);

                  await fetch(`${API_BASE_URL}/payments/confirm`, {
                    method: 'POST',
                    headers: { 'Authorization': headers.Authorization },
                    body: confirmForm,
                  });

                  Alert.alert('Payment Successful', 'Your consultation has been paid for.');
                  loadConsultations();
                }
              } catch (error) {
                Alert.alert('Payment Failed', 'Unable to process payment');
              } finally {
                setIsProcessingPayment(false);
                setPayingConsultation(null);
              }
            }
          }
        ]
      );
    } else {
      setShowPaymentModal(true);
    }
  };

  // Rate consultation
  const handleRateConsultation = async (consultation: Consultation, rating: number) => {
    try {
      const headers = await getAuthHeaders();
      const formData = new FormData();
      formData.append('rating', rating.toString());

      const response = await fetch(
        `${API_BASE_URL}/consultations/${consultation.id}/rate`,
        {
          method: 'POST',
          headers: { 'Authorization': headers.Authorization },
          body: formData,
        }
      );

      if (response.ok) {
        Alert.alert('Thank You', 'Your rating has been submitted');
        loadConsultations();
        setShowDetailModal(false);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to submit rating');
    }
  };

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'scheduled': return '#3b82f6';
      case 'confirmed': return '#10b981';
      case 'in_progress': return '#f59e0b';
      case 'completed': return '#6b7280';
      case 'cancelled': return '#ef4444';
      case 'no_show': return '#dc2626';
      default: return '#6b7280';
    }
  };

  // Render consultation card
  const renderConsultationCard = (consultation: Consultation) => {
    const isUpcoming = ['scheduled', 'confirmed'].includes(consultation.status);
    const canJoin = isUpcoming && new Date(consultation.scheduled_datetime) <= new Date(Date.now() + 15 * 60 * 1000);
    const needsPayment = consultation.payment_status === 'pending' && consultation.consultation_fee > 0;

    return (
      <TouchableOpacity
        key={consultation.id}
        style={styles.consultationCard}
        onPress={() => {
          setSelectedConsultation(consultation);
          setShowDetailModal(true);
        }}
      >
        <View style={styles.cardHeader}>
          <View>
            <Text style={styles.doctorName}>{consultation.dermatologist_name || 'Dermatologist'}</Text>
            <Text style={styles.consultationType}>{consultation.consultation_type.replace('_', ' ')}</Text>
          </View>
          <View style={[styles.statusBadge, { backgroundColor: getStatusColor(consultation.status) }]}>
            <Text style={styles.statusText}>{consultation.status.toUpperCase()}</Text>
          </View>
        </View>

        <View style={styles.cardDetails}>
          <View style={styles.detailRow}>
            <Ionicons name="calendar" size={16} color="#6b7280" />
            <Text style={styles.detailText}>{formatDate(consultation.scheduled_datetime)}</Text>
          </View>
          <View style={styles.detailRow}>
            <Ionicons name="time" size={16} color="#6b7280" />
            <Text style={styles.detailText}>{formatTime(consultation.scheduled_datetime)}</Text>
          </View>
          <View style={styles.detailRow}>
            <Ionicons name="videocam" size={16} color="#6b7280" />
            <Text style={styles.detailText}>{consultation.video_platform || 'Video'}</Text>
          </View>
        </View>

        {consultation.consultation_reason && (
          <Text style={styles.reasonText} numberOfLines={2}>
            {consultation.consultation_reason}
          </Text>
        )}

        {/* Action buttons */}
        <View style={styles.cardActions}>
          {canJoin && (
            <TouchableOpacity
              style={[styles.actionButton, styles.joinButton]}
              onPress={() => handleJoinVideoCall(consultation)}
            >
              <Ionicons name="videocam" size={18} color="#fff" />
              <Text style={styles.actionButtonText}>Join Call</Text>
            </TouchableOpacity>
          )}

          {needsPayment && (
            <TouchableOpacity
              style={[styles.actionButton, styles.payButton]}
              onPress={() => handlePayment(consultation)}
            >
              <Ionicons name="card" size={18} color="#fff" />
              <Text style={styles.actionButtonText}>Pay ${consultation.consultation_fee}</Text>
            </TouchableOpacity>
          )}

          {isUpcoming && !canJoin && (
            <TouchableOpacity
              style={[styles.actionButton, styles.cancelButton]}
              onPress={() => handleCancelConsultation(consultation)}
            >
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
          )}

          {consultation.status === 'completed' && !consultation.rating && (
            <TouchableOpacity
              style={[styles.actionButton, styles.rateButton]}
              onPress={() => {
                setSelectedConsultation(consultation);
                setShowDetailModal(true);
              }}
            >
              <Ionicons name="star" size={18} color="#f59e0b" />
              <Text style={styles.rateButtonText}>Rate</Text>
            </TouchableOpacity>
          )}
        </View>
      </TouchableOpacity>
    );
  };

  // Render dermatologist card
  const renderDermatologistCard = (derm: Dermatologist) => (
    <TouchableOpacity
      key={derm.id}
      style={[
        styles.dermCard,
        selectedDermatologist?.id === derm.id && styles.dermCardSelected
      ]}
      onPress={() => setSelectedDermatologist(derm)}
    >
      <View style={styles.dermHeader}>
        <View>
          <Text style={styles.dermName}>{derm.full_name}</Text>
          <Text style={styles.dermCredentials}>{derm.credentials}</Text>
        </View>
        {derm.is_verified && (
          <Ionicons name="checkmark-circle" size={24} color="#10b981" />
        )}
      </View>

      <Text style={styles.dermPractice}>{derm.practice_name}</Text>
      <Text style={styles.dermLocation}>{derm.city}, {derm.state}</Text>

      <View style={styles.dermStats}>
        <View style={styles.dermStat}>
          <Ionicons name="star" size={16} color="#f59e0b" />
          <Text style={styles.dermStatText}>{derm.average_rating?.toFixed(1) || 'N/A'}</Text>
        </View>
        <View style={styles.dermStat}>
          <Ionicons name="time" size={16} color="#6b7280" />
          <Text style={styles.dermStatText}>{derm.typical_wait_time_days} days</Text>
        </View>
        <View style={styles.dermStat}>
          <Ionicons name="cash" size={16} color="#10b981" />
          <Text style={styles.dermStatText}>${derm.consultation_fee || 100}</Text>
        </View>
      </View>

      {derm.specializations?.length > 0 && (
        <View style={styles.specializations}>
          {derm.specializations.slice(0, 3).map((spec, i) => (
            <View key={i} style={styles.specTag}>
              <Text style={styles.specText}>{spec}</Text>
            </View>
          ))}
        </View>
      )}

      {selectedDermatologist?.id === derm.id && (
        <TouchableOpacity
          style={styles.selectButton}
          onPress={() => setShowBookingModal(true)}
        >
          <Text style={styles.selectButtonText}>Book Consultation</Text>
        </TouchableOpacity>
      )}
    </TouchableOpacity>
  );

  // Render booking modal
  const renderBookingModal = () => (
    <Modal
      visible={showBookingModal}
      animationType="slide"
      transparent
      onRequestClose={() => setShowBookingModal(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Book Consultation</Text>
            <TouchableOpacity onPress={() => setShowBookingModal(false)}>
              <Ionicons name="close" size={24} color="#333" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.modalScroll}>
            {selectedDermatologist && (
              <View style={styles.selectedDermInfo}>
                <Text style={styles.selectedDermName}>{selectedDermatologist.full_name}</Text>
                <Text style={styles.selectedDermDetail}>{selectedDermatologist.practice_name}</Text>
              </View>
            )}

            <Text style={styles.inputLabel}>Consultation Type</Text>
            <View style={styles.typeButtons}>
              {['initial', 'follow_up', 'urgent'].map(type => (
                <TouchableOpacity
                  key={type}
                  style={[
                    styles.typeButton,
                    bookingType === type && styles.typeButtonActive
                  ]}
                  onPress={() => setBookingType(type as any)}
                >
                  <Text style={[
                    styles.typeButtonText,
                    bookingType === type && styles.typeButtonTextActive
                  ]}>
                    {type.replace('_', ' ')}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.inputLabel}>Reason for Consultation *</Text>
            <TextInput
              style={styles.textInput}
              value={bookingReason}
              onChangeText={setBookingReason}
              placeholder="Describe your concern..."
              multiline
              numberOfLines={3}
            />

            <Text style={styles.inputLabel}>Preferred Date *</Text>
            <TextInput
              style={styles.textInput}
              value={bookingDate}
              onChangeText={setBookingDate}
              placeholder="YYYY-MM-DD"
            />

            <Text style={styles.inputLabel}>Preferred Time *</Text>
            <TextInput
              style={styles.textInput}
              value={bookingTime}
              onChangeText={setBookingTime}
              placeholder="HH:MM (24-hour format)"
            />

            <Text style={styles.inputLabel}>Additional Notes</Text>
            <TextInput
              style={styles.textInput}
              value={bookingNotes}
              onChangeText={setBookingNotes}
              placeholder="Any additional information..."
              multiline
              numberOfLines={2}
            />

            <View style={styles.feeInfo}>
              <Text style={styles.feeLabel}>Consultation Fee:</Text>
              <Text style={styles.feeAmount}>${selectedDermatologist?.consultation_fee || 100}</Text>
            </View>
          </ScrollView>

          <TouchableOpacity
            style={[styles.bookButton, isBooking && styles.bookButtonDisabled]}
            onPress={handleBookConsultation}
            disabled={isBooking}
          >
            {isBooking ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.bookButtonText}>Confirm Booking</Text>
            )}
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  // Render detail modal
  const renderDetailModal = () => {
    if (!selectedConsultation) return null;

    return (
      <Modal
        visible={showDetailModal}
        animationType="slide"
        transparent
        onRequestClose={() => setShowDetailModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Consultation Details</Text>
              <TouchableOpacity onPress={() => setShowDetailModal(false)}>
                <Ionicons name="close" size={24} color="#333" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalScroll}>
              <View style={styles.detailSection}>
                <Text style={styles.detailSectionTitle}>Appointment</Text>
                <Text style={styles.detailItem}>
                  Doctor: {selectedConsultation.dermatologist_name}
                </Text>
                <Text style={styles.detailItem}>
                  Date: {formatDate(selectedConsultation.scheduled_datetime)}
                </Text>
                <Text style={styles.detailItem}>
                  Time: {formatTime(selectedConsultation.scheduled_datetime)}
                </Text>
                <Text style={styles.detailItem}>
                  Duration: {selectedConsultation.duration_minutes} minutes
                </Text>
                <Text style={styles.detailItem}>
                  Platform: {selectedConsultation.video_platform}
                </Text>
              </View>

              {selectedConsultation.consultation_reason && (
                <View style={styles.detailSection}>
                  <Text style={styles.detailSectionTitle}>Reason</Text>
                  <Text style={styles.detailItem}>{selectedConsultation.consultation_reason}</Text>
                </View>
              )}

              {selectedConsultation.diagnosis && (
                <View style={styles.detailSection}>
                  <Text style={styles.detailSectionTitle}>Diagnosis</Text>
                  <Text style={styles.detailItem}>{selectedConsultation.diagnosis}</Text>
                </View>
              )}

              {selectedConsultation.treatment_plan && (
                <View style={styles.detailSection}>
                  <Text style={styles.detailSectionTitle}>Treatment Plan</Text>
                  <Text style={styles.detailItem}>{selectedConsultation.treatment_plan}</Text>
                </View>
              )}

              {selectedConsultation.dermatologist_notes && (
                <View style={styles.detailSection}>
                  <Text style={styles.detailSectionTitle}>Doctor's Notes</Text>
                  <Text style={styles.detailItem}>{selectedConsultation.dermatologist_notes}</Text>
                </View>
              )}

              {selectedConsultation.follow_up_needed && (
                <View style={styles.followUpBadge}>
                  <Ionicons name="calendar" size={18} color="#f59e0b" />
                  <Text style={styles.followUpText}>
                    Follow-up recommended: {selectedConsultation.follow_up_timeframe}
                  </Text>
                </View>
              )}

              {/* Rating section for completed consultations */}
              {selectedConsultation.status === 'completed' && (
                <View style={styles.ratingSection}>
                  <Text style={styles.detailSectionTitle}>Rate Your Experience</Text>
                  <View style={styles.ratingStars}>
                    {[1, 2, 3, 4, 5].map(star => (
                      <TouchableOpacity
                        key={star}
                        onPress={() => handleRateConsultation(selectedConsultation, star)}
                      >
                        <Ionicons
                          name={selectedConsultation.rating && star <= selectedConsultation.rating ? 'star' : 'star-outline'}
                          size={36}
                          color="#f59e0b"
                        />
                      </TouchableOpacity>
                    ))}
                  </View>
                </View>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>
    );
  };

  if (isLoading) {
    return (
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2563eb" />
          <Text style={styles.loadingText}>Loading consultations...</Text>
        </View>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#2563eb" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Consultations</Text>
        <View style={styles.headerSpacer} />
      </View>

      {/* Tabs */}
      <View style={styles.tabs}>
        {(['upcoming', 'past', 'book'] as TabType[]).map(tab => (
          <TouchableOpacity
            key={tab}
            style={[styles.tab, activeTab === tab && styles.tabActive]}
            onPress={() => setActiveTab(tab)}
          >
            <Text style={[styles.tabText, activeTab === tab && styles.tabTextActive]}>
              {tab === 'upcoming' ? 'Upcoming' : tab === 'past' ? 'Past' : 'Book New'}
            </Text>
            {tab === 'upcoming' && upcomingConsultations.length > 0 && (
              <View style={styles.badge}>
                <Text style={styles.badgeText}>{upcomingConsultations.length}</Text>
              </View>
            )}
          </TouchableOpacity>
        ))}
      </View>

      {/* Content */}
      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {activeTab === 'upcoming' && (
          <>
            {upcomingConsultations.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="calendar-outline" size={64} color="#9ca3af" />
                <Text style={styles.emptyTitle}>No Upcoming Consultations</Text>
                <Text style={styles.emptyText}>Book a consultation with a dermatologist to get started</Text>
                <TouchableOpacity
                  style={styles.emptyButton}
                  onPress={() => setActiveTab('book')}
                >
                  <Text style={styles.emptyButtonText}>Book Now</Text>
                </TouchableOpacity>
              </View>
            ) : (
              upcomingConsultations.map(renderConsultationCard)
            )}
          </>
        )}

        {activeTab === 'past' && (
          <>
            {pastConsultations.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="time-outline" size={64} color="#9ca3af" />
                <Text style={styles.emptyTitle}>No Past Consultations</Text>
                <Text style={styles.emptyText}>Your consultation history will appear here</Text>
              </View>
            ) : (
              pastConsultations.map(renderConsultationCard)
            )}
          </>
        )}

        {activeTab === 'book' && (
          <>
            {/* Search filters */}
            <View style={styles.searchSection}>
              <TextInput
                style={styles.searchInput}
                value={searchCity}
                onChangeText={setSearchCity}
                placeholder="City"
                onSubmitEditing={loadDermatologists}
              />
              <TextInput
                style={styles.searchInput}
                value={searchSpecialization}
                onChangeText={setSearchSpecialization}
                placeholder="Specialization"
                onSubmitEditing={loadDermatologists}
              />
              <TouchableOpacity style={styles.searchButton} onPress={loadDermatologists}>
                <Ionicons name="search" size={20} color="#fff" />
              </TouchableOpacity>
            </View>

            {dermatologists.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="people-outline" size={64} color="#9ca3af" />
                <Text style={styles.emptyTitle}>No Dermatologists Found</Text>
                <Text style={styles.emptyText}>Try adjusting your search criteria</Text>
              </View>
            ) : (
              dermatologists.filter(d => d.accepts_video_consultations).map(renderDermatologistCard)
            )}
          </>
        )}

        <View style={styles.bottomSpacer} />
      </ScrollView>

      {renderBookingModal()}
      {renderDetailModal()}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 20,
    paddingBottom: 15,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  headerSpacer: {
    width: 40,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    color: '#6b7280',
    fontSize: 16,
  },
  tabs: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
  },
  tabActive: {
    backgroundColor: '#dbeafe',
  },
  tabText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
  },
  tabTextActive: {
    color: '#2563eb',
  },
  badge: {
    backgroundColor: '#2563eb',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
    marginLeft: 6,
  },
  badgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  consultationCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  doctorName: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  consultationType: {
    fontSize: 14,
    color: '#6b7280',
    textTransform: 'capitalize',
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '700',
  },
  cardDetails: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  detailText: {
    fontSize: 14,
    color: '#6b7280',
  },
  reasonText: {
    fontSize: 14,
    color: '#4b5563',
    marginBottom: 12,
  },
  cardActions: {
    flexDirection: 'row',
    gap: 8,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    gap: 6,
  },
  joinButton: {
    backgroundColor: '#10b981',
    flex: 1,
  },
  payButton: {
    backgroundColor: '#2563eb',
    flex: 1,
  },
  cancelButton: {
    backgroundColor: '#f3f4f6',
    flex: 1,
  },
  rateButton: {
    backgroundColor: '#fef3c7',
    flex: 1,
  },
  actionButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  cancelButtonText: {
    color: '#6b7280',
    fontWeight: '600',
  },
  rateButtonText: {
    color: '#92400e',
    fontWeight: '600',
  },
  searchSection: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
  },
  searchInput: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  searchButton: {
    backgroundColor: '#2563eb',
    borderRadius: 8,
    padding: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  dermCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  dermCardSelected: {
    borderColor: '#2563eb',
  },
  dermHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  dermName: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  dermCredentials: {
    fontSize: 14,
    color: '#6b7280',
  },
  dermPractice: {
    fontSize: 14,
    color: '#4b5563',
    marginTop: 4,
  },
  dermLocation: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 12,
  },
  dermStats: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 12,
  },
  dermStat: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  dermStatText: {
    fontSize: 14,
    color: '#4b5563',
  },
  specializations: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  specTag: {
    backgroundColor: '#e0f2fe',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  specText: {
    fontSize: 12,
    color: '#0369a1',
  },
  selectButton: {
    backgroundColor: '#2563eb',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 12,
  },
  selectButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginTop: 8,
    paddingHorizontal: 40,
  },
  emptyButton: {
    backgroundColor: '#2563eb',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    marginTop: 20,
  },
  emptyButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  bottomSpacer: {
    height: 40,
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    maxHeight: '90%',
    paddingBottom: Platform.OS === 'ios' ? 40 : 20,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  modalScroll: {
    padding: 20,
  },
  selectedDermInfo: {
    backgroundColor: '#f0f9ff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 20,
  },
  selectedDermName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  selectedDermDetail: {
    fontSize: 14,
    color: '#6b7280',
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
    marginTop: 12,
  },
  typeButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  typeButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
  },
  typeButtonActive: {
    backgroundColor: '#2563eb',
  },
  typeButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
    textTransform: 'capitalize',
  },
  typeButtonTextActive: {
    color: '#fff',
  },
  textInput: {
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    fontSize: 16,
  },
  feeInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#f0fdf4',
    padding: 16,
    borderRadius: 8,
    marginTop: 20,
  },
  feeLabel: {
    fontSize: 16,
    color: '#166534',
  },
  feeAmount: {
    fontSize: 24,
    fontWeight: '700',
    color: '#166534',
  },
  bookButton: {
    backgroundColor: '#2563eb',
    marginHorizontal: 20,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 16,
  },
  bookButtonDisabled: {
    backgroundColor: '#93c5fd',
  },
  bookButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  detailSection: {
    marginBottom: 20,
  },
  detailSectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 8,
  },
  detailItem: {
    fontSize: 14,
    color: '#4b5563',
    marginBottom: 4,
  },
  followUpBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#fef3c7',
    padding: 12,
    borderRadius: 8,
    marginTop: 12,
  },
  followUpText: {
    fontSize: 14,
    color: '#92400e',
  },
  ratingSection: {
    alignItems: 'center',
    marginTop: 20,
    padding: 16,
    backgroundColor: '#f9fafb',
    borderRadius: 12,
  },
  ratingStars: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 12,
  },
});
