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
  StatusBar
} from 'react-native';
import { router } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';

const API_BASE_URL = 'http://localhost:8000';

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
  accepts_second_opinions: boolean;
  availability_status: string;
  typical_wait_time_days: number;
  average_rating: number;
  years_experience: number;
  is_verified: boolean;
}

interface Consultation {
  id: number;
  dermatologist_name: string;
  consultation_type: string;
  scheduled_datetime: string;
  status: string;
  video_meeting_url?: string;
}

interface Referral {
  id: number;
  dermatologist_name: string;
  referral_reason: string;
  urgency_level: string;
  status: string;
  created_at: string;
}

interface SecondOpinion {
  id: number;
  dermatologist_name?: string;
  original_diagnosis: string;
  status: string;
  urgency: string;
  created_at: string;
}

export default function DermatologistIntegrationScreen() {
  const { user } = useAuth();
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState<'directory' | 'consultations' | 'referrals' | 'second-opinions'>('directory');
  const [isLoading, setIsLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // Directory state
  const [dermatologists, setDermatologists] = useState<Dermatologist[]>([]);
  const [searchCity, setSearchCity] = useState('');
  const [searchState, setSearchState] = useState('');
  const [filterSpecialization, setFilterSpecialization] = useState('');
  const [selectedDermatologist, setSelectedDermatologist] = useState<Dermatologist | null>(null);

  // Consultation state
  const [consultations, setConsultations] = useState<Consultation[]>([]);
  const [showBookingForm, setShowBookingForm] = useState(false);
  const [consultationType, setConsultationType] = useState('initial');
  const [consultationReason, setConsultationReason] = useState('');
  const [scheduledDate, setScheduledDate] = useState('');
  const [scheduledTime, setScheduledTime] = useState('');
  const [patientNotes, setPatientNotes] = useState('');

  // Referral state
  const [referrals, setReferrals] = useState<Referral[]>([]);
  const [showReferralForm, setShowReferralForm] = useState(false);
  const [referralReason, setReferralReason] = useState('suspicious_lesion');
  const [primaryConcern, setPrimaryConcern] = useState('');
  const [urgencyLevel, setUrgencyLevel] = useState('routine');
  const [referringProvider, setReferringProvider] = useState('');

  // Second Opinion state
  const [secondOpinions, setSecondOpinions] = useState<SecondOpinion[]>([]);
  const [showSecondOpinionForm, setShowSecondOpinionForm] = useState(false);
  const [originalDiagnosis, setOriginalDiagnosis] = useState('');
  const [reasonForSecondOpinion, setReasonForSecondOpinion] = useState('uncertainty');
  const [concerns, setConcerns] = useState('');
  const [secondOpinionUrgency, setSecondOpinionUrgency] = useState('routine');

  useEffect(() => {
    if (activeTab === 'directory') {
      loadDermatologists();
    } else if (activeTab === 'consultations') {
      loadConsultations();
    } else if (activeTab === 'referrals') {
      loadReferrals();
    } else if (activeTab === 'second-opinions') {
      loadSecondOpinions();
    }
  }, [activeTab]);

  const loadDermatologists = async () => {
    try {
      setIsLoading(true);
      let url = `${API_BASE_URL}/dermatologists?limit=50`;
      if (searchCity) url += `&city=${encodeURIComponent(searchCity)}`;
      if (searchState) url += `&state=${encodeURIComponent(searchState)}`;
      if (filterSpecialization) url += `&specialization=${encodeURIComponent(filterSpecialization)}`;

      const response = await fetch(url, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setDermatologists(data.dermatologists || []);
      } else {
        console.log('No dermatologists found');
        setDermatologists([]);
      }
    } catch (error) {
      console.log('Error loading dermatologists:', error);
      setDermatologists([]);
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const loadConsultations = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/consultations`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setConsultations(data.consultations || []);
      } else {
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

  const loadReferrals = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/referrals`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setReferrals(data.referrals || []);
      } else {
        setReferrals([]);
      }
    } catch (error) {
      console.log('Error loading referrals:', error);
      setReferrals([]);
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const loadSecondOpinions = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/second-opinions`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setSecondOpinions(data.second_opinions || []);
      } else {
        setSecondOpinions([]);
      }
    } catch (error) {
      console.log('Error loading second opinions:', error);
      setSecondOpinions([]);
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const handleBookConsultation = async () => {
    if (!selectedDermatologist) {
      Alert.alert(t('dermatologist.common.error'), t('dermatologist.consultations.selectDermFirst'));
      return;
    }

    if (!consultationReason.trim() || !scheduledDate || !scheduledTime) {
      Alert.alert(t('dermatologist.consultations.requiredFields'), t('dermatologist.consultations.fillAllFields'));
      return;
    }

    try {
      const datetime = `${scheduledDate}T${scheduledTime}:00`;
      const formData = new FormData();
      formData.append('dermatologist_id', selectedDermatologist.id.toString());
      formData.append('consultation_type', consultationType);
      formData.append('consultation_reason', consultationReason);
      formData.append('scheduled_datetime', datetime);
      formData.append('duration_minutes', '30');
      if (patientNotes) formData.append('patient_notes', patientNotes);

      const response = await fetch(`${API_BASE_URL}/consultations`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user?.token}` },
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(t('dermatologist.common.success'), t('dermatologist.consultations.bookingSuccess', { name: data.dermatologist_name }));
        setShowBookingForm(false);
        setConsultationReason('');
        setScheduledDate('');
        setScheduledTime('');
        setPatientNotes('');
        setActiveTab('consultations');
      } else {
        const error = await response.json();
        Alert.alert(t('dermatologist.common.error'), error.detail || t('dermatologist.consultations.bookingFailed'));
      }
    } catch (error) {
      Alert.alert(t('dermatologist.common.error'), t('dermatologist.common.networkError'));
    }
  };

  const handleCreateReferral = async () => {
    if (!primaryConcern.trim()) {
      Alert.alert(t('dermatologist.common.required'), t('dermatologist.referrals.describeConcern'));
      return;
    }

    try {
      const formData = new FormData();
      formData.append('referral_reason', referralReason);
      formData.append('primary_concern', primaryConcern);
      formData.append('urgency_level', urgencyLevel);
      if (selectedDermatologist) {
        formData.append('dermatologist_id', selectedDermatologist.id.toString());
      }
      if (referringProvider) {
        formData.append('referring_provider_name', referringProvider);
      }

      const response = await fetch(`${API_BASE_URL}/referrals`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user?.token}` },
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(t('dermatologist.common.success'), t('dermatologist.referrals.referralSuccess'));
        setShowReferralForm(false);
        setPrimaryConcern('');
        setReferringProvider('');
        setActiveTab('referrals');
      } else {
        const error = await response.json();
        Alert.alert(t('dermatologist.common.error'), error.detail || t('dermatologist.referrals.referralFailed'));
      }
    } catch (error) {
      Alert.alert(t('dermatologist.common.error'), t('dermatologist.common.networkError'));
    }
  };

  const handleRequestSecondOpinion = async () => {
    if (!originalDiagnosis.trim()) {
      Alert.alert(t('dermatologist.common.required'), t('dermatologist.secondOpinions.enterDiagnosis'));
      return;
    }

    try {
      const formData = new FormData();
      formData.append('original_diagnosis', originalDiagnosis);
      formData.append('reason_for_second_opinion', reasonForSecondOpinion);
      formData.append('urgency', secondOpinionUrgency);
      if (concerns) formData.append('concerns', concerns);
      if (selectedDermatologist) {
        formData.append('dermatologist_id', selectedDermatologist.id.toString());
      }

      const response = await fetch(`${API_BASE_URL}/second-opinions`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user?.token}` },
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(t('dermatologist.common.success'), t('dermatologist.secondOpinions.requestSuccess'));
        setShowSecondOpinionForm(false);
        setOriginalDiagnosis('');
        setConcerns('');
        setActiveTab('second-opinions');
      } else {
        const error = await response.json();
        Alert.alert(t('dermatologist.common.error'), error.detail || t('dermatologist.secondOpinions.requestFailed'));
      }
    } catch (error) {
      Alert.alert(t('dermatologist.common.error'), t('dermatologist.common.networkError'));
    }
  };

  const renderDirectoryTab = () => (
    <View style={styles.tabContent}>
      {/* Search Filters */}
      <View style={styles.filterSection}>
        <Text style={styles.sectionTitle}>{t('dermatologist.directory.findDermatologist')}</Text>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>{t('dermatologist.directory.city')}</Text>
          <TextInput
            style={styles.input}
            placeholder={t('dermatologist.directory.enterCity')}
            value={searchCity}
            onChangeText={setSearchCity}
            placeholderTextColor="#9ca3af"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>{t('dermatologist.directory.state')}</Text>
          <TextInput
            style={styles.input}
            placeholder={t('dermatologist.directory.enterState')}
            value={searchState}
            onChangeText={setSearchState}
            placeholderTextColor="#9ca3af"
          />
        </View>

        <Pressable
          style={styles.searchButton}
          onPress={loadDermatologists}
        >
          <Text style={styles.searchButtonText}>{t('dermatologist.directory.search')}</Text>
        </Pressable>
      </View>

      {/* Results */}
      {isLoading ? (
        <ActivityIndicator size="large" color="#0284c7" style={{ marginTop: 20 }} />
      ) : dermatologists.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateText}>{t('dermatologist.directory.noResults')}</Text>
          <Text style={styles.emptyStateSubtext}>{t('dermatologist.directory.noResultsSubtext')}</Text>
        </View>
      ) : (
        <ScrollView style={styles.resultsList}>
          {dermatologists.map((derm) => (
            <Pressable
              key={derm.id}
              style={[
                styles.dermatologistCard,
                selectedDermatologist?.id === derm.id && styles.dermatologistCardSelected
              ]}
              onPress={() => setSelectedDermatologist(derm)}
            >
              <View style={styles.dermHeader}>
                <Text style={styles.dermName}>{derm.full_name}</Text>
                {derm.is_verified && <Text style={styles.verifiedBadge}>{t('dermatologist.directory.verified')}</Text>}
              </View>
              <Text style={styles.dermCredentials}>{derm.credentials}</Text>
              <Text style={styles.dermPractice}>{derm.practice_name}</Text>
              <Text style={styles.dermLocation}>{derm.city}, {derm.state}</Text>

              {derm.specializations && derm.specializations.length > 0 && (
                <View style={styles.specializationsContainer}>
                  {derm.specializations.map((spec, idx) => (
                    <Text key={idx} style={styles.specializationTag}>{spec}</Text>
                  ))}
                </View>
              )}

              <View style={styles.dermServices}>
                {derm.accepts_video_consultations && (
                  <Text style={styles.serviceTag}>{t('dermatologist.directory.videoTag')}</Text>
                )}
                {derm.accepts_referrals && (
                  <Text style={styles.serviceTag}>{t('dermatologist.directory.referralsTag')}</Text>
                )}
                {derm.accepts_second_opinions && (
                  <Text style={styles.serviceTag}>{t('dermatologist.directory.secondOpinionTag')}</Text>
                )}
              </View>

              {derm.average_rating && (
                <Text style={styles.rating}>⭐ {derm.average_rating.toFixed(1)} / 5.0</Text>
              )}

              {selectedDermatologist?.id === derm.id && (
                <View style={styles.actionButtons}>
                  {derm.accepts_video_consultations && (
                    <Pressable
                      style={styles.actionButton}
                      onPress={() => {
                        setShowBookingForm(true);
                        setActiveTab('consultations');
                      }}
                    >
                      <Text style={styles.actionButtonText}>{t('dermatologist.directory.bookVideoCall')}</Text>
                    </Pressable>
                  )}
                  {derm.accepts_referrals && (
                    <Pressable
                      style={[styles.actionButton, styles.actionButtonSecondary]}
                      onPress={() => {
                        setShowReferralForm(true);
                        setActiveTab('referrals');
                      }}
                    >
                      <Text style={[styles.actionButtonText, styles.actionButtonTextSecondary]}>{t('dermatologist.directory.createReferral')}</Text>
                    </Pressable>
                  )}
                  {derm.accepts_second_opinions && (
                    <Pressable
                      style={[styles.actionButton, styles.actionButtonSecondary]}
                      onPress={() => {
                        setShowSecondOpinionForm(true);
                        setActiveTab('second-opinions');
                      }}
                    >
                      <Text style={[styles.actionButtonText, styles.actionButtonTextSecondary]}>{t('dermatologist.directory.get2ndOpinion')}</Text>
                    </Pressable>
                  )}
                </View>
              )}
            </Pressable>
          ))}
        </ScrollView>
      )}
    </View>
  );

  const renderConsultationsTab = () => (
    <View style={styles.tabContent}>
      {showBookingForm ? (
        <View style={styles.formContainer}>
          <Text style={styles.formTitle}>{t('dermatologist.consultations.bookVideoConsultation')}</Text>
          {selectedDermatologist && (
            <View style={styles.selectedDermInfo}>
              <Text style={styles.selectedDermName}>{selectedDermatologist.full_name}</Text>
              <Text style={styles.selectedDermDetails}>{selectedDermatologist.practice_name}</Text>
            </View>
          )}

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.consultations.consultationType')}</Text>
            <View style={styles.radioGroup}>
              {['initial', 'follow_up', 'urgent'].map((type) => (
                <Pressable
                  key={type}
                  style={[
                    styles.radioButton,
                    consultationType === type && styles.radioButtonSelected
                  ]}
                  onPress={() => setConsultationType(type)}
                >
                  <Text style={[
                    styles.radioButtonText,
                    consultationType === type && styles.radioButtonTextSelected
                  ]}>
                    {type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.consultations.reason')}</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder={t('dermatologist.consultations.reasonPlaceholder')}
              value={consultationReason}
              onChangeText={setConsultationReason}
              multiline
              numberOfLines={4}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.consultations.preferredDate')}</Text>
            <TextInput
              style={styles.input}
              placeholder={t('dermatologist.consultations.datePlaceholder')}
              value={scheduledDate}
              onChangeText={setScheduledDate}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.consultations.preferredTime')}</Text>
            <TextInput
              style={styles.input}
              placeholder={t('dermatologist.consultations.timePlaceholder')}
              value={scheduledTime}
              onChangeText={setScheduledTime}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.consultations.additionalNotes')}</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder={t('dermatologist.consultations.notesPlaceholder')}
              value={patientNotes}
              onChangeText={setPatientNotes}
              multiline
              numberOfLines={3}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.formActions}>
            <Pressable style={styles.submitButton} onPress={handleBookConsultation}>
              <Text style={styles.submitButtonText}>{t('dermatologist.consultations.bookConsultation')}</Text>
            </Pressable>
            <Pressable
              style={styles.cancelButton}
              onPress={() => setShowBookingForm(false)}
            >
              <Text style={styles.cancelButtonText}>{t('dermatologist.common.cancel')}</Text>
            </Pressable>
          </View>
        </View>
      ) : (
        <>
          {consultations.length === 0 ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyStateText}>{t('dermatologist.consultations.noConsultations')}</Text>
              <Text style={styles.emptyStateSubtext}>{t('dermatologist.consultations.browseDirectory')}</Text>
            </View>
          ) : (
            <ScrollView style={styles.resultsList}>
              {consultations.map((consult) => (
                <View key={consult.id} style={styles.consultationCard}>
                  <View style={styles.consultationHeader}>
                    <Text style={styles.consultationDerm}>{consult.dermatologist_name}</Text>
                    <View style={[
                      styles.statusBadge,
                      consult.status === 'completed' && styles.statusBadgeCompleted,
                      consult.status === 'cancelled' && styles.statusBadgeCancelled
                    ]}>
                      <Text style={styles.statusBadgeText}>{consult.status}</Text>
                    </View>
                  </View>
                  <Text style={styles.consultationType}>{consult.consultation_type.replace('_', ' ')}</Text>
                  <Text style={styles.consultationDate}>
                    📅 {new Date(consult.scheduled_datetime).toLocaleString()}
                  </Text>
                  {consult.video_meeting_url && consult.status === 'scheduled' && (
                    <Pressable
                      style={styles.joinButton}
                      onPress={() => {
                        // In a real app, would open video platform
                        Alert.alert(t('dermatologist.consultations.videoLink'), consult.video_meeting_url);
                      }}
                    >
                      <Text style={styles.joinButtonText}>{t('dermatologist.consultations.joinVideoCall')}</Text>
                    </Pressable>
                  )}
                </View>
              ))}
            </ScrollView>
          )}
        </>
      )}
    </View>
  );

  const renderReferralsTab = () => (
    <View style={styles.tabContent}>
      {showReferralForm ? (
        <View style={styles.formContainer}>
          <Text style={styles.formTitle}>{t('dermatologist.referrals.createReferral')}</Text>
          {selectedDermatologist && (
            <View style={styles.selectedDermInfo}>
              <Text style={styles.selectedDermName}>{selectedDermatologist.full_name}</Text>
              <Text style={styles.selectedDermDetails}>{selectedDermatologist.practice_name}</Text>
            </View>
          )}

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.referrals.referralReason')}</Text>
            <View style={styles.radioGroup}>
              {[
                { value: 'suspicious_lesion', label: t('dermatologist.referrals.suspiciousLesion') },
                { value: 'skin_cancer_screening', label: t('dermatologist.referrals.screening') },
                { value: 'treatment_resistant', label: t('dermatologist.referrals.treatmentResistant') }
              ].map((option) => (
                <Pressable
                  key={option.value}
                  style={[
                    styles.radioButton,
                    referralReason === option.value && styles.radioButtonSelected
                  ]}
                  onPress={() => setReferralReason(option.value)}
                >
                  <Text style={[
                    styles.radioButtonText,
                    referralReason === option.value && styles.radioButtonTextSelected
                  ]}>{option.label}</Text>
                </Pressable>
              ))}
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.referrals.primaryConcern')}</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder={t('dermatologist.referrals.concernPlaceholder')}
              value={primaryConcern}
              onChangeText={setPrimaryConcern}
              multiline
              numberOfLines={4}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.referrals.urgencyLevel')}</Text>
            <View style={styles.radioGroup}>
              {['routine', 'semi_urgent', 'urgent'].map((level) => (
                <Pressable
                  key={level}
                  style={[
                    styles.radioButton,
                    urgencyLevel === level && styles.radioButtonSelected
                  ]}
                  onPress={() => setUrgencyLevel(level)}
                >
                  <Text style={[
                    styles.radioButtonText,
                    urgencyLevel === level && styles.radioButtonTextSelected
                  ]}>
                    {level.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.referrals.referringProvider')}</Text>
            <TextInput
              style={styles.input}
              placeholder={t('dermatologist.referrals.providerPlaceholder')}
              value={referringProvider}
              onChangeText={setReferringProvider}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.formActions}>
            <Pressable style={styles.submitButton} onPress={handleCreateReferral}>
              <Text style={styles.submitButtonText}>{t('dermatologist.referrals.submitReferral')}</Text>
            </Pressable>
            <Pressable
              style={styles.cancelButton}
              onPress={() => setShowReferralForm(false)}
            >
              <Text style={styles.cancelButtonText}>{t('dermatologist.common.cancel')}</Text>
            </Pressable>
          </View>
        </View>
      ) : (
        <>
          <Pressable
            style={styles.addButton}
            onPress={() => {
              setShowReferralForm(true);
            }}
          >
            <Text style={styles.addButtonText}>{t('dermatologist.referrals.createNewReferral')}</Text>
          </Pressable>

          {referrals.length === 0 ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyStateText}>{t('dermatologist.referrals.noReferrals')}</Text>
              <Text style={styles.emptyStateSubtext}>{t('dermatologist.referrals.noReferralsSubtext')}</Text>
            </View>
          ) : (
            <ScrollView style={styles.resultsList}>
              {referrals.map((referral) => (
                <View key={referral.id} style={styles.referralCard}>
                  <View style={styles.referralHeader}>
                    <Text style={styles.referralReason}>{referral.referral_reason.replace(/_/g, ' ')}</Text>
                    <View style={[
                      styles.urgencyBadge,
                      referral.urgency_level === 'urgent' && styles.urgencyBadgeUrgent,
                      referral.urgency_level === 'semi_urgent' && styles.urgencyBadgeSemiUrgent
                    ]}>
                      <Text style={styles.urgencyBadgeText}>{referral.urgency_level}</Text>
                    </View>
                  </View>
                  {referral.dermatologist_name && (
                    <Text style={styles.referralDerm}>To: {referral.dermatologist_name}</Text>
                  )}
                  <Text style={styles.referralStatus}>Status: {referral.status}</Text>
                  <Text style={styles.referralDate}>
                    Created: {new Date(referral.created_at).toLocaleDateString()}
                  </Text>
                </View>
              ))}
            </ScrollView>
          )}
        </>
      )}
    </View>
  );

  const renderSecondOpinionsTab = () => (
    <View style={styles.tabContent}>
      {showSecondOpinionForm ? (
        <View style={styles.formContainer}>
          <Text style={styles.formTitle}>{t('dermatologist.secondOpinions.requestSecondOpinion')}</Text>
          {selectedDermatologist && (
            <View style={styles.selectedDermInfo}>
              <Text style={styles.selectedDermName}>{selectedDermatologist.full_name}</Text>
              <Text style={styles.selectedDermDetails}>{selectedDermatologist.practice_name}</Text>
            </View>
          )}

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.secondOpinions.originalDiagnosis')}</Text>
            <TextInput
              style={styles.input}
              placeholder={t('dermatologist.secondOpinions.diagnosisPlaceholder')}
              value={originalDiagnosis}
              onChangeText={setOriginalDiagnosis}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.secondOpinions.reasonForOpinion')}</Text>
            <View style={styles.radioGroup}>
              {[
                { value: 'uncertainty', label: t('dermatologist.secondOpinions.uncertainty') },
                { value: 'high_risk_diagnosis', label: t('dermatologist.secondOpinions.highRisk') },
                { value: 'peace_of_mind', label: t('dermatologist.secondOpinions.peaceOfMind') }
              ].map((option) => (
                <Pressable
                  key={option.value}
                  style={[
                    styles.radioButton,
                    reasonForSecondOpinion === option.value && styles.radioButtonSelected
                  ]}
                  onPress={() => setReasonForSecondOpinion(option.value)}
                >
                  <Text style={[
                    styles.radioButtonText,
                    reasonForSecondOpinion === option.value && styles.radioButtonTextSelected
                  ]}>{option.label}</Text>
                </Pressable>
              ))}
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.secondOpinions.yourConcerns')}</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder={t('dermatologist.secondOpinions.concernsPlaceholder')}
              value={concerns}
              onChangeText={setConcerns}
              multiline
              numberOfLines={4}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>{t('dermatologist.secondOpinions.urgency')}</Text>
            <View style={styles.radioGroup}>
              {['routine', 'semi_urgent', 'urgent'].map((level) => (
                <Pressable
                  key={level}
                  style={[
                    styles.radioButton,
                    secondOpinionUrgency === level && styles.radioButtonSelected
                  ]}
                  onPress={() => setSecondOpinionUrgency(level)}
                >
                  <Text style={[
                    styles.radioButtonText,
                    secondOpinionUrgency === level && styles.radioButtonTextSelected
                  ]}>
                    {level.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>

          <View style={styles.formActions}>
            <Pressable style={styles.submitButton} onPress={handleRequestSecondOpinion}>
              <Text style={styles.submitButtonText}>{t('dermatologist.secondOpinions.submitRequest')}</Text>
            </Pressable>
            <Pressable
              style={styles.cancelButton}
              onPress={() => setShowSecondOpinionForm(false)}
            >
              <Text style={styles.cancelButtonText}>{t('dermatologist.common.cancel')}</Text>
            </Pressable>
          </View>
        </View>
      ) : (
        <>
          <Pressable
            style={styles.addButton}
            onPress={() => {
              setShowSecondOpinionForm(true);
            }}
          >
            <Text style={styles.addButtonText}>{t('dermatologist.secondOpinions.requestNewOpinion')}</Text>
          </Pressable>

          {secondOpinions.length === 0 ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyStateText}>{t('dermatologist.secondOpinions.noOpinions')}</Text>
              <Text style={styles.emptyStateSubtext}>{t('dermatologist.secondOpinions.noOpinionsSubtext')}</Text>
            </View>
          ) : (
            <ScrollView style={styles.resultsList}>
              {secondOpinions.map((opinion) => (
                <View key={opinion.id} style={styles.opinionCard}>
                  <Text style={styles.opinionDiagnosis}>Original: {opinion.original_diagnosis}</Text>
                  {opinion.dermatologist_name && (
                    <Text style={styles.opinionDerm}>Reviewer: {opinion.dermatologist_name}</Text>
                  )}
                  <View style={styles.opinionStatus}>
                    <View style={[
                      styles.statusBadge,
                      opinion.status === 'completed' && styles.statusBadgeCompleted
                    ]}>
                      <Text style={styles.statusBadgeText}>{opinion.status}</Text>
                    </View>
                    <View style={[
                      styles.urgencyBadge,
                      opinion.urgency === 'urgent' && styles.urgencyBadgeUrgent
                    ]}>
                      <Text style={styles.urgencyBadgeText}>{opinion.urgency}</Text>
                    </View>
                  </View>
                  <Text style={styles.opinionDate}>
                    Submitted: {new Date(opinion.created_at).toLocaleDateString()}
                  </Text>
                </View>
              ))}
            </ScrollView>
          )}
        </>
      )}
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable onPress={() => router.back()} style={styles.backButton}>
          <Text style={styles.backButtonText}>{t('dermatologist.backButton')}</Text>
        </Pressable>
        <Text style={styles.headerTitle}>{t('dermatologist.title')}</Text>
      </View>

      {/* Tabs */}
      <View style={styles.tabBar}>
        <Pressable
          style={[styles.tab, activeTab === 'directory' && styles.tabActive]}
          onPress={() => setActiveTab('directory')}
        >
          <Text style={[styles.tabText, activeTab === 'directory' && styles.tabTextActive]}>
            {t('dermatologist.tabs.directory')}
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tab, activeTab === 'consultations' && styles.tabActive]}
          onPress={() => setActiveTab('consultations')}
        >
          <Text style={[styles.tabText, activeTab === 'consultations' && styles.tabTextActive]}>
            {t('dermatologist.tabs.consultations')}
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tab, activeTab === 'referrals' && styles.tabActive]}
          onPress={() => setActiveTab('referrals')}
        >
          <Text style={[styles.tabText, activeTab === 'referrals' && styles.tabTextActive]}>
            {t('dermatologist.tabs.referrals')}
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tab, activeTab === 'second-opinions' && styles.tabActive]}
          onPress={() => setActiveTab('second-opinions')}
        >
          <Text style={[styles.tabText, activeTab === 'second-opinions' && styles.tabTextActive]}>
            {t('dermatologist.tabs.secondOpinions')}
          </Text>
        </Pressable>
      </View>

      {/* Content */}
      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={() => {
              setRefreshing(true);
              if (activeTab === 'directory') loadDermatologists();
              else if (activeTab === 'consultations') loadConsultations();
              else if (activeTab === 'referrals') loadReferrals();
              else loadSecondOpinions();
            }}
          />
        }
      >
        {activeTab === 'directory' && renderDirectoryTab()}
        {activeTab === 'consultations' && renderConsultationsTab()}
        {activeTab === 'referrals' && renderReferralsTab()}
        {activeTab === 'second-opinions' && renderSecondOpinionsTab()}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: Platform.OS === 'ios' ? 50 : 16,
    paddingBottom: 16,
    paddingHorizontal: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
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
    color: '#0284c7',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  tab: {
    flex: 1,
    paddingVertical: 14,
    alignItems: 'center',
    borderBottomWidth: 2,
    borderBottomColor: 'transparent',
  },
  tabActive: {
    borderBottomColor: '#0284c7',
  },
  tabText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
  },
  tabTextActive: {
    color: '#0284c7',
  },
  content: {
    flex: 1,
  },
  tabContent: {
    padding: 16,
  },
  filterSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 6,
  },
  input: {
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    color: '#1f2937',
    backgroundColor: '#fff',
  },
  textArea: {
    minHeight: 100,
    textAlignVertical: 'top',
  },
  searchButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  searchButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
  },
  emptyStateText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 8,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#9ca3af',
  },
  resultsList: {
    marginTop: 8,
  },
  dermatologistCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 2,
    borderColor: 'transparent',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  dermatologistCardSelected: {
    borderColor: '#0284c7',
  },
  dermHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  dermName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    flex: 1,
  },
  verifiedBadge: {
    fontSize: 12,
    fontWeight: '600',
    color: '#10b981',
    backgroundColor: '#d1fae5',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  dermCredentials: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 4,
  },
  dermPractice: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
    marginBottom: 2,
  },
  dermLocation: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 8,
  },
  specializationsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 8,
  },
  specializationTag: {
    fontSize: 11,
    color: '#0284c7',
    backgroundColor: '#dbeafe',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    marginRight: 6,
    marginBottom: 4,
  },
  dermServices: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 8,
  },
  serviceTag: {
    fontSize: 12,
    color: '#059669',
    backgroundColor: '#d1fae5',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    marginRight: 6,
    marginBottom: 4,
  },
  rating: {
    fontSize: 13,
    color: '#f59e0b',
    fontWeight: '600',
  },
  actionButtons: {
    marginTop: 12,
    gap: 8,
  },
  actionButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  actionButtonSecondary: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#0284c7',
  },
  actionButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  actionButtonTextSecondary: {
    color: '#0284c7',
  },
  formContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  formTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
  },
  selectedDermInfo: {
    backgroundColor: '#dbeafe',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
  },
  selectedDermName: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#1e40af',
    marginBottom: 2,
  },
  selectedDermDetails: {
    fontSize: 13,
    color: '#1e40af',
  },
  radioGroup: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  radioButton: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#d1d5db',
    backgroundColor: '#fff',
  },
  radioButtonSelected: {
    backgroundColor: '#0284c7',
    borderColor: '#0284c7',
  },
  radioButtonText: {
    fontSize: 13,
    fontWeight: '500',
    color: '#374151',
  },
  radioButtonTextSelected: {
    color: '#fff',
  },
  formActions: {
    marginTop: 16,
    gap: 10,
  },
  submitButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  submitButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  cancelButton: {
    backgroundColor: '#f3f4f6',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
  },
  addButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 16,
  },
  addButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#fff',
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
  consultationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  consultationDerm: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    flex: 1,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: '#fef3c7',
  },
  statusBadgeCompleted: {
    backgroundColor: '#d1fae5',
  },
  statusBadgeCancelled: {
    backgroundColor: '#fee2e2',
  },
  statusBadgeText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#78350f',
  },
  consultationType: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 4,
  },
  consultationDate: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 8,
  },
  joinButton: {
    backgroundColor: '#059669',
    paddingVertical: 10,
    borderRadius: 8,
    alignItems: 'center',
  },
  joinButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  referralCard: {
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
  referralHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  referralReason: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#1f2937',
    textTransform: 'capitalize',
    flex: 1,
  },
  urgencyBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: '#d1d5db',
  },
  urgencyBadgeUrgent: {
    backgroundColor: '#fee2e2',
  },
  urgencyBadgeSemiUrgent: {
    backgroundColor: '#fef3c7',
  },
  urgencyBadgeText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#374151',
    textTransform: 'lowercase',
  },
  referralDerm: {
    fontSize: 14,
    color: '#059669',
    marginBottom: 4,
  },
  referralStatus: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 4,
  },
  referralDate: {
    fontSize: 12,
    color: '#9ca3af',
  },
  opinionCard: {
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
  opinionDiagnosis: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 6,
  },
  opinionDerm: {
    fontSize: 14,
    color: '#059669',
    marginBottom: 6,
  },
  opinionStatus: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 6,
  },
  opinionDate: {
    fontSize: 12,
    color: '#9ca3af',
  },
});
