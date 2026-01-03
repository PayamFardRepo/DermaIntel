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
import { router, useLocalSearchParams } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import { API_BASE_URL } from '../config';
import DoctorSearchService from '../services/DoctorSearchService';

interface Dermatologist {
  id?: number;
  placeId?: string;
  name: string;
  full_name?: string;
  credentials?: string;
  practice_name?: string;
  address?: string;
  city?: string;
  state?: string;
  specializations?: string[];
  accepts_video_consultations?: boolean;
  accepts_referrals?: boolean;
  accepts_second_opinions?: boolean;
  availability_status?: string;
  typical_wait_time_days?: number;
  average_rating?: number;
  rating?: number | string;
  userRatingsTotal?: number;
  years_experience?: number;
  is_verified?: boolean;
  distance?: number;
  isOpen?: boolean | null;
  phone?: string;
  location?: { lat: number; lng: number };
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
  const params = useLocalSearchParams<{ tab?: string; action?: string }>();

  // Initialize tab based on route params
  const initialTab = params.tab === 'consultations' ? 'consultations' :
                     params.tab === 'referrals' ? 'referrals' :
                     params.tab === 'second-opinions' ? 'second-opinions' : 'directory';

  const [activeTab, setActiveTab] = useState<'directory' | 'consultations' | 'referrals' | 'second-opinions'>(initialTab);
  const [isLoading, setIsLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // Directory state
  const [dermatologists, setDermatologists] = useState<Dermatologist[]>([]);
  const [selectedDermatologist, setSelectedDermatologist] = useState<Dermatologist | null>(null);
  const [minRating, setMinRating] = useState<number>(0); // 0 = no filter, 1-5 = minimum rating
  const [searchRadius, setSearchRadius] = useState<number>(10); // miles
  const [sortBy, setSortBy] = useState<'distance' | 'rating'>('distance');
  const [specialistType, setSpecialistType] = useState<'dermatologist' | 'oncologist' | 'both'>('dermatologist');
  const [hasSearched, setHasSearched] = useState(false); // Track if user has initiated a search

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
    // Don't auto-load dermatologists - wait for user to configure filters and search
    if (activeTab === 'directory') {
      // Reset search state when switching to directory tab
      setHasSearched(false);
      setDermatologists([]);
    } else if (activeTab === 'consultations') {
      loadConsultations();
    } else if (activeTab === 'referrals') {
      loadReferrals();
    } else if (activeTab === 'second-opinions') {
      loadSecondOpinions();
    }
  }, [activeTab]);

  // Handle action=request param to show booking form automatically
  useEffect(() => {
    if (params.action === 'request' && params.tab === 'consultations') {
      setShowBookingForm(true);
    }
  }, [params.action, params.tab]);

  const loadDermatologists = async () => {
    try {
      setIsLoading(true);
      setHasSearched(true);

      // Use Google Places API to search for specialists
      const location = await DoctorSearchService.getCurrentLocation();
      // Convert miles to meters (1 mile = 1609.34 meters)
      const radiusMeters = searchRadius * 1609.34;

      let allResults: any[] = [];

      // Search based on specialist type
      if (specialistType === 'dermatologist' || specialistType === 'both') {
        const dermResults = await DoctorSearchService.searchNearbyDoctors(
          'Dermatologist',
          location.latitude,
          location.longitude,
          radiusMeters
        );
        allResults = [...allResults, ...dermResults.map((doc: any) => ({ ...doc, specialistType: 'Dermatologist' }))];
      }

      if (specialistType === 'oncologist' || specialistType === 'both') {
        const oncoResults = await DoctorSearchService.searchNearbyDoctors(
          'Oncologist',
          location.latitude,
          location.longitude,
          radiusMeters
        );
        allResults = [...allResults, ...oncoResults.map((doc: any) => ({ ...doc, specialistType: 'Oncologist' }))];
      }

      // Remove duplicates based on placeId
      const uniqueResults = allResults.filter((doc, index, self) =>
        index === self.findIndex((d) => d.placeId === doc.placeId)
      );

      // Map results to our interface
      let mappedResults: Dermatologist[] = uniqueResults.map((doc: any, index: number) => ({
        id: index + 1,
        placeId: doc.placeId,
        name: doc.name,
        full_name: doc.name,
        address: doc.address,
        rating: doc.rating,
        average_rating: typeof doc.rating === 'number' ? doc.rating : undefined,
        userRatingsTotal: doc.userRatingsTotal,
        distance: doc.distance,
        isOpen: doc.isOpen,
        phone: doc.phone,
        location: doc.location,
        specializations: [doc.specialistType],
        // These are assumed capabilities for specialists found via search
        accepts_video_consultations: true,
        accepts_referrals: true,
        accepts_second_opinions: true,
        is_verified: doc.userRatingsTotal > 50, // Consider verified if many reviews
      }));

      // Apply minimum rating filter
      if (minRating > 0) {
        mappedResults = mappedResults.filter(derm => {
          const rating = typeof derm.rating === 'number' ? derm.rating : (derm.average_rating || 0);
          return rating >= minRating;
        });
      }

      // Sort results
      if (sortBy === 'rating') {
        mappedResults.sort((a, b) => {
          const ratingA = typeof a.rating === 'number' ? a.rating : (a.average_rating || 0);
          const ratingB = typeof b.rating === 'number' ? b.rating : (b.average_rating || 0);
          return ratingB - ratingA; // Descending (highest rating first)
        });
      } else {
        // Sort by distance (ascending)
        mappedResults.sort((a, b) => (a.distance || 999) - (b.distance || 999));
      }

      setDermatologists(mappedResults);
    } catch (error: any) {
      console.log('Error loading specialists:', error);
      if (error.message?.includes('permission')) {
        Alert.alert(
          'Location Required',
          'Please enable location access to find specialists near you.',
          [{ text: 'OK' }]
        );
      }
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
    if (!consultationReason.trim() || !scheduledDate || !scheduledTime) {
      Alert.alert(t('dermatologist.consultations.requiredFields'), t('dermatologist.consultations.fillAllFields'));
      return;
    }

    try {
      const datetime = `${scheduledDate}T${scheduledTime}:00`;
      const formData = new FormData();
      // Dermatologist is now optional - system can assign one
      if (selectedDermatologist?.id) {
        formData.append('dermatologist_id', selectedDermatologist.id.toString());
      }
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
        const successMessage = data.dermatologist_name
          ? t('dermatologist.consultations.bookingSuccess', { name: data.dermatologist_name })
          : 'Consultation request submitted successfully! A dermatologist will be assigned.';
        Alert.alert(t('dermatologist.common.success'), successMessage);
        setShowBookingForm(false);
        setConsultationReason('');
        setScheduledDate('');
        setScheduledTime('');
        setPatientNotes('');
        loadConsultations(); // Refresh the list
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
        <Text style={styles.sectionTitle}>Find Specialists Near You</Text>
        <Text style={styles.searchDescription}>
          Configure your search preferences below, then tap "Search" to find specialists in your area.
        </Text>

        {/* Specialist Type Filter */}
        <View style={styles.filterRow}>
          <Text style={styles.filterLabel}>Specialist Type:</Text>
          <View style={styles.specialistOptions}>
            <Pressable
              style={[
                styles.specialistOption,
                specialistType === 'dermatologist' && styles.specialistOptionActive
              ]}
              onPress={() => setSpecialistType('dermatologist')}
            >
              <Text style={styles.specialistOptionIcon}>🩺</Text>
              <Text style={[
                styles.specialistOptionText,
                specialistType === 'dermatologist' && styles.specialistOptionTextActive
              ]}>
                Dermatologist
              </Text>
            </Pressable>
            <Pressable
              style={[
                styles.specialistOption,
                specialistType === 'oncologist' && styles.specialistOptionActive
              ]}
              onPress={() => setSpecialistType('oncologist')}
            >
              <Text style={styles.specialistOptionIcon}>🔬</Text>
              <Text style={[
                styles.specialistOptionText,
                specialistType === 'oncologist' && styles.specialistOptionTextActive
              ]}>
                Oncologist
              </Text>
            </Pressable>
            <Pressable
              style={[
                styles.specialistOption,
                specialistType === 'both' && styles.specialistOptionActive
              ]}
              onPress={() => setSpecialistType('both')}
            >
              <Text style={styles.specialistOptionIcon}>👥</Text>
              <Text style={[
                styles.specialistOptionText,
                specialistType === 'both' && styles.specialistOptionTextActive
              ]}>
                Both
              </Text>
            </Pressable>
          </View>
        </View>

        {/* Search Radius Filter */}
        <View style={styles.filterRow}>
          <Text style={styles.filterLabel}>Search Radius:</Text>
          <View style={styles.radiusOptions}>
            {[5, 10, 25, 50].map((radius) => (
              <Pressable
                key={radius}
                style={[
                  styles.radiusOption,
                  searchRadius === radius && styles.radiusOptionActive
                ]}
                onPress={() => setSearchRadius(radius)}
              >
                <Text style={[
                  styles.radiusOptionText,
                  searchRadius === radius && styles.radiusOptionTextActive
                ]}>
                  {radius} mi
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        {/* Minimum Rating Filter */}
        <View style={styles.filterRow}>
          <Text style={styles.filterLabel}>Minimum Rating:</Text>
          <View style={styles.ratingOptions}>
            <Pressable
              style={[
                styles.ratingOption,
                minRating === 0 && styles.ratingOptionActive
              ]}
              onPress={() => setMinRating(0)}
            >
              <Text style={[
                styles.ratingOptionText,
                minRating === 0 && styles.ratingOptionTextActive
              ]}>
                Any
              </Text>
            </Pressable>
            {[3, 3.5, 4, 4.5].map((rating) => (
              <Pressable
                key={rating}
                style={[
                  styles.ratingOption,
                  minRating === rating && styles.ratingOptionActive
                ]}
                onPress={() => setMinRating(rating)}
              >
                <Text style={[
                  styles.ratingOptionText,
                  minRating === rating && styles.ratingOptionTextActive
                ]}>
                  ⭐ {rating}+
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        {/* Sort By */}
        <View style={styles.filterRow}>
          <Text style={styles.filterLabel}>Sort By:</Text>
          <View style={styles.sortOptions}>
            <Pressable
              style={[
                styles.sortOption,
                sortBy === 'distance' && styles.sortOptionActive
              ]}
              onPress={() => setSortBy('distance')}
            >
              <Text style={[
                styles.sortOptionText,
                sortBy === 'distance' && styles.sortOptionTextActive
              ]}>
                📍 Nearest
              </Text>
            </Pressable>
            <Pressable
              style={[
                styles.sortOption,
                sortBy === 'rating' && styles.sortOptionActive
              ]}
              onPress={() => setSortBy('rating')}
            >
              <Text style={[
                styles.sortOptionText,
                sortBy === 'rating' && styles.sortOptionTextActive
              ]}>
                ⭐ Highest Rated
              </Text>
            </Pressable>
          </View>
        </View>

        <Pressable
          style={styles.searchButton}
          onPress={loadDermatologists}
          disabled={isLoading}
        >
          <Text style={styles.searchButtonText}>
            {isLoading ? 'Searching...' : '🔍 Search Specialists'}
          </Text>
        </Pressable>
      </View>

      {/* Results - Only show after user has searched */}
      {isLoading ? (
        <ActivityIndicator size="large" color="#0284c7" style={{ marginTop: 20 }} />
      ) : !hasSearched ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateIcon}>🔍</Text>
          <Text style={styles.emptyStateText}>Configure Your Search</Text>
          <Text style={styles.emptyStateSubtext}>
            Select your preferred specialist type, search radius, and minimum rating above, then tap "Search Specialists" to find providers near you.
          </Text>
        </View>
      ) : dermatologists.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateIcon}>😕</Text>
          <Text style={styles.emptyStateText}>No Results Found</Text>
          <Text style={styles.emptyStateSubtext}>
            Try expanding your search radius or lowering the minimum rating requirement.
          </Text>
        </View>
      ) : (
        <ScrollView style={styles.resultsList}>
          {dermatologists.map((derm) => (
            <Pressable
              key={derm.placeId || derm.id}
              style={[
                styles.dermatologistCard,
                selectedDermatologist?.placeId === derm.placeId && styles.dermatologistCardSelected
              ]}
              onPress={() => setSelectedDermatologist(derm)}
            >
              <View style={styles.dermHeader}>
                <View style={styles.dermNameRow}>
                  <Text style={styles.dermName}>{derm.name || derm.full_name}</Text>
                  {derm.is_verified && <Text style={styles.verifiedBadge}>{t('dermatologist.directory.verified')}</Text>}
                </View>
                {derm.specializations && derm.specializations.length > 0 && (
                  <View style={styles.specialistBadge}>
                    <Text style={styles.specialistBadgeText}>
                      {derm.specializations[0] === 'Oncologist' ? '🔬' : '🩺'} {derm.specializations[0]}
                    </Text>
                  </View>
                )}
              </View>

              {/* Address and distance */}
              <Text style={styles.dermLocation}>{derm.address}</Text>

              {/* Distance and open status */}
              <View style={styles.distanceRow}>
                {derm.distance !== undefined && (
                  <Text style={styles.distanceText}>📍 {derm.distance} miles away</Text>
                )}
                {derm.isOpen !== null && (
                  <Text style={[styles.openStatus, derm.isOpen ? styles.openNow : styles.closedNow]}>
                    {derm.isOpen ? '● Open Now' : '● Closed'}
                  </Text>
                )}
              </View>

              {/* Rating and reviews */}
              {(derm.rating || derm.average_rating) && (
                <View style={styles.ratingRow}>
                  <Text style={styles.rating}>
                    ⭐ {typeof derm.rating === 'number' ? derm.rating.toFixed(1) : derm.average_rating?.toFixed(1)} / 5.0
                  </Text>
                  {derm.userRatingsTotal !== undefined && derm.userRatingsTotal > 0 && (
                    <Text style={styles.reviewCount}>({derm.userRatingsTotal} reviews)</Text>
                  )}
                </View>
              )}

              {selectedDermatologist?.placeId === derm.placeId && (
                <View style={styles.actionButtons}>
                  {/* Call button */}
                  {derm.phone && (
                    <Pressable
                      style={styles.actionButton}
                      onPress={() => DoctorSearchService.callPhone(derm.phone!)}
                    >
                      <Text style={styles.actionButtonText}>📞 Call Office</Text>
                    </Pressable>
                  )}

                  {/* Directions button */}
                  {derm.location && (
                    <Pressable
                      style={[styles.actionButton, styles.actionButtonSecondary]}
                      onPress={() => DoctorSearchService.openInMaps(
                        derm.location!.lat,
                        derm.location!.lng,
                        derm.name || derm.full_name || 'Dermatologist'
                      )}
                    >
                      <Text style={[styles.actionButtonText, styles.actionButtonTextSecondary]}>🗺️ Get Directions</Text>
                    </Pressable>
                  )}

                  {/* Get more details */}
                  {derm.placeId && (
                    <Pressable
                      style={[styles.actionButton, styles.actionButtonSecondary]}
                      onPress={async () => {
                        const details = await DoctorSearchService.getDoctorDetails(derm.placeId!);
                        if (details) {
                          Alert.alert(
                            details.name,
                            `📍 ${details.address}\n📞 ${details.phone || 'Not available'}\n🌐 ${details.website || 'No website'}\n\n${details.openingHours?.join('\n') || 'Hours not available'}`,
                            [{ text: 'OK' }]
                          );
                        }
                      }}
                    >
                      <Text style={[styles.actionButtonText, styles.actionButtonTextSecondary]}>ℹ️ More Info</Text>
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
          {selectedDermatologist ? (
            <View style={styles.selectedDermInfo}>
              <Text style={styles.selectedDermName}>{selectedDermatologist.full_name}</Text>
              <Text style={styles.selectedDermDetails}>{selectedDermatologist.practice_name}</Text>
            </View>
          ) : (
            <View style={styles.noSelectedDermInfo}>
              <Text style={styles.noSelectedDermText}>
                No specific dermatologist selected. A dermatologist will be assigned to your consultation.
              </Text>
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
          {/* Request Consultation Button */}
          <Pressable
            style={styles.addButton}
            onPress={() => {
              setShowBookingForm(true);
            }}
          >
            <Text style={styles.addButtonText}>Request Consultation</Text>
          </Pressable>

          {consultations.length === 0 ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyStateText}>{t('dermatologist.consultations.noConsultations')}</Text>
              <Text style={styles.emptyStateSubtext}>Tap "Request Consultation" above to schedule your first consultation with a dermatologist.</Text>
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
    marginBottom: 8,
  },
  searchDescription: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 16,
    lineHeight: 20,
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
  filterRow: {
    marginBottom: 16,
  },
  filterLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  radiusOptions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  radiusOption: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#d1d5db',
    backgroundColor: '#fff',
  },
  radiusOptionActive: {
    backgroundColor: '#0284c7',
    borderColor: '#0284c7',
  },
  radiusOptionText: {
    fontSize: 14,
    color: '#4b5563',
  },
  radiusOptionTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  ratingOptions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  ratingOption: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#d1d5db',
    backgroundColor: '#fff',
  },
  ratingOptionActive: {
    backgroundColor: '#f59e0b',
    borderColor: '#f59e0b',
  },
  ratingOptionText: {
    fontSize: 14,
    color: '#4b5563',
  },
  ratingOptionTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  sortOptions: {
    flexDirection: 'row',
    gap: 8,
  },
  sortOption: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#d1d5db',
    backgroundColor: '#fff',
    alignItems: 'center',
  },
  sortOptionActive: {
    backgroundColor: '#10b981',
    borderColor: '#10b981',
  },
  sortOptionText: {
    fontSize: 14,
    color: '#4b5563',
  },
  sortOptionTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  specialistOptions: {
    flexDirection: 'row',
    gap: 8,
  },
  specialistOption: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#d1d5db',
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  specialistOptionActive: {
    backgroundColor: '#7c3aed',
    borderColor: '#7c3aed',
  },
  specialistOptionIcon: {
    fontSize: 24,
    marginBottom: 4,
  },
  specialistOptionText: {
    fontSize: 12,
    color: '#4b5563',
    textAlign: 'center',
  },
  specialistOptionTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  emptyStateIcon: {
    fontSize: 48,
    marginBottom: 16,
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
    marginBottom: 8,
  },
  dermNameRow: {
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
  specialistBadge: {
    backgroundColor: '#ede9fe',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    alignSelf: 'flex-start',
  },
  specialistBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#7c3aed',
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
    marginBottom: 6,
  },
  distanceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    gap: 12,
  },
  distanceText: {
    fontSize: 13,
    color: '#0284c7',
    fontWeight: '500',
  },
  openStatus: {
    fontSize: 12,
    fontWeight: '600',
  },
  openNow: {
    color: '#10b981',
  },
  closedNow: {
    color: '#ef4444',
  },
  ratingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  reviewCount: {
    fontSize: 12,
    color: '#6b7280',
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
  noSelectedDermInfo: {
    backgroundColor: '#fef3c7',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  noSelectedDermText: {
    fontSize: 13,
    color: '#92400e',
    lineHeight: 18,
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
