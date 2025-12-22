/**
 * Referrals Management Screen
 *
 * Features:
 * - Create and manage referrals to dermatologists
 * - Browse dermatologist directory with filters
 * - Track referral status through the process
 * - View referral details and outcomes
 * - Schedule appointments from referrals
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Platform,
  RefreshControl,
  Modal,
  TextInput,
  Dimensions,
  Linking,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Referral reasons
const REFERRAL_REASONS = [
  { id: 'suspicious_lesion', label: 'Suspicious Lesion', icon: 'warning-outline' },
  { id: 'skin_cancer_screening', label: 'Skin Cancer Screening', icon: 'search-outline' },
  { id: 'treatment_resistant', label: 'Treatment Resistant Condition', icon: 'medical-outline' },
  { id: 'specialized_care', label: 'Specialized Care Needed', icon: 'person-outline' },
  { id: 'second_opinion', label: 'Second Opinion', icon: 'chatbubbles-outline' },
  { id: 'cosmetic', label: 'Cosmetic Concern', icon: 'sparkles-outline' },
];

// Urgency levels
const URGENCY_LEVELS = [
  { id: 'urgent', label: 'Urgent', description: 'Within 1-2 weeks', color: '#ef4444' },
  { id: 'semi_urgent', label: 'Semi-Urgent', description: 'Within 2-4 weeks', color: '#f59e0b' },
  { id: 'routine', label: 'Routine', description: 'Within 1-3 months', color: '#10b981' },
];

// Status configurations
const STATUS_CONFIG: { [key: string]: { label: string; color: string; icon: string } } = {
  pending: { label: 'Pending', color: '#f59e0b', icon: 'time-outline' },
  accepted: { label: 'Accepted', color: '#3b82f6', icon: 'checkmark-circle-outline' },
  appointment_scheduled: { label: 'Scheduled', color: '#8b5cf6', icon: 'calendar-outline' },
  seen: { label: 'Completed', color: '#10b981', icon: 'checkmark-done-outline' },
  declined: { label: 'Declined', color: '#ef4444', icon: 'close-circle-outline' },
  cancelled: { label: 'Cancelled', color: '#6b7280', icon: 'ban-outline' },
};

interface Referral {
  id: number;
  dermatologist_id: number | null;
  dermatologist_name: string | null;
  referring_provider_name: string | null;
  referral_reason: string;
  primary_concern: string;
  urgency_level: string;
  status: string;
  status_notes: string | null;
  appointment_scheduled_date: string | null;
  created_at: string;
}

interface ReferralDetail extends Referral {
  clinical_summary: string | null;
  analysis_id: number | null;
  lesion_group_id: number | null;
  supporting_documents: any;
  insurance_authorization_required: boolean;
  insurance_authorization_number: string | null;
  insurance_approved: boolean | null;
  dermatologist_accepted: boolean | null;
  dermatologist_response: string | null;
  dermatologist_diagnosis: string | null;
  treatment_provided: string | null;
  outcome_report: string | null;
  appointment_completed_date: string | null;
}

interface Dermatologist {
  id: number;
  full_name: string;
  credentials: string | null;
  practice_name: string | null;
  city: string | null;
  state: string | null;
  specializations: string[];
  languages_spoken: string[];
  accepts_video_consultations: boolean;
  accepts_referrals: boolean;
  availability_status: string | null;
  typical_wait_time_days: number | null;
  average_rating: number | null;
  total_reviews: number | null;
  photo_url: string | null;
  bio: string | null;
  is_verified: boolean;
}

export default function ReferralsScreen() {
  const { t } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // State
  const [activeTab, setActiveTab] = useState<'referrals' | 'directory'>('referrals');
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Referrals state
  const [referrals, setReferrals] = useState<Referral[]>([]);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [selectedReferral, setSelectedReferral] = useState<ReferralDetail | null>(null);

  // Directory state
  const [dermatologists, setDermatologists] = useState<Dermatologist[]>([]);
  const [citySearch, setCitySearch] = useState('');
  const [stateSearch, setStateSearch] = useState('');
  const [specializationFilter, setSpecializationFilter] = useState<string | null>(null);

  // Modal state
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [updateStatusModalVisible, setUpdateStatusModalVisible] = useState(false);
  const [dermatologistDetailModalVisible, setDermatologistDetailModalVisible] = useState(false);
  const [selectedDermatologist, setSelectedDermatologist] = useState<Dermatologist | null>(null);

  // New referral form state
  const [newReferralReason, setNewReferralReason] = useState('suspicious_lesion');
  const [newPrimaryConcern, setNewPrimaryConcern] = useState('');
  const [newClinicalSummary, setNewClinicalSummary] = useState('');
  const [newUrgencyLevel, setNewUrgencyLevel] = useState('routine');
  const [newDermatologistId, setNewDermatologistId] = useState<number | null>(null);
  const [newReferringProvider, setNewReferringProvider] = useState('');

  // Status update state
  const [newStatus, setNewStatus] = useState('');
  const [statusNotes, setStatusNotes] = useState('');
  const [appointmentDate, setAppointmentDate] = useState('');

  const getAuthHeaders = async () => {
    const token = await AsyncStorage.getItem('accessToken');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  };

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
      return;
    }
    loadReferrals();
    loadDermatologists();
  }, [isAuthenticated]);

  const loadReferrals = async () => {
    try {
      const headers = await getAuthHeaders();
      let url = `${API_BASE_URL}/referrals?limit=50`;
      if (statusFilter) {
        url += `&status=${statusFilter}`;
      }

      const response = await fetch(url, { headers });

      if (response.ok) {
        const data = await response.json();
        setReferrals(data.referrals || []);
      }
    } catch (error) {
      console.error('Error loading referrals:', error);
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const loadDermatologists = async () => {
    try {
      const headers = await getAuthHeaders();
      let url = `${API_BASE_URL}/dermatologists?accepts_referrals=true&limit=50`;

      if (citySearch) url += `&city=${encodeURIComponent(citySearch)}`;
      if (stateSearch) url += `&state=${encodeURIComponent(stateSearch)}`;
      if (specializationFilter) url += `&specialization=${encodeURIComponent(specializationFilter)}`;

      const response = await fetch(url, { headers });

      if (response.ok) {
        const data = await response.json();
        setDermatologists(data.dermatologists || []);
      }
    } catch (error) {
      console.error('Error loading dermatologists:', error);
    }
  };

  const loadReferralDetail = async (referralId: number) => {
    try {
      setIsLoading(true);
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/referrals/${referralId}`, { headers });

      if (response.ok) {
        const data = await response.json();
        setSelectedReferral(data);
        setDetailModalVisible(true);
      }
    } catch (error) {
      console.error('Error loading referral detail:', error);
      Alert.alert('Error', 'Failed to load referral details');
    } finally {
      setIsLoading(false);
    }
  };

  const createReferral = async () => {
    if (!newPrimaryConcern.trim()) {
      Alert.alert('Missing Information', 'Please describe your primary concern');
      return;
    }

    try {
      setIsLoading(true);
      const token = await AsyncStorage.getItem('accessToken');
      const formData = new FormData();

      formData.append('referral_reason', newReferralReason);
      formData.append('primary_concern', newPrimaryConcern);
      formData.append('urgency_level', newUrgencyLevel);
      if (newClinicalSummary) formData.append('clinical_summary', newClinicalSummary);
      if (newDermatologistId) formData.append('dermatologist_id', newDermatologistId.toString());
      if (newReferringProvider) formData.append('referring_provider_name', newReferringProvider);

      const response = await fetch(`${API_BASE_URL}/referrals`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        Alert.alert('Success', 'Referral created successfully');
        setCreateModalVisible(false);
        resetNewReferralForm();
        loadReferrals();
      } else {
        throw new Error('Failed to create referral');
      }
    } catch (error) {
      console.error('Error creating referral:', error);
      Alert.alert('Error', 'Failed to create referral');
    } finally {
      setIsLoading(false);
    }
  };

  const updateReferralStatus = async () => {
    if (!selectedReferral || !newStatus) return;

    try {
      setIsLoading(true);
      const token = await AsyncStorage.getItem('accessToken');
      const formData = new FormData();

      formData.append('status', newStatus);
      if (statusNotes) formData.append('status_notes', statusNotes);
      if (appointmentDate) formData.append('appointment_date', appointmentDate);

      const response = await fetch(`${API_BASE_URL}/referrals/${selectedReferral.id}/status`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        Alert.alert('Success', 'Referral status updated');
        setUpdateStatusModalVisible(false);
        setDetailModalVisible(false);
        resetStatusForm();
        loadReferrals();
      } else {
        throw new Error('Failed to update status');
      }
    } catch (error) {
      console.error('Error updating status:', error);
      Alert.alert('Error', 'Failed to update referral status');
    } finally {
      setIsLoading(false);
    }
  };

  const resetNewReferralForm = () => {
    setNewReferralReason('suspicious_lesion');
    setNewPrimaryConcern('');
    setNewClinicalSummary('');
    setNewUrgencyLevel('routine');
    setNewDermatologistId(null);
    setNewReferringProvider('');
  };

  const resetStatusForm = () => {
    setNewStatus('');
    setStatusNotes('');
    setAppointmentDate('');
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadReferrals();
    loadDermatologists();
  }, [statusFilter]);

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const getStatusConfig = (status: string) => {
    return STATUS_CONFIG[status] || STATUS_CONFIG.pending;
  };

  // Render Referrals Tab
  const renderReferralsTab = () => (
    <ScrollView
      style={styles.tabContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Create Referral Button */}
      <TouchableOpacity
        style={styles.createButton}
        onPress={() => setCreateModalVisible(true)}
      >
        <LinearGradient
          colors={['#2563eb', '#1d4ed8']}
          style={styles.createButtonGradient}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
        >
          <Ionicons name="add-circle-outline" size={24} color="#fff" />
          <Text style={styles.createButtonText}>Create New Referral</Text>
        </LinearGradient>
      </TouchableOpacity>

      {/* Status Filter */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterScroll}>
        <View style={styles.filterRow}>
          <TouchableOpacity
            style={[styles.filterChip, !statusFilter && styles.filterChipActive]}
            onPress={() => {
              setStatusFilter(null);
              setIsLoading(true);
              loadReferrals();
            }}
          >
            <Text style={[styles.filterChipText, !statusFilter && styles.filterChipTextActive]}>
              All
            </Text>
          </TouchableOpacity>
          {Object.entries(STATUS_CONFIG).map(([key, config]) => (
            <TouchableOpacity
              key={key}
              style={[styles.filterChip, statusFilter === key && styles.filterChipActive]}
              onPress={() => {
                setStatusFilter(key);
                setIsLoading(true);
                loadReferrals();
              }}
            >
              <Ionicons
                name={config.icon as any}
                size={14}
                color={statusFilter === key ? '#fff' : config.color}
              />
              <Text style={[styles.filterChipText, statusFilter === key && styles.filterChipTextActive]}>
                {config.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>

      {/* Referrals List */}
      {referrals.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="document-text-outline" size={64} color="#9ca3af" />
          <Text style={styles.emptyTitle}>No Referrals</Text>
          <Text style={styles.emptyText}>
            {statusFilter
              ? `No ${STATUS_CONFIG[statusFilter]?.label.toLowerCase()} referrals found`
              : 'Create your first referral to a dermatologist'}
          </Text>
        </View>
      ) : (
        referrals.map(referral => {
          const statusConfig = getStatusConfig(referral.status);
          const urgencyConfig = URGENCY_LEVELS.find(u => u.id === referral.urgency_level);

          return (
            <TouchableOpacity
              key={referral.id}
              style={styles.referralCard}
              onPress={() => loadReferralDetail(referral.id)}
            >
              <View style={styles.referralHeader}>
                <View style={[styles.statusBadge, { backgroundColor: `${statusConfig.color}20` }]}>
                  <Ionicons name={statusConfig.icon as any} size={14} color={statusConfig.color} />
                  <Text style={[styles.statusText, { color: statusConfig.color }]}>
                    {statusConfig.label}
                  </Text>
                </View>
                <Text style={styles.referralDate}>{formatDate(referral.created_at)}</Text>
              </View>

              <Text style={styles.referralReason}>
                {REFERRAL_REASONS.find(r => r.id === referral.referral_reason)?.label || referral.referral_reason}
              </Text>
              <Text style={styles.referralConcern} numberOfLines={2}>
                {referral.primary_concern}
              </Text>

              <View style={styles.referralFooter}>
                {referral.dermatologist_name && (
                  <View style={styles.referralDetail}>
                    <Ionicons name="person-outline" size={14} color="#6b7280" />
                    <Text style={styles.referralDetailText}>{referral.dermatologist_name}</Text>
                  </View>
                )}
                {urgencyConfig && (
                  <View style={[styles.urgencyBadge, { backgroundColor: `${urgencyConfig.color}20` }]}>
                    <Text style={[styles.urgencyText, { color: urgencyConfig.color }]}>
                      {urgencyConfig.label}
                    </Text>
                  </View>
                )}
              </View>

              {referral.appointment_scheduled_date && (
                <View style={styles.appointmentInfo}>
                  <Ionicons name="calendar" size={14} color="#8b5cf6" />
                  <Text style={styles.appointmentText}>
                    Appointment: {formatDate(referral.appointment_scheduled_date)}
                  </Text>
                </View>
              )}
            </TouchableOpacity>
          );
        })
      )}

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Render Directory Tab
  const renderDirectoryTab = () => (
    <ScrollView
      style={styles.tabContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Search Filters */}
      <View style={styles.searchCard}>
        <Text style={styles.searchTitle}>Find a Dermatologist</Text>
        <View style={styles.searchRow}>
          <View style={styles.searchInputContainer}>
            <Ionicons name="location-outline" size={18} color="#6b7280" />
            <TextInput
              style={styles.searchInput}
              placeholder="City"
              value={citySearch}
              onChangeText={setCitySearch}
              placeholderTextColor="#9ca3af"
            />
          </View>
          <View style={styles.searchInputContainer}>
            <Ionicons name="map-outline" size={18} color="#6b7280" />
            <TextInput
              style={styles.searchInput}
              placeholder="State"
              value={stateSearch}
              onChangeText={setStateSearch}
              placeholderTextColor="#9ca3af"
            />
          </View>
        </View>
        <TouchableOpacity
          style={styles.searchButton}
          onPress={() => loadDermatologists()}
        >
          <Ionicons name="search" size={18} color="#fff" />
          <Text style={styles.searchButtonText}>Search</Text>
        </TouchableOpacity>
      </View>

      {/* Specialization Filter */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.specFilterScroll}>
        <View style={styles.specFilterRow}>
          {['Mohs Surgery', 'Pediatric', 'Cosmetic', 'Medical', 'Skin Cancer'].map(spec => (
            <TouchableOpacity
              key={spec}
              style={[
                styles.specFilterChip,
                specializationFilter === spec && styles.specFilterChipActive,
              ]}
              onPress={() => {
                setSpecializationFilter(specializationFilter === spec ? null : spec);
                loadDermatologists();
              }}
            >
              <Text style={[
                styles.specFilterText,
                specializationFilter === spec && styles.specFilterTextActive,
              ]}>
                {spec}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>

      {/* Dermatologists List */}
      <Text style={styles.sectionTitle}>
        {dermatologists.length} Dermatologists Found
      </Text>

      {dermatologists.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="people-outline" size={64} color="#9ca3af" />
          <Text style={styles.emptyTitle}>No Results</Text>
          <Text style={styles.emptyText}>
            Try adjusting your search criteria
          </Text>
        </View>
      ) : (
        dermatologists.map(derm => (
          <TouchableOpacity
            key={derm.id}
            style={styles.dermCard}
            onPress={() => {
              setSelectedDermatologist(derm);
              setDermatologistDetailModalVisible(true);
            }}
          >
            <View style={styles.dermHeader}>
              <View style={styles.dermAvatar}>
                <Ionicons name="person" size={28} color="#2563eb" />
              </View>
              <View style={styles.dermInfo}>
                <View style={styles.dermNameRow}>
                  <Text style={styles.dermName}>{derm.full_name}</Text>
                  {derm.is_verified && (
                    <Ionicons name="checkmark-circle" size={16} color="#10b981" />
                  )}
                </View>
                <Text style={styles.dermCredentials}>{derm.credentials}</Text>
                {derm.practice_name && (
                  <Text style={styles.dermPractice}>{derm.practice_name}</Text>
                )}
              </View>
              {derm.average_rating && (
                <View style={styles.ratingBadge}>
                  <Ionicons name="star" size={14} color="#f59e0b" />
                  <Text style={styles.ratingText}>{derm.average_rating.toFixed(1)}</Text>
                </View>
              )}
            </View>

            <View style={styles.dermDetails}>
              {derm.city && derm.state && (
                <View style={styles.dermDetailItem}>
                  <Ionicons name="location-outline" size={14} color="#6b7280" />
                  <Text style={styles.dermDetailText}>{derm.city}, {derm.state}</Text>
                </View>
              )}
              {derm.typical_wait_time_days && (
                <View style={styles.dermDetailItem}>
                  <Ionicons name="time-outline" size={14} color="#6b7280" />
                  <Text style={styles.dermDetailText}>
                    ~{derm.typical_wait_time_days} days wait
                  </Text>
                </View>
              )}
            </View>

            {derm.specializations.length > 0 && (
              <View style={styles.specsRow}>
                {derm.specializations.slice(0, 3).map((spec, index) => (
                  <View key={index} style={styles.specTag}>
                    <Text style={styles.specTagText}>{spec}</Text>
                  </View>
                ))}
                {derm.specializations.length > 3 && (
                  <Text style={styles.moreSpecs}>+{derm.specializations.length - 3} more</Text>
                )}
              </View>
            )}

            <View style={styles.dermBadges}>
              {derm.accepts_video_consultations && (
                <View style={styles.featureBadge}>
                  <Ionicons name="videocam-outline" size={12} color="#8b5cf6" />
                  <Text style={styles.featureBadgeText}>Video</Text>
                </View>
              )}
              {derm.accepts_referrals && (
                <View style={styles.featureBadge}>
                  <Ionicons name="document-text-outline" size={12} color="#10b981" />
                  <Text style={styles.featureBadgeText}>Referrals</Text>
                </View>
              )}
            </View>

            <TouchableOpacity
              style={styles.selectDermButton}
              onPress={(e) => {
                e.stopPropagation();
                setNewDermatologistId(derm.id);
                setActiveTab('referrals');
                setCreateModalVisible(true);
              }}
            >
              <Ionicons name="send-outline" size={16} color="#2563eb" />
              <Text style={styles.selectDermText}>Create Referral</Text>
            </TouchableOpacity>
          </TouchableOpacity>
        ))
      )}

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Create Referral Modal
  const renderCreateModal = () => (
    <Modal
      visible={createModalVisible}
      animationType="slide"
      onRequestClose={() => setCreateModalVisible(false)}
    >
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setCreateModalVisible(false)}>
            <Ionicons name="close" size={28} color="#1e3a5f" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Create Referral</Text>
          <View style={{ width: 28 }} />
        </View>

        <ScrollView style={styles.modalContent}>
          {/* Referral Reason */}
          <Text style={styles.formLabel}>Reason for Referral *</Text>
          <View style={styles.reasonGrid}>
            {REFERRAL_REASONS.map(reason => (
              <TouchableOpacity
                key={reason.id}
                style={[
                  styles.reasonOption,
                  newReferralReason === reason.id && styles.reasonOptionSelected,
                ]}
                onPress={() => setNewReferralReason(reason.id)}
              >
                <Ionicons
                  name={reason.icon as any}
                  size={24}
                  color={newReferralReason === reason.id ? '#fff' : '#2563eb'}
                />
                <Text style={[
                  styles.reasonText,
                  newReferralReason === reason.id && styles.reasonTextSelected,
                ]}>
                  {reason.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          {/* Primary Concern */}
          <Text style={styles.formLabel}>Primary Concern *</Text>
          <TextInput
            style={[styles.textInput, styles.textAreaInput]}
            placeholder="Describe your main concern in detail..."
            value={newPrimaryConcern}
            onChangeText={setNewPrimaryConcern}
            multiline
            numberOfLines={4}
            placeholderTextColor="#9ca3af"
          />

          {/* Clinical Summary */}
          <Text style={styles.formLabel}>Clinical Summary (Optional)</Text>
          <TextInput
            style={[styles.textInput, styles.textAreaInput]}
            placeholder="Additional medical history or context..."
            value={newClinicalSummary}
            onChangeText={setNewClinicalSummary}
            multiline
            numberOfLines={3}
            placeholderTextColor="#9ca3af"
          />

          {/* Urgency Level */}
          <Text style={styles.formLabel}>Urgency Level</Text>
          <View style={styles.urgencyOptions}>
            {URGENCY_LEVELS.map(level => (
              <TouchableOpacity
                key={level.id}
                style={[
                  styles.urgencyOption,
                  newUrgencyLevel === level.id && { borderColor: level.color, backgroundColor: `${level.color}10` },
                ]}
                onPress={() => setNewUrgencyLevel(level.id)}
              >
                <View style={[styles.urgencyDot, { backgroundColor: level.color }]} />
                <View>
                  <Text style={[
                    styles.urgencyLabel,
                    newUrgencyLevel === level.id && { color: level.color },
                  ]}>
                    {level.label}
                  </Text>
                  <Text style={styles.urgencyDesc}>{level.description}</Text>
                </View>
              </TouchableOpacity>
            ))}
          </View>

          {/* Referring Provider */}
          <Text style={styles.formLabel}>Referring Provider (Optional)</Text>
          <TextInput
            style={styles.textInput}
            placeholder="Dr. Name or self-referral"
            value={newReferringProvider}
            onChangeText={setNewReferringProvider}
            placeholderTextColor="#9ca3af"
          />

          {/* Selected Dermatologist */}
          {newDermatologistId && (
            <View style={styles.selectedDermCard}>
              <Ionicons name="person-circle-outline" size={24} color="#2563eb" />
              <Text style={styles.selectedDermText}>
                Dermatologist ID: {newDermatologistId}
              </Text>
              <TouchableOpacity onPress={() => setNewDermatologistId(null)}>
                <Ionicons name="close-circle" size={20} color="#ef4444" />
              </TouchableOpacity>
            </View>
          )}

          {!newDermatologistId && (
            <TouchableOpacity
              style={styles.browseDermButton}
              onPress={() => {
                setCreateModalVisible(false);
                setActiveTab('directory');
              }}
            >
              <Ionicons name="search-outline" size={20} color="#2563eb" />
              <Text style={styles.browseDermText}>Browse Dermatologists</Text>
            </TouchableOpacity>
          )}

          {/* Submit Button */}
          <TouchableOpacity
            style={styles.submitButton}
            onPress={createReferral}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <>
                <Ionicons name="send" size={20} color="#fff" />
                <Text style={styles.submitButtonText}>Submit Referral</Text>
              </>
            )}
          </TouchableOpacity>

          <View style={styles.bottomSpacer} />
        </ScrollView>
      </LinearGradient>
    </Modal>
  );

  // Referral Detail Modal
  const renderDetailModal = () => (
    <Modal
      visible={detailModalVisible}
      animationType="slide"
      onRequestClose={() => setDetailModalVisible(false)}
    >
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setDetailModalVisible(false)}>
            <Ionicons name="arrow-back" size={28} color="#1e3a5f" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Referral Details</Text>
          <TouchableOpacity
            onPress={() => {
              setNewStatus(selectedReferral?.status || '');
              setUpdateStatusModalVisible(true);
            }}
          >
            <Ionicons name="create-outline" size={24} color="#2563eb" />
          </TouchableOpacity>
        </View>

        {selectedReferral && (
          <ScrollView style={styles.modalContent}>
            {/* Status Card */}
            <View style={styles.detailStatusCard}>
              <View style={[
                styles.detailStatusBadge,
                { backgroundColor: `${getStatusConfig(selectedReferral.status).color}20` }
              ]}>
                <Ionicons
                  name={getStatusConfig(selectedReferral.status).icon as any}
                  size={24}
                  color={getStatusConfig(selectedReferral.status).color}
                />
                <Text style={[
                  styles.detailStatusText,
                  { color: getStatusConfig(selectedReferral.status).color }
                ]}>
                  {getStatusConfig(selectedReferral.status).label}
                </Text>
              </View>
              <Text style={styles.detailDate}>Created {formatDate(selectedReferral.created_at)}</Text>
            </View>

            {/* Reason & Concern */}
            <View style={styles.detailCard}>
              <Text style={styles.detailCardTitle}>Referral Information</Text>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Reason:</Text>
                <Text style={styles.detailValue}>
                  {REFERRAL_REASONS.find(r => r.id === selectedReferral.referral_reason)?.label}
                </Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Urgency:</Text>
                <Text style={[
                  styles.detailValue,
                  { color: URGENCY_LEVELS.find(u => u.id === selectedReferral.urgency_level)?.color }
                ]}>
                  {URGENCY_LEVELS.find(u => u.id === selectedReferral.urgency_level)?.label}
                </Text>
              </View>
              <Text style={styles.detailLabel}>Primary Concern:</Text>
              <Text style={styles.detailParagraph}>{selectedReferral.primary_concern}</Text>
              {selectedReferral.clinical_summary && (
                <>
                  <Text style={styles.detailLabel}>Clinical Summary:</Text>
                  <Text style={styles.detailParagraph}>{selectedReferral.clinical_summary}</Text>
                </>
              )}
            </View>

            {/* Appointment Info */}
            {selectedReferral.appointment_scheduled_date && (
              <View style={styles.detailCard}>
                <Text style={styles.detailCardTitle}>Appointment</Text>
                <View style={styles.appointmentDetailRow}>
                  <Ionicons name="calendar" size={20} color="#8b5cf6" />
                  <Text style={styles.appointmentDetailText}>
                    Scheduled: {formatDate(selectedReferral.appointment_scheduled_date)}
                  </Text>
                </View>
                {selectedReferral.appointment_completed_date && (
                  <View style={styles.appointmentDetailRow}>
                    <Ionicons name="checkmark-circle" size={20} color="#10b981" />
                    <Text style={styles.appointmentDetailText}>
                      Completed: {formatDate(selectedReferral.appointment_completed_date)}
                    </Text>
                  </View>
                )}
              </View>
            )}

            {/* Insurance Info */}
            {selectedReferral.insurance_authorization_required && (
              <View style={styles.detailCard}>
                <Text style={styles.detailCardTitle}>Insurance Authorization</Text>
                <View style={styles.detailRow}>
                  <Text style={styles.detailLabel}>Required:</Text>
                  <Text style={styles.detailValue}>Yes</Text>
                </View>
                {selectedReferral.insurance_authorization_number && (
                  <View style={styles.detailRow}>
                    <Text style={styles.detailLabel}>Auth #:</Text>
                    <Text style={styles.detailValue}>
                      {selectedReferral.insurance_authorization_number}
                    </Text>
                  </View>
                )}
                <View style={styles.detailRow}>
                  <Text style={styles.detailLabel}>Status:</Text>
                  <Text style={[
                    styles.detailValue,
                    { color: selectedReferral.insurance_approved ? '#10b981' : '#f59e0b' }
                  ]}>
                    {selectedReferral.insurance_approved ? 'Approved' : 'Pending'}
                  </Text>
                </View>
              </View>
            )}

            {/* Outcome */}
            {selectedReferral.dermatologist_diagnosis && (
              <View style={styles.detailCard}>
                <Text style={styles.detailCardTitle}>Outcome</Text>
                {selectedReferral.dermatologist_diagnosis && (
                  <>
                    <Text style={styles.detailLabel}>Diagnosis:</Text>
                    <Text style={styles.detailParagraph}>{selectedReferral.dermatologist_diagnosis}</Text>
                  </>
                )}
                {selectedReferral.treatment_provided && (
                  <>
                    <Text style={styles.detailLabel}>Treatment:</Text>
                    <Text style={styles.detailParagraph}>{selectedReferral.treatment_provided}</Text>
                  </>
                )}
                {selectedReferral.outcome_report && (
                  <>
                    <Text style={styles.detailLabel}>Report:</Text>
                    <Text style={styles.detailParagraph}>{selectedReferral.outcome_report}</Text>
                  </>
                )}
              </View>
            )}

            {/* Status Notes */}
            {selectedReferral.status_notes && (
              <View style={styles.notesCard}>
                <Ionicons name="chatbubble-outline" size={20} color="#6b7280" />
                <Text style={styles.notesText}>{selectedReferral.status_notes}</Text>
              </View>
            )}

            <View style={styles.bottomSpacer} />
          </ScrollView>
        )}
      </LinearGradient>
    </Modal>
  );

  // Update Status Modal
  const renderUpdateStatusModal = () => (
    <Modal
      visible={updateStatusModalVisible}
      transparent
      animationType="slide"
      onRequestClose={() => setUpdateStatusModalVisible(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.updateStatusContent}>
          <View style={styles.updateStatusHeader}>
            <Text style={styles.updateStatusTitle}>Update Status</Text>
            <TouchableOpacity onPress={() => setUpdateStatusModalVisible(false)}>
              <Ionicons name="close" size={24} color="#6b7280" />
            </TouchableOpacity>
          </View>

          <View style={styles.statusOptions}>
            {Object.entries(STATUS_CONFIG).map(([key, config]) => (
              <TouchableOpacity
                key={key}
                style={[
                  styles.statusOption,
                  newStatus === key && { borderColor: config.color, backgroundColor: `${config.color}10` },
                ]}
                onPress={() => setNewStatus(key)}
              >
                <Ionicons name={config.icon as any} size={20} color={config.color} />
                <Text style={[styles.statusOptionText, { color: config.color }]}>
                  {config.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          <Text style={styles.formLabel}>Notes (Optional)</Text>
          <TextInput
            style={[styles.textInput, styles.textAreaInput]}
            placeholder="Add any notes about this status update..."
            value={statusNotes}
            onChangeText={setStatusNotes}
            multiline
            numberOfLines={3}
            placeholderTextColor="#9ca3af"
          />

          {newStatus === 'appointment_scheduled' && (
            <>
              <Text style={styles.formLabel}>Appointment Date</Text>
              <TextInput
                style={styles.textInput}
                placeholder="YYYY-MM-DD"
                value={appointmentDate}
                onChangeText={setAppointmentDate}
                placeholderTextColor="#9ca3af"
              />
            </>
          )}

          <TouchableOpacity
            style={styles.updateButton}
            onPress={updateReferralStatus}
            disabled={isLoading || !newStatus}
          >
            {isLoading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.updateButtonText}>Update Status</Text>
            )}
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  // Dermatologist Detail Modal
  const renderDermatologistDetailModal = () => (
    <Modal
      visible={dermatologistDetailModalVisible}
      animationType="slide"
      onRequestClose={() => setDermatologistDetailModalVisible(false)}
    >
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setDermatologistDetailModalVisible(false)}>
            <Ionicons name="arrow-back" size={28} color="#1e3a5f" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Dermatologist Profile</Text>
          <View style={{ width: 28 }} />
        </View>

        {selectedDermatologist && (
          <ScrollView style={styles.modalContent}>
            {/* Profile Header */}
            <View style={styles.profileHeader}>
              <View style={styles.profileAvatar}>
                <Ionicons name="person" size={48} color="#2563eb" />
              </View>
              <Text style={styles.profileName}>{selectedDermatologist.full_name}</Text>
              <Text style={styles.profileCredentials}>{selectedDermatologist.credentials}</Text>
              {selectedDermatologist.is_verified && (
                <View style={styles.verifiedBadge}>
                  <Ionicons name="checkmark-circle" size={16} color="#10b981" />
                  <Text style={styles.verifiedText}>Verified Provider</Text>
                </View>
              )}
            </View>

            {/* Rating & Stats */}
            {selectedDermatologist.average_rating && (
              <View style={styles.statsRow}>
                <View style={styles.statCard}>
                  <Ionicons name="star" size={24} color="#f59e0b" />
                  <Text style={styles.statValue}>{selectedDermatologist.average_rating.toFixed(1)}</Text>
                  <Text style={styles.statLabel}>Rating</Text>
                </View>
                <View style={styles.statCard}>
                  <Ionicons name="chatbubbles-outline" size={24} color="#6b7280" />
                  <Text style={styles.statValue}>{selectedDermatologist.total_reviews || 0}</Text>
                  <Text style={styles.statLabel}>Reviews</Text>
                </View>
                <View style={styles.statCard}>
                  <Ionicons name="time-outline" size={24} color="#6b7280" />
                  <Text style={styles.statValue}>{selectedDermatologist.typical_wait_time_days || 'N/A'}</Text>
                  <Text style={styles.statLabel}>Days Wait</Text>
                </View>
              </View>
            )}

            {/* Practice Info */}
            <View style={styles.detailCard}>
              <Text style={styles.detailCardTitle}>Practice Information</Text>
              {selectedDermatologist.practice_name && (
                <View style={styles.profileDetailRow}>
                  <Ionicons name="business-outline" size={18} color="#6b7280" />
                  <Text style={styles.profileDetailText}>{selectedDermatologist.practice_name}</Text>
                </View>
              )}
              {selectedDermatologist.city && selectedDermatologist.state && (
                <View style={styles.profileDetailRow}>
                  <Ionicons name="location-outline" size={18} color="#6b7280" />
                  <Text style={styles.profileDetailText}>
                    {selectedDermatologist.city}, {selectedDermatologist.state}
                  </Text>
                </View>
              )}
            </View>

            {/* Specializations */}
            {selectedDermatologist.specializations.length > 0 && (
              <View style={styles.detailCard}>
                <Text style={styles.detailCardTitle}>Specializations</Text>
                <View style={styles.specTagsWrap}>
                  {selectedDermatologist.specializations.map((spec, index) => (
                    <View key={index} style={styles.specTagLarge}>
                      <Text style={styles.specTagTextLarge}>{spec}</Text>
                    </View>
                  ))}
                </View>
              </View>
            )}

            {/* Bio */}
            {selectedDermatologist.bio && (
              <View style={styles.detailCard}>
                <Text style={styles.detailCardTitle}>About</Text>
                <Text style={styles.bioText}>{selectedDermatologist.bio}</Text>
              </View>
            )}

            {/* Languages */}
            {selectedDermatologist.languages_spoken.length > 0 && (
              <View style={styles.detailCard}>
                <Text style={styles.detailCardTitle}>Languages</Text>
                <Text style={styles.languagesText}>
                  {selectedDermatologist.languages_spoken.join(', ')}
                </Text>
              </View>
            )}

            {/* Action Buttons */}
            <TouchableOpacity
              style={styles.createReferralButton}
              onPress={() => {
                setNewDermatologistId(selectedDermatologist.id);
                setDermatologistDetailModalVisible(false);
                setActiveTab('referrals');
                setCreateModalVisible(true);
              }}
            >
              <Ionicons name="send" size={20} color="#fff" />
              <Text style={styles.createReferralButtonText}>Create Referral</Text>
            </TouchableOpacity>

            <View style={styles.bottomSpacer} />
          </ScrollView>
        )}
      </LinearGradient>
    </Modal>
  );

  if (isLoading && referrals.length === 0) {
    return (
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2563eb" />
          <Text style={styles.loadingText}>Loading...</Text>
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
        <Text style={styles.headerTitle}>Referrals</Text>
        <View style={{ width: 40 }} />
      </View>

      {/* Tabs */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'referrals' && styles.activeTab]}
          onPress={() => setActiveTab('referrals')}
        >
          <Ionicons
            name="document-text-outline"
            size={18}
            color={activeTab === 'referrals' ? '#2563eb' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'referrals' && styles.activeTabText]}>
            My Referrals
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'directory' && styles.activeTab]}
          onPress={() => setActiveTab('directory')}
        >
          <Ionicons
            name="people-outline"
            size={18}
            color={activeTab === 'directory' ? '#2563eb' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'directory' && styles.activeTabText]}>
            Find Specialists
          </Text>
        </TouchableOpacity>
      </View>

      {/* Tab Content */}
      {activeTab === 'referrals' && renderReferralsTab()}
      {activeTab === 'directory' && renderDirectoryTab()}

      {/* Modals */}
      {renderCreateModal()}
      {renderDetailModal()}
      {renderUpdateStatusModal()}
      {renderDermatologistDetailModal()}
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
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    gap: 6,
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: '#2563eb',
  },
  tabText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
  },
  activeTabText: {
    color: '#2563eb',
  },
  tabContent: {
    flex: 1,
    padding: 16,
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
  createButton: {
    marginBottom: 16,
    borderRadius: 12,
    overflow: 'hidden',
  },
  createButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    gap: 10,
  },
  createButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  filterScroll: {
    marginBottom: 16,
  },
  filterRow: {
    flexDirection: 'row',
    gap: 8,
    paddingVertical: 4,
  },
  filterChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    gap: 6,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  filterChipActive: {
    backgroundColor: '#2563eb',
    borderColor: '#2563eb',
  },
  filterChipText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
  },
  filterChipTextActive: {
    color: '#fff',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 48,
    backgroundColor: '#fff',
    borderRadius: 16,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  referralCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
  },
  referralHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 4,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
  },
  referralDate: {
    fontSize: 12,
    color: '#9ca3af',
  },
  referralReason: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 4,
  },
  referralConcern: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
  },
  referralFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  referralDetail: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  referralDetailText: {
    fontSize: 13,
    color: '#6b7280',
  },
  urgencyBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  urgencyText: {
    fontSize: 12,
    fontWeight: '600',
  },
  appointmentInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
    gap: 6,
  },
  appointmentText: {
    fontSize: 13,
    color: '#8b5cf6',
    fontWeight: '600',
  },
  searchCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  searchTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 16,
  },
  searchRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 12,
  },
  searchInputContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 10,
    paddingHorizontal: 12,
    gap: 8,
  },
  searchInput: {
    flex: 1,
    paddingVertical: 12,
    fontSize: 15,
    color: '#1e3a5f',
  },
  searchButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563eb',
    borderRadius: 10,
    padding: 14,
    gap: 8,
  },
  searchButtonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
  specFilterScroll: {
    marginBottom: 16,
  },
  specFilterRow: {
    flexDirection: 'row',
    gap: 8,
  },
  specFilterChip: {
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  specFilterChipActive: {
    backgroundColor: '#2563eb',
    borderColor: '#2563eb',
  },
  specFilterText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
  },
  specFilterTextActive: {
    color: '#fff',
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 12,
  },
  dermCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  dermHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  dermAvatar: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#eff6ff',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  dermInfo: {
    flex: 1,
  },
  dermNameRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  dermName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  dermCredentials: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  dermPractice: {
    fontSize: 13,
    color: '#9ca3af',
    marginTop: 2,
  },
  ratingBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fef3c7',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
    gap: 4,
  },
  ratingText: {
    fontSize: 13,
    fontWeight: '700',
    color: '#92400e',
  },
  dermDetails: {
    flexDirection: 'row',
    marginTop: 12,
    gap: 16,
  },
  dermDetailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  dermDetailText: {
    fontSize: 12,
    color: '#6b7280',
  },
  specsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 12,
    gap: 6,
  },
  specTag: {
    backgroundColor: '#f0f9ff',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },
  specTagText: {
    fontSize: 11,
    color: '#1e40af',
    fontWeight: '600',
  },
  moreSpecs: {
    fontSize: 11,
    color: '#9ca3af',
    alignSelf: 'center',
  },
  dermBadges: {
    flexDirection: 'row',
    marginTop: 12,
    gap: 8,
  },
  featureBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
    gap: 4,
  },
  featureBadgeText: {
    fontSize: 11,
    color: '#6b7280',
    fontWeight: '600',
  },
  selectDermButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#eff6ff',
    borderRadius: 8,
    padding: 12,
    marginTop: 12,
    gap: 6,
  },
  selectDermText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2563eb',
  },
  modalContainer: {
    flex: 1,
  },
  modalHeader: {
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
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  modalContent: {
    flex: 1,
    padding: 16,
  },
  formLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
    marginTop: 16,
  },
  reasonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  reasonOption: {
    width: (SCREEN_WIDTH - 52) / 2,
    alignItems: 'center',
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    padding: 16,
    borderWidth: 2,
    borderColor: '#bfdbfe',
  },
  reasonOptionSelected: {
    backgroundColor: '#2563eb',
    borderColor: '#2563eb',
  },
  reasonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#1e3a5f',
    marginTop: 8,
    textAlign: 'center',
  },
  reasonTextSelected: {
    color: '#fff',
  },
  textInput: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 10,
    padding: 14,
    fontSize: 15,
    color: '#1e3a5f',
  },
  textAreaInput: {
    minHeight: 100,
    textAlignVertical: 'top',
  },
  urgencyOptions: {
    gap: 10,
  },
  urgencyOption: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#e5e7eb',
    borderRadius: 12,
    padding: 16,
    gap: 12,
  },
  urgencyDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  urgencyLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  urgencyDesc: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  selectedDermCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    gap: 10,
  },
  selectedDermText: {
    flex: 1,
    fontSize: 14,
    color: '#1e40af',
    fontWeight: '600',
  },
  browseDermButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    gap: 8,
  },
  browseDermText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#2563eb',
  },
  submitButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 18,
    marginTop: 24,
    gap: 10,
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  detailStatusCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    marginBottom: 16,
  },
  detailStatusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    gap: 8,
  },
  detailStatusText: {
    fontSize: 18,
    fontWeight: '700',
  },
  detailDate: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 8,
  },
  detailCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  detailCardTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  detailLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
    marginTop: 8,
  },
  detailValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  detailParagraph: {
    fontSize: 14,
    color: '#374151',
    lineHeight: 22,
    marginTop: 4,
  },
  appointmentDetailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 8,
  },
  appointmentDetailText: {
    fontSize: 14,
    color: '#374151',
  },
  notesCard: {
    flexDirection: 'row',
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    gap: 12,
  },
  notesText: {
    flex: 1,
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  updateStatusContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    paddingBottom: Platform.OS === 'ios' ? 40 : 20,
  },
  updateStatusHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  updateStatusTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  statusOptions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 16,
  },
  statusOption: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    borderWidth: 2,
    borderColor: '#e5e7eb',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    gap: 6,
  },
  statusOptionText: {
    fontSize: 13,
    fontWeight: '600',
  },
  updateButton: {
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 16,
  },
  updateButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  profileHeader: {
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    marginBottom: 16,
  },
  profileAvatar: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: '#eff6ff',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  profileName: {
    fontSize: 22,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  profileCredentials: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 4,
  },
  verifiedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ecfdf5',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    marginTop: 12,
    gap: 6,
  },
  verifiedText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#10b981',
  },
  statsRow: {
    flexDirection: 'row',
    marginBottom: 16,
    gap: 12,
  },
  statCard: {
    flex: 1,
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
  },
  statValue: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  profileDetailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 8,
  },
  profileDetailText: {
    fontSize: 14,
    color: '#374151',
  },
  specTagsWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  specTagLarge: {
    backgroundColor: '#eff6ff',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
  },
  specTagTextLarge: {
    fontSize: 13,
    color: '#1e40af',
    fontWeight: '600',
  },
  bioText: {
    fontSize: 14,
    color: '#374151',
    lineHeight: 22,
  },
  languagesText: {
    fontSize: 14,
    color: '#374151',
  },
  createReferralButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 18,
    marginTop: 8,
    gap: 10,
  },
  createReferralButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  bottomSpacer: {
    height: 30,
  },
});
