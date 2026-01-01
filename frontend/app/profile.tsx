import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
  RefreshControl,
  Modal,
  Switch,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import { API_BASE_URL } from '../config';

interface UserProfile {
  id?: number;
  user_id?: number;
  date_of_birth?: string;
  gender?: string;
  phone_number?: string;
  address?: string;
  emergency_contact?: string;
  medical_history?: string;
  skin_type?: string;
  family_history?: string;
  city?: string;
  state?: string;
  country?: string;
  zip_code?: string;
  latitude?: number;
  longitude?: number;
  preferred_distance_miles?: number;
  notification_preferences?: NotificationPreferences;
  privacy_settings?: PrivacySettings;
  total_analyses?: number;
  last_analysis_date?: string;
  created_at?: string;
  updated_at?: string;
}

interface NotificationPreferences {
  email_notifications?: boolean;
  push_notifications?: boolean;
  sms_notifications?: boolean;
  appointment_reminders?: boolean;
  analysis_results?: boolean;
  promotional?: boolean;
}

interface PrivacySettings {
  share_with_dermatologists?: boolean;
  allow_research_use?: boolean;
  show_in_directory?: boolean;
}

interface ExtendedUserInfo {
  id: number;
  username: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  created_at: string;
  profile?: UserProfile;
  total_analyses: number;
  last_analysis_date?: string;
}

const SKIN_TYPES = [
  { value: 'I', label: 'Type I', description: 'Very fair, always burns, never tans' },
  { value: 'II', label: 'Type II', description: 'Fair, usually burns, tans minimally' },
  { value: 'III', label: 'Type III', description: 'Medium, sometimes burns, tans gradually' },
  { value: 'IV', label: 'Type IV', description: 'Olive, rarely burns, tans easily' },
  { value: 'V', label: 'Type V', description: 'Brown, very rarely burns, tans very easily' },
  { value: 'VI', label: 'Type VI', description: 'Dark brown/black, never burns' },
];

const GENDERS = [
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
  { value: 'other', label: 'Other' },
  { value: 'prefer_not_to_say', label: 'Prefer not to say' },
];

export default function ProfileScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'personal' | 'medical' | 'location' | 'preferences'>('personal');
  const [userInfo, setUserInfo] = useState<ExtendedUserInfo | null>(null);
  const [profile, setProfile] = useState<UserProfile>({});
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  // Full name editing
  const [fullName, setFullName] = useState<string>('');
  const [hasFullNameChanges, setHasFullNameChanges] = useState(false);

  // Skin type picker modal
  const [showSkinTypePicker, setShowSkinTypePicker] = useState(false);

  useEffect(() => {
    fetchUserInfo();
  }, []);

  const fetchUserInfo = async () => {
    try {
      const token = await SecureStore.getItemAsync('auth_token');
      const response = await fetch(`${API_BASE_URL}/me/extended`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setUserInfo(data);
        setProfile(data.profile || {});
        setFullName(data.full_name || '');
      }
    } catch (error) {
      console.error('Error fetching user info:', error);
      Alert.alert('Error', 'Failed to fetch profile');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchUserInfo();
  }, []);

  const updateProfile = (field: keyof UserProfile, value: any) => {
    setProfile(prev => ({ ...prev, [field]: value }));
    setHasChanges(true);
  };

  const updateFullName = (value: string) => {
    setFullName(value);
    setHasFullNameChanges(value !== (userInfo?.full_name || ''));
  };

  const updateNotificationPreferences = (field: keyof NotificationPreferences, value: boolean) => {
    setProfile(prev => ({
      ...prev,
      notification_preferences: {
        ...prev.notification_preferences,
        [field]: value,
      },
    }));
    setHasChanges(true);
  };

  const updatePrivacySettings = (field: keyof PrivacySettings, value: boolean) => {
    setProfile(prev => ({
      ...prev,
      privacy_settings: {
        ...prev.privacy_settings,
        [field]: value,
      },
    }));
    setHasChanges(true);
  };

  const handleSaveProfile = async () => {
    setSaving(true);
    try {
      const token = await SecureStore.getItemAsync('auth_token');

      // Save full name if changed
      if (hasFullNameChanges) {
        const settingsResponse = await fetch(`${API_BASE_URL}/me/settings?full_name=${encodeURIComponent(fullName)}`, {
          method: 'PUT',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        if (!settingsResponse.ok) {
          const error = await settingsResponse.json();
          Alert.alert('Error', error.detail || 'Failed to update full name');
          setSaving(false);
          return;
        }
      }

      // Save profile data if changed
      if (hasChanges) {
        const response = await fetch(`${API_BASE_URL}/profile`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(profile),
        });

        if (!response.ok) {
          const error = await response.json();
          Alert.alert('Error', error.detail || 'Failed to update profile');
          setSaving(false);
          return;
        }
      }

      Alert.alert('Success', 'Profile updated successfully');
      setHasChanges(false);
      setHasFullNameChanges(false);
      fetchUserInfo();
    } catch (error) {
      console.error('Error saving profile:', error);
      Alert.alert('Error', 'Failed to save profile');
    } finally {
      setSaving(false);
    }
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'Not set';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  const getSkinTypeLabel = (value?: string) => {
    const skinType = SKIN_TYPES.find(st => st.value === value);
    return skinType ? `${skinType.label} - ${skinType.description}` : 'Not set';
  };

  const renderPersonalTab = () => (
    <View style={styles.tabContent}>
      {/* Account Info Card */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="person-circle" size={24} color="#667eea" />
          <Text style={styles.cardTitle}>Account Information</Text>
        </View>

        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Username</Text>
          <Text style={styles.infoValue}>{userInfo?.username || 'N/A'}</Text>
        </View>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Email</Text>
          <Text style={styles.infoValue}>{userInfo?.email || 'N/A'}</Text>
        </View>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Full Name</Text>
          <TextInput
            style={[styles.infoValue, styles.editableInput]}
            value={fullName}
            onChangeText={updateFullName}
            placeholder="Enter your full name"
            placeholderTextColor="#999"
          />
        </View>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Member Since</Text>
          <Text style={styles.infoValue}>{formatDate(userInfo?.created_at)}</Text>
        </View>
      </View>

      {/* Personal Details Card */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="id-card" size={24} color="#667eea" />
          <Text style={styles.cardTitle}>Personal Details</Text>
        </View>

        <Text style={styles.inputLabel}>Date of Birth</Text>
        <TextInput
          style={styles.textInput}
          value={profile.date_of_birth?.split('T')[0] || ''}
          onChangeText={(text) => updateProfile('date_of_birth', text)}
          placeholder="YYYY-MM-DD"
        />

        <Text style={styles.inputLabel}>Gender</Text>
        <View style={styles.genderOptions}>
          {GENDERS.map((gender) => (
            <TouchableOpacity
              key={gender.value}
              style={[
                styles.genderOption,
                profile.gender === gender.value && styles.genderOptionSelected,
              ]}
              onPress={() => updateProfile('gender', gender.value)}
            >
              <Text
                style={[
                  styles.genderOptionText,
                  profile.gender === gender.value && styles.genderOptionTextSelected,
                ]}
              >
                {gender.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <Text style={styles.inputLabel}>Phone Number</Text>
        <TextInput
          style={styles.textInput}
          value={profile.phone_number || ''}
          onChangeText={(text) => updateProfile('phone_number', text)}
          placeholder="(555) 123-4567"
          keyboardType="phone-pad"
        />

        <Text style={styles.inputLabel}>Emergency Contact</Text>
        <TextInput
          style={styles.textInput}
          value={profile.emergency_contact || ''}
          onChangeText={(text) => updateProfile('emergency_contact', text)}
          placeholder="Name and phone number"
        />
      </View>

      {/* Statistics Card */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="stats-chart" size={24} color="#667eea" />
          <Text style={styles.cardTitle}>Your Statistics</Text>
        </View>

        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{userInfo?.total_analyses || 0}</Text>
            <Text style={styles.statLabel}>Total Analyses</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statValue}>
              {userInfo?.last_analysis_date ? formatDate(userInfo.last_analysis_date).split(',')[0] : 'Never'}
            </Text>
            <Text style={styles.statLabel}>Last Analysis</Text>
          </View>
        </View>
      </View>
    </View>
  );

  const renderMedicalTab = () => (
    <View style={styles.tabContent}>
      {/* Skin Type Card */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="color-palette" size={24} color="#667eea" />
          <Text style={styles.cardTitle}>Skin Type (Fitzpatrick Scale)</Text>
        </View>

        <TouchableOpacity
          style={styles.pickerButton}
          onPress={() => setShowSkinTypePicker(true)}
        >
          <Text style={styles.pickerButtonText}>
            {profile.skin_type ? getSkinTypeLabel(profile.skin_type) : 'Select your skin type'}
          </Text>
          <Ionicons name="chevron-down" size={20} color="#666" />
        </TouchableOpacity>

        <Text style={styles.helperText}>
          Your Fitzpatrick skin type helps us provide more accurate risk assessments
        </Text>
      </View>

      {/* Medical History Card */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="medical" size={24} color="#667eea" />
          <Text style={styles.cardTitle}>Medical History</Text>
        </View>

        <Text style={styles.inputLabel}>Medical Conditions & History</Text>
        <TextInput
          style={[styles.textInput, styles.textArea]}
          value={profile.medical_history || ''}
          onChangeText={(text) => updateProfile('medical_history', text)}
          placeholder="List any relevant medical conditions, allergies, or past procedures..."
          multiline
          numberOfLines={4}
        />

        <Text style={styles.inputLabel}>Family History</Text>
        <TextInput
          style={[styles.textInput, styles.textArea]}
          value={profile.family_history || ''}
          onChangeText={(text) => updateProfile('family_history', text)}
          placeholder="Any family history of skin cancer or other skin conditions..."
          multiline
          numberOfLines={4}
        />
      </View>

      {/* Quick Actions */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="flash" size={24} color="#667eea" />
          <Text style={styles.cardTitle}>Detailed Medical Records</Text>
        </View>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => router.push('/family-history')}
        >
          <Ionicons name="people" size={20} color="#667eea" />
          <Text style={styles.actionButtonText}>Family History Details</Text>
          <Ionicons name="chevron-forward" size={20} color="#999" />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => router.push('/medication-checker')}
        >
          <Ionicons name="medkit" size={20} color="#667eea" />
          <Text style={styles.actionButtonText}>Medications</Text>
          <Ionicons name="chevron-forward" size={20} color="#999" />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => router.push('/risk-calculator')}
        >
          <Ionicons name="calculator" size={20} color="#667eea" />
          <Text style={styles.actionButtonText}>Risk Assessment</Text>
          <Ionicons name="chevron-forward" size={20} color="#999" />
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderLocationTab = () => (
    <View style={styles.tabContent}>
      {/* Address Card */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="location" size={24} color="#667eea" />
          <Text style={styles.cardTitle}>Address</Text>
        </View>

        <Text style={styles.inputLabel}>Street Address</Text>
        <TextInput
          style={[styles.textInput, styles.textArea]}
          value={profile.address || ''}
          onChangeText={(text) => updateProfile('address', text)}
          placeholder="123 Main Street, Apt 4B"
          multiline
          numberOfLines={2}
        />

        <View style={styles.row}>
          <View style={styles.halfInput}>
            <Text style={styles.inputLabel}>City</Text>
            <TextInput
              style={styles.textInput}
              value={profile.city || ''}
              onChangeText={(text) => updateProfile('city', text)}
              placeholder="City"
            />
          </View>
          <View style={styles.halfInput}>
            <Text style={styles.inputLabel}>State</Text>
            <TextInput
              style={styles.textInput}
              value={profile.state || ''}
              onChangeText={(text) => updateProfile('state', text)}
              placeholder="State"
            />
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.halfInput}>
            <Text style={styles.inputLabel}>ZIP Code</Text>
            <TextInput
              style={styles.textInput}
              value={profile.zip_code || ''}
              onChangeText={(text) => updateProfile('zip_code', text)}
              placeholder="12345"
              keyboardType="numeric"
            />
          </View>
          <View style={styles.halfInput}>
            <Text style={styles.inputLabel}>Country</Text>
            <TextInput
              style={styles.textInput}
              value={profile.country || 'USA'}
              onChangeText={(text) => updateProfile('country', text)}
              placeholder="USA"
            />
          </View>
        </View>
      </View>

      {/* Dermatologist Preferences */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="navigate" size={24} color="#667eea" />
          <Text style={styles.cardTitle}>Dermatologist Search Preferences</Text>
        </View>

        <Text style={styles.inputLabel}>Maximum Distance (miles)</Text>
        <View style={styles.distanceOptions}>
          {[10, 25, 50, 100].map((distance) => (
            <TouchableOpacity
              key={distance}
              style={[
                styles.distanceOption,
                profile.preferred_distance_miles === distance && styles.distanceOptionSelected,
              ]}
              onPress={() => updateProfile('preferred_distance_miles', distance)}
            >
              <Text
                style={[
                  styles.distanceOptionText,
                  profile.preferred_distance_miles === distance && styles.distanceOptionTextSelected,
                ]}
              >
                {distance} mi
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <Text style={styles.helperText}>
          This helps us find dermatologists within your preferred travel distance
        </Text>
      </View>
    </View>
  );

  const renderPreferencesTab = () => (
    <View style={styles.tabContent}>
      {/* Notification Preferences */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="notifications" size={24} color="#667eea" />
          <Text style={styles.cardTitle}>Notification Preferences</Text>
        </View>

        <View style={styles.switchRow}>
          <View style={styles.switchInfo}>
            <Text style={styles.switchLabel}>Email Notifications</Text>
            <Text style={styles.switchDescription}>Receive updates via email</Text>
          </View>
          <Switch
            value={profile.notification_preferences?.email_notifications ?? true}
            onValueChange={(value) => updateNotificationPreferences('email_notifications', value)}
            trackColor={{ false: '#ddd', true: '#667eea' }}
            thumbColor="#fff"
          />
        </View>

        <View style={styles.switchRow}>
          <View style={styles.switchInfo}>
            <Text style={styles.switchLabel}>Push Notifications</Text>
            <Text style={styles.switchDescription}>Receive push notifications on your device</Text>
          </View>
          <Switch
            value={profile.notification_preferences?.push_notifications ?? true}
            onValueChange={(value) => updateNotificationPreferences('push_notifications', value)}
            trackColor={{ false: '#ddd', true: '#667eea' }}
            thumbColor="#fff"
          />
        </View>

        <View style={styles.switchRow}>
          <View style={styles.switchInfo}>
            <Text style={styles.switchLabel}>SMS Notifications</Text>
            <Text style={styles.switchDescription}>Receive text message alerts</Text>
          </View>
          <Switch
            value={profile.notification_preferences?.sms_notifications ?? false}
            onValueChange={(value) => updateNotificationPreferences('sms_notifications', value)}
            trackColor={{ false: '#ddd', true: '#667eea' }}
            thumbColor="#fff"
          />
        </View>

        <View style={styles.switchRow}>
          <View style={styles.switchInfo}>
            <Text style={styles.switchLabel}>Appointment Reminders</Text>
            <Text style={styles.switchDescription}>Get reminded about upcoming appointments</Text>
          </View>
          <Switch
            value={profile.notification_preferences?.appointment_reminders ?? true}
            onValueChange={(value) => updateNotificationPreferences('appointment_reminders', value)}
            trackColor={{ false: '#ddd', true: '#667eea' }}
            thumbColor="#fff"
          />
        </View>

        <View style={styles.switchRow}>
          <View style={styles.switchInfo}>
            <Text style={styles.switchLabel}>Analysis Results</Text>
            <Text style={styles.switchDescription}>Notify when analysis is complete</Text>
          </View>
          <Switch
            value={profile.notification_preferences?.analysis_results ?? true}
            onValueChange={(value) => updateNotificationPreferences('analysis_results', value)}
            trackColor={{ false: '#ddd', true: '#667eea' }}
            thumbColor="#fff"
          />
        </View>
      </View>

      {/* Privacy Settings */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="shield-checkmark" size={24} color="#667eea" />
          <Text style={styles.cardTitle}>Privacy Settings</Text>
        </View>

        <View style={styles.switchRow}>
          <View style={styles.switchInfo}>
            <Text style={styles.switchLabel}>Share with Dermatologists</Text>
            <Text style={styles.switchDescription}>Allow sharing analysis data with healthcare providers</Text>
          </View>
          <Switch
            value={profile.privacy_settings?.share_with_dermatologists ?? true}
            onValueChange={(value) => updatePrivacySettings('share_with_dermatologists', value)}
            trackColor={{ false: '#ddd', true: '#667eea' }}
            thumbColor="#fff"
          />
        </View>

        <View style={styles.switchRow}>
          <View style={styles.switchInfo}>
            <Text style={styles.switchLabel}>Research Participation</Text>
            <Text style={styles.switchDescription}>Allow anonymized data to be used for research</Text>
          </View>
          <Switch
            value={profile.privacy_settings?.allow_research_use ?? false}
            onValueChange={(value) => updatePrivacySettings('allow_research_use', value)}
            trackColor={{ false: '#ddd', true: '#667eea' }}
            thumbColor="#fff"
          />
        </View>
      </View>

      {/* Data Management */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="folder" size={24} color="#667eea" />
          <Text style={styles.cardTitle}>Data Management</Text>
        </View>

        <TouchableOpacity style={styles.actionButton} onPress={() => router.push('/history')}>
          <Ionicons name="time" size={20} color="#667eea" />
          <Text style={styles.actionButtonText}>View Analysis History</Text>
          <Ionicons name="chevron-forward" size={20} color="#999" />
        </TouchableOpacity>

        <TouchableOpacity style={styles.actionButton} onPress={() => router.push('/fhir-export')}>
          <Ionicons name="download" size={20} color="#667eea" />
          <Text style={styles.actionButtonText}>Export Health Data (FHIR)</Text>
          <Ionicons name="chevron-forward" size={20} color="#999" />
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderSkinTypeModal = () => (
    <Modal visible={showSkinTypePicker} animationType="slide" transparent>
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Select Skin Type</Text>
            <TouchableOpacity onPress={() => setShowSkinTypePicker(false)}>
              <Ionicons name="close" size={24} color="#333" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.skinTypeList}>
            {SKIN_TYPES.map((skinType) => (
              <TouchableOpacity
                key={skinType.value}
                style={[
                  styles.skinTypeItem,
                  profile.skin_type === skinType.value && styles.skinTypeItemSelected,
                ]}
                onPress={() => {
                  updateProfile('skin_type', skinType.value);
                  setShowSkinTypePicker(false);
                }}
              >
                <View style={styles.skinTypeInfo}>
                  <Text style={styles.skinTypeLabel}>{skinType.label}</Text>
                  <Text style={styles.skinTypeDescription}>{skinType.description}</Text>
                </View>
                {profile.skin_type === skinType.value && (
                  <Ionicons name="checkmark-circle" size={24} color="#4CAF50" />
                )}
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );

  if (loading) {
    return (
      <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#fff" />
          <Text style={styles.loadingText}>Loading profile...</Text>
        </View>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>My Profile</Text>
        {(hasChanges || hasFullNameChanges) ? (
          <TouchableOpacity onPress={handleSaveProfile} style={styles.saveButton} disabled={saving}>
            {saving ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <Text style={styles.saveButtonText}>Save</Text>
            )}
          </TouchableOpacity>
        ) : (
          <View style={styles.headerRight} />
        )}
      </View>

      {/* Profile Avatar */}
      <View style={styles.avatarSection}>
        <View style={styles.avatar}>
          <Ionicons name="person" size={48} color="#667eea" />
        </View>
        <Text style={styles.userName}>{userInfo?.full_name || userInfo?.username || 'User'}</Text>
        <Text style={styles.userEmail}>{userInfo?.email}</Text>
      </View>

      {/* Tabs */}
      <View style={styles.tabContainer}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.tabScroll}>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'personal' && styles.activeTab]}
            onPress={() => setActiveTab('personal')}
          >
            <Ionicons name="person" size={16} color={activeTab === 'personal' ? '#667eea' : '#666'} />
            <Text style={[styles.tabText, activeTab === 'personal' && styles.activeTabText]}>Personal</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'medical' && styles.activeTab]}
            onPress={() => setActiveTab('medical')}
          >
            <Ionicons name="medical" size={16} color={activeTab === 'medical' ? '#667eea' : '#666'} />
            <Text style={[styles.tabText, activeTab === 'medical' && styles.activeTabText]}>Medical</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'location' && styles.activeTab]}
            onPress={() => setActiveTab('location')}
          >
            <Ionicons name="location" size={16} color={activeTab === 'location' ? '#667eea' : '#666'} />
            <Text style={[styles.tabText, activeTab === 'location' && styles.activeTabText]}>Location</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'preferences' && styles.activeTab]}
            onPress={() => setActiveTab('preferences')}
          >
            <Ionicons name="settings" size={16} color={activeTab === 'preferences' ? '#667eea' : '#666'} />
            <Text style={[styles.tabText, activeTab === 'preferences' && styles.activeTabText]}>Preferences</Text>
          </TouchableOpacity>
        </ScrollView>
      </View>

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#fff" />
        }
      >
        {activeTab === 'personal' && renderPersonalTab()}
        {activeTab === 'medical' && renderMedicalTab()}
        {activeTab === 'location' && renderLocationTab()}
        {activeTab === 'preferences' && renderPreferencesTab()}
      </ScrollView>

      {renderSkinTypeModal()}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#fff',
    marginTop: 10,
    fontSize: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 15,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  headerRight: {
    width: 60,
  },
  saveButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 8,
  },
  saveButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  avatarSection: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  userName: {
    fontSize: 20,
    fontWeight: '600',
    color: '#fff',
  },
  userEmail: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 4,
  },
  tabContainer: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    borderRadius: 12,
    marginBottom: 15,
  },
  tabScroll: {
    padding: 4,
  },
  tab: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 10,
    marginRight: 4,
    gap: 6,
  },
  activeTab: {
    backgroundColor: '#f0f0f0',
  },
  tabText: {
    fontSize: 13,
    color: '#666',
    fontWeight: '500',
  },
  activeTabText: {
    color: '#667eea',
    fontWeight: '600',
  },
  content: {
    flex: 1,
  },
  tabContent: {
    paddingHorizontal: 20,
    paddingBottom: 30,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  infoLabel: {
    fontSize: 14,
    color: '#666',
  },
  infoValue: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  editableInput: {
    backgroundColor: '#f5f5f5',
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 4,
    minWidth: 150,
    textAlign: 'right',
  },
  inputLabel: {
    fontSize: 13,
    color: '#666',
    marginBottom: 6,
    marginTop: 12,
  },
  textInput: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    fontSize: 15,
    color: '#333',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  textArea: {
    minHeight: 80,
    textAlignVertical: 'top',
  },
  genderOptions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 4,
  },
  genderOption: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#f5f5f5',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  genderOptionSelected: {
    backgroundColor: '#667eea',
    borderColor: '#667eea',
  },
  genderOptionText: {
    fontSize: 13,
    color: '#666',
  },
  genderOptionTextSelected: {
    color: '#fff',
  },
  statsRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#667eea',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  statDivider: {
    width: 1,
    height: 40,
    backgroundColor: '#eee',
    marginHorizontal: 20,
  },
  pickerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  pickerButtonText: {
    fontSize: 14,
    color: '#333',
    flex: 1,
  },
  helperText: {
    fontSize: 12,
    color: '#999',
    marginTop: 8,
    fontStyle: 'italic',
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
    gap: 12,
  },
  actionButtonText: {
    flex: 1,
    fontSize: 15,
    color: '#333',
  },
  row: {
    flexDirection: 'row',
    gap: 12,
  },
  halfInput: {
    flex: 1,
  },
  distanceOptions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 4,
  },
  distanceOption: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: '#f5f5f5',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  distanceOptionSelected: {
    backgroundColor: '#667eea',
    borderColor: '#667eea',
  },
  distanceOptionText: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  distanceOptionTextSelected: {
    color: '#fff',
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  switchInfo: {
    flex: 1,
    marginRight: 12,
  },
  switchLabel: {
    fontSize: 15,
    color: '#333',
    fontWeight: '500',
  },
  switchDescription: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '70%',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  skinTypeList: {
    padding: 10,
  },
  skinTypeItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    marginBottom: 8,
    backgroundColor: '#f5f5f5',
  },
  skinTypeItemSelected: {
    backgroundColor: '#E8EAF6',
    borderWidth: 1,
    borderColor: '#667eea',
  },
  skinTypeInfo: {
    flex: 1,
  },
  skinTypeLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  skinTypeDescription: {
    fontSize: 13,
    color: '#666',
    marginTop: 4,
  },
});
