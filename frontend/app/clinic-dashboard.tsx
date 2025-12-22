import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  RefreshControl,
  ActivityIndicator,
  Alert,
  Modal,
  TextInput,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as SecureStore from 'expo-secure-store';
import { API_URL } from '../config';
import { useUserSettings } from '../contexts/UserSettingsContext';

interface Clinic {
  id: number;
  name: string;
  address?: string;
  phone?: string;
  email?: string;
  description?: string;
  specialty?: string;
  clinic_code: string;
  is_verified: boolean;
  is_active: boolean;
  staff_count?: number;
  patient_count?: number;
  created_at: string;
}

interface Invitation {
  id: number;
  clinic_id: number;
  clinic_name: string;
  invitation_code: string;
  consent_level: string;
  message?: string;
  status: string;
  expires_at: string;
}

interface RoleSummary {
  user_id: number;
  account_type: string;
  display_mode: string;
  is_verified_professional: boolean;
  clinics_as_staff: Clinic[];
  clinics_as_patient: Clinic[];
  pending_invitations: Invitation[];
}

export default function ClinicDashboard() {
  const router = useRouter();
  const { settings } = useUserSettings();
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [roleSummary, setRoleSummary] = useState<RoleSummary | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newClinicName, setNewClinicName] = useState('');
  const [newClinicAddress, setNewClinicAddress] = useState('');
  const [newClinicPhone, setNewClinicPhone] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const getAuthToken = async (): Promise<string | null> => {
    try {
      return await SecureStore.getItemAsync('auth_token');
    } catch {
      return null;
    }
  };

  const fetchRoleSummary = useCallback(async () => {
    try {
      const token = await getAuthToken();
      if (!token) {
        router.replace('/');
        return;
      }

      const response = await fetch(`${API_URL}/api/clinics/user/role-summary`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setRoleSummary(data);
      } else if (response.status === 401) {
        router.replace('/');
      }
    } catch (error) {
      console.error('Error fetching role summary:', error);
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  }, [router]);

  useEffect(() => {
    fetchRoleSummary();
  }, [fetchRoleSummary]);

  const onRefresh = () => {
    setRefreshing(true);
    fetchRoleSummary();
  };

  const handleCreateClinic = async () => {
    if (!newClinicName.trim()) {
      Alert.alert('Error', 'Please enter a clinic name');
      return;
    }

    if (!settings.isVerifiedProfessional) {
      Alert.alert(
        'Verification Required',
        'You need to be a verified healthcare professional to create a clinic. Would you like to start the verification process?',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Verify Now', onPress: () => router.push('/professional-verification' as any) }
        ]
      );
      return;
    }

    setIsCreating(true);
    try {
      const token = await getAuthToken();
      if (!token) return;

      const response = await fetch(`${API_URL}/api/clinics/`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: newClinicName.trim(),
          address: newClinicAddress.trim() || undefined,
          phone: newClinicPhone.trim() || undefined,
        }),
      });

      if (response.ok) {
        const newClinic = await response.json();
        Alert.alert('Success', `Clinic "${newClinic.name}" created successfully!`);
        setShowCreateModal(false);
        setNewClinicName('');
        setNewClinicAddress('');
        setNewClinicPhone('');
        fetchRoleSummary();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to create clinic');
      }
    } catch (error) {
      Alert.alert('Error', 'Network error. Please try again.');
    } finally {
      setIsCreating(false);
    }
  };

  const handleRespondToInvitation = async (invitationCode: string, accept: boolean) => {
    try {
      const token = await getAuthToken();
      if (!token) return;

      const response = await fetch(`${API_URL}/api/clinics/invitations/respond`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          invitation_code: invitationCode,
          accept: accept,
        }),
      });

      if (response.ok || response.status === 200) {
        Alert.alert('Success', accept ? 'You have joined the clinic!' : 'Invitation declined');
        fetchRoleSummary();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to respond to invitation');
      }
    } catch (error) {
      Alert.alert('Error', 'Network error. Please try again.');
    }
  };

  const renderClinicCard = (clinic: Clinic, isStaff: boolean) => (
    <TouchableOpacity
      key={`${isStaff ? 'staff' : 'patient'}-${clinic.id}`}
      style={styles.clinicCard}
      onPress={() => {
        // Navigate to clinic details
        // router.push(`/clinic/${clinic.id}` as any);
      }}
    >
      <View style={styles.clinicHeader}>
        <View style={styles.clinicInfo}>
          <Text style={styles.clinicName}>{clinic.name}</Text>
          {clinic.specialty && (
            <Text style={styles.clinicSpecialty}>{clinic.specialty}</Text>
          )}
        </View>
        <View style={[styles.roleBadge, isStaff ? styles.staffBadge : styles.patientBadge]}>
          <Ionicons
            name={isStaff ? 'medkit' : 'person'}
            size={12}
            color={isStaff ? '#2E7D32' : '#1976D2'}
          />
          <Text style={[styles.roleBadgeText, isStaff ? styles.staffText : styles.patientText]}>
            {isStaff ? 'Staff' : 'Patient'}
          </Text>
        </View>
      </View>

      {clinic.address && (
        <View style={styles.clinicDetail}>
          <Ionicons name="location-outline" size={14} color="#666" />
          <Text style={styles.clinicDetailText}>{clinic.address}</Text>
        </View>
      )}

      {isStaff && (
        <View style={styles.clinicStats}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{clinic.staff_count || 0}</Text>
            <Text style={styles.statLabel}>Staff</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{clinic.patient_count || 0}</Text>
            <Text style={styles.statLabel}>Patients</Text>
          </View>
          <View style={styles.statItem}>
            <View style={[styles.statusDot, clinic.is_verified ? styles.verifiedDot : styles.pendingDot]} />
            <Text style={styles.statLabel}>{clinic.is_verified ? 'Verified' : 'Pending'}</Text>
          </View>
        </View>
      )}

      <View style={styles.clinicCode}>
        <Text style={styles.codeLabel}>Clinic Code:</Text>
        <Text style={styles.codeValue}>{clinic.clinic_code}</Text>
      </View>
    </TouchableOpacity>
  );

  const renderInvitation = (invitation: Invitation) => (
    <View key={invitation.id} style={styles.invitationCard}>
      <View style={styles.invitationHeader}>
        <Ionicons name="mail-outline" size={24} color="#4A90A4" />
        <View style={styles.invitationInfo}>
          <Text style={styles.invitationTitle}>{invitation.clinic_name}</Text>
          <Text style={styles.invitationSubtitle}>
            Invited to join as patient ({invitation.consent_level} access)
          </Text>
        </View>
      </View>

      {invitation.message && (
        <Text style={styles.invitationMessage}>"{invitation.message}"</Text>
      )}

      <Text style={styles.expiryText}>
        Expires: {new Date(invitation.expires_at).toLocaleDateString()}
      </Text>

      <View style={styles.invitationActions}>
        <TouchableOpacity
          style={styles.declineButton}
          onPress={() => handleRespondToInvitation(invitation.invitation_code, false)}
        >
          <Text style={styles.declineButtonText}>Decline</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.acceptButton}
          onPress={() => handleRespondToInvitation(invitation.invitation_code, true)}
        >
          <Text style={styles.acceptButtonText}>Accept</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4A90A4" />
          <Text style={styles.loadingText}>Loading clinics...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color="#333" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>My Clinics</Text>
        {settings.isVerifiedProfessional && (
          <TouchableOpacity style={styles.addButton} onPress={() => setShowCreateModal(true)}>
            <Ionicons name="add" size={24} color="#4A90A4" />
          </TouchableOpacity>
        )}
      </View>

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {/* Pending Invitations */}
        {roleSummary?.pending_invitations && roleSummary.pending_invitations.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>
              <Ionicons name="mail" size={16} color="#4A90A4" /> Pending Invitations
            </Text>
            {roleSummary.pending_invitations.map(renderInvitation)}
          </View>
        )}

        {/* Clinics as Staff */}
        {roleSummary?.clinics_as_staff && roleSummary.clinics_as_staff.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>
              <Ionicons name="medkit" size={16} color="#2E7D32" /> Clinics (Staff)
            </Text>
            {roleSummary.clinics_as_staff.map(clinic => renderClinicCard(clinic, true))}
          </View>
        )}

        {/* Clinics as Patient */}
        {roleSummary?.clinics_as_patient && roleSummary.clinics_as_patient.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>
              <Ionicons name="person" size={16} color="#1976D2" /> Clinics (Patient)
            </Text>
            {roleSummary.clinics_as_patient.map(clinic => renderClinicCard(clinic, false))}
          </View>
        )}

        {/* Empty State */}
        {(!roleSummary ||
          (roleSummary.clinics_as_staff.length === 0 &&
            roleSummary.clinics_as_patient.length === 0 &&
            roleSummary.pending_invitations.length === 0)) && (
          <View style={styles.emptyState}>
            <Ionicons name="business-outline" size={64} color="#ccc" />
            <Text style={styles.emptyTitle}>No Clinics Yet</Text>
            <Text style={styles.emptyText}>
              {settings.isVerifiedProfessional
                ? 'Create a clinic to start managing patients, or wait for a clinic to invite you as a patient.'
                : 'You are not linked to any clinics. Ask your healthcare provider to invite you.'}
            </Text>
            {settings.isVerifiedProfessional && (
              <TouchableOpacity
                style={styles.createClinicButton}
                onPress={() => setShowCreateModal(true)}
              >
                <Ionicons name="add-circle" size={20} color="#fff" />
                <Text style={styles.createClinicButtonText}>Create a Clinic</Text>
              </TouchableOpacity>
            )}
          </View>
        )}

        {/* Professional Verification CTA */}
        {!settings.isVerifiedProfessional && (
          <TouchableOpacity
            style={styles.verificationCTA}
            onPress={() => router.push('/professional-verification' as any)}
          >
            <Ionicons name="shield-checkmark-outline" size={24} color="#4A90A4" />
            <View style={styles.ctaContent}>
              <Text style={styles.ctaTitle}>Are you a healthcare professional?</Text>
              <Text style={styles.ctaSubtitle}>
                Verify your credentials to create clinics and access professional features.
              </Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#999" />
          </TouchableOpacity>
        )}
      </ScrollView>

      {/* Create Clinic Modal */}
      <Modal
        visible={showCreateModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowCreateModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Create New Clinic</Text>
              <TouchableOpacity onPress={() => setShowCreateModal(false)}>
                <Ionicons name="close" size={24} color="#666" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalBody}>
              <Text style={styles.inputLabel}>Clinic Name *</Text>
              <TextInput
                style={styles.input}
                placeholder="Enter clinic name"
                value={newClinicName}
                onChangeText={setNewClinicName}
              />

              <Text style={styles.inputLabel}>Address</Text>
              <TextInput
                style={styles.input}
                placeholder="Enter address (optional)"
                value={newClinicAddress}
                onChangeText={setNewClinicAddress}
              />

              <Text style={styles.inputLabel}>Phone</Text>
              <TextInput
                style={styles.input}
                placeholder="Enter phone number (optional)"
                value={newClinicPhone}
                onChangeText={setNewClinicPhone}
                keyboardType="phone-pad"
              />
            </ScrollView>

            <View style={styles.modalFooter}>
              <TouchableOpacity
                style={styles.cancelButton}
                onPress={() => setShowCreateModal(false)}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.submitButton, isCreating && styles.disabledButton]}
                onPress={handleCreateClinic}
                disabled={isCreating}
              >
                {isCreating ? (
                  <ActivityIndicator size="small" color="#fff" />
                ) : (
                  <Text style={styles.submitButtonText}>Create Clinic</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    paddingVertical: 16,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  backButton: {
    marginRight: 12,
  },
  headerTitle: {
    flex: 1,
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  addButton: {
    padding: 4,
  },
  content: {
    flex: 1,
  },
  section: {
    marginTop: 16,
    paddingHorizontal: 16,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
    marginBottom: 12,
  },
  clinicCard: {
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
  clinicHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  clinicInfo: {
    flex: 1,
  },
  clinicName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  clinicSpecialty: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  roleBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  staffBadge: {
    backgroundColor: '#e8f5e9',
  },
  patientBadge: {
    backgroundColor: '#e3f2fd',
  },
  roleBadgeText: {
    fontSize: 11,
    fontWeight: '600',
    marginLeft: 4,
  },
  staffText: {
    color: '#2E7D32',
  },
  patientText: {
    color: '#1976D2',
  },
  clinicDetail: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  clinicDetailText: {
    fontSize: 13,
    color: '#666',
    marginLeft: 6,
  },
  clinicStats: {
    flexDirection: 'row',
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginBottom: 4,
  },
  verifiedDot: {
    backgroundColor: '#4CAF50',
  },
  pendingDot: {
    backgroundColor: '#FFC107',
  },
  clinicCode: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  codeLabel: {
    fontSize: 12,
    color: '#999',
  },
  codeValue: {
    fontSize: 12,
    fontWeight: '600',
    color: '#4A90A4',
    marginLeft: 4,
    fontFamily: 'monospace',
  },
  invitationCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#4A90A4',
  },
  invitationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  invitationInfo: {
    flex: 1,
    marginLeft: 12,
  },
  invitationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  invitationSubtitle: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  invitationMessage: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
    marginTop: 12,
    paddingLeft: 12,
    borderLeftWidth: 2,
    borderLeftColor: '#ddd',
  },
  expiryText: {
    fontSize: 12,
    color: '#999',
    marginTop: 12,
  },
  invitationActions: {
    flexDirection: 'row',
    marginTop: 16,
    gap: 12,
  },
  declineButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ddd',
    alignItems: 'center',
  },
  declineButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#666',
  },
  acceptButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: '#4A90A4',
    alignItems: 'center',
  },
  acceptButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 48,
    paddingHorizontal: 32,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
    lineHeight: 20,
  },
  createClinicButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#4A90A4',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    marginTop: 24,
  },
  createClinicButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginLeft: 8,
  },
  verificationCTA: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  ctaContent: {
    flex: 1,
    marginLeft: 12,
  },
  ctaTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#333',
  },
  ctaSubtitle: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
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
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  modalBody: {
    padding: 16,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
    marginBottom: 6,
    marginTop: 12,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
  },
  modalFooter: {
    flexDirection: 'row',
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    gap: 12,
  },
  cancelButton: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ddd',
    alignItems: 'center',
  },
  cancelButtonText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#666',
  },
  submitButton: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 8,
    backgroundColor: '#4A90A4',
    alignItems: 'center',
  },
  disabledButton: {
    opacity: 0.6,
  },
  submitButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
});
