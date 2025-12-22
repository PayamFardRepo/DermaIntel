import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
  Alert,
  ActivityIndicator,
  Modal,
  FlatList,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useUserSettings } from '../contexts/UserSettingsContext';

const US_STATES = [
  'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
  'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
  'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
  'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
  'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
  'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
  'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia',
  'Wisconsin', 'Wyoming', 'District of Columbia'
];

export default function ProfessionalVerification() {
  const router = useRouter();
  const { settings, submitProfessionalVerification, refreshSettings } = useUserSettings();

  const [licenseNumber, setLicenseNumber] = useState(settings.professionalLicenseNumber || '');
  const [licenseState, setLicenseState] = useState(settings.professionalLicenseState || '');
  const [npiNumber, setNpiNumber] = useState(settings.npiNumber || '');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showStatePicker, setShowStatePicker] = useState(false);

  const handleSubmit = async () => {
    if (!licenseNumber.trim()) {
      Alert.alert('Error', 'Please enter your medical license number');
      return;
    }
    if (!licenseState) {
      Alert.alert('Error', 'Please select the state where your license was issued');
      return;
    }

    setIsSubmitting(true);
    try {
      const result = await submitProfessionalVerification(
        licenseNumber.trim(),
        licenseState,
        npiNumber.trim() || undefined
      );

      if (result.success) {
        Alert.alert(
          'Verification Submitted',
          'Your professional verification request has been submitted. You will be notified once your credentials are verified.',
          [{ text: 'OK', onPress: () => router.back() }]
        );
      } else {
        Alert.alert('Error', result.message);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to submit verification request');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color="#333" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Professional Verification</Text>
      </View>

      <ScrollView style={styles.content}>
        {/* Status Banner */}
        {settings.isVerifiedProfessional ? (
          <View style={styles.verifiedBanner}>
            <Ionicons name="checkmark-circle" size={32} color="#2E7D32" />
            <View style={styles.bannerContent}>
              <Text style={styles.bannerTitle}>Verified Healthcare Professional</Text>
              <Text style={styles.bannerSubtitle}>
                Your credentials have been verified. You have full access to professional features.
              </Text>
            </View>
          </View>
        ) : settings.professionalLicenseNumber ? (
          <View style={styles.pendingBanner}>
            <Ionicons name="time" size={32} color="#F57C00" />
            <View style={styles.bannerContent}>
              <Text style={styles.bannerTitle}>Verification Pending</Text>
              <Text style={styles.bannerSubtitle}>
                Your verification request is being reviewed. This typically takes 1-2 business days.
              </Text>
            </View>
          </View>
        ) : null}

        {/* Information Section */}
        <View style={styles.infoSection}>
          <Ionicons name="information-circle" size={24} color="#4A90A4" />
          <Text style={styles.infoText}>
            Healthcare professional verification enables you to create and manage clinics,
            access detailed clinical data, and receive patient-shared analyses.
          </Text>
        </View>

        {/* Form Section */}
        <View style={styles.formSection}>
          <Text style={styles.sectionTitle}>Credentials</Text>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Medical License Number *</Text>
            <TextInput
              style={styles.input}
              placeholder="Enter your license number"
              value={licenseNumber}
              onChangeText={setLicenseNumber}
              autoCapitalize="characters"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>License State *</Text>
            <TouchableOpacity
              style={styles.pickerButton}
              onPress={() => setShowStatePicker(true)}
            >
              <Text style={licenseState ? styles.pickerText : styles.pickerPlaceholder}>
                {licenseState || 'Select state'}
              </Text>
              <Ionicons name="chevron-down" size={20} color="#666" />
            </TouchableOpacity>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>NPI Number (Optional)</Text>
            <TextInput
              style={styles.input}
              placeholder="Enter your NPI number"
              value={npiNumber}
              onChangeText={setNpiNumber}
              keyboardType="numeric"
              maxLength={10}
            />
            <Text style={styles.helperText}>
              National Provider Identifier - 10 digit number
            </Text>
          </View>
        </View>

        {/* Benefits Section */}
        <View style={styles.benefitsSection}>
          <Text style={styles.sectionTitle}>Benefits of Verification</Text>

          <View style={styles.benefitItem}>
            <Ionicons name="business" size={20} color="#4A90A4" />
            <Text style={styles.benefitText}>Create and manage your own clinics</Text>
          </View>

          <View style={styles.benefitItem}>
            <Ionicons name="people" size={20} color="#4A90A4" />
            <Text style={styles.benefitText}>Invite patients to share their analyses</Text>
          </View>

          <View style={styles.benefitItem}>
            <Ionicons name="analytics" size={20} color="#4A90A4" />
            <Text style={styles.benefitText}>Access detailed ABCDE analysis and clinical metrics</Text>
          </View>

          <View style={styles.benefitItem}>
            <Ionicons name="document-text" size={20} color="#4A90A4" />
            <Text style={styles.benefitText}>Add clinical notes to patient records</Text>
          </View>

          <View style={styles.benefitItem}>
            <Ionicons name="shield-checkmark" size={20} color="#4A90A4" />
            <Text style={styles.benefitText}>Verified badge on your profile</Text>
          </View>
        </View>

        {/* Submit Button */}
        {!settings.isVerifiedProfessional && (
          <TouchableOpacity
            style={[styles.submitButton, isSubmitting && styles.disabledButton]}
            onPress={handleSubmit}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <>
                <Ionicons name="shield-checkmark" size={20} color="#fff" />
                <Text style={styles.submitButtonText}>
                  {settings.professionalLicenseNumber ? 'Update Verification' : 'Submit for Verification'}
                </Text>
              </>
            )}
          </TouchableOpacity>
        )}

        {/* Disclaimer */}
        <Text style={styles.disclaimer}>
          By submitting your credentials, you confirm that you are a licensed healthcare
          professional and that the information provided is accurate. False claims may
          result in account suspension.
        </Text>
      </ScrollView>

      {/* State Picker Modal */}
      <Modal
        visible={showStatePicker}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowStatePicker(false)}
      >
        <View style={styles.pickerModalOverlay}>
          <View style={styles.pickerModal}>
            <View style={styles.pickerHeader}>
              <TouchableOpacity onPress={() => setShowStatePicker(false)}>
                <Text style={styles.pickerCancel}>Cancel</Text>
              </TouchableOpacity>
              <Text style={styles.pickerTitle}>Select State</Text>
              <TouchableOpacity onPress={() => setShowStatePicker(false)}>
                <Text style={styles.pickerDone}>Done</Text>
              </TouchableOpacity>
            </View>
            <FlatList
              data={US_STATES}
              keyExtractor={(item) => item}
              style={styles.stateList}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={[
                    styles.stateItem,
                    licenseState === item && styles.stateItemSelected
                  ]}
                  onPress={() => {
                    setLicenseState(item);
                    setShowStatePicker(false);
                  }}
                >
                  <Text style={[
                    styles.stateItemText,
                    licenseState === item && styles.stateItemTextSelected
                  ]}>
                    {item}
                  </Text>
                  {licenseState === item && (
                    <Ionicons name="checkmark" size={20} color="#4A90A4" />
                  )}
                </TouchableOpacity>
              )}
            />
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
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  verifiedBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#e8f5e9',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  pendingBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff3e0',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  bannerContent: {
    flex: 1,
    marginLeft: 12,
  },
  bannerTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  bannerSubtitle: {
    fontSize: 13,
    color: '#666',
    marginTop: 4,
  },
  infoSection: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#e3f2fd',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
  },
  infoText: {
    flex: 1,
    fontSize: 14,
    color: '#1976D2',
    marginLeft: 12,
    lineHeight: 20,
  },
  formSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 16,
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
    marginBottom: 6,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    backgroundColor: '#fff',
  },
  pickerButton: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    backgroundColor: '#fff',
  },
  pickerText: {
    fontSize: 16,
    color: '#333',
  },
  pickerPlaceholder: {
    fontSize: 16,
    color: '#999',
  },
  helperText: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  benefitsSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  benefitItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  benefitText: {
    fontSize: 14,
    color: '#333',
    marginLeft: 12,
    flex: 1,
  },
  submitButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#4A90A4',
    paddingVertical: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  disabledButton: {
    opacity: 0.6,
  },
  submitButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginLeft: 8,
  },
  disclaimer: {
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
    lineHeight: 18,
    marginBottom: 32,
  },
  pickerModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  pickerModal: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '70%',
  },
  pickerHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  pickerTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  pickerCancel: {
    fontSize: 16,
    color: '#666',
  },
  pickerDone: {
    fontSize: 16,
    color: '#4A90A4',
    fontWeight: '600',
  },
  stateList: {
    maxHeight: 400,
  },
  stateItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  stateItemSelected: {
    backgroundColor: '#e3f2fd',
  },
  stateItemText: {
    fontSize: 16,
    color: '#333',
  },
  stateItemTextSelected: {
    color: '#4A90A4',
    fontWeight: '500',
  },
});
