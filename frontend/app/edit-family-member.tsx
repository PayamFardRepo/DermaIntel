import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Platform,
  Switch
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';

const RELATIONSHIP_TYPES = [
  { label: 'Parent', value: 'parent' },
  { label: 'Sibling', value: 'sibling' },
  { label: 'Grandparent', value: 'grandparent' },
  { label: 'Aunt/Uncle', value: 'aunt_uncle' },
  { label: 'Cousin', value: 'cousin' },
  { label: 'Child', value: 'child' }
];

const RELATIONSHIP_SIDES = [
  { label: 'Maternal', value: 'maternal' },
  { label: 'Paternal', value: 'paternal' },
  { label: 'Own', value: 'own' }
];

const SKIN_TYPES = [
  { label: 'Type I (Very Fair)', value: 'I' },
  { label: 'Type II (Fair)', value: 'II' },
  { label: 'Type III (Medium)', value: 'III' },
  { label: 'Type IV (Olive)', value: 'IV' },
  { label: 'Type V (Brown)', value: 'V' },
  { label: 'Type VI (Dark Brown)', value: 'VI' }
];

const HAIR_COLORS = ['blonde', 'red', 'brown', 'black', 'gray', 'other'];
const EYE_COLORS = ['blue', 'green', 'hazel', 'brown', 'gray', 'other'];
const MELANOMA_OUTCOME_VALUES = ['survived', 'deceased', 'unknown'];

export default function EditFamilyMemberScreen() {
  const { isAuthenticated, logout } = useAuth();
  const router = useRouter();
  const params = useLocalSearchParams();
  const memberId = params.id;

  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  // Basic Information
  const [name, setName] = useState('');
  const [relationshipType, setRelationshipType] = useState('');
  const [relationshipSide, setRelationshipSide] = useState('');
  const [gender, setGender] = useState('');
  const [yearOfBirth, setYearOfBirth] = useState('');
  const [isAlive, setIsAlive] = useState(true);
  const [ageAtDeath, setAgeAtDeath] = useState('');

  // Skin Cancer Information
  const [hasSkinCancer, setHasSkinCancer] = useState(false);
  const [hasMelanoma, setHasMelanoma] = useState(false);
  const [melanomaCount, setMelanomaCount] = useState('0');
  const [melanomaAgeAtDiagnosis, setMelanomaAgeAtDiagnosis] = useState('');
  const [melanomaOutcome, setMelanomaOutcome] = useState('');
  const [skinCancerDetails, setSkinCancerDetails] = useState('');

  // Physical Characteristics
  const [skinType, setSkinType] = useState('');
  const [hairColor, setHairColor] = useState('');
  const [eyeColor, setEyeColor] = useState('');
  const [hasManyMoles, setHasManyMoles] = useState(false);
  const [hasAtypicalMoles, setHasAtypicalMoles] = useState(false);

  // Additional Information
  const [notes, setNotes] = useState('');

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    } else if (memberId) {
      loadFamilyMember();
    }
  }, [isAuthenticated, memberId]);

  const loadFamilyMember = async () => {
    try {
      const token = AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/family-history`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        const member = data.family_members.find((m: any) => m.id === parseInt(memberId as string));

        if (member) {
          setName(member.name || '');
          setRelationshipType(member.relationship_type || '');
          setRelationshipSide(member.relationship_side || '');
          setGender(member.gender || '');
          setYearOfBirth(member.year_of_birth?.toString() || '');
          setIsAlive(member.is_alive);
          setAgeAtDeath(member.age_at_death?.toString() || '');
          setHasSkinCancer(member.has_skin_cancer);
          setHasMelanoma(member.has_melanoma);
          setMelanomaCount(member.melanoma_count?.toString() || '0');
          setMelanomaAgeAtDiagnosis(member.melanoma_age_at_diagnosis?.toString() || '');
          setMelanomaOutcome(member.melanoma_outcome || '');

          // Convert skin cancer types array to string format
          if (member.skin_cancer_types && Array.isArray(member.skin_cancer_types)) {
            const details = member.skin_cancer_types.map((cancer: any) => {
              return `${cancer.type}@${cancer.age_diagnosed || ''}@${cancer.location || ''}`;
            }).join(', ');
            setSkinCancerDetails(details);
          }

          setSkinType(member.skin_type || '');
          setHairColor(member.hair_color || '');
          setEyeColor(member.eye_color || '');
          setHasManyMoles(member.has_many_moles || false);
          setHasAtypicalMoles(member.has_atypical_moles || false);
          setNotes(member.notes || '');
        } else {
          Alert.alert('Error', 'Family member not found');
          router.back();
        }
      }
    } catch (error) {
      console.error('Error loading family member:', error);
      Alert.alert('Error', 'Failed to load family member data');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!relationshipType) {
      Alert.alert('Validation Error', 'Please select a relationship type');
      return;
    }

    setSubmitting(true);

    try {
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert('Authentication Error', 'Please log in again');
        logout();
        return;
      }

      // Prepare skin cancer types array
      const skinCancerTypes = skinCancerDetails
        ? skinCancerDetails.split(',').map(detail => {
            const parts = detail.trim().split('@');
            return {
              type: parts[0] || 'unknown',
              age_diagnosed: parts[1] ? parseInt(parts[1]) : null,
              location: parts[2] || null
            };
          })
        : [];

      const formData = new FormData();
      formData.append('relationship_type', relationshipType);
      if (relationshipSide) formData.append('relationship_side', relationshipSide);
      if (name) formData.append('name', name);
      if (gender) formData.append('gender', gender);
      if (yearOfBirth) formData.append('year_of_birth', yearOfBirth);
      formData.append('is_alive', isAlive.toString());
      if (!isAlive && ageAtDeath) formData.append('age_at_death', ageAtDeath);
      formData.append('has_skin_cancer', hasSkinCancer.toString());
      if (hasSkinCancer && skinCancerTypes.length > 0) {
        formData.append('skin_cancer_types', JSON.stringify(skinCancerTypes));
      }
      formData.append('has_melanoma', hasMelanoma.toString());
      if (hasMelanoma) {
        formData.append('melanoma_count', melanomaCount);
        if (melanomaAgeAtDiagnosis) formData.append('melanoma_age_at_diagnosis', melanomaAgeAtDiagnosis);
        if (melanomaOutcome) formData.append('melanoma_outcome', melanomaOutcome);
      }
      if (skinType) formData.append('skin_type', skinType);
      if (hairColor) formData.append('hair_color', hairColor);
      if (eyeColor) formData.append('eye_color', eyeColor);
      formData.append('has_many_moles', hasManyMoles.toString());
      formData.append('has_atypical_moles', hasAtypicalMoles.toString());
      if (notes) formData.append('notes', notes);

      const response = await fetch(`${API_BASE_URL}/family-history/${memberId}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });

      if (response.ok) {
        Alert.alert('Success', 'Family member updated successfully', [
          {
            text: 'OK',
            onPress: () => router.back()
          }
        ]);
      } else if (response.status === 401) {
        Alert.alert('Session Expired', 'Please log in again');
        logout();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to update family member');
      }
    } catch (error) {
      console.error('Error updating family member:', error);
      Alert.alert('Error', 'Network error. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const renderSelector = (
    label: string,
    value: string,
    options: { label: string; value: string }[],
    onSelect: (value: string) => void,
    required = false
  ) => (
    <View style={styles.fieldContainer}>
      <Text style={styles.fieldLabel}>
        {label} {required && <Text style={styles.required}>*</Text>}
      </Text>
      <View style={styles.optionsContainer}>
        {options.map((option) => (
          <TouchableOpacity
            key={option.value}
            style={[
              styles.optionButton,
              value === option.value && styles.optionButtonSelected
            ]}
            onPress={() => onSelect(option.value)}
          >
            <Text
              style={[
                styles.optionButtonText,
                value === option.value && styles.optionButtonTextSelected
              ]}
            >
              {option.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#8b5cf6" />
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    );
  }

  return (
    <LinearGradient
      colors={['#1e3a8a', '#3b82f6', '#60a5fa']}
      style={styles.container}
    >
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="white" />
        </TouchableOpacity>
        <Text style={styles.title}>Edit Family Member</Text>
        <View style={styles.placeholder} />
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.formCard}>
          {/* Basic Information Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Basic Information</Text>

            <View style={styles.fieldContainer}>
              <Text style={styles.fieldLabel}>Name (Optional)</Text>
              <TextInput
                style={styles.textInput}
                value={name}
                onChangeText={setName}
                placeholder="e.g., Mom, Dad, Sister Jane"
                placeholderTextColor="#9ca3af"
              />
            </View>

            {renderSelector(
              'Relationship',
              relationshipType,
              RELATIONSHIP_TYPES,
              setRelationshipType,
              true
            )}

            {relationshipType && relationshipType !== 'child' && (
              renderSelector(
                'Side',
                relationshipSide,
                RELATIONSHIP_SIDES,
                setRelationshipSide
              )
            )}

            {renderSelector(
              'Gender',
              gender,
              [
                { label: 'Male', value: 'male' },
                { label: 'Female', value: 'female' },
                { label: 'Other', value: 'other' }
              ],
              setGender
            )}

            <View style={styles.fieldContainer}>
              <Text style={styles.fieldLabel}>Year of Birth</Text>
              <TextInput
                style={styles.textInput}
                value={yearOfBirth}
                onChangeText={setYearOfBirth}
                placeholder="e.g., 1960"
                placeholderTextColor="#9ca3af"
                keyboardType="numeric"
              />
            </View>

            <View style={styles.switchContainer}>
              <Text style={styles.fieldLabel}>Currently Alive</Text>
              <Switch
                value={isAlive}
                onValueChange={setIsAlive}
                trackColor={{ false: '#d1d5db', true: '#8b5cf6' }}
                thumbColor={isAlive ? '#fff' : '#f3f4f6'}
              />
            </View>

            {!isAlive && (
              <View style={styles.fieldContainer}>
                <Text style={styles.fieldLabel}>Age at Death</Text>
                <TextInput
                  style={styles.textInput}
                  value={ageAtDeath}
                  onChangeText={setAgeAtDeath}
                  placeholder="e.g., 75"
                  placeholderTextColor="#9ca3af"
                  keyboardType="numeric"
                />
              </View>
            )}
          </View>

          {/* Skin Cancer History Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Skin Cancer History</Text>

            <View style={styles.switchContainer}>
              <Text style={styles.fieldLabel}>Has Skin Cancer</Text>
              <Switch
                value={hasSkinCancer}
                onValueChange={setHasSkinCancer}
                trackColor={{ false: '#d1d5db', true: '#dc2626' }}
                thumbColor={hasSkinCancer ? '#fff' : '#f3f4f6'}
              />
            </View>

            {hasSkinCancer && (
              <>
                <View style={styles.switchContainer}>
                  <Text style={styles.fieldLabel}>Has Melanoma</Text>
                  <Switch
                    value={hasMelanoma}
                    onValueChange={setHasMelanoma}
                    trackColor={{ false: '#d1d5db', true: '#dc2626' }}
                    thumbColor={hasMelanoma ? '#fff' : '#f3f4f6'}
                  />
                </View>

                {hasMelanoma && (
                  <>
                    <View style={styles.fieldContainer}>
                      <Text style={styles.fieldLabel}>Number of Melanomas</Text>
                      <TextInput
                        style={styles.textInput}
                        value={melanomaCount}
                        onChangeText={setMelanomaCount}
                        placeholder="e.g., 1"
                        placeholderTextColor="#9ca3af"
                        keyboardType="numeric"
                      />
                    </View>

                    <View style={styles.fieldContainer}>
                      <Text style={styles.fieldLabel}>Age at Diagnosis</Text>
                      <TextInput
                        style={styles.textInput}
                        value={melanomaAgeAtDiagnosis}
                        onChangeText={setMelanomaAgeAtDiagnosis}
                        placeholder="Age when melanoma was diagnosed"
                        placeholderTextColor="#9ca3af"
                        keyboardType="numeric"
                      />
                    </View>

                    {renderSelector(
                      'Outcome',
                      melanomaOutcome,
                      MELANOMA_OUTCOME_VALUES.map(v => ({
                        label: v === 'survived' ? 'Survived' : v === 'deceased' ? 'Deceased' : 'Unknown',
                        value: v
                      })),
                      setMelanomaOutcome
                    )}
                  </>
                )}

                <View style={styles.fieldContainer}>
                  <Text style={styles.fieldLabel}>Skin Cancer Details</Text>
                  <Text style={styles.fieldHint}>
                    Format: Type@Age@Location, separated by commas
                  </Text>
                  <Text style={styles.fieldExample}>
                    Example: melanoma@45@back, BCC@60@face
                  </Text>
                  <TextInput
                    style={[styles.textInput, styles.textArea]}
                    value={skinCancerDetails}
                    onChangeText={setSkinCancerDetails}
                    placeholder="melanoma@45@back"
                    placeholderTextColor="#9ca3af"
                    multiline
                    numberOfLines={3}
                  />
                </View>
              </>
            )}
          </View>

          {/* Physical Characteristics Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Physical Characteristics</Text>

            {renderSelector('Skin Type', skinType, SKIN_TYPES, setSkinType)}

            {renderSelector(
              'Hair Color',
              hairColor,
              HAIR_COLORS.map(c => ({ label: c.charAt(0).toUpperCase() + c.slice(1), value: c })),
              setHairColor
            )}

            {renderSelector(
              'Eye Color',
              eyeColor,
              EYE_COLORS.map(c => ({ label: c.charAt(0).toUpperCase() + c.slice(1), value: c })),
              setEyeColor
            )}

            <View style={styles.switchContainer}>
              <Text style={styles.fieldLabel}>Has Many Moles (&gt;50)</Text>
              <Switch
                value={hasManyMoles}
                onValueChange={setHasManyMoles}
                trackColor={{ false: '#d1d5db', true: '#f59e0b' }}
                thumbColor={hasManyMoles ? '#fff' : '#f3f4f6'}
              />
            </View>

            <View style={styles.switchContainer}>
              <Text style={styles.fieldLabel}>Has Atypical/Dysplastic Moles</Text>
              <Switch
                value={hasAtypicalMoles}
                onValueChange={setHasAtypicalMoles}
                trackColor={{ false: '#d1d5db', true: '#f59e0b' }}
                thumbColor={hasAtypicalMoles ? '#fff' : '#f3f4f6'}
              />
            </View>
          </View>

          {/* Additional Notes Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Additional Notes</Text>

            <View style={styles.fieldContainer}>
              <Text style={styles.fieldLabel}>Notes</Text>
              <TextInput
                style={[styles.textInput, styles.textArea]}
                value={notes}
                onChangeText={setNotes}
                placeholder="Any additional medical information..."
                placeholderTextColor="#9ca3af"
                multiline
                numberOfLines={4}
              />
            </View>
          </View>

          {/* Submit Button */}
          <TouchableOpacity
            style={[styles.submitButton, submitting && styles.submitButtonDisabled]}
            onPress={handleSubmit}
            disabled={submitting}
          >
            {submitting ? (
              <ActivityIndicator color="white" />
            ) : (
              <>
                <Ionicons name="checkmark-circle" size={24} color="white" />
                <Text style={styles.submitButtonText}>Update Family Member</Text>
              </>
            )}
          </TouchableOpacity>
        </View>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1e3a8a',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: 'white',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 16,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.2)',
  },
  backButton: {
    padding: 8,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    flex: 1,
    textAlign: 'center',
  },
  placeholder: {
    width: 40,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  formCard: {
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
    borderBottomWidth: 2,
    borderBottomColor: '#8b5cf6',
    paddingBottom: 8,
  },
  fieldContainer: {
    marginBottom: 16,
  },
  fieldLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  required: {
    color: '#dc2626',
  },
  fieldHint: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4,
  },
  fieldExample: {
    fontSize: 12,
    color: '#8b5cf6',
    fontStyle: 'italic',
    marginBottom: 8,
  },
  textInput: {
    backgroundColor: '#f9fafb',
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    color: '#1f2937',
  },
  textArea: {
    height: 80,
    textAlignVertical: 'top',
  },
  optionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  optionButton: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#d1d5db',
    backgroundColor: '#f9fafb',
  },
  optionButtonSelected: {
    borderColor: '#8b5cf6',
    backgroundColor: '#ede9fe',
  },
  optionButtonText: {
    fontSize: 14,
    color: '#6b7280',
    fontWeight: '500',
  },
  optionButtonTextSelected: {
    color: '#7c3aed',
    fontWeight: '600',
  },
  switchContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  submitButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#3b82f6',
    padding: 16,
    borderRadius: 12,
    gap: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 5,
    marginTop: 8,
  },
  submitButtonDisabled: {
    opacity: 0.6,
  },
  submitButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
