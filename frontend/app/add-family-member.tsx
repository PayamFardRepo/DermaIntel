import React, { useState } from 'react';
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
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';

const RELATIONSHIP_TYPE_VALUES = ['parent', 'sibling', 'grandparent', 'aunt_uncle', 'cousin', 'child'];
const RELATIONSHIP_SIDE_VALUES = ['maternal', 'paternal', 'own'];
const SKIN_TYPE_VALUES = ['I', 'II', 'III', 'IV', 'V', 'VI'];
const HAIR_COLOR_VALUES = ['blonde', 'red', 'brown', 'black', 'gray', 'other'];
const EYE_COLOR_VALUES = ['blue', 'green', 'hazel', 'brown', 'gray', 'other'];
const GENDER_VALUES = ['male', 'female', 'other'];
const MELANOMA_OUTCOME_VALUES = ['survived', 'deceased', 'unknown'];

export default function AddFamilyMemberScreen() {
  const { t } = useTranslation();
  const { isAuthenticated, logout } = useAuth();
  const router = useRouter();
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

  React.useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    }
  }, [isAuthenticated]);

  const handleSubmit = async () => {
    // Validation
    if (!relationshipType) {
      Alert.alert(t('addFamilyMember.validationError'), t('addFamilyMember.selectRelationship'));
      return;
    }

    setSubmitting(true);

    try {
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert(t('addFamilyMember.authError'), t('addFamilyMember.pleaseLoginAgain'));
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
      formData.append('genetic_testing_done', 'false');
      if (notes) formData.append('notes', notes);

      const response = await fetch(`${API_BASE_URL}/family-history/add`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });

      if (response.ok) {
        Alert.alert(t('addFamilyMember.success'), t('addFamilyMember.addedSuccessfully'), [
          {
            text: t('addFamilyMember.ok'),
            onPress: () => router.back()
          }
        ]);
      } else if (response.status === 401) {
        Alert.alert(t('addFamilyMember.sessionExpired'), t('addFamilyMember.pleaseLoginAgain'));
        logout();
      } else {
        const error = await response.json();
        Alert.alert(t('addFamilyMember.error'), error.detail || t('addFamilyMember.failedToAdd'));
      }
    } catch (error) {
      console.error('Error adding family member:', error);
      Alert.alert(t('addFamilyMember.error'), t('addFamilyMember.networkError'));
    } finally{
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

  return (
    <LinearGradient
      colors={['#1e3a8a', '#3b82f6', '#60a5fa']}
      style={styles.container}
    >
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="white" />
        </TouchableOpacity>
        <Text style={styles.title}>{t('addFamilyMember.title')}</Text>
        <View style={styles.placeholder} />
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.formCard}>
          {/* Basic Information Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>{t('addFamilyMember.basicInfo')}</Text>

            <View style={styles.fieldContainer}>
              <Text style={styles.fieldLabel}>{t('addFamilyMember.name')}</Text>
              <TextInput
                style={styles.textInput}
                value={name}
                onChangeText={setName}
                placeholder={t('addFamilyMember.namePlaceholder')}
                placeholderTextColor="#9ca3af"
              />
            </View>

            {renderSelector(
              t('addFamilyMember.relationship'),
              relationshipType,
              RELATIONSHIP_TYPE_VALUES.map(v => ({ label: t(`addFamilyMember.relationshipTypes.${v}`), value: v })),
              setRelationshipType,
              true
            )}

            {relationshipType && relationshipType !== 'child' && (
              renderSelector(
                t('addFamilyMember.side'),
                relationshipSide,
                RELATIONSHIP_SIDE_VALUES.map(v => ({ label: t(`addFamilyMember.relationshipSides.${v}`), value: v })),
                setRelationshipSide
              )
            )}

            {renderSelector(
              t('addFamilyMember.gender'),
              gender,
              GENDER_VALUES.map(v => ({ label: t(`addFamilyMember.genders.${v}`), value: v })),
              setGender
            )}

            <View style={styles.fieldContainer}>
              <Text style={styles.fieldLabel}>{t('addFamilyMember.yearOfBirth')}</Text>
              <TextInput
                style={styles.textInput}
                value={yearOfBirth}
                onChangeText={setYearOfBirth}
                placeholder={t('addFamilyMember.yearOfBirthPlaceholder')}
                placeholderTextColor="#9ca3af"
                keyboardType="numeric"
              />
            </View>

            <View style={styles.switchContainer}>
              <Text style={styles.fieldLabel}>{t('addFamilyMember.currentlyAlive')}</Text>
              <Switch
                value={isAlive}
                onValueChange={setIsAlive}
                trackColor={{ false: '#d1d5db', true: '#8b5cf6' }}
                thumbColor={isAlive ? '#fff' : '#f3f4f6'}
              />
            </View>

            {!isAlive && (
              <View style={styles.fieldContainer}>
                <Text style={styles.fieldLabel}>{t('addFamilyMember.ageAtDeath')}</Text>
                <TextInput
                  style={styles.textInput}
                  value={ageAtDeath}
                  onChangeText={setAgeAtDeath}
                  placeholder={t('addFamilyMember.ageAtDeathPlaceholder')}
                  placeholderTextColor="#9ca3af"
                  keyboardType="numeric"
                />
              </View>
            )}
          </View>

          {/* Skin Cancer History Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>{t('addFamilyMember.skinCancerHistory')}</Text>

            <View style={styles.switchContainer}>
              <Text style={styles.fieldLabel}>{t('addFamilyMember.hasSkinCancer')}</Text>
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
                  <Text style={styles.fieldLabel}>{t('addFamilyMember.hasMelanoma')}</Text>
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
                      <Text style={styles.fieldLabel}>{t('addFamilyMember.numberOfMelanomas')}</Text>
                      <TextInput
                        style={styles.textInput}
                        value={melanomaCount}
                        onChangeText={setMelanomaCount}
                        placeholder={t('addFamilyMember.numberOfMelanomasPlaceholder')}
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
                  <Text style={styles.fieldLabel}>{t('addFamilyMember.skinCancerDetails')}</Text>
                  <Text style={styles.fieldHint}>
                    {t('addFamilyMember.skinCancerDetailsHint')}
                  </Text>
                  <Text style={styles.fieldExample}>
                    {t('addFamilyMember.skinCancerDetailsExample')}
                  </Text>
                  <TextInput
                    style={[styles.textInput, styles.textArea]}
                    value={skinCancerDetails}
                    onChangeText={setSkinCancerDetails}
                    placeholder={t('addFamilyMember.skinCancerDetailsPlaceholder')}
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
            <Text style={styles.sectionTitle}>{t('addFamilyMember.physicalCharacteristics')}</Text>

            {renderSelector(
              t('addFamilyMember.skinType'),
              skinType,
              SKIN_TYPE_VALUES.map(v => ({ label: t(`addFamilyMember.skinTypes.${v}`), value: v })),
              setSkinType
            )}

            {renderSelector(
              t('addFamilyMember.hairColor'),
              hairColor,
              HAIR_COLOR_VALUES.map(v => ({ label: t(`addFamilyMember.hairColors.${v}`), value: v })),
              setHairColor
            )}

            {renderSelector(
              t('addFamilyMember.eyeColor'),
              eyeColor,
              EYE_COLOR_VALUES.map(v => ({ label: t(`addFamilyMember.eyeColors.${v}`), value: v })),
              setEyeColor
            )}

            <View style={styles.switchContainer}>
              <Text style={styles.fieldLabel}>{t('addFamilyMember.hasManyMoles')}</Text>
              <Switch
                value={hasManyMoles}
                onValueChange={setHasManyMoles}
                trackColor={{ false: '#d1d5db', true: '#f59e0b' }}
                thumbColor={hasManyMoles ? '#fff' : '#f3f4f6'}
              />
            </View>

            <View style={styles.switchContainer}>
              <Text style={styles.fieldLabel}>{t('addFamilyMember.hasAtypicalMoles')}</Text>
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
            <Text style={styles.sectionTitle}>{t('addFamilyMember.additionalNotes')}</Text>

            <View style={styles.fieldContainer}>
              <Text style={styles.fieldLabel}>{t('addFamilyMember.notes')}</Text>
              <TextInput
                style={[styles.textInput, styles.textArea]}
                value={notes}
                onChangeText={setNotes}
                placeholder={t('addFamilyMember.notesPlaceholder')}
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
                <Text style={styles.submitButtonText}>{t('addFamilyMember.submit')}</Text>
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
    backgroundColor: '#10b981',
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
