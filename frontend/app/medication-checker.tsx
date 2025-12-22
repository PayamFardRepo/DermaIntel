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
  Platform,
  StatusBar,
  Modal,
  Switch
} from 'react-native';
import { router } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import { API_BASE_URL } from '../config';

interface Medication {
  name: string;
  category: string;
  common_uses: string[];
}

interface DrugInteraction {
  drug1: string;
  drug2: string;
  severity: string;
  interaction_type: string;
  description: string;
  mechanism: string;
  clinical_effects: string[];
  management: string;
}

interface Contraindication {
  drug: string;
  condition: string;
  severity: string;
  description: string;
  reason: string;
  alternatives: string[];
}

interface PhotosensitivityWarning {
  drug: string;
  type: string;
  severity: string;
  onset_timeframe: string;
  duration_after_stopping: string;
  uva_uvb_sensitivity: string;
  clinical_presentation: string;
  precautions: string[];
  spf_recommendation: number;
}

interface AgeWarning {
  drug: string;
  min_age?: number;
  max_age?: number;
  severity: string;
  pediatric_concerns: string[];
  geriatric_concerns: string[];
  dose_adjustment: string;
}

interface PregnancyWarning {
  drug: string;
  pregnancy_category: string;
  severity: string;
  first_trimester_risk: string;
  second_trimester_risk: string;
  third_trimester_risk: string;
  lactation_safe: boolean;
  lactation_notes: string;
  alternatives_during_pregnancy: string[];
  alternatives_during_lactation: string[];
}

interface CheckResult {
  medication: string;
  is_safe: boolean;
  overall_risk_level: string;
  drug_interactions: DrugInteraction[];
  contraindications: Contraindication[];
  photosensitivity_warnings: PhotosensitivityWarning[];
  age_warnings: AgeWarning[];
  pregnancy_warnings: PregnancyWarning[];
  dosage_issues: any[];
  recommendations: string[];
  requires_provider_review: boolean;
}

export default function MedicationCheckerScreen() {
  const { user } = useAuth();
  const { t } = useTranslation();

  // Form state
  const [medication, setMedication] = useState('');
  const [currentMedications, setCurrentMedications] = useState<string[]>([]);
  const [newMedication, setNewMedication] = useState('');
  const [patientConditions, setPatientConditions] = useState<string[]>([]);
  const [newCondition, setNewCondition] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [isPregnant, setIsPregnant] = useState(false);
  const [isBreastfeeding, setIsBreastfeeding] = useState(false);
  const [dose, setDose] = useState('');
  const [frequency, setFrequency] = useState('');
  const [renalFunction, setRenalFunction] = useState('normal');
  const [hepaticFunction, setHepaticFunction] = useState('normal');
  const [sunExposure, setSunExposure] = useState('moderate');

  // Results state
  const [checkResult, setCheckResult] = useState<CheckResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Common medications
  const [commonMedications, setCommonMedications] = useState<Medication[]>([]);
  const [showMedicationPicker, setShowMedicationPicker] = useState(false);
  const [medicationFilter, setMedicationFilter] = useState('');

  // Detail modals
  const [showInteractionDetail, setShowInteractionDetail] = useState(false);
  const [selectedInteraction, setSelectedInteraction] = useState<DrugInteraction | null>(null);
  const [showPhotoDetail, setShowPhotoDetail] = useState(false);
  const [selectedPhotoWarning, setSelectedPhotoWarning] = useState<PhotosensitivityWarning | null>(null);

  useEffect(() => {
    loadCommonMedications();
  }, []);

  const loadCommonMedications = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/medications/common-dermatological`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setCommonMedications(data.medications || []);
      }
    } catch (error) {
      console.log('Error loading medications:', error);
    }
  };

  const addCurrentMedication = () => {
    if (newMedication.trim() && !currentMedications.includes(newMedication.trim().toLowerCase())) {
      setCurrentMedications([...currentMedications, newMedication.trim().toLowerCase()]);
      setNewMedication('');
    }
  };

  const removeCurrentMedication = (med: string) => {
    setCurrentMedications(currentMedications.filter(m => m !== med));
  };

  const addCondition = () => {
    if (newCondition.trim() && !patientConditions.includes(newCondition.trim().toLowerCase())) {
      setPatientConditions([...patientConditions, newCondition.trim().toLowerCase()]);
      setNewCondition('');
    }
  };

  const removeCondition = (condition: string) => {
    setPatientConditions(patientConditions.filter(c => c !== condition));
  };

  const checkInteractions = async () => {
    if (!medication.trim()) {
      Alert.alert('Required', 'Please enter a medication to check');
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('medication', medication.trim());
      formData.append('current_medications', JSON.stringify(currentMedications));
      formData.append('patient_conditions', JSON.stringify(patientConditions));
      if (patientAge) formData.append('patient_age', patientAge);
      formData.append('is_pregnant', isPregnant.toString());
      formData.append('is_breastfeeding', isBreastfeeding.toString());
      if (dose) formData.append('dose', dose);
      if (frequency) formData.append('frequency', frequency);
      formData.append('renal_function', renalFunction);
      formData.append('hepatic_function', hepaticFunction);
      formData.append('sun_exposure_level', sunExposure);

      const response = await fetch(`${API_BASE_URL}/medications/check-interaction`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user?.token}` },
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        setCheckResult(data);
      } else {
        Alert.alert('Error', 'Failed to check medication');
      }
    } catch (error) {
      console.error('Error checking medication:', error);
      Alert.alert('Error', 'Failed to check medication interactions');
    } finally {
      setIsLoading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'contraindicated': return '#D32F2F';
      case 'severe': return '#F44336';
      case 'moderate': return '#FF9800';
      case 'mild': return '#FFC107';
      case 'info': return '#2196F3';
      default: return '#9E9E9E';
    }
  };

  const getSeverityBackground = (severity: string) => {
    switch (severity) {
      case 'contraindicated': return '#FFEBEE';
      case 'severe': return '#FFEBEE';
      case 'moderate': return '#FFF3E0';
      case 'mild': return '#FFFDE7';
      case 'info': return '#E3F2FD';
      default: return '#F5F5F5';
    }
  };

  const selectMedication = (med: Medication) => {
    setMedication(med.name);
    setShowMedicationPicker(false);
  };

  const filteredMedications = commonMedications.filter(med =>
    med.name.toLowerCase().includes(medicationFilter.toLowerCase()) ||
    med.category.toLowerCase().includes(medicationFilter.toLowerCase())
  );

  const renderForm = () => (
    <ScrollView style={styles.formContainer}>
      <Text style={styles.sectionTitle}>Medication to Check</Text>
      <View style={styles.medicationInputRow}>
        <TextInput
          style={[styles.input, { flex: 1 }]}
          value={medication}
          onChangeText={setMedication}
          placeholder="Enter medication name"
          placeholderTextColor="#999"
        />
        <Pressable
          style={styles.browseButton}
          onPress={() => setShowMedicationPicker(true)}
        >
          <Text style={styles.browseButtonText}>Browse</Text>
        </Pressable>
      </View>

      <View style={styles.row}>
        <View style={styles.halfWidth}>
          <Text style={styles.label}>Dose</Text>
          <TextInput
            style={styles.input}
            value={dose}
            onChangeText={setDose}
            placeholder="e.g., 100mg"
            placeholderTextColor="#999"
          />
        </View>
        <View style={styles.halfWidth}>
          <Text style={styles.label}>Frequency</Text>
          <TextInput
            style={styles.input}
            value={frequency}
            onChangeText={setFrequency}
            placeholder="e.g., twice daily"
            placeholderTextColor="#999"
          />
        </View>
      </View>

      <Text style={styles.sectionTitle}>Current Medications</Text>
      <View style={styles.tagInputRow}>
        <TextInput
          style={[styles.input, { flex: 1 }]}
          value={newMedication}
          onChangeText={setNewMedication}
          placeholder="Add current medication"
          placeholderTextColor="#999"
          onSubmitEditing={addCurrentMedication}
        />
        <Pressable style={styles.addButton} onPress={addCurrentMedication}>
          <Text style={styles.addButtonText}>+</Text>
        </Pressable>
      </View>
      <View style={styles.tagsContainer}>
        {currentMedications.map((med, index) => (
          <View key={index} style={styles.tag}>
            <Text style={styles.tagText}>{med}</Text>
            <Pressable onPress={() => removeCurrentMedication(med)}>
              <Text style={styles.tagRemove}>x</Text>
            </Pressable>
          </View>
        ))}
      </View>

      <Text style={styles.sectionTitle}>Patient Conditions</Text>
      <View style={styles.tagInputRow}>
        <TextInput
          style={[styles.input, { flex: 1 }]}
          value={newCondition}
          onChangeText={setNewCondition}
          placeholder="Add condition (e.g., liver disease)"
          placeholderTextColor="#999"
          onSubmitEditing={addCondition}
        />
        <Pressable style={styles.addButton} onPress={addCondition}>
          <Text style={styles.addButtonText}>+</Text>
        </Pressable>
      </View>
      <View style={styles.tagsContainer}>
        {patientConditions.map((condition, index) => (
          <View key={index} style={[styles.tag, { backgroundColor: '#E3F2FD' }]}>
            <Text style={[styles.tagText, { color: '#1976D2' }]}>{condition}</Text>
            <Pressable onPress={() => removeCondition(condition)}>
              <Text style={[styles.tagRemove, { color: '#1976D2' }]}>x</Text>
            </Pressable>
          </View>
        ))}
      </View>

      <Text style={styles.sectionTitle}>Patient Profile</Text>

      <Text style={styles.label}>Age</Text>
      <TextInput
        style={styles.input}
        value={patientAge}
        onChangeText={setPatientAge}
        placeholder="Enter age"
        placeholderTextColor="#999"
        keyboardType="numeric"
      />

      <View style={styles.switchRow}>
        <Text style={styles.switchLabel}>Pregnant</Text>
        <Switch
          value={isPregnant}
          onValueChange={setIsPregnant}
          trackColor={{ false: '#E0E0E0', true: '#81C784' }}
        />
      </View>

      <View style={styles.switchRow}>
        <Text style={styles.switchLabel}>Breastfeeding</Text>
        <Switch
          value={isBreastfeeding}
          onValueChange={setIsBreastfeeding}
          trackColor={{ false: '#E0E0E0', true: '#81C784' }}
        />
      </View>

      <Text style={styles.label}>Kidney Function</Text>
      <View style={styles.optionsRow}>
        {['normal', 'mild', 'moderate', 'severe'].map(option => (
          <Pressable
            key={option}
            style={[
              styles.optionButton,
              renalFunction === option && styles.optionButtonSelected
            ]}
            onPress={() => setRenalFunction(option)}
          >
            <Text style={[
              styles.optionButtonText,
              renalFunction === option && styles.optionButtonTextSelected
            ]}>
              {option.charAt(0).toUpperCase() + option.slice(1)}
            </Text>
          </Pressable>
        ))}
      </View>

      <Text style={styles.label}>Liver Function</Text>
      <View style={styles.optionsRow}>
        {['normal', 'mild', 'moderate', 'severe'].map(option => (
          <Pressable
            key={option}
            style={[
              styles.optionButton,
              hepaticFunction === option && styles.optionButtonSelected
            ]}
            onPress={() => setHepaticFunction(option)}
          >
            <Text style={[
              styles.optionButtonText,
              hepaticFunction === option && styles.optionButtonTextSelected
            ]}>
              {option.charAt(0).toUpperCase() + option.slice(1)}
            </Text>
          </Pressable>
        ))}
      </View>

      <Text style={styles.label}>Sun Exposure Level</Text>
      <View style={styles.optionsRow}>
        {['minimal', 'moderate', 'high', 'very_high'].map(option => (
          <Pressable
            key={option}
            style={[
              styles.optionButton,
              sunExposure === option && styles.optionButtonSelected
            ]}
            onPress={() => setSunExposure(option)}
          >
            <Text style={[
              styles.optionButtonText,
              sunExposure === option && styles.optionButtonTextSelected
            ]}>
              {option.replace('_', ' ')}
            </Text>
          </Pressable>
        ))}
      </View>

      <Pressable
        style={[styles.checkButton, isLoading && styles.checkButtonDisabled]}
        onPress={checkInteractions}
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.checkButtonText}>Check Medication Safety</Text>
        )}
      </Pressable>
    </ScrollView>
  );

  const renderResults = () => {
    if (!checkResult) return null;

    const totalWarnings =
      checkResult.drug_interactions.length +
      checkResult.contraindications.length +
      checkResult.photosensitivity_warnings.length +
      checkResult.age_warnings.length +
      checkResult.pregnancy_warnings.length +
      checkResult.dosage_issues.length;

    return (
      <ScrollView style={styles.resultsContainer}>
        {/* Safety Summary */}
        <View style={[
          styles.summaryCard,
          { backgroundColor: getSeverityBackground(checkResult.overall_risk_level) }
        ]}>
          <View style={styles.summaryHeader}>
            <Text style={[
              styles.summaryTitle,
              { color: getSeverityColor(checkResult.overall_risk_level) }
            ]}>
              {checkResult.is_safe ? 'Generally Safe' : 'Caution Required'}
            </Text>
            <View style={[
              styles.severityBadge,
              { backgroundColor: getSeverityColor(checkResult.overall_risk_level) }
            ]}>
              <Text style={styles.severityBadgeText}>
                {checkResult.overall_risk_level.toUpperCase()}
              </Text>
            </View>
          </View>
          <Text style={styles.summarySubtitle}>
            {checkResult.medication} - {totalWarnings} warning{totalWarnings !== 1 ? 's' : ''} found
          </Text>
          {checkResult.requires_provider_review && (
            <View style={styles.providerReviewBanner}>
              <Text style={styles.providerReviewText}>
                Provider review recommended before prescribing
              </Text>
            </View>
          )}
        </View>

        {/* Drug Interactions */}
        {checkResult.drug_interactions.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionHeader}>Drug Interactions</Text>
            {checkResult.drug_interactions.map((interaction, index) => (
              <Pressable
                key={index}
                style={[
                  styles.warningCard,
                  { borderLeftColor: getSeverityColor(interaction.severity) }
                ]}
                onPress={() => {
                  setSelectedInteraction(interaction);
                  setShowInteractionDetail(true);
                }}
              >
                <View style={styles.warningHeader}>
                  <Text style={styles.warningDrugs}>
                    {interaction.drug1} + {interaction.drug2}
                  </Text>
                  <View style={[
                    styles.severityPill,
                    { backgroundColor: getSeverityColor(interaction.severity) }
                  ]}>
                    <Text style={styles.severityPillText}>{interaction.severity}</Text>
                  </View>
                </View>
                <Text style={styles.warningDescription}>{interaction.description}</Text>
                <Text style={styles.tapForMore}>Tap for details</Text>
              </Pressable>
            ))}
          </View>
        )}

        {/* Contraindications */}
        {checkResult.contraindications.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionHeader}>Contraindications</Text>
            {checkResult.contraindications.map((contra, index) => (
              <View
                key={index}
                style={[
                  styles.warningCard,
                  { borderLeftColor: getSeverityColor(contra.severity) }
                ]}
              >
                <View style={styles.warningHeader}>
                  <Text style={styles.warningCondition}>{contra.condition}</Text>
                  <View style={[
                    styles.severityPill,
                    { backgroundColor: getSeverityColor(contra.severity) }
                  ]}>
                    <Text style={styles.severityPillText}>{contra.severity}</Text>
                  </View>
                </View>
                <Text style={styles.warningDescription}>{contra.description}</Text>
                {contra.alternatives.length > 0 && (
                  <View style={styles.alternativesBox}>
                    <Text style={styles.alternativesTitle}>Alternatives:</Text>
                    <Text style={styles.alternativesText}>
                      {contra.alternatives.join(', ')}
                    </Text>
                  </View>
                )}
              </View>
            ))}
          </View>
        )}

        {/* Photosensitivity Warnings */}
        {checkResult.photosensitivity_warnings.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionHeader}>Sun Sensitivity Warnings</Text>
            {checkResult.photosensitivity_warnings.map((warning, index) => (
              <Pressable
                key={index}
                style={[
                  styles.warningCard,
                  { borderLeftColor: '#FF9800', backgroundColor: '#FFF8E1' }
                ]}
                onPress={() => {
                  setSelectedPhotoWarning(warning);
                  setShowPhotoDetail(true);
                }}
              >
                <View style={styles.sunWarningHeader}>
                  <Text style={styles.sunIcon}>sun</Text>
                  <View>
                    <Text style={styles.photoType}>{warning.type}</Text>
                    <Text style={styles.spfRecommendation}>
                      SPF {warning.spf_recommendation}+ recommended
                    </Text>
                  </View>
                </View>
                <Text style={styles.warningDescription}>{warning.clinical_presentation}</Text>
                <Text style={styles.tapForMore}>Tap for precautions</Text>
              </Pressable>
            ))}
          </View>
        )}

        {/* Age Warnings */}
        {checkResult.age_warnings.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionHeader}>Age-Related Warnings</Text>
            {checkResult.age_warnings.map((warning, index) => (
              <View
                key={index}
                style={[
                  styles.warningCard,
                  { borderLeftColor: '#9C27B0' }
                ]}
              >
                {warning.pediatric_concerns.length > 0 && (
                  <View style={styles.concernBox}>
                    <Text style={styles.concernTitle}>Pediatric Concerns:</Text>
                    {warning.pediatric_concerns.map((concern, i) => (
                      <Text key={i} style={styles.concernItem}>  {concern}</Text>
                    ))}
                  </View>
                )}
                {warning.geriatric_concerns.length > 0 && (
                  <View style={styles.concernBox}>
                    <Text style={styles.concernTitle}>Geriatric Concerns:</Text>
                    {warning.geriatric_concerns.map((concern, i) => (
                      <Text key={i} style={styles.concernItem}>  {concern}</Text>
                    ))}
                  </View>
                )}
                <Text style={styles.doseAdjustment}>{warning.dose_adjustment}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Pregnancy Warnings */}
        {checkResult.pregnancy_warnings.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionHeader}>Pregnancy/Lactation</Text>
            {checkResult.pregnancy_warnings.map((warning, index) => (
              <View
                key={index}
                style={[
                  styles.warningCard,
                  { borderLeftColor: '#E91E63' }
                ]}
              >
                <View style={styles.pregnancyHeader}>
                  <Text style={styles.pregnancyCategory}>
                    Category {warning.pregnancy_category}
                  </Text>
                  <View style={[
                    styles.lactationBadge,
                    { backgroundColor: warning.lactation_safe ? '#4CAF50' : '#F44336' }
                  ]}>
                    <Text style={styles.lactationBadgeText}>
                      Lactation: {warning.lactation_safe ? 'Safe' : 'Avoid'}
                    </Text>
                  </View>
                </View>
                <View style={styles.trimesterBox}>
                  <Text style={styles.trimesterLabel}>1st Trimester:</Text>
                  <Text style={styles.trimesterRisk}>{warning.first_trimester_risk}</Text>
                </View>
                {warning.alternatives_during_pregnancy.length > 0 && (
                  <View style={styles.alternativesBox}>
                    <Text style={styles.alternativesTitle}>Alternatives during pregnancy:</Text>
                    <Text style={styles.alternativesText}>
                      {warning.alternatives_during_pregnancy.join(', ')}
                    </Text>
                  </View>
                )}
              </View>
            ))}
          </View>
        )}

        {/* Dosage Issues */}
        {checkResult.dosage_issues.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionHeader}>Dosage Considerations</Text>
            {checkResult.dosage_issues.map((issue, index) => (
              <View
                key={index}
                style={[
                  styles.warningCard,
                  {
                    borderLeftColor: issue.severity === 'critical_error' ? '#D32F2F' : '#FF9800'
                  }
                ]}
              >
                <Text style={[
                  styles.dosageIssueText,
                  issue.severity === 'critical_error' && { color: '#D32F2F', fontWeight: '700' }
                ]}>
                  {issue.message}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Recommendations */}
        {checkResult.recommendations.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionHeader}>Recommendations</Text>
            <View style={styles.recommendationsCard}>
              {checkResult.recommendations.map((rec, index) => (
                <View key={index} style={styles.recommendationItem}>
                  <Text style={styles.recommendationBullet}>*</Text>
                  <Text style={styles.recommendationText}>{rec}</Text>
                </View>
              ))}
            </View>
          </View>
        )}

        <Pressable
          style={styles.newCheckButton}
          onPress={() => {
            setCheckResult(null);
            setMedication('');
          }}
        >
          <Text style={styles.newCheckButtonText}>Check Another Medication</Text>
        </Pressable>
      </ScrollView>
    );
  };

  const renderMedicationPicker = () => (
    <Modal
      visible={showMedicationPicker}
      animationType="slide"
      onRequestClose={() => setShowMedicationPicker(false)}
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <Pressable onPress={() => setShowMedicationPicker(false)}>
            <Text style={styles.closeButton}>Cancel</Text>
          </Pressable>
          <Text style={styles.modalTitle}>Select Medication</Text>
          <View style={{ width: 60 }} />
        </View>

        <TextInput
          style={styles.searchInput}
          value={medicationFilter}
          onChangeText={setMedicationFilter}
          placeholder="Search medications..."
          placeholderTextColor="#999"
        />

        <ScrollView style={styles.medicationList}>
          {filteredMedications.map((med, index) => (
            <Pressable
              key={index}
              style={styles.medicationItem}
              onPress={() => selectMedication(med)}
            >
              <View>
                <Text style={styles.medicationName}>{med.name}</Text>
                <Text style={styles.medicationCategory}>{med.category}</Text>
              </View>
              <Text style={styles.medicationUses}>
                {med.common_uses.slice(0, 2).join(', ')}
              </Text>
            </Pressable>
          ))}
        </ScrollView>
      </View>
    </Modal>
  );

  const renderInteractionDetail = () => (
    <Modal
      visible={showInteractionDetail}
      transparent
      animationType="fade"
      onRequestClose={() => setShowInteractionDetail(false)}
    >
      <View style={styles.detailModalOverlay}>
        <View style={styles.detailModalContent}>
          {selectedInteraction && (
            <>
              <Text style={styles.detailModalTitle}>
                {selectedInteraction.drug1} + {selectedInteraction.drug2}
              </Text>
              <View style={[
                styles.detailSeverityBadge,
                { backgroundColor: getSeverityColor(selectedInteraction.severity) }
              ]}>
                <Text style={styles.detailSeverityText}>
                  {selectedInteraction.severity.toUpperCase()}
                </Text>
              </View>

              <Text style={styles.detailSectionTitle}>Description</Text>
              <Text style={styles.detailText}>{selectedInteraction.description}</Text>

              <Text style={styles.detailSectionTitle}>Mechanism</Text>
              <Text style={styles.detailText}>{selectedInteraction.mechanism}</Text>

              <Text style={styles.detailSectionTitle}>Clinical Effects</Text>
              {selectedInteraction.clinical_effects.map((effect, i) => (
                <Text key={i} style={styles.detailListItem}>* {effect}</Text>
              ))}

              <Text style={styles.detailSectionTitle}>Management</Text>
              <Text style={styles.detailText}>{selectedInteraction.management}</Text>

              <Pressable
                style={styles.detailCloseButton}
                onPress={() => setShowInteractionDetail(false)}
              >
                <Text style={styles.detailCloseButtonText}>Close</Text>
              </Pressable>
            </>
          )}
        </View>
      </View>
    </Modal>
  );

  const renderPhotoDetail = () => (
    <Modal
      visible={showPhotoDetail}
      transparent
      animationType="fade"
      onRequestClose={() => setShowPhotoDetail(false)}
    >
      <View style={styles.detailModalOverlay}>
        <View style={styles.detailModalContent}>
          {selectedPhotoWarning && (
            <>
              <Text style={styles.detailModalTitle}>
                Photosensitivity Warning
              </Text>
              <Text style={styles.photoTypeDetail}>{selectedPhotoWarning.type}</Text>

              <Text style={styles.detailSectionTitle}>When does it occur?</Text>
              <Text style={styles.detailText}>{selectedPhotoWarning.onset_timeframe}</Text>

              <Text style={styles.detailSectionTitle}>Duration after stopping</Text>
              <Text style={styles.detailText}>{selectedPhotoWarning.duration_after_stopping}</Text>

              <Text style={styles.detailSectionTitle}>UV Sensitivity</Text>
              <Text style={styles.detailText}>{selectedPhotoWarning.uva_uvb_sensitivity}</Text>

              <Text style={styles.detailSectionTitle}>Precautions</Text>
              {selectedPhotoWarning.precautions.map((precaution, i) => (
                <Text key={i} style={styles.detailListItem}>* {precaution}</Text>
              ))}

              <View style={styles.spfBox}>
                <Text style={styles.spfLabel}>Recommended SPF:</Text>
                <Text style={styles.spfValue}>{selectedPhotoWarning.spf_recommendation}+</Text>
              </View>

              <Pressable
                style={styles.detailCloseButton}
                onPress={() => setShowPhotoDetail(false)}
              >
                <Text style={styles.detailCloseButtonText}>Close</Text>
              </Pressable>
            </>
          )}
        </View>
      </View>
    </Modal>
  );

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" />

      <View style={styles.header}>
        <Pressable onPress={() => router.back()} style={styles.backButton}>
          <Text style={styles.backButtonText}>leftArrow</Text>
        </Pressable>
        <Text style={styles.headerTitle}>Medication Checker</Text>
        <View style={{ width: 40 }} />
      </View>

      {checkResult ? renderResults() : renderForm()}

      {renderMedicationPicker()}
      {renderInteractionDetail()}
      {renderPhotoDetail()}
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
  medicationInputRow: {
    flexDirection: 'row',
    gap: 8,
  },
  browseButton: {
    backgroundColor: '#4A90A4',
    paddingHorizontal: 16,
    justifyContent: 'center',
    borderRadius: 8,
  },
  browseButtonText: {
    color: '#fff',
    fontWeight: '500',
  },
  row: {
    flexDirection: 'row',
    gap: 12,
  },
  halfWidth: {
    flex: 1,
  },
  tagInputRow: {
    flexDirection: 'row',
    gap: 8,
  },
  addButton: {
    backgroundColor: '#4A90A4',
    width: 44,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 8,
  },
  addButtonText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: '500',
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 8,
  },
  tag: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#E8F5E9',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    gap: 6,
  },
  tagText: {
    color: '#388E3C',
    fontSize: 14,
  },
  tagRemove: {
    color: '#388E3C',
    fontSize: 16,
    fontWeight: '600',
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  switchLabel: {
    fontSize: 16,
    color: '#333',
  },
  optionsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  optionButton: {
    paddingHorizontal: 14,
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
    fontSize: 13,
    color: '#666',
  },
  optionButtonTextSelected: {
    color: '#fff',
  },
  checkButton: {
    backgroundColor: '#4A90A4',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 24,
    marginBottom: 40,
  },
  checkButtonDisabled: {
    backgroundColor: '#B0BEC5',
  },
  checkButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  resultsContainer: {
    flex: 1,
    padding: 16,
  },
  summaryCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  summaryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  summaryTitle: {
    fontSize: 20,
    fontWeight: '700',
  },
  severityBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  severityBadgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  summarySubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  providerReviewBanner: {
    backgroundColor: '#FFF3E0',
    padding: 10,
    borderRadius: 6,
    marginTop: 12,
  },
  providerReviewText: {
    color: '#E65100',
    fontWeight: '500',
    textAlign: 'center',
  },
  section: {
    marginBottom: 20,
  },
  sectionHeader: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  warningCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 14,
    marginBottom: 10,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  warningHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  warningDrugs: {
    fontSize: 15,
    fontWeight: '600',
    color: '#333',
    flex: 1,
  },
  warningCondition: {
    fontSize: 15,
    fontWeight: '600',
    color: '#333',
    textTransform: 'capitalize',
  },
  severityPill: {
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderRadius: 10,
  },
  severityPillText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  warningDescription: {
    fontSize: 14,
    color: '#555',
    lineHeight: 20,
  },
  tapForMore: {
    fontSize: 12,
    color: '#4A90A4',
    marginTop: 8,
  },
  alternativesBox: {
    backgroundColor: '#E8F5E9',
    padding: 10,
    borderRadius: 6,
    marginTop: 10,
  },
  alternativesTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#388E3C',
  },
  alternativesText: {
    fontSize: 13,
    color: '#388E3C',
    marginTop: 2,
  },
  sunWarningHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 8,
  },
  sunIcon: {
    fontSize: 28,
  },
  photoType: {
    fontSize: 15,
    fontWeight: '600',
    color: '#E65100',
    textTransform: 'capitalize',
  },
  spfRecommendation: {
    fontSize: 13,
    color: '#F57C00',
  },
  concernBox: {
    marginBottom: 8,
  },
  concernTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#7B1FA2',
  },
  concernItem: {
    fontSize: 13,
    color: '#555',
    marginTop: 2,
  },
  doseAdjustment: {
    fontSize: 13,
    color: '#333',
    fontStyle: 'italic',
    marginTop: 6,
  },
  pregnancyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  pregnancyCategory: {
    fontSize: 16,
    fontWeight: '700',
    color: '#C2185B',
  },
  lactationBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  lactationBadgeText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
  },
  trimesterBox: {
    marginBottom: 8,
  },
  trimesterLabel: {
    fontSize: 12,
    color: '#666',
  },
  trimesterRisk: {
    fontSize: 13,
    color: '#333',
  },
  dosageIssueText: {
    fontSize: 14,
    color: '#333',
  },
  recommendationsCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 14,
  },
  recommendationItem: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  recommendationBullet: {
    fontSize: 14,
    color: '#4A90A4',
    marginRight: 8,
    fontWeight: '700',
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  newCheckButton: {
    backgroundColor: '#4A90A4',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 16,
    marginBottom: 40,
  },
  newCheckButtonText: {
    color: '#fff',
    fontSize: 15,
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
    fontSize: 16,
    color: '#4A90A4',
    fontWeight: '500',
  },
  searchInput: {
    backgroundColor: '#fff',
    margin: 16,
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    fontSize: 16,
  },
  medicationList: {
    flex: 1,
  },
  medicationItem: {
    backgroundColor: '#fff',
    padding: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  medicationName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  medicationCategory: {
    fontSize: 12,
    color: '#4A90A4',
    marginTop: 2,
  },
  medicationUses: {
    fontSize: 12,
    color: '#999',
    marginTop: 4,
  },
  detailModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  detailModalContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    width: '90%',
    maxWidth: 400,
    maxHeight: '80%',
  },
  detailModalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#333',
    textAlign: 'center',
  },
  detailSeverityBadge: {
    alignSelf: 'center',
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 16,
    marginTop: 10,
    marginBottom: 16,
  },
  detailSeverityText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
  },
  detailSectionTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#666',
    marginTop: 14,
    marginBottom: 4,
    textTransform: 'uppercase',
  },
  detailText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  detailListItem: {
    fontSize: 14,
    color: '#333',
    marginTop: 4,
  },
  detailCloseButton: {
    backgroundColor: '#4A90A4',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 20,
  },
  detailCloseButtonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
  photoTypeDetail: {
    fontSize: 16,
    color: '#E65100',
    textAlign: 'center',
    marginTop: 4,
    textTransform: 'capitalize',
  },
  spfBox: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#FFF3E0',
    padding: 12,
    borderRadius: 8,
    marginTop: 16,
    gap: 8,
  },
  spfLabel: {
    fontSize: 14,
    color: '#E65100',
  },
  spfValue: {
    fontSize: 20,
    fontWeight: '700',
    color: '#E65100',
  },
});
