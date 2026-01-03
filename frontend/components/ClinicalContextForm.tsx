import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  Modal,
  Switch,
  TextInput,
} from 'react-native';

// Clinical context data structure
export interface ClinicalContext {
  // Demographics
  patient_age?: number;
  fitzpatrick_skin_type?: string;

  // Lesion history
  lesion_duration?: string;
  is_new_lesion?: boolean;
  has_changed_recently?: boolean;

  // ABCDE changes
  abcde_changes?: {
    asymmetry_changed?: boolean;
    border_changed?: boolean;
    color_changed?: boolean;
    diameter_changed?: boolean;
    evolving?: boolean;
  };

  // Location
  body_location?: string;
  is_sun_exposed_area?: boolean;
  is_high_risk_location?: boolean;

  // Symptoms
  symptoms?: {
    itching?: boolean;
    bleeding?: boolean;
    pain?: boolean;
    crusting?: boolean;
    oozing?: boolean;
    ulceration?: boolean;
  };

  // Medical history
  personal_history_melanoma?: boolean;
  personal_history_skin_cancer?: boolean;
  personal_history_atypical_moles?: boolean;
  family_history_melanoma?: boolean;
  family_history_skin_cancer?: boolean;

  // Risk factors
  history_severe_sunburns?: boolean;
  uses_tanning_beds?: boolean;
  immunosuppressed?: boolean;
  many_moles?: boolean;

  // Notes
  patient_concerns?: string;
}

interface ClinicalContextFormProps {
  visible: boolean;
  onClose: () => void;
  onSubmit: (context: ClinicalContext) => void;
  initialContext?: ClinicalContext;
}

const FITZPATRICK_TYPES = [
  { value: 'I', label: 'Type I', description: 'Very fair, always burns, never tans' },
  { value: 'II', label: 'Type II', description: 'Fair, usually burns, tans minimally' },
  { value: 'III', label: 'Type III', description: 'Medium, sometimes burns, tans uniformly' },
  { value: 'IV', label: 'Type IV', description: 'Olive, rarely burns, tans easily' },
  { value: 'V', label: 'Type V', description: 'Brown, very rarely burns, tans very easily' },
  { value: 'VI', label: 'Type VI', description: 'Dark brown/black, never burns' },
];

const LESION_DURATIONS = [
  { value: 'new', label: 'New', description: 'Less than 1 month' },
  { value: 'recent', label: 'Recent', description: '1-6 months' },
  { value: 'months', label: 'Several months', description: '6-12 months' },
  { value: 'one_year', label: '1-2 years', description: 'Stable for 1-2 years' },
  { value: 'years', label: '2-5 years', description: 'Present for years' },
  { value: 'long_term', label: 'Long term', description: 'More than 5 years' },
  { value: 'unknown', label: 'Unknown', description: "I don't know" },
];

export const ClinicalContextForm: React.FC<ClinicalContextFormProps> = ({
  visible,
  onClose,
  onSubmit,
  initialContext = {},
}) => {
  const [context, setContext] = useState<ClinicalContext>(initialContext);
  const [currentSection, setCurrentSection] = useState(0);
  const [hasAutoAdvanced, setHasAutoAdvanced] = useState(false);

  // Auto-advance past Patient Info section if age and skin_type are pre-populated
  useEffect(() => {
    if (visible && !hasAutoAdvanced) {
      // Update context with initialContext when modal becomes visible
      setContext(initialContext);

      // If both age and skin_type are pre-filled, auto-advance to section 1 (Lesion History)
      if (initialContext.patient_age && initialContext.fitzpatrick_skin_type) {
        setCurrentSection(1);
        setHasAutoAdvanced(true);
      }
    }

    // Reset when modal closes
    if (!visible) {
      setHasAutoAdvanced(false);
      setCurrentSection(0);
    }
  }, [visible, initialContext]);

  const sections = [
    { title: 'Patient Info', icon: '👤' },
    { title: 'Lesion History', icon: '📋' },
    { title: 'Symptoms', icon: '🩺' },
    { title: 'Medical History', icon: '📁' },
    { title: 'Risk Factors', icon: '⚠️' },
  ];

  const updateContext = (key: string, value: any) => {
    setContext(prev => ({ ...prev, [key]: value }));
  };

  const updateNestedContext = (parent: string, key: string, value: any) => {
    setContext(prev => ({
      ...prev,
      [parent]: {
        ...(prev[parent as keyof ClinicalContext] as object || {}),
        [key]: value,
      },
    }));
  };

  const handleSubmit = () => {
    onSubmit(context);
    onClose();
  };

  const handleSkip = () => {
    onSubmit({});
    onClose();
  };

  const renderSectionIndicator = () => (
    <View style={styles.sectionIndicator}>
      {sections.map((section, index) => (
        <Pressable
          key={index}
          style={[
            styles.sectionDot,
            currentSection === index && styles.sectionDotActive,
          ]}
          onPress={() => setCurrentSection(index)}
        >
          <Text style={styles.sectionIcon}>{section.icon}</Text>
        </Pressable>
      ))}
    </View>
  );

  const renderPatientInfo = () => {
    const isPrePopulated = initialContext.patient_age && initialContext.fitzpatrick_skin_type;

    return (
      <View style={styles.sectionContent}>
        <Text style={styles.sectionTitle}>Patient Information</Text>
        <Text style={styles.sectionDescription}>
          Basic information helps calibrate the risk assessment
        </Text>

        {/* Show pre-populated notice if data came from profile */}
        {isPrePopulated && (
          <View style={styles.prePopulatedNotice}>
            <Text style={styles.prePopulatedText}>
              ✓ Pre-filled from your profile. You can edit if needed.
            </Text>
          </View>
        )}

        {/* Age Input */}
        <View style={styles.inputGroup}>
          <Text style={styles.inputLabel}>Age (years)</Text>
          <TextInput
            style={styles.textInput}
            placeholder="Enter age"
            keyboardType="numeric"
            value={context.patient_age?.toString() || ''}
            onChangeText={(text) => updateContext('patient_age', text ? parseInt(text) : undefined)}
          />
        </View>

        {/* Fitzpatrick Skin Type */}
        <View style={styles.inputGroup}>
          <Text style={styles.inputLabel}>Skin Type (Fitzpatrick Scale)</Text>
          <View style={styles.optionGrid}>
            {FITZPATRICK_TYPES.map((type) => (
              <Pressable
                key={type.value}
                style={[
                  styles.optionButton,
                  context.fitzpatrick_skin_type === type.value && styles.optionButtonActive,
                ]}
                onPress={() => updateContext('fitzpatrick_skin_type', type.value)}
              >
                <Text style={[
                  styles.optionLabel,
                  context.fitzpatrick_skin_type === type.value && styles.optionLabelActive,
                ]}>
                  {type.label}
                </Text>
                <Text style={styles.optionDescription}>{type.description}</Text>
              </Pressable>
            ))}
          </View>
        </View>
      </View>
    );
  };

  const renderLesionHistory = () => (
    <View style={styles.sectionContent}>
      <Text style={styles.sectionTitle}>Lesion History</Text>
      <Text style={styles.sectionDescription}>
        How long has the lesion been present and has it changed?
      </Text>

      {/* Duration */}
      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>How long has this lesion been present?</Text>
        <View style={styles.optionGrid}>
          {LESION_DURATIONS.map((duration) => (
            <Pressable
              key={duration.value}
              style={[
                styles.optionButton,
                context.lesion_duration === duration.value && styles.optionButtonActive,
              ]}
              onPress={() => updateContext('lesion_duration', duration.value)}
            >
              <Text style={[
                styles.optionLabel,
                context.lesion_duration === duration.value && styles.optionLabelActive,
              ]}>
                {duration.label}
              </Text>
              <Text style={styles.optionDescription}>{duration.description}</Text>
            </Pressable>
          ))}
        </View>
      </View>

      {/* Recent Changes */}
      <View style={styles.switchGroup}>
        <View style={styles.switchLabel}>
          <Text style={styles.switchTitle}>Has it changed recently?</Text>
          <Text style={styles.switchDescription}>Any changes in the past 3 months</Text>
        </View>
        <Switch
          value={context.has_changed_recently || false}
          onValueChange={(value) => updateContext('has_changed_recently', value)}
          trackColor={{ false: '#e2e8f0', true: '#3b82f6' }}
        />
      </View>

      {/* ABCDE Changes */}
      <Text style={styles.subheading}>ABCDE Criteria Changes</Text>
      <Text style={styles.helperText}>Has the lesion shown any of these changes?</Text>

      {[
        { key: 'asymmetry_changed', label: 'Asymmetry', desc: 'Shape becoming asymmetric' },
        { key: 'border_changed', label: 'Border', desc: 'Border becoming irregular' },
        { key: 'color_changed', label: 'Color', desc: 'Color changes or multiple colors' },
        { key: 'diameter_changed', label: 'Diameter', desc: 'Growing larger' },
        { key: 'evolving', label: 'Evolution', desc: 'Any noticeable evolution' },
      ].map((item) => (
        <View key={item.key} style={styles.switchGroup}>
          <View style={styles.switchLabel}>
            <Text style={styles.switchTitle}>{item.label}</Text>
            <Text style={styles.switchDescription}>{item.desc}</Text>
          </View>
          <Switch
            value={context.abcde_changes?.[item.key as keyof typeof context.abcde_changes] || false}
            onValueChange={(value) => updateNestedContext('abcde_changes', item.key, value)}
            trackColor={{ false: '#e2e8f0', true: '#f59e0b' }}
          />
        </View>
      ))}
    </View>
  );

  const renderSymptoms = () => (
    <View style={styles.sectionContent}>
      <Text style={styles.sectionTitle}>Symptoms</Text>
      <Text style={styles.sectionDescription}>
        Does the lesion have any of these symptoms?
      </Text>

      {[
        { key: 'itching', label: 'Itching', desc: 'The lesion itches', color: '#f59e0b' },
        { key: 'bleeding', label: 'Bleeding', desc: 'Bleeds spontaneously or easily', color: '#ef4444' },
        { key: 'pain', label: 'Pain', desc: 'Painful or tender', color: '#f59e0b' },
        { key: 'crusting', label: 'Crusting', desc: 'Forms crusts', color: '#f59e0b' },
        { key: 'oozing', label: 'Oozing', desc: 'Oozes fluid', color: '#f59e0b' },
        { key: 'ulceration', label: 'Ulceration', desc: 'Has an open sore', color: '#ef4444' },
      ].map((item) => (
        <View key={item.key} style={styles.switchGroup}>
          <View style={styles.switchLabel}>
            <Text style={styles.switchTitle}>{item.label}</Text>
            <Text style={styles.switchDescription}>{item.desc}</Text>
          </View>
          <Switch
            value={context.symptoms?.[item.key as keyof typeof context.symptoms] || false}
            onValueChange={(value) => updateNestedContext('symptoms', item.key, value)}
            trackColor={{ false: '#e2e8f0', true: item.color }}
          />
        </View>
      ))}
    </View>
  );

  const renderMedicalHistory = () => (
    <View style={styles.sectionContent}>
      <Text style={styles.sectionTitle}>Medical History</Text>
      <Text style={styles.sectionDescription}>
        Personal and family history of skin conditions
      </Text>

      <Text style={styles.subheading}>Personal History</Text>

      {[
        { key: 'personal_history_melanoma', label: 'Previous Melanoma', desc: 'You have had melanoma before' },
        { key: 'personal_history_skin_cancer', label: 'Other Skin Cancer', desc: 'Previous BCC, SCC, or other skin cancer' },
        { key: 'personal_history_atypical_moles', label: 'Atypical Moles', desc: 'History of dysplastic/atypical nevi' },
      ].map((item) => (
        <View key={item.key} style={styles.switchGroup}>
          <View style={styles.switchLabel}>
            <Text style={styles.switchTitle}>{item.label}</Text>
            <Text style={styles.switchDescription}>{item.desc}</Text>
          </View>
          <Switch
            value={context[item.key as keyof ClinicalContext] as boolean || false}
            onValueChange={(value) => updateContext(item.key, value)}
            trackColor={{ false: '#e2e8f0', true: '#ef4444' }}
          />
        </View>
      ))}

      <Text style={styles.subheading}>Family History</Text>

      {[
        { key: 'family_history_melanoma', label: 'Family Melanoma', desc: 'First-degree relative with melanoma' },
        { key: 'family_history_skin_cancer', label: 'Family Skin Cancer', desc: 'Family history of skin cancer' },
      ].map((item) => (
        <View key={item.key} style={styles.switchGroup}>
          <View style={styles.switchLabel}>
            <Text style={styles.switchTitle}>{item.label}</Text>
            <Text style={styles.switchDescription}>{item.desc}</Text>
          </View>
          <Switch
            value={context[item.key as keyof ClinicalContext] as boolean || false}
            onValueChange={(value) => updateContext(item.key, value)}
            trackColor={{ false: '#e2e8f0', true: '#f59e0b' }}
          />
        </View>
      ))}
    </View>
  );

  const renderRiskFactors = () => (
    <View style={styles.sectionContent}>
      <Text style={styles.sectionTitle}>Risk Factors</Text>
      <Text style={styles.sectionDescription}>
        Lifestyle and other factors that may affect risk
      </Text>

      {[
        { key: 'history_severe_sunburns', label: 'Severe Sunburns', desc: 'History of blistering sunburns' },
        { key: 'uses_tanning_beds', label: 'Tanning Beds', desc: 'Use of indoor tanning beds' },
        { key: 'immunosuppressed', label: 'Immunosuppressed', desc: 'Transplant, HIV, immunosuppressive medications' },
        { key: 'many_moles', label: 'Many Moles', desc: 'More than 50 moles on body' },
      ].map((item) => (
        <View key={item.key} style={styles.switchGroup}>
          <View style={styles.switchLabel}>
            <Text style={styles.switchTitle}>{item.label}</Text>
            <Text style={styles.switchDescription}>{item.desc}</Text>
          </View>
          <Switch
            value={context[item.key as keyof ClinicalContext] as boolean || false}
            onValueChange={(value) => updateContext(item.key, value)}
            trackColor={{ false: '#e2e8f0', true: '#f59e0b' }}
          />
        </View>
      ))}

      {/* Patient Concerns */}
      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>Any specific concerns? (Optional)</Text>
        <TextInput
          style={[styles.textInput, styles.textArea]}
          placeholder="What worries you about this lesion?"
          multiline
          numberOfLines={3}
          value={context.patient_concerns || ''}
          onChangeText={(text) => updateContext('patient_concerns', text || undefined)}
        />
      </View>
    </View>
  );

  const renderCurrentSection = () => {
    switch (currentSection) {
      case 0:
        return renderPatientInfo();
      case 1:
        return renderLesionHistory();
      case 2:
        return renderSymptoms();
      case 3:
        return renderMedicalHistory();
      case 4:
        return renderRiskFactors();
      default:
        return renderPatientInfo();
    }
  };

  return (
    <Modal visible={visible} animationType="slide" transparent>
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          {/* Header */}
          <View style={styles.header}>
            <View>
              <Text style={styles.headerTitle}>Clinical Context</Text>
              <Text style={styles.headerSubtitle}>
                Help improve analysis accuracy
              </Text>
            </View>
            <Pressable onPress={onClose} style={styles.closeButton}>
              <Text style={styles.closeButtonText}>X</Text>
            </Pressable>
          </View>

          {/* Section Indicator */}
          {renderSectionIndicator()}

          {/* Content */}
          <ScrollView style={styles.scrollContent} showsVerticalScrollIndicator={false}>
            {renderCurrentSection()}
          </ScrollView>

          {/* Navigation */}
          <View style={styles.navigation}>
            {currentSection > 0 ? (
              <Pressable
                style={styles.navButton}
                onPress={() => setCurrentSection(currentSection - 1)}
              >
                <Text style={styles.navButtonText}>Back</Text>
              </Pressable>
            ) : (
              <Pressable style={styles.skipButton} onPress={handleSkip}>
                <Text style={styles.skipButtonText}>Skip</Text>
              </Pressable>
            )}

            {currentSection < sections.length - 1 ? (
              <Pressable
                style={[styles.navButton, styles.navButtonPrimary]}
                onPress={() => setCurrentSection(currentSection + 1)}
              >
                <Text style={styles.navButtonTextPrimary}>Next</Text>
              </Pressable>
            ) : (
              <Pressable
                style={[styles.navButton, styles.navButtonSubmit]}
                onPress={handleSubmit}
              >
                <Text style={styles.navButtonTextPrimary}>Analyze</Text>
              </Pressable>
            )}
          </View>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    maxHeight: '90%',
    minHeight: '70%',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e293b',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#64748b',
    marginTop: 2,
  },
  closeButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#f1f5f9',
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeButtonText: {
    fontSize: 16,
    color: '#64748b',
    fontWeight: '600',
  },
  sectionIndicator: {
    flexDirection: 'row',
    justifyContent: 'center',
    paddingVertical: 16,
    gap: 12,
  },
  sectionDot: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#f1f5f9',
    justifyContent: 'center',
    alignItems: 'center',
  },
  sectionDotActive: {
    backgroundColor: '#3b82f6',
  },
  sectionIcon: {
    fontSize: 18,
  },
  scrollContent: {
    flex: 1,
    paddingHorizontal: 20,
  },
  sectionContent: {
    paddingBottom: 20,
  },
  sectionTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 8,
  },
  sectionDescription: {
    fontSize: 14,
    color: '#64748b',
    marginBottom: 20,
  },
  prePopulatedNotice: {
    backgroundColor: '#ecfdf5',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#a7f3d0',
  },
  prePopulatedText: {
    fontSize: 13,
    color: '#059669',
    fontWeight: '500',
  },
  subheading: {
    fontSize: 16,
    fontWeight: '600',
    color: '#334155',
    marginTop: 20,
    marginBottom: 12,
  },
  helperText: {
    fontSize: 13,
    color: '#64748b',
    marginBottom: 12,
  },
  inputGroup: {
    marginBottom: 20,
  },
  inputLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: '#334155',
    marginBottom: 10,
  },
  textInput: {
    backgroundColor: '#f8fafc',
    borderRadius: 12,
    padding: 14,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  textArea: {
    height: 100,
    textAlignVertical: 'top',
  },
  optionGrid: {
    gap: 10,
  },
  optionButton: {
    backgroundColor: '#f8fafc',
    borderRadius: 12,
    padding: 14,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  optionButtonActive: {
    borderColor: '#3b82f6',
    backgroundColor: '#eff6ff',
  },
  optionLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: '#334155',
  },
  optionLabelActive: {
    color: '#2563eb',
  },
  optionDescription: {
    fontSize: 13,
    color: '#64748b',
    marginTop: 2,
  },
  switchGroup: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  switchLabel: {
    flex: 1,
    marginRight: 16,
  },
  switchTitle: {
    fontSize: 15,
    fontWeight: '500',
    color: '#334155',
  },
  switchDescription: {
    fontSize: 13,
    color: '#64748b',
    marginTop: 2,
  },
  navigation: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    backgroundColor: '#fff',
  },
  navButton: {
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 12,
    backgroundColor: '#f1f5f9',
  },
  navButtonPrimary: {
    backgroundColor: '#3b82f6',
  },
  navButtonSubmit: {
    backgroundColor: '#10b981',
  },
  navButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#64748b',
  },
  navButtonTextPrimary: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  skipButton: {
    paddingVertical: 14,
    paddingHorizontal: 28,
  },
  skipButtonText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#64748b',
  },
});

export default ClinicalContextForm;
