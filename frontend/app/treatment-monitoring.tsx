import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  TextInput,
  Alert,
  ActivityIndicator,
  Switch,
  Image,
  Modal,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';

// SCORAD Body regions with percentages
const SCORAD_BODY_REGIONS = [
  { id: 'head', name: 'Head & Neck', percentage: 9 },
  { id: 'upper_limbs', name: 'Upper Limbs', percentage: 18 },
  { id: 'lower_limbs', name: 'Lower Limbs', percentage: 36 },
  { id: 'anterior_trunk', name: 'Anterior Trunk', percentage: 18 },
  { id: 'posterior_trunk', name: 'Posterior Trunk', percentage: 18 },
  { id: 'genitals', name: 'Genitals', percentage: 1 },
];

// PASI Body regions with percentages
const PASI_BODY_REGIONS = [
  { id: 'head', name: 'Head', weight: 0.1 },
  { id: 'upper_limbs', name: 'Upper Limbs', weight: 0.2 },
  { id: 'trunk', name: 'Trunk', weight: 0.3 },
  { id: 'lower_limbs', name: 'Lower Limbs', weight: 0.4 },
];

export default function TreatmentMonitoringScreen() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const { t } = useTranslation();

  // Tab state - now includes SCORAD, PASI, and Photo Comparison
  const [activeTab, setActiveTab] = useState<'treatments' | 'add' | 'log' | 'scorad' | 'pasi' | 'compare'>('treatments');

  // Treatments list
  const [treatments, setTreatments] = useState([]);
  const [selectedTreatment, setSelectedTreatment] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Add treatment form
  const [treatmentName, setTreatmentName] = useState('');
  const [treatmentType, setTreatmentType] = useState('topical');
  const [activeIngredient, setActiveIngredient] = useState('');
  const [dosage, setDosage] = useState('');
  const [frequency, setFrequency] = useState('once_daily');
  const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0]);
  const [indication, setIndication] = useState('');
  const [targetBodyArea, setTargetBodyArea] = useState('');
  const [notes, setNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Log dose form
  const [logDate, setLogDate] = useState(new Date().toISOString().split('T')[0]);
  const [takenAsPrescribed, setTakenAsPrescribed] = useState(true);
  const [missedDose, setMissedDose] = useState(false);
  const [immediateReaction, setImmediateReaction] = useState('none');
  const [logNotes, setLogNotes] = useState('');

  // Effectiveness assessment
  const [effectiveness, setEffectiveness] = useState([]);
  const [showEffectivenessForm, setShowEffectivenessForm] = useState(false);
  const [baselineSize, setBaselineSize] = useState('');
  const [currentSize, setCurrentSize] = useState('');
  const [patientRating, setPatientRating] = useState(3);
  const [improvementsNoted, setImprovementsNoted] = useState<string[]>([]);
  const [sideEffects, setSideEffects] = useState<string[]>([]);

  // SCORAD Calculator state
  const [scoradAffectedAreas, setScoradAffectedAreas] = useState<Record<string, number>>({});
  const [scoradErythema, setScoradErythema] = useState(0);
  const [scoradEdema, setScoradEdema] = useState(0);
  const [scoradOozing, setScoradOozing] = useState(0);
  const [scoradExcoriation, setScoradExcoriation] = useState(0);
  const [scoradLichenification, setScoradLichenification] = useState(0);
  const [scoradDryness, setScoradDryness] = useState(0);
  const [scoradItch, setScoradItch] = useState(0);
  const [scoradSleepLoss, setScoradSleepLoss] = useState(0);
  const [scoradResult, setScoradResult] = useState<any>(null);

  // PASI Calculator state
  const [pasiRegions, setPasiRegions] = useState<Record<string, {
    area: number;
    erythema: number;
    induration: number;
    desquamation: number;
  }>>({
    head: { area: 0, erythema: 0, induration: 0, desquamation: 0 },
    upper_limbs: { area: 0, erythema: 0, induration: 0, desquamation: 0 },
    trunk: { area: 0, erythema: 0, induration: 0, desquamation: 0 },
    lower_limbs: { area: 0, erythema: 0, induration: 0, desquamation: 0 },
  });
  const [pasiResult, setPasiResult] = useState<any>(null);

  // Photo comparison state
  const [lesionHistory, setLesionHistory] = useState<any[]>([]);
  const [selectedLesion, setSelectedLesion] = useState<any>(null);
  const [beforePhoto, setBeforePhoto] = useState<any>(null);
  const [afterPhoto, setAfterPhoto] = useState<any>(null);
  const [comparisonResult, setComparisonResult] = useState<any>(null);
  const [showComparisonModal, setShowComparisonModal] = useState(false);

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    }
  }, [isAuthenticated]);

  useEffect(() => {
    if (isAuthenticated && activeTab === 'treatments') {
      loadTreatments();
    }
  }, [activeTab, isAuthenticated]);

  const loadTreatments = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/treatments`, {
        headers: {
          'Authorization': `Bearer ${user?.token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setTreatments(data.treatments || []);
      }
    } catch (error) {
      console.log('Could not load treatments:', error);
      setTreatments([]);
    } finally {
      setIsLoading(false);
    }
  };

  const loadEffectiveness = async (treatmentId: number) => {
    try {
      const response = await fetch(`${API_BASE_URL}/treatment-effectiveness/${treatmentId}`, {
        headers: {
          'Authorization': `Bearer ${user?.token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setEffectiveness(data.assessments || []);
      }
    } catch (error) {
      console.log('Could not load effectiveness:', error);
    }
  };

  // Load lesion history for photo comparison
  const loadLesionHistory = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/analysis-history`, {
        headers: {
          'Authorization': `Bearer ${user?.token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setLesionHistory(data.analyses || []);
      }
    } catch (error) {
      console.log('Could not load lesion history:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Calculate SCORAD score
  const calculateScorad = async () => {
    setIsSubmitting(true);
    try {
      // Calculate affected area (A)
      let totalAffectedArea = 0;
      Object.entries(scoradAffectedAreas).forEach(([regionId, percentage]) => {
        const region = SCORAD_BODY_REGIONS.find(r => r.id === regionId);
        if (region) {
          totalAffectedArea += (percentage / 100) * region.percentage;
        }
      });

      const requestBody = {
        affected_area_percentage: totalAffectedArea,
        erythema: scoradErythema,
        edema: scoradEdema,
        oozing_crusting: scoradOozing,
        excoriation: scoradExcoriation,
        lichenification: scoradLichenification,
        dryness: scoradDryness,
        pruritus_vvas: scoradItch,
        sleep_loss_vvas: scoradSleepLoss,
        treatment_id: selectedTreatment?.id || null,
      };

      const response = await fetch(`${API_BASE_URL}/calculate-scorad`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${user?.token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const data = await response.json();
        setScoradResult(data);
        Alert.alert(
          'SCORAD Calculated',
          `Score: ${data.total_score.toFixed(1)}\nSeverity: ${data.severity_category}`
        );
      } else {
        throw new Error('Failed to calculate SCORAD');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to calculate SCORAD score');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Calculate PASI score
  const calculatePasi = async () => {
    setIsSubmitting(true);
    try {
      const requestBody = {
        head: pasiRegions.head,
        upper_limbs: pasiRegions.upper_limbs,
        trunk: pasiRegions.trunk,
        lower_limbs: pasiRegions.lower_limbs,
        treatment_id: selectedTreatment?.id || null,
      };

      const response = await fetch(`${API_BASE_URL}/calculate-pasi`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${user?.token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const data = await response.json();
        setPasiResult(data);
        Alert.alert(
          'PASI Calculated',
          `Score: ${data.total_score.toFixed(1)}\nSeverity: ${data.severity_category}`
        );
      } else {
        throw new Error('Failed to calculate PASI');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to calculate PASI score');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Compare before/after photos
  const compareLesionPhotos = async () => {
    if (!beforePhoto || !afterPhoto) {
      Alert.alert('Select Photos', 'Please select both before and after photos to compare');
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/compare-lesion-analyses`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${user?.token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysis_id_1: beforePhoto.id,
          analysis_id_2: afterPhoto.id,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setComparisonResult(data);
        setShowComparisonModal(true);
      } else {
        throw new Error('Failed to compare photos');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to compare lesion photos');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleAddTreatment = async () => {
    if (!treatmentName || !indication) {
      Alert.alert(t('treatmentMonitoring.add.validationError'), t('treatmentMonitoring.add.validationMessage'));
      return;
    }

    setIsSubmitting(true);
    try {
      const requestBody = {
        treatment_name: treatmentName,
        treatment_type: treatmentType,
        start_date: new Date(startDate).toISOString(),
        active_ingredient: activeIngredient || null,
        dosage: dosage || null,
        frequency,
        indication,
        target_body_area: targetBodyArea || null,
        notes: notes || null,
      };

      const response = await fetch(`${API_BASE_URL}/treatments`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${user?.token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        Alert.alert(t('treatmentMonitoring.add.success'), t('treatmentMonitoring.add.successMessage'));
        resetAddForm();
        setActiveTab('treatments');
      } else {
        throw new Error('Failed to add treatment');
      }
    } catch (error) {
      Alert.alert(t('treatmentMonitoring.add.error'), t('treatmentMonitoring.add.errorMessage'));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleLogDose = async () => {
    if (!selectedTreatment) {
      Alert.alert(t('treatmentMonitoring.log.selectError'), t('treatmentMonitoring.log.selectErrorMessage'));
      return;
    }

    setIsSubmitting(true);
    try {
      const requestBody = {
        treatment_id: selectedTreatment.id,
        administered_date: new Date(logDate).toISOString(),
        taken_as_prescribed: takenAsPrescribed,
        missed_dose: missedDose,
        immediate_reaction: immediateReaction,
        notes: logNotes || null,
      };

      const response = await fetch(`${API_BASE_URL}/treatment-logs`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${user?.token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        Alert.alert(t('treatmentMonitoring.log.doseSuccess'), t('treatmentMonitoring.log.doseSuccessMessage'));
        resetLogForm();
      } else {
        throw new Error('Failed to log dose');
      }
    } catch (error) {
      Alert.alert(t('treatmentMonitoring.log.doseError'), t('treatmentMonitoring.log.doseErrorMessage'));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCreateEffectivenessAssessment = async () => {
    if (!selectedTreatment) {
      Alert.alert(t('treatmentMonitoring.log.selectError'), t('treatmentMonitoring.log.selectErrorMessage'));
      return;
    }

    setIsSubmitting(true);
    try {
      const requestBody = {
        treatment_id: selectedTreatment.id,
        assessment_date: new Date().toISOString(),
        baseline_size_mm: baselineSize ? parseFloat(baselineSize) : null,
        current_size_mm: currentSize ? parseFloat(currentSize) : null,
        patient_effectiveness_rating: patientRating,
        improvements_noted: improvementsNoted,
        side_effects: sideEffects,
        treatment_outcome: 'stable',
      };

      const response = await fetch(`${API_BASE_URL}/treatment-effectiveness`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${user?.token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(
          t('treatmentMonitoring.log.assessmentSuccess'),
          t('treatmentMonitoring.log.assessmentSuccessMessage', { score: data.objective_effectiveness_score || 0 })
        );
        setShowEffectivenessForm(false);
        loadEffectiveness(selectedTreatment.id);
      } else {
        throw new Error('Failed to create assessment');
      }
    } catch (error) {
      Alert.alert(t('treatmentMonitoring.log.assessmentError'), t('treatmentMonitoring.log.assessmentErrorMessage'));
    } finally {
      setIsSubmitting(false);
    }
  };

  const resetAddForm = () => {
    setTreatmentName('');
    setActiveIngredient('');
    setDosage('');
    setIndication('');
    setTargetBodyArea('');
    setNotes('');
    setStartDate(new Date().toISOString().split('T')[0]);
  };

  const resetLogForm = () => {
    setLogDate(new Date().toISOString().split('T')[0]);
    setTakenAsPrescribed(true);
    setMissedDose(false);
    setImmediateReaction('none');
    setLogNotes('');
  };

  const toggleImprovement = (improvement: string) => {
    if (improvementsNoted.includes(improvement)) {
      setImprovementsNoted(improvementsNoted.filter(i => i !== improvement));
    } else {
      setImprovementsNoted([...improvementsNoted, improvement]);
    }
  };

  const toggleSideEffect = (effect: string) => {
    if (sideEffects.includes(effect)) {
      setSideEffects(sideEffects.filter(e => e !== effect));
    } else {
      setSideEffects([...sideEffects, effect]);
    }
  };

  const renderTreatmentsList = () => (
    <ScrollView style={styles.tabContent}>
      {isLoading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4299e1" />
        </View>
      ) : treatments.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateText}>{t('treatmentMonitoring.list.emptyTitle')}</Text>
          <Text style={styles.emptyStateSubtext}>{t('treatmentMonitoring.list.emptySubtext')}</Text>
          <Pressable style={styles.addButton} onPress={() => setActiveTab('add')}>
            <Text style={styles.addButtonText}>{t('treatmentMonitoring.list.addTreatment')}</Text>
          </Pressable>
        </View>
      ) : (
        <>
          {treatments.map((treatment: any) => (
            <Pressable
              key={treatment.id}
              style={[
                styles.treatmentCard,
                selectedTreatment?.id === treatment.id && styles.treatmentCardSelected
              ]}
              onPress={() => {
                setSelectedTreatment(treatment);
                loadEffectiveness(treatment.id);
              }}
            >
              <View style={styles.treatmentHeader}>
                <Text style={styles.treatmentName}>{treatment.treatment_name}</Text>
                {treatment.is_active && <View style={styles.activeBadge}><Text style={styles.activeBadgeText}>{t('treatmentMonitoring.list.active')}</Text></View>}
              </View>

              <Text style={styles.treatmentDetail}>
                {treatment.treatment_type} • {treatment.frequency}
              </Text>
              {treatment.active_ingredient && (
                <Text style={styles.treatmentDetail}>{t('treatmentMonitoring.list.ingredient')} {treatment.active_ingredient}</Text>
              )}
              {treatment.dosage && (
                <Text style={styles.treatmentDetail}>{t('treatmentMonitoring.list.dosage')} {treatment.dosage}</Text>
              )}
              <Text style={styles.treatmentDetail}>{t('treatmentMonitoring.list.for')} {treatment.indication}</Text>
              <Text style={styles.treatmentDetail}>
                {t('treatmentMonitoring.list.started')} {new Date(treatment.start_date).toLocaleDateString()}
              </Text>

              {selectedTreatment?.id === treatment.id && effectiveness.length > 0 && (
                <View style={styles.effectivenessPreview}>
                  <Text style={styles.effectivenessTitle}>{t('treatmentMonitoring.list.latestAssessment')}</Text>
                  {effectiveness[0] && (
                    <>
                      <Text style={styles.effectivenessText}>
                        {t('treatmentMonitoring.list.score')} {effectiveness[0].objective_effectiveness_score || 0}/100
                      </Text>
                      {effectiveness[0].size_change_percent && (
                        <Text style={styles.effectivenessText}>
                          {t('treatmentMonitoring.list.sizeChange')} {effectiveness[0].size_change_percent > 0 ? '+' : ''}
                          {effectiveness[0].size_change_percent}%
                        </Text>
                      )}
                    </>
                  )}
                </View>
              )}
            </Pressable>
          ))}

          {selectedTreatment && (
            <View style={styles.actionButtons}>
              <Pressable
                style={styles.actionButton}
                onPress={() => {
                  setActiveTab('log');
                  setShowEffectivenessForm(false);
                }}
              >
                <Text style={styles.actionButtonText}>{t('treatmentMonitoring.list.logDose')}</Text>
              </Pressable>
              <Pressable
                style={[styles.actionButton, styles.actionButtonSecondary]}
                onPress={() => {
                  setActiveTab('log');
                  setShowEffectivenessForm(true);
                }}
              >
                <Text style={styles.actionButtonText}>{t('treatmentMonitoring.list.assessEffectiveness')}</Text>
              </Pressable>
            </View>
          )}
        </>
      )}
      <View style={{ height: 20 }} />
    </ScrollView>
  );

  const renderAddForm = () => (
    <ScrollView style={styles.tabContent}>
      <View style={styles.form}>
        <Text style={styles.formTitle}>{t('treatmentMonitoring.add.title')}</Text>

        <Text style={styles.label}>{t('treatmentMonitoring.add.treatmentNameRequired')}</Text>
        <TextInput
          style={styles.input}
          value={treatmentName}
          onChangeText={setTreatmentName}
          placeholder={t('treatmentMonitoring.add.namePlaceholder')}
        />

        <Text style={styles.label}>{t('treatmentMonitoring.add.type')}</Text>
        <View style={styles.buttonGroup}>
          {[
            { key: 'topical', label: t('treatmentMonitoring.add.topical') },
            { key: 'oral_medication', label: t('treatmentMonitoring.add.oralMedication') },
            { key: 'injection', label: t('treatmentMonitoring.add.injection') },
            { key: 'procedure', label: t('treatmentMonitoring.add.procedure') }
          ].map(type => (
            <Pressable
              key={type.key}
              style={[styles.optionButton, treatmentType === type.key && styles.optionButtonActive]}
              onPress={() => setTreatmentType(type.key)}
            >
              <Text style={[styles.optionButtonText, treatmentType === type.key && styles.optionButtonTextActive]}>
                {type.label}
              </Text>
            </Pressable>
          ))}
        </View>

        <Text style={styles.label}>{t('treatmentMonitoring.add.activeIngredient')}</Text>
        <TextInput
          style={styles.input}
          value={activeIngredient}
          onChangeText={setActiveIngredient}
          placeholder={t('treatmentMonitoring.add.ingredientPlaceholder')}
        />

        <Text style={styles.label}>{t('treatmentMonitoring.add.dosage')}</Text>
        <TextInput
          style={styles.input}
          value={dosage}
          onChangeText={setDosage}
          placeholder={t('treatmentMonitoring.add.dosagePlaceholder')}
        />

        <Text style={styles.label}>{t('treatmentMonitoring.add.frequency')}</Text>
        <View style={styles.buttonGroup}>
          {[
            { key: 'once_daily', label: t('treatmentMonitoring.add.onceDaily') },
            { key: 'twice_daily', label: t('treatmentMonitoring.add.twiceDaily') },
            { key: 'as_needed', label: t('treatmentMonitoring.add.asNeeded') },
            { key: 'weekly', label: t('treatmentMonitoring.add.weekly') }
          ].map(freq => (
            <Pressable
              key={freq.key}
              style={[styles.optionButton, frequency === freq.key && styles.optionButtonActive]}
              onPress={() => setFrequency(freq.key)}
            >
              <Text style={[styles.optionButtonText, frequency === freq.key && styles.optionButtonTextActive]}>
                {freq.label}
              </Text>
            </Pressable>
          ))}
        </View>

        <Text style={styles.label}>{t('treatmentMonitoring.add.indicationRequired')}</Text>
        <TextInput
          style={styles.input}
          value={indication}
          onChangeText={setIndication}
          placeholder={t('treatmentMonitoring.add.indicationPlaceholder')}
        />

        <Text style={styles.label}>{t('treatmentMonitoring.add.targetBodyArea')}</Text>
        <TextInput
          style={styles.input}
          value={targetBodyArea}
          onChangeText={setTargetBodyArea}
          placeholder={t('treatmentMonitoring.add.bodyAreaPlaceholder')}
        />

        <Text style={styles.label}>{t('treatmentMonitoring.add.startDate')}</Text>
        <TextInput
          style={styles.input}
          value={startDate}
          onChangeText={setStartDate}
          placeholder={t('treatmentMonitoring.log.datePlaceholder')}
        />

        <Text style={styles.label}>{t('treatmentMonitoring.add.notes')}</Text>
        <TextInput
          style={[styles.input, styles.textArea]}
          value={notes}
          onChangeText={setNotes}
          placeholder={t('treatmentMonitoring.add.notesPlaceholder')}
          multiline
          numberOfLines={3}
        />

        <Pressable
          style={[styles.submitButton, isSubmitting && styles.submitButtonDisabled]}
          onPress={handleAddTreatment}
          disabled={isSubmitting}
        >
          {isSubmitting ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.submitButtonText}>{t('treatmentMonitoring.add.addTreatment')}</Text>
          )}
        </Pressable>
      </View>
      <View style={{ height: 40 }} />
    </ScrollView>
  );

  const renderLogForm = () => (
    <ScrollView style={styles.tabContent}>
      {!selectedTreatment ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateText}>{t('treatmentMonitoring.log.selectTreatmentFirst')}</Text>
          <Text style={styles.emptyStateSubtext}>{t('treatmentMonitoring.log.selectTreatmentSubtext')}</Text>
        </View>
      ) : showEffectivenessForm ? (
        <View style={styles.form}>
          <Text style={styles.formTitle}>{t('treatmentMonitoring.log.effectivenessTitle')}</Text>
          <Text style={styles.formSubtitle}>{t('treatmentMonitoring.log.for')} {selectedTreatment.treatment_name}</Text>

          <Text style={styles.label}>{t('treatmentMonitoring.log.baselineSize')}</Text>
          <TextInput
            style={styles.input}
            value={baselineSize}
            onChangeText={setBaselineSize}
            placeholder={t('treatmentMonitoring.log.baselinePlaceholder')}
            keyboardType="decimal-pad"
          />

          <Text style={styles.label}>{t('treatmentMonitoring.log.currentSize')}</Text>
          <TextInput
            style={styles.input}
            value={currentSize}
            onChangeText={setCurrentSize}
            placeholder={t('treatmentMonitoring.log.currentPlaceholder')}
            keyboardType="decimal-pad"
          />

          <Text style={styles.label}>{t('treatmentMonitoring.log.yourRating')}</Text>
          <View style={styles.starRating}>
            {[1, 2, 3, 4, 5].map(star => (
              <Pressable
                key={star}
                onPress={() => setPatientRating(star)}
              >
                <Text style={styles.star}>{star <= patientRating ? '⭐' : '☆'}</Text>
              </Pressable>
            ))}
          </View>

          <Text style={styles.label}>{t('treatmentMonitoring.log.improvementsNoted')}</Text>
          <View style={styles.checkboxGroup}>
            {['size_reduced', 'less_red', 'less_itchy', 'flatter'].map(improvement => (
              <Pressable
                key={improvement}
                style={[styles.checkbox, improvementsNoted.includes(improvement) && styles.checkboxChecked]}
                onPress={() => toggleImprovement(improvement)}
              >
                <Text style={styles.checkboxText}>
                  {t(`treatmentMonitoring.log.${improvement === 'size_reduced' ? 'sizeReduced' : improvement === 'less_red' ? 'lessRed' : improvement === 'less_itchy' ? 'lessItchy' : 'flatter'}`)}
                </Text>
              </Pressable>
            ))}
          </View>

          <Text style={styles.label}>{t('treatmentMonitoring.log.sideEffects')}</Text>
          <View style={styles.checkboxGroup}>
            {['redness', 'irritation', 'dryness', 'burning'].map(effect => (
              <Pressable
                key={effect}
                style={[styles.checkbox, sideEffects.includes(effect) && styles.checkboxChecked]}
                onPress={() => toggleSideEffect(effect)}
              >
                <Text style={styles.checkboxText}>
                  {t(`treatmentMonitoring.log.${effect}`)}
                </Text>
              </Pressable>
            ))}
          </View>

          <Pressable
            style={[styles.submitButton, isSubmitting && styles.submitButtonDisabled]}
            onPress={handleCreateEffectivenessAssessment}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.submitButtonText}>{t('treatmentMonitoring.log.createAssessment')}</Text>
            )}
          </Pressable>
        </View>
      ) : (
        <View style={styles.form}>
          <Text style={styles.formTitle}>{t('treatmentMonitoring.log.logDoseTitle')}</Text>
          <Text style={styles.formSubtitle}>{t('treatmentMonitoring.log.for')} {selectedTreatment.treatment_name}</Text>

          <Text style={styles.label}>{t('treatmentMonitoring.log.date')}</Text>
          <TextInput
            style={styles.input}
            value={logDate}
            onChangeText={setLogDate}
            placeholder={t('treatmentMonitoring.log.datePlaceholder')}
          />

          <View style={styles.switchRow}>
            <Text style={styles.switchLabel}>{t('treatmentMonitoring.log.takenAsPrescribed')}</Text>
            <Switch value={takenAsPrescribed} onValueChange={setTakenAsPrescribed} />
          </View>

          <View style={styles.switchRow}>
            <Text style={styles.switchLabel}>{t('treatmentMonitoring.log.missedDose')}</Text>
            <Switch value={missedDose} onValueChange={setMissedDose} />
          </View>

          <Text style={styles.label}>{t('treatmentMonitoring.log.immediateReaction')}</Text>
          <View style={styles.buttonGroup}>
            {['none', 'burning', 'stinging', 'redness', 'itching'].map(reaction => (
              <Pressable
                key={reaction}
                style={[styles.optionButton, immediateReaction === reaction && styles.optionButtonActive]}
                onPress={() => setImmediateReaction(reaction)}
              >
                <Text style={[styles.optionButtonText, immediateReaction === reaction && styles.optionButtonTextActive]}>
                  {t(`treatmentMonitoring.log.${reaction}`)}
                </Text>
              </Pressable>
            ))}
          </View>

          <Text style={styles.label}>{t('treatmentMonitoring.log.notes')}</Text>
          <TextInput
            style={[styles.input, styles.textArea]}
            value={logNotes}
            onChangeText={setLogNotes}
            placeholder={t('treatmentMonitoring.log.notesPlaceholder')}
            multiline
            numberOfLines={3}
          />

          <Pressable
            style={[styles.submitButton, isSubmitting && styles.submitButtonDisabled]}
            onPress={handleLogDose}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.submitButtonText}>{t('treatmentMonitoring.log.logDose')}</Text>
            )}
          </Pressable>
        </View>
      )}
      <View style={{ height: 40 }} />
    </ScrollView>
  );

  // SCORAD Calculator render
  const renderScoradCalculator = () => (
    <ScrollView style={styles.tabContent}>
      <View style={styles.form}>
        <Text style={styles.formTitle}>SCORAD Calculator</Text>
        <Text style={styles.formSubtitle}>Severity Scoring of Atopic Dermatitis</Text>

        {/* Part A: Affected Body Surface Area */}
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>A. Affected Body Surface Area</Text>
          <Text style={styles.sectionSubtitle}>Estimate % of each region affected</Text>
        </View>

        {SCORAD_BODY_REGIONS.map(region => (
          <View key={region.id} style={styles.regionRow}>
            <View style={styles.regionInfo}>
              <Text style={styles.regionName}>{region.name}</Text>
              <Text style={styles.regionPercent}>({region.percentage}% of body)</Text>
            </View>
            <View style={styles.sliderContainer}>
              <View style={styles.sliderLabels}>
                {[0, 25, 50, 75, 100].map(val => (
                  <Pressable
                    key={val}
                    style={[
                      styles.sliderOption,
                      (scoradAffectedAreas[region.id] || 0) === val && styles.sliderOptionActive
                    ]}
                    onPress={() => setScoradAffectedAreas({ ...scoradAffectedAreas, [region.id]: val })}
                  >
                    <Text style={[
                      styles.sliderOptionText,
                      (scoradAffectedAreas[region.id] || 0) === val && styles.sliderOptionTextActive
                    ]}>
                      {val}%
                    </Text>
                  </Pressable>
                ))}
              </View>
            </View>
          </View>
        ))}

        {/* Part B: Intensity */}
        <View style={[styles.sectionHeader, { marginTop: 24 }]}>
          <Text style={styles.sectionTitle}>B. Intensity (0-3 scale)</Text>
          <Text style={styles.sectionSubtitle}>0=None, 1=Mild, 2=Moderate, 3=Severe</Text>
        </View>

        {[
          { key: 'erythema', label: 'Erythema (Redness)', setter: setScoradErythema, value: scoradErythema },
          { key: 'edema', label: 'Edema/Papulation', setter: setScoradEdema, value: scoradEdema },
          { key: 'oozing', label: 'Oozing/Crusting', setter: setScoradOozing, value: scoradOozing },
          { key: 'excoriation', label: 'Excoriation', setter: setScoradExcoriation, value: scoradExcoriation },
          { key: 'lichenification', label: 'Lichenification', setter: setScoradLichenification, value: scoradLichenification },
          { key: 'dryness', label: 'Dryness', setter: setScoradDryness, value: scoradDryness },
        ].map(item => (
          <View key={item.key} style={styles.intensityRow}>
            <Text style={styles.intensityLabel}>{item.label}</Text>
            <View style={styles.intensityButtons}>
              {[0, 1, 2, 3].map(val => (
                <Pressable
                  key={val}
                  style={[styles.intensityButton, item.value === val && styles.intensityButtonActive]}
                  onPress={() => item.setter(val)}
                >
                  <Text style={[styles.intensityButtonText, item.value === val && styles.intensityButtonTextActive]}>
                    {val}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>
        ))}

        {/* Part C: Subjective Symptoms */}
        <View style={[styles.sectionHeader, { marginTop: 24 }]}>
          <Text style={styles.sectionTitle}>C. Subjective Symptoms (VAS 0-10)</Text>
          <Text style={styles.sectionSubtitle}>Patient-reported severity over last 3 days/nights</Text>
        </View>

        <View style={styles.vasRow}>
          <Text style={styles.vasLabel}>Pruritus (Itch)</Text>
          <View style={styles.vasButtons}>
            {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(val => (
              <Pressable
                key={val}
                style={[styles.vasButton, scoradItch === val && styles.vasButtonActive]}
                onPress={() => setScoradItch(val)}
              >
                <Text style={[styles.vasButtonText, scoradItch === val && styles.vasButtonTextActive]}>
                  {val}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        <View style={styles.vasRow}>
          <Text style={styles.vasLabel}>Sleep Loss</Text>
          <View style={styles.vasButtons}>
            {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(val => (
              <Pressable
                key={val}
                style={[styles.vasButton, scoradSleepLoss === val && styles.vasButtonActive]}
                onPress={() => setScoradSleepLoss(val)}
              >
                <Text style={[styles.vasButtonText, scoradSleepLoss === val && styles.vasButtonTextActive]}>
                  {val}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        {/* Results */}
        {scoradResult && (
          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>SCORAD Result</Text>
            <Text style={styles.resultScore}>{scoradResult.total_score.toFixed(1)}</Text>
            <View style={[styles.severityBadge, { backgroundColor: getSeverityColor(scoradResult.severity_category) }]}>
              <Text style={styles.severityText}>{scoradResult.severity_category}</Text>
            </View>
            <View style={styles.resultBreakdown}>
              <Text style={styles.breakdownItem}>Area (A): {scoradResult.area_score?.toFixed(1) || 'N/A'}</Text>
              <Text style={styles.breakdownItem}>Intensity (B): {scoradResult.intensity_score?.toFixed(1) || 'N/A'}</Text>
              <Text style={styles.breakdownItem}>Symptoms (C): {scoradResult.subjective_score?.toFixed(1) || 'N/A'}</Text>
            </View>
            <Text style={styles.resultInterpretation}>{scoradResult.interpretation || ''}</Text>
          </View>
        )}

        <Pressable
          style={[styles.submitButton, isSubmitting && styles.submitButtonDisabled]}
          onPress={calculateScorad}
          disabled={isSubmitting}
        >
          {isSubmitting ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.submitButtonText}>Calculate SCORAD</Text>
          )}
        </Pressable>
      </View>
      <View style={{ height: 40 }} />
    </ScrollView>
  );

  // PASI Calculator render
  const renderPasiCalculator = () => (
    <ScrollView style={styles.tabContent}>
      <View style={styles.form}>
        <Text style={styles.formTitle}>PASI Calculator</Text>
        <Text style={styles.formSubtitle}>Psoriasis Area and Severity Index</Text>

        {PASI_BODY_REGIONS.map(region => (
          <View key={region.id} style={styles.pasiRegionCard}>
            <View style={styles.pasiRegionHeader}>
              <Text style={styles.pasiRegionName}>{region.name}</Text>
              <Text style={styles.pasiRegionWeight}>Weight: {(region.weight * 100)}%</Text>
            </View>

            {/* Area involvement */}
            <Text style={styles.pasiLabel}>Area Involvement (0-6)</Text>
            <View style={styles.pasiScaleRow}>
              {[0, 1, 2, 3, 4, 5, 6].map(val => (
                <Pressable
                  key={val}
                  style={[styles.pasiScaleButton, pasiRegions[region.id]?.area === val && styles.pasiScaleButtonActive]}
                  onPress={() => setPasiRegions({
                    ...pasiRegions,
                    [region.id]: { ...pasiRegions[region.id], area: val }
                  })}
                >
                  <Text style={[styles.pasiScaleText, pasiRegions[region.id]?.area === val && styles.pasiScaleTextActive]}>
                    {val}
                  </Text>
                </Pressable>
              ))}
            </View>
            <Text style={styles.pasiScaleHint}>0=0%, 1=1-9%, 2=10-29%, 3=30-49%, 4=50-69%, 5=70-89%, 6=90-100%</Text>

            {/* Erythema */}
            <Text style={styles.pasiLabel}>Erythema (0-4)</Text>
            <View style={styles.pasiScaleRow}>
              {[0, 1, 2, 3, 4].map(val => (
                <Pressable
                  key={val}
                  style={[styles.pasiScaleButton, pasiRegions[region.id]?.erythema === val && styles.pasiScaleButtonActive]}
                  onPress={() => setPasiRegions({
                    ...pasiRegions,
                    [region.id]: { ...pasiRegions[region.id], erythema: val }
                  })}
                >
                  <Text style={[styles.pasiScaleText, pasiRegions[region.id]?.erythema === val && styles.pasiScaleTextActive]}>
                    {val}
                  </Text>
                </Pressable>
              ))}
            </View>

            {/* Induration */}
            <Text style={styles.pasiLabel}>Induration/Thickness (0-4)</Text>
            <View style={styles.pasiScaleRow}>
              {[0, 1, 2, 3, 4].map(val => (
                <Pressable
                  key={val}
                  style={[styles.pasiScaleButton, pasiRegions[region.id]?.induration === val && styles.pasiScaleButtonActive]}
                  onPress={() => setPasiRegions({
                    ...pasiRegions,
                    [region.id]: { ...pasiRegions[region.id], induration: val }
                  })}
                >
                  <Text style={[styles.pasiScaleText, pasiRegions[region.id]?.induration === val && styles.pasiScaleTextActive]}>
                    {val}
                  </Text>
                </Pressable>
              ))}
            </View>

            {/* Desquamation */}
            <Text style={styles.pasiLabel}>Desquamation/Scaling (0-4)</Text>
            <View style={styles.pasiScaleRow}>
              {[0, 1, 2, 3, 4].map(val => (
                <Pressable
                  key={val}
                  style={[styles.pasiScaleButton, pasiRegions[region.id]?.desquamation === val && styles.pasiScaleButtonActive]}
                  onPress={() => setPasiRegions({
                    ...pasiRegions,
                    [region.id]: { ...pasiRegions[region.id], desquamation: val }
                  })}
                >
                  <Text style={[styles.pasiScaleText, pasiRegions[region.id]?.desquamation === val && styles.pasiScaleTextActive]}>
                    {val}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>
        ))}

        {/* Results */}
        {pasiResult && (
          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>PASI Result</Text>
            <Text style={styles.resultScore}>{pasiResult.total_score.toFixed(1)}</Text>
            <View style={[styles.severityBadge, { backgroundColor: getSeverityColor(pasiResult.severity_category) }]}>
              <Text style={styles.severityText}>{pasiResult.severity_category}</Text>
            </View>
            <View style={styles.resultBreakdown}>
              {pasiResult.region_scores && Object.entries(pasiResult.region_scores).map(([key, value]: [string, any]) => (
                <Text key={key} style={styles.breakdownItem}>
                  {key.replace('_', ' ')}: {typeof value === 'number' ? value.toFixed(2) : value}
                </Text>
              ))}
            </View>
            <Text style={styles.resultInterpretation}>{pasiResult.interpretation || ''}</Text>
          </View>
        )}

        <Pressable
          style={[styles.submitButton, isSubmitting && styles.submitButtonDisabled]}
          onPress={calculatePasi}
          disabled={isSubmitting}
        >
          {isSubmitting ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.submitButtonText}>Calculate PASI</Text>
          )}
        </Pressable>
      </View>
      <View style={{ height: 40 }} />
    </ScrollView>
  );

  // Photo Comparison render
  const renderPhotoComparison = () => (
    <ScrollView style={styles.tabContent}>
      <View style={styles.form}>
        <Text style={styles.formTitle}>Treatment Photo Comparison</Text>
        <Text style={styles.formSubtitle}>Compare before and after lesion photos</Text>

        {/* Load history button */}
        {lesionHistory.length === 0 && (
          <Pressable style={styles.loadButton} onPress={loadLesionHistory}>
            <Text style={styles.loadButtonText}>Load Analysis History</Text>
          </Pressable>
        )}

        {isLoading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#4299e1" />
          </View>
        ) : lesionHistory.length > 0 ? (
          <>
            {/* Before Photo Selection */}
            <Text style={styles.comparisonLabel}>Select BEFORE Photo (Earlier)</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.photoScroll}>
              {lesionHistory.map((analysis: any) => (
                <Pressable
                  key={`before-${analysis.id}`}
                  style={[
                    styles.photoCard,
                    beforePhoto?.id === analysis.id && styles.photoCardSelected
                  ]}
                  onPress={() => setBeforePhoto(analysis)}
                >
                  {analysis.image_url ? (
                    <Image source={{ uri: `${API_BASE_URL}${analysis.image_url}` }} style={styles.photoThumbnail} />
                  ) : (
                    <View style={styles.photoPlaceholder}>
                      <Text style={styles.photoPlaceholderText}>No Image</Text>
                    </View>
                  )}
                  <Text style={styles.photoDate}>
                    {new Date(analysis.created_at).toLocaleDateString()}
                  </Text>
                  <Text style={styles.photoCondition} numberOfLines={1}>
                    {analysis.predicted_condition || 'Unknown'}
                  </Text>
                </Pressable>
              ))}
            </ScrollView>

            {/* After Photo Selection */}
            <Text style={[styles.comparisonLabel, { marginTop: 20 }]}>Select AFTER Photo (Later)</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.photoScroll}>
              {lesionHistory.map((analysis: any) => (
                <Pressable
                  key={`after-${analysis.id}`}
                  style={[
                    styles.photoCard,
                    afterPhoto?.id === analysis.id && styles.photoCardSelected
                  ]}
                  onPress={() => setAfterPhoto(analysis)}
                >
                  {analysis.image_url ? (
                    <Image source={{ uri: `${API_BASE_URL}${analysis.image_url}` }} style={styles.photoThumbnail} />
                  ) : (
                    <View style={styles.photoPlaceholder}>
                      <Text style={styles.photoPlaceholderText}>No Image</Text>
                    </View>
                  )}
                  <Text style={styles.photoDate}>
                    {new Date(analysis.created_at).toLocaleDateString()}
                  </Text>
                  <Text style={styles.photoCondition} numberOfLines={1}>
                    {analysis.predicted_condition || 'Unknown'}
                  </Text>
                </Pressable>
              ))}
            </ScrollView>

            {/* Selected Photos Preview */}
            {(beforePhoto || afterPhoto) && (
              <View style={styles.selectionPreview}>
                <View style={styles.previewColumn}>
                  <Text style={styles.previewLabel}>BEFORE</Text>
                  {beforePhoto ? (
                    <>
                      {beforePhoto.image_url ? (
                        <Image source={{ uri: `${API_BASE_URL}${beforePhoto.image_url}` }} style={styles.previewImage} />
                      ) : (
                        <View style={styles.previewPlaceholder}><Text>No Image</Text></View>
                      )}
                      <Text style={styles.previewDate}>{new Date(beforePhoto.created_at).toLocaleDateString()}</Text>
                    </>
                  ) : (
                    <View style={styles.previewPlaceholder}><Text>Select photo</Text></View>
                  )}
                </View>
                <View style={styles.previewArrow}>
                  <Text style={styles.arrowText}>→</Text>
                </View>
                <View style={styles.previewColumn}>
                  <Text style={styles.previewLabel}>AFTER</Text>
                  {afterPhoto ? (
                    <>
                      {afterPhoto.image_url ? (
                        <Image source={{ uri: `${API_BASE_URL}${afterPhoto.image_url}` }} style={styles.previewImage} />
                      ) : (
                        <View style={styles.previewPlaceholder}><Text>No Image</Text></View>
                      )}
                      <Text style={styles.previewDate}>{new Date(afterPhoto.created_at).toLocaleDateString()}</Text>
                    </>
                  ) : (
                    <View style={styles.previewPlaceholder}><Text>Select photo</Text></View>
                  )}
                </View>
              </View>
            )}

            <Pressable
              style={[styles.submitButton, (!beforePhoto || !afterPhoto || isSubmitting) && styles.submitButtonDisabled]}
              onPress={compareLesionPhotos}
              disabled={!beforePhoto || !afterPhoto || isSubmitting}
            >
              {isSubmitting ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.submitButtonText}>Compare Photos</Text>
              )}
            </Pressable>
          </>
        ) : (
          <View style={styles.emptyState}>
            <Text style={styles.emptyStateText}>No Analysis History</Text>
            <Text style={styles.emptyStateSubtext}>Take photos of lesions first to compare them over time</Text>
          </View>
        )}
      </View>

      {/* Comparison Results Modal */}
      <Modal
        visible={showComparisonModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowComparisonModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Comparison Results</Text>

            {comparisonResult && (
              <ScrollView style={styles.modalScroll}>
                {/* Time difference */}
                <View style={styles.comparisonSection}>
                  <Text style={styles.comparisonSectionTitle}>Time Between Photos</Text>
                  <Text style={styles.comparisonValue}>{comparisonResult.time_difference || 'N/A'}</Text>
                </View>

                {/* Size change */}
                {comparisonResult.size_change && (
                  <View style={styles.comparisonSection}>
                    <Text style={styles.comparisonSectionTitle}>Size Change</Text>
                    <Text style={[
                      styles.comparisonValue,
                      { color: comparisonResult.size_change.percent < 0 ? '#10b981' : '#ef4444' }
                    ]}>
                      {comparisonResult.size_change.percent > 0 ? '+' : ''}{comparisonResult.size_change.percent?.toFixed(1)}%
                    </Text>
                    <Text style={styles.comparisonDetail}>
                      {comparisonResult.size_change.before_mm?.toFixed(1)}mm → {comparisonResult.size_change.after_mm?.toFixed(1)}mm
                    </Text>
                  </View>
                )}

                {/* Confidence change */}
                {comparisonResult.confidence_change !== undefined && (
                  <View style={styles.comparisonSection}>
                    <Text style={styles.comparisonSectionTitle}>Classification Confidence</Text>
                    <Text style={styles.comparisonDetail}>
                      Before: {(comparisonResult.confidence_before * 100)?.toFixed(1)}%
                    </Text>
                    <Text style={styles.comparisonDetail}>
                      After: {(comparisonResult.confidence_after * 100)?.toFixed(1)}%
                    </Text>
                  </View>
                )}

                {/* Overall assessment */}
                {comparisonResult.overall_assessment && (
                  <View style={styles.comparisonSection}>
                    <Text style={styles.comparisonSectionTitle}>Assessment</Text>
                    <Text style={styles.comparisonAssessment}>{comparisonResult.overall_assessment}</Text>
                  </View>
                )}

                {/* Recommendations */}
                {comparisonResult.recommendations && comparisonResult.recommendations.length > 0 && (
                  <View style={styles.comparisonSection}>
                    <Text style={styles.comparisonSectionTitle}>Recommendations</Text>
                    {comparisonResult.recommendations.map((rec: string, index: number) => (
                      <Text key={index} style={styles.recommendationItem}>• {rec}</Text>
                    ))}
                  </View>
                )}
              </ScrollView>
            )}

            <Pressable style={styles.closeButton} onPress={() => setShowComparisonModal(false)}>
              <Text style={styles.closeButtonText}>Close</Text>
            </Pressable>
          </View>
        </View>
      </Modal>

      <View style={{ height: 40 }} />
    </ScrollView>
  );

  // Helper function for severity colors
  const getSeverityColor = (severity: string): string => {
    switch (severity?.toLowerCase()) {
      case 'mild': return '#10b981';
      case 'moderate': return '#f59e0b';
      case 'severe': return '#ef4444';
      case 'very severe': return '#7c2d12';
      case 'clear': return '#22c55e';
      default: return '#6b7280';
    }
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.background}
      />

      <View style={styles.header}>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Text style={styles.backButtonText}>← {t('common.back')}</Text>
        </Pressable>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>💊 {t('treatmentMonitoring.title')}</Text>
          <Text style={styles.headerSubtitle}>{t('treatmentMonitoring.subtitle')}</Text>
        </View>
      </View>

      {/* Primary Tab Bar */}
      <View style={styles.tabBar}>
        <Pressable
          style={[styles.tab, activeTab === 'treatments' && styles.tabActive]}
          onPress={() => setActiveTab('treatments')}
        >
          <Text style={[styles.tabText, activeTab === 'treatments' && styles.tabTextActive]}>
            {t('treatmentMonitoring.tabs.treatments')}
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tab, activeTab === 'add' && styles.tabActive]}
          onPress={() => setActiveTab('add')}
        >
          <Text style={[styles.tabText, activeTab === 'add' && styles.tabTextActive]}>
            {t('treatmentMonitoring.tabs.addNew')}
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tab, activeTab === 'log' && styles.tabActive]}
          onPress={() => setActiveTab('log')}
        >
          <Text style={[styles.tabText, activeTab === 'log' && styles.tabTextActive]}>
            {t('treatmentMonitoring.tabs.logTrack')}
          </Text>
        </Pressable>
      </View>

      {/* Secondary Tab Bar for Calculators & Comparison */}
      <View style={styles.secondaryTabBar}>
        <Pressable
          style={[styles.secondaryTab, activeTab === 'scorad' && styles.secondaryTabActive]}
          onPress={() => setActiveTab('scorad')}
        >
          <Text style={[styles.secondaryTabText, activeTab === 'scorad' && styles.secondaryTabTextActive]}>
            SCORAD
          </Text>
        </Pressable>
        <Pressable
          style={[styles.secondaryTab, activeTab === 'pasi' && styles.secondaryTabActive]}
          onPress={() => setActiveTab('pasi')}
        >
          <Text style={[styles.secondaryTabText, activeTab === 'pasi' && styles.secondaryTabTextActive]}>
            PASI
          </Text>
        </Pressable>
        <Pressable
          style={[styles.secondaryTab, activeTab === 'compare' && styles.secondaryTabActive]}
          onPress={() => {
            setActiveTab('compare');
            if (lesionHistory.length === 0) loadLesionHistory();
          }}
        >
          <Text style={[styles.secondaryTabText, activeTab === 'compare' && styles.secondaryTabTextActive]}>
            Photo Compare
          </Text>
        </Pressable>
      </View>

      {activeTab === 'treatments' && renderTreatmentsList()}
      {activeTab === 'add' && renderAddForm()}
      {activeTab === 'log' && renderLogForm()}
      {activeTab === 'scorad' && renderScoradCalculator()}
      {activeTab === 'pasi' && renderPasiCalculator()}
      {activeTab === 'compare' && renderPhotoComparison()}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  background: { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0 },
  header: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 20, paddingTop: 60, paddingBottom: 20, borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.2)' },
  backButton: { backgroundColor: 'rgba(66, 153, 225, 0.9)', paddingHorizontal: 15, paddingVertical: 8, borderRadius: 20, marginRight: 15 },
  backButtonText: { color: 'white', fontSize: 14, fontWeight: 'bold' },
  headerContent: { flex: 1 },
  headerTitle: { fontSize: 24, fontWeight: 'bold', color: '#2c5282' },
  headerSubtitle: { fontSize: 14, color: '#4a5568', marginTop: 4 },
  tabBar: { flexDirection: 'row', backgroundColor: 'rgba(255, 255, 255, 0.9)', borderBottomWidth: 1, borderBottomColor: '#e2e8f0' },
  tab: { flex: 1, paddingVertical: 16, alignItems: 'center', borderBottomWidth: 2, borderBottomColor: 'transparent' },
  tabActive: { borderBottomColor: '#4299e1' },
  tabText: { fontSize: 14, fontWeight: '600', color: '#718096' },
  tabTextActive: { color: '#4299e1' },
  tabContent: { flex: 1, paddingHorizontal: 20, paddingVertical: 20 },
  loadingContainer: { paddingVertical: 40, alignItems: 'center' },
  emptyState: { paddingVertical: 60, alignItems: 'center' },
  emptyStateText: { fontSize: 20, color: '#4a5568', marginBottom: 8 },
  emptyStateSubtext: { fontSize: 14, color: '#718096', textAlign: 'center', marginBottom: 20 },
  addButton: { backgroundColor: '#4299e1', paddingHorizontal: 24, paddingVertical: 12, borderRadius: 8 },
  addButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  treatmentCard: { backgroundColor: '#fff', borderRadius: 12, padding: 16, marginBottom: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 2, elevation: 2 },
  treatmentCardSelected: { borderWidth: 2, borderColor: '#4299e1' },
  treatmentHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  treatmentName: { fontSize: 18, fontWeight: 'bold', color: '#2d3748', flex: 1 },
  activeBadge: { backgroundColor: '#10b981', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8 },
  activeBadgeText: { color: '#fff', fontSize: 11, fontWeight: 'bold' },
  treatmentDetail: { fontSize: 13, color: '#64748b', marginBottom: 4 },
  effectivenessPreview: { marginTop: 12, paddingTop: 12, borderTopWidth: 1, borderTopColor: '#e2e8f0' },
  effectivenessTitle: { fontSize: 14, fontWeight: '600', color: '#4a5568', marginBottom: 6 },
  effectivenessText: { fontSize: 13, color: '#64748b', marginBottom: 2 },
  actionButtons: { flexDirection: 'row', gap: 12, marginTop: 12 },
  actionButton: { flex: 1, backgroundColor: '#4299e1', paddingVertical: 12, borderRadius: 8, alignItems: 'center' },
  actionButtonSecondary: { backgroundColor: '#10b981' },
  actionButtonText: { color: '#fff', fontSize: 14, fontWeight: 'bold' },
  form: { backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: 16, padding: 20 },
  formTitle: { fontSize: 20, fontWeight: 'bold', color: '#2c5282', marginBottom: 4 },
  formSubtitle: { fontSize: 14, color: '#64748b', marginBottom: 16 },
  label: { fontSize: 14, fontWeight: '600', color: '#4a5568', marginBottom: 8, marginTop: 12 },
  input: { backgroundColor: '#f7fafc', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 10, fontSize: 14, borderWidth: 1, borderColor: '#e2e8f0' },
  textArea: { height: 80, textAlignVertical: 'top' },
  buttonGroup: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 8 },
  optionButton: { backgroundColor: '#f7fafc', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, borderWidth: 1, borderColor: '#e2e8f0' },
  optionButtonActive: { backgroundColor: '#4299e1', borderColor: '#4299e1' },
  optionButtonText: { fontSize: 13, color: '#4a5568', fontWeight: '500' },
  optionButtonTextActive: { color: '#fff' },
  switchRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#e2e8f0' },
  switchLabel: { fontSize: 14, color: '#2d3748' },
  submitButton: { backgroundColor: '#4299e1', paddingVertical: 16, borderRadius: 12, alignItems: 'center', marginTop: 20 },
  submitButtonDisabled: { opacity: 0.6 },
  submitButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  starRating: { flexDirection: 'row', gap: 8, marginTop: 8 },
  star: { fontSize: 32 },
  checkboxGroup: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 8 },
  checkbox: { backgroundColor: '#f7fafc', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, borderWidth: 1, borderColor: '#e2e8f0' },
  checkboxChecked: { backgroundColor: '#10b981', borderColor: '#10b981' },
  checkboxText: { fontSize: 13, color: '#2d3748' },

  // Secondary tab bar styles
  secondaryTabBar: { flexDirection: 'row', backgroundColor: 'rgba(240, 248, 255, 0.95)', paddingVertical: 8, paddingHorizontal: 12, gap: 8 },
  secondaryTab: { flex: 1, paddingVertical: 10, paddingHorizontal: 12, borderRadius: 20, backgroundColor: '#e2e8f0', alignItems: 'center' },
  secondaryTabActive: { backgroundColor: '#4299e1' },
  secondaryTabText: { fontSize: 12, fontWeight: '600', color: '#64748b' },
  secondaryTabTextActive: { color: '#fff' },

  // SCORAD styles
  sectionHeader: { marginTop: 16, marginBottom: 12 },
  sectionTitle: { fontSize: 16, fontWeight: 'bold', color: '#2c5282' },
  sectionSubtitle: { fontSize: 12, color: '#718096', marginTop: 2 },
  regionRow: { marginBottom: 16 },
  regionInfo: { marginBottom: 8 },
  regionName: { fontSize: 14, fontWeight: '600', color: '#2d3748' },
  regionPercent: { fontSize: 11, color: '#718096' },
  sliderContainer: { marginTop: 4 },
  sliderLabels: { flexDirection: 'row', justifyContent: 'space-between', gap: 4 },
  sliderOption: { flex: 1, paddingVertical: 8, backgroundColor: '#f7fafc', borderRadius: 6, alignItems: 'center', borderWidth: 1, borderColor: '#e2e8f0' },
  sliderOptionActive: { backgroundColor: '#4299e1', borderColor: '#4299e1' },
  sliderOptionText: { fontSize: 11, color: '#4a5568', fontWeight: '500' },
  sliderOptionTextActive: { color: '#fff' },
  intensityRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#e2e8f0' },
  intensityLabel: { flex: 1, fontSize: 13, color: '#2d3748' },
  intensityButtons: { flexDirection: 'row', gap: 6 },
  intensityButton: { width: 36, height: 36, borderRadius: 18, backgroundColor: '#f7fafc', alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: '#e2e8f0' },
  intensityButtonActive: { backgroundColor: '#4299e1', borderColor: '#4299e1' },
  intensityButtonText: { fontSize: 14, fontWeight: '600', color: '#4a5568' },
  intensityButtonTextActive: { color: '#fff' },
  vasRow: { marginBottom: 16 },
  vasLabel: { fontSize: 14, fontWeight: '600', color: '#2d3748', marginBottom: 8 },
  vasButtons: { flexDirection: 'row', flexWrap: 'wrap', gap: 4 },
  vasButton: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#f7fafc', alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: '#e2e8f0' },
  vasButtonActive: { backgroundColor: '#4299e1', borderColor: '#4299e1' },
  vasButtonText: { fontSize: 11, fontWeight: '500', color: '#4a5568' },
  vasButtonTextActive: { color: '#fff' },
  resultCard: { backgroundColor: '#f0f9ff', borderRadius: 16, padding: 20, marginTop: 20, alignItems: 'center' },
  resultTitle: { fontSize: 14, fontWeight: '600', color: '#4a5568', marginBottom: 8 },
  resultScore: { fontSize: 48, fontWeight: 'bold', color: '#2c5282' },
  severityBadge: { paddingHorizontal: 16, paddingVertical: 6, borderRadius: 16, marginTop: 8 },
  severityText: { fontSize: 14, fontWeight: 'bold', color: '#fff', textTransform: 'uppercase' },
  resultBreakdown: { marginTop: 16, width: '100%' },
  breakdownItem: { fontSize: 13, color: '#4a5568', marginBottom: 4 },
  resultInterpretation: { fontSize: 12, color: '#64748b', marginTop: 12, textAlign: 'center', fontStyle: 'italic' },

  // PASI styles
  pasiRegionCard: { backgroundColor: '#f8fafc', borderRadius: 12, padding: 16, marginBottom: 16 },
  pasiRegionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  pasiRegionName: { fontSize: 16, fontWeight: 'bold', color: '#2c5282' },
  pasiRegionWeight: { fontSize: 12, color: '#718096' },
  pasiLabel: { fontSize: 13, fontWeight: '600', color: '#4a5568', marginTop: 12, marginBottom: 6 },
  pasiScaleRow: { flexDirection: 'row', gap: 4 },
  pasiScaleButton: { flex: 1, paddingVertical: 8, backgroundColor: '#fff', borderRadius: 6, alignItems: 'center', borderWidth: 1, borderColor: '#e2e8f0' },
  pasiScaleButtonActive: { backgroundColor: '#4299e1', borderColor: '#4299e1' },
  pasiScaleText: { fontSize: 12, fontWeight: '600', color: '#4a5568' },
  pasiScaleTextActive: { color: '#fff' },
  pasiScaleHint: { fontSize: 10, color: '#94a3b8', marginTop: 4 },

  // Photo comparison styles
  loadButton: { backgroundColor: '#4299e1', paddingVertical: 14, borderRadius: 10, alignItems: 'center', marginVertical: 20 },
  loadButtonText: { color: '#fff', fontSize: 14, fontWeight: 'bold' },
  comparisonLabel: { fontSize: 14, fontWeight: '600', color: '#2c5282', marginBottom: 8 },
  photoScroll: { marginBottom: 8 },
  photoCard: { width: 100, marginRight: 12, borderRadius: 10, overflow: 'hidden', backgroundColor: '#f7fafc', borderWidth: 2, borderColor: 'transparent' },
  photoCardSelected: { borderColor: '#4299e1' },
  photoThumbnail: { width: '100%', height: 80, resizeMode: 'cover' },
  photoPlaceholder: { width: '100%', height: 80, backgroundColor: '#e2e8f0', alignItems: 'center', justifyContent: 'center' },
  photoPlaceholderText: { fontSize: 10, color: '#94a3b8' },
  photoDate: { fontSize: 10, color: '#4a5568', padding: 4, textAlign: 'center' },
  photoCondition: { fontSize: 9, color: '#718096', paddingHorizontal: 4, paddingBottom: 4, textAlign: 'center' },
  selectionPreview: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginTop: 20, marginBottom: 20 },
  previewColumn: { flex: 1, alignItems: 'center' },
  previewLabel: { fontSize: 12, fontWeight: 'bold', color: '#4a5568', marginBottom: 8 },
  previewImage: { width: 120, height: 120, borderRadius: 12, resizeMode: 'cover' },
  previewPlaceholder: { width: 120, height: 120, borderRadius: 12, backgroundColor: '#e2e8f0', alignItems: 'center', justifyContent: 'center' },
  previewDate: { fontSize: 11, color: '#64748b', marginTop: 4 },
  previewArrow: { paddingHorizontal: 12 },
  arrowText: { fontSize: 24, color: '#4299e1' },

  // Modal styles
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'center', alignItems: 'center', padding: 20 },
  modalContent: { backgroundColor: '#fff', borderRadius: 20, padding: 24, width: '100%', maxHeight: '80%' },
  modalTitle: { fontSize: 20, fontWeight: 'bold', color: '#2c5282', marginBottom: 16, textAlign: 'center' },
  modalScroll: { maxHeight: 400 },
  comparisonSection: { backgroundColor: '#f8fafc', borderRadius: 12, padding: 16, marginBottom: 12 },
  comparisonSectionTitle: { fontSize: 14, fontWeight: '600', color: '#4a5568', marginBottom: 8 },
  comparisonValue: { fontSize: 24, fontWeight: 'bold', color: '#2c5282' },
  comparisonDetail: { fontSize: 13, color: '#64748b', marginTop: 4 },
  comparisonAssessment: { fontSize: 14, color: '#2d3748', lineHeight: 20 },
  recommendationItem: { fontSize: 13, color: '#4a5568', marginBottom: 4 },
  closeButton: { backgroundColor: '#4299e1', paddingVertical: 14, borderRadius: 10, alignItems: 'center', marginTop: 16 },
  closeButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
});
