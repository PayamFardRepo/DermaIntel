import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  Pressable,
  Alert,
  Image,
  Dimensions,
  Linking,
  Modal,
  Platform,
  TextInput,
  Clipboard,
  TouchableOpacity
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../../contexts/AuthContext';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { useTranslation } from 'react-i18next';
import * as SecureStore from 'expo-secure-store';
import AnalysisHistoryService from '../../services/AnalysisHistoryService';
import AuthService from '../../services/AuthService';
import * as Sharing from 'expo-sharing';
import DoctorSearchService from '../../services/DoctorSearchService';
import { API_BASE_URL } from '../../config';
import MeasurementTool from '../../components/MeasurementTool';
import SymptomTracker from '../../components/SymptomTracker';
import MedicationList, { Medication } from '../../components/MedicationList';
import MedicalHistory, { MedicalHistoryData } from '../../components/MedicalHistory';
import TeledermatologyShare from '../../components/TeledermatologyShare';
import * as Speech from 'expo-speech';
// Use legacy FileSystem API to avoid deprecation errors
import * as FileSystemLegacy from 'expo-file-system/legacy';

const screenWidth = Dimensions.get('window').width;

export default function AnalysisDetailScreen() {
  const { t } = useTranslation();
  const [analysis, setAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showMeasurementTool, setShowMeasurementTool] = useState(false);
  const [measurements, setMeasurements] = useState([]);
  const [showBiopsyForm, setShowBiopsyForm] = useState(false);
  const [biopsyData, setBiopsyData] = useState({
    biopsy_result: '',
    biopsy_date: '',
    biopsy_notes: '',
    biopsy_facility: '',
    pathologist_name: ''
  });
  const [showSymptomForm, setShowSymptomForm] = useState(false);
  const [symptomData, setSymptomData] = useState<any>(null);
  const [showMedicationForm, setShowMedicationForm] = useState(false);
  const [medicationData, setMedicationData] = useState<Medication[]>([]);
  const [showMedicalHistoryForm, setShowMedicalHistoryForm] = useState(false);
  const [medicalHistoryData, setMedicalHistoryData] = useState<MedicalHistoryData | null>(null);
  const [showTeledermatologyForm, setShowTeledermatologyForm] = useState(false);
  const [showDoctorSearchModal, setShowDoctorSearchModal] = useState(false);
  const [selectedSpecialist, setSelectedSpecialist] = useState(null);
  const [nearbyDoctors, setNearbyDoctors] = useState([]);
  const [isLoadingDoctors, setIsLoadingDoctors] = useState(false);
  const [userLocation, setUserLocation] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [showFormModal, setShowFormModal] = useState(false);
  const [selectedFormType, setSelectedFormType] = useState<'cms1500' | 'ub04' | 'generic' | null>(null);
  const [selectedFormData, setSelectedFormData] = useState<any>(null);
  const [labResults, setLabResults] = useState<any>(null);
  const [isLoadingLabResults, setIsLoadingLabResults] = useState(false);
  const [suggestedLabTests, setSuggestedLabTests] = useState<any>(null);

  // AI Explanation state
  const [aiExplanation, setAiExplanation] = useState<string | null>(null);
  const [isLoadingAiExplanation, setIsLoadingAiExplanation] = useState(false);
  const [showAiExplanation, setShowAiExplanation] = useState(false);
  const [aiExplanationError, setAiExplanationError] = useState<string | null>(null);

  // Differential Reasoning state
  const [differentialReasoning, setDifferentialReasoning] = useState<string | null>(null);
  const [isLoadingReasoning, setIsLoadingReasoning] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const [reasoningError, setReasoningError] = useState<string | null>(null);

  const { user, isAuthenticated, logout } = useAuth();
  const router = useRouter();
  const params = useLocalSearchParams();
  const id = Array.isArray(params.id) ? params.id[0] : params.id;

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    }
  }, [isAuthenticated]);

  const loadAnalysisDetail = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const analysisData = await AnalysisHistoryService.getAnalysisById(parseInt(id));
      console.log('=== ANALYSIS DATA ===');
      console.log('Full data:', JSON.stringify(analysisData, null, 2));
      console.log('Has literature_references?', !!analysisData.literature_references);
      console.log('Literature references:', analysisData.literature_references);
      console.log('=== CLINICAL DECISION SUPPORT DEBUG ===');
      console.log('Has clinical_decision_support?', !!analysisData.clinical_decision_support);
      console.log('Clinical decision support data:', analysisData.clinical_decision_support);
      console.log('=== LAB CONTEXT DEBUG ===');
      console.log('Has lab_context?', !!analysisData.lab_context);
      console.log('Lab context data:', JSON.stringify(analysisData.lab_context, null, 2));
      console.log('=== OTC RECOMMENDATIONS DEBUG ===');
      console.log('Has otc_recommendations?', !!analysisData.otc_recommendations);
      console.log('OTC applicable?', analysisData.otc_recommendations?.applicable);
      console.log('OTC data:', JSON.stringify(analysisData.otc_recommendations, null, 2));
      console.log('=== SYMPTOM DATA ===');
      console.log('symptom_duration:', analysisData.symptom_duration);
      console.log('symptom_itching:', analysisData.symptom_itching);
      console.log('symptom_pain:', analysisData.symptom_pain);
      console.log('symptom_bleeding:', analysisData.symptom_bleeding);
      console.log('symptom_changes:', analysisData.symptom_changes);
      console.log('symptom_notes:', analysisData.symptom_notes);
      setAnalysis(analysisData);

    } catch (error) {
      console.error('Failed to load analysis detail:', error);
      setError(error.message);

      if (error.message.includes('Authentication') || error.message.includes('401')) {
        Alert.alert(
          'Session Expired',
          'Please log in again to view this analysis.',
          [{ text: 'OK', onPress: () => logout() }]
        );
      } else if (error.message.includes('not found')) {
        Alert.alert(
          'Analysis Not Found',
          'This analysis could not be found or you do not have permission to view it.',
          [{ text: 'OK', onPress: () => router.back() }]
        );
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isAuthenticated && id) {
      loadAnalysisDetail();
    }
  }, [isAuthenticated, id]);

  // Function to fetch AI explanation for a condition
  const fetchAIExplanation = async (condition: string, severity?: string) => {
    if (!condition || condition === 'Unknown') {
      setAiExplanationError('No condition to explain');
      return;
    }

    setIsLoadingAiExplanation(true);
    setAiExplanationError(null);
    setShowAiExplanation(true);

    try {
      const token = await SecureStore.getItemAsync('auth_token');

      const formData = new FormData();
      formData.append('condition', condition);
      if (severity) {
        formData.append('severity', severity);
      }

      const response = await fetch(`${API_BASE_URL}/ai-chat/explain-condition`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get AI explanation');
      }

      const data = await response.json();
      setAiExplanation(data.explanation);
    } catch (error: any) {
      console.error('AI explanation error:', error);
      setAiExplanationError(error.message || 'Failed to load explanation. Please try again.');
    } finally {
      setIsLoadingAiExplanation(false);
    }
  };

  // Function to fetch differential reasoning
  const fetchDifferentialReasoning = async () => {
    if (!analysis) return;

    setIsLoadingReasoning(true);
    setReasoningError(null);
    setShowReasoning(true);

    try {
      const token = await SecureStore.getItemAsync('auth_token');

      const formData = new FormData();
      formData.append('primary_diagnosis', analysis.predicted_class || analysis.inflammatory_condition || 'Unknown');

      if (analysis.confidence) {
        formData.append('confidence', analysis.confidence.toString());
      }

      // Include differential diagnoses if available
      if (analysis.differential_diagnoses) {
        let differentials = [];
        if (analysis.differential_diagnoses.primary) {
          differentials = differentials.concat(analysis.differential_diagnoses.primary);
        }
        if (analysis.differential_diagnoses.inflammatory) {
          differentials = differentials.concat(analysis.differential_diagnoses.inflammatory);
        }
        if (analysis.differential_diagnoses.infectious) {
          differentials = differentials.concat(analysis.differential_diagnoses.infectious);
        }
        if (differentials.length > 0) {
          formData.append('differential_diagnoses', JSON.stringify(differentials.slice(0, 5)));
        }
      }

      // Include clinical context if available
      const clinicalContext: any = {};
      if (analysis.body_location) clinicalContext.body_location = analysis.body_location;
      if (analysis.symptom_duration) clinicalContext.symptom_duration = analysis.symptom_duration;
      if (analysis.symptom_itching !== undefined) clinicalContext.itching = analysis.symptom_itching;
      if (analysis.symptom_pain !== undefined) clinicalContext.pain = analysis.symptom_pain;
      if (analysis.symptom_bleeding !== undefined) clinicalContext.bleeding = analysis.symptom_bleeding;
      if (Object.keys(clinicalContext).length > 0) {
        formData.append('clinical_context', JSON.stringify(clinicalContext));
      }

      if (analysis.risk_level) {
        formData.append('risk_level', analysis.risk_level);
      }

      const response = await fetch(`${API_BASE_URL}/ai-chat/differential-reasoning`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get differential reasoning');
      }

      const data = await response.json();
      setDifferentialReasoning(data.reasoning);
    } catch (error: any) {
      console.error('Differential reasoning error:', error);
      setReasoningError(error.message || 'Failed to load reasoning. Please try again.');
    } finally {
      setIsLoadingReasoning(false);
    }
  };

  // Load lab results when analysis is available
  const loadLabResultsInsights = async () => {
    if (!analysis?.predicted_class) return;

    try {
      setIsLoadingLabResults(true);
      const token = AuthService.getToken();
      if (!token) return;

      // Fetch user's most recent lab results
      const labResponse = await fetch(`${API_BASE_URL}/lab-results`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (labResponse.ok) {
        const labData = await labResponse.json();
        if (labData.length > 0) {
          // Get the most recent lab result with full analysis
          const recentLab = labData[0];
          const analysisResponse = await fetch(`${API_BASE_URL}/lab-results/analysis/${recentLab.id}`, {
            headers: { 'Authorization': `Bearer ${token}` }
          });
          if (analysisResponse.ok) {
            const analysisData = await analysisResponse.json();
            setLabResults(analysisData);
          }
        }
      }

      // Fetch suggested lab tests for this skin condition
      const suggestedResponse = await fetch(
        `${API_BASE_URL}/lab-results/suggested-tests/${encodeURIComponent(analysis.predicted_class)}`,
        { headers: { 'Authorization': `Bearer ${token}` } }
      );

      if (suggestedResponse.ok) {
        const suggestedData = await suggestedResponse.json();
        setSuggestedLabTests(suggestedData);
      }
    } catch (error) {
      console.log('Lab results not available:', error);
    } finally {
      setIsLoadingLabResults(false);
    }
  };

  useEffect(() => {
    if (analysis?.predicted_class) {
      loadLabResultsInsights();
    }
  }, [analysis?.predicted_class]);

  const exportPreAuthPDF = async () => {
    try {
      // Get authentication token
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert('Error', 'Authentication required. Please log in again.');
        return;
      }

      // Download PDF from backend using legacy FileSystem API (stable)
      const downloadUrl = `${API_BASE_URL}/analysis/preauth-pdf/${id}`;
      const fileName = `PreAuth_${id}_${Date.now()}.pdf`;
      const fileUri = FileSystemLegacy.documentDirectory + fileName;

      console.log('Downloading PDF from:', downloadUrl);
      console.log('Saving to:', fileUri);

      // Download using legacy API with proper authentication
      const downloadResult = await FileSystemLegacy.downloadAsync(
        downloadUrl,
        fileUri,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );

      console.log('PDF downloaded successfully:', downloadResult.uri);

      // Show success alert and offer to share immediately
      Alert.alert(
        '✅ PDF Ready',
        'Your pre-authorization PDF has been generated successfully!',
        [
          {
            text: 'Open & Share',
            onPress: async () => {
              // Check if sharing is available
              const isAvailable = await Sharing.isAvailableAsync();
              if (isAvailable) {
                await Sharing.shareAsync(downloadResult.uri, {
                  mimeType: 'application/pdf',
                  dialogTitle: 'Save or Share Pre-Authorization PDF',
                  UTI: 'com.adobe.pdf'
                });
              } else {
                Alert.alert(
                  'PDF Downloaded',
                  `PDF has been saved to: ${downloadResult.uri}`,
                  [{ text: 'OK' }]
                );
              }
            }
          },
          {
            text: 'Later',
            style: 'cancel'
          }
        ]
      );
    } catch (error) {
      console.error('PDF export error:', error);
      Alert.alert(
        'Export Failed',
        `Could not generate PDF: ${error.message}`,
        [{ text: 'OK' }]
      );
    }
  };

  const sharePreAuthDocumentation = async () => {
    try {
      if (!analysis?.insurance_preauthorization) {
        Alert.alert('Error', 'No pre-authorization data available');
        return;
      }

      const preauth = analysis.insurance_preauthorization;

      // Prepare text content
      let shareText = '📋 INSURANCE PRE-AUTHORIZATION DOCUMENTATION\n\n';

      // Add summary
      if (preauth.form_data) {
        shareText += '=== AUTHORIZATION SUMMARY ===\n';
        shareText += `Diagnosis: ${preauth.form_data.diagnosis?.primary_diagnosis || analysis.predicted_class}\n`;
        shareText += `ICD-10: ${preauth.form_data.diagnosis?.icd10_code || 'N/A'}\n`;
        shareText += `Urgency: ${preauth.form_data.urgency || 'N/A'}\n`;
        shareText += `Confidence: ${preauth.form_data.diagnosis?.confidence_level || 'N/A'}\n\n`;

        // Add procedures
        if (preauth.form_data.procedures_requested && preauth.form_data.procedures_requested.length > 0) {
          shareText += '=== REQUESTED PROCEDURES ===\n';
          preauth.form_data.procedures_requested.forEach((proc, idx) => {
            shareText += `${idx + 1}. ${proc.description} (CPT: ${proc.code})\n`;
            shareText += `   ${proc.rationale}\n\n`;
          });
        }
      }

      // Add clinical summary preview
      if (preauth.clinical_summary) {
        shareText += '=== CLINICAL SUMMARY ===\n';
        shareText += preauth.clinical_summary.substring(0, 500) + '...\n\n';
      }

      shareText += '---\n';
      shareText += 'Generated by Skin Classifier AI\n';
      shareText += 'For complete documentation including medical necessity letter and supporting evidence, please export as PDF.\n';

      // Show share options
      Alert.alert(
        'Share Documentation',
        'Choose how you would like to share:',
        [
          {
            text: 'Copy Text',
            onPress: () => {
              Clipboard.setString(shareText);
              Alert.alert('Copied', 'Pre-authorization text copied to clipboard');
            }
          },
          {
            text: 'Export PDF',
            onPress: exportPreAuthPDF
          },
          {
            text: 'Cancel',
            style: 'cancel'
          }
        ]
      );
    } catch (error) {
      console.error('Share error:', error);
      Alert.alert('Error', `Could not share documentation: ${error.message}`);
    }
  };

  const updatePreAuthStatus = async (newStatus: string) => {
    try {
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert('Error', 'Authentication required');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/analysis/preauth-status/${id}?status=${newStatus}`, {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to update status');
      }

      // Reload the analysis to get updated data
      await loadAnalysisDetail();

      Alert.alert('Success', `Status updated to: ${newStatus.replace('_', ' ')}`);
    } catch (error) {
      console.error('Status update error:', error);
      Alert.alert('Error', `Could not update status: ${error.message}`);
    }
  };

  const openFormModal = (formType: 'cms1500' | 'ub04' | 'generic', formData: any) => {
    console.log('=== OPENING FORM MODAL ===');
    console.log('Form Type:', formType);
    console.log('Form Data Keys:', Object.keys(formData || {}));
    console.log('Full Form Data:', JSON.stringify(formData, null, 2));
    setSelectedFormType(formType);
    setSelectedFormData(formData);
    setShowFormModal(true);
  };

  const renderFormField = (label: string, value: any, depth: number = 0) => {
    const indent = depth * 20;

    if (value === null || value === undefined) {
      return null;
    }

    // Skip metadata fields that aren't part of the form itself
    if (['form_type', 'form_version', 'provider', 'generated_at', 'instructions'].includes(label)) {
      return null;
    }

    if (typeof value === 'object' && !Array.isArray(value)) {
      return (
        <View key={label} style={[styles.formSection, { marginLeft: indent }]}>
          <Text style={styles.formSectionTitle}>
            {depth === 0 ? '📋 ' : ''}
            {label.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </Text>
          {Object.entries(value).map(([key, val]) => renderFormField(key, val, depth + 1))}
        </View>
      );
    }

    if (Array.isArray(value)) {
      return (
        <View key={label} style={[styles.formSection, { marginLeft: indent }]}>
          <Text style={styles.formSectionTitle}>
            {label.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} ({value.length} items)
          </Text>
          {value.map((item, index) => (
            <View key={index} style={styles.formArrayItem}>
              <Text style={styles.formArrayItemHeader}>
                {label.includes('service') ? `Service Line ${index + 1}` :
                 label.includes('revenue') ? `Revenue Code ${index + 1}` :
                 `Item ${index + 1}`}
              </Text>
              {typeof item === 'object' ? (
                Object.entries(item).map(([key, val]) => renderFormField(key, val, depth + 1))
              ) : (
                <Text style={styles.formFieldValue}>• {item}</Text>
              )}
            </View>
          ))}
        </View>
      );
    }

    // Highlight placeholder fields
    const isPlaceholder = String(value).startsWith('[') && String(value).endsWith(']');

    return (
      <View key={label} style={[styles.formField, { marginLeft: indent }]}>
        <Text style={styles.formFieldLabel}>
          {label.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
        </Text>
        <Text style={[
          styles.formFieldValue,
          isPlaceholder && styles.formFieldPlaceholder
        ]}>
          {String(value)}
        </Text>
      </View>
    );
  };

  const formatAnalysisForDisplay = (analysis) => {
    return AnalysisHistoryService.formatAnalysisForDisplay(analysis);
  };

  // Get recommended specialists based on the predicted condition
  const getRecommendedSpecialists = (predictedClass: string) => {
    const specialistMap: { [key: string]: Array<{ name: string; reason: string; urgency: 'urgent' | 'recommended' | 'optional' }> } = {
      'Melanoma': [
        { name: 'Dermatologist', reason: 'For diagnosis confirmation and surgical planning', urgency: 'urgent' },
        { name: 'Surgical Oncologist', reason: 'For melanoma excision and staging', urgency: 'urgent' },
        { name: 'Medical Oncologist', reason: 'For advanced melanoma treatment planning', urgency: 'urgent' },
        { name: 'Radiation Oncologist', reason: 'If radiation therapy is needed', urgency: 'recommended' }
      ],
      'Basal Cell Carcinoma': [
        { name: 'Dermatologist', reason: 'For diagnosis confirmation and treatment options', urgency: 'urgent' },
        { name: 'Mohs Surgeon', reason: 'For precise removal with tissue preservation', urgency: 'recommended' },
        { name: 'Plastic Surgeon', reason: 'For reconstruction after removal', urgency: 'optional' }
      ],
      'Squamous Cell Carcinoma': [
        { name: 'Dermatologist', reason: 'For diagnosis and treatment planning', urgency: 'urgent' },
        { name: 'Mohs Surgeon', reason: 'For complete removal with margin control', urgency: 'recommended' },
        { name: 'Surgical Oncologist', reason: 'For high-risk or metastatic cases', urgency: 'recommended' }
      ],
      'Actinic Keratoses': [
        { name: 'Dermatologist', reason: 'For treatment and cancer prevention', urgency: 'recommended' },
        { name: 'Primary Care Physician', reason: 'For routine monitoring', urgency: 'optional' }
      ],
      'Atopic Dermatitis (Eczema)': [
        { name: 'Dermatologist', reason: 'For comprehensive eczema management', urgency: 'recommended' },
        { name: 'Allergist', reason: 'To identify and manage triggers', urgency: 'recommended' },
        { name: 'Primary Care Physician', reason: 'For ongoing care and prescriptions', urgency: 'optional' }
      ],
      'Psoriasis': [
        { name: 'Dermatologist', reason: 'For treatment planning and biologics', urgency: 'recommended' },
        { name: 'Rheumatologist', reason: 'If psoriatic arthritis is suspected', urgency: 'recommended' },
        { name: 'Primary Care Physician', reason: 'For overall health management', urgency: 'optional' }
      ],
      'Acne': [
        { name: 'Dermatologist', reason: 'For prescription treatments and scar management', urgency: 'recommended' },
        { name: 'Endocrinologist', reason: 'If hormonal acne is suspected', urgency: 'optional' }
      ],
      'Rosacea': [
        { name: 'Dermatologist', reason: 'For prescription treatments and management', urgency: 'recommended' },
        { name: 'Ophthalmologist', reason: 'If eye symptoms are present', urgency: 'recommended' }
      ],
      'Urticaria (Hives)': [
        { name: 'Allergist', reason: 'To identify triggers and allergens', urgency: 'recommended' },
        { name: 'Dermatologist', reason: 'For chronic urticaria management', urgency: 'recommended' },
        { name: 'Immunologist', reason: 'For autoimmune-related urticaria', urgency: 'optional' }
      ],
      'Contact Dermatitis': [
        { name: 'Dermatologist', reason: 'For patch testing and treatment', urgency: 'recommended' },
        { name: 'Allergist', reason: 'For allergen identification', urgency: 'recommended' }
      ],
      'Seborrheic Dermatitis': [
        { name: 'Dermatologist', reason: 'For treatment and management', urgency: 'recommended' },
        { name: 'Primary Care Physician', reason: 'For initial treatment', urgency: 'optional' }
      ]
    };

    // Return specialists for the condition, or default general recommendations
    return specialistMap[predictedClass] || [
      { name: 'Dermatologist', reason: 'For professional evaluation and treatment', urgency: 'recommended' },
      { name: 'Primary Care Physician', reason: 'For initial assessment', urgency: 'optional' }
    ];
  };

  /**
   * Get plain English explanation of the skin condition
   * @param {string} predictedClass - The predicted skin condition
   * @returns {string} Simple, conversational explanation
   */
  const getPlainEnglishExplanation = (predictedClass: string) => {
    const explanations: { [key: string]: string } = {
      'Melanoma': "Think of melanoma as the most serious type of skin cancer. It starts in the cells that give your skin its color. The good news is that when caught early, it's very treatable. The key is to see a doctor right away. They'll examine it closely and may recommend removing it. Early detection is crucial, so don't delay getting this checked out by a specialist.",

      'Basal Cell Carcinoma': "This is the most common type of skin cancer, but also one of the least dangerous. It grows slowly and rarely spreads to other parts of your body. Think of it like a small bump or sore that doesn't heal properly. It usually appears on areas that get a lot of sun, like your face or neck. While it's not an emergency, you should have it removed to prevent it from growing larger.",

      'Squamous Cell Carcinoma': "This is another type of skin cancer that's more serious than basal cell but less aggressive than melanoma. It can look like a scaly patch, a firm bump, or a sore that keeps coming back. It grows faster than basal cell carcinoma and can spread if left untreated, so it's important to have it checked and removed by a doctor soon.",

      'Actinic Keratoses': "These are rough, scaly patches on your skin caused by too much sun exposure over the years. Think of them as 'pre-cancers' - they're not cancer yet, but they could develop into skin cancer if left untreated. The good news is they're very treatable with creams, freezing, or other simple procedures. Your doctor can easily remove them before they become a problem.",

      'Atopic Dermatitis (Eczema)': "Eczema is like your skin getting overly sensitive and irritated. It makes your skin dry, itchy, and sometimes red or inflamed. It's not contagious and it's not an infection - it's more like your skin's protective barrier isn't working as well as it should. Many people have it, especially those with allergies or asthma in their family. With the right moisturizers and sometimes prescription creams, you can keep it under control.",

      'Psoriasis': "Imagine your skin cells growing way too fast - that's psoriasis. Normally, skin cells take about a month to grow and shed, but with psoriasis, this happens in just a few days. The result is thick, scaly patches that can be itchy or uncomfortable. It's not contagious, and while there's no cure, there are many effective treatments that can clear it up and keep it under control.",

      'Acne': "Acne happens when the tiny openings in your skin (pores) get clogged with oil and dead skin cells. This creates an environment where bacteria can grow, leading to pimples, blackheads, or deeper bumps. It's incredibly common, especially during teenage years, but adults can get it too. The good news is there are many effective treatments, from over-the-counter creams to prescription medications.",

      'Rosacea': "Rosacea makes your face look flushed or red, almost like you're blushing all the time. It often affects the cheeks, nose, and forehead. Sometimes you'll see small visible blood vessels or bumps that look like acne. It's a chronic condition, meaning it comes and goes, but it's manageable with the right treatments and by avoiding triggers like spicy foods, alcohol, or extreme temperatures.",

      'Urticaria (Hives)': "Hives are raised, itchy bumps on your skin that can appear suddenly and disappear just as quickly. They're usually caused by an allergic reaction to something - maybe a food, medication, or even stress. Each hive might last a few hours, but new ones can keep appearing. Most cases clear up on their own, but if they last more than a few weeks or make it hard to breathe, see a doctor right away.",

      'Contact Dermatitis': "This is basically an allergic reaction on your skin. Your skin touched something it didn't like - maybe a certain soap, metal, plant, or chemical - and now it's red, itchy, and possibly swollen. The most important thing is figuring out what caused it so you can avoid it in the future. Once you stop touching the irritant, your skin should heal on its own, though creams can help speed up the process.",

      'Seborrheic Dermatitis': "This condition causes scaly, flaky patches on oily areas of your skin, especially your scalp (where it's known as dandruff), face, and chest. It's not caused by poor hygiene - it's actually related to a type of yeast that lives on everyone's skin but grows more in some people. Special shampoos and creams can control it, though it tends to come and go.",

      'Benign Keratosis': "These are harmless, non-cancerous growths that look like warts or rough patches. Think of them as just extra skin cells that have built up over time. They're very common as we age and are sometimes called 'age spots' or 'wisdom spots.' They don't need treatment unless they bother you or you want them removed for cosmetic reasons.",

      'Dermatofibroma': "This is a small, harmless bump in your skin that's usually firm and brownish in color. It often appears on your legs and might be the result of a minor injury like a bug bite, though no one knows for sure. They're completely benign (not cancerous) and don't need treatment, though a doctor can remove them if they bother you.",

      'Nevus (Mole)': "A nevus is just the medical term for a common mole. Most people have between 10 and 40 moles on their body. They're clusters of pigmented cells and are usually harmless. The important thing is to watch for changes - if a mole starts changing in size, shape, or color, or starts bleeding or itching, have it checked by a doctor to make sure it's not turning into melanoma.",

      'Vascular Lesion': "This is a red or purple spot caused by blood vessels that have grown abnormally or have become damaged. Think of it like a small cluster of tiny blood vessels near the surface of your skin. Common types include cherry angiomas (small red dots) or spider veins. Most are harmless and don't require treatment unless you want them removed for appearance."
    };

    return explanations[predictedClass] ||
      `This is a skin condition that your doctor can explain in detail. The AI has identified it as ${predictedClass}, which is a type of skin concern that may need professional evaluation. Your healthcare provider will be able to explain what this means for you specifically, discuss treatment options, and answer any questions you have. Remember, this is just a screening tool - always consult with a medical professional for a proper diagnosis.`;
  };

  /**
   * Speak the plain English explanation using text-to-speech
   */
  const handleSpeakExplanation = async (text: string) => {
    if (isSpeaking) {
      // Stop speaking
      Speech.stop();
      setIsSpeaking(false);
    } else {
      // Start speaking
      setIsSpeaking(true);
      Speech.speak(text, {
        language: 'en-US',
        pitch: 1.0,
        rate: 0.9, // Slightly slower for clarity
        onDone: () => setIsSpeaking(false),
        onStopped: () => setIsSpeaking(false),
        onError: () => {
          setIsSpeaking(false);
          Alert.alert('Error', 'Unable to play audio. Please check your device settings.');
        }
      });
    }
  };

  /**
   * Handle specialist card click - search for nearby doctors
   */
  const handleSpecialistClick = async (specialist) => {
    try {
      setSelectedSpecialist(specialist);
      setShowDoctorSearchModal(true);
      setIsLoadingDoctors(true);
      setNearbyDoctors([]);

      // Get user's current location
      const location = await DoctorSearchService.getCurrentLocation();
      setUserLocation(location);

      // Search for nearby doctors
      const doctors = await DoctorSearchService.searchNearbyDoctors(
        specialist.name,
        location.latitude,
        location.longitude
      );

      setNearbyDoctors(doctors);
    } catch (error) {
      console.error('Error searching for doctors:', error);

      if (error.message.includes('permission')) {
        Alert.alert(
          'Location Permission Required',
          'Please enable location services to find nearby doctors. You can enable this in your device settings.',
          [{ text: 'OK' }]
        );
      } else {
        Alert.alert(
          'Error',
          'Unable to search for nearby doctors. Please try again later.',
          [{ text: 'OK' }]
        );
      }
    } finally {
      setIsLoadingDoctors(false);
    }
  };

  /**
   * Call a doctor's phone number
   */
  const handleCallDoctor = (phoneNumber) => {
    if (phoneNumber) {
      DoctorSearchService.callPhone(phoneNumber);
    }
  };

  /**
   * Open doctor's location in maps
   */
  const handleOpenInMaps = (doctor) => {
    if (doctor.location) {
      DoctorSearchService.openInMaps(doctor.location.lat, doctor.location.lng, doctor.name);
    } else if (doctor.address) {
      Linking.openURL(`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(doctor.address)}`);
    }
  };

  const handleMeasurementsComplete = (newMeasurements) => {
    setMeasurements(newMeasurements);
    setShowMeasurementTool(false);

    if (newMeasurements.length > 0) {
      Alert.alert(
        'Measurements Saved',
        `${newMeasurements.length} measurement(s) have been saved to this analysis.`,
        [{ text: 'OK' }]
      );
    }
  };

  const handleSymptomChange = (symptoms: any) => {
    setSymptomData(symptoms);
  };

  const handleSubmitSymptoms = async () => {
    try {
      console.log('=== SUBMITTING SYMPTOMS ===');
      console.log('Symptom data:', JSON.stringify(symptomData, null, 2));

      if (!symptomData) {
        Alert.alert('Error', 'Please provide symptom information.');
        return;
      }

      // Check if user actually entered any meaningful data (not just default false values)
      const hasAnyData = (symptomData.symptom_duration && symptomData.symptom_duration.trim()) ||
                         (symptomData.symptom_changes && symptomData.symptom_changes.trim()) ||
                         symptomData.symptom_itching === true ||  // Must be explicitly true
                         symptomData.symptom_pain === true ||     // Must be explicitly true
                         symptomData.symptom_bleeding === true || // Must be explicitly true
                         (symptomData.symptom_notes && symptomData.symptom_notes.trim());

      if (!hasAnyData) {
        Alert.alert(
          'No Symptoms Entered',
          'Please enter at least one symptom before saving:\n\n• Enter duration (e.g., "2 weeks")\n• Describe changes\n• Toggle itching, pain, or bleeding ON\n• Add notes',
          [{ text: 'OK' }]
        );
        return;
      }

      const token = await AnalysisHistoryService.getAuthToken();
      if (!token) {
        Alert.alert('Error', 'Authentication required. Please log in again.');
        return;
      }

      // Create form data
      const formData = new FormData();
      if (symptomData.symptom_duration) {
        console.log('Adding duration:', symptomData.symptom_duration);
        formData.append('symptom_duration', symptomData.symptom_duration);
      }
      if (symptomData.symptom_duration_value) formData.append('symptom_duration_value', symptomData.symptom_duration_value.toString());
      if (symptomData.symptom_duration_unit) formData.append('symptom_duration_unit', symptomData.symptom_duration_unit);
      if (symptomData.symptom_changes) {
        console.log('Adding changes:', symptomData.symptom_changes);
        formData.append('symptom_changes', symptomData.symptom_changes);
      }
      formData.append('symptom_itching', symptomData.symptom_itching ? 'true' : 'false');
      if (symptomData.symptom_itching_severity) formData.append('symptom_itching_severity', symptomData.symptom_itching_severity.toString());
      formData.append('symptom_pain', symptomData.symptom_pain ? 'true' : 'false');
      if (symptomData.symptom_pain_severity) formData.append('symptom_pain_severity', symptomData.symptom_pain_severity.toString());
      formData.append('symptom_bleeding', symptomData.symptom_bleeding ? 'true' : 'false');
      if (symptomData.symptom_bleeding_frequency) formData.append('symptom_bleeding_frequency', symptomData.symptom_bleeding_frequency);
      if (symptomData.symptom_notes) {
        console.log('Adding notes:', symptomData.symptom_notes);
        formData.append('symptom_notes', symptomData.symptom_notes);
      }

      const response = await fetch(`${API_BASE_URL}/analysis/symptoms/${id}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Failed to submit symptoms: ${response.status}`);
      }

      const result = await response.json();
      console.log('Backend response:', result);

      // Close form and reload analysis
      setShowSymptomForm(false);
      await loadAnalysisDetail();

      Alert.alert(
        'Symptoms Recorded',
        `Your symptom information has been saved successfully.\n\nDuration: ${symptomData.symptom_duration || 'Not specified'}\nItching: ${symptomData.symptom_itching ? 'Yes' : 'No'}\nPain: ${symptomData.symptom_pain ? 'Yes' : 'No'}\nBleeding: ${symptomData.symptom_bleeding ? 'Yes' : 'No'}`,
        [{ text: 'OK' }]
      );

    } catch (error) {
      console.error('Symptom submission error:', error);
      Alert.alert('Error', error.message || 'Failed to submit symptoms. Please try again.');
    }
  };

  const handleMedicationChange = (medications: Medication[]) => {
    setMedicationData(medications);
  };

  const handleSubmitMedications = async () => {
    try {
      // Filter out empty medications (where name is empty)
      const validMedications = medicationData.filter(med => med.name && med.name.trim());

      if (validMedications.length === 0) {
        Alert.alert(
          'No Medications Entered',
          'Please add at least one medication with a name.',
          [{ text: 'OK' }]
        );
        return;
      }

      const token = await AnalysisHistoryService.getAuthToken();
      if (!token) {
        Alert.alert('Error', 'Authentication required. Please log in again.');
        return;
      }

      // Create form data with JSON string
      const formData = new FormData();
      formData.append('medications', JSON.stringify(validMedications));

      const response = await fetch(`${API_BASE_URL}/analysis/medications/${id}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Failed to submit medications: ${response.status}`);
      }

      const result = await response.json();

      // Close form and reload analysis
      setShowMedicationForm(false);
      await loadAnalysisDetail();

      Alert.alert(
        'Medications Recorded',
        `${validMedications.length} medication(s) have been saved successfully.`,
        [{ text: 'OK' }]
      );

    } catch (error) {
      console.error('Medication submission error:', error);
      Alert.alert('Error', error.message || 'Failed to submit medications. Please try again.');
    }
  };

  const handleMedicalHistoryChange = (history: MedicalHistoryData) => {
    setMedicalHistoryData(history);
  };

  const handleSubmitMedicalHistory = async () => {
    try {
      if (!medicalHistoryData) {
        Alert.alert('Error', 'Please provide medical history information.');
        return;
      }

      // Check if any data was entered
      const hasAnyData = medicalHistoryData.family_history_skin_cancer === true ||
                         medicalHistoryData.previous_skin_cancers === true ||
                         medicalHistoryData.immunosuppression === true ||
                         medicalHistoryData.sun_exposure_level ||
                         medicalHistoryData.history_of_sunburns === true ||
                         medicalHistoryData.tanning_bed_use === true ||
                         (medicalHistoryData.other_risk_factors && medicalHistoryData.other_risk_factors.trim());

      if (!hasAnyData) {
        Alert.alert(
          'No Data Entered',
          'Please provide at least one risk factor or medical history detail.',
          [{ text: 'OK' }]
        );
        return;
      }

      const token = await AnalysisHistoryService.getAuthToken();
      if (!token) {
        Alert.alert('Error', 'Authentication required. Please log in again.');
        return;
      }

      // Create form data
      const formData = new FormData();
      formData.append('family_history_skin_cancer', medicalHistoryData.family_history_skin_cancer ? 'true' : 'false');
      if (medicalHistoryData.family_history_details) formData.append('family_history_details', medicalHistoryData.family_history_details);
      formData.append('previous_skin_cancers', medicalHistoryData.previous_skin_cancers ? 'true' : 'false');
      if (medicalHistoryData.previous_skin_cancers_details) formData.append('previous_skin_cancers_details', medicalHistoryData.previous_skin_cancers_details);
      formData.append('immunosuppression', medicalHistoryData.immunosuppression ? 'true' : 'false');
      if (medicalHistoryData.immunosuppression_details) formData.append('immunosuppression_details', medicalHistoryData.immunosuppression_details);
      if (medicalHistoryData.sun_exposure_level) formData.append('sun_exposure_level', medicalHistoryData.sun_exposure_level);
      if (medicalHistoryData.sun_exposure_details) formData.append('sun_exposure_details', medicalHistoryData.sun_exposure_details);
      formData.append('history_of_sunburns', medicalHistoryData.history_of_sunburns ? 'true' : 'false');
      if (medicalHistoryData.sunburn_details) formData.append('sunburn_details', medicalHistoryData.sunburn_details);
      formData.append('tanning_bed_use', medicalHistoryData.tanning_bed_use ? 'true' : 'false');
      if (medicalHistoryData.tanning_bed_frequency) formData.append('tanning_bed_frequency', medicalHistoryData.tanning_bed_frequency);
      if (medicalHistoryData.other_risk_factors) formData.append('other_risk_factors', medicalHistoryData.other_risk_factors);

      const response = await fetch(`${API_BASE_URL}/analysis/medical-history/${id}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Failed to submit medical history: ${response.status}`);
      }

      const result = await response.json();

      // Close form and reload analysis
      setShowMedicalHistoryForm(false);
      await loadAnalysisDetail();

      Alert.alert(
        'Medical History Recorded',
        'Your risk factors and medical history have been saved successfully.',
        [{ text: 'OK' }]
      );

    } catch (error) {
      console.error('Medical history submission error:', error);
      Alert.alert('Error', error.message || 'Failed to submit medical history. Please try again.');
    }
  };

  const handleTeledermatologyShare = async (shareData: any) => {
    try {
      const token = await AnalysisHistoryService.getAuthToken();
      if (!token) {
        Alert.alert('Error', 'Authentication required. Please log in again.');
        return;
      }

      // Create form data
      const formData = new FormData();
      formData.append('dermatologist_name', shareData.dermatologist_name);
      formData.append('dermatologist_email', shareData.dermatologist_email);
      if (shareData.share_message) {
        formData.append('share_message', shareData.share_message);
      }

      const response = await fetch(`${API_BASE_URL}/analysis/share-with-dermatologist/${id}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Failed to share with dermatologist: ${response.status}`);
      }

      const result = await response.json();

      // Close form and reload analysis
      setShowTeledermatologyForm(false);
      await loadAnalysisDetail();

      // Copy URL to clipboard
      const shareUrl = `${API_BASE_URL}${result.share_url}`;
      Clipboard.setString(shareUrl);

      // Show success with options to share
      Alert.alert(
        'Shared Successfully! 🎉',
        `Analysis shared with ${result.dermatologist_name}.\n\nThe share link has been copied to your clipboard. You can now paste and send it via email, text, or any messaging app.`,
        [
          {
            text: 'Send via Email',
            onPress: async () => {
              const subject = 'Skin Analysis for Review';
              const pdfUrl = `${shareUrl}/pdf`;
              const body = `Dear ${result.dermatologist_name},\n\nI would like to share my skin analysis with you for professional review.\n\nPlease click the link below to view the analysis:\n${shareUrl}\n\nYou can also download a PDF report here:\n${pdfUrl}\n\nThank you for your time.\n\nBest regards`;
              const emailUrl = `mailto:${shareData.dermatologist_email}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

              const canOpen = await Linking.canOpenURL(emailUrl);
              if (canOpen) {
                Linking.openURL(emailUrl);
              } else {
                Alert.alert('Cannot open email', 'Please copy the link from your clipboard and send it manually.');
              }
            }
          },
          {
            text: 'Copy Link Again',
            onPress: () => {
              Clipboard.setString(shareUrl);
              Alert.alert('Copied!', 'Share link copied to clipboard.');
            }
          },
          { text: 'Done' }
        ]
      );

    } catch (error) {
      console.error('Teledermatology sharing error:', error);
      Alert.alert('Error', error.message || 'Failed to share with dermatologist. Please try again.');
    }
  };

  const handleSubmitBiopsy = async () => {
    try {
      if (!biopsyData.biopsy_result.trim()) {
        Alert.alert('Error', 'Please enter the biopsy result.');
        return;
      }

      const token = await AnalysisHistoryService.getAuthToken();
      if (!token) {
        Alert.alert('Error', 'Authentication required. Please log in again.');
        return;
      }

      // Create form data
      const formData = new FormData();
      formData.append('biopsy_result', biopsyData.biopsy_result);
      if (biopsyData.biopsy_date) formData.append('biopsy_date', biopsyData.biopsy_date);
      if (biopsyData.biopsy_notes) formData.append('biopsy_notes', biopsyData.biopsy_notes);
      if (biopsyData.biopsy_facility) formData.append('biopsy_facility', biopsyData.biopsy_facility);
      if (biopsyData.pathologist_name) formData.append('pathologist_name', biopsyData.pathologist_name);

      const response = await fetch(`${API_BASE_URL}/analysis/biopsy/${id}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Failed to submit biopsy result: ${response.status}`);
      }

      const result = await response.json();

      // Close form and reload analysis
      setShowBiopsyForm(false);
      await loadAnalysisDetail();

      // Show success message with accuracy info
      const accuracyMessage = result.prediction_correct
        ? `✅ AI prediction was CORRECT!\n\nAI: ${result.ai_prediction}\nBiopsy: ${result.biopsy_result}`
        : `❌ AI prediction differed from biopsy.\n\nAI: ${result.ai_prediction}\nBiopsy: ${result.biopsy_result}\n\nCategory: ${result.accuracy_category}`;

      Alert.alert('Biopsy Result Added', accuracyMessage, [{ text: 'OK' }]);

    } catch (error) {
      console.error('Biopsy submission error:', error);
      Alert.alert('Error', error.message || 'Failed to submit biopsy result. Please try again.');
    }
  };

  const handleExportFHIR = async () => {
    try {
      console.log('Starting FHIR export...');
      const token = await AnalysisHistoryService.getAuthToken();
      if (!token) {
        Alert.alert('Error', 'Authentication required. Please log in again.');
        return;
      }

      console.log('Fetching FHIR report from API...');
      // Fetch FHIR report from API
      const response = await fetch(`${API_BASE_URL}/analysis/export/fhir/${id}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept': 'application/fhir+json'
        }
      });

      console.log('Response status:', response.status);
      if (!response.ok) {
        const errorText = await response.text();
        console.error('API error response:', errorText);
        throw new Error(`Failed to export FHIR report (${response.status}): ${errorText}`);
      }

      const fhirData = await response.json();
      console.log('FHIR data received:', fhirData);

      const filename = `fhir-report-${id}-${Date.now()}.json`;
      const fhirJsonString = JSON.stringify(fhirData, null, 2);

      console.log('Platform:', Platform.OS);

      // Check if we're in a browser environment (web or React Native Web)
      const isWeb = typeof document !== 'undefined' && typeof window !== 'undefined';
      console.log('Is web environment:', isWeb);

      if (isWeb) {
        // Web/Browser platform - use DOM APIs
        console.log('Using web download...');
        const blob = new Blob([fhirJsonString], { type: 'application/fhir+json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        Alert.alert(
          'Export Successful',
          `FHIR report "${filename}" has been downloaded.`,
          [{ text: 'OK' }]
        );
      } else {
        // Mobile platform (iOS/Android) - use legacy FileSystem API and share
        console.log('Using mobile file sharing...');

        const fileUri = FileSystemLegacy.cacheDirectory + filename;

        // Write to cache directory using legacy API
        await FileSystemLegacy.writeAsStringAsync(fileUri, fhirJsonString);

        console.log('File written to:', fileUri);

        // Share the file
        const canShare = await Sharing.isAvailableAsync();
        if (canShare) {
          console.log('Sharing file...');
          await Sharing.shareAsync(fileUri, {
            mimeType: 'application/json',
            dialogTitle: 'Export FHIR Report',
            UTI: 'public.json'
          });
        } else {
          Alert.alert(
            'Export Successful',
            `FHIR report saved to:\n${fileUri}\n\nYou can access it through your file manager.`,
            [{ text: 'OK' }]
          );
        }
      }

    } catch (error) {
      console.error('FHIR export error details:', error);
      console.error('Error stack:', error.stack);
      Alert.alert(
        'Export Failed',
        `${error.message || 'Failed to export FHIR report. Please try again.'}\n\nCheck console for details.`,
        [{ text: 'OK' }]
      );
    }
  };

  const getRiskBadgeStyle = (riskLevel) => {
    const baseStyle = { ...styles.riskBadge };
    switch (riskLevel?.toLowerCase()) {
      case 'high':
        return { ...baseStyle, backgroundColor: '#dc3545' };
      case 'medium':
        return { ...baseStyle, backgroundColor: '#ffc107', color: '#000' };
      case 'low':
        return { ...baseStyle, backgroundColor: '#28a745' };
      default:
        return { ...baseStyle, backgroundColor: '#6c757d' };
    }
  };

  if (isLoading) {
    return (
      <View style={styles.container}>
        <LinearGradient
          colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.backgroundContainer}
        />
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4299e1" />
          <Text style={styles.loadingText}>Loading analysis details...</Text>
        </View>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.container}>
        <LinearGradient
          colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.backgroundContainer}
        />
        <View style={styles.errorContainer}>
          <Text style={styles.errorTitle}>⚠️ Unable to Load Analysis</Text>
          <Text style={styles.errorText}>{error}</Text>
          <View style={styles.errorButtons}>
            <Pressable style={styles.retryButton} onPress={loadAnalysisDetail}>
              <Text style={styles.retryButtonText}>Try Again</Text>
            </Pressable>
            <Pressable style={styles.backButton} onPress={() => router.back()}>
              <Text style={styles.backButtonText}>{t('common.goBack')}</Text>
            </Pressable>
          </View>
        </View>
      </View>
    );
  }

  if (!analysis) {
    return null;
  }

  // Format analysis for display - uses confidence from backend API
  const formatted = formatAnalysisForDisplay(analysis);

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.backgroundContainer}
      />

      {/* Header */}
      <View style={styles.header}>
        <Pressable style={styles.backHeaderButton} onPress={() => router.back()}>
          <Text style={styles.backHeaderButtonText}>← {t('common.back')}</Text>
        </Pressable>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>📋 Analysis Report</Text>
          <Text style={styles.headerSubtitle}>Detailed Results</Text>
        </View>
      </View>

      <ScrollView style={styles.scrollContainer} contentContainerStyle={styles.scrollContent}>
        {/* Main Analysis Card */}
        <View style={styles.analysisCard}>
          {/* Header Section */}
          <View style={styles.cardHeader}>
            <View style={styles.analysisInfo}>
              <Text style={styles.analysisDate}>{formatted.createdAt.date}</Text>
              <Text style={styles.analysisTime}>{formatted.createdAt.time}</Text>
              <Text style={styles.analysisRelative}>{formatted.createdAt.relative}</Text>
            </View>
            <View style={getRiskBadgeStyle(formatted.riskLevel)}>
              <Text style={styles.riskBadgeText}>{formatted.riskLevel?.toUpperCase() || 'UNKNOWN'}</Text>
            </View>
          </View>

          {/* Image Display */}
          {analysis.image_url && (
            <View style={styles.imageSection}>
              <Text style={styles.sectionTitle}>📷 Analyzed Image</Text>
              <View style={styles.imageContainer}>
                <Image
                  source={{ uri: `${API_BASE_URL}${analysis.image_url}` }}
                  style={styles.analysisImage}
                  resizeMode="cover"
                />
              </View>
            </View>
          )}

          {/* Explainability Heatmap */}
          {analysis.explainability_heatmap && (
            <View style={styles.explainabilitySection}>
              <Text style={styles.sectionTitle}>🔥 Explainability Heatmap</Text>
              <Text style={styles.explainabilityDescription}>
                Highlighted regions show which areas of the image influenced the AI's diagnosis.
                Red/warm colors indicate high importance, blue/cool colors indicate low importance.
              </Text>
              <View style={styles.heatmapContainer}>
                <Image
                  source={{ uri: analysis.explainability_heatmap }}
                  style={styles.heatmapImage}
                  resizeMode="contain"
                />
              </View>
              <Text style={styles.explainabilityNote}>
                ℹ️ This visualization helps understand the AI's decision-making process
              </Text>
            </View>
          )}

          {/* Dermoscopy Analysis Section */}
          {analysis.dermoscopy && analysis.dermoscopy.is_dermoscopy_image && (
            <View style={styles.dermoscopySection}>
              <Text style={styles.sectionTitle}>🔬 Dermoscopy Analysis</Text>
              <Text style={styles.dermoscopyBadge}>
                {analysis.dermoscopy.image_type === 'dermoscopy' ? '✓ Dermoscopy Image Detected' : 'Clinical Photo'}
              </Text>
              <Text style={styles.dermoscopyConfidence}>
                Detection Confidence: {Math.round(analysis.dermoscopy.detection_confidence * 100)}%
              </Text>

              {analysis.dermoscopy.detection_features && analysis.dermoscopy.detection_features.length > 0 && (
                <View style={styles.dermoscopyFeatures}>
                  <Text style={styles.dermoscopyFeaturesTitle}>Image Characteristics:</Text>
                  {analysis.dermoscopy.detection_features.map((feature, index) => (
                    <Text key={index} style={styles.dermoscopyFeature}>• {feature}</Text>
                  ))}
                </View>
              )}

              {analysis.dermoscopy.dermoscopic_structures && Object.keys(analysis.dermoscopy.dermoscopic_structures).length > 0 && (
                <View style={styles.dermoscopyStructures}>
                  <Text style={styles.dermoscopyStructuresTitle}>Dermoscopic Structures Detected:</Text>

                  {analysis.dermoscopy.dermoscopic_structures.pigment_network && (
                    <View style={styles.structureItem}>
                      <Text style={styles.structureName}>🕸️ Pigment Network:</Text>
                      <Text style={styles.structureDetail}>
                        {analysis.dermoscopy.dermoscopic_structures.pigment_network.density} pattern detected
                      </Text>
                    </View>
                  )}

                  {analysis.dermoscopy.dermoscopic_structures.dots_globules && (
                    <View style={styles.structureItem}>
                      <Text style={styles.structureName}>⚫ Dots & Globules:</Text>
                      <Text style={styles.structureDetail}>
                        {analysis.dermoscopy.dermoscopic_structures.dots_globules.count} detected -
                        {analysis.dermoscopy.dermoscopic_structures.dots_globules.pattern} pattern
                      </Text>
                    </View>
                  )}

                  {analysis.dermoscopy.dermoscopic_structures.blue_white_veil && (
                    <View style={styles.structureItem}>
                      <Text style={[styles.structureName, styles.riskIndicator]}>⚠️ Blue-White Veil:</Text>
                      <Text style={styles.structureDetail}>
                        Present ({Math.round(analysis.dermoscopy.dermoscopic_structures.blue_white_veil.coverage * 100)}% coverage)
                      </Text>
                      <Text style={styles.riskNote}>Note: May indicate melanoma risk</Text>
                    </View>
                  )}

                  {analysis.dermoscopy.dermoscopic_structures.vascular_structures && (
                    <View style={styles.structureItem}>
                      <Text style={styles.structureName}>🩸 Vascular Structures:</Text>
                      <Text style={styles.structureDetail}>
                        {analysis.dermoscopy.dermoscopic_structures.vascular_structures.type} pattern
                      </Text>
                    </View>
                  )}

                  {analysis.dermoscopy.dermoscopic_structures.asymmetry && (
                    <View style={styles.structureItem}>
                      <Text style={styles.structureName}>⚖️ Asymmetry:</Text>
                      <Text style={styles.structureDetail}>
                        {analysis.dermoscopy.dermoscopic_structures.asymmetry.interpretation} asymmetry
                        (score: {analysis.dermoscopy.dermoscopic_structures.asymmetry.score.toFixed(2)})
                      </Text>
                    </View>
                  )}
                </View>
              )}

              <Text style={styles.dermoscopyNote}>
                ℹ️ Dermoscopy provides enhanced visualization of subsurface skin structures
              </Text>
            </View>
          )}

          {/* Measurement & Calibration Section */}
          <View style={styles.calibrationSection}>
            <Text style={styles.sectionTitle}>📏 Measurement & Calibration</Text>

            {analysis.calibration && (
              <>

              {analysis.calibration.calibration_found ? (
                <>
                  <View style={styles.calibrationInfo}>
                    <Text style={styles.calibrationFound}>✓ Calibration Detected</Text>
                    <View style={styles.calibrationDetails}>
                      <View style={styles.calibrationItem}>
                        <Text style={styles.calibrationLabel}>Type:</Text>
                        <Text style={styles.calibrationValue}>
                          {analysis.calibration.calibration_type === 'coin' ? '🪙 Coin' : '📏 Ruler'}
                        </Text>
                      </View>
                      <View style={styles.calibrationItem}>
                        <Text style={styles.calibrationLabel}>Resolution:</Text>
                        <Text style={styles.calibrationValue}>
                          {analysis.calibration.pixels_per_mm?.toFixed(2)} pixels/mm
                        </Text>
                      </View>
                      <View style={styles.calibrationItem}>
                        <Text style={styles.calibrationLabel}>Confidence:</Text>
                        <Text style={styles.calibrationValue}>
                          {Math.round((analysis.calibration.confidence || 0) * 100)}%
                        </Text>
                      </View>
                    </View>
                  </View>

                  <View style={styles.measurementCapability}>
                    <Text style={styles.measurementCapabilityText}>
                      ✨ This image is calibrated for accurate measurements
                    </Text>
                    <Text style={styles.measurementInstructions}>
                      You can use the measurement tool to draw lines and get real-world dimensions in millimeters.
                    </Text>
                    <Pressable
                      style={styles.measurementButton}
                      onPress={() => setShowMeasurementTool(true)}
                    >
                      <Text style={styles.measurementButtonText}>📏 Open Measurement Tool</Text>
                    </Pressable>
                  </View>

                  {analysis.calibration.detected_objects && analysis.calibration.detected_objects.length > 0 && (
                    <View style={styles.detectedObjects}>
                      <Text style={styles.detectedObjectsTitle}>Detected Reference Objects:</Text>
                      {analysis.calibration.detected_objects.map((obj, index) => (
                        <View key={index} style={styles.detectedObject}>
                          <Text style={styles.detectedObjectType}>
                            {obj.type === 'coin_circular' ? '🪙 Circular object (assumed: US Quarter)' : '📏 ' + obj.type}
                          </Text>
                          {obj.assumed_size_mm && (
                            <Text style={styles.detectedObjectSize}>
                              Reference size: {obj.assumed_size_mm} mm
                            </Text>
                          )}
                        </View>
                      ))}
                    </View>
                  )}
                </>
              ) : (
                <View style={styles.noCalibration}>
                  <Text style={styles.noCalibrationText}>⚠ No automatic calibration detected</Text>
                  <Text style={styles.noCalibrationHint}>
                    For accurate measurements, include a reference object (coin or ruler) in your photos.
                  </Text>
                  <Pressable
                    style={styles.measurementButtonSecondary}
                    onPress={() => setShowMeasurementTool(true)}
                  >
                    <Text style={styles.measurementButtonText}>📏 Use Manual Calibration</Text>
                  </Pressable>
                </View>
              )}
              </>
            )}

            {/* Always show measurement tool button */}
            {!analysis.calibration && (
              <View style={styles.measurementToolSection}>
                <Text style={styles.measurementToolText}>
                  📏 Draw and measure lesion dimensions
                </Text>
                <Pressable
                  style={styles.measurementToolButton}
                  onPress={() => setShowMeasurementTool(true)}
                >
                  <Text style={styles.measurementToolButtonText}>📏 Open Measurement Tool</Text>
                </Pressable>
              </View>
            )}
          </View>

          {/* Results Section */}
          <View style={styles.resultsSection}>
            <Text style={styles.sectionTitle}>🔍 Analysis Results</Text>

            {/* Model Disagreement Warning */}
            {(() => {
              const malignantTypes = ['Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Actinic Keratosis'];
              const binaryLesionProb = analysis.binary_probabilities?.lesion || 0;
              const predictedClass = analysis.predicted_class || '';
              const isMalignantPrediction = malignantTypes.some(type =>
                predictedClass.toLowerCase().includes(type.toLowerCase())
              );
              const modelsDisagree = binaryLesionProb < 0.5 && isMalignantPrediction && analysis.lesion_confidence > 0.3;

              if (modelsDisagree) {
                return (
                  <View style={styles.modelDisagreementBanner}>
                    <Text style={styles.modelDisagreementIcon}>⚠️</Text>
                    <View style={styles.modelDisagreementContent}>
                      <Text style={styles.modelDisagreementTitle}>Requires Professional Review</Text>
                      <Text style={styles.modelDisagreementText}>
                        Initial screening was uncertain, but pattern analysis detected features consistent with {predictedClass}.
                        Dermatologist consultation is recommended for definitive diagnosis.
                      </Text>
                    </View>
                  </View>
                );
              }
              return null;
            })()}

            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Classification:</Text>
              <Text style={styles.resultValue}>
                {formatted.isLesion ? '🔬 Lesion Detected' : '✅ No Lesion Detected'}
              </Text>
            </View>

            {formatted.predictedClass && (
              <View style={styles.resultItem}>
                <Text style={styles.resultLabel}>Predicted Class:</Text>
                <Text style={styles.diagnosisValue}>{formatted.predictedClass}</Text>
              </View>
            )}

            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Classification Confidence:</Text>
              {(() => {
                // Use lesion_confidence (cancer type probability) - NOT binary_confidence (lesion detection)
                const conf = analysis.lesion_confidence !== null && analysis.lesion_confidence !== undefined
                  ? analysis.lesion_confidence
                  : (analysis.binary_confidence || 0);
                const pct = Math.round(conf * 100);
                const level = pct >= 80 ? 'High' : pct >= 60 ? 'Medium' : 'Low';
                const clr = pct >= 70 ? '#28a745' : pct >= 50 ? '#ffc107' : '#dc3545';
                return (
                  <Text style={[styles.confidenceValue, { color: clr }]}>
                    {pct}% ({level})
                  </Text>
                );
              })()}
            </View>

            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Analysis Type:</Text>
              <Text style={styles.resultValue}>{formatted.type}</Text>
            </View>

            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Inflammatory Condition:</Text>
              <Text style={styles.diagnosisValue}>
                {analysis.inflammatory_condition || 'None detected'}
              </Text>
            </View>

            {analysis.inflammatory_confidence && (
              <View style={styles.resultItem}>
                <Text style={styles.resultLabel}>Inflammatory Confidence:</Text>
                <Text style={[styles.confidenceValue, { color: analysis.inflammatory_confidence > 0.7 ? '#28a745' : analysis.inflammatory_confidence > 0.5 ? '#ffc107' : '#dc3545' }]}>
                  {Math.round((analysis.inflammatory_confidence || 0) * 100)}%
                </Text>
              </View>
            )}

            {/* AI Explanation Section */}
            {(formatted.predictedClass || analysis.inflammatory_condition) && (
              <View style={styles.aiExplanationContainer}>
                <Pressable
                  style={[styles.learnMoreButton, showAiExplanation && styles.learnMoreButtonActive]}
                  onPress={() => {
                    if (showAiExplanation) {
                      setShowAiExplanation(false);
                    } else {
                      const condition = formatted.predictedClass || analysis.inflammatory_condition;
                      const severity = analysis.risk_level;
                      fetchAIExplanation(condition, severity);
                    }
                  }}
                >
                  <Ionicons
                    name={showAiExplanation ? "chevron-up-circle" : "information-circle"}
                    size={20}
                    color="#667eea"
                    style={{ marginRight: 8 }}
                  />
                  <Text style={styles.learnMoreButtonText}>
                    {showAiExplanation ? 'Hide Explanation' : 'Learn More About This Condition'}
                  </Text>
                  {isLoadingAiExplanation && (
                    <ActivityIndicator size="small" color="#667eea" style={{ marginLeft: 8 }} />
                  )}
                </Pressable>

                {showAiExplanation && (
                  <View style={styles.aiExplanationContent}>
                    {isLoadingAiExplanation ? (
                      <View style={styles.aiExplanationLoading}>
                        <ActivityIndicator size="large" color="#667eea" />
                        <Text style={styles.aiExplanationLoadingText}>
                          Getting AI explanation...
                        </Text>
                      </View>
                    ) : aiExplanationError ? (
                      <View style={styles.aiExplanationError}>
                        <Ionicons name="alert-circle" size={24} color="#e53e3e" />
                        <Text style={styles.aiExplanationErrorText}>{aiExplanationError}</Text>
                        <Pressable
                          style={styles.retryButton}
                          onPress={() => {
                            const condition = formatted.predictedClass || analysis.inflammatory_condition;
                            const severity = analysis.risk_level;
                            fetchAIExplanation(condition, severity);
                          }}
                        >
                          <Text style={styles.retryButtonText}>Try Again</Text>
                        </Pressable>
                      </View>
                    ) : aiExplanation ? (
                      <View style={styles.aiExplanationText}>
                        <View style={styles.aiExplanationHeader}>
                          <Ionicons name="sparkles" size={18} color="#667eea" />
                          <Text style={styles.aiExplanationTitle}>
                            About {formatted.predictedClass || analysis.inflammatory_condition}
                          </Text>
                        </View>
                        <Text style={styles.aiExplanationBody}>{aiExplanation}</Text>
                        <View style={styles.aiExplanationDisclaimer}>
                          <Ionicons name="information-circle-outline" size={14} color="#718096" />
                          <Text style={styles.aiExplanationDisclaimerText}>
                            This AI explanation is for educational purposes only. Please consult a healthcare provider for medical advice.
                          </Text>
                        </View>
                      </View>
                    ) : null}
                  </View>
                )}
              </View>
            )}

            {/* Differential Reasoning Section */}
            {(formatted.predictedClass || analysis.inflammatory_condition) && (
              <View style={styles.reasoningContainer}>
                <Pressable
                  style={[styles.reasoningButton, showReasoning && styles.reasoningButtonActive]}
                  onPress={() => {
                    if (showReasoning) {
                      setShowReasoning(false);
                    } else {
                      fetchDifferentialReasoning();
                    }
                  }}
                >
                  <Ionicons
                    name={showReasoning ? "chevron-up-circle" : "git-branch"}
                    size={20}
                    color="#8b5cf6"
                    style={{ marginRight: 8 }}
                  />
                  <Text style={styles.reasoningButtonText}>
                    {showReasoning ? 'Hide Diagnostic Reasoning' : 'Show Diagnostic Reasoning'}
                  </Text>
                  {isLoadingReasoning && (
                    <ActivityIndicator size="small" color="#8b5cf6" style={{ marginLeft: 8 }} />
                  )}
                </Pressable>

                {showReasoning && (
                  <View style={styles.reasoningContent}>
                    {isLoadingReasoning ? (
                      <View style={styles.reasoningLoading}>
                        <ActivityIndicator size="large" color="#8b5cf6" />
                        <Text style={styles.reasoningLoadingText}>
                          Analyzing diagnostic reasoning...
                        </Text>
                      </View>
                    ) : reasoningError ? (
                      <View style={styles.reasoningError}>
                        <Ionicons name="alert-circle" size={24} color="#e53e3e" />
                        <Text style={styles.reasoningErrorText}>{reasoningError}</Text>
                        <Pressable
                          style={styles.reasoningRetryButton}
                          onPress={fetchDifferentialReasoning}
                        >
                          <Text style={styles.reasoningRetryButtonText}>Try Again</Text>
                        </Pressable>
                      </View>
                    ) : differentialReasoning ? (
                      <View style={styles.reasoningText}>
                        <View style={styles.reasoningHeader}>
                          <Ionicons name="analytics" size={18} color="#8b5cf6" />
                          <Text style={styles.reasoningTitle}>
                            Diagnostic Reasoning
                          </Text>
                        </View>
                        <Text style={styles.reasoningBody}>{differentialReasoning}</Text>
                        <View style={styles.reasoningDisclaimer}>
                          <Ionicons name="information-circle-outline" size={14} color="#718096" />
                          <Text style={styles.reasoningDisclaimerText}>
                            This AI reasoning is for educational purposes only. Please consult a healthcare provider for medical diagnosis.
                          </Text>
                        </View>
                      </View>
                    ) : null}
                  </View>
                )}
              </View>
            )}
          </View>

          {/* Infectious Disease Section */}
          {analysis.infectious_disease && (
            <View style={styles.infectiousSection}>
              <Text style={styles.sectionTitle}>🦠 Infectious Disease Analysis</Text>

              <View style={styles.resultItem}>
                <Text style={styles.resultLabel}>Detected Disease:</Text>
                <Text style={styles.diagnosisValue}>{analysis.infectious_disease.replace(/_/g, ' ')}</Text>
              </View>

              <View style={styles.resultItem}>
                <Text style={styles.resultLabel}>Confidence:</Text>
                <Text style={[styles.confidenceValue, { color: analysis.infectious_confidence > 0.7 ? '#28a745' : analysis.infectious_confidence > 0.5 ? '#ffc107' : '#dc3545' }]}>
                  {Math.round((analysis.infectious_confidence || 0) * 100)}%
                </Text>
              </View>

              {/* Infection Type Badge */}
              {analysis.infection_type && analysis.infection_type !== 'none' && (
                <View style={styles.resultItem}>
                  <Text style={styles.resultLabel}>Infection Type:</Text>
                  <View style={[
                    styles.infectionTypeBadge,
                    analysis.infection_type === 'bacterial' && styles.bacterialBadge,
                    analysis.infection_type === 'fungal' && styles.fungalBadge,
                    analysis.infection_type === 'viral' && styles.viralBadge,
                    analysis.infection_type === 'parasitic' && styles.parasiticBadge,
                  ]}>
                    <Text style={styles.infectionTypeBadgeText}>
                      {analysis.infection_type.toUpperCase()}
                    </Text>
                  </View>
                </View>
              )}

              {/* Severity */}
              {analysis.infectious_severity && (
                <View style={styles.resultItem}>
                  <Text style={styles.resultLabel}>Severity:</Text>
                  <View style={[
                    styles.severityBadge,
                    analysis.infectious_severity === 'severe' && styles.severeSeverity,
                    analysis.infectious_severity === 'moderate' && styles.moderateSeverity,
                    analysis.infectious_severity === 'mild' && styles.mildSeverity,
                  ]}>
                    <Text style={styles.severityBadgeText}>
                      {analysis.infectious_severity.toUpperCase()}
                    </Text>
                  </View>
                </View>
              )}

              {/* Contagion Warning */}
              {analysis.contagious && (
                <View style={styles.contagionWarning}>
                  <Text style={styles.contagionWarningTitle}>⚠️ CONTAGIOUS</Text>
                  <Text style={styles.contagionWarningText}>
                    Transmission Risk: <Text style={styles.boldText}>{analysis.transmission_risk?.toUpperCase() || 'UNKNOWN'}</Text>
                  </Text>
                  <Text style={styles.contagionWarningSubtext}>
                    Follow hygiene protocols to prevent spread. Avoid close contact and sharing personal items.
                  </Text>
                </View>
              )}

              {/* Treatment Recommendations */}
              {analysis.treatment_recommendations?.infectious && (
                <View style={styles.treatmentSection}>
                  <Text style={styles.treatmentTitle}>💊 Treatment Recommendations</Text>
                  {analysis.treatment_recommendations.infectious.primary_treatment?.map((treatment, index) => (
                    <View key={index} style={styles.treatmentItem}>
                      <Text style={styles.treatmentBullet}>•</Text>
                      <Text style={styles.treatmentText}>{treatment}</Text>
                    </View>
                  ))}
                  {analysis.treatment_recommendations.infectious.urgency && (
                    <View style={styles.urgencyBox}>
                      <Text style={styles.urgencyText}>{analysis.treatment_recommendations.infectious.urgency}</Text>
                    </View>
                  )}
                </View>
              )}

              {/* Differential Diagnoses for Infectious Diseases */}
              {analysis.differential_diagnoses?.infectious && analysis.differential_diagnoses.infectious.length > 0 && (
                <View style={styles.differentialSection}>
                  <Text style={styles.differentialTitle}>📊 Differential Diagnoses</Text>
                  {analysis.differential_diagnoses.infectious.slice(0, 5).map((diagnosis, index) => (
                    <View key={index} style={styles.differentialItem}>
                      <View style={styles.differentialHeader}>
                        <Text style={styles.differentialCondition}>{diagnosis.condition.replace(/_/g, ' ')}</Text>
                        <Text style={styles.differentialProbability}>{Math.round(diagnosis.probability * 100)}%</Text>
                      </View>
                      {diagnosis.description && (
                        <Text style={styles.differentialDescription}>{diagnosis.description}</Text>
                      )}
                    </View>
                  ))}
                </View>
              )}
            </View>
          )}

          {/* Burn Severity Assessment */}
          {console.log('BURN DATA CHECK:', {
            burn_severity: analysis.burn_severity,
            burn_confidence: analysis.burn_confidence,
            burn_differential: analysis.differential_diagnoses?.burn
          })}
          {(analysis.burn_severity || analysis.differential_diagnoses?.burn) && (
            <View style={styles.classificationCard}>
              <View style={styles.classificationHeader}>
                <Text style={styles.classificationIcon}>🔥</Text>
                <View style={styles.classificationInfo}>
                  <Text style={styles.classificationLabel}>Burn Severity Assessment</Text>
                  {analysis.burn_severity && (
                    <Text style={styles.classificationValue}>{analysis.burn_severity}</Text>
                  )}
                  {analysis.burn_confidence && (
                    <Text style={styles.confidenceText}>
                      Confidence: {Math.round(analysis.burn_confidence * 100)}%
                    </Text>
                  )}
                </View>
              </View>

              {/* Burn Urgency */}
              {analysis.burn_urgency && (
                <View style={styles.urgencyBox}>
                  <Text style={styles.urgencyText}>{analysis.burn_urgency}</Text>
                </View>
              )}

              {/* Burn Treatment Advice */}
              {analysis.burn_treatment_advice && (
                <View style={styles.treatmentSection}>
                  <Text style={styles.treatmentTitle}>🏥 Treatment Advice</Text>
                  <Text style={styles.treatmentText}>{analysis.burn_treatment_advice}</Text>
                </View>
              )}

              {/* Differential Diagnoses for Burns */}
              {analysis.differential_diagnoses?.burn && analysis.differential_diagnoses.burn.length > 0 && (
                <View style={styles.differentialSection}>
                  <Text style={styles.differentialTitle}>📊 Burn Assessment Details</Text>
                  {analysis.differential_diagnoses.burn.map((diagnosis, index) => (
                    <View key={index} style={styles.differentialItem}>
                      <View style={styles.differentialHeader}>
                        <Text style={styles.differentialCondition}>{diagnosis.condition}</Text>
                        <Text style={styles.differentialProbability}>{Math.round(diagnosis.probability * 100)}%</Text>
                      </View>
                      {diagnosis.urgency && (
                        <Text style={styles.differentialDescription}>⚠️ {diagnosis.urgency}</Text>
                      )}
                    </View>
                  ))}
                </View>
              )}
            </View>
          )}

          {/* Plain English Explanation Section */}
          {analysis.predicted_class && (
            <View style={styles.plainExplanationSection}>
              <View style={styles.plainExplanationHeader}>
                <Text style={styles.sectionTitle}>💬 What Does This Mean?</Text>
                <Pressable
                  style={[styles.speakButton, isSpeaking && styles.speakButtonActive]}
                  onPress={() => handleSpeakExplanation(getPlainEnglishExplanation(analysis.predicted_class))}
                >
                  <Text style={styles.speakButtonText}>
                    {isSpeaking ? '⏸️ Pause' : '🔊 Listen'}
                  </Text>
                </Pressable>
              </View>

              <View style={styles.plainExplanationCard}>
                <Text style={styles.plainExplanationIntro}>
                  In simple terms:
                </Text>
                <Text style={styles.plainExplanationText}>
                  {getPlainEnglishExplanation(analysis.predicted_class)}
                </Text>
              </View>

              <View style={styles.plainExplanationNote}>
                <Text style={styles.plainExplanationNoteText}>
                  💡 This explanation is written in everyday language to help you understand your condition better.
                  Tap the "Listen" button above to hear it read aloud. Remember to discuss this with your doctor
                  for personalized medical advice.
                </Text>
              </View>
            </View>
          )}

          {/* Export to EMR Section - Prominent Position */}
          <View style={styles.exportSection}>
            <Text style={styles.sectionTitle}>📤 Export to EMR</Text>
            <Text style={styles.exportDescription}>
              Download this analysis in HL7 FHIR format for integration with Electronic Medical Records (EMR) systems.
            </Text>
            <Pressable
              style={styles.exportButton}
              onPress={() => handleExportFHIR()}
            >
              <Text style={styles.exportButtonText}>⬇️ Download FHIR Report</Text>
            </Pressable>
            <Text style={styles.exportNote}>
              ℹ️ FHIR (Fast Healthcare Interoperability Resources) is the international standard for exchanging healthcare information electronically.
            </Text>
          </View>

          {/* Biopsy Correlation Section - HIGHLY VISIBLE */}
          <View style={[styles.biopsySection, { borderWidth: 3, borderColor: '#f59e0b' }]}>
            <Text style={[styles.sectionTitle, { fontSize: 24 }]}>🔬 BIOPSY CORRELATION</Text>
            <Text style={{ fontSize: 12, color: '#92400e', marginBottom: 10 }}>
              [DEBUG: biopsy_performed = {JSON.stringify(analysis.biopsy_performed)}]
            </Text>
            {console.log('Biopsy section rendering, biopsy_performed:', analysis.biopsy_performed)}

            {analysis.biopsy_performed === true ? (
              // Show existing biopsy result
              <View>
                <View style={[styles.biopsyResultCard, analysis.prediction_correct ? styles.biopsyCorrect : styles.biopsyIncorrect]}>
                  <Text style={styles.biopsyResultTitle}>
                    {analysis.prediction_correct ? '✅ Prediction Confirmed' : '⚠️ Prediction Differs from Biopsy'}
                  </Text>

                  <View style={styles.biopsyComparison}>
                    <View style={styles.biopsyComparisonItem}>
                      <Text style={styles.biopsyComparisonLabel}>AI Prediction:</Text>
                      <Text style={styles.biopsyComparisonValue}>{analysis.predicted_class}</Text>
                      <Text style={styles.biopsyConfidence}>
                        {Math.round((analysis.lesion_confidence || 0) * 100)}% confidence
                      </Text>
                    </View>

                    <Text style={styles.biopsyVs}>vs</Text>

                    <View style={styles.biopsyComparisonItem}>
                      <Text style={styles.biopsyComparisonLabel}>Biopsy Result:</Text>
                      <Text style={styles.biopsyComparisonValue}>{analysis.biopsy_result}</Text>
                      <Text style={styles.biopsyDate}>
                        {analysis.biopsy_date ? new Date(analysis.biopsy_date).toLocaleDateString() : 'Date not recorded'}
                      </Text>
                    </View>
                  </View>

                  {analysis.accuracy_category && (
                    <View style={styles.accuracyCategoryBadge}>
                      <Text style={styles.accuracyCategoryText}>
                        Category: {analysis.accuracy_category.replace('_', ' ').toUpperCase()}
                      </Text>
                    </View>
                  )}

                  {analysis.biopsy_facility && (
                    <Text style={styles.biopsyDetail}>📍 Facility: {analysis.biopsy_facility}</Text>
                  )}

                  {analysis.pathologist_name && (
                    <Text style={styles.biopsyDetail}>👨‍⚕️ Pathologist: {analysis.pathologist_name}</Text>
                  )}

                  {analysis.biopsy_notes && (
                    <View style={styles.biopsyNotes}>
                      <Text style={styles.biopsyNotesLabel}>Notes:</Text>
                      <Text style={styles.biopsyNotesText}>{analysis.biopsy_notes}</Text>
                    </View>
                  )}
                </View>
              </View>
            ) : (
              // Show form to add biopsy result
              <View>
                <Text style={styles.biopsyDescription}>
                  Link pathology/biopsy results to this AI prediction to track accuracy and improve the model.
                </Text>

                <Pressable
                  style={styles.addBiopsyButton}
                  onPress={() => setShowBiopsyForm(true)}
                >
                  <Text style={styles.addBiopsyButtonText}>➕ Add Biopsy Result</Text>
                </Pressable>
              </View>
            )}
          </View>

          {/* Body Map Location */}
          {(analysis.body_location || analysis.body_map_coordinates) && (
            <View style={styles.bodyMapSection}>
              <Text style={styles.sectionTitle}>📍 Lesion Location</Text>

              {analysis.body_location && (
                <View style={styles.locationInfo}>
                  <View style={styles.locationItem}>
                    <Text style={styles.locationLabel}>Body Region:</Text>
                    <Text style={styles.locationValue}>
                      {analysis.body_location.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                    </Text>
                  </View>

                  {analysis.body_sublocation && (
                    <View style={styles.locationItem}>
                      <Text style={styles.locationLabel}>Specific Area:</Text>
                      <Text style={styles.locationValue}>
                        {analysis.body_sublocation.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                      </Text>
                    </View>
                  )}

                  {analysis.body_side && (
                    <View style={styles.locationItem}>
                      <Text style={styles.locationLabel}>Side:</Text>
                      <Text style={styles.locationValue}>
                        {analysis.body_side.replace(/\b\w/g, (l: string) => l.toUpperCase())}
                      </Text>
                    </View>
                  )}
                </View>
              )}

              <Text style={styles.locationNote}>
                ℹ️ Tracking lesion locations helps monitor changes over time and identify patterns across different body areas.
              </Text>
            </View>
          )}

          {/* List of Specialists to Follow Up With */}
          {analysis.predicted_class && (
            <View style={styles.specialistSection}>
              <Text style={styles.sectionTitle}>👨‍⚕️ Specialists to Follow Up With</Text>

              <Text style={styles.specialistDescription}>
                Based on the AI prediction of {analysis.predicted_class}, we recommend consulting with the following specialists:
              </Text>

              {getRecommendedSpecialists(analysis.predicted_class).map((specialist, index) => (
                <Pressable
                  key={index}
                  style={({ pressed }) => [
                    styles.specialistCard,
                    specialist.urgency === 'urgent' && styles.specialistCardUrgent,
                    specialist.urgency === 'recommended' && styles.specialistCardRecommended,
                    specialist.urgency === 'optional' && styles.specialistCardOptional,
                    pressed && styles.specialistCardPressed
                  ]}
                  onPress={() => handleSpecialistClick(specialist)}
                >
                  <View style={styles.specialistHeader}>
                    <View style={styles.specialistTitleContainer}>
                      <Text style={styles.specialistName}>{specialist.name}</Text>
                      <View style={[
                        styles.urgencyBadge,
                        specialist.urgency === 'urgent' && styles.urgencyBadgeUrgent,
                        specialist.urgency === 'recommended' && styles.urgencyBadgeRecommended,
                        specialist.urgency === 'optional' && styles.urgencyBadgeOptional
                      ]}>
                        <Text style={[
                          styles.urgencyText,
                          specialist.urgency === 'urgent' && styles.urgencyTextUrgent,
                          specialist.urgency === 'recommended' && styles.urgencyTextRecommended,
                          specialist.urgency === 'optional' && styles.urgencyTextOptional
                        ]}>
                          {specialist.urgency === 'urgent' && '🔴 URGENT'}
                          {specialist.urgency === 'recommended' && '🟡 Recommended'}
                          {specialist.urgency === 'optional' && '🟢 Optional'}
                        </Text>
                      </View>
                    </View>
                  </View>

                  <Text style={styles.specialistReason}>{specialist.reason}</Text>

                  {specialist.urgency === 'urgent' && (
                    <Text style={styles.specialistUrgentNote}>
                      ⚠️ Please schedule an appointment as soon as possible
                    </Text>
                  )}

                  <Text style={styles.specialistFindNearby}>
                    👆 Tap to find nearby {specialist.name}s
                  </Text>
                </Pressable>
              ))}

              <View style={styles.specialistDisclaimer}>
                <Text style={styles.specialistDisclaimerText}>
                  ℹ️ These recommendations are based on the AI prediction and general medical guidelines.
                  Your primary care physician may recommend additional or different specialists based on your
                  specific situation and medical history.
                </Text>
              </View>
            </View>
          )}

          {/* Symptom Tracker Section */}
          <View style={styles.symptomSection}>
            <Text style={styles.sectionTitle}>📋 Symptom Tracker</Text>

            {(analysis.symptom_duration || analysis.symptom_itching || analysis.symptom_pain || analysis.symptom_bleeding) ? (
              // Show existing symptoms
              <View>
                {analysis.symptom_duration && (
                  <View style={styles.symptomItem}>
                    <Text style={styles.symptomLabel}>Duration:</Text>
                    <Text style={styles.symptomValue}>{analysis.symptom_duration}</Text>
                  </View>
                )}

                {analysis.symptom_changes && (
                  <View style={styles.symptomItem}>
                    <Text style={styles.symptomLabel}>Changes Observed:</Text>
                    <Text style={styles.symptomText}>{analysis.symptom_changes}</Text>
                  </View>
                )}

                <View style={styles.symptomFlags}>
                  {analysis.symptom_itching && (
                    <View style={styles.symptomFlag}>
                      <Text style={styles.symptomFlagText}>
                        🤚 Itching {analysis.symptom_itching_severity ? `(${analysis.symptom_itching_severity}/10)` : ''}
                      </Text>
                    </View>
                  )}

                  {analysis.symptom_pain && (
                    <View style={styles.symptomFlag}>
                      <Text style={styles.symptomFlagText}>
                        😖 Pain {analysis.symptom_pain_severity ? `(${analysis.symptom_pain_severity}/10)` : ''}
                      </Text>
                    </View>
                  )}

                  {analysis.symptom_bleeding && (
                    <View style={styles.symptomFlag}>
                      <Text style={styles.symptomFlagText}>
                        🩸 Bleeding {analysis.symptom_bleeding_frequency ? `(${analysis.symptom_bleeding_frequency})` : ''}
                      </Text>
                    </View>
                  )}
                </View>

                {analysis.symptom_notes && (
                  <View style={styles.symptomItem}>
                    <Text style={styles.symptomLabel}>Additional Notes:</Text>
                    <Text style={styles.symptomText}>{analysis.symptom_notes}</Text>
                  </View>
                )}

                <Pressable
                  style={styles.updateSymptomsButton}
                  onPress={() => setShowSymptomForm(true)}
                >
                  <Text style={styles.updateSymptomsButtonText}>✏️ Update Symptoms</Text>
                </Pressable>
              </View>
            ) : (
              // Show add button
              <View>
                <Text style={styles.symptomDescription}>
                  Track symptoms like duration, changes, itching, pain, and bleeding to monitor the lesion over time.
                </Text>

                <Pressable
                  style={styles.addSymptomButton}
                  onPress={() => setShowSymptomForm(true)}
                >
                  <Text style={styles.addSymptomButtonText}>➕ Record Symptoms</Text>
                </Pressable>
              </View>
            )}
          </View>

          {/* Medication List Section */}
          <View style={styles.medicationSection}>
            <Text style={styles.sectionTitle}>💊 Medication List</Text>

            {(() => {
              try {
                const medications = analysis.medications ? JSON.parse(analysis.medications) : [];
                return medications.length > 0 ? (
                  // Show existing medications
                  <View>
                    {medications.map((med: Medication, index: number) => (
                      <View key={index} style={styles.medicationCard}>
                        <View style={styles.medicationHeader}>
                          <Text style={styles.medicationName}>{med.name}</Text>
                          {med.skin_reaction === 'yes' && (
                            <View style={styles.reactionBadge}>
                              <Text style={styles.reactionBadgeText}>⚠️ Suspected Reaction</Text>
                            </View>
                          )}
                        </View>

                        {med.dosage && (
                          <Text style={styles.medicationDetail}>
                            <Text style={styles.medicationDetailLabel}>Dosage: </Text>
                            {med.dosage}
                          </Text>
                        )}

                        {med.start_date && (
                          <Text style={styles.medicationDetail}>
                            <Text style={styles.medicationDetailLabel}>Started: </Text>
                            {med.start_date}
                          </Text>
                        )}

                        {med.purpose && (
                          <Text style={styles.medicationDetail}>
                            <Text style={styles.medicationDetailLabel}>Purpose: </Text>
                            {med.purpose}
                          </Text>
                        )}
                      </View>
                    ))}

                    <Pressable
                      style={styles.updateMedicationsButton}
                      onPress={() => setShowMedicationForm(true)}
                    >
                      <Text style={styles.updateMedicationsButtonText}>✏️ Update Medications</Text>
                    </Pressable>
                  </View>
                ) : (
                  // Show add button
                  <View>
                    <Text style={styles.medicationDescription}>
                      Document medications that might cause skin reactions or photosensitivity. This helps identify potential drug-induced skin conditions.
                    </Text>

                    <Pressable
                      style={styles.addMedicationButton}
                      onPress={() => setShowMedicationForm(true)}
                    >
                      <Text style={styles.addMedicationButtonText}>➕ Add Medications</Text>
                    </Pressable>
                  </View>
                );
              } catch (e) {
                console.error('Error parsing medications:', e);
                return (
                  <View>
                    <Text style={styles.medicationDescription}>
                      Document medications that might cause skin reactions or photosensitivity.
                    </Text>

                    <Pressable
                      style={styles.addMedicationButton}
                      onPress={() => setShowMedicationForm(true)}
                    >
                      <Text style={styles.addMedicationButtonText}>➕ Add Medications</Text>
                    </Pressable>
                  </View>
                );
              }
            })()}
          </View>

          {/* Medical History Section */}
          <View style={styles.medicalHistorySection}>
            <Text style={styles.sectionTitle}>🏥 Medical History & Risk Factors</Text>

            {(analysis.family_history_skin_cancer || analysis.previous_skin_cancers ||
              analysis.immunosuppression || analysis.sun_exposure_level || analysis.history_of_sunburns ||
              analysis.tanning_bed_use || analysis.other_risk_factors) ? (
              // Show existing medical history
              <View>
                {analysis.family_history_skin_cancer && (
                  <View style={styles.riskFactorItem}>
                    <Text style={styles.riskFactorTitle}>👨‍👩‍👧‍👦 Family History of Skin Cancer</Text>
                    {analysis.family_history_details && (
                      <Text style={styles.riskFactorDetail}>{analysis.family_history_details}</Text>
                    )}
                  </View>
                )}

                {analysis.previous_skin_cancers && (
                  <View style={styles.riskFactorItem}>
                    <Text style={styles.riskFactorTitle}>📋 Previous Skin Cancers</Text>
                    {analysis.previous_skin_cancers_details && (
                      <Text style={styles.riskFactorDetail}>{analysis.previous_skin_cancers_details}</Text>
                    )}
                  </View>
                )}

                {analysis.immunosuppression && (
                  <View style={styles.riskFactorItem}>
                    <Text style={styles.riskFactorTitle}>🛡️ Immunosuppression</Text>
                    {analysis.immunosuppression_details && (
                      <Text style={styles.riskFactorDetail}>{analysis.immunosuppression_details}</Text>
                    )}
                  </View>
                )}

                {analysis.sun_exposure_level && (
                  <View style={styles.riskFactorItem}>
                    <Text style={styles.riskFactorTitle}>☀️ Sun Exposure</Text>
                    <Text style={styles.riskFactorBadge}>
                      {analysis.sun_exposure_level === 'minimal' && '🌙 Minimal'}
                      {analysis.sun_exposure_level === 'moderate' && '⛅ Moderate'}
                      {analysis.sun_exposure_level === 'high' && '☀️ High'}
                      {analysis.sun_exposure_level === 'very_high' && '🔥 Very High'}
                    </Text>
                    {analysis.sun_exposure_details && (
                      <Text style={styles.riskFactorDetail}>{analysis.sun_exposure_details}</Text>
                    )}
                  </View>
                )}

                {analysis.history_of_sunburns && (
                  <View style={styles.riskFactorItem}>
                    <Text style={styles.riskFactorTitle}>🔥 History of Severe Sunburns</Text>
                    {analysis.sunburn_details && (
                      <Text style={styles.riskFactorDetail}>{analysis.sunburn_details}</Text>
                    )}
                  </View>
                )}

                {analysis.tanning_bed_use && (
                  <View style={styles.riskFactorItem}>
                    <Text style={styles.riskFactorTitle}>🛏️ Tanning Bed Use</Text>
                    {analysis.tanning_bed_frequency && (
                      <Text style={styles.riskFactorBadge}>{analysis.tanning_bed_frequency}</Text>
                    )}
                  </View>
                )}

                {analysis.other_risk_factors && (
                  <View style={styles.riskFactorItem}>
                    <Text style={styles.riskFactorTitle}>📝 Other Risk Factors</Text>
                    <Text style={styles.riskFactorDetail}>{analysis.other_risk_factors}</Text>
                  </View>
                )}

                <Pressable
                  style={styles.updateMedicalHistoryButton}
                  onPress={() => setShowMedicalHistoryForm(true)}
                >
                  <Text style={styles.updateMedicalHistoryButtonText}>✏️ Update Medical History</Text>
                </Pressable>
              </View>
            ) : (
              // Show add button
              <View>
                <Text style={styles.medicalHistoryDescription}>
                  Track important risk factors like family history, previous skin cancers, immunosuppression, and sun exposure to assess overall risk level.
                </Text>

                <Pressable
                  style={styles.addMedicalHistoryButton}
                  onPress={() => setShowMedicalHistoryForm(true)}
                >
                  <Text style={styles.addMedicalHistoryButtonText}>➕ Add Medical History</Text>
                </Pressable>
              </View>
            )}
          </View>

          {/* Teledermatology Section */}
          <View style={styles.teledermatologySection}>
            <Pressable
              style={styles.teledermatologyButton}
              onPress={() => setShowTeledermatologyForm(true)}
            >
              <Text style={styles.teledermatologyButtonText}>
                {analysis.shared_with_dermatologist ? '👨‍⚕️ View Dermatologist Consultation' : '📤 Share with Dermatologist'}
              </Text>
            </Pressable>

            {analysis.shared_with_dermatologist && (
              <View style={styles.shareStatusBox}>
                <Text style={styles.shareStatusText}>
                  ✅ Shared with {analysis.dermatologist_name}
                </Text>
                {analysis.dermatologist_reviewed ? (
                  <Text style={styles.reviewedStatusText}>
                    ✓ Reviewed {analysis.dermatologist_review_date ? `on ${new Date(analysis.dermatologist_review_date).toLocaleDateString()}` : ''}
                  </Text>
                ) : (
                  <Text style={styles.pendingStatusText}>⏳ Awaiting review</Text>
                )}
              </View>
            )}
          </View>

          {/* Prediction Uncertainty & Confidence Intervals */}
          {analysis.probabilities_with_uncertainty && (
            <View style={styles.uncertaintySection}>
              <Text style={styles.sectionTitle}>📊 Prediction Uncertainty</Text>
              <Text style={styles.uncertaintyDescription}>
                95% Confidence Intervals show the range where we expect the true probability to fall.
                Wider ranges indicate higher uncertainty.
              </Text>

              {Object.entries(analysis.probabilities_with_uncertainty).map(([className, data]) => {
                const mean = data.mean * 100;
                const lower = data.lower * 100;
                const upper = data.upper * 100;
                const range = upper - lower;
                const isTopPrediction = className === analysis.predicted_class;

                return (
                  <View key={className} style={styles.uncertaintyItem}>
                    <View style={styles.uncertaintyHeader}>
                      <Text style={[
                        styles.uncertaintyClassName,
                        isTopPrediction && styles.uncertaintyClassNameHighlight
                      ]}>
                        {className}
                        {isTopPrediction && ' ★'}
                      </Text>
                      <Text style={styles.uncertaintyMean}>
                        {mean.toFixed(1)}%
                      </Text>
                    </View>

                    <View style={styles.uncertaintyBarContainer}>
                      {/* Background track */}
                      <View style={styles.uncertaintyTrack} />

                      {/* Confidence interval range */}
                      <View style={[
                        styles.uncertaintyRange,
                        {
                          left: `${lower}%`,
                          width: `${range}%`,
                          backgroundColor: isTopPrediction ? '#4299e1' : '#cbd5e0'
                        }
                      ]} />

                      {/* Point estimate marker */}
                      <View style={[
                        styles.uncertaintyPoint,
                        {
                          left: `${mean}%`,
                          backgroundColor: isTopPrediction ? '#2c5282' : '#4a5568'
                        }
                      ]} />
                    </View>

                    <View style={styles.uncertaintyLabels}>
                      <Text style={styles.uncertaintyLabel}>
                        95% CI: [{lower.toFixed(1)}%, {upper.toFixed(1)}%]
                      </Text>
                      <Text style={[
                        styles.uncertaintyRangeLabel,
                        { color: range > 20 ? '#dc3545' : range > 10 ? '#ffc107' : '#28a745' }
                      ]}>
                        ±{(range/2).toFixed(1)}%
                      </Text>
                    </View>
                  </View>
                );
              })}

              {analysis.uncertainty_metrics && (
                <View style={styles.uncertaintyMetrics}>
                  <Text style={styles.uncertaintyMetricsTitle}>Model Uncertainty Metrics:</Text>

                  {analysis.uncertainty_metrics.epistemic_uncertainty !== undefined && (
                    <View style={styles.uncertaintyMetricItem}>
                      <Text style={styles.uncertaintyMetricLabel}>Epistemic Uncertainty (Model):</Text>
                      <Text style={styles.uncertaintyMetricValue}>
                        {(analysis.uncertainty_metrics.epistemic_uncertainty * 100).toFixed(2)}%
                      </Text>
                    </View>
                  )}

                  {analysis.uncertainty_metrics.aleatoric_uncertainty !== undefined && (
                    <View style={styles.uncertaintyMetricItem}>
                      <Text style={styles.uncertaintyMetricLabel}>Aleatoric Uncertainty (Data):</Text>
                      <Text style={styles.uncertaintyMetricValue}>
                        {(analysis.uncertainty_metrics.aleatoric_uncertainty * 100).toFixed(2)}%
                      </Text>
                    </View>
                  )}

                  <Text style={styles.uncertaintyExplanation}>
                    Lower uncertainty values indicate higher model confidence in the prediction.
                  </Text>
                </View>
              )}
            </View>
          )}

          {/* Differential Diagnoses Section */}
          {analysis.differential_diagnoses && (
            (analysis.differential_diagnoses.lesion?.length > 0 || analysis.differential_diagnoses.inflammatory?.length > 0) && (
              <View style={styles.differentialSection}>
                <Text style={styles.sectionTitle}>🩺 Differential Diagnoses</Text>
                <Text style={styles.differentialSubtext}>
                  Ranked by probability from most to least likely
                </Text>

                {/* Lesion Differential Diagnoses */}
                {analysis.differential_diagnoses.lesion?.length > 0 && (
                  <View style={styles.differentialSubsection}>
                    <Text style={styles.differentialCategory}>Lesion Classification</Text>
                    {analysis.differential_diagnoses.lesion
                      .filter(diagnosis => typeof diagnosis.probability === 'number' && !isNaN(diagnosis.probability))
                      .map((diagnosis, index) => (
                      <View key={index} style={styles.diagnosisCard}>
                        <View style={styles.diagnosisHeader}>
                          <Text style={styles.diagnosisRank}>#{index + 1}</Text>
                          <Text style={styles.diagnosisCondition}>{diagnosis.condition}</Text>
                          <Text style={styles.diagnosisProbability}>
                            {Math.round(diagnosis.probability * 100)}%
                          </Text>
                        </View>
                        <View style={[
                          styles.severityBadge,
                          {
                            backgroundColor:
                              diagnosis.severity === 'critical' ? '#dc3545' :
                              diagnosis.severity === 'high' ? '#fd7e14' :
                              diagnosis.severity === 'medium' ? '#ffc107' :
                              '#28a745'
                          }
                        ]}>
                          <Text style={styles.severityText}>{diagnosis.severity.toUpperCase()}</Text>
                        </View>
                        <Text style={styles.diagnosisUrgency}>⏱️ {diagnosis.urgency}</Text>
                        <Text style={styles.diagnosisDescription}>{diagnosis.description}</Text>
                        <Text style={styles.diagnosisFeatures}>
                          <Text style={styles.featuresLabel}>Key features: </Text>
                          {diagnosis.key_features}
                        </Text>
                      </View>
                    ))}
                  </View>
                )}

                {/* Inflammatory Differential Diagnoses */}
                {analysis.differential_diagnoses.inflammatory?.length > 0 &&
                 analysis.differential_diagnoses.inflammatory.filter(d => typeof d.probability === 'number' && !isNaN(d.probability)).length > 0 && (
                  <View style={styles.differentialSubsection}>
                    <Text style={styles.differentialCategory}>Inflammatory Conditions</Text>
                    {analysis.differential_diagnoses.inflammatory
                      .filter(diagnosis => typeof diagnosis.probability === 'number' && !isNaN(diagnosis.probability))
                      .map((diagnosis, index) => (
                      <View key={index} style={styles.diagnosisCard}>
                        <View style={styles.diagnosisHeader}>
                          <Text style={styles.diagnosisRank}>#{index + 1}</Text>
                          <Text style={styles.diagnosisCondition}>{diagnosis.condition}</Text>
                          <Text style={styles.diagnosisProbability}>
                            {Math.round(diagnosis.probability * 100)}%
                          </Text>
                        </View>
                        <View style={[
                          styles.severityBadge,
                          {
                            backgroundColor:
                              diagnosis.severity === 'high' ? '#fd7e14' :
                              diagnosis.severity === 'medium' ? '#ffc107' :
                              '#28a745'
                          }
                        ]}>
                          <Text style={styles.severityText}>{diagnosis.severity.toUpperCase()}</Text>
                        </View>
                        <Text style={styles.diagnosisUrgency}>⏱️ {diagnosis.urgency}</Text>
                        <Text style={styles.diagnosisDescription}>{diagnosis.description}</Text>
                        <Text style={styles.diagnosisFeatures}>
                          <Text style={styles.featuresLabel}>Key features: </Text>
                          {diagnosis.key_features}
                        </Text>
                      </View>
                    ))}
                  </View>
                )}
              </View>
            )
          )}

          {/* Red Flag Indicators Section */}
          {analysis.red_flag_indicators && (
            <View style={styles.redFlagSection}>
              <Text style={styles.sectionTitle}>🚩 Melanoma Red Flag Indicators (ABCDE)</Text>
              <View style={styles.redFlagOverview}>
                <Text style={styles.redFlagRiskLabel}>Overall Risk:</Text>
                <Text style={[
                  styles.redFlagRiskValue,
                  analysis.red_flag_indicators.overall_risk === 'High' && styles.redFlagHigh,
                  analysis.red_flag_indicators.overall_risk === 'Moderate' && styles.redFlagModerate,
                  analysis.red_flag_indicators.overall_risk === 'Low-Moderate' && styles.redFlagLowModerate,
                  analysis.red_flag_indicators.overall_risk === 'Low' && styles.redFlagLow
                ]}>
                  {analysis.red_flag_indicators.overall_risk} ({analysis.red_flag_indicators.red_flags_count} flags)
                </Text>
              </View>

              <View style={styles.redFlagGrid}>
                {/* Asymmetry */}
                <View style={[
                  styles.redFlagItem,
                  analysis.red_flag_indicators.asymmetry.flag && styles.redFlagItemActive
                ]}>
                  <Text style={styles.redFlagTitle}>
                    {analysis.red_flag_indicators.asymmetry.flag ? '⚠️ ' : '✓ '}Asymmetry
                  </Text>
                  <Text style={styles.redFlagDescription}>
                    {analysis.red_flag_indicators.asymmetry.description}
                  </Text>
                </View>

                {/* Border Irregularity */}
                <View style={[
                  styles.redFlagItem,
                  analysis.red_flag_indicators.border_irregularity.flag && styles.redFlagItemActive
                ]}>
                  <Text style={styles.redFlagTitle}>
                    {analysis.red_flag_indicators.border_irregularity.flag ? '⚠️ ' : '✓ '}Border
                  </Text>
                  <Text style={styles.redFlagDescription}>
                    {analysis.red_flag_indicators.border_irregularity.description}
                  </Text>
                </View>

                {/* Color Variation */}
                <View style={[
                  styles.redFlagItem,
                  analysis.red_flag_indicators.color_variation.flag && styles.redFlagItemActive
                ]}>
                  <Text style={styles.redFlagTitle}>
                    {analysis.red_flag_indicators.color_variation.flag ? '⚠️ ' : '✓ '}Color
                  </Text>
                  <Text style={styles.redFlagDescription}>
                    {analysis.red_flag_indicators.color_variation.description}
                  </Text>
                </View>

                {/* Diameter */}
                <View style={[
                  styles.redFlagItem,
                  analysis.red_flag_indicators.diameter.flag && styles.redFlagItemActive
                ]}>
                  <Text style={styles.redFlagTitle}>
                    {analysis.red_flag_indicators.diameter.flag ? '⚠️ ' : '✓ '}Diameter
                  </Text>
                  <Text style={styles.redFlagDescription}>
                    {analysis.red_flag_indicators.diameter.description}
                  </Text>
                </View>

                {/* Evolution */}
                {analysis.red_flag_indicators.evolution && (
                  <View style={[
                    styles.redFlagItem,
                    styles.redFlagEvolution,
                    analysis.red_flag_indicators.evolution.flag && styles.redFlagItemActive
                  ]}>
                    <Text style={styles.redFlagTitle}>
                      {analysis.red_flag_indicators.evolution.flag ? '⚠️ ' : '📊 '}Evolution
                    </Text>
                    <Text style={styles.redFlagDescription}>
                      {analysis.red_flag_indicators.evolution.description}
                    </Text>
                    {analysis.red_flag_indicators.evolution.changes_detected?.length > 0 && (
                      <View style={styles.evolutionChanges}>
                        {analysis.red_flag_indicators.evolution.changes_detected.map((change, idx) => (
                          <Text key={idx} style={styles.evolutionChange}>• {change}</Text>
                        ))}
                      </View>
                    )}
                    <Text style={styles.evolutionRecommendation}>
                      {analysis.red_flag_indicators.evolution.recommendation}
                    </Text>
                  </View>
                )}
              </View>
            </View>
          )}

          {/* Second Opinion Section */}
          {analysis.second_opinion && analysis.second_opinion.requires_review && (
            <View style={styles.secondOpinionSection}>
              <View style={styles.secondOpinionHeader}>
                <Text style={styles.sectionTitle}>👨‍⚕️ Second Opinion Recommended</Text>
                <View style={[
                  styles.priorityBadge,
                  analysis.second_opinion.priority === 'Critical' && styles.priorityCritical,
                  analysis.second_opinion.priority === 'High' && styles.priorityHigh,
                  analysis.second_opinion.priority === 'Medium' && styles.priorityMedium,
                  analysis.second_opinion.priority === 'Low' && styles.priorityLow
                ]}>
                  <Text style={styles.priorityText}>{analysis.second_opinion.priority} Priority</Text>
                </View>
              </View>

              <Text style={styles.secondOpinionSummary}>{analysis.second_opinion.summary}</Text>

              {/* Reasons for Review */}
              {analysis.second_opinion.reasons && analysis.second_opinion.reasons.length > 0 && (
                <View style={styles.secondOpinionReasons}>
                  <Text style={styles.secondOpinionSubtitle}>Why Review is Needed:</Text>
                  {analysis.second_opinion.reasons.map((reason, index) => (
                    <View key={index} style={styles.reasonItem}>
                      <Text style={styles.reasonBullet}>•</Text>
                      <Text style={styles.reasonText}>{reason}</Text>
                    </View>
                  ))}
                </View>
              )}

              {/* Clinical Recommendations */}
              {analysis.second_opinion.recommendations && analysis.second_opinion.recommendations.length > 0 && (
                <View style={styles.secondOpinionRecommendations}>
                  <Text style={styles.secondOpinionSubtitle}>Recommended Actions:</Text>
                  {analysis.second_opinion.recommendations.map((recommendation, index) => (
                    <View key={index} style={styles.recommendationItem}>
                      <Text style={styles.recommendationNumber}>{index + 1}.</Text>
                      <Text style={styles.recommendationText}>{recommendation}</Text>
                    </View>
                  ))}
                </View>
              )}
            </View>
          )}

          {/* Treatment Recommendations Section */}
          {analysis.treatment_recommendations && (analysis.treatment_recommendations.lesion || analysis.treatment_recommendations.inflammatory) && (
            <View style={styles.treatmentSection}>
              <Text style={styles.sectionTitle}>💊 Treatment Recommendations</Text>

              {/* Lesion Treatment */}
              {analysis.treatment_recommendations.lesion && (
                <View style={styles.treatmentSubsection}>
                  <Text style={styles.treatmentSubtitle}>For {analysis.predicted_class}:</Text>

                  {/* Urgency */}
                  <View style={styles.treatmentUrgency}>
                    <Text style={styles.treatmentUrgencyText}>
                      ⏱️ {analysis.treatment_recommendations.lesion.urgency}
                    </Text>
                  </View>

                  {/* Warning */}
                  {analysis.treatment_recommendations.lesion.warning && (
                    <View style={styles.treatmentWarning}>
                      <Text style={styles.treatmentWarningText}>
                        {analysis.treatment_recommendations.lesion.warning}
                      </Text>
                    </View>
                  )}

                  {/* Confidence Note */}
                  {analysis.treatment_recommendations.lesion.confidence_note && (
                    <View style={styles.treatmentConfidenceNote}>
                      <Text style={styles.treatmentConfidenceNoteText}>
                        {analysis.treatment_recommendations.lesion.confidence_note}
                      </Text>
                    </View>
                  )}

                  {/* First-Line Treatment */}
                  {analysis.treatment_recommendations.lesion.first_line && analysis.treatment_recommendations.lesion.first_line.length > 0 && (
                    <View style={styles.treatmentGroup}>
                      <Text style={styles.treatmentGroupTitle}>🔹 First-Line Treatment:</Text>
                      {analysis.treatment_recommendations.lesion.first_line.map((treatment, index) => (
                        <View key={index} style={styles.treatmentItem}>
                          <Text style={styles.treatmentBullet}>•</Text>
                          <Text style={styles.treatmentText}>{treatment}</Text>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* Second-Line Treatment */}
                  {analysis.treatment_recommendations.lesion.second_line && analysis.treatment_recommendations.lesion.second_line.length > 0 && (
                    <View style={styles.treatmentGroup}>
                      <Text style={styles.treatmentGroupTitle}>🔸 Second-Line / Alternative Treatment:</Text>
                      {analysis.treatment_recommendations.lesion.second_line.map((treatment, index) => (
                        <View key={index} style={styles.treatmentItem}>
                          <Text style={styles.treatmentBullet}>•</Text>
                          <Text style={styles.treatmentText}>{treatment}</Text>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* General Care */}
                  {analysis.treatment_recommendations.lesion.general_care && analysis.treatment_recommendations.lesion.general_care.length > 0 && (
                    <View style={styles.treatmentGroup}>
                      <Text style={styles.treatmentGroupTitle}>🏠 General Care & Prevention:</Text>
                      {analysis.treatment_recommendations.lesion.general_care.map((care, index) => (
                        <View key={index} style={styles.treatmentItem}>
                          <Text style={styles.treatmentBullet}>•</Text>
                          <Text style={styles.treatmentText}>{care}</Text>
                        </View>
                      ))}
                    </View>
                  )}
                </View>
              )}

              {/* Inflammatory Condition Treatment */}
              {analysis.treatment_recommendations.inflammatory && (
                <View style={styles.treatmentSubsection}>
                  <Text style={styles.treatmentSubtitle}>For {analysis.inflammatory_condition}:</Text>

                  {/* Urgency */}
                  <View style={styles.treatmentUrgency}>
                    <Text style={styles.treatmentUrgencyText}>
                      ⏱️ {analysis.treatment_recommendations.inflammatory.urgency}
                    </Text>
                  </View>

                  {/* Warning */}
                  {analysis.treatment_recommendations.inflammatory.warning && (
                    <View style={styles.treatmentWarning}>
                      <Text style={styles.treatmentWarningText}>
                        {analysis.treatment_recommendations.inflammatory.warning}
                      </Text>
                    </View>
                  )}

                  {/* Confidence Note */}
                  {analysis.treatment_recommendations.inflammatory.confidence_note && (
                    <View style={styles.treatmentConfidenceNote}>
                      <Text style={styles.treatmentConfidenceNoteText}>
                        {analysis.treatment_recommendations.inflammatory.confidence_note}
                      </Text>
                    </View>
                  )}

                  {/* First-Line Treatment */}
                  {analysis.treatment_recommendations.inflammatory.first_line && analysis.treatment_recommendations.inflammatory.first_line.length > 0 && (
                    <View style={styles.treatmentGroup}>
                      <Text style={styles.treatmentGroupTitle}>🔹 First-Line Treatment:</Text>
                      {analysis.treatment_recommendations.inflammatory.first_line.map((treatment, index) => (
                        <View key={index} style={styles.treatmentItem}>
                          <Text style={styles.treatmentBullet}>•</Text>
                          <Text style={styles.treatmentText}>{treatment}</Text>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* Second-Line Treatment */}
                  {analysis.treatment_recommendations.inflammatory.second_line && analysis.treatment_recommendations.inflammatory.second_line.length > 0 && (
                    <View style={styles.treatmentGroup}>
                      <Text style={styles.treatmentGroupTitle}>🔸 Second-Line / Alternative Treatment:</Text>
                      {analysis.treatment_recommendations.inflammatory.second_line.map((treatment, index) => (
                        <View key={index} style={styles.treatmentItem}>
                          <Text style={styles.treatmentBullet}>•</Text>
                          <Text style={styles.treatmentText}>{treatment}</Text>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* General Care */}
                  {analysis.treatment_recommendations.inflammatory.general_care && analysis.treatment_recommendations.inflammatory.general_care.length > 0 && (
                    <View style={styles.treatmentGroup}>
                      <Text style={styles.treatmentGroupTitle}>🏠 General Care & Prevention:</Text>
                      {analysis.treatment_recommendations.inflammatory.general_care.map((care, index) => (
                        <View key={index} style={styles.treatmentItem}>
                          <Text style={styles.treatmentBullet}>•</Text>
                          <Text style={styles.treatmentText}>{care}</Text>
                        </View>
                      ))}
                    </View>
                  )}
                </View>
              )}

              <View style={styles.treatmentDisclaimer}>
                <Text style={styles.treatmentDisclaimerText}>
                  ⚕️ Important: These are general treatment recommendations. Always consult with a healthcare provider before starting any treatment. Your dermatologist will tailor treatment based on your specific case, medical history, and individual factors.
                </Text>
              </View>
            </View>
          )}

          {/* OTC Recommendations Section */}
          {analysis.otc_recommendations && (
            <View style={styles.otcSection}>
              <Text style={styles.sectionTitle}>🏪 Over-the-Counter Options</Text>

              {analysis.otc_recommendations.applicable ? (
                <View>
                  {/* OTC Recommendations List */}
                  {analysis.otc_recommendations.recommendations && analysis.otc_recommendations.recommendations.map((rec, index) => (
                    <View key={index} style={styles.otcCategory}>
                      <Text style={styles.otcCategoryTitle}>{rec.category}</Text>

                      {/* Examples */}
                      {rec.examples && rec.examples.length > 0 && (
                        <View style={styles.otcExamples}>
                          <Text style={styles.otcLabel}>Products:</Text>
                          {rec.examples.map((example, idx) => (
                            <View key={idx} style={styles.otcItem}>
                              <Text style={styles.otcBullet}>•</Text>
                              <Text style={styles.otcText}>{example}</Text>
                            </View>
                          ))}
                        </View>
                      )}

                      {/* Usage */}
                      {rec.usage && (
                        <View style={styles.otcUsage}>
                          <Text style={styles.otcLabel}>How to use:</Text>
                          <Text style={styles.otcUsageText}>{rec.usage}</Text>
                        </View>
                      )}

                      {/* Duration */}
                      {rec.duration && (
                        <View style={styles.otcDuration}>
                          <Text style={styles.otcLabel}>Duration:</Text>
                          <Text style={styles.otcDurationText}>{rec.duration}</Text>
                        </View>
                      )}

                      {/* Warnings */}
                      {rec.warnings && rec.warnings.length > 0 && (
                        <View style={styles.otcWarnings}>
                          <Text style={styles.otcWarningLabel}>⚠️ Warnings:</Text>
                          {rec.warnings.map((warning, idx) => (
                            <View key={idx} style={styles.otcItem}>
                              <Text style={styles.otcBullet}>•</Text>
                              <Text style={styles.otcWarningText}>{warning}</Text>
                            </View>
                          ))}
                        </View>
                      )}

                      {/* Contraindications */}
                      {rec.contraindications && rec.contraindications.length > 0 && (
                        <View style={styles.otcContraindications}>
                          <Text style={styles.otcContraindicationLabel}>🚫 Do not use if:</Text>
                          {rec.contraindications.map((contra, idx) => (
                            <View key={idx} style={styles.otcItem}>
                              <Text style={styles.otcBullet}>•</Text>
                              <Text style={styles.otcContraindicationText}>{contra}</Text>
                            </View>
                          ))}
                        </View>
                      )}
                    </View>
                  ))}

                  {/* General Advice */}
                  {analysis.otc_recommendations.general_advice && analysis.otc_recommendations.general_advice.length > 0 && (
                    <View style={styles.otcGeneralAdvice}>
                      <Text style={styles.otcAdviceTitle}>💡 General Advice:</Text>
                      {analysis.otc_recommendations.general_advice.map((advice, idx) => (
                        <View key={idx} style={styles.otcItem}>
                          <Text style={styles.otcBullet}>•</Text>
                          <Text style={styles.otcText}>{advice}</Text>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* When to Seek Care */}
                  {analysis.otc_recommendations.when_to_seek_care && analysis.otc_recommendations.when_to_seek_care.length > 0 && (
                    <View style={styles.otcSeekCare}>
                      <Text style={styles.otcSeekCareTitle}>🏥 See a doctor if:</Text>
                      {analysis.otc_recommendations.when_to_seek_care.map((reason, idx) => (
                        <View key={idx} style={styles.otcItem}>
                          <Text style={styles.otcBullet}>•</Text>
                          <Text style={styles.otcSeekCareText}>{reason}</Text>
                        </View>
                      ))}
                    </View>
                  )}
                </View>
              ) : (
                <View style={styles.otcNotApplicable}>
                  <Text style={styles.otcNotApplicableTitle}>
                    ⚕️ {analysis.otc_recommendations.reason_not_applicable || 'OTC treatment not recommended'}
                  </Text>

                  {analysis.otc_recommendations.see_doctor_reasons && analysis.otc_recommendations.see_doctor_reasons.length > 0 && (
                    <View style={styles.otcSeeDoctorReasons}>
                      {analysis.otc_recommendations.see_doctor_reasons.map((reason, idx) => (
                        <View key={idx} style={styles.otcItem}>
                          <Text style={styles.otcBullet}>•</Text>
                          <Text style={styles.otcSeeDoctorText}>{reason}</Text>
                        </View>
                      ))}
                    </View>
                  )}
                </View>
              )}

              {/* Disclaimer */}
              <View style={styles.otcDisclaimer}>
                <Text style={styles.otcDisclaimerText}>
                  {analysis.otc_recommendations.disclaimer ||
                    '⚠️ These suggestions are for educational purposes only and do not constitute medical advice. Always read product labels and consult a healthcare provider before starting any treatment.'}
                </Text>
              </View>
            </View>
          )}

          {/* Lab Results Insights Section */}
          {(labResults || suggestedLabTests) && (
            <View style={styles.labResultsSection}>
              <Text style={styles.sectionTitle}>🔬 Lab Results & Skin Health</Text>

              {isLoadingLabResults ? (
                <ActivityIndicator size="small" color="#6366f1" />
              ) : (
                <>
                  {/* User's Lab Results Analysis */}
                  {labResults && (
                    <View style={styles.labResultsCard}>
                      <View style={styles.labResultsHeader}>
                        <Text style={styles.labResultsHeaderText}>Your Recent Lab Results</Text>
                        <View style={[
                          styles.labHealthScoreBadge,
                          { backgroundColor: labResults.skin_health_score >= 80 ? '#10b981' :
                                            labResults.skin_health_score >= 60 ? '#f59e0b' : '#ef4444' }
                        ]}>
                          <Text style={styles.labHealthScoreText}>
                            {labResults.skin_health_score}% Skin Health
                          </Text>
                        </View>
                      </View>

                      {/* Abnormal Values Summary */}
                      {labResults.abnormal_count > 0 && (
                        <View style={styles.abnormalSummary}>
                          <Text style={styles.abnormalSummaryText}>
                            {labResults.abnormal_count} value(s) outside normal range
                          </Text>
                        </View>
                      )}

                      {/* Skin-Relevant Findings */}
                      {labResults.skin_relevance_analysis?.relevant_findings?.length > 0 && (
                        <View style={styles.labFindingsContainer}>
                          <Text style={styles.labFindingsTitle}>Findings Relevant to Your Skin:</Text>
                          {labResults.skin_relevance_analysis.relevant_findings.map((finding: any, idx: number) => (
                            <View key={idx} style={styles.labFindingItem}>
                              <View style={styles.labFindingHeader}>
                                <Text style={styles.labFindingName}>{finding.finding}</Text>
                                <View style={[
                                  styles.labSeverityBadge,
                                  { backgroundColor: finding.severity === 'high' ? '#ef4444' :
                                                    finding.severity === 'moderate' ? '#f59e0b' : '#6366f1' }
                                ]}>
                                  <Text style={styles.labSeverityText}>{finding.severity}</Text>
                                </View>
                              </View>
                              <Text style={styles.labFindingExplanation}>{finding.skin_impact}</Text>
                              {finding.related_conditions?.length > 0 && (
                                <Text style={styles.labRelatedConditions}>
                                  Related conditions: {finding.related_conditions.join(', ')}
                                </Text>
                              )}
                            </View>
                          ))}
                        </View>
                      )}

                      {/* Recommendations from Lab Analysis */}
                      {labResults.skin_relevance_analysis?.recommendations?.length > 0 && (
                        <View style={styles.labRecommendationsContainer}>
                          <Text style={styles.labRecommendationsTitle}>Recommendations:</Text>
                          {labResults.skin_relevance_analysis.recommendations.map((rec: string, idx: number) => (
                            <View key={idx} style={styles.labRecommendationItem}>
                              <Text style={styles.labRecommendationBullet}>•</Text>
                              <Text style={styles.labRecommendationText}>{rec}</Text>
                            </View>
                          ))}
                        </View>
                      )}

                      <Pressable
                        style={styles.viewAllLabsButton}
                        onPress={() => router.push('/lab-results')}
                      >
                        <Text style={styles.viewAllLabsButtonText}>View All Lab Results →</Text>
                      </Pressable>
                    </View>
                  )}

                  {/* No Lab Results - Suggest Getting Tested */}
                  {/* Only show if neither labResults nor lab_context has data */}
                  {!labResults && !analysis?.lab_context?.has_lab_context && suggestedLabTests && (
                    <View style={styles.noLabResultsCard}>
                      <Text style={styles.noLabResultsTitle}>
                        No lab results on file
                      </Text>
                      <Text style={styles.noLabResultsDescription}>
                        Lab tests can provide valuable insights about factors affecting your skin health.
                      </Text>
                      <Pressable
                        style={styles.addLabResultsButton}
                        onPress={() => router.push('/lab-results')}
                      >
                        <Text style={styles.addLabResultsButtonText}>+ Add Lab Results</Text>
                      </Pressable>
                    </View>
                  )}

                  {/* Suggested Lab Tests for This Condition */}
                  {suggestedLabTests?.suggested_tests?.length > 0 && (
                    <View style={styles.suggestedTestsContainer}>
                      <Text style={styles.suggestedTestsTitle}>
                        Suggested Lab Tests for {analysis.predicted_class}
                      </Text>
                      <Text style={styles.suggestedTestsSubtitle}>
                        These tests may help identify underlying factors:
                      </Text>
                      {suggestedLabTests.suggested_tests.map((test: any, idx: number) => (
                        <View key={idx} style={styles.suggestedTestItem}>
                          <View style={styles.suggestedTestHeader}>
                            <Text style={styles.suggestedTestName}>{test.test_name}</Text>
                            <View style={[
                              styles.priorityBadge,
                              { backgroundColor: test.priority === 'high' ? '#ef4444' :
                                                test.priority === 'medium' ? '#f59e0b' : '#6366f1' }
                            ]}>
                              <Text style={styles.priorityText}>{test.priority} priority</Text>
                            </View>
                          </View>
                          <Text style={styles.suggestedTestReason}>{test.reason}</Text>
                        </View>
                      ))}
                    </View>
                  )}

                  <View style={styles.labDisclaimer}>
                    <Text style={styles.labDisclaimerText}>
                      Lab test recommendations are for informational purposes. Consult your healthcare provider
                      to determine which tests are appropriate for your situation.
                    </Text>
                  </View>
                </>
              )}
            </View>
          )}

          {/* Literature References Section */}
          {analysis.literature_references && (analysis.literature_references.lesion || analysis.literature_references.inflammatory) && (
            <View style={styles.literatureSection}>
              <Text style={styles.sectionTitle}>📚 Clinical Literature & Guidelines</Text>

              {/* Lesion Literature */}
              {analysis.literature_references.lesion && (
                <View style={styles.literatureSubsection}>
                  <Text style={styles.literatureSubtitle}>For {analysis.predicted_class}:</Text>

                  {/* Clinical Guidelines */}
                  {analysis.literature_references.lesion.guidelines && analysis.literature_references.lesion.guidelines.length > 0 && (
                    <View style={styles.referenceGroup}>
                      <Text style={styles.referenceGroupTitle}>📋 Clinical Practice Guidelines:</Text>
                      {analysis.literature_references.lesion.guidelines.map((guideline, index) => (
                        <View key={index} style={styles.referenceItem}>
                          <Text style={styles.referenceBullet}>•</Text>
                          <View style={styles.referenceContent}>
                            <Text style={styles.referenceTitle}>{guideline.title}</Text>
                            <Text style={styles.referenceOrg}>{guideline.organization} ({guideline.year})</Text>
                            <Text style={styles.referenceLink} onPress={() => Linking.openURL(guideline.url)}>
                              View Guideline →
                            </Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* Review Articles */}
                  {analysis.literature_references.lesion.reviews && analysis.literature_references.lesion.reviews.length > 0 && (
                    <View style={styles.referenceGroup}>
                      <Text style={styles.referenceGroupTitle}>🔬 Research & Review Articles:</Text>
                      {analysis.literature_references.lesion.reviews.map((review, index) => (
                        <View key={index} style={styles.referenceItem}>
                          <Text style={styles.referenceBullet}>•</Text>
                          <View style={styles.referenceContent}>
                            <Text style={styles.referenceTitle}>{review.title}</Text>
                            <Text style={styles.referenceOrg}>{review.journal}</Text>
                            <Text style={styles.referenceLink} onPress={() => Linking.openURL(review.url)}>
                              Read Article →
                            </Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* Patient Resources */}
                  {analysis.literature_references.lesion.patient_resources && analysis.literature_references.lesion.patient_resources.length > 0 && (
                    <View style={styles.referenceGroup}>
                      <Text style={styles.referenceGroupTitle}>ℹ️ Patient Education Resources:</Text>
                      {analysis.literature_references.lesion.patient_resources.map((resource, index) => (
                        <View key={index} style={styles.referenceItem}>
                          <Text style={styles.referenceBullet}>•</Text>
                          <View style={styles.referenceContent}>
                            <Text style={styles.referenceTitle}>{resource.title}</Text>
                            <Text style={styles.referenceOrg}>{resource.organization}</Text>
                            <Text style={styles.referenceLink} onPress={() => Linking.openURL(resource.url)}>
                              Learn More →
                            </Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  )}
                </View>
              )}

              {/* Inflammatory Condition Literature */}
              {analysis.literature_references.inflammatory && (
                <View style={styles.literatureSubsection}>
                  <Text style={styles.literatureSubtitle}>For {analysis.inflammatory_condition}:</Text>

                  {/* Clinical Guidelines */}
                  {analysis.literature_references.inflammatory.guidelines && analysis.literature_references.inflammatory.guidelines.length > 0 && (
                    <View style={styles.referenceGroup}>
                      <Text style={styles.referenceGroupTitle}>📋 Clinical Practice Guidelines:</Text>
                      {analysis.literature_references.inflammatory.guidelines.map((guideline, index) => (
                        <View key={index} style={styles.referenceItem}>
                          <Text style={styles.referenceBullet}>•</Text>
                          <View style={styles.referenceContent}>
                            <Text style={styles.referenceTitle}>{guideline.title}</Text>
                            <Text style={styles.referenceOrg}>{guideline.organization} ({guideline.year})</Text>
                            <Text style={styles.referenceLink} onPress={() => Linking.openURL(guideline.url)}>
                              View Guideline →
                            </Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* Review Articles */}
                  {analysis.literature_references.inflammatory.reviews && analysis.literature_references.inflammatory.reviews.length > 0 && (
                    <View style={styles.referenceGroup}>
                      <Text style={styles.referenceGroupTitle}>🔬 Research & Review Articles:</Text>
                      {analysis.literature_references.inflammatory.reviews.map((review, index) => (
                        <View key={index} style={styles.referenceItem}>
                          <Text style={styles.referenceBullet}>•</Text>
                          <View style={styles.referenceContent}>
                            <Text style={styles.referenceTitle}>{review.title}</Text>
                            <Text style={styles.referenceOrg}>{review.journal}</Text>
                            <Text style={styles.referenceLink} onPress={() => Linking.openURL(review.url)}>
                              Read Article →
                            </Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* Patient Resources */}
                  {analysis.literature_references.inflammatory.patient_resources && analysis.literature_references.inflammatory.patient_resources.length > 0 && (
                    <View style={styles.referenceGroup}>
                      <Text style={styles.referenceGroupTitle}>ℹ️ Patient Education Resources:</Text>
                      {analysis.literature_references.inflammatory.patient_resources.map((resource, index) => (
                        <View key={index} style={styles.referenceItem}>
                          <Text style={styles.referenceBullet}>•</Text>
                          <View style={styles.referenceContent}>
                            <Text style={styles.referenceTitle}>{resource.title}</Text>
                            <Text style={styles.referenceOrg}>{resource.organization}</Text>
                            <Text style={styles.referenceLink} onPress={() => Linking.openURL(resource.url)}>
                              Learn More →
                            </Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  )}
                </View>
              )}
            </View>
          )}

          {/* Clinical Decision Support System */}
          {analysis.clinical_decision_support && !analysis.clinical_decision_support.error && (
            <View style={styles.clinicalSupportSection}>
              <View style={styles.clinicalSupportHeader}>
                <Text style={styles.clinicalSupportTitle}>⚕️ Clinical Decision Support</Text>
                <Text style={styles.clinicalSupportSubtitle}>Evidence-Based Treatment Protocol</Text>
              </View>

              {/* Urgency Banner */}
              {analysis.clinical_decision_support.urgency && (
                <View style={[
                  styles.urgencyBanner,
                  analysis.clinical_decision_support.urgency === 'URGENT' && styles.urgencyBannerCritical,
                  analysis.clinical_decision_support.urgency === 'HIGH' && styles.urgencyBannerHigh,
                  analysis.clinical_decision_support.urgency === 'MODERATE' && styles.urgencyBannerModerate,
                  analysis.clinical_decision_support.urgency === 'ROUTINE' && styles.urgencyBannerRoutine
                ]}>
                  <Text style={styles.urgencyText}>
                    {analysis.clinical_decision_support.urgency === 'URGENT' && '🚨 URGENT'}
                    {analysis.clinical_decision_support.urgency === 'HIGH' && '⚠️ HIGH PRIORITY'}
                    {analysis.clinical_decision_support.urgency === 'MODERATE' && '📋 MODERATE'}
                    {analysis.clinical_decision_support.urgency === 'ROUTINE' && '✓ ROUTINE'}
                  </Text>
                  <Text style={styles.urgencyTimeline}>
                    {analysis.clinical_decision_support.timeline}
                  </Text>
                </View>
              )}

              {/* Treatment Protocol Steps */}
              {analysis.clinical_decision_support.first_line && analysis.clinical_decision_support.first_line.length > 0 && (
                <View style={styles.treatmentProtocolCard}>
                  <Text style={styles.cardTitle}>📋 Treatment Protocol</Text>

                  {analysis.clinical_decision_support.first_line.map((step, index) => (
                    <View key={index} style={styles.protocolStep}>
                      <View style={styles.stepHeader}>
                        <View style={[
                          styles.stepNumber,
                          step.priority === 'critical' && styles.stepNumberCritical,
                          step.priority === 'high' && styles.stepNumberHigh,
                          step.priority === 'moderate' && styles.stepNumberModerate
                        ]}>
                          <Text style={styles.stepNumberText}>{index + 1}</Text>
                        </View>
                        <View style={styles.stepContent}>
                          <Text style={styles.stepAction}>{step.action}</Text>
                          <Text style={styles.stepTimeframe}>⏱️ {step.timeframe}</Text>
                        </View>
                      </View>
                      <Text style={styles.stepRationale}>
                        💡 {step.rationale}
                      </Text>
                    </View>
                  ))}
                </View>
              )}

              {/* Medications */}
              {analysis.clinical_decision_support.medications && analysis.clinical_decision_support.medications.length > 0 && (
                <View style={styles.medicationsCard}>
                  <Text style={styles.cardTitle}>💊 Medication Options</Text>

                  {analysis.clinical_decision_support.medications.map((med, index) => (
                    <View key={index} style={styles.medicationCard}>
                      <View style={styles.medicationHeader}>
                        <Text style={styles.medicationName}>{med.name}</Text>
                        {med.response_rate && (
                          <View style={styles.responseRateBadge}>
                            <Text style={styles.responseRateText}>
                              {med.response_rate} effective
                            </Text>
                          </View>
                        )}
                      </View>

                      <Text style={styles.medicationIndication}>
                        📌 {med.indication}
                      </Text>

                      <View style={styles.medicationDetails}>
                        <View style={styles.detailRow}>
                          <Text style={styles.detailLabel}>Dosage:</Text>
                          <Text style={styles.detailValue}>{med.dosage}</Text>
                        </View>
                        <View style={styles.detailRow}>
                          <Text style={styles.detailLabel}>Duration:</Text>
                          <Text style={styles.detailValue}>{med.duration}</Text>
                        </View>
                        {med.cost_range && (
                          <View style={styles.detailRow}>
                            <Text style={styles.detailLabel}>Cost:</Text>
                            <Text style={styles.detailValue}>{med.cost_range}</Text>
                          </View>
                        )}
                        {med.insurance_coverage && (
                          <View style={styles.detailRow}>
                            <Text style={styles.detailLabel}>Insurance:</Text>
                            <Text style={[
                              styles.detailValue,
                              med.insurance_coverage.includes('covered') ? styles.insuranceCovered : styles.insuranceNotCovered
                            ]}>
                              {med.insurance_coverage}
                            </Text>
                          </View>
                        )}
                      </View>

                      {med.contraindications && med.contraindications.length > 0 && (
                        <View style={styles.contraindicationsBox}>
                          <Text style={styles.contraindicationsTitle}>⚠️ Contraindications:</Text>
                          <Text style={styles.contraindicationsText}>
                            {med.contraindications.join(', ')}
                          </Text>
                        </View>
                      )}

                      {med.side_effects && med.side_effects.length > 0 && (
                        <View style={styles.sideEffectsBox}>
                          <Text style={styles.sideEffectsTitle}>Side Effects:</Text>
                          <Text style={styles.sideEffectsText}>
                            {med.side_effects.join(', ')}
                          </Text>
                        </View>
                      )}

                      {med.monitoring && (
                        <View style={styles.monitoringBox}>
                          <Text style={styles.monitoringTitle}>🔬 Monitoring Required:</Text>
                          <Text style={styles.monitoringText}>{med.monitoring}</Text>
                        </View>
                      )}
                    </View>
                  ))}
                </View>
              )}

              {/* Drug Interactions */}
              {analysis.clinical_decision_support.drug_interactions && analysis.clinical_decision_support.drug_interactions.length > 0 && (
                <View style={styles.drugInteractionsCard}>
                  <Text style={styles.cardTitle}>⚠️ Drug Interaction Warnings</Text>

                  {analysis.clinical_decision_support.drug_interactions.map((interaction, index) => (
                    <View key={index} style={[
                      styles.interactionWarning,
                      interaction.severity === 'MAJOR' && styles.interactionMajor,
                      interaction.severity === 'CRITICAL' && styles.interactionCritical
                    ]}>
                      <View style={styles.interactionHeader}>
                        <Text style={styles.interactionSeverity}>
                          {interaction.severity === 'CRITICAL' && '🔴 CRITICAL'}
                          {interaction.severity === 'MAJOR' && '🟠 MAJOR'}
                          {interaction.severity === 'MODERATE' && '🟡 MODERATE'}
                        </Text>
                      </View>
                      <Text style={styles.interactionWarningText}>{interaction.warning}</Text>
                      <Text style={styles.interactionAction}>
                        → {interaction.action}
                      </Text>
                    </View>
                  ))}
                </View>
              )}

              {/* Insurance & Billing */}
              {analysis.clinical_decision_support.insurance_codes && (
                <View style={styles.insuranceCard}>
                  <Text style={styles.cardTitle}>💳 Insurance & Billing</Text>

                  <View style={styles.insuranceContent}>
                    <View style={styles.codeRow}>
                      <Text style={styles.codeLabel}>ICD-10 Code:</Text>
                      <Text style={styles.codeValue}>{analysis.clinical_decision_support.insurance_codes.icd10}</Text>
                    </View>

                    {analysis.clinical_decision_support.insurance_codes.cpt && (
                      <View style={styles.codeRow}>
                        <Text style={styles.codeLabel}>CPT Codes:</Text>
                        <Text style={styles.codeValue}>
                          {Array.isArray(analysis.clinical_decision_support.insurance_codes.cpt)
                            ? analysis.clinical_decision_support.insurance_codes.cpt.join(', ')
                            : analysis.clinical_decision_support.insurance_codes.cpt}
                        </Text>
                      </View>
                    )}

                    {analysis.clinical_decision_support.insurance_codes.pre_auth_required && (
                      <View style={styles.preAuthWarning}>
                        <Text style={styles.preAuthText}>
                          ⚠️ Pre-Authorization Required
                        </Text>
                        <Text style={styles.preAuthSubtext}>
                          Documentation needed: {analysis.clinical_decision_support.insurance_codes.documentation}
                        </Text>
                      </View>
                    )}
                  </View>
                </View>
              )}

              {/* Follow-Up Schedule */}
              {analysis.clinical_decision_support.follow_up && (
                <View style={styles.followUpCard}>
                  <Text style={styles.cardTitle}>📅 Follow-Up Schedule</Text>

                  <View style={styles.followUpTimeline}>
                    {analysis.clinical_decision_support.follow_up.initial && (
                      <View style={styles.followUpItem}>
                        <View style={styles.followUpDot} />
                        <View style={styles.followUpContent}>
                          <Text style={styles.followUpLabel}>Initial Phase</Text>
                          <Text style={styles.followUpValue}>{analysis.clinical_decision_support.follow_up.initial}</Text>
                        </View>
                      </View>
                    )}

                    {analysis.clinical_decision_support.follow_up.long_term && (
                      <View style={styles.followUpItem}>
                        <View style={styles.followUpDot} />
                        <View style={styles.followUpContent}>
                          <Text style={styles.followUpLabel}>Long-term</Text>
                          <Text style={styles.followUpValue}>{analysis.clinical_decision_support.follow_up.long_term}</Text>
                        </View>
                      </View>
                    )}

                    {analysis.clinical_decision_support.follow_up.imaging && (
                      <View style={styles.followUpItem}>
                        <View style={styles.followUpDot} />
                        <View style={styles.followUpContent}>
                          <Text style={styles.followUpLabel}>Imaging</Text>
                          <Text style={styles.followUpValue}>{analysis.clinical_decision_support.follow_up.imaging}</Text>
                        </View>
                      </View>
                    )}
                  </View>
                </View>
              )}

              {/* Patient Education */}
              {analysis.clinical_decision_support.patient_education && analysis.clinical_decision_support.patient_education.length > 0 && (
                <View style={styles.patientEducationCard}>
                  <Text style={styles.cardTitle}>📚 Patient Education</Text>
                  {analysis.clinical_decision_support.patient_education.map((item, index) => (
                    <View key={index} style={styles.educationItem}>
                      <Text style={styles.educationBullet}>•</Text>
                      <Text style={styles.educationText}>{item}</Text>
                    </View>
                  ))}
                </View>
              )}

              <View style={styles.clinicalDisclaimer}>
                <Text style={styles.clinicalDisclaimerText}>
                  ⚕️ This clinical decision support is based on AAD/NCCN evidence-based guidelines.
                  Final treatment decisions should be made by a licensed physician considering
                  patient-specific factors. This is not a substitute for professional medical judgment.
                </Text>
              </View>
            </View>
          )}

          {/* Lab Results Context - Shows how user's lab values affect this diagnosis */}
          {analysis.lab_context && (
            <View style={styles.labContextSection}>
              <View style={styles.labContextHeader}>
                <Text style={styles.labContextTitle}>🧪 Lab Results Impact</Text>
                <Text style={styles.labContextSubtitle}>
                  {analysis.lab_context.has_lab_context
                    ? `Analysis enhanced with your lab results`
                    : 'Add lab results for personalized insights'}
                </Text>
              </View>

              {analysis.lab_context.has_lab_context ? (
                <>
                  {/* Lab Data Info */}
                  {analysis.lab_context.lab_data_date && (
                    <View style={styles.labDataInfo}>
                      <Text style={styles.labDataInfoText}>
                        📅 Using lab results from {analysis.lab_context.lab_data_date}
                      </Text>
                      {analysis.lab_context.abnormal_labs_count > 0 && (
                        <Text style={styles.labDataInfoText}>
                          ⚠️ {analysis.lab_context.abnormal_labs_count} abnormal value(s) with skin relevance
                        </Text>
                      )}
                    </View>
                  )}

                  {/* Supporting Labs - Labs that reinforce the diagnosis */}
                  {analysis.lab_context.supporting_labs && analysis.lab_context.supporting_labs.length > 0 && (
                    <View style={styles.labSupportingCard}>
                      <Text style={styles.labCardTitle}>✓ Labs Supporting This Diagnosis</Text>
                      {analysis.lab_context.supporting_labs.map((lab: any, index: number) => (
                        <View key={index} style={styles.labItem}>
                          <View style={styles.labItemHeader}>
                            <Text style={styles.labName}>{lab.name}</Text>
                            <View style={[
                              styles.labStatusBadge,
                              lab.status === 'low' ? styles.labStatusLow : styles.labStatusHigh
                            ]}>
                              <Text style={styles.labStatusText}>
                                {lab.status === 'low' ? '↓ LOW' : '↑ HIGH'}
                              </Text>
                            </View>
                          </View>
                          <Text style={styles.labValue}>Value: {lab.value}</Text>
                          <Text style={styles.labExplanation}>{lab.explanation}</Text>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* Lab Insights - How labs affected confidence */}
                  {analysis.lab_context.lab_insights && analysis.lab_context.lab_insights.length > 0 && (
                    <View style={styles.labInsightsCard}>
                      <Text style={styles.labCardTitle}>📊 Confidence Adjustments</Text>
                      {analysis.lab_context.lab_insights.map((insight: any, index: number) => (
                        <View key={index} style={styles.insightItem}>
                          <Text style={styles.insightCondition}>{insight.condition}</Text>
                          <View style={styles.insightAdjustment}>
                            <Text style={styles.insightOriginal}>
                              {insight.original_confidence}%
                            </Text>
                            <Text style={styles.insightArrow}>→</Text>
                            <Text style={[
                              styles.insightAdjusted,
                              insight.adjustment > 0 ? styles.adjustmentPositive : styles.adjustmentNegative
                            ]}>
                              {insight.adjusted_confidence}%
                            </Text>
                            <Text style={[
                              styles.insightDelta,
                              insight.adjustment > 0 ? styles.adjustmentPositive : styles.adjustmentNegative
                            ]}>
                              ({insight.adjustment > 0 ? '+' : ''}{insight.adjustment}%)
                            </Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* Other Relevant Labs */}
                  {analysis.lab_context.other_relevant_labs && analysis.lab_context.other_relevant_labs.length > 0 && (
                    <View style={styles.labOtherCard}>
                      <Text style={styles.labCardTitle}>ℹ️ Other Abnormal Labs</Text>
                      <Text style={styles.labOtherSubtitle}>
                        These may affect skin health but aren't directly related to this diagnosis
                      </Text>
                      {analysis.lab_context.other_relevant_labs.map((lab: any, index: number) => (
                        <View key={index} style={styles.labItemCompact}>
                          <Text style={styles.labNameCompact}>{lab.name}</Text>
                          <Text style={styles.labStatusCompact}>
                            {lab.status === 'low' ? '↓' : '↑'} {lab.value}
                          </Text>
                        </View>
                      ))}
                    </View>
                  )}

                  {/* Lab-Based Recommendations */}
                  {analysis.lab_context.lab_recommendations && analysis.lab_context.lab_recommendations.length > 0 && (
                    <View style={styles.labRecommendationsCard}>
                      <Text style={styles.labCardTitle}>💡 Lab-Based Recommendations</Text>
                      {analysis.lab_context.lab_recommendations.map((rec: string, index: number) => (
                        <View key={index} style={styles.labRecItem}>
                          <Text style={styles.labRecBullet}>•</Text>
                          <Text style={styles.labRecText}>{rec}</Text>
                        </View>
                      ))}
                    </View>
                  )}
                </>
              ) : (
                <View style={styles.noLabDataCard}>
                  <Text style={styles.noLabDataText}>
                    {analysis.lab_context.message || 'No lab results available'}
                  </Text>
                  <Pressable
                    style={styles.addLabButton}
                    onPress={() => router.push('/lab-results')}
                  >
                    <Text style={styles.addLabButtonText}>+ Add Lab Results</Text>
                  </Pressable>
                </View>
              )}
            </View>
          )}

          {/* Insurance Pre-Authorization */}
          {analysis.insurance_preauthorization && (
            <View style={styles.insurancePreAuthSection}>
              <View style={styles.insurancePreAuthHeader}>
                <Text style={styles.insurancePreAuthTitle}>📋 Insurance Pre-Authorization</Text>
                <Text style={styles.insurancePreAuthSubtitle}>Documentation Ready for Submission</Text>
              </View>

              {/* Quick Summary */}
              {analysis.insurance_preauthorization.form_data && (
                <View style={styles.preAuthSummaryCard}>
                  <Text style={styles.cardTitle}>📝 Authorization Summary</Text>

                  <View style={styles.summaryRow}>
                    <Text style={styles.summaryLabel}>Diagnosis:</Text>
                    <Text style={styles.summaryValue}>
                      {analysis.insurance_preauthorization.form_data.diagnosis?.primary_diagnosis || analysis.predicted_class}
                    </Text>
                  </View>

                  <View style={styles.summaryRow}>
                    <Text style={styles.summaryLabel}>ICD-10 Code:</Text>
                    <Text style={styles.summaryValue}>
                      {analysis.insurance_preauthorization.form_data.diagnosis?.icd10_code || 'N/A'}
                    </Text>
                  </View>

                  <View style={styles.summaryRow}>
                    <Text style={styles.summaryLabel}>Urgency:</Text>
                    <Text style={[
                      styles.summaryValue,
                      analysis.insurance_preauthorization.form_data.urgency?.includes('Urgent') && styles.urgentText
                    ]}>
                      {analysis.insurance_preauthorization.form_data.urgency || 'Routine'}
                    </Text>
                  </View>

                  <View style={styles.summaryRow}>
                    <Text style={styles.summaryLabel}>Confidence:</Text>
                    <Text style={styles.summaryValue}>
                      {analysis.insurance_preauthorization.form_data.diagnosis?.confidence_level || 'N/A'}
                    </Text>
                  </View>
                </View>
              )}

              {/* Approval Likelihood Prediction */}
              {analysis.insurance_preauthorization.approval_likelihood && (
                <View style={styles.approvalLikelihoodCard}>
                  <Text style={styles.cardTitle}>🎯 Approval Likelihood Prediction</Text>

                  <View style={[
                    styles.approvalBadge,
                    { backgroundColor: analysis.insurance_preauthorization.approval_likelihood.category_color }
                  ]}>
                    <Text style={styles.approvalProbability}>
                      {analysis.insurance_preauthorization.approval_likelihood.probability}%
                    </Text>
                    <Text style={styles.approvalCategory}>
                      {analysis.insurance_preauthorization.approval_likelihood.category}
                    </Text>
                  </View>

                  <Text style={styles.approvalRecommendation}>
                    {analysis.insurance_preauthorization.approval_likelihood.recommendation}
                  </Text>

                  <View style={styles.approvalFactors}>
                    <Text style={styles.factorsTitle}>Factors Contributing to Score:</Text>
                    {analysis.insurance_preauthorization.approval_likelihood.factors.map((factor, index) => (
                      <View key={index} style={styles.factorItem}>
                        <View style={styles.factorHeader}>
                          <Text style={styles.factorName}>{factor.factor}</Text>
                          <Text style={styles.factorScore}>{factor.score}/{factor.max}</Text>
                        </View>
                        <Text style={styles.factorDescription}>{factor.description}</Text>
                        <View style={styles.factorBar}>
                          <View style={[
                            styles.factorBarFill,
                            { width: `${(factor.score / factor.max) * 100}%` },
                            factor.impact === 'CRITICAL' && { backgroundColor: '#dc2626' },
                            factor.impact === 'HIGH' && { backgroundColor: '#10b981' },
                            factor.impact === 'MEDIUM' && { backgroundColor: '#3b82f6' },
                            factor.impact === 'LOW' && { backgroundColor: '#6b7280' }
                          ]} />
                        </View>
                      </View>
                    ))}
                  </View>
                </View>
              )}

              {/* Submission Status Tracking */}
              {analysis.insurance_preauthorization.submission_status && (
                <View style={styles.statusTrackingCard}>
                  <Text style={styles.cardTitle}>📊 Submission Status</Text>

                  <View style={[
                    styles.currentStatus,
                    analysis.insurance_preauthorization.submission_status.current_status === 'APPROVED' && { backgroundColor: '#d1fae5', borderColor: '#10b981' },
                    analysis.insurance_preauthorization.submission_status.current_status === 'DENIED' && { backgroundColor: '#fee2e2', borderColor: '#ef4444' },
                    analysis.insurance_preauthorization.submission_status.current_status === 'UNDER_REVIEW' && { backgroundColor: '#dbeafe', borderColor: '#3b82f6' },
                    analysis.insurance_preauthorization.submission_status.current_status === 'SUBMITTED' && { backgroundColor: '#fef3c7', borderColor: '#f59e0b' }
                  ]}>
                    <Text style={styles.statusText}>
                      {analysis.insurance_preauthorization.submission_status.status_description}
                    </Text>
                  </View>

                  {analysis.insurance_preauthorization.submission_status.submitted_date && (
                    <Text style={styles.statusDate}>
                      Submitted: {new Date(analysis.insurance_preauthorization.submission_status.submitted_date).toLocaleDateString()}
                    </Text>
                  )}

                  {analysis.insurance_preauthorization.submission_status.decision_date && (
                    <Text style={styles.statusDate}>
                      Decision: {new Date(analysis.insurance_preauthorization.submission_status.decision_date).toLocaleDateString()}
                    </Text>
                  )}

                  <TouchableOpacity
                    style={styles.updateStatusButton}
                    onPress={() => {
                      Alert.alert(
                        'Update Status',
                        'Status tracking allows you to monitor your pre-authorization through the approval process.',
                        [
                          { text: 'Mark as Submitted', onPress: () => updatePreAuthStatus('SUBMITTED') },
                          { text: 'Mark as Under Review', onPress: () => updatePreAuthStatus('UNDER_REVIEW') },
                          { text: 'Mark as Approved', onPress: () => updatePreAuthStatus('APPROVED') },
                          { text: 'Mark as Denied', onPress: () => updatePreAuthStatus('DENIED') },
                          { text: 'Cancel', style: 'cancel' }
                        ]
                      );
                    }}
                  >
                    <Text style={styles.updateStatusButtonText}>Update Status</Text>
                  </TouchableOpacity>
                </View>
              )}

              {/* Requested Procedures */}
              {analysis.insurance_preauthorization.form_data?.procedures_requested &&
               analysis.insurance_preauthorization.form_data.procedures_requested.length > 0 && (
                <View style={styles.proceduresCard}>
                  <Text style={styles.cardTitle}>🏥 Requested Procedures</Text>

                  {analysis.insurance_preauthorization.form_data.procedures_requested.map((proc, index) => (
                    <View key={index} style={styles.procedureItem}>
                      <View style={styles.procedureHeader}>
                        <Text style={styles.procedureCode}>CPT: {proc.code}</Text>
                        <Text style={styles.procedureDescription}>{proc.description}</Text>
                      </View>
                      <Text style={styles.procedureRationale}>
                        💡 {proc.rationale}
                      </Text>
                    </View>
                  ))}
                </View>
              )}

              {/* Medical Necessity Letter Preview */}
              {analysis.insurance_preauthorization.medical_necessity_letter && (
                <View style={styles.letterPreviewCard}>
                  <Text style={styles.cardTitle}>📄 Medical Necessity Letter</Text>
                  <Text style={styles.letterPreviewText} numberOfLines={10}>
                    {analysis.insurance_preauthorization.medical_necessity_letter.substring(0, 500)}...
                  </Text>
                  <TouchableOpacity
                    style={styles.viewFullLetterButton}
                    onPress={() => {
                      // TODO: Open full letter in modal or new screen
                      Alert.alert('Medical Necessity Letter', analysis.insurance_preauthorization.medical_necessity_letter, [
                        { text: 'Close', style: 'cancel' }
                      ]);
                    }}
                  >
                    <Text style={styles.viewFullLetterButtonText}>📖 View Full Letter</Text>
                  </TouchableOpacity>
                </View>
              )}

              {/* Clinical Summary */}
              {analysis.insurance_preauthorization.clinical_summary && (
                <View style={styles.clinicalSummaryCard}>
                  <Text style={styles.cardTitle}>📊 Clinical Summary</Text>
                  <Text style={styles.clinicalSummaryText} numberOfLines={8}>
                    {analysis.insurance_preauthorization.clinical_summary.substring(0, 400)}...
                  </Text>
                  <TouchableOpacity
                    style={styles.viewFullSummaryButton}
                    onPress={() => {
                      Alert.alert('Clinical Summary', analysis.insurance_preauthorization.clinical_summary, [
                        { text: 'Close', style: 'cancel' }
                      ]);
                    }}
                  >
                    <Text style={styles.viewFullSummaryButtonText}>📄 View Full Summary</Text>
                  </TouchableOpacity>
                </View>
              )}

              {/* Supporting Evidence */}
              {analysis.insurance_preauthorization.supporting_evidence && (
                <View style={styles.supportingEvidenceCard}>
                  <Text style={styles.cardTitle}>📚 Supporting Evidence</Text>

                  {analysis.insurance_preauthorization.supporting_evidence.clinical_guidelines && (
                    <View style={styles.guidelinesBox}>
                      <Text style={styles.evidenceSubtitle}>Clinical Guidelines:</Text>
                      {analysis.insurance_preauthorization.supporting_evidence.clinical_guidelines.map((guideline, index) => (
                        <Text key={index} style={styles.guidelineText}>• {guideline}</Text>
                      ))}
                    </View>
                  )}

                  {analysis.insurance_preauthorization.supporting_evidence.diagnostic_accuracy && (
                    <View style={styles.accuracyBox}>
                      <Text style={styles.evidenceSubtitle}>Diagnostic Method:</Text>
                      <Text style={styles.evidenceText}>
                        {analysis.insurance_preauthorization.supporting_evidence.diagnostic_accuracy.method}
                      </Text>
                      <Text style={styles.evidenceText}>
                        {analysis.insurance_preauthorization.supporting_evidence.diagnostic_accuracy.validation}
                      </Text>
                    </View>
                  )}
                </View>
              )}

              {/* Auto-Fill Forms */}
              {analysis.insurance_preauthorization.autofill_forms && (
                <View style={styles.autoFillFormsCard}>
                  <Text style={styles.cardTitle}>📝 Auto-Fill Insurance Forms</Text>

                  <Text style={styles.autoFillDescription}>
                    Pre-filled forms ready for common insurance providers. Fields marked with [BRACKETS] need your information.
                  </Text>

                  <TouchableOpacity
                    style={styles.formTypeButton}
                    onPress={() => {
                      const forms = analysis.insurance_preauthorization.autofill_forms;
                      openFormModal('cms1500', forms.cms_1500);
                    }}
                  >
                    <Text style={styles.formTypeButtonText}>📄 CMS-1500 Form (Professional)</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={styles.formTypeButton}
                    onPress={() => {
                      const forms = analysis.insurance_preauthorization.autofill_forms;
                      openFormModal('ub04', forms.ub_04);
                    }}
                  >
                    <Text style={styles.formTypeButtonText}>📄 UB-04 Form (Facility)</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={styles.formTypeButton}
                    onPress={() => {
                      const forms = analysis.insurance_preauthorization.autofill_forms;
                      openFormModal('generic', forms.insurance_specific);
                    }}
                  >
                    <Text style={styles.formTypeButtonText}>📄 Generic Pre-Auth Request</Text>
                  </TouchableOpacity>

                  <View style={styles.formInstructions}>
                    <Text style={styles.formInstructionsTitle}>📋 Instructions:</Text>
                    <Text style={styles.formInstructionsText}>
                      • Fill in fields marked with [BRACKETS]{'\n'}
                      • Physician signature required{'\n'}
                      • Attach supporting documentation{'\n'}
                      • Submit per insurance requirements
                    </Text>
                  </View>
                </View>
              )}

              {/* Export/Share Actions */}
              <View style={styles.preAuthActionsCard}>
                <Text style={styles.cardTitle}>📤 Export & Share</Text>

                <TouchableOpacity
                  style={styles.exportPDFButton}
                  onPress={exportPreAuthPDF}
                >
                  <Text style={styles.exportPDFButtonText}>📄 Export as PDF</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.shareButton}
                  onPress={sharePreAuthDocumentation}
                >
                  <Text style={styles.shareButtonText}>📧 Share Documentation</Text>
                </TouchableOpacity>
              </View>

              <View style={styles.insuranceDisclaimer}>
                <Text style={styles.insuranceDisclaimerText}>
                  ⚕️ This pre-authorization documentation is based on evidence-based clinical guidelines
                  and AI-assisted diagnostic analysis. All information should be reviewed and signed by
                  a licensed physician before submission to insurance providers. This documentation is
                  intended to streamline the authorization process but does not guarantee approval.
                </Text>
              </View>
            </View>
          )}

          {/* Probabilities Section */}
          {(analysis.binary_probabilities || analysis.lesion_probabilities) && (
            <View style={styles.probabilitiesSection}>
              <Text style={styles.sectionTitle}>📊 Detailed Probabilities</Text>

              {/* Binary Classification Probabilities */}
              {analysis.binary_probabilities && Object.keys(analysis.binary_probabilities).length > 0 && (
                <View style={styles.probabilitySubsection}>
                  <Text style={styles.probabilitySubtitle}>🔍 Lesion Detection</Text>
                  {/* Display in order: Lesion first (concerning outcome), then Non-Lesion */}
                  {['lesion', 'non_lesion'].map((key) => {
                    const probability = analysis.binary_probabilities[key];
                    if (probability === undefined) return null;
                    const displayName = key === 'lesion' ? 'Lesion' : 'Non-Lesion';
                    const barColor = key === 'lesion' ? '#dc3545' : '#28a745';
                    return (
                      <View key={key} style={styles.probabilityItem}>
                        <Text style={styles.probabilityLabel}>{displayName}:</Text>
                        <View style={styles.probabilityBarContainer}>
                          <View style={[styles.probabilityBar, { width: `${probability * 100}%`, backgroundColor: barColor }]} />
                          <Text style={styles.probabilityValue}>{Math.round(probability * 100)}%</Text>
                        </View>
                      </View>
                    );
                  })}
                </View>
              )}

              {/* Lesion Classification Probabilities (7 cancer types) */}
              {analysis.lesion_probabilities && Object.keys(analysis.lesion_probabilities).length > 0 && formatted.isLesion && (
                <View style={styles.probabilitySubsection}>
                  <Text style={styles.probabilitySubtitle}>🔬 Cancer Type Classification</Text>
                  {Object.entries(analysis.lesion_probabilities).map(([className, probability]) => (
                    <View key={className} style={styles.probabilityItem}>
                      <Text style={styles.probabilityLabel}>{className}:</Text>
                      <View style={styles.probabilityBarContainer}>
                        <View style={[styles.probabilityBar, { width: `${probability * 100}%` }]} />
                        <Text style={styles.probabilityValue}>{Math.round(probability * 100)}%</Text>
                      </View>
                    </View>
                  ))}
                </View>
              )}

              {/* Inflammatory Condition Classification */}
              {analysis.inflammatory_probabilities && Object.keys(analysis.inflammatory_probabilities).length > 0 && (
                <View style={styles.probabilitySubsection}>
                  <Text style={styles.probabilitySubtitle}>🔥 Inflammatory Conditions</Text>
                  {Object.entries(analysis.inflammatory_probabilities)
                    .filter(([className, probability]) => !className.startsWith('_') && typeof probability === 'number')
                    .map(([className, probability]) => (
                    <View key={className} style={styles.probabilityItem}>
                      <Text style={styles.probabilityLabel}>{className}:</Text>
                      <View style={styles.probabilityBarContainer}>
                        <View style={[styles.probabilityBar, { width: `${probability * 100}%`, backgroundColor: '#ff6b35' }]} />
                        <Text style={styles.probabilityValue}>{Math.round(probability * 100)}%</Text>
                      </View>
                    </View>
                  ))}
                </View>
              )}
            </View>
          )}

          {/* Recommendation Section */}
          {formatted.recommendation && (
            <View style={styles.recommendationSection}>
              <Text style={styles.sectionTitle}>💡 Recommendation</Text>
              <Text style={styles.recommendationText}>{formatted.recommendation}</Text>
            </View>
          )}

          {/* Technical Details Section */}
          <View style={styles.technicalSection}>
            <Text style={styles.sectionTitle}>⚙️ Technical Details</Text>

            <View style={styles.technicalItem}>
              <Text style={styles.technicalLabel}>Processing Time:</Text>
              <Text style={styles.technicalValue}>{formatted.processingTime?.toFixed(2) || 0}s</Text>
            </View>

            <View style={styles.technicalItem}>
              <Text style={styles.technicalLabel}>Model Version:</Text>
              <Text style={styles.technicalValue}>{formatted.modelVersion || 'N/A'}</Text>
            </View>

            <View style={styles.technicalItem}>
              <Text style={styles.technicalLabel}>Analysis ID:</Text>
              <Text style={styles.technicalValue}>#{analysis.id}</Text>
            </View>

            {analysis.image_filename && (
              <View style={styles.technicalItem}>
                <Text style={styles.technicalLabel}>Image Filename:</Text>
                <Text style={styles.technicalValue}>{analysis.image_filename}</Text>
              </View>
            )}
          </View>
        </View>

        {/* Limitations Disclaimer */}
        <View style={styles.disclaimerSection}>
          <Text style={styles.disclaimerTitle}>⚠️ Important Limitations & Disclaimer</Text>

          <View style={styles.disclaimerContent}>
            <Text style={styles.disclaimerHeading}>What This AI Can Do:</Text>
            <Text style={styles.disclaimerText}>• Provide preliminary analysis of skin lesion images</Text>
            <Text style={styles.disclaimerText}>• Identify potential skin conditions based on visual features</Text>
            <Text style={styles.disclaimerText}>• Suggest risk levels and recommend appropriate follow-up</Text>
            <Text style={styles.disclaimerText}>• Serve as an educational and screening tool</Text>
          </View>

          <View style={styles.disclaimerContent}>
            <Text style={styles.disclaimerHeading}>What This AI Cannot Do:</Text>
            <Text style={styles.disclaimerText}>• Replace professional medical diagnosis or treatment</Text>
            <Text style={styles.disclaimerText}>• Detect all skin conditions or cancers with 100% accuracy</Text>
            <Text style={styles.disclaimerText}>• Account for your personal medical history or risk factors</Text>
            <Text style={styles.disclaimerText}>• Perform physical examination or laboratory testing</Text>
            <Text style={styles.disclaimerText}>• Prescribe medications or treatment plans</Text>
          </View>

          <View style={styles.disclaimerContent}>
            <Text style={styles.disclaimerHeading}>When to Seek In-Person Care:</Text>
            <Text style={styles.disclaimerUrgent}>🚨 SEEK IMMEDIATE CARE if you have:</Text>
            <Text style={styles.disclaimerText}>• Rapidly changing or growing lesions</Text>
            <Text style={styles.disclaimerText}>• Bleeding, ulceration, or non-healing wounds</Text>
            <Text style={styles.disclaimerText}>• Signs of infection (warmth, pus, severe pain)</Text>
            <Text style={styles.disclaimerText}>• Severe symptoms affecting daily life</Text>
            <Text style={styles.disclaimerText}>• Any HIGH RISK classification from this analysis</Text>
          </View>

          <View style={styles.disclaimerContent}>
            <Text style={styles.disclaimerHeading}>📋 Always Consult a Healthcare Provider:</Text>
            <Text style={styles.disclaimerText}>• For definitive diagnosis and treatment recommendations</Text>
            <Text style={styles.disclaimerText}>• Before making any medical decisions based on this analysis</Text>
            <Text style={styles.disclaimerText}>• If you have concerns about any skin changes</Text>
            <Text style={styles.disclaimerText}>• For regular skin cancer screenings, especially if you have risk factors</Text>
          </View>

          <View style={styles.disclaimerWarning}>
            <Text style={styles.disclaimerWarningText}>
              This tool is designed to assist, not replace, professional medical judgment.
              AI predictions may contain errors. No medical decisions should be made solely based on this analysis.
            </Text>
          </View>
        </View>

        {/* Saved Measurements Section */}
        {measurements.length > 0 && (
          <View style={styles.savedMeasurementsSection}>
            <Text style={styles.sectionTitle}>📏 Saved Measurements</Text>
            {measurements.map((measurement, index) => (
              <View key={measurement.id} style={styles.savedMeasurementItem}>
                <Text style={styles.savedMeasurementLabel}>{measurement.label}</Text>
                <Text style={styles.savedMeasurementValue}>
                  {measurement.distanceMm
                    ? `${measurement.distanceMm.toFixed(2)} mm`
                    : 'Not calibrated'}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Action Buttons */}
        <View style={styles.actionSection}>
          <Pressable style={styles.actionButton} onPress={() => router.push('/home')}>
            <Text style={styles.actionButtonText}>📷 New Analysis</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => router.push('/history')}>
            <Text style={styles.secondaryButtonText}>📊 View History</Text>
          </Pressable>
        </View>
      </ScrollView>

      {/* Biopsy Form Modal */}
      <Modal
        visible={showBiopsyForm}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowBiopsyForm(false)}
      >
        <View style={styles.biopsyModalOverlay}>
          <View style={styles.biopsyModalContent}>
            <ScrollView>
              <View style={styles.biopsyModalHeader}>
                <Text style={styles.biopsyModalTitle}>Add Biopsy Result</Text>
                <Pressable onPress={() => setShowBiopsyForm(false)}>
                  <Text style={styles.biopsyModalClose}>✕</Text>
                </Pressable>
              </View>

              <Text style={styles.biopsyFormLabel}>Biopsy Result *</Text>
              <TextInput
                style={styles.biopsyInput}
                placeholder="e.g., Melanoma, Basal Cell Carcinoma"
                value={biopsyData.biopsy_result}
                onChangeText={(text) => setBiopsyData({...biopsyData, biopsy_result: text})}
              />

              <Text style={styles.biopsyFormLabel}>Biopsy Date</Text>
              <TextInput
                style={styles.biopsyInput}
                placeholder="YYYY-MM-DD"
                value={biopsyData.biopsy_date}
                onChangeText={(text) => setBiopsyData({...biopsyData, biopsy_date: text})}
              />

              <Text style={styles.biopsyFormLabel}>Lab/Facility</Text>
              <TextInput
                style={styles.biopsyInput}
                placeholder="e.g., City Medical Center"
                value={biopsyData.biopsy_facility}
                onChangeText={(text) => setBiopsyData({...biopsyData, biopsy_facility: text})}
              />

              <Text style={styles.biopsyFormLabel}>Pathologist Name</Text>
              <TextInput
                style={styles.biopsyInput}
                placeholder="e.g., Dr. Smith"
                value={biopsyData.pathologist_name}
                onChangeText={(text) => setBiopsyData({...biopsyData, pathologist_name: text})}
              />

              <Text style={styles.biopsyFormLabel}>Additional Notes</Text>
              <TextInput
                style={[styles.biopsyInput, styles.biopsyTextArea]}
                placeholder="Any additional pathology notes..."
                value={biopsyData.biopsy_notes}
                onChangeText={(text) => setBiopsyData({...biopsyData, biopsy_notes: text})}
                multiline
                numberOfLines={4}
              />

              <View style={styles.biopsyFormButtons}>
                <Pressable
                  style={styles.biopsyCancelButton}
                  onPress={() => setShowBiopsyForm(false)}
                >
                  <Text style={styles.biopsyCancelButtonText}>Cancel</Text>
                </Pressable>

                <Pressable
                  style={styles.biopsySubmitButton}
                  onPress={handleSubmitBiopsy}
                >
                  <Text style={styles.biopsySubmitButtonText}>Submit Biopsy Result</Text>
                </Pressable>
              </View>
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* Symptom Tracker Modal */}
      <Modal
        visible={showSymptomForm}
        animationType="slide"
        presentationStyle="fullScreen"
        onRequestClose={() => setShowSymptomForm(false)}
      >
        <View style={styles.fullScreenModal}>
          <View style={styles.fullScreenModalHeader}>
            <Pressable onPress={() => setShowSymptomForm(false)}>
              <Text style={styles.fullScreenModalClose}>← {t('common.back')}</Text>
            </Pressable>
            <Text style={styles.fullScreenModalTitle}>Record Symptoms</Text>
            <View style={{ width: 60 }} />
          </View>

          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={{ flexGrow: 1 }}
            showsVerticalScrollIndicator={true}
          >
            <SymptomTracker
              onSymptomChange={handleSymptomChange}
              initialSymptoms={{
                symptom_duration: analysis?.symptom_duration,
                symptom_duration_value: analysis?.symptom_duration_value,
                symptom_duration_unit: analysis?.symptom_duration_unit,
                symptom_changes: analysis?.symptom_changes,
                symptom_itching: analysis?.symptom_itching,
                symptom_itching_severity: analysis?.symptom_itching_severity,
                symptom_pain: analysis?.symptom_pain,
                symptom_pain_severity: analysis?.symptom_pain_severity,
                symptom_bleeding: analysis?.symptom_bleeding,
                symptom_bleeding_frequency: analysis?.symptom_bleeding_frequency,
                symptom_notes: analysis?.symptom_notes
              }}
            />
          </ScrollView>

          <View style={styles.fullScreenModalFooter}>
            <Pressable
              style={styles.biopsyCancelButton}
              onPress={() => setShowSymptomForm(false)}
            >
              <Text style={styles.biopsyCancelButtonText}>Cancel</Text>
            </Pressable>

            <Pressable
              style={styles.biopsySubmitButton}
              onPress={handleSubmitSymptoms}
            >
              <Text style={styles.biopsySubmitButtonText}>Save Symptoms</Text>
            </Pressable>
          </View>
        </View>
      </Modal>

      {/* Medication List Modal */}
      <Modal
        visible={showMedicationForm}
        animationType="slide"
        presentationStyle="fullScreen"
        onRequestClose={() => setShowMedicationForm(false)}
      >
        <View style={styles.fullScreenModal}>
          <View style={styles.fullScreenModalHeader}>
            <Pressable onPress={() => setShowMedicationForm(false)}>
              <Text style={styles.fullScreenModalClose}>← {t('common.back')}</Text>
            </Pressable>
            <Text style={styles.fullScreenModalTitle}>Medication List</Text>
            <View style={{ width: 60 }} />
          </View>

          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={{ flexGrow: 1 }}
            showsVerticalScrollIndicator={true}
          >
            <MedicationList
              onMedicationChange={handleMedicationChange}
              initialMedications={(() => {
                try {
                  return analysis?.medications ? JSON.parse(analysis.medications) : undefined;
                } catch (e) {
                  return undefined;
                }
              })()}
            />
          </ScrollView>

          <View style={styles.fullScreenModalFooter}>
            <Pressable
              style={styles.biopsyCancelButton}
              onPress={() => setShowMedicationForm(false)}
            >
              <Text style={styles.biopsyCancelButtonText}>Cancel</Text>
            </Pressable>

            <Pressable
              style={styles.biopsySubmitButton}
              onPress={handleSubmitMedications}
            >
              <Text style={styles.biopsySubmitButtonText}>Save Medications</Text>
            </Pressable>
          </View>
        </View>
      </Modal>

      {/* Medical History Modal */}
      <Modal
        visible={showMedicalHistoryForm}
        animationType="slide"
        presentationStyle="fullScreen"
        onRequestClose={() => setShowMedicalHistoryForm(false)}
      >
        <View style={styles.fullScreenModal}>
          <View style={[styles.fullScreenModalHeader, { backgroundColor: '#dc2626' }]}>
            <Pressable onPress={() => setShowMedicalHistoryForm(false)}>
              <Text style={styles.fullScreenModalClose}>← {t('common.back')}</Text>
            </Pressable>
            <Text style={styles.fullScreenModalTitle}>Medical History</Text>
            <View style={{ width: 60 }} />
          </View>

          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={{ flexGrow: 1 }}
            showsVerticalScrollIndicator={true}
          >
            <MedicalHistory
              onMedicalHistoryChange={handleMedicalHistoryChange}
              initialHistory={{
                family_history_skin_cancer: analysis?.family_history_skin_cancer,
                family_history_details: analysis?.family_history_details,
                previous_skin_cancers: analysis?.previous_skin_cancers,
                previous_skin_cancers_details: analysis?.previous_skin_cancers_details,
                immunosuppression: analysis?.immunosuppression,
                immunosuppression_details: analysis?.immunosuppression_details,
                sun_exposure_level: analysis?.sun_exposure_level,
                sun_exposure_details: analysis?.sun_exposure_details,
                history_of_sunburns: analysis?.history_of_sunburns,
                sunburn_details: analysis?.sunburn_details,
                tanning_bed_use: analysis?.tanning_bed_use,
                tanning_bed_frequency: analysis?.tanning_bed_frequency,
                other_risk_factors: analysis?.other_risk_factors
              }}
            />
          </ScrollView>

          <View style={styles.fullScreenModalFooter}>
            <Pressable
              style={styles.biopsyCancelButton}
              onPress={() => setShowMedicalHistoryForm(false)}
            >
              <Text style={styles.biopsyCancelButtonText}>Cancel</Text>
            </Pressable>

            <Pressable
              style={styles.biopsySubmitButton}
              onPress={handleSubmitMedicalHistory}
            >
              <Text style={styles.biopsySubmitButtonText}>Save Medical History</Text>
            </Pressable>
          </View>
        </View>
      </Modal>

      {/* Teledermatology Modal */}
      <Modal
        visible={showTeledermatologyForm}
        animationType="slide"
        presentationStyle="fullScreen"
        onRequestClose={() => setShowTeledermatologyForm(false)}
      >
        <View style={styles.fullScreenModal}>
          <View style={[styles.fullScreenModalHeader, { backgroundColor: '#0284c7' }]}>
            <Pressable onPress={() => setShowTeledermatologyForm(false)}>
              <Text style={styles.fullScreenModalClose}>← {t('common.back')}</Text>
            </Pressable>
            <Text style={styles.fullScreenModalTitle}>Teledermatology</Text>
            <View style={{ width: 60 }} />
          </View>

          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={{ flexGrow: 1 }}
            showsVerticalScrollIndicator={true}
          >
            <TeledermatologyShare
              analysisId={parseInt(id as string)}
              onShareComplete={handleTeledermatologyShare}
              existingShare={analysis?.shared_with_dermatologist ? {
                dermatologist_name: analysis.dermatologist_name,
                dermatologist_email: analysis.dermatologist_email,
                share_date: analysis.share_date,
                share_message: analysis.share_message,
                dermatologist_reviewed: analysis.dermatologist_reviewed,
                dermatologist_notes: analysis.dermatologist_notes,
                dermatologist_recommendation: analysis.dermatologist_recommendation
              } : undefined}
            />
          </ScrollView>
        </View>
      </Modal>

      {/* Measurement Tool Modal */}
      <Modal
        visible={showMeasurementTool}
        animationType="slide"
        presentationStyle="fullScreen"
        onRequestClose={() => setShowMeasurementTool(false)}
      >
        <View style={styles.measurementModalContainer}>
          <View style={styles.measurementModalHeader}>
            <Pressable
              style={styles.closeMeasurementButton}
              onPress={() => setShowMeasurementTool(false)}
            >
              <Text style={styles.closeMeasurementButtonText}>✕ Close</Text>
            </Pressable>
            <Text style={styles.measurementModalTitle}>Lesion Measurement Tool</Text>
          </View>

          {analysis && analysis.image_url && (
            <MeasurementTool
              imageUri={`${API_BASE_URL}${analysis.image_url}`}
              calibration={analysis.calibration || {
                calibration_found: false,
                calibration_type: null,
                pixels_per_mm: null,
                confidence: 0,
                detected_objects: []
              }}
              onMeasurementsComplete={handleMeasurementsComplete}
            />
          )}
        </View>
      </Modal>

      {/* Doctor Search Modal */}
      <Modal
        visible={showDoctorSearchModal}
        animationType="slide"
        transparent={false}
        onRequestClose={() => setShowDoctorSearchModal(false)}
      >
        <View style={styles.doctorModalContainer}>
          <LinearGradient
            colors={['#3b82f6', '#1e40af']}
            style={styles.doctorModalHeader}
          >
            <Pressable
              onPress={() => setShowDoctorSearchModal(false)}
              style={styles.doctorModalBackButton}
            >
              <Text style={styles.doctorModalBackText}>← {t('common.back')}</Text>
            </Pressable>
            <Text style={styles.doctorModalTitle}>
              {selectedSpecialist ? `Find ${selectedSpecialist.name}` : 'Find Specialist'}
            </Text>
            {selectedSpecialist && (
              <Text style={styles.doctorModalSubtitle}>
                {selectedSpecialist.reason}
              </Text>
            )}
          </LinearGradient>

          <ScrollView style={styles.doctorModalContent}>
            {isLoadingDoctors ? (
              <View style={styles.doctorLoadingContainer}>
                <ActivityIndicator size="large" color="#3b82f6" />
                <Text style={styles.doctorLoadingText}>Searching for nearby doctors...</Text>
                <Text style={styles.doctorLoadingSubtext}>
                  Using your location to find the best options
                </Text>
              </View>
            ) : nearbyDoctors.length > 0 ? (
              <>
                <View style={styles.doctorResultsHeader}>
                  <Text style={styles.doctorResultsTitle}>
                    Found {nearbyDoctors.length} {selectedSpecialist?.name}s near you
                  </Text>
                  <Text style={styles.doctorResultsSubtitle}>
                    Tap on a doctor for more options
                  </Text>
                </View>

                {nearbyDoctors.map((doctor, index) => (
                  <View key={index} style={styles.doctorCard}>
                    <View style={styles.doctorCardHeader}>
                      <Text style={styles.doctorName}>{doctor.name}</Text>
                      {doctor.rating && doctor.rating !== 'N/A' && (
                        <View style={styles.doctorRating}>
                          <Text style={styles.doctorRatingText}>⭐ {doctor.rating}</Text>
                          {doctor.userRatingsTotal > 0 && (
                            <Text style={styles.doctorRatingCount}>({doctor.userRatingsTotal})</Text>
                          )}
                        </View>
                      )}
                    </View>

                    <View style={styles.doctorInfo}>
                      <Text style={styles.doctorInfoIcon}>📍</Text>
                      <Text style={styles.doctorAddress}>{doctor.address}</Text>
                    </View>

                    {doctor.distance && (
                      <View style={styles.doctorInfo}>
                        <Text style={styles.doctorInfoIcon}>🚗</Text>
                        <Text style={styles.doctorDistance}>{doctor.distance} miles away</Text>
                      </View>
                    )}

                    {doctor.isOpen !== null && (
                      <View style={styles.doctorInfo}>
                        <Text style={styles.doctorInfoIcon}>🕐</Text>
                        <Text style={[
                          styles.doctorStatus,
                          doctor.isOpen ? styles.doctorStatusOpen : styles.doctorStatusClosed
                        ]}>
                          {doctor.isOpen ? 'Open now' : 'Closed'}
                        </Text>
                      </View>
                    )}

                    {doctor.phone && (
                      <View style={styles.doctorInfo}>
                        <Text style={styles.doctorInfoIcon}>📞</Text>
                        <Text style={styles.doctorPhone}>{doctor.phone}</Text>
                      </View>
                    )}

                    <View style={styles.doctorActions}>
                      {doctor.phone && (
                        <Pressable
                          style={styles.doctorActionButton}
                          onPress={() => handleCallDoctor(doctor.phone)}
                        >
                          <Text style={styles.doctorActionButtonText}>📞 Call</Text>
                        </Pressable>
                      )}

                      <Pressable
                        style={[styles.doctorActionButton, styles.doctorActionButtonPrimary]}
                        onPress={() => handleOpenInMaps(doctor)}
                      >
                        <Text style={[styles.doctorActionButtonText, styles.doctorActionButtonTextPrimary]}>
                          🗺️ Directions
                        </Text>
                      </Pressable>
                    </View>
                  </View>
                ))}

                <View style={styles.doctorDisclaimer}>
                  <Text style={styles.doctorDisclaimerText}>
                    ℹ️ These results are provided by Google Places. Please verify credentials and
                    availability before scheduling an appointment. Always consult with your
                    primary care physician for referrals.
                  </Text>
                </View>
              </>
            ) : (
              <View style={styles.doctorEmptyContainer}>
                <Text style={styles.doctorEmptyIcon}>🔍</Text>
                <Text style={styles.doctorEmptyTitle}>No doctors found nearby</Text>
                <Text style={styles.doctorEmptyText}>
                  We couldn't find any {selectedSpecialist?.name}s in your area.
                  Try widening your search or consult your primary care physician for a referral.
                </Text>
              </View>
            )}
          </ScrollView>
        </View>
      </Modal>

      {/* Form Viewer Modal */}
      <Modal
        visible={showFormModal}
        animationType="slide"
        transparent={false}
        onRequestClose={() => setShowFormModal(false)}
      >
        <View style={styles.formModalContainer}>
          <View style={styles.formModalHeader}>
            <Text style={styles.formModalTitle}>
              {selectedFormType === 'cms1500' && '📄 CMS-1500 Form (Professional Claim)'}
              {selectedFormType === 'ub04' && '📄 UB-04 Form (Facility Billing)'}
              {selectedFormType === 'generic' && '📄 Generic Pre-Authorization Request'}
            </Text>
            <TouchableOpacity
              style={styles.formModalCloseButton}
              onPress={() => setShowFormModal(false)}
            >
              <Text style={styles.formModalCloseButtonText}>✕</Text>
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.formModalContent}>
            <View style={styles.formModalNotice}>
              <Text style={styles.formModalNoticeText}>
                💡 Fields marked with [BRACKETS] require your personal information.
                Review all fields before submission.
              </Text>
            </View>

            {selectedFormType === 'cms1500' && selectedFormData && (
              <View>
                <Text style={styles.formTypeDescription}>
                  CMS-1500 is the standard claim form for professional services. This form is used by physicians,
                  therapists, and other healthcare professionals to bill insurance for services.
                </Text>

                <View style={styles.formDataSummary}>
                  <Text style={styles.formDataSummaryTitle}>📊 CMS-1500 Form Structure:</Text>
                  <Text style={styles.formDataSummaryText}>
                    ✓ Patient Information Section (7 fields){'\n'}
                    ✓ Physician Information Section (7 fields){'\n'}
                    ✓ Service Lines with CPT Codes{'\n'}
                    ✓ Diagnosis Codes (ICD-10){'\n'}
                    {'\n'}
                    <Text style={{fontWeight: '700', color: '#dc2626'}}>
                      Fields in RED are placeholders - add your information
                    </Text>
                  </Text>
                </View>

                {selectedFormData.form_type && (
                  <View style={styles.formMetadata}>
                    <Text style={styles.formFieldValue}>Form Type: {selectedFormData.form_type}</Text>
                    {selectedFormData.form_version && (
                      <Text style={styles.formFieldValue}>Version: {selectedFormData.form_version}</Text>
                    )}
                  </View>
                )}

                {Object.entries(selectedFormData).map(([key, value]) =>
                  renderFormField(key, value)
                )}
              </View>
            )}

            {selectedFormType === 'ub04' && selectedFormData && (
              <View>
                <Text style={styles.formTypeDescription}>
                  UB-04 is the standard claim form for institutional providers (hospitals, skilled nursing facilities,
                  etc.). This form is used for facility billing and includes revenue codes.
                </Text>

                <View style={styles.formDataSummary}>
                  <Text style={styles.formDataSummaryTitle}>📊 UB-04 Form Structure:</Text>
                  <Text style={styles.formDataSummaryText}>
                    ✓ Facility Information (8 fields){'\n'}
                    ✓ Patient Demographics{'\n'}
                    ✓ Diagnosis Codes (Principal + Other){'\n'}
                    ✓ Revenue Codes with HCPCS{'\n'}
                    {'\n'}
                    <Text style={{fontWeight: '700', color: '#dc2626'}}>
                      Fields in RED are placeholders - add your information
                    </Text>
                  </Text>
                </View>

                {selectedFormData.form_type && (
                  <View style={styles.formMetadata}>
                    <Text style={styles.formFieldValue}>Form Type: {selectedFormData.form_type}</Text>
                  </View>
                )}

                {Object.entries(selectedFormData).map(([key, value]) =>
                  renderFormField(key, value)
                )}
              </View>
            )}

            {selectedFormType === 'generic' && selectedFormData && (
              <View>
                <Text style={styles.formTypeDescription}>
                  Generic Pre-Authorization Request form that can be adapted for various insurance providers.
                  This includes all essential fields required for prior authorization.
                </Text>

                <View style={styles.formDataSummary}>
                  <Text style={styles.formDataSummaryTitle}>📊 Generic Pre-Auth Structure:</Text>
                  <Text style={styles.formDataSummaryText}>
                    ✓ Member & Provider Information{'\n'}
                    ✓ Diagnosis & Procedure Codes{'\n'}
                    ✓ Clinical Rationale{'\n'}
                    ✓ Urgency Level{'\n'}
                    ✓ Supporting Documentation List{'\n'}
                    {'\n'}
                    <Text style={{fontWeight: '700', color: '#dc2626'}}>
                      Fields in RED are placeholders - add your information
                    </Text>
                  </Text>
                </View>

                {Object.entries(selectedFormData).map(([key, value]) =>
                  renderFormField(key, value)
                )}
              </View>
            )}

            <View style={styles.formModalFooter}>
              <Text style={styles.formModalFooterText}>
                📋 This form is pre-filled based on AI analysis. All clinical information should be reviewed and signed by a licensed physician before submission to insurance providers.
              </Text>
            </View>
          </ScrollView>

          <View style={styles.formModalActions}>
            <TouchableOpacity
              style={styles.formModalActionButton}
              onPress={async () => {
                // Close modal first for better UX
                setShowFormModal(false);
                // Export the complete pre-authorization PDF which includes all forms
                await exportPreAuthPDF();
              }}
            >
              <Text style={styles.formModalActionButtonText}>📄 Export as PDF</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.formModalActionButton, styles.formModalActionButtonSecondary]}
              onPress={() => setShowFormModal(false)}
            >
              <Text style={[styles.formModalActionButtonText, styles.formModalActionButtonTextSecondary]}>
                Close
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  backgroundContainer: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#4a5568',
    textAlign: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#e53e3e',
    marginBottom: 12,
    textAlign: 'center',
  },
  errorText: {
    fontSize: 16,
    color: '#4a5568',
    textAlign: 'center',
    marginBottom: 20,
    lineHeight: 22,
  },
  errorButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  retryButton: {
    backgroundColor: '#4299e1',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  backButton: {
    backgroundColor: '#6c757d',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  backButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.2)',
  },
  backHeaderButton: {
    backgroundColor: 'rgba(66, 153, 225, 0.9)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 16,
  },
  backHeaderButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  headerContent: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#4a5568',
    marginTop: 4,
  },
  scrollContainer: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  analysisCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  classificationCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  classificationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  classificationIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  classificationInfo: {
    flex: 1,
  },
  classificationLabel: {
    fontSize: 14,
    color: '#718096',
    marginBottom: 4,
  },
  classificationValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2d3748',
  },
  confidenceText: {
    fontSize: 14,
    color: '#4a5568',
    marginTop: 4,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  analysisInfo: {
    flex: 1,
  },
  analysisDate: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2d3748',
  },
  analysisTime: {
    fontSize: 14,
    color: '#718096',
    marginTop: 2,
  },
  analysisRelative: {
    fontSize: 12,
    color: '#a0aec0',
    marginTop: 2,
  },
  riskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  riskBadgeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  imageSection: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
  },
  imageContainer: {
    alignItems: 'center',
    backgroundColor: '#f7fafc',
    borderRadius: 12,
    padding: 12,
  },
  analysisImage: {
    width: screenWidth - 80,
    height: 250,
    borderRadius: 12,
    backgroundColor: '#e2e8f0',
  },
  resultsSection: {
    marginBottom: 24,
  },
  modelDisagreementBanner: {
    flexDirection: 'row',
    backgroundColor: '#fff3cd',
    borderWidth: 1,
    borderColor: '#ffc107',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
    alignItems: 'flex-start',
  },
  modelDisagreementIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  modelDisagreementContent: {
    flex: 1,
  },
  modelDisagreementTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#856404',
    marginBottom: 4,
  },
  modelDisagreementText: {
    fontSize: 13,
    color: '#856404',
    lineHeight: 18,
  },
  resultItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f7fafc',
  },
  resultLabel: {
    fontSize: 14,
    color: '#4a5568',
    fontWeight: '500',
    flex: 1,
  },
  resultValue: {
    fontSize: 14,
    color: '#2d3748',
    fontWeight: '600',
    flex: 1,
    textAlign: 'right',
  },
  diagnosisValue: {
    fontSize: 14,
    color: '#2c5282',
    fontWeight: '600',
    flex: 1,
    textAlign: 'right',
  },
  confidenceValue: {
    fontSize: 14,
    fontWeight: '600',
    flex: 1,
    textAlign: 'right',
  },
  probabilitiesSection: {
    marginBottom: 24,
  },
  probabilitySubsection: {
    marginBottom: 20,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  probabilitySubtitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c5282',
    marginBottom: 12,
  },
  probabilityItem: {
    marginBottom: 12,
  },
  probabilityLabel: {
    fontSize: 14,
    color: '#4a5568',
    fontWeight: '500',
    marginBottom: 4,
  },
  probabilityBarContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f7fafc',
    borderRadius: 8,
    height: 24,
    paddingHorizontal: 8,
  },
  probabilityBar: {
    backgroundColor: '#4299e1',
    height: 16,
    borderRadius: 8,
    position: 'absolute',
    left: 0,
    top: 4,
  },
  probabilityValue: {
    fontSize: 12,
    color: '#2d3748',
    fontWeight: '600',
    marginLeft: 'auto',
    zIndex: 1,
  },
  recommendationSection: {
    marginBottom: 24,
    backgroundColor: '#f0f8ff',
    padding: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#4299e1',
  },
  recommendationSectionText: {
    fontSize: 14,
    color: '#2c5282',
    lineHeight: 20,
    fontStyle: 'italic',
  },
  technicalSection: {
    backgroundColor: '#f8f9fa',
    padding: 16,
    borderRadius: 12,
  },
  technicalItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 6,
  },
  technicalLabel: {
    fontSize: 13,
    color: '#6c757d',
    fontWeight: '500',
  },
  technicalValue: {
    fontSize: 13,
    color: '#495057',
    fontWeight: '600',
  },
  actionSection: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#38a169',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  actionButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: '#4299e1',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  secondaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  differentialSection: {
    marginBottom: 24,
    backgroundColor: '#f8f9ff',
    padding: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#4299e1',
  },
  redFlagSection: {
    marginBottom: 24,
    backgroundColor: '#fff5f5',
    padding: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#f56565',
  },
  redFlagOverview: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#fed7d7',
  },
  redFlagRiskLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: '#2d3748',
    marginRight: 8,
  },
  redFlagRiskValue: {
    fontSize: 15,
    fontWeight: 'bold',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 6,
  },
  redFlagHigh: {
    backgroundColor: '#fc8181',
    color: '#742a2a',
  },
  redFlagModerate: {
    backgroundColor: '#f6ad55',
    color: '#7c2d12',
  },
  redFlagLowModerate: {
    backgroundColor: '#fbd38d',
    color: '#744210',
  },
  redFlagLow: {
    backgroundColor: '#68d391',
    color: '#22543d',
  },
  redFlagGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  redFlagItem: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: 'white',
    padding: 12,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#68d391',
  },
  redFlagItemActive: {
    borderColor: '#fc8181',
    backgroundColor: '#fff5f5',
  },
  redFlagTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2d3748',
    marginBottom: 4,
  },
  redFlagDescription: {
    fontSize: 12,
    color: '#4a5568',
    lineHeight: 16,
  },
  redFlagEvolution: {
    minWidth: '100%',
  },
  evolutionChanges: {
    marginTop: 8,
    paddingLeft: 8,
  },
  evolutionChange: {
    fontSize: 11,
    color: '#d97706',
    marginBottom: 3,
    lineHeight: 15,
  },
  evolutionRecommendation: {
    fontSize: 12,
    color: '#2563eb',
    fontWeight: '600',
    marginTop: 8,
    fontStyle: 'italic',
  },
  // Second Opinion Styles
  secondOpinionSection: {
    backgroundColor: '#fff3cd',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: '#ffc107',
  },
  secondOpinionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  priorityBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  priorityCritical: {
    backgroundColor: '#dc3545',
  },
  priorityHigh: {
    backgroundColor: '#fd7e14',
  },
  priorityMedium: {
    backgroundColor: '#ffc107',
  },
  priorityLow: {
    backgroundColor: '#20c997',
  },
  priorityText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
  },
  secondOpinionSummary: {
    fontSize: 14,
    color: '#856404',
    fontWeight: '600',
    marginBottom: 16,
  },
  secondOpinionSubtitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#333',
    marginBottom: 8,
    marginTop: 8,
  },
  secondOpinionReasons: {
    marginBottom: 12,
  },
  reasonItem: {
    flexDirection: 'row',
    marginBottom: 6,
    paddingLeft: 4,
  },
  reasonBullet: {
    fontSize: 14,
    color: '#dc3545',
    fontWeight: '700',
    marginRight: 8,
  },
  reasonText: {
    flex: 1,
    fontSize: 12,
    color: '#333',
    lineHeight: 18,
  },
  secondOpinionRecommendations: {
    backgroundColor: '#fff',
    padding: 12,
    borderRadius: 8,
    marginTop: 8,
  },
  recommendationItem: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  recommendationNumber: {
    fontSize: 12,
    fontWeight: '700',
    color: '#007bff',
    marginRight: 8,
    minWidth: 20,
  },
  recommendationText: {
    flex: 1,
    fontSize: 12,
    color: '#333',
    lineHeight: 18,
  },
  // Explainability Styles
  explainabilitySection: {
    backgroundColor: '#f8f9fa',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#dee2e6',
  },
  explainabilityDescription: {
    fontSize: 13,
    color: '#495057',
    marginBottom: 12,
    lineHeight: 20,
  },
  heatmapContainer: {
    backgroundColor: '#000',
    borderRadius: 8,
    overflow: 'hidden',
    marginBottom: 12,
  },
  heatmapImage: {
    width: '100%',
    height: 300,
  },
  explainabilityNote: {
    fontSize: 11,
    color: '#6c757d',
    fontStyle: 'italic',
    textAlign: 'center',
  },
  differentialSubtext: {
    fontSize: 13,
    color: '#6c757d',
    fontStyle: 'italic',
    marginBottom: 16,
  },
  differentialSubsection: {
    marginBottom: 20,
  },
  differentialCategory: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
  },
  diagnosisCard: {
    backgroundColor: 'white',
    padding: 14,
    borderRadius: 10,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  diagnosisHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  diagnosisRank: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#4299e1',
    marginRight: 8,
  },
  diagnosisCondition: {
    fontSize: 15,
    fontWeight: '600',
    color: '#2d3748',
    flex: 1,
  },
  diagnosisProbability: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#4299e1',
  },
  severityBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    alignSelf: 'flex-start',
    marginBottom: 8,
  },
  severityText: {
    color: 'white',
    fontSize: 11,
    fontWeight: 'bold',
  },
  diagnosisUrgency: {
    fontSize: 13,
    color: '#e67e22',
    fontWeight: '600',
    marginBottom: 6,
  },
  diagnosisDescription: {
    fontSize: 13,
    color: '#4a5568',
    marginBottom: 6,
    lineHeight: 18,
  },
  diagnosisFeatures: {
    fontSize: 12,
    color: '#718096',
    lineHeight: 17,
  },
  featuresLabel: {
    fontWeight: '600',
    color: '#4a5568',
  },
  treatmentSection: {
    backgroundColor: '#f0fdf4',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#bbf7d0',
  },
  treatmentSubsection: {
    marginBottom: 20,
  },
  treatmentSubtitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#065f46',
    marginBottom: 12,
  },
  treatmentUrgency: {
    backgroundColor: '#fef3c7',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  treatmentUrgencyText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#92400e',
  },
  treatmentWarning: {
    backgroundColor: '#fee2e2',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#dc2626',
  },
  treatmentWarningText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#991b1b',
    lineHeight: 18,
  },
  treatmentConfidenceNote: {
    backgroundColor: '#dbeafe',
    padding: 10,
    borderRadius: 6,
    marginBottom: 12,
  },
  treatmentConfidenceNoteText: {
    fontSize: 12,
    color: '#1e40af',
    fontStyle: 'italic',
  },
  treatmentGroup: {
    marginBottom: 16,
    backgroundColor: '#ffffff',
    padding: 12,
    borderRadius: 8,
  },
  treatmentGroupTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#047857',
    marginBottom: 10,
  },
  treatmentItem: {
    flexDirection: 'row',
    marginBottom: 8,
    paddingLeft: 4,
  },
  treatmentBullet: {
    fontSize: 14,
    color: '#059669',
    marginRight: 8,
    marginTop: 2,
    fontWeight: 'bold',
  },
  treatmentText: {
    flex: 1,
    fontSize: 13,
    color: '#1f2937',
    lineHeight: 19,
  },
  treatmentDisclaimer: {
    backgroundColor: '#fef9c3',
    padding: 12,
    borderRadius: 8,
    marginTop: 8,
    borderWidth: 1,
    borderColor: '#fde047',
  },
  treatmentDisclaimerText: {
    fontSize: 12,
    color: '#713f12',
    lineHeight: 17,
    fontStyle: 'italic',
  },
  // OTC Recommendations Styles
  otcSection: {
    backgroundColor: '#f0f9ff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#bae6fd',
  },
  otcCategory: {
    backgroundColor: '#ffffff',
    padding: 14,
    borderRadius: 10,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: '#e0f2fe',
  },
  otcCategoryTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0369a1',
    marginBottom: 12,
  },
  otcLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#0c4a6e',
    marginBottom: 6,
  },
  otcExamples: {
    marginBottom: 10,
  },
  otcItem: {
    flexDirection: 'row',
    marginBottom: 4,
    paddingLeft: 4,
  },
  otcBullet: {
    fontSize: 14,
    color: '#0284c7',
    marginRight: 8,
    fontWeight: 'bold',
  },
  otcText: {
    flex: 1,
    fontSize: 13,
    color: '#1e293b',
    lineHeight: 18,
  },
  otcUsage: {
    backgroundColor: '#f0fdf4',
    padding: 10,
    borderRadius: 6,
    marginBottom: 10,
  },
  otcUsageText: {
    fontSize: 13,
    color: '#166534',
    lineHeight: 18,
  },
  otcDuration: {
    backgroundColor: '#fef3c7',
    padding: 10,
    borderRadius: 6,
    marginBottom: 10,
  },
  otcDurationText: {
    fontSize: 13,
    color: '#92400e',
    lineHeight: 18,
  },
  otcWarnings: {
    backgroundColor: '#fef2f2',
    padding: 10,
    borderRadius: 6,
    marginBottom: 10,
  },
  otcWarningLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#dc2626',
    marginBottom: 6,
  },
  otcWarningText: {
    flex: 1,
    fontSize: 12,
    color: '#991b1b',
    lineHeight: 17,
  },
  otcContraindications: {
    backgroundColor: '#fce7f3',
    padding: 10,
    borderRadius: 6,
    marginBottom: 10,
  },
  otcContraindicationLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#be185d',
    marginBottom: 6,
  },
  otcContraindicationText: {
    flex: 1,
    fontSize: 12,
    color: '#9d174d',
    lineHeight: 17,
  },
  otcGeneralAdvice: {
    backgroundColor: '#ecfdf5',
    padding: 12,
    borderRadius: 8,
    marginTop: 8,
    marginBottom: 12,
  },
  otcAdviceTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#047857',
    marginBottom: 8,
  },
  otcSeekCare: {
    backgroundColor: '#fff7ed',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#f97316',
  },
  otcSeekCareTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#c2410c',
    marginBottom: 8,
  },
  otcSeekCareText: {
    flex: 1,
    fontSize: 13,
    color: '#9a3412',
    lineHeight: 18,
  },
  otcNotApplicable: {
    backgroundColor: '#fef2f2',
    padding: 14,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#ef4444',
  },
  otcNotApplicableTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#b91c1c',
    marginBottom: 10,
  },
  otcSeeDoctorReasons: {
    marginTop: 8,
  },
  otcSeeDoctorText: {
    flex: 1,
    fontSize: 13,
    color: '#7f1d1d',
    lineHeight: 18,
  },
  otcDisclaimer: {
    backgroundColor: '#fefce8',
    padding: 12,
    borderRadius: 8,
    marginTop: 12,
    borderWidth: 1,
    borderColor: '#fef08a',
  },
  otcDisclaimerText: {
    fontSize: 11,
    color: '#854d0e',
    lineHeight: 16,
    fontStyle: 'italic',
  },

  // Lab Results Insights Styles
  labResultsSection: {
    backgroundColor: '#f0f9ff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#bae6fd',
  },
  labResultsCard: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 10,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e0e7ff',
  },
  labResultsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  labResultsHeaderText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  labHealthScoreBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  labHealthScoreText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#fff',
  },
  abnormalSummary: {
    backgroundColor: '#fef3c7',
    padding: 8,
    borderRadius: 6,
    marginBottom: 12,
  },
  abnormalSummaryText: {
    fontSize: 13,
    color: '#92400e',
    fontWeight: '500',
  },
  labFindingsContainer: {
    marginBottom: 12,
  },
  labFindingsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  labFindingItem: {
    backgroundColor: '#f8fafc',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#6366f1',
  },
  labFindingHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  labFindingName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    flex: 1,
  },
  labSeverityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  labSeverityText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#fff',
    textTransform: 'capitalize',
  },
  labFindingExplanation: {
    fontSize: 13,
    color: '#4b5563',
    lineHeight: 18,
    marginBottom: 4,
  },
  labRelatedConditions: {
    fontSize: 12,
    color: '#6b7280',
    fontStyle: 'italic',
  },
  labRecommendationsContainer: {
    backgroundColor: '#ecfdf5',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  labRecommendationsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#065f46',
    marginBottom: 8,
  },
  labRecommendationItem: {
    flexDirection: 'row',
    marginBottom: 4,
  },
  labRecommendationBullet: {
    fontSize: 13,
    color: '#10b981',
    marginRight: 8,
  },
  labRecommendationText: {
    fontSize: 13,
    color: '#047857',
    flex: 1,
    lineHeight: 18,
  },
  viewAllLabsButton: {
    alignSelf: 'flex-end',
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  viewAllLabsButtonText: {
    fontSize: 14,
    color: '#6366f1',
    fontWeight: '600',
  },
  noLabResultsCard: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 10,
    marginBottom: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderStyle: 'dashed',
  },
  noLabResultsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 8,
  },
  noLabResultsDescription: {
    fontSize: 13,
    color: '#9ca3af',
    textAlign: 'center',
    marginBottom: 12,
  },
  addLabResultsButton: {
    backgroundColor: '#6366f1',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  addLabResultsButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  suggestedTestsContainer: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 10,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e0e7ff',
  },
  suggestedTestsTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1e3a5f',
    marginBottom: 4,
  },
  suggestedTestsSubtitle: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 12,
  },
  suggestedTestItem: {
    backgroundColor: '#f8fafc',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
  },
  suggestedTestHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  suggestedTestName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    flex: 1,
  },
  priorityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  priorityText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#fff',
    textTransform: 'capitalize',
  },
  suggestedTestReason: {
    fontSize: 13,
    color: '#4b5563',
    lineHeight: 18,
  },
  labDisclaimer: {
    backgroundColor: '#f0f9ff',
    padding: 10,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#bae6fd',
  },
  labDisclaimerText: {
    fontSize: 11,
    color: '#0369a1',
    lineHeight: 15,
    fontStyle: 'italic',
    textAlign: 'center',
  },

  literatureSection: {
    backgroundColor: '#f8f9fa',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
  },
  literatureSubsection: {
    marginBottom: 16,
  },
  literatureSubtitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 12,
  },
  referenceGroup: {
    marginBottom: 16,
  },
  referenceGroupTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#495057',
    marginBottom: 8,
  },
  referenceItem: {
    flexDirection: 'row',
    marginBottom: 12,
    paddingLeft: 8,
  },
  referenceBullet: {
    fontSize: 16,
    color: '#6c757d',
    marginRight: 8,
    marginTop: 2,
  },
  referenceContent: {
    flex: 1,
  },
  referenceTitle: {
    fontSize: 14,
    fontWeight: '500',
    color: '#2c3e50',
    marginBottom: 4,
  },
  referenceOrg: {
    fontSize: 12,
    color: '#6c757d',
    marginBottom: 4,
  },
  referenceLink: {
    fontSize: 13,
    color: '#007bff',
    textDecorationLine: 'underline',
  },
  disclaimerSection: {
    backgroundColor: '#fff3cd',
    borderWidth: 2,
    borderColor: '#ffc107',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  disclaimerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#856404',
    marginBottom: 16,
    textAlign: 'center',
  },
  disclaimerContent: {
    marginBottom: 16,
  },
  disclaimerHeading: {
    fontSize: 15,
    fontWeight: '600',
    color: '#856404',
    marginBottom: 8,
  },
  disclaimerText: {
    fontSize: 13,
    color: '#856404',
    lineHeight: 20,
    marginBottom: 4,
    marginLeft: 8,
  },
  disclaimerUrgent: {
    fontSize: 14,
    fontWeight: '700',
    color: '#dc3545',
    marginBottom: 8,
  },
  disclaimerWarning: {
    backgroundColor: '#f8d7da',
    borderWidth: 1,
    borderColor: '#dc3545',
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
  },
  disclaimerWarningText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#721c24',
    lineHeight: 18,
    textAlign: 'center',
  },
  uncertaintySection: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  uncertaintyDescription: {
    fontSize: 13,
    color: '#64748b',
    marginBottom: 16,
    lineHeight: 18,
  },
  uncertaintyItem: {
    marginBottom: 20,
  },
  uncertaintyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  uncertaintyClassName: {
    fontSize: 14,
    fontWeight: '500',
    color: '#475569',
    flex: 1,
  },
  uncertaintyClassNameHighlight: {
    fontSize: 15,
    fontWeight: '700',
    color: '#2c5282',
  },
  uncertaintyMean: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e293b',
  },
  uncertaintyBarContainer: {
    height: 24,
    backgroundColor: '#f1f5f9',
    borderRadius: 12,
    position: 'relative',
    marginBottom: 6,
  },
  uncertaintyTrack: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    backgroundColor: '#e2e8f0',
    borderRadius: 12,
  },
  uncertaintyRange: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    borderRadius: 12,
    opacity: 0.5,
  },
  uncertaintyPoint: {
    position: 'absolute',
    top: 4,
    width: 4,
    height: 16,
    borderRadius: 2,
  },
  uncertaintyLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  uncertaintyLabel: {
    fontSize: 12,
    color: '#64748b',
  },
  uncertaintyRangeLabel: {
    fontSize: 13,
    fontWeight: '600',
  },
  uncertaintyMetrics: {
    backgroundColor: '#f8fafc',
    borderRadius: 8,
    padding: 12,
    marginTop: 16,
    borderWidth: 1,
    borderColor: '#cbd5e1',
  },
  uncertaintyMetricsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#334155',
    marginBottom: 12,
  },
  uncertaintyMetricItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  uncertaintyMetricLabel: {
    fontSize: 13,
    color: '#475569',
    flex: 1,
  },
  uncertaintyMetricValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e293b',
  },
  uncertaintyExplanation: {
    fontSize: 12,
    color: '#64748b',
    fontStyle: 'italic',
    marginTop: 8,
  },
  dermoscopySection: {
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#bae6fd',
  },
  dermoscopyBadge: {
    fontSize: 16,
    fontWeight: '600',
    color: '#0c4a6e',
    backgroundColor: '#e0f2fe',
    padding: 8,
    borderRadius: 6,
    marginBottom: 8,
    textAlign: 'center',
  },
  dermoscopyConfidence: {
    fontSize: 14,
    color: '#075985',
    marginBottom: 12,
  },
  dermoscopyFeatures: {
    backgroundColor: '#ffffff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  dermoscopyFeaturesTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#0c4a6e',
    marginBottom: 8,
  },
  dermoscopyFeature: {
    fontSize: 13,
    color: '#475569',
    marginBottom: 4,
  },
  dermoscopyStructures: {
    backgroundColor: '#ffffff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  dermoscopyStructuresTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#0c4a6e',
    marginBottom: 12,
  },
  structureItem: {
    marginBottom: 12,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  structureName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e293b',
    marginBottom: 4,
  },
  structureDetail: {
    fontSize: 13,
    color: '#64748b',
    marginBottom: 2,
  },
  riskIndicator: {
    color: '#dc2626',
  },
  riskNote: {
    fontSize: 12,
    fontStyle: 'italic',
    color: '#dc2626',
    marginTop: 4,
  },
  dermoscopyNote: {
    fontSize: 12,
    color: '#64748b',
    fontStyle: 'italic',
    marginTop: 8,
  },
  calibrationSection: {
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#bae6fd',
  },
  calibrationInfo: {
    backgroundColor: '#ffffff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  calibrationFound: {
    fontSize: 16,
    fontWeight: '600',
    color: '#0c4a6e',
    marginBottom: 12,
  },
  calibrationDetails: {
    gap: 8,
  },
  calibrationItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
  },
  calibrationLabel: {
    fontSize: 14,
    color: '#64748b',
    fontWeight: '500',
  },
  calibrationValue: {
    fontSize: 14,
    color: '#0c4a6e',
    fontWeight: '600',
  },
  measurementCapability: {
    backgroundColor: '#e0f2fe',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  measurementCapabilityText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#075985',
    marginBottom: 6,
  },
  measurementInstructions: {
    fontSize: 13,
    color: '#0c4a6e',
    lineHeight: 18,
    marginBottom: 12,
  },
  measurementButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 4,
  },
  measurementButtonSecondary: {
    backgroundColor: '#3b82f6',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 12,
  },
  measurementButtonText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '600',
  },
  detectedObjects: {
    backgroundColor: '#ffffff',
    padding: 12,
    borderRadius: 8,
  },
  detectedObjectsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#0c4a6e',
    marginBottom: 8,
  },
  detectedObject: {
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: '#e0f2fe',
  },
  detectedObjectType: {
    fontSize: 13,
    color: '#475569',
    marginBottom: 3,
  },
  detectedObjectSize: {
    fontSize: 12,
    color: '#64748b',
  },
  noCalibration: {
    backgroundColor: '#fff7ed',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#fed7aa',
  },
  noCalibrationText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#9a3412',
    marginBottom: 6,
  },
  noCalibrationHint: {
    fontSize: 13,
    color: '#c2410c',
    lineHeight: 18,
  },
  measurementModalContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  measurementModalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: 60,
    paddingBottom: 16,
    backgroundColor: '#0284c7',
    borderBottomWidth: 1,
    borderBottomColor: '#0369a1',
  },
  closeMeasurementButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 16,
  },
  closeMeasurementButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  measurementModalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    flex: 1,
  },
  savedMeasurementsSection: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  savedMeasurementItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: '#f0f9ff',
    borderRadius: 8,
    marginBottom: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#0284c7',
  },
  savedMeasurementLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#0c4a6e',
    flex: 1,
  },
  savedMeasurementValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#0369a1',
  },
  measurementToolSection: {
    backgroundColor: '#e0f2fe',
    padding: 16,
    borderRadius: 8,
    marginTop: 12,
  },
  measurementToolText: {
    fontSize: 14,
    color: '#0c4a6e',
    marginBottom: 12,
    textAlign: 'center',
  },
  measurementToolButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
  },
  measurementToolButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  exportSection: {
    backgroundColor: '#f0f9ff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    borderWidth: 1,
    borderColor: '#bae6fd',
  },
  exportDescription: {
    fontSize: 14,
    color: '#475569',
    marginBottom: 16,
    lineHeight: 20,
  },
  exportButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 3,
  },
  exportButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  exportNote: {
    fontSize: 12,
    color: '#64748b',
    fontStyle: 'italic',
    lineHeight: 17,
    marginTop: 8,
  },
  // Biopsy Correlation Styles
  biopsySection: {
    backgroundColor: '#fef3c7',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    borderWidth: 1,
    borderColor: '#fde047',
  },
  biopsyDescription: {
    fontSize: 14,
    color: '#78350f',
    marginBottom: 16,
    lineHeight: 20,
  },
  addBiopsyButton: {
    backgroundColor: '#eab308',
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 10,
    alignItems: 'center',
  },
  addBiopsyButtonText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '600',
  },
  biopsyResultCard: {
    borderRadius: 12,
    padding: 16,
    borderWidth: 2,
  },
  biopsyCorrect: {
    backgroundColor: '#d1fae5',
    borderColor: '#10b981',
  },
  biopsyIncorrect: {
    backgroundColor: '#fee2e2',
    borderColor: '#ef4444',
  },
  biopsyResultTitle: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 16,
    color: '#1f2937',
  },
  biopsyComparison: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  biopsyComparisonItem: {
    flex: 1,
  },
  biopsyComparisonLabel: {
    fontSize: 12,
    color: '#6b7280',
    fontWeight: '600',
    marginBottom: 4,
  },
  biopsyComparisonValue: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 4,
  },
  biopsyConfidence: {
    fontSize: 11,
    color: '#6b7280',
    fontStyle: 'italic',
  },
  biopsyDate: {
    fontSize: 11,
    color: '#6b7280',
  },
  biopsyVs: {
    fontSize: 14,
    fontWeight: '700',
    color: '#9ca3af',
    marginHorizontal: 12,
  },
  accuracyCategoryBadge: {
    backgroundColor: '#fbbf24',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 6,
    alignSelf: 'flex-start',
    marginBottom: 12,
  },
  accuracyCategoryText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#78350f',
  },
  biopsyDetail: {
    fontSize: 13,
    color: '#374151',
    marginBottom: 6,
  },
  biopsyNotes: {
    marginTop: 12,
    padding: 12,
    backgroundColor: '#ffffff',
    borderRadius: 8,
  },
  biopsyNotesLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 6,
  },
  biopsyNotesText: {
    fontSize: 13,
    color: '#374151',
    lineHeight: 19,
  },
  // Biopsy Modal Styles
  biopsyModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  biopsyModalContent: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    width: '100%',
    maxWidth: 500,
    maxHeight: '90%',
  },
  biopsyModalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  biopsyModalTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1f2937',
  },
  biopsyModalClose: {
    fontSize: 28,
    color: '#6b7280',
    fontWeight: '300',
  },
  biopsyFormLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
    marginTop: 12,
  },
  biopsyInput: {
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    backgroundColor: '#ffffff',
  },
  biopsyTextArea: {
    minHeight: 100,
    textAlignVertical: 'top',
  },
  biopsyFormButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 24,
    gap: 12,
  },
  biopsyCancelButton: {
    flex: 1,
    backgroundColor: '#f3f4f6',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  biopsyCancelButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#6b7280',
  },
  biopsySubmitButton: {
    flex: 1,
    backgroundColor: '#eab308',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  biopsySubmitButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#ffffff',
  },
  // Body Map Section Styles
  bodyMapSection: {
    backgroundColor: '#dbeafe',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: '#3b82f6',
  },
  locationInfo: {
    marginTop: 12,
    marginBottom: 12,
  },
  locationItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#bfdbfe',
  },
  locationLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e40af',
  },
  locationValue: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#1e3a8a',
  },
  locationNote: {
    fontSize: 12,
    color: '#1e40af',
    fontStyle: 'italic',
    marginTop: 8,
  },
  // Specialist Section Styles
  specialistSection: {
    backgroundColor: '#f0f9ff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: '#3b82f6',
  },
  specialistDescription: {
    fontSize: 14,
    color: '#1e40af',
    marginBottom: 16,
    lineHeight: 20,
  },
  specialistCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 2,
    borderLeftWidth: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  specialistCardUrgent: {
    borderColor: '#ef4444',
    borderLeftColor: '#dc2626',
    backgroundColor: '#fef2f2',
  },
  specialistCardRecommended: {
    borderColor: '#f59e0b',
    borderLeftColor: '#d97706',
    backgroundColor: '#fffbeb',
  },
  specialistCardOptional: {
    borderColor: '#10b981',
    borderLeftColor: '#059669',
    backgroundColor: '#f0fdf4',
  },
  specialistHeader: {
    marginBottom: 8,
  },
  specialistTitleContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  specialistName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1e293b',
    flex: 1,
  },
  urgencyBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    marginLeft: 8,
  },
  urgencyBadgeUrgent: {
    backgroundColor: '#fee2e2',
  },
  urgencyBadgeRecommended: {
    backgroundColor: '#fef3c7',
  },
  urgencyBadgeOptional: {
    backgroundColor: '#d1fae5',
  },
  urgencyBadgeText: {
    fontSize: 11,
    fontWeight: '600',
  },
  urgencyTextUrgent: {
    color: '#991b1b',
  },
  urgencyTextRecommended: {
    color: '#92400e',
  },
  urgencyTextOptional: {
    color: '#065f46',
  },
  specialistReason: {
    fontSize: 14,
    color: '#64748b',
    lineHeight: 20,
    marginBottom: 4,
  },
  specialistUrgentNote: {
    fontSize: 13,
    color: '#dc2626',
    fontWeight: '600',
    marginTop: 8,
    fontStyle: 'italic',
  },
  specialistDisclaimer: {
    backgroundColor: '#e0f2fe',
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#0ea5e9',
  },
  specialistDisclaimerText: {
    fontSize: 12,
    color: '#0c4a6e',
    lineHeight: 18,
  },
  specialistCardPressed: {
    opacity: 0.7,
    transform: [{ scale: 0.98 }],
  },
  specialistFindNearby: {
    fontSize: 12,
    color: '#3b82f6',
    fontWeight: '600',
    marginTop: 8,
    textAlign: 'center',
  },
  // Plain English Explanation Styles
  plainExplanationSection: {
    backgroundColor: '#fef9f3',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: '#fb923c',
  },
  plainExplanationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  speakButton: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  speakButtonActive: {
    backgroundColor: '#ef4444',
  },
  speakButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  plainExplanationCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#fb923c',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  plainExplanationIntro: {
    fontSize: 14,
    fontWeight: '700',
    color: '#ea580c',
    marginBottom: 12,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  plainExplanationText: {
    fontSize: 16,
    lineHeight: 26,
    color: '#1e293b',
    fontWeight: '400',
  },
  plainExplanationNote: {
    backgroundColor: '#fef3c7',
    borderRadius: 8,
    padding: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#f59e0b',
  },
  plainExplanationNoteText: {
    fontSize: 12,
    color: '#92400e',
    lineHeight: 18,
  },
  // Doctor Search Modal Styles
  doctorModalContainer: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  doctorModalHeader: {
    paddingTop: 50,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  doctorModalBackButton: {
    marginBottom: 10,
  },
  doctorModalBackText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  doctorModalTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  doctorModalSubtitle: {
    fontSize: 14,
    color: '#dbeafe',
    lineHeight: 20,
  },
  doctorModalContent: {
    flex: 1,
    padding: 16,
  },
  doctorLoadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
    marginTop: 100,
  },
  doctorLoadingText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1e293b',
    marginTop: 16,
  },
  doctorLoadingSubtext: {
    fontSize: 14,
    color: '#64748b',
    marginTop: 8,
    textAlign: 'center',
  },
  doctorResultsHeader: {
    marginBottom: 16,
  },
  doctorResultsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1e293b',
    marginBottom: 4,
  },
  doctorResultsSubtitle: {
    fontSize: 14,
    color: '#64748b',
  },
  doctorCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  doctorCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  doctorName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1e293b',
    flex: 1,
    marginRight: 8,
  },
  doctorRating: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  doctorRatingText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#f59e0b',
  },
  doctorRatingCount: {
    fontSize: 12,
    color: '#64748b',
    marginLeft: 4,
  },
  doctorInfo: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  doctorInfoIcon: {
    fontSize: 16,
    marginRight: 8,
    width: 20,
  },
  doctorAddress: {
    fontSize: 14,
    color: '#475569',
    flex: 1,
    lineHeight: 20,
  },
  doctorDistance: {
    fontSize: 14,
    color: '#3b82f6',
    fontWeight: '500',
  },
  doctorStatus: {
    fontSize: 14,
    fontWeight: '600',
  },
  doctorStatusOpen: {
    color: '#10b981',
  },
  doctorStatusClosed: {
    color: '#ef4444',
  },
  doctorPhone: {
    fontSize: 14,
    color: '#475569',
  },
  doctorActions: {
    flexDirection: 'row',
    marginTop: 12,
    gap: 8,
  },
  doctorActionButton: {
    flex: 1,
    backgroundColor: '#f1f5f9',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  doctorActionButtonPrimary: {
    backgroundColor: '#3b82f6',
    borderColor: '#3b82f6',
  },
  doctorActionButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#475569',
  },
  doctorActionButtonTextPrimary: {
    color: '#ffffff',
  },
  doctorDisclaimer: {
    backgroundColor: '#fef3c7',
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
    marginBottom: 20,
    borderLeftWidth: 3,
    borderLeftColor: '#f59e0b',
  },
  doctorDisclaimerText: {
    fontSize: 12,
    color: '#92400e',
    lineHeight: 18,
  },
  doctorEmptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
    marginTop: 100,
  },
  doctorEmptyIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  doctorEmptyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1e293b',
    marginBottom: 8,
    textAlign: 'center',
  },
  doctorEmptyText: {
    fontSize: 14,
    color: '#64748b',
    textAlign: 'center',
    lineHeight: 20,
  },
  // Symptom Section Styles
  symptomSection: {
    backgroundColor: '#f0fdf4',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: '#22c55e',
  },
  symptomDescription: {
    fontSize: 14,
    color: '#15803d',
    marginBottom: 16,
    lineHeight: 20,
  },
  symptomItem: {
    marginBottom: 12,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#bbf7d0',
  },
  symptomLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#15803d',
    marginBottom: 4,
  },
  symptomValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#166534',
  },
  symptomText: {
    fontSize: 14,
    color: '#166534',
    lineHeight: 20,
  },
  symptomFlags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 12,
  },
  symptomFlag: {
    backgroundColor: '#dcfce7',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#86efac',
  },
  symptomFlagText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#166534',
  },
  addSymptomButton: {
    backgroundColor: '#22c55e',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  addSymptomButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  updateSymptomsButton: {
    backgroundColor: '#16a34a',
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 12,
  },
  updateSymptomsButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#ffffff',
  },
  medicationSection: {
    backgroundColor: '#fef3c7',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: '#f59e0b',
  },
  medicationDescription: {
    fontSize: 14,
    color: '#92400e',
    marginBottom: 16,
    lineHeight: 20,
  },
  medicationCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#fcd34d',
  },
  medicationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  medicationName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#92400e',
    flex: 1,
  },
  reactionBadge: {
    backgroundColor: '#fee2e2',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#fca5a5',
  },
  reactionBadgeText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#dc2626',
  },
  medicationDetail: {
    fontSize: 14,
    color: '#78350f',
    marginBottom: 4,
  },
  medicationDetailLabel: {
    fontWeight: '600',
  },
  addMedicationButton: {
    backgroundColor: '#f59e0b',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  addMedicationButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  updateMedicationsButton: {
    backgroundColor: '#d97706',
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 12,
  },
  updateMedicationsButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#ffffff',
  },
  medicalHistorySection: {
    backgroundColor: '#fee2e2',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: '#dc2626',
  },
  medicalHistoryDescription: {
    fontSize: 14,
    color: '#7f1d1d',
    marginBottom: 16,
    lineHeight: 20,
  },
  riskFactorItem: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 14,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#dc2626',
  },
  riskFactorTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#7f1d1d',
    marginBottom: 6,
  },
  riskFactorDetail: {
    fontSize: 14,
    color: '#991b1b',
    lineHeight: 20,
  },
  riskFactorBadge: {
    fontSize: 14,
    fontWeight: '600',
    color: '#dc2626',
    marginTop: 4,
  },
  addMedicalHistoryButton: {
    backgroundColor: '#dc2626',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  addMedicalHistoryButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  updateMedicalHistoryButton: {
    backgroundColor: '#b91c1c',
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 12,
  },
  updateMedicalHistoryButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#ffffff',
  },
  teledermatologySection: {
    backgroundColor: '#dbeafe',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: '#0284c7',
  },
  teledermatologyButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 16,
    borderRadius: 10,
    alignItems: 'center',
  },
  teledermatologyButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  shareStatusBox: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 14,
    marginTop: 14,
    borderLeftWidth: 4,
    borderLeftColor: '#10b981',
  },
  shareStatusText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#065f46',
    marginBottom: 6,
  },
  reviewedStatusText: {
    fontSize: 14,
    color: '#059669',
  },
  pendingStatusText: {
    fontSize: 14,
    color: '#f59e0b',
    fontStyle: 'italic',
  },
  fullScreenModal: {
    flex: 1,
    backgroundColor: '#fff',
  },
  fullScreenModalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 60,
    paddingBottom: 16,
    backgroundColor: '#22c55e',
    borderBottomWidth: 1,
    borderBottomColor: '#16a34a',
  },
  fullScreenModalClose: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
    width: 60,
  },
  fullScreenModalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
  },
  fullScreenModalContent: {
    backgroundColor: '#f9fafb',
  },
  fullScreenModalFooter: {
    flexDirection: 'row',
    gap: 12,
    padding: 16,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  // Infectious Disease Styles
  infectiousSection: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginTop: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    borderLeftWidth: 4,
    borderLeftColor: '#8b5cf6',
  },
  infectionTypeBadge: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    alignSelf: 'flex-start',
  },
  bacterialBadge: {
    backgroundColor: '#fee2e2',
  },
  fungalBadge: {
    backgroundColor: '#fef3c7',
  },
  viralBadge: {
    backgroundColor: '#dbeafe',
  },
  parasiticBadge: {
    backgroundColor: '#e9d5ff',
  },
  infectionTypeBadgeText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  clinicalSeverityBadge: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    alignSelf: 'flex-start',
  },
  severeSeverity: {
    backgroundColor: '#fecaca',
  },
  moderateSeverity: {
    backgroundColor: '#fed7aa',
  },
  mildSeverity: {
    backgroundColor: '#bbf7d0',
  },
  severityBadgeText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  contagionWarning: {
    backgroundColor: '#fef3c7',
    borderRadius: 12,
    padding: 16,
    marginTop: 12,
    borderWidth: 2,
    borderColor: '#f59e0b',
  },
  contagionWarningTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#92400e',
    marginBottom: 8,
  },
  contagionWarningText: {
    fontSize: 14,
    color: '#78350f',
    marginBottom: 4,
  },
  contagionWarningSubtext: {
    fontSize: 12,
    color: '#92400e',
    marginTop: 8,
    fontStyle: 'italic',
  },
  boldText: {
    fontWeight: 'bold',
  },
  clinicalTreatmentSection: {
    backgroundColor: '#f0fdf4',
    borderRadius: 12,
    padding: 16,
    marginTop: 12,
    borderWidth: 1,
    borderColor: '#86efac',
  },
  treatmentTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#166534',
    marginBottom: 12,
  },
  clinicalTreatmentItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  clinicalTreatmentBullet: {
    fontSize: 14,
    color: '#166534',
    marginRight: 8,
  },
  clinicalTreatmentText: {
    fontSize: 14,
    color: '#166534',
    flex: 1,
  },
  urgencyBox: {
    backgroundColor: '#dbeafe',
    borderRadius: 8,
    padding: 12,
    marginTop: 12,
    borderWidth: 1,
    borderColor: '#3b82f6',
  },
  urgencyInfoText: {
    fontSize: 13,
    color: '#1e40af',
    fontWeight: '600',
    textAlign: 'center',
  },
  clinicalDifferentialSection: {
    marginTop: 16,
    padding: 16,
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  differentialTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 12,
  },
  differentialItem: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  differentialHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  differentialCondition: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    flex: 1,
  },
  differentialProbability: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#3b82f6',
  },
  differentialDescription: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  // Clinical Decision Support Styles
  clinicalSupportSection: {
    backgroundColor: '#f8fafc',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 3,
    borderColor: '#3b82f6',
  },
  clinicalSupportHeader: {
    marginBottom: 16,
    borderBottomWidth: 2,
    borderBottomColor: '#e2e8f0',
    paddingBottom: 12,
  },
  clinicalSupportTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#1e293b',
    marginBottom: 4,
  },
  clinicalSupportSubtitle: {
    fontSize: 14,
    color: '#64748b',
    fontStyle: 'italic',
  },
  urgencyBanner: {
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    borderLeftWidth: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  urgencyBannerCritical: {
    backgroundColor: '#fee2e2',
    borderLeftColor: '#dc2626',
  },
  urgencyBannerHigh: {
    backgroundColor: '#fed7aa',
    borderLeftColor: '#ea580c',
  },
  urgencyBannerModerate: {
    backgroundColor: '#fef3c7',
    borderLeftColor: '#f59e0b',
  },
  urgencyBannerRoutine: {
    backgroundColor: '#d1fae5',
    borderLeftColor: '#10b981',
  },
  urgencyBannerText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  urgencyTimeline: {
    fontSize: 14,
    color: '#475569',
    fontWeight: '600',
  },
  treatmentProtocolCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1e293b',
    marginBottom: 16,
  },
  protocolStep: {
    marginBottom: 16,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  stepHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  stepNumber: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  stepNumberCritical: {
    backgroundColor: '#dc2626',
  },
  stepNumberHigh: {
    backgroundColor: '#ea580c',
  },
  stepNumberModerate: {
    backgroundColor: '#f59e0b',
  },
  stepNumberText: {
    color: '#ffffff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  stepContent: {
    flex: 1,
  },
  stepAction: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e293b',
    marginBottom: 4,
  },
  stepTimeframe: {
    fontSize: 13,
    color: '#3b82f6',
    fontWeight: '500',
  },
  stepRationale: {
    fontSize: 13,
    color: '#64748b',
    lineHeight: 19,
    fontStyle: 'italic',
    marginLeft: 48,
  },
  medicationsCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  medicationItemCard: {
    backgroundColor: '#f8fafc',
    borderRadius: 10,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#cbd5e1',
  },
  medicationItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  medicationItemName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1e293b',
    flex: 1,
  },
  responseRateBadge: {
    backgroundColor: '#dbeafe',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  responseRateText: {
    fontSize: 11,
    color: '#1e40af',
    fontWeight: '600',
  },
  medicationIndication: {
    fontSize: 14,
    color: '#475569',
    marginBottom: 12,
    lineHeight: 20,
  },
  medicationDetails: {
    marginBottom: 8,
  },
  detailRow: {
    flexDirection: 'row',
    marginBottom: 6,
  },
  detailLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#64748b',
    width: 80,
  },
  detailValue: {
    fontSize: 13,
    color: '#1e293b',
    flex: 1,
  },
  insuranceCovered: {
    color: '#10b981',
    fontWeight: '600',
  },
  insuranceNotCovered: {
    color: '#ef4444',
    fontWeight: '600',
  },
  contraindicationsBox: {
    backgroundColor: '#fee2e2',
    borderRadius: 6,
    padding: 10,
    marginTop: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#dc2626',
  },
  contraindicationsTitle: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#991b1b',
    marginBottom: 4,
  },
  contraindicationsText: {
    fontSize: 12,
    color: '#7f1d1d',
    lineHeight: 18,
  },
  sideEffectsBox: {
    backgroundColor: '#fef3c7',
    borderRadius: 6,
    padding: 10,
    marginTop: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#f59e0b',
  },
  sideEffectsTitle: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#92400e',
    marginBottom: 4,
  },
  sideEffectsText: {
    fontSize: 12,
    color: '#78350f',
    lineHeight: 18,
  },
  monitoringBox: {
    backgroundColor: '#dbeafe',
    borderRadius: 6,
    padding: 10,
    marginTop: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#3b82f6',
  },
  monitoringTitle: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#1e40af',
    marginBottom: 4,
  },
  monitoringText: {
    fontSize: 12,
    color: '#1e3a8a',
    lineHeight: 18,
  },
  drugInteractionsCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 2,
    borderColor: '#ef4444',
  },
  interactionWarning: {
    backgroundColor: '#fff7ed',
    borderRadius: 8,
    padding: 12,
    marginBottom: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  interactionCritical: {
    backgroundColor: '#fee2e2',
    borderLeftColor: '#dc2626',
  },
  interactionMajor: {
    backgroundColor: '#fed7aa',
    borderLeftColor: '#ea580c',
  },
  interactionHeader: {
    marginBottom: 6,
  },
  interactionSeverity: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#991b1b',
  },
  interactionWarningText: {
    fontSize: 13,
    color: '#1e293b',
    lineHeight: 19,
    marginBottom: 6,
  },
  interactionAction: {
    fontSize: 12,
    color: '#3b82f6',
    fontWeight: '600',
    fontStyle: 'italic',
  },
  insuranceCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  insuranceContent: {
    marginTop: 8,
  },
  codeRow: {
    flexDirection: 'row',
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  codeLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#64748b',
    width: 100,
  },
  codeValue: {
    fontSize: 13,
    color: '#1e293b',
    fontWeight: '600',
    flex: 1,
  },
  preAuthWarning: {
    backgroundColor: '#fef3c7',
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#f59e0b',
  },
  preAuthText: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#92400e',
    marginBottom: 4,
  },
  preAuthSubtext: {
    fontSize: 12,
    color: '#78350f',
    lineHeight: 18,
  },
  followUpCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  followUpTimeline: {
    marginTop: 8,
  },
  followUpItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  followUpDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#3b82f6',
    marginTop: 4,
    marginRight: 12,
  },
  followUpContent: {
    flex: 1,
  },
  followUpLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#64748b',
    marginBottom: 4,
  },
  followUpValue: {
    fontSize: 14,
    color: '#1e293b',
    lineHeight: 20,
  },
  patientEducationCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  educationItem: {
    flexDirection: 'row',
    marginBottom: 10,
  },
  educationBullet: {
    fontSize: 16,
    color: '#3b82f6',
    marginRight: 10,
    fontWeight: 'bold',
  },
  educationText: {
    fontSize: 14,
    color: '#475569',
    lineHeight: 21,
    flex: 1,
  },
  clinicalDisclaimer: {
    backgroundColor: '#e0f2fe',
    borderRadius: 8,
    padding: 14,
    marginTop: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#0ea5e9',
  },
  clinicalDisclaimerText: {
    fontSize: 11,
    color: '#0c4a6e',
    lineHeight: 17,
    fontStyle: 'italic',
  },

  // Insurance Pre-Authorization Styles
  insurancePreAuthSection: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: '#10b981',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  insurancePreAuthHeader: {
    marginBottom: 20,
    borderBottomWidth: 3,
    borderBottomColor: '#10b981',
    paddingBottom: 12,
  },
  insurancePreAuthTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#047857',
    marginBottom: 4,
  },
  insurancePreAuthSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    fontStyle: 'italic',
  },
  preAuthSummaryCard: {
    backgroundColor: '#f0fdf4',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#10b981',
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#d1fae5',
  },
  summaryLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#065f46',
  },
  summaryValue: {
    fontSize: 14,
    color: '#047857',
    flex: 1,
    textAlign: 'right',
  },
  urgentText: {
    color: '#dc2626',
    fontWeight: 'bold',
  },
  proceduresCard: {
    backgroundColor: '#fefce8',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#eab308',
  },
  procedureItem: {
    marginBottom: 12,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#fef3c7',
  },
  procedureHeader: {
    marginBottom: 6,
  },
  procedureCode: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#854d0e',
    marginBottom: 4,
  },
  procedureDescription: {
    fontSize: 15,
    fontWeight: '600',
    color: '#a16207',
  },
  procedureRationale: {
    fontSize: 13,
    color: '#78716c',
    fontStyle: 'italic',
    marginTop: 4,
  },
  letterPreviewCard: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
  },
  letterPreviewText: {
    fontSize: 13,
    color: '#1e40af',
    lineHeight: 20,
    marginBottom: 12,
  },
  viewFullLetterButton: {
    backgroundColor: '#3b82f6',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  viewFullLetterButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ffffff',
  },
  clinicalSummaryCard: {
    backgroundColor: '#f5f3ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#8b5cf6',
  },
  clinicalSummaryText: {
    fontSize: 13,
    color: '#5b21b6',
    lineHeight: 20,
    marginBottom: 12,
  },
  viewFullSummaryButton: {
    backgroundColor: '#8b5cf6',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  viewFullSummaryButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ffffff',
  },
  supportingEvidenceCard: {
    backgroundColor: '#fef2f2',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#f87171',
  },
  guidelinesBox: {
    marginBottom: 12,
  },
  evidenceSubtitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#991b1b',
    marginBottom: 8,
  },
  guidelineText: {
    fontSize: 13,
    color: '#7f1d1d',
    lineHeight: 20,
    marginBottom: 4,
  },
  accuracyBox: {
    marginTop: 8,
  },
  evidenceText: {
    fontSize: 13,
    color: '#7f1d1d',
    lineHeight: 20,
    marginBottom: 4,
  },
  preAuthActionsCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  exportPDFButton: {
    backgroundColor: '#10b981',
    borderRadius: 10,
    padding: 16,
    alignItems: 'center',
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  exportPDFButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  shareButton: {
    backgroundColor: '#0ea5e9',
    borderRadius: 10,
    padding: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  shareButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  insuranceDisclaimer: {
    backgroundColor: '#fef3c7',
    borderRadius: 8,
    padding: 14,
    marginTop: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  insuranceDisclaimerText: {
    fontSize: 11,
    color: '#78350f',
    lineHeight: 17,
    fontStyle: 'italic',
  },

  // Approval Likelihood Styles
  approvalLikelihoodCard: {
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
  },
  approvalBadge: {
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    marginBottom: 16,
  },
  approvalProbability: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  approvalCategory: {
    fontSize: 18,
    fontWeight: '600',
    color: '#ffffff',
    marginTop: 4,
  },
  approvalRecommendation: {
    fontSize: 14,
    color: '#1e40af',
    lineHeight: 21,
    marginBottom: 16,
  },
  approvalFactors: {
    marginTop: 8,
  },
  factorsTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1e40af',
    marginBottom: 12,
  },
  factorItem: {
    marginBottom: 16,
  },
  factorHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  factorName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a8a',
  },
  factorScore: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#3b82f6',
  },
  factorDescription: {
    fontSize: 12,
    color: '#64748b',
    marginBottom: 8,
    lineHeight: 18,
  },
  factorBar: {
    height: 8,
    backgroundColor: '#e0e7ff',
    borderRadius: 4,
    overflow: 'hidden',
  },
  factorBarFill: {
    height: '100%',
    backgroundColor: '#3b82f6',
  },

  // Status Tracking Styles
  statusTrackingCard: {
    backgroundColor: '#fef9c3',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#eab308',
  },
  currentStatus: {
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#fef3c7',
    borderWidth: 2,
    borderColor: '#f59e0b',
    marginBottom: 12,
  },
  statusText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#78350f',
    textAlign: 'center',
  },
  statusDate: {
    fontSize: 13,
    color: '#78350f',
    marginBottom: 6,
  },
  updateStatusButton: {
    backgroundColor: '#eab308',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
    marginTop: 12,
  },
  updateStatusButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ffffff',
  },

  // Auto-Fill Forms Styles
  autoFillFormsCard: {
    backgroundColor: '#f5f3ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#8b5cf6',
  },
  autoFillDescription: {
    fontSize: 13,
    color: '#5b21b6',
    marginBottom: 12,
    lineHeight: 20,
  },
  formTypeButton: {
    backgroundColor: '#8b5cf6',
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
    marginBottom: 10,
  },
  formTypeButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ffffff',
  },
  formInstructions: {
    marginTop: 12,
    padding: 12,
    backgroundColor: '#ede9fe',
    borderRadius: 8,
  },
  formInstructionsTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#5b21b6',
    marginBottom: 6,
  },
  formInstructionsText: {
    fontSize: 12,
    color: '#6b21a8',
    lineHeight: 18,
  },

  // Form Modal Styles
  formModalContainer: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  formModalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#8b5cf6',
    borderBottomWidth: 1,
    borderBottomColor: '#7c3aed',
  },
  formModalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    flex: 1,
  },
  formModalCloseButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  formModalCloseButtonText: {
    fontSize: 20,
    color: '#ffffff',
    fontWeight: 'bold',
  },
  formModalContent: {
    flex: 1,
    padding: 16,
  },
  formModalNotice: {
    backgroundColor: '#fef3c7',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  formModalNoticeText: {
    fontSize: 13,
    color: '#92400e',
    lineHeight: 18,
  },
  formSection: {
    marginBottom: 16,
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#8b5cf6',
  },
  formSectionTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 8,
    textTransform: 'capitalize',
  },
  formField: {
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  formFieldLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 4,
    textTransform: 'capitalize',
  },
  formFieldValue: {
    fontSize: 14,
    color: '#111827',
    lineHeight: 20,
  },
  formArrayItem: {
    marginBottom: 12,
    paddingLeft: 12,
    paddingTop: 8,
    paddingBottom: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#8b5cf6',
    backgroundColor: '#faf5ff',
    borderRadius: 6,
  },
  formArrayItemHeader: {
    fontSize: 13,
    fontWeight: '700',
    color: '#6b21a8',
    marginBottom: 8,
    paddingBottom: 6,
    borderBottomWidth: 1,
    borderBottomColor: '#e9d5ff',
  },
  formFieldPlaceholder: {
    color: '#dc2626',
    fontWeight: '600',
    backgroundColor: '#fef2f2',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  formModalFooter: {
    marginTop: 20,
    padding: 16,
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#6366f1',
  },
  formModalFooterText: {
    fontSize: 12,
    color: '#4b5563',
    lineHeight: 18,
  },
  formModalActions: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: '#f9fafb',
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
    gap: 12,
  },
  formModalActionButton: {
    flex: 1,
    backgroundColor: '#8b5cf6',
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
  },
  formModalActionButtonSecondary: {
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#8b5cf6',
  },
  formModalActionButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ffffff',
  },
  formModalActionButtonTextSecondary: {
    color: '#8b5cf6',
  },
  formTypeDescription: {
    fontSize: 13,
    color: '#4b5563',
    backgroundColor: '#f0f9ff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    lineHeight: 20,
    borderLeftWidth: 3,
    borderLeftColor: '#3b82f6',
  },
  formMetadata: {
    backgroundColor: '#ede9fe',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  formDataSummary: {
    backgroundColor: '#fff7ed',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#f97316',
  },
  formDataSummaryTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#9a3412',
    marginBottom: 6,
  },
  formDataSummaryText: {
    fontSize: 12,
    color: '#7c2d12',
    lineHeight: 18,
  },

  // Lab Context Styles
  labContextSection: {
    backgroundColor: '#f0f9ff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#bae6fd',
  },
  labContextHeader: {
    marginBottom: 16,
  },
  labContextTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#0c4a6e',
    marginBottom: 4,
  },
  labContextSubtitle: {
    fontSize: 14,
    color: '#0369a1',
  },
  labDataInfo: {
    backgroundColor: '#e0f2fe',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
  },
  labDataInfoText: {
    fontSize: 13,
    color: '#0369a1',
    marginBottom: 4,
  },
  labSupportingCard: {
    backgroundColor: '#dcfce7',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#22c55e',
  },
  labCardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  labItem: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  labItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  labName: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937',
  },
  labStatusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  labStatusLow: {
    backgroundColor: '#fef3c7',
  },
  labStatusHigh: {
    backgroundColor: '#fee2e2',
  },
  labStatusText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#1f2937',
  },
  labValue: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 4,
  },
  labExplanation: {
    fontSize: 13,
    color: '#166534',
    fontStyle: 'italic',
  },
  labInsightsCard: {
    backgroundColor: '#fef3c7',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  insightItem: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  insightCondition: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  insightAdjustment: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  insightOriginal: {
    fontSize: 14,
    color: '#6b7280',
  },
  insightArrow: {
    fontSize: 14,
    color: '#9ca3af',
    marginHorizontal: 8,
  },
  insightAdjusted: {
    fontSize: 15,
    fontWeight: '600',
  },
  insightDelta: {
    fontSize: 13,
    marginLeft: 6,
  },
  adjustmentPositive: {
    color: '#16a34a',
  },
  adjustmentNegative: {
    color: '#dc2626',
  },
  labOtherCard: {
    backgroundColor: '#f3f4f6',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  labOtherSubtitle: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 12,
    fontStyle: 'italic',
  },
  labItemCompact: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  labNameCompact: {
    fontSize: 14,
    color: '#374151',
  },
  labStatusCompact: {
    fontSize: 14,
    fontWeight: '500',
    color: '#6b7280',
  },
  labRecommendationsCard: {
    backgroundColor: '#ede9fe',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#8b5cf6',
  },
  labRecItem: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  labRecBullet: {
    fontSize: 14,
    color: '#6d28d9',
    marginRight: 8,
  },
  labRecText: {
    fontSize: 14,
    color: '#5b21b6',
    flex: 1,
  },
  noLabDataCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
  },
  noLabDataText: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 16,
  },
  addLabButton: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
  },
  addLabButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },

  // AI Explanation Styles
  aiExplanationContainer: {
    marginTop: 16,
    backgroundColor: 'rgba(102, 126, 234, 0.05)',
    borderRadius: 12,
    overflow: 'hidden',
  },
  learnMoreButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: 20,
    backgroundColor: 'rgba(102, 126, 234, 0.08)',
  },
  learnMoreButtonActive: {
    backgroundColor: 'rgba(102, 126, 234, 0.15)',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(102, 126, 234, 0.2)',
  },
  learnMoreButtonText: {
    color: '#667eea',
    fontSize: 15,
    fontWeight: '600',
  },
  aiExplanationContent: {
    padding: 16,
  },
  aiExplanationLoading: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 30,
  },
  aiExplanationLoadingText: {
    marginTop: 12,
    color: '#667eea',
    fontSize: 14,
  },
  aiExplanationError: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  aiExplanationErrorText: {
    color: '#e53e3e',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 12,
    paddingVertical: 8,
    paddingHorizontal: 20,
    backgroundColor: '#667eea',
    borderRadius: 8,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  aiExplanationText: {
    // Container for explanation text
  },
  aiExplanationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  aiExplanationTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#2d3748',
    marginLeft: 8,
  },
  aiExplanationBody: {
    fontSize: 14,
    lineHeight: 22,
    color: '#4a5568',
  },
  aiExplanationDisclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginTop: 16,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0, 0, 0, 0.08)',
  },
  aiExplanationDisclaimerText: {
    flex: 1,
    marginLeft: 6,
    fontSize: 11,
    color: '#718096',
    lineHeight: 16,
  },

  // Differential Reasoning Styles
  reasoningContainer: {
    marginTop: 16,
    backgroundColor: 'rgba(139, 92, 246, 0.05)',
    borderRadius: 12,
    overflow: 'hidden',
  },
  reasoningButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: 20,
    backgroundColor: 'rgba(139, 92, 246, 0.08)',
  },
  reasoningButtonActive: {
    backgroundColor: 'rgba(139, 92, 246, 0.15)',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(139, 92, 246, 0.2)',
  },
  reasoningButtonText: {
    color: '#8b5cf6',
    fontSize: 15,
    fontWeight: '600',
  },
  reasoningContent: {
    padding: 16,
  },
  reasoningLoading: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 30,
  },
  reasoningLoadingText: {
    marginTop: 12,
    color: '#8b5cf6',
    fontSize: 14,
  },
  reasoningError: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  reasoningErrorText: {
    color: '#e53e3e',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
  reasoningRetryButton: {
    marginTop: 12,
    paddingVertical: 8,
    paddingHorizontal: 20,
    backgroundColor: '#8b5cf6',
    borderRadius: 8,
  },
  reasoningRetryButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  reasoningText: {
    // Container for reasoning text
  },
  reasoningHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  reasoningTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#2d3748',
    marginLeft: 8,
  },
  reasoningBody: {
    fontSize: 14,
    lineHeight: 22,
    color: '#4a5568',
  },
  reasoningDisclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginTop: 16,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0, 0, 0, 0.08)',
  },
  reasoningDisclaimerText: {
    flex: 1,
    marginLeft: 6,
    fontSize: 11,
    color: '#718096',
    lineHeight: 16,
  },
});