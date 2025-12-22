import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  Pressable,
  ActivityIndicator,
  Alert,
  Modal,
  Platform,
  Switch,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { API_BASE_URL } from '../config';
import AuthService from '../services/AuthService';
import * as DocumentPicker from 'expo-document-picker';

type TabType = 'blood' | 'urine' | 'stool';

interface LabResult {
  id: number;
  test_date: string;
  test_type: string;
  lab_name: string;
  abnormal_flags: string[];
  skin_relevance_analysis: any;
  created_at: string;
}

export default function LabResultsScreen() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  const [activeTab, setActiveTab] = useState<TabType>('blood');
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [labResults, setLabResults] = useState<LabResult[]>([]);
  const [showEntryForm, setShowEntryForm] = useState(false);
  const [showResultDetail, setShowResultDetail] = useState<LabResult | null>(null);
  const [isParsing, setIsParsing] = useState(false);
  const [useOCR, setUseOCR] = useState(false);
  const [parseResult, setParseResult] = useState<any>(null);
  const [editingLabId, setEditingLabId] = useState<number | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Form state
  const [formData, setFormData] = useState({
    test_date: new Date().toISOString().split('T')[0],
    test_type: 'comprehensive',
    lab_name: '',
    ordering_physician: '',
    notes: '',

    // Blood - CBC
    wbc: '',
    rbc: '',
    hemoglobin: '',
    hematocrit: '',
    platelets: '',
    mcv: '',
    mch: '',
    mchc: '',
    rdw: '',
    mpv: '',

    // Blood - WBC Differential (%)
    neutrophils: '',
    lymphocytes: '',
    monocytes: '',
    eosinophils: '',
    basophils: '',

    // Blood - Absolute WBC Counts
    neutrophils_abs: '',
    lymphocytes_abs: '',
    monocytes_abs: '',
    eosinophils_abs: '',
    basophils_abs: '',

    // Blood - Metabolic Panel
    glucose_fasting: '',
    hba1c: '',
    eag: '',
    bun: '',
    creatinine: '',
    bun_creatinine_ratio: '',
    egfr: '',
    egfr_african_american: '',
    sodium: '',
    potassium: '',
    chloride: '',
    co2: '',
    calcium: '',
    magnesium: '',

    // Blood - Liver Function
    alt: '',
    ast: '',
    alp: '',
    bilirubin_total: '',
    albumin: '',
    total_protein: '',
    globulin: '',
    albumin_globulin_ratio: '',

    // Blood - Lipid Panel
    cholesterol_total: '',
    ldl: '',
    hdl: '',
    triglycerides: '',
    chol_hdl_ratio: '',
    non_hdl_cholesterol: '',

    // Blood - Thyroid Panel
    tsh: '',
    t3_uptake: '',
    t4_total: '',
    free_t4_index: '',
    t4_free: '',

    // Blood - Iron Studies
    iron: '',
    ferritin: '',
    tibc: '',

    // Blood - Vitamins
    vitamin_d: '',
    vitamin_b12: '',
    folate: '',

    // Blood - Inflammatory Markers
    crp: '',
    esr: '',

    // Blood - Autoimmune
    ana_positive: false,

    // Blood - Allergy
    ige_total: '',

    // Urinalysis - Physical
    urine_color: '',
    urine_appearance: '',
    urine_specific_gravity: '',
    urine_ph: '',

    // Urinalysis - Chemical
    urine_protein: '',
    urine_glucose: '',
    urine_ketones: '',
    urine_blood: '',
    urine_bilirubin: '',
    urine_urobilinogen: '',
    urine_nitrite: '',
    urine_leukocyte_esterase: '',

    // Urinalysis - Microscopic
    urine_wbc: '',
    urine_rbc: '',
    urine_bacteria: '',
    urine_squamous_epithelial: '',
    urine_hyaline_cast: '',

    // Stool
    stool_color: '',
    stool_occult_blood: '',
    stool_parasites: '',
    stool_calprotectin: '',
  });

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
      return;
    }
    loadLabResults();
  }, [isAuthenticated]);

  const loadLabResults = async () => {
    try {
      setIsLoading(true);
      const token = await AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/lab-results`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setLabResults(data.lab_results || []);
      }
    } catch (error) {
      console.error('Failed to load lab results:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (field: string, value: string | boolean) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async () => {
    try {
      setIsSaving(true);
      const token = await AuthService.getToken();

      // Convert string values to numbers where needed
      const payload: any = {
        test_date: formData.test_date,
        test_type: formData.test_type,
        lab_name: formData.lab_name || null,
        ordering_physician: formData.ordering_physician || null,
        notes: formData.notes || null,
        is_manually_entered: true,
      };

      // Add numeric values only if they're not empty
      const numericFields = [
        // CBC
        'wbc', 'rbc', 'hemoglobin', 'hematocrit', 'platelets',
        'mcv', 'mch', 'mchc', 'rdw', 'mpv',
        // WBC Differential
        'neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils',
        // Absolute WBC
        'neutrophils_abs', 'lymphocytes_abs', 'monocytes_abs', 'eosinophils_abs', 'basophils_abs',
        // Metabolic
        'glucose_fasting', 'hba1c', 'eag', 'bun', 'creatinine', 'bun_creatinine_ratio',
        'egfr', 'egfr_african_american', 'sodium', 'potassium', 'chloride', 'co2', 'calcium', 'magnesium',
        // Liver
        'alt', 'ast', 'alp', 'bilirubin_total', 'albumin', 'total_protein', 'globulin', 'albumin_globulin_ratio',
        // Lipids
        'cholesterol_total', 'ldl', 'hdl', 'triglycerides', 'chol_hdl_ratio', 'non_hdl_cholesterol',
        // Thyroid
        'tsh', 't3_uptake', 't4_total', 'free_t4_index', 't4_free',
        // Iron
        'iron', 'ferritin', 'tibc',
        // Vitamins
        'vitamin_d', 'vitamin_b12', 'folate',
        // Inflammatory
        'crp', 'esr',
        // Allergy
        'ige_total',
        // Urine numeric
        'urine_specific_gravity', 'urine_ph',
        // Stool numeric
        'stool_calprotectin',
      ];

      numericFields.forEach(field => {
        const value = formData[field as keyof typeof formData];
        if (value && value !== '') {
          payload[field] = parseFloat(value as string);
        }
      });

      // Add string fields
      const stringFields = [
        // Urine
        'urine_color', 'urine_appearance', 'urine_protein', 'urine_glucose', 'urine_ketones',
        'urine_blood', 'urine_bilirubin', 'urine_urobilinogen', 'urine_nitrite', 'urine_leukocyte_esterase',
        'urine_wbc', 'urine_rbc', 'urine_bacteria', 'urine_squamous_epithelial', 'urine_hyaline_cast',
        // Stool
        'stool_color', 'stool_occult_blood', 'stool_parasites',
      ];

      stringFields.forEach(field => {
        const value = formData[field as keyof typeof formData];
        if (value && value !== '') {
          payload[field] = value;
        }
      });

      // Add boolean fields
      payload.ana_positive = formData.ana_positive;

      // Use PUT for editing, POST for new
      const url = editingLabId
        ? `${API_BASE_URL}/lab-results/${editingLabId}`
        : `${API_BASE_URL}/lab-results`;
      const method = editingLabId ? 'PUT' : 'POST';

      const response = await fetch(url, {
        method,
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        const result = await response.json();
        Alert.alert(
          'Success',
          editingLabId
            ? 'Lab results updated successfully!'
            : `Lab results saved! Found ${result.abnormal_count} values with skin relevance.`,
          [{ text: 'OK', onPress: () => {
            setShowEntryForm(false);
            setEditingLabId(null);
            loadLabResults();
            // Reset form
            setFormData({
              ...formData,
              wbc: '', rbc: '', hemoglobin: '', hematocrit: '', platelets: '', eosinophils: '',
              glucose_fasting: '', hba1c: '', creatinine: '', egfr: '',
              alt: '', ast: '', bilirubin_total: '',
              tsh: '', t4_free: '',
              iron: '', ferritin: '',
              vitamin_d: '', vitamin_b12: '',
              crp: '', esr: '',
              ana_positive: false,
              ige_total: '',
              urine_color: '', urine_ph: '', urine_protein: '', urine_glucose: '', urine_blood: '',
              stool_occult_blood: '', stool_parasites: '',
            });
          }}]
        );
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to save lab results');
      }
    } catch (error) {
      console.error('Failed to save lab results:', error);
      Alert.alert('Error', 'Failed to save lab results');
    } finally {
      setIsSaving(false);
    }
  };

  const handlePDFUpload = async () => {
    try {
      // Pick a PDF document
      const result = await DocumentPicker.getDocumentAsync({
        type: 'application/pdf',
        copyToCacheDirectory: true,
      });

      if (result.canceled || !result.assets || result.assets.length === 0) {
        return;
      }

      const file = result.assets[0];
      setIsParsing(true);
      setParseResult(null);

      const token = await AuthService.getToken();

      // Create form data
      const formDataToSend = new FormData();
      formDataToSend.append('file', {
        uri: file.uri,
        type: 'application/pdf',
        name: file.name || 'lab_results.pdf',
      } as any);
      formDataToSend.append('use_ocr', useOCR ? 'true' : 'false');

      const response = await fetch(`${API_BASE_URL}/lab-results/parse-pdf`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formDataToSend,
      });

      if (response.ok) {
        const data = await response.json();
        setParseResult(data);

        // Populate form with extracted values
        if (data.extracted_values) {
          const newFormData = { ...formData };

          // Update form with extracted values
          Object.keys(data.extracted_values).forEach(key => {
            const value = data.extracted_values[key];
            if (key in newFormData) {
              if (typeof value === 'boolean') {
                (newFormData as any)[key] = value;
              } else if (typeof value === 'number') {
                (newFormData as any)[key] = value.toString();
              } else {
                (newFormData as any)[key] = value;
              }
            }
          });

          // Update lab name and test date if found
          if (data.lab_name) {
            newFormData.lab_name = data.lab_name;
          }
          if (data.test_date) {
            newFormData.test_date = data.test_date;
          }

          setFormData(newFormData);
        }

        Alert.alert(
          'PDF Parsed Successfully',
          `Extracted ${data.values_found} lab values.\n\nConfidence: ${data.parse_confidence}\n\nPlease review the values and make any corrections before saving.`,
          [{ text: 'OK' }]
        );
      } else {
        const error = await response.json();
        Alert.alert(
          'Parsing Failed',
          error.detail || 'Could not extract values from PDF. Try enabling OCR for scanned documents.',
          [{ text: 'OK' }]
        );
      }
    } catch (error) {
      console.error('PDF upload error:', error);
      Alert.alert('Error', 'Failed to upload PDF. Please try again.');
    } finally {
      setIsParsing(false);
    }
  };

  const handleEditLabResult = async (labId: number) => {
    try {
      setIsLoading(true);
      const token = await AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/lab-results/${labId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();

        // Populate form with existing data
        const newFormData = { ...formData };

        // Metadata
        newFormData.test_date = data.test_date || new Date().toISOString().split('T')[0];
        newFormData.test_type = data.test_type || 'comprehensive';
        newFormData.lab_name = data.lab_name || '';
        newFormData.ordering_physician = data.ordering_physician || '';
        newFormData.notes = data.notes || '';

        // Blood panel - CBC
        const cbc = data.blood_panel?.cbc || {};
        newFormData.wbc = cbc.wbc?.toString() || '';
        newFormData.rbc = cbc.rbc?.toString() || '';
        newFormData.hemoglobin = cbc.hemoglobin?.toString() || '';
        newFormData.hematocrit = cbc.hematocrit?.toString() || '';
        newFormData.platelets = cbc.platelets?.toString() || '';
        newFormData.mcv = cbc.mcv?.toString() || '';
        newFormData.mch = cbc.mch?.toString() || '';
        newFormData.mchc = cbc.mchc?.toString() || '';
        newFormData.rdw = cbc.rdw?.toString() || '';
        newFormData.mpv = cbc.mpv?.toString() || '';

        // WBC Differential
        const wbcDiff = data.blood_panel?.wbc_differential || {};
        newFormData.neutrophils = wbcDiff.neutrophils?.toString() || '';
        newFormData.lymphocytes = wbcDiff.lymphocytes?.toString() || '';
        newFormData.monocytes = wbcDiff.monocytes?.toString() || '';
        newFormData.eosinophils = wbcDiff.eosinophils?.toString() || '';
        newFormData.basophils = wbcDiff.basophils?.toString() || '';

        // WBC Absolute
        const wbcAbs = data.blood_panel?.wbc_absolute || {};
        newFormData.neutrophils_abs = wbcAbs.neutrophils_abs?.toString() || '';
        newFormData.lymphocytes_abs = wbcAbs.lymphocytes_abs?.toString() || '';
        newFormData.monocytes_abs = wbcAbs.monocytes_abs?.toString() || '';
        newFormData.eosinophils_abs = wbcAbs.eosinophils_abs?.toString() || '';
        newFormData.basophils_abs = wbcAbs.basophils_abs?.toString() || '';

        // Metabolic
        const metabolic = data.blood_panel?.metabolic || {};
        newFormData.glucose_fasting = metabolic.glucose_fasting?.toString() || '';
        newFormData.hba1c = metabolic.hba1c?.toString() || '';
        newFormData.eag = metabolic.eag?.toString() || '';
        newFormData.bun = metabolic.bun?.toString() || '';
        newFormData.creatinine = metabolic.creatinine?.toString() || '';
        newFormData.bun_creatinine_ratio = metabolic.bun_creatinine_ratio?.toString() || '';
        newFormData.egfr = metabolic.egfr?.toString() || '';
        newFormData.egfr_african_american = metabolic.egfr_african_american?.toString() || '';
        newFormData.sodium = metabolic.sodium?.toString() || '';
        newFormData.potassium = metabolic.potassium?.toString() || '';
        newFormData.chloride = metabolic.chloride?.toString() || '';
        newFormData.co2 = metabolic.co2?.toString() || '';
        newFormData.calcium = metabolic.calcium?.toString() || '';
        newFormData.magnesium = metabolic.magnesium?.toString() || '';

        // Liver
        const liver = data.blood_panel?.liver || {};
        newFormData.alt = liver.alt?.toString() || '';
        newFormData.ast = liver.ast?.toString() || '';
        newFormData.alp = liver.alp?.toString() || '';
        newFormData.bilirubin_total = liver.bilirubin_total?.toString() || '';
        newFormData.albumin = liver.albumin?.toString() || '';
        newFormData.total_protein = liver.total_protein?.toString() || '';
        newFormData.globulin = liver.globulin?.toString() || '';
        newFormData.albumin_globulin_ratio = liver.albumin_globulin_ratio?.toString() || '';

        // Lipid
        const lipid = data.blood_panel?.lipid || {};
        newFormData.cholesterol_total = lipid.cholesterol_total?.toString() || '';
        newFormData.ldl = lipid.ldl?.toString() || '';
        newFormData.hdl = lipid.hdl?.toString() || '';
        newFormData.triglycerides = lipid.triglycerides?.toString() || '';
        newFormData.chol_hdl_ratio = lipid.chol_hdl_ratio?.toString() || '';
        newFormData.non_hdl_cholesterol = lipid.non_hdl_cholesterol?.toString() || '';

        // Thyroid
        const thyroid = data.blood_panel?.thyroid || {};
        newFormData.tsh = thyroid.tsh?.toString() || '';
        newFormData.t3_uptake = thyroid.t3_uptake?.toString() || '';
        newFormData.t4_total = thyroid.t4_total?.toString() || '';
        newFormData.free_t4_index = thyroid.free_t4_index?.toString() || '';
        newFormData.t4_free = thyroid.t4_free?.toString() || '';

        // Iron
        const ironStudies = data.blood_panel?.iron || {};
        newFormData.iron = ironStudies.iron?.toString() || '';
        newFormData.ferritin = ironStudies.ferritin?.toString() || '';
        newFormData.tibc = ironStudies.tibc?.toString() || '';

        // Vitamins
        const vitamins = data.blood_panel?.vitamins || {};
        newFormData.vitamin_d = vitamins.vitamin_d?.toString() || '';
        newFormData.vitamin_b12 = vitamins.vitamin_b12?.toString() || '';
        newFormData.folate = vitamins.folate?.toString() || '';

        // Inflammatory
        const inflammatory = data.blood_panel?.inflammatory || {};
        newFormData.crp = inflammatory.crp?.toString() || '';
        newFormData.esr = inflammatory.esr?.toString() || '';

        // Autoimmune
        const autoimmune = data.blood_panel?.autoimmune || {};
        newFormData.ana_positive = autoimmune.ana_positive || false;

        // Allergy
        const allergy = data.blood_panel?.allergy || {};
        newFormData.ige_total = allergy.ige_total?.toString() || '';

        // Urinalysis
        const urinePhysical = data.urinalysis?.physical || {};
        newFormData.urine_color = urinePhysical.color || '';
        newFormData.urine_appearance = urinePhysical.appearance || '';
        newFormData.urine_specific_gravity = urinePhysical.specific_gravity?.toString() || '';
        newFormData.urine_ph = urinePhysical.ph?.toString() || '';

        const urineChemical = data.urinalysis?.chemical || {};
        newFormData.urine_protein = urineChemical.protein || '';
        newFormData.urine_glucose = urineChemical.glucose || '';
        newFormData.urine_ketones = urineChemical.ketones || '';
        newFormData.urine_blood = urineChemical.blood || '';
        newFormData.urine_bilirubin = urineChemical.bilirubin || '';
        newFormData.urine_urobilinogen = urineChemical.urobilinogen || '';
        newFormData.urine_nitrite = urineChemical.nitrite || '';
        newFormData.urine_leukocyte_esterase = urineChemical.leukocyte_esterase || '';

        const urineMicro = data.urinalysis?.microscopic || {};
        newFormData.urine_wbc = urineMicro.wbc || '';
        newFormData.urine_rbc = urineMicro.rbc || '';
        newFormData.urine_bacteria = urineMicro.bacteria || '';
        newFormData.urine_squamous_epithelial = urineMicro.squamous_epithelial || '';

        const urineCasts = data.urinalysis?.casts || {};
        newFormData.urine_hyaline_cast = urineCasts.hyaline_cast || '';

        // Stool
        const stool = data.stool_test || {};
        newFormData.stool_color = stool.color || '';
        newFormData.stool_occult_blood = stool.occult_blood || '';
        newFormData.stool_parasites = stool.parasites || '';
        newFormData.stool_calprotectin = stool.calprotectin?.toString() || '';

        setFormData(newFormData);
        setEditingLabId(labId);
        setShowResultDetail(null);
        setShowEntryForm(true);
      } else {
        Alert.alert('Error', 'Failed to load lab result details');
      }
    } catch (error) {
      console.error('Failed to load lab result for editing:', error);
      Alert.alert('Error', 'Failed to load lab result');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteLabResult = async (labId: number) => {
    Alert.alert(
      'Delete Lab Result',
      'Are you sure you want to delete this lab result? This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              setIsDeleting(true);
              const token = await AuthService.getToken();
              const response = await fetch(`${API_BASE_URL}/lab-results/${labId}`, {
                method: 'DELETE',
                headers: {
                  'Authorization': `Bearer ${token}`,
                },
              });

              if (response.ok) {
                Alert.alert('Success', 'Lab result deleted successfully');
                setShowResultDetail(null);
                loadLabResults();
              } else {
                Alert.alert('Error', 'Failed to delete lab result');
              }
            } catch (error) {
              console.error('Failed to delete lab result:', error);
              Alert.alert('Error', 'Failed to delete lab result');
            } finally {
              setIsDeleting(false);
            }
          },
        },
      ]
    );
  };

  const resetForm = () => {
    setFormData({
      test_date: new Date().toISOString().split('T')[0],
      test_type: 'comprehensive',
      lab_name: '',
      ordering_physician: '',
      notes: '',
      wbc: '', rbc: '', hemoglobin: '', hematocrit: '', platelets: '',
      mcv: '', mch: '', mchc: '', rdw: '', mpv: '',
      neutrophils: '', lymphocytes: '', monocytes: '', eosinophils: '', basophils: '',
      neutrophils_abs: '', lymphocytes_abs: '', monocytes_abs: '', eosinophils_abs: '', basophils_abs: '',
      glucose_fasting: '', hba1c: '', eag: '', bun: '', creatinine: '', bun_creatinine_ratio: '',
      egfr: '', egfr_african_american: '', sodium: '', potassium: '', chloride: '', co2: '', calcium: '', magnesium: '',
      alt: '', ast: '', alp: '', bilirubin_total: '', albumin: '', total_protein: '', globulin: '', albumin_globulin_ratio: '',
      cholesterol_total: '', ldl: '', hdl: '', triglycerides: '', chol_hdl_ratio: '', non_hdl_cholesterol: '',
      tsh: '', t3_uptake: '', t4_total: '', free_t4_index: '', t4_free: '',
      iron: '', ferritin: '', tibc: '',
      vitamin_d: '', vitamin_b12: '', folate: '',
      crp: '', esr: '',
      ana_positive: false,
      ige_total: '',
      urine_color: '', urine_appearance: '', urine_specific_gravity: '', urine_ph: '',
      urine_protein: '', urine_glucose: '', urine_ketones: '', urine_blood: '',
      urine_bilirubin: '', urine_urobilinogen: '', urine_nitrite: '', urine_leukocyte_esterase: '',
      urine_wbc: '', urine_rbc: '', urine_bacteria: '', urine_squamous_epithelial: '', urine_hyaline_cast: '',
      stool_color: '', stool_occult_blood: '', stool_parasites: '', stool_calprotectin: '',
    });
    setEditingLabId(null);
    setParseResult(null);
  };

  const renderLabInput = (label: string, field: string, normalRange: string, unit?: string) => (
    <View style={styles.inputGroup}>
      <Text style={styles.inputLabel}>{label} <Text style={styles.normalRangeHint}>(Normal: {normalRange})</Text></Text>
      <View style={styles.inputRow}>
        <TextInput
          style={styles.input}
          value={formData[field as keyof typeof formData] as string}
          onChangeText={(value) => handleInputChange(field, value)}
          placeholder="Enter value"
          keyboardType="decimal-pad"
          placeholderTextColor="#9ca3af"
        />
        {unit && <Text style={styles.unitText}>{unit}</Text>}
      </View>
    </View>
  );

  const renderBloodTab = () => (
    <View>
      <Text style={styles.sectionHeader}>Complete Blood Count (CBC)</Text>
      {renderLabInput('White Blood Cells', 'wbc', '3.8-10.8', 'K/uL')}
      {renderLabInput('Red Blood Cells', 'rbc', '4.2-5.8', 'M/uL')}
      {renderLabInput('Hemoglobin', 'hemoglobin', '13.2-17.1', 'g/dL')}
      {renderLabInput('Hematocrit', 'hematocrit', '38.5-50.0', '%')}
      {renderLabInput('Platelets', 'platelets', '140-400', 'K/uL')}
      {renderLabInput('MCV', 'mcv', '80-100', 'fL')}
      {renderLabInput('MCH', 'mch', '27-33', 'pg')}
      {renderLabInput('MCHC', 'mchc', '32-36', 'g/dL')}
      {renderLabInput('RDW', 'rdw', '11-15', '%')}
      {renderLabInput('MPV', 'mpv', '7.5-12.5', 'fL')}

      <Text style={styles.sectionHeader}>WBC Differential (%)</Text>
      {renderLabInput('Neutrophils', 'neutrophils', '40-70', '%')}
      {renderLabInput('Lymphocytes', 'lymphocytes', '20-40', '%')}
      {renderLabInput('Monocytes', 'monocytes', '2-8', '%')}
      {renderLabInput('Eosinophils', 'eosinophils', '1-4', '%')}
      {renderLabInput('Basophils', 'basophils', '0-1', '%')}

      <Text style={styles.sectionHeader}>Absolute WBC Counts</Text>
      {renderLabInput('Abs Neutrophils', 'neutrophils_abs', '1500-7800', 'cells/uL')}
      {renderLabInput('Abs Lymphocytes', 'lymphocytes_abs', '850-3900', 'cells/uL')}
      {renderLabInput('Abs Monocytes', 'monocytes_abs', '200-950', 'cells/uL')}
      {renderLabInput('Abs Eosinophils', 'eosinophils_abs', '15-500', 'cells/uL')}
      {renderLabInput('Abs Basophils', 'basophils_abs', '0-200', 'cells/uL')}

      <Text style={styles.sectionHeader}>Comprehensive Metabolic Panel</Text>
      {renderLabInput('Glucose (Fasting)', 'glucose_fasting', '65-99', 'mg/dL')}
      {renderLabInput('HbA1c', 'hba1c', '<5.7', '%')}
      {renderLabInput('eAG', 'eag', 'calc', 'mg/dL')}
      {renderLabInput('BUN', 'bun', '7-25', 'mg/dL')}
      {renderLabInput('Creatinine', 'creatinine', '0.70-1.25', 'mg/dL')}
      {renderLabInput('BUN/Creatinine Ratio', 'bun_creatinine_ratio', '6-22', '')}
      {renderLabInput('eGFR', 'egfr', '>60', 'mL/min')}
      {renderLabInput('eGFR (African Am.)', 'egfr_african_american', '>60', 'mL/min')}
      {renderLabInput('Sodium', 'sodium', '135-146', 'mmol/L')}
      {renderLabInput('Potassium', 'potassium', '3.5-5.3', 'mmol/L')}
      {renderLabInput('Chloride', 'chloride', '98-110', 'mmol/L')}
      {renderLabInput('CO2', 'co2', '20-32', 'mmol/L')}
      {renderLabInput('Calcium', 'calcium', '8.6-10.3', 'mg/dL')}
      {renderLabInput('Magnesium', 'magnesium', '1.6-2.3', 'mg/dL')}

      <Text style={styles.sectionHeader}>Liver Function</Text>
      {renderLabInput('ALT', 'alt', '9-46', 'U/L')}
      {renderLabInput('AST', 'ast', '10-35', 'U/L')}
      {renderLabInput('Alkaline Phosphatase', 'alp', '40-115', 'U/L')}
      {renderLabInput('Bilirubin (Total)', 'bilirubin_total', '0.2-1.2', 'mg/dL')}
      {renderLabInput('Protein (Total)', 'total_protein', '6.1-8.1', 'g/dL')}
      {renderLabInput('Albumin', 'albumin', '3.6-5.1', 'g/dL')}
      {renderLabInput('Globulin', 'globulin', '1.9-3.7', 'g/dL')}
      {renderLabInput('A/G Ratio', 'albumin_globulin_ratio', '1.0-2.5', '')}

      <Text style={styles.sectionHeader}>Lipid Panel</Text>
      {renderLabInput('Cholesterol (Total)', 'cholesterol_total', '<200', 'mg/dL')}
      {renderLabInput('HDL Cholesterol', 'hdl', '>40', 'mg/dL')}
      {renderLabInput('LDL Cholesterol', 'ldl', '<100', 'mg/dL')}
      {renderLabInput('Triglycerides', 'triglycerides', '<150', 'mg/dL')}
      {renderLabInput('Chol/HDL Ratio', 'chol_hdl_ratio', '<5.0', '')}
      {renderLabInput('Non-HDL Cholesterol', 'non_hdl_cholesterol', '<130', 'mg/dL')}

      <Text style={styles.sectionHeader}>Thyroid Panel</Text>
      {renderLabInput('TSH', 'tsh', '0.40-4.50', 'mIU/L')}
      {renderLabInput('T3 Uptake', 't3_uptake', '22-35', '%')}
      {renderLabInput('T4 (Total)', 't4_total', '4.9-10.5', 'mcg/dL')}
      {renderLabInput('Free T4 Index', 'free_t4_index', '1.4-3.8', '')}
      {renderLabInput('Free T4', 't4_free', '0.8-1.8', 'ng/dL')}

      <Text style={styles.sectionHeader}>Iron Studies</Text>
      {renderLabInput('Iron', 'iron', '60-170', 'µg/dL')}
      {renderLabInput('Ferritin', 'ferritin', '12-300', 'ng/mL')}
      {renderLabInput('TIBC', 'tibc', '250-370', 'µg/dL')}

      <Text style={styles.sectionHeader}>Vitamins</Text>
      {renderLabInput('Vitamin D', 'vitamin_d', '30-100', 'ng/mL')}
      {renderLabInput('Vitamin B12', 'vitamin_b12', '200-900', 'pg/mL')}
      {renderLabInput('Folate', 'folate', '2.7-17', 'ng/mL')}

      <Text style={styles.sectionHeader}>Inflammatory Markers</Text>
      {renderLabInput('CRP', 'crp', '<8.0', 'mg/L')}
      {renderLabInput('ESR (Sed Rate)', 'esr', '<20', 'mm/hr')}

      <Text style={styles.sectionHeader}>Autoimmune</Text>
      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>ANA (Antinuclear Antibody)</Text>
        <View style={styles.toggleRow}>
          <Pressable
            style={[styles.toggleButton, !formData.ana_positive && styles.toggleButtonActive]}
            onPress={() => handleInputChange('ana_positive', false)}
          >
            <Text style={[styles.toggleText, !formData.ana_positive && styles.toggleTextActive]}>Negative</Text>
          </Pressable>
          <Pressable
            style={[styles.toggleButton, formData.ana_positive && styles.toggleButtonActive]}
            onPress={() => handleInputChange('ana_positive', true)}
          >
            <Text style={[styles.toggleText, formData.ana_positive && styles.toggleTextActive]}>Positive</Text>
          </Pressable>
        </View>
      </View>

      <Text style={styles.sectionHeader}>Allergy</Text>
      {renderLabInput('Total IgE', 'ige_total', '<100', 'IU/mL')}
    </View>
  );

  const renderTextInput = (label: string, field: string, placeholder: string) => (
    <View style={styles.inputGroup}>
      <Text style={styles.inputLabel}>{label}</Text>
      <TextInput
        style={styles.input}
        value={formData[field as keyof typeof formData] as string}
        onChangeText={(value) => handleInputChange(field, value)}
        placeholder={placeholder}
        placeholderTextColor="#9ca3af"
      />
    </View>
  );

  const renderUrineTab = () => (
    <View>
      <Text style={styles.sectionHeader}>Physical Characteristics</Text>
      {renderTextInput('Color', 'urine_color', 'yellow')}
      {renderTextInput('Appearance', 'urine_appearance', 'clear')}
      {renderLabInput('Specific Gravity', 'urine_specific_gravity', '1.001-1.035', '')}
      {renderLabInput('pH', 'urine_ph', '5.0-8.0', '')}

      <Text style={styles.sectionHeader}>Chemical Analysis</Text>
      {renderTextInput('Protein', 'urine_protein', 'negative, trace, 1+, 2+, 3+')}
      {renderTextInput('Glucose', 'urine_glucose', 'negative, trace, 1+, etc.')}
      {renderTextInput('Ketones', 'urine_ketones', 'negative, trace, small, etc.')}
      {renderTextInput('Blood (Occult)', 'urine_blood', 'negative, trace, 1+, etc.')}
      {renderTextInput('Bilirubin', 'urine_bilirubin', 'negative, 1+, 2+, 3+')}
      {renderTextInput('Urobilinogen', 'urine_urobilinogen', 'normal, 0.2-1.0 mg/dL')}
      {renderTextInput('Nitrite', 'urine_nitrite', 'negative or positive')}
      {renderTextInput('Leukocyte Esterase', 'urine_leukocyte_esterase', 'negative, trace, 1+, etc.')}

      <Text style={styles.sectionHeader}>Microscopic Examination</Text>
      {renderTextInput('WBC', 'urine_wbc', 'none seen, <5 /HPF')}
      {renderTextInput('RBC', 'urine_rbc', 'none seen, <2 /HPF')}
      {renderTextInput('Bacteria', 'urine_bacteria', 'none seen, few, moderate')}
      {renderTextInput('Squamous Epithelial', 'urine_squamous_epithelial', 'none seen, <5 /HPF')}
      {renderTextInput('Hyaline Cast', 'urine_hyaline_cast', 'none seen')}
    </View>
  );

  const renderStoolTab = () => (
    <View>
      <Text style={styles.sectionHeader}>Stool Test</Text>
      {renderTextInput('Color', 'stool_color', 'brown, black, red, clay, etc.')}
      {renderTextInput('Occult Blood', 'stool_occult_blood', 'positive or negative')}
      {renderTextInput('Parasites', 'stool_parasites', 'none detected, or specify')}
      {renderLabInput('Calprotectin', 'stool_calprotectin', '<50', 'µg/g')}
    </View>
  );

  const renderResultCard = (result: LabResult) => {
    const abnormalCount = result.abnormal_flags?.length || 0;
    const skinScore = result.skin_relevance_analysis?.overall_skin_health_score;

    return (
      <Pressable
        key={result.id}
        style={styles.resultCard}
        onPress={() => setShowResultDetail(result)}
      >
        <View style={styles.resultHeader}>
          <Text style={styles.resultDate}>{result.test_date}</Text>
          <Text style={styles.resultType}>{result.test_type}</Text>
        </View>
        {result.lab_name && (
          <Text style={styles.resultLab}>{result.lab_name}</Text>
        )}
        <View style={styles.resultStats}>
          {abnormalCount > 0 && (
            <View style={styles.abnormalBadge}>
              <Text style={styles.abnormalText}>{abnormalCount} abnormal</Text>
            </View>
          )}
          {skinScore !== undefined && (
            <View style={[styles.scoreBadge, skinScore >= 80 ? styles.scoreGood : skinScore >= 60 ? styles.scoreMedium : styles.scoreLow]}>
              <Text style={styles.scoreText}>Skin Score: {skinScore}</Text>
            </View>
          )}
        </View>
      </Pressable>
    );
  };

  const renderResultDetailModal = () => {
    if (!showResultDetail) return null;

    const analysis = showResultDetail.skin_relevance_analysis;

    return (
      <Modal
        visible={!!showResultDetail}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setShowResultDetail(null)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Lab Results Analysis</Text>
            <Pressable onPress={() => setShowResultDetail(null)}>
              <Text style={styles.closeButton}>Close</Text>
            </Pressable>
          </View>

          <ScrollView style={styles.modalContent}>
            <Text style={styles.modalDate}>{showResultDetail.test_date}</Text>

            {/* Edit and Delete Buttons */}
            <View style={styles.actionButtonsRow}>
              <Pressable
                style={styles.editButton}
                onPress={() => handleEditLabResult(showResultDetail.id)}
                disabled={isLoading}
              >
                <Text style={styles.editButtonText}>
                  {isLoading ? 'Loading...' : 'Edit Results'}
                </Text>
              </Pressable>
              <Pressable
                style={styles.deleteButton}
                onPress={() => handleDeleteLabResult(showResultDetail.id)}
                disabled={isDeleting}
              >
                <Text style={styles.deleteButtonText}>
                  {isDeleting ? 'Deleting...' : 'Delete'}
                </Text>
              </Pressable>
            </View>

            {analysis && (
              <>
                {/* Skin Health Score */}
                <View style={styles.scoreSection}>
                  <Text style={styles.scoreSectionTitle}>Skin Health Score</Text>
                  <View style={[styles.scoreCircle,
                    analysis.overall_skin_health_score >= 80 ? styles.scoreCircleGood :
                    analysis.overall_skin_health_score >= 60 ? styles.scoreCircleMedium :
                    styles.scoreCircleLow
                  ]}>
                    <Text style={styles.scoreValue}>{analysis.overall_skin_health_score}</Text>
                    <Text style={styles.scoreLabel}>/ 100</Text>
                  </View>
                </View>

                {/* Key Concerns */}
                {analysis.key_concerns?.length > 0 && (
                  <View style={styles.concernsSection}>
                    <Text style={styles.concernsTitle}>Key Concerns</Text>
                    {analysis.key_concerns.map((concern: string, idx: number) => (
                      <View key={idx} style={styles.concernItem}>
                        <Text style={styles.concernBullet}>!</Text>
                        <Text style={styles.concernText}>{concern}</Text>
                      </View>
                    ))}
                  </View>
                )}

                {/* Abnormal Findings */}
                {analysis.abnormal_findings?.length > 0 && (
                  <View style={styles.findingsSection}>
                    <Text style={styles.findingsTitle}>Abnormal Findings with Skin Impact</Text>
                    {analysis.abnormal_findings.map((finding: any, idx: number) => (
                      <View key={idx} style={styles.findingCard}>
                        <View style={styles.findingHeader}>
                          <Text style={styles.findingName}>{finding.lab_name}</Text>
                          <Text style={[styles.findingStatus,
                            finding.status.includes('high') ? styles.statusHigh : styles.statusLow
                          ]}>
                            {finding.status.toUpperCase()}
                          </Text>
                        </View>
                        <Text style={styles.findingValue}>
                          {finding.value} {finding.unit} (Normal: {finding.normal_range})
                        </Text>
                        {finding.skin_implications?.length > 0 && (
                          <View style={styles.implicationsBox}>
                            <Text style={styles.implicationsLabel}>Skin Implications:</Text>
                            {finding.skin_implications.slice(0, 3).map((impl: string, i: number) => (
                              <Text key={i} style={styles.implicationText}>- {impl}</Text>
                            ))}
                          </View>
                        )}
                      </View>
                    ))}
                  </View>
                )}

                {/* Recommendations */}
                {analysis.recommendations?.length > 0 && (
                  <View style={styles.recommendationsSection}>
                    <Text style={styles.recommendationsTitle}>Recommendations</Text>
                    {analysis.recommendations.map((rec: string, idx: number) => (
                      <View key={idx} style={styles.recItem}>
                        <Text style={styles.recBullet}>+</Text>
                        <Text style={styles.recText}>{rec}</Text>
                      </View>
                    ))}
                  </View>
                )}

                {/* Disclaimer */}
                <View style={styles.disclaimerBox}>
                  <Text style={styles.disclaimerText}>
                    {analysis.disclaimer}
                  </Text>
                </View>
              </>
            )}
          </ScrollView>
        </View>
      </Modal>
    );
  };

  return (
    <View style={styles.container}>
      <LinearGradient colors={['#0f172a', '#1e293b']} style={styles.header}>
        <Pressable onPress={() => router.back()} style={styles.backButton}>
          <Text style={styles.backText}>Back</Text>
        </Pressable>
        <Text style={styles.headerTitle}>Lab Results</Text>
        <Pressable onPress={() => setShowEntryForm(true)} style={styles.addButton}>
          <Text style={styles.addButtonText}>+ Add</Text>
        </Pressable>
      </LinearGradient>

      {isLoading ? (
        <ActivityIndicator size="large" color="#3b82f6" style={styles.loader} />
      ) : (
        <ScrollView style={styles.content}>
          {labResults.length === 0 ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyIcon}>🧪</Text>
              <Text style={styles.emptyTitle}>No Lab Results Yet</Text>
              <Text style={styles.emptyText}>
                Add your blood, urine, or stool test results to get personalized skin health insights.
              </Text>
              <Pressable style={styles.emptyButton} onPress={() => setShowEntryForm(true)}>
                <Text style={styles.emptyButtonText}>Add Lab Results</Text>
              </Pressable>
            </View>
          ) : (
            <View style={styles.resultsList}>
              <Text style={styles.resultsCount}>{labResults.length} lab result(s)</Text>
              {labResults.map(renderResultCard)}
            </View>
          )}

          <View style={styles.infoBox}>
            <Text style={styles.infoTitle}>Why Add Lab Results?</Text>
            <Text style={styles.infoText}>
              Your blood, urine, and stool test results can reveal important connections to your skin health:
            </Text>
            <Text style={styles.infoBullet}>- Thyroid issues can cause dry skin and hair loss</Text>
            <Text style={styles.infoBullet}>- Low vitamin D worsens psoriasis and eczema</Text>
            <Text style={styles.infoBullet}>- High blood sugar affects wound healing</Text>
            <Text style={styles.infoBullet}>- Iron deficiency causes pale skin and brittle nails</Text>
            <Text style={styles.infoBullet}>- Elevated IgE indicates allergic skin conditions</Text>
          </View>
        </ScrollView>
      )}

      {/* Entry Form Modal */}
      <Modal
        visible={showEntryForm}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => {
          setShowEntryForm(false);
          resetForm();
        }}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Pressable onPress={() => {
              setShowEntryForm(false);
              resetForm();
            }}>
              <Text style={styles.cancelButton}>Cancel</Text>
            </Pressable>
            <Text style={styles.modalTitle}>
              {editingLabId ? 'Edit Lab Results' : 'Enter Lab Results'}
            </Text>
            <Pressable onPress={handleSubmit} disabled={isSaving || isParsing}>
              <Text style={[styles.saveButton, (isSaving || isParsing) && styles.saveButtonDisabled]}>
                {isSaving ? 'Saving...' : editingLabId ? 'Update' : 'Save'}
              </Text>
            </Pressable>
          </View>

          {/* PDF Upload Section */}
          <View style={styles.pdfUploadSection}>
            <Text style={styles.pdfUploadTitle}>Upload Lab Results PDF</Text>
            <Text style={styles.pdfUploadSubtitle}>
              Upload your lab report PDF to auto-fill the form
            </Text>

            <View style={styles.ocrToggleRow}>
              <Text style={styles.ocrLabel}>Use OCR (for scanned documents)</Text>
              <Switch
                value={useOCR}
                onValueChange={setUseOCR}
                trackColor={{ false: '#e2e8f0', true: '#93c5fd' }}
                thumbColor={useOCR ? '#3b82f6' : '#f4f3f4'}
              />
            </View>

            <Pressable
              style={[styles.pdfUploadButton, isParsing && styles.pdfUploadButtonDisabled]}
              onPress={handlePDFUpload}
              disabled={isParsing}
            >
              {isParsing ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <Text style={styles.pdfUploadButtonText}>Select PDF File</Text>
              )}
            </Pressable>

            {parseResult && (
              <View style={styles.parseResultBox}>
                <Text style={styles.parseResultTitle}>
                  Extracted {parseResult.values_found} values
                </Text>
                <Text style={styles.parseResultConfidence}>
                  Confidence: {parseResult.parse_confidence}
                </Text>
                {parseResult.validation_warnings?.length > 0 && (
                  <View style={styles.parseWarnings}>
                    {parseResult.validation_warnings.map((warning: string, idx: number) => (
                      <Text key={idx} style={styles.parseWarningText}>! {warning}</Text>
                    ))}
                  </View>
                )}
              </View>
            )}

            <View style={styles.dividerRow}>
              <View style={styles.dividerLine} />
              <Text style={styles.dividerText}>or enter manually</Text>
              <View style={styles.dividerLine} />
            </View>
          </View>

          {/* Tabs */}
          <View style={styles.tabContainer}>
            <Pressable
              style={[styles.tab, activeTab === 'blood' && styles.tabActive]}
              onPress={() => setActiveTab('blood')}
            >
              <Text style={[styles.tabText, activeTab === 'blood' && styles.tabTextActive]}>
                Blood
              </Text>
            </Pressable>
            <Pressable
              style={[styles.tab, activeTab === 'urine' && styles.tabActive]}
              onPress={() => setActiveTab('urine')}
            >
              <Text style={[styles.tabText, activeTab === 'urine' && styles.tabTextActive]}>
                Urine
              </Text>
            </Pressable>
            <Pressable
              style={[styles.tab, activeTab === 'stool' && styles.tabActive]}
              onPress={() => setActiveTab('stool')}
            >
              <Text style={[styles.tabText, activeTab === 'stool' && styles.tabTextActive]}>
                Stool
              </Text>
            </Pressable>
          </View>

          <ScrollView style={styles.formContent}>
            {/* Common fields */}
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Test Date</Text>
              <TextInput
                style={styles.input}
                value={formData.test_date}
                onChangeText={(value) => handleInputChange('test_date', value)}
                placeholder="YYYY-MM-DD"
                placeholderTextColor="#9ca3af"
              />
            </View>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Lab Name (Optional)</Text>
              <TextInput
                style={styles.input}
                value={formData.lab_name}
                onChangeText={(value) => handleInputChange('lab_name', value)}
                placeholder="Quest, LabCorp, etc."
                placeholderTextColor="#9ca3af"
              />
            </View>

            {/* Tab content */}
            {activeTab === 'blood' && renderBloodTab()}
            {activeTab === 'urine' && renderUrineTab()}
            {activeTab === 'stool' && renderStoolTab()}

            <View style={{ height: 100 }} />
          </ScrollView>
        </View>
      </Modal>

      {renderResultDetailModal()}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  header: {
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 20,
    paddingHorizontal: 20,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  backButton: {
    padding: 8,
  },
  backText: {
    color: '#60a5fa',
    fontSize: 16,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '700',
  },
  addButton: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  addButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  loader: {
    flex: 1,
    justifyContent: 'center',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyIcon: {
    fontSize: 60,
    marginBottom: 16,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 8,
  },
  emptyText: {
    fontSize: 14,
    color: '#64748b',
    textAlign: 'center',
    paddingHorizontal: 40,
    marginBottom: 24,
  },
  emptyButton: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  emptyButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
  },
  resultsList: {
    marginBottom: 20,
  },
  resultsCount: {
    fontSize: 14,
    color: '#64748b',
    marginBottom: 12,
  },
  resultCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  resultDate: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e293b',
  },
  resultType: {
    fontSize: 14,
    color: '#64748b',
    textTransform: 'capitalize',
  },
  resultLab: {
    fontSize: 14,
    color: '#64748b',
    marginBottom: 8,
  },
  resultStats: {
    flexDirection: 'row',
    gap: 8,
  },
  abnormalBadge: {
    backgroundColor: '#fef2f2',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  abnormalText: {
    color: '#dc2626',
    fontSize: 12,
    fontWeight: '600',
  },
  scoreBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  scoreGood: {
    backgroundColor: '#f0fdf4',
  },
  scoreMedium: {
    backgroundColor: '#fefce8',
  },
  scoreLow: {
    backgroundColor: '#fef2f2',
  },
  scoreText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#1e293b',
  },
  infoBox: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    marginTop: 8,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e40af',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 14,
    color: '#1e40af',
    marginBottom: 12,
  },
  infoBullet: {
    fontSize: 13,
    color: '#3b82f6',
    marginBottom: 4,
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 60 : 20,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e293b',
  },
  cancelButton: {
    color: '#64748b',
    fontSize: 16,
  },
  saveButton: {
    color: '#3b82f6',
    fontSize: 16,
    fontWeight: '600',
  },
  saveButtonDisabled: {
    color: '#94a3b8',
  },
  closeButton: {
    color: '#3b82f6',
    fontSize: 16,
    fontWeight: '600',
  },
  pdfUploadSection: {
    backgroundColor: '#f0f9ff',
    padding: 16,
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#bae6fd',
  },
  pdfUploadTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0c4a6e',
    marginBottom: 4,
  },
  pdfUploadSubtitle: {
    fontSize: 13,
    color: '#0369a1',
    marginBottom: 12,
  },
  ocrToggleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
    backgroundColor: '#fff',
    padding: 12,
    borderRadius: 8,
  },
  ocrLabel: {
    fontSize: 14,
    color: '#475569',
  },
  pdfUploadButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  pdfUploadButtonDisabled: {
    backgroundColor: '#7dd3fc',
  },
  pdfUploadButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  parseResultBox: {
    marginTop: 12,
    backgroundColor: '#ecfdf5',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#a7f3d0',
  },
  parseResultTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#065f46',
  },
  parseResultConfidence: {
    fontSize: 13,
    color: '#047857',
    marginTop: 4,
  },
  parseWarnings: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#a7f3d0',
  },
  parseWarningText: {
    fontSize: 12,
    color: '#b45309',
    marginBottom: 2,
  },
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 16,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#cbd5e1',
  },
  dividerText: {
    paddingHorizontal: 12,
    fontSize: 13,
    color: '#64748b',
  },
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: '#f1f5f9',
    margin: 16,
    borderRadius: 8,
    padding: 4,
  },
  tab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 6,
  },
  tabActive: {
    backgroundColor: '#fff',
  },
  tabText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#64748b',
  },
  tabTextActive: {
    color: '#3b82f6',
  },
  formContent: {
    flex: 1,
    paddingHorizontal: 16,
  },
  sectionHeader: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e293b',
    marginTop: 20,
    marginBottom: 12,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  inputGroup: {
    marginBottom: 16,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#475569',
    marginBottom: 6,
  },
  normalRangeHint: {
    fontSize: 12,
    fontWeight: '400',
    color: '#94a3b8',
    fontStyle: 'italic',
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  input: {
    flex: 1,
    backgroundColor: '#f8fafc',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
    color: '#1e293b',
  },
  unitText: {
    marginLeft: 8,
    color: '#64748b',
    fontSize: 14,
    minWidth: 60,
  },
  toggleRow: {
    flexDirection: 'row',
    gap: 8,
  },
  toggleButton: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
    backgroundColor: '#f1f5f9',
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  toggleButtonActive: {
    backgroundColor: '#3b82f6',
    borderColor: '#3b82f6',
  },
  toggleText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#64748b',
  },
  toggleTextActive: {
    color: '#fff',
  },
  modalContent: {
    flex: 1,
    padding: 16,
  },
  modalDate: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 20,
  },
  scoreSection: {
    alignItems: 'center',
    marginBottom: 24,
  },
  scoreSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#64748b',
    marginBottom: 12,
  },
  scoreCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreCircleGood: {
    backgroundColor: '#dcfce7',
  },
  scoreCircleMedium: {
    backgroundColor: '#fef9c3',
  },
  scoreCircleLow: {
    backgroundColor: '#fee2e2',
  },
  scoreValue: {
    fontSize: 36,
    fontWeight: '700',
    color: '#1e293b',
  },
  scoreLabel: {
    fontSize: 14,
    color: '#64748b',
  },
  concernsSection: {
    backgroundColor: '#fef2f2',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  concernsTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#dc2626',
    marginBottom: 12,
  },
  concernItem: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  concernBullet: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: '#dc2626',
    color: '#fff',
    textAlign: 'center',
    lineHeight: 20,
    marginRight: 10,
    fontWeight: '700',
  },
  concernText: {
    flex: 1,
    fontSize: 14,
    color: '#7f1d1d',
  },
  findingsSection: {
    marginBottom: 16,
  },
  findingsTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 12,
  },
  findingCard: {
    backgroundColor: '#f8fafc',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  findingHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  findingName: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1e293b',
  },
  findingStatus: {
    fontSize: 12,
    fontWeight: '700',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  statusHigh: {
    backgroundColor: '#fee2e2',
    color: '#dc2626',
  },
  statusLow: {
    backgroundColor: '#fef3c7',
    color: '#d97706',
  },
  findingValue: {
    fontSize: 13,
    color: '#64748b',
    marginBottom: 8,
  },
  implicationsBox: {
    backgroundColor: '#fff',
    borderRadius: 6,
    padding: 10,
  },
  implicationsLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#64748b',
    marginBottom: 4,
  },
  implicationText: {
    fontSize: 13,
    color: '#475569',
    marginBottom: 2,
  },
  recommendationsSection: {
    backgroundColor: '#f0fdf4',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  recommendationsTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#166534',
    marginBottom: 12,
  },
  recItem: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  recBullet: {
    width: 20,
    color: '#16a34a',
    fontWeight: '700',
    fontSize: 16,
  },
  recText: {
    flex: 1,
    fontSize: 14,
    color: '#166534',
  },
  disclaimerBox: {
    backgroundColor: '#fefce8',
    borderRadius: 8,
    padding: 12,
    marginTop: 16,
    marginBottom: 40,
  },
  disclaimerText: {
    fontSize: 12,
    color: '#854d0e',
    lineHeight: 18,
    fontStyle: 'italic',
  },
  actionButtonsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    gap: 12,
  },
  editButton: {
    flex: 1,
    backgroundColor: '#3b82f6',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  editButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
  deleteButton: {
    flex: 1,
    backgroundColor: '#fee2e2',
    borderWidth: 1,
    borderColor: '#fecaca',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  deleteButtonText: {
    color: '#dc2626',
    fontWeight: '600',
    fontSize: 14,
  },
});
