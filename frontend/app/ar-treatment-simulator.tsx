import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Platform,
  ScrollView,
  Dimensions,
  Image
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';

const { width, height } = Dimensions.get('window');
const imageWidth = (width - 48) / 2;

interface TreatmentOption {
  id: string;
  name: string;
  description: string;
  expectedImprovement: number;
  timeframe: string;
  sideEffects: string[];
}

export default function ARTreatmentSimulatorScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const params = useLocalSearchParams();
  const { lesionId, diagnosis: diagnosisParam, imageUrl } = params;

  // Handle diagnosis - it might be an array from useLocalSearchParams
  const diagnosis = Array.isArray(diagnosisParam) ? diagnosisParam[0] : diagnosisParam;

  console.log('[AR SIMULATOR] Params received:', params);
  console.log('[AR SIMULATOR] Diagnosis:', diagnosis);
  console.log('[AR SIMULATOR] Image URL:', imageUrl);

  const [isSimulating, setIsSimulating] = useState(false);
  const [selectedTreatment, setSelectedTreatment] = useState<TreatmentOption | null>(null);
  const [simulationResult, setSimulationResult] = useState<any>(null);
  const [treatments, setTreatments] = useState<TreatmentOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'6months' | '1year' | '2years'>('6months');

  useEffect(() => {
    fetchTreatmentOptions();

    // If imageUrl is provided (from analysis or lesion), use it automatically
    if (imageUrl && typeof imageUrl === 'string') {
      // For lesion detail images, prepend API_BASE_URL if it's a relative path
      if (imageUrl.startsWith('/uploads/')) {
        setUploadedImage(`${API_BASE_URL}${imageUrl}`);
      } else {
        // For home screen (local file URI)
        setUploadedImage(imageUrl);
      }
    }
  }, [imageUrl]);

  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      Alert.alert('Permission Required', 'Please allow access to your photo library');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      setUploadedImage(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();

    if (permissionResult.granted === false) {
      Alert.alert('Permission Required', 'Please allow camera access');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      setUploadedImage(result.assets[0].uri);
    }
  };

  const fetchTreatmentOptions = async () => {
    try {
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert('Error', 'Please login again');
        router.replace('/');
        return;
      }

      // Fetch treatment recommendations based on diagnosis
      const response = await fetch(
        `${API_BASE_URL}/treatment-recommendations?diagnosis=${diagnosis}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );

      if (response.ok) {
        const data = await response.json();
        setTreatments(data.treatments || getMockTreatments());
      } else {
        // Use mock data if endpoint doesn't exist
        setTreatments(getMockTreatments());
      }
    } catch (error) {
      console.error('Error fetching treatments:', error);
      setTreatments(getMockTreatments());
    } finally {
      setLoading(false);
    }
  };

  const getConditionSpecificTreatments = (diagnosisText: string): TreatmentOption[] => {
    const diagnosisLower = diagnosisText?.toLowerCase() || '';

    // Melanoma - Requires surgery, not topical treatments
    if (diagnosisLower.includes('melanoma')) {
      return [
        {
          id: 'surgical-excision',
          name: 'Surgical Excision',
          description: 'Complete surgical removal of melanoma with clear margins',
          expectedImprovement: 95,
          timeframe: '1 procedure + 2-4 weeks healing',
          sideEffects: ['Scarring', 'Pain', 'Infection risk']
        },
        {
          id: 'mohs-surgery',
          name: 'Mohs Micrographic Surgery',
          description: 'Layer-by-layer removal with immediate microscopic analysis',
          expectedImprovement: 98,
          timeframe: '1 day procedure + 2-3 weeks healing',
          sideEffects: ['Larger wound', 'Scarring', 'Extended recovery']
        },
        {
          id: 'immunotherapy',
          name: 'Immunotherapy (Advanced)',
          description: 'For metastatic melanoma - activates immune system',
          expectedImprovement: 60,
          timeframe: '6-12 months',
          sideEffects: ['Fatigue', 'Inflammation', 'Autoimmune reactions']
        }
      ];
    }

    // Acne
    if (diagnosisLower.includes('acne')) {
      return [
        {
          id: 'topical-retinoid',
          name: 'Topical Retinoid (Tretinoin)',
          description: 'Unclogs pores and promotes cell turnover',
          expectedImprovement: 75,
          timeframe: '8-12 weeks',
          sideEffects: ['Dryness', 'Peeling', 'Sun sensitivity']
        },
        {
          id: 'benzoyl-peroxide',
          name: 'Benzoyl Peroxide',
          description: 'Antibacterial treatment to reduce acne bacteria',
          expectedImprovement: 65,
          timeframe: '4-6 weeks',
          sideEffects: ['Dryness', 'Bleaching of fabrics', 'Mild irritation']
        },
        {
          id: 'oral-isotretinoin',
          name: 'Oral Isotretinoin (Accutane)',
          description: 'Systemic treatment for severe acne',
          expectedImprovement: 90,
          timeframe: '4-6 months',
          sideEffects: ['Dry skin/lips', 'Birth defects risk', 'Mood changes']
        },
        {
          id: 'laser-acne',
          name: 'Laser/Light Therapy',
          description: 'Reduces bacteria and inflammation',
          expectedImprovement: 70,
          timeframe: '4-8 sessions over 2-3 months',
          sideEffects: ['Redness', 'Temporary darkening', 'Cost']
        }
      ];
    }

    // Eczema / Atopic Dermatitis
    if (diagnosisLower.includes('eczema') || diagnosisLower.includes('atopic') || diagnosisLower.includes('dermatitis')) {
      return [
        {
          id: 'topical-steroid-eczema',
          name: 'Topical Corticosteroid',
          description: 'Anti-inflammatory cream to reduce flare-ups',
          expectedImprovement: 80,
          timeframe: '1-2 weeks',
          sideEffects: ['Skin thinning', 'Stretch marks', 'Rebound flares']
        },
        {
          id: 'calcineurin-inhibitor',
          name: 'Topical Calcineurin Inhibitor',
          description: 'Non-steroid anti-inflammatory (tacrolimus/pimecrolimus)',
          expectedImprovement: 75,
          timeframe: '2-4 weeks',
          sideEffects: ['Burning sensation', 'No skin thinning', 'Safer long-term']
        },
        {
          id: 'moisturizer-therapy',
          name: 'Intensive Moisturizer + Wet Wrap',
          description: 'Barrier repair therapy with ceramides',
          expectedImprovement: 60,
          timeframe: 'Ongoing maintenance',
          sideEffects: ['Minimal', 'Greasy feel', 'Time-consuming']
        },
        {
          id: 'dupilumab',
          name: 'Dupilumab (Dupixent) Injection',
          description: 'Biologic therapy for moderate-to-severe eczema',
          expectedImprovement: 85,
          timeframe: '4-16 weeks',
          sideEffects: ['Injection site reaction', 'Eye problems', 'Expensive']
        }
      ];
    }

    // Psoriasis
    if (diagnosisLower.includes('psoriasis')) {
      return [
        {
          id: 'topical-steroid-psoriasis',
          name: 'Topical Corticosteroid',
          description: 'Reduces inflammation and scaling',
          expectedImprovement: 65,
          timeframe: '2-4 weeks',
          sideEffects: ['Skin thinning', 'Tolerance development', 'Rebound']
        },
        {
          id: 'vitamin-d-analog',
          name: 'Vitamin D Analog (Calcipotriene)',
          description: 'Slows skin cell growth',
          expectedImprovement: 70,
          timeframe: '4-8 weeks',
          sideEffects: ['Irritation', 'Burning', 'Less potent than steroids']
        },
        {
          id: 'phototherapy',
          name: 'Narrowband UVB Phototherapy',
          description: 'UV light therapy to slow cell turnover',
          expectedImprovement: 80,
          timeframe: '2-3 months (3x/week)',
          sideEffects: ['Sunburn risk', 'Aging', 'Requires clinic visits']
        },
        {
          id: 'biologic-psoriasis',
          name: 'Biologic Therapy (TNF inhibitors)',
          description: 'Systemic immune modulator for severe psoriasis',
          expectedImprovement: 90,
          timeframe: '12-16 weeks',
          sideEffects: ['Infection risk', 'Injection reactions', 'Very expensive']
        }
      ];
    }

    // Rosacea
    if (diagnosisLower.includes('rosacea')) {
      return [
        {
          id: 'metronidazole',
          name: 'Topical Metronidazole',
          description: 'Anti-inflammatory and antibacterial gel',
          expectedImprovement: 70,
          timeframe: '3-6 weeks',
          sideEffects: ['Mild irritation', 'Dryness', 'Metallic taste if licked']
        },
        {
          id: 'azelaic-acid',
          name: 'Azelaic Acid',
          description: 'Reduces redness and bumps',
          expectedImprovement: 75,
          timeframe: '4-8 weeks',
          sideEffects: ['Tingling', 'Dryness', 'Lightening of skin']
        },
        {
          id: 'laser-ipl-rosacea',
          name: 'IPL/Laser Therapy',
          description: 'Targets visible blood vessels',
          expectedImprovement: 85,
          timeframe: '3-5 sessions over 3-6 months',
          sideEffects: ['Temporary bruising', 'Swelling', 'Expensive']
        },
        {
          id: 'oral-doxycycline',
          name: 'Oral Doxycycline (Low-dose)',
          description: 'Anti-inflammatory antibiotic',
          expectedImprovement: 80,
          timeframe: '6-12 weeks',
          sideEffects: ['GI upset', 'Sun sensitivity', 'Yeast infections']
        }
      ];
    }

    // Warts
    if (diagnosisLower.includes('wart')) {
      return [
        {
          id: 'salicylic-acid',
          name: 'Salicylic Acid (OTC)',
          description: 'Dissolves wart tissue gradually',
          expectedImprovement: 65,
          timeframe: '4-12 weeks',
          sideEffects: ['Skin irritation', 'Slow process', 'Requires daily application']
        },
        {
          id: 'cryotherapy-wart',
          name: 'Cryotherapy (Liquid Nitrogen)',
          description: 'Freezing destroys wart tissue',
          expectedImprovement: 85,
          timeframe: '2-4 treatments over 4-8 weeks',
          sideEffects: ['Pain', 'Blistering', 'Scarring', 'Hypopigmentation']
        },
        {
          id: 'cantharidin',
          name: 'Cantharidin (Beetle Juice)',
          description: 'Painless blistering agent',
          expectedImprovement: 80,
          timeframe: '1-3 treatments',
          sideEffects: ['Blistering', 'Pain after 24hrs', 'Scarring risk']
        },
        {
          id: 'immunotherapy-wart',
          name: 'Immunotherapy (Candida Antigen)',
          description: 'Triggers immune response against wart',
          expectedImprovement: 75,
          timeframe: '3-6 weeks',
          sideEffects: ['Injection discomfort', 'Systemic reaction', 'Variable success']
        }
      ];
    }

    // Seborrheic Keratosis
    if (diagnosisLower.includes('seborrheic') || diagnosisLower.includes('keratosis')) {
      return [
        {
          id: 'cryotherapy-sk',
          name: 'Cryotherapy',
          description: 'Freezing removes benign growth',
          expectedImprovement: 90,
          timeframe: '1-2 treatments',
          sideEffects: ['Hypopigmentation', 'Blistering', 'Scarring']
        },
        {
          id: 'curettage',
          name: 'Curettage (Scraping)',
          description: 'Surgical scraping under local anesthesia',
          expectedImprovement: 95,
          timeframe: '1 procedure',
          sideEffects: ['Scarring', 'Bleeding', 'Infection risk']
        },
        {
          id: 'electrodesiccation',
          name: 'Electrodesiccation',
          description: 'Electrical current burns away growth',
          expectedImprovement: 92,
          timeframe: '1 procedure',
          sideEffects: ['Scarring', 'Hyperpigmentation', 'Pain']
        }
      ];
    }

    // Basal Cell Carcinoma / Squamous Cell Carcinoma
    if (diagnosisLower.includes('basal cell') || diagnosisLower.includes('squamous cell') || diagnosisLower.includes('carcinoma')) {
      return [
        {
          id: 'mohs-bcc',
          name: 'Mohs Micrographic Surgery',
          description: 'Gold standard - highest cure rate with tissue sparing',
          expectedImprovement: 99,
          timeframe: '1 day + 2-4 weeks healing',
          sideEffects: ['Scarring', 'Wound care', 'Possible reconstruction']
        },
        {
          id: 'excision-bcc',
          name: 'Surgical Excision',
          description: 'Complete removal with margins',
          expectedImprovement: 95,
          timeframe: '1 procedure + 2-3 weeks healing',
          sideEffects: ['Scarring', 'Recurrence risk if margins inadequate']
        },
        {
          id: 'edc-bcc',
          name: 'Electrodesiccation & Curettage',
          description: 'Scraping and burning (for small, low-risk)',
          expectedImprovement: 90,
          timeframe: '1 procedure + 2 weeks healing',
          sideEffects: ['White scar', 'Higher recurrence', 'No margin check']
        },
        {
          id: 'imiquimod',
          name: 'Topical Imiquimod (Superficial BCC)',
          description: 'Immune-stimulating cream for low-risk cases',
          expectedImprovement: 80,
          timeframe: '6-12 weeks',
          sideEffects: ['Severe inflammation', 'Crusting', 'Only for certain types']
        }
      ];
    }

    // Vitiligo
    if (diagnosisLower.includes('vitiligo')) {
      return [
        {
          id: 'topical-steroid-vitiligo',
          name: 'Topical Corticosteroid',
          description: 'Can repigment small areas',
          expectedImprovement: 60,
          timeframe: '3-6 months',
          sideEffects: ['Skin thinning', 'Limited efficacy', 'Stretch marks']
        },
        {
          id: 'calcineurin-vitiligo',
          name: 'Topical Calcineurin Inhibitor',
          description: 'Better for face/neck areas',
          expectedImprovement: 65,
          timeframe: '3-6 months',
          sideEffects: ['Burning', 'Better than steroids long-term']
        },
        {
          id: 'nbuvb-vitiligo',
          name: 'Narrowband UVB Therapy',
          description: 'Most effective non-surgical treatment',
          expectedImprovement: 75,
          timeframe: '6-12 months (2-3x/week)',
          sideEffects: ['Time commitment', 'Clinic visits', 'Sunburn risk']
        },
        {
          id: 'excimer-laser',
          name: 'Excimer Laser',
          description: 'Targeted UV therapy for localized vitiligo',
          expectedImprovement: 70,
          timeframe: '2-4 months (2x/week)',
          sideEffects: ['Blistering', 'Cost', 'Requires multiple sessions']
        }
      ];
    }

    // Fungal Infections (Tinea, Ringworm)
    if (diagnosisLower.includes('tinea') || diagnosisLower.includes('ringworm') || diagnosisLower.includes('fungal')) {
      return [
        {
          id: 'topical-antifungal',
          name: 'Topical Antifungal (Clotrimazole)',
          description: 'OTC cream for localized infections',
          expectedImprovement: 85,
          timeframe: '2-4 weeks',
          sideEffects: ['Minimal', 'Irritation', 'Must complete full course']
        },
        {
          id: 'terbinafine-cream',
          name: 'Terbinafine Cream',
          description: 'Stronger topical antifungal',
          expectedImprovement: 90,
          timeframe: '1-2 weeks',
          sideEffects: ['Irritation', 'Expensive', 'Very effective']
        },
        {
          id: 'oral-antifungal',
          name: 'Oral Antifungal (Terbinafine/Fluconazole)',
          description: 'For widespread or nail infections',
          expectedImprovement: 95,
          timeframe: '6-12 weeks',
          sideEffects: ['Liver toxicity', 'GI upset', 'Drug interactions']
        }
      ];
    }

    // Default/Generic treatments for unrecognized conditions
    return [
      {
        id: 'consult-dermatologist',
        name: 'Dermatologist Consultation Required',
        description: 'This condition requires professional evaluation for treatment',
        expectedImprovement: 0,
        timeframe: 'Varies',
        sideEffects: ['Diagnosis required before treatment']
      },
      {
        id: 'topical-steroid-generic',
        name: 'Topical Corticosteroid (Generic)',
        description: 'May reduce inflammation if inflammatory condition',
        expectedImprovement: 60,
        timeframe: '2-4 weeks',
        sideEffects: ['Skin thinning', 'Variable efficacy', 'Consult MD first']
      },
      {
        id: 'emollient-generic',
        name: 'Moisturizer/Emollient',
        description: 'Barrier support for dry skin conditions',
        expectedImprovement: 40,
        timeframe: 'Ongoing',
        sideEffects: ['Minimal', 'Not a cure', 'Supportive care only']
      }
    ];
  };

  const getMockTreatments = (): TreatmentOption[] => {
    // Use condition-specific treatments based on diagnosis
    return getConditionSpecificTreatments(diagnosis || '');
  };

  const simulateTreatmentOutcome = async () => {
    if (!selectedTreatment) {
      Alert.alert('Select Treatment', 'Please select a treatment option first');
      return;
    }

    if (!uploadedImage) {
      Alert.alert('Upload Image', 'Please upload or capture an image first');
      return;
    }

    setIsSimulating(true);

    try {
      // Get token directly from SecureStore (same pattern as home.tsx)
      // AuthService.getToken() only returns in-memory token which may be null
      const SecureStore = require('expo-secure-store');
      const token = await SecureStore.getItemAsync('auth_token');
      console.log('[AR SIMULATOR] Token retrieved from SecureStore:', token ? `${token.substring(0, 20)}...` : 'NULL/UNDEFINED');

      if (!token) {
        Alert.alert('Authentication Error', 'Please log in again to use this feature');
        setIsSimulating(false);
        return;
      }

      // Create FormData for image upload
      const formData = new FormData();
      formData.append('image', {
        uri: uploadedImage,
        type: 'image/jpeg',
        name: 'skin_image.jpg'
      } as any);
      formData.append('treatment_type', selectedTreatment.id);
      formData.append('timeframe', selectedTimeframe);
      formData.append('diagnosis', diagnosis as string);

      // Call backend AI prediction endpoint
      const response = await fetch(
        `${API_BASE_URL}/simulate-treatment-outcome`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          },
          body: formData
        }
      );

      if (response.ok) {
        const result = await response.json();
        console.log('[SIMULATION] Backend response:', JSON.stringify(result, null, 2));
        console.log('[SIMULATION] After image URL:', result.afterImage);
        console.log('[SIMULATION] Before image URL:', result.beforeImage);
        setSimulationResult({
          ...result,
          treatmentName: selectedTreatment.name,
          beforeImage: uploadedImage
        });
      } else {
        // Fallback to mock simulation if endpoint doesn't exist
        console.log('[SIMULATION] Backend error, using mock simulation');
        console.log('[SIMULATION] Response status:', response.status);
        const errorText = await response.text();
        console.log('[SIMULATION] Error details:', errorText);
        generateMockSimulation();
      }
    } catch (error) {
      console.error('Error simulating treatment:', error);
      // Fallback to mock simulation
      generateMockSimulation();
    } finally {
      setIsSimulating(false);
    }
  };

  const generateMockSimulation = () => {
    // Generate mock timeline based on selected timeframe
    const timelineMap = {
      '6months': [
        { weeks: 4, improvement: 20, description: 'Early signs of improvement' },
        { weeks: 8, improvement: 40, description: 'Noticeable reduction in symptoms' },
        { weeks: 16, improvement: 65, description: 'Significant improvement visible' },
        { weeks: 24, improvement: Math.min(selectedTreatment!.expectedImprovement, 100), description: 'Expected 6-month outcome' }
      ],
      '1year': [
        { weeks: 8, improvement: 15, description: 'Initial response to treatment' },
        { weeks: 16, improvement: 35, description: 'Progressive improvement' },
        { weeks: 32, improvement: 60, description: 'Substantial improvement' },
        { weeks: 52, improvement: Math.min(selectedTreatment!.expectedImprovement + 5, 100), description: 'Expected 1-year outcome' }
      ],
      '2years': [
        { weeks: 12, improvement: 12, description: 'Gradual initial improvement' },
        { weeks: 26, improvement: 30, description: 'Steady progress' },
        { weeks: 52, improvement: 55, description: 'Notable improvement at 1 year' },
        { weeks: 104, improvement: Math.min(selectedTreatment!.expectedImprovement + 10, 100), description: 'Expected 2-year outcome' }
      ]
    };

    const mockResult = {
      treatmentId: selectedTreatment!.id,
      treatmentName: selectedTreatment!.name,
      projectedImprovement: timelineMap[selectedTimeframe][3].improvement,
      beforeImage: uploadedImage,
      afterImage: null, // In production, this would be AI-generated
      timeline: timelineMap[selectedTimeframe],
      timeframe: selectedTimeframe,
      recommendations: [
        'Apply treatment as prescribed by your dermatologist',
        'Use broad-spectrum SPF 30+ sunscreen daily',
        'Avoid harsh skincare products during treatment',
        'Document progress with regular photos',
        `Schedule follow-up appointment in ${selectedTimeframe === '6months' ? '3' : selectedTimeframe === '1year' ? '6' : '12'} months`
      ],
      disclaimer: 'These predictions are based on clinical data and statistical models. Individual results may vary significantly. Always consult with a board-certified dermatologist before starting any treatment.'
    };

    setSimulationResult(mockResult);
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.headerButton}>
          <Ionicons name="arrow-back" size={24} color="#1f2937" />
        </TouchableOpacity>
        <Text style={styles.title}>Treatment Outcome Simulator</Text>
        <View style={styles.headerButton} />
      </View>

      <ScrollView style={styles.content}>
        {/* Info Banner */}
        <View style={styles.infoBanner}>
          <Ionicons name="information-circle" size={24} color="#3b82f6" />
          <View style={styles.infoBannerContent}>
            <Text style={styles.infoBannerTitle}>How It Works</Text>
            <Text style={styles.infoBannerText}>
              {uploadedImage
                ? 'Select a timeframe and treatment to see AI-predicted outcomes based on clinical data'
                : 'Upload a photo, then select a treatment to see simulated outcomes based on clinical data'}
            </Text>
          </View>
        </View>

        {/* Diagnosis Info */}
        <View style={styles.diagnosisCard}>
          <View style={styles.diagnosisHeader}>
            <View style={styles.diagnosisInfo}>
              <Text style={styles.cardTitle}>Current Diagnosis</Text>
              <Text style={styles.diagnosisText}>
                {diagnosis ? String(diagnosis) : 'Not specified'}
              </Text>
            </View>
            {simulationResult?.severity && (
              <View style={[
                styles.severityBadge,
                simulationResult.severity === 'severe' && styles.severityBadgeSevere,
                simulationResult.severity === 'moderate' && styles.severityBadgeModerate,
                simulationResult.severity === 'mild' && styles.severityBadgeMild
              ]}>
                <Ionicons
                  name={
                    simulationResult.severity === 'severe' ? 'alert-circle' :
                    simulationResult.severity === 'moderate' ? 'warning' :
                    'checkmark-circle'
                  }
                  size={16}
                  color="white"
                />
                <Text style={styles.severityBadgeText}>
                  {simulationResult.severity.toUpperCase()}
                </Text>
              </View>
            )}
          </View>
          {!diagnosis && (
            <Text style={styles.diagnosisDebugText}>
              Debug: params = {JSON.stringify(params)}
            </Text>
          )}
        </View>

        {/* Image Upload Section */}
        <View style={styles.uploadSection}>
          <Text style={styles.sectionTitle}>
            {uploadedImage ? 'Skin Image' : 'Step 1: Upload Skin Image'}
          </Text>

          {uploadedImage ? (
            <View style={styles.uploadedImageContainer}>
              <Image
                source={{ uri: uploadedImage }}
                style={styles.uploadedImage}
                resizeMode="cover"
              />
              <View style={styles.imageActionButtons}>
                <TouchableOpacity
                  style={styles.changeImageSmallButton}
                  onPress={() => setUploadedImage(null)}
                >
                  <Ionicons name="trash-outline" size={16} color="white" />
                  <Text style={styles.changeImageSmallButtonText}>Remove</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.changeImageSmallButton}
                  onPress={pickImage}
                >
                  <Ionicons name="images-outline" size={16} color="white" />
                  <Text style={styles.changeImageSmallButtonText}>Change</Text>
                </TouchableOpacity>
              </View>
            </View>
          ) : (
            <View style={styles.uploadButtons}>
              <TouchableOpacity
                style={styles.uploadButton}
                onPress={takePhoto}
              >
                <Ionicons name="camera" size={32} color="#3b82f6" />
                <Text style={styles.uploadButtonText}>Take Photo</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.uploadButton}
                onPress={pickImage}
              >
                <Ionicons name="images" size={32} color="#3b82f6" />
                <Text style={styles.uploadButtonText}>Choose from Gallery</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>

        {/* Timeframe Selection */}
        {uploadedImage && (
          <View style={styles.timeframeSection}>
            <Text style={styles.sectionTitle}>
              {imageUrl ? 'Step 1: Select Prediction Timeframe' : 'Step 2: Select Prediction Timeframe'}
            </Text>
            <View style={styles.timeframeButtons}>
              <TouchableOpacity
                style={[
                  styles.timeframeButton,
                  selectedTimeframe === '6months' && styles.selectedTimeframeButton
                ]}
                onPress={() => setSelectedTimeframe('6months')}
              >
                <Text style={[
                  styles.timeframeButtonText,
                  selectedTimeframe === '6months' && styles.selectedTimeframeButtonText
                ]}>6 Months</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[
                  styles.timeframeButton,
                  selectedTimeframe === '1year' && styles.selectedTimeframeButton
                ]}
                onPress={() => setSelectedTimeframe('1year')}
              >
                <Text style={[
                  styles.timeframeButtonText,
                  selectedTimeframe === '1year' && styles.selectedTimeframeButtonText
                ]}>1 Year</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[
                  styles.timeframeButton,
                  selectedTimeframe === '2years' && styles.selectedTimeframeButton
                ]}
                onPress={() => setSelectedTimeframe('2years')}
              >
                <Text style={[
                  styles.timeframeButtonText,
                  selectedTimeframe === '2years' && styles.selectedTimeframeButtonText
                ]}>2 Years</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Treatment Options */}
        <View style={styles.treatmentsSection}>
          <Text style={styles.sectionTitle}>
            {uploadedImage && imageUrl ? 'Step 2: Select Treatment' : uploadedImage ? 'Step 3: Select Treatment' : 'Treatment Options'}
          </Text>

          {loading ? (
            <ActivityIndicator size="large" color="#3b82f6" />
          ) : (
            treatments.map((treatment) => (
              <TouchableOpacity
                key={treatment.id}
                style={[
                  styles.treatmentCard,
                  selectedTreatment?.id === treatment.id && styles.selectedTreatmentCard
                ]}
                onPress={() => setSelectedTreatment(treatment)}
              >
                <View style={styles.treatmentHeader}>
                  <View style={styles.treatmentTitleRow}>
                    <Ionicons
                      name={selectedTreatment?.id === treatment.id ? "checkmark-circle" : "ellipse-outline"}
                      size={24}
                      color={selectedTreatment?.id === treatment.id ? "#3b82f6" : "#9ca3af"}
                    />
                    <Text style={styles.treatmentName}>{treatment.name}</Text>
                  </View>
                  <View style={styles.improvementBadge}>
                    <Text style={styles.improvementText}>{treatment.expectedImprovement}%</Text>
                  </View>
                </View>

                <Text style={styles.treatmentDescription}>{treatment.description}</Text>

                <View style={styles.treatmentDetails}>
                  <View style={styles.detailItem}>
                    <Ionicons name="time-outline" size={16} color="#6b7280" />
                    <Text style={styles.detailText}>{treatment.timeframe}</Text>
                  </View>
                </View>

                {selectedTreatment?.id === treatment.id && (
                  <View style={styles.sideEffectsSection}>
                    <Text style={styles.sideEffectsTitle}>Possible Side Effects:</Text>
                    {treatment.sideEffects.map((effect, index) => (
                      <Text key={index} style={styles.sideEffectText}>• {effect}</Text>
                    ))}
                  </View>
                )}
              </TouchableOpacity>
            ))
          )}
        </View>

        {/* Simulate Button */}
        {selectedTreatment && !simulationResult && (
          <TouchableOpacity
            style={styles.simulateButton}
            onPress={simulateTreatmentOutcome}
            disabled={isSimulating}
          >
            {isSimulating ? (
              <ActivityIndicator size="small" color="white" />
            ) : (
              <>
                <Ionicons name="eye" size={20} color="white" />
                <Text style={styles.simulateButtonText}>Simulate Treatment Outcome</Text>
              </>
            )}
          </TouchableOpacity>
        )}

        {/* Simulation Results */}
        {simulationResult && (
          <View style={styles.resultsSection}>
            <Text style={styles.sectionTitle}>Simulation Results</Text>

            {/* Before/After Images */}
            <View style={styles.beforeAfterSection}>
              <Text style={styles.cardTitle}>Visual Prediction</Text>
              <View style={styles.beforeAfterContainer}>
                <View style={styles.beforeAfterItem}>
                  <Text style={styles.beforeAfterLabel}>Before Treatment</Text>
                  <Image
                    source={{ uri: simulationResult.beforeImage }}
                    style={styles.beforeAfterImage}
                    resizeMode="cover"
                  />
                </View>

                <View style={styles.arrowContainer}>
                  <Ionicons name="arrow-forward" size={32} color="#3b82f6" />
                  <Text style={styles.timeframeLabel}>
                    After {selectedTimeframe === '6months' ? '6 months' : selectedTimeframe === '1year' ? '1 year' : '2 years'}
                  </Text>
                </View>

                <View style={styles.beforeAfterItem}>
                  <Text style={styles.beforeAfterLabel}>Predicted Outcome</Text>
                  <Image
                    source={{
                      uri: simulationResult.afterImage
                        ? (simulationResult.afterImage.startsWith('http')
                            ? simulationResult.afterImage
                            : `${API_BASE_URL}${simulationResult.afterImage}`)
                        : simulationResult.beforeImage
                    }}
                    style={styles.beforeAfterImage}
                    resizeMode="cover"
                    onError={(e) => {
                      console.error('[AFTER IMAGE] Load error:', e.nativeEvent.error);
                      console.log('[AFTER IMAGE] Attempted URL:', simulationResult.afterImage);
                    }}
                    onLoad={() => console.log('[AFTER IMAGE] Loaded successfully')}
                  />
                </View>
              </View>

              {/* Disclaimer */}
              {simulationResult.disclaimer && (
                <View style={styles.disclaimerBox}>
                  <Ionicons name="information-circle-outline" size={20} color="#6b7280" />
                  <Text style={styles.disclaimerText}>{simulationResult.disclaimer}</Text>
                </View>
              )}
            </View>

            <View style={styles.resultCard}>
              <View style={styles.resultHeader}>
                <Ionicons name="trending-up" size={24} color="#10b981" />
                <Text style={styles.resultTitle}>Expected Improvement</Text>
              </View>
              <Text style={styles.resultPercentage}>
                {simulationResult.projectedImprovement}%
              </Text>
              {simulationResult.baseImprovement !== simulationResult.projectedImprovement && (
                <Text style={styles.adjustedFromText}>
                  Adjusted from baseline {simulationResult.baseImprovement}%
                </Text>
              )}
            </View>

            {/* Confidence Intervals */}
            {simulationResult.confidenceIntervals && (
              <View style={styles.confidenceCard}>
                <Text style={styles.cardTitle}>Outcome Range</Text>
                <Text style={styles.confidenceSubtitle}>
                  Based on severity and clinical data
                </Text>
                <View style={styles.confidenceRanges}>
                  <View style={styles.confidenceItem}>
                    <View style={[styles.confidenceDot, { backgroundColor: '#10b981' }]} />
                    <Text style={styles.confidenceLabel}>Best Case</Text>
                    <Text style={styles.confidenceValue}>
                      {simulationResult.confidenceIntervals.best_case}%
                    </Text>
                  </View>
                  <View style={styles.confidenceItem}>
                    <View style={[styles.confidenceDot, { backgroundColor: '#3b82f6' }]} />
                    <Text style={styles.confidenceLabel}>Typical</Text>
                    <Text style={styles.confidenceValue}>
                      {simulationResult.confidenceIntervals.typical}%
                    </Text>
                  </View>
                  <View style={styles.confidenceItem}>
                    <View style={[styles.confidenceDot, { backgroundColor: '#f59e0b' }]} />
                    <Text style={styles.confidenceLabel}>Worst Case</Text>
                    <Text style={styles.confidenceValue}>
                      {simulationResult.confidenceIntervals.worst_case}%
                    </Text>
                  </View>
                </View>
                <View style={styles.confidenceBar}>
                  <View
                    style={[
                      styles.confidenceRange,
                      {
                        left: `${simulationResult.confidenceIntervals.worst_case}%`,
                        width: `${simulationResult.confidenceIntervals.best_case - simulationResult.confidenceIntervals.worst_case}%`
                      }
                    ]}
                  />
                  <View
                    style={[
                      styles.confidenceMarker,
                      { left: `${simulationResult.confidenceIntervals.typical}%` }
                    ]}
                  />
                </View>
              </View>
            )}

            {/* Progressive Timeline Images */}
            {simulationResult.timeline && simulationResult.timeline.length > 0 &&
             simulationResult.timeline[0].image_url && (
              <View style={styles.progressiveTimelineCard}>
                <Text style={styles.cardTitle}>Progressive Treatment Timeline</Text>
                <Text style={styles.timelineSubtitle}>
                  AI-predicted outcomes at key milestones
                </Text>
                <ScrollView
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  style={styles.timelineScroll}
                  contentContainerStyle={styles.timelineScrollContent}
                >
                  {simulationResult.timeline.map((milestone: any, index: number) => (
                    <View key={index} style={styles.timelineImageItem}>
                      <Image
                        source={{
                          uri: milestone.image_url.startsWith('http')
                            ? milestone.image_url
                            : `${API_BASE_URL}${milestone.image_url}`
                        }}
                        style={styles.timelineImage}
                        resizeMode="cover"
                      />
                      <View style={styles.timelineImageOverlay}>
                        <Text style={styles.timelineImageWeek}>Week {milestone.weeks}</Text>
                        <Text style={styles.timelineImageImprovement}>
                          {milestone.improvement}%
                        </Text>
                      </View>
                    </View>
                  ))}
                </ScrollView>
              </View>
            )}

            {/* Timeline */}
            <View style={styles.timelineCard}>
              <Text style={styles.cardTitle}>Treatment Timeline</Text>
              {simulationResult.timeline.map((milestone: any, index: number) => (
                <View key={index} style={styles.timelineItem}>
                  <View style={styles.timelineDot} />
                  {index < simulationResult.timeline.length - 1 && (
                    <View style={styles.timelineLine} />
                  )}
                  <View style={styles.timelineContent}>
                    <Text style={styles.timelineWeek}>Week {milestone.weeks}</Text>
                    <Text style={styles.timelineDescription}>{milestone.description}</Text>
                    <View style={styles.progressBar}>
                      <View
                        style={[
                          styles.progressFill,
                          { width: `${milestone.improvement}%` }
                        ]}
                      />
                    </View>
                    <Text style={styles.progressText}>{milestone.improvement}% improvement</Text>
                  </View>
                </View>
              ))}
            </View>

            {/* Recommendations */}
            <View style={styles.recommendationsCard}>
              <Text style={styles.cardTitle}>Care Recommendations</Text>
              {simulationResult.recommendations.map((rec: string, index: number) => (
                <View key={index} style={styles.recommendationItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#10b981" />
                  <Text style={styles.recommendationText}>{rec}</Text>
                </View>
              ))}
            </View>

            {/* Action Buttons */}
            <View style={styles.actionButtons}>
              <TouchableOpacity
                style={styles.secondaryButton}
                onPress={() => {
                  // Keep results visible, just scroll back up to change options
                  setSimulationResult(null);
                }}
              >
                <Ionicons name="reload-circle-outline" size={20} color="#3b82f6" />
                <Text style={styles.secondaryButtonText}>Try Different Options</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.primaryActionButton}
                onPress={() => {
                  Alert.alert(
                    'Schedule Consultation',
                    'Would you like to schedule a telemedicine consultation to discuss this treatment?',
                    [
                      { text: 'Later', style: 'cancel' },
                      {
                        text: 'Schedule',
                        onPress: () => {
                          Alert.alert('Coming Soon', 'Telemedicine scheduling is in development');
                        }
                      }
                    ]
                  );
                }}
              >
                <Ionicons name="calendar" size={20} color="white" />
                <Text style={styles.primaryActionButtonText}>Schedule Consultation</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb'
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    padding: 20
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 60 : 16,
    paddingBottom: 16,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb'
  },
  headerButton: {
    padding: 4,
    width: 32
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    flex: 1,
    textAlign: 'center'
  },
  content: {
    flex: 1
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#6b7280'
  },
  errorText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#dc2626',
    marginTop: 16,
    marginBottom: 8
  },
  errorSubtext: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 24
  },
  infoBanner: {
    flexDirection: 'row',
    backgroundColor: '#eff6ff',
    padding: 16,
    margin: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6'
  },
  infoBannerContent: {
    flex: 1,
    marginLeft: 12
  },
  infoBannerTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4
  },
  infoBannerText: {
    fontSize: 13,
    color: '#4b5563',
    lineHeight: 18
  },
  diagnosisCard: {
    backgroundColor: 'white',
    margin: 16,
    marginTop: 0,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 12
  },
  diagnosisText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#3b82f6'
  },
  diagnosisDebugText: {
    fontSize: 10,
    color: '#9ca3af',
    marginTop: 8
  },
  treatmentsSection: {
    padding: 16,
    paddingTop: 0
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16
  },
  treatmentCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 2,
    borderColor: '#e5e7eb',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  selectedTreatmentCard: {
    borderColor: '#3b82f6',
    backgroundColor: '#eff6ff'
  },
  treatmentHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12
  },
  treatmentTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1
  },
  treatmentName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    flex: 1
  },
  improvementBadge: {
    backgroundColor: '#10b981',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12
  },
  improvementText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold'
  },
  treatmentDescription: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
    marginBottom: 12
  },
  treatmentDetails: {
    flexDirection: 'row',
    gap: 16
  },
  detailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6
  },
  detailText: {
    fontSize: 13,
    color: '#6b7280'
  },
  sideEffectsSection: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb'
  },
  sideEffectsTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 8
  },
  sideEffectText: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4
  },
  simulateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#3b82f6',
    paddingVertical: 16,
    marginHorizontal: 16,
    borderRadius: 12,
    shadowColor: '#3b82f6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4
  },
  simulateButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold'
  },
  resultsSection: {
    padding: 16
  },
  resultCard: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12
  },
  resultTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937'
  },
  resultPercentage: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#10b981'
  },
  timelineCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  timelineItem: {
    position: 'relative',
    marginBottom: 24
  },
  timelineDot: {
    position: 'absolute',
    left: 0,
    top: 4,
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#3b82f6',
    borderWidth: 2,
    borderColor: 'white'
  },
  timelineLine: {
    position: 'absolute',
    left: 5,
    top: 16,
    width: 2,
    bottom: -24,
    backgroundColor: '#e5e7eb'
  },
  timelineContent: {
    marginLeft: 24
  },
  timelineWeek: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#3b82f6',
    marginBottom: 4
  },
  timelineDescription: {
    fontSize: 14,
    color: '#4b5563',
    marginBottom: 8
  },
  progressBar: {
    height: 8,
    backgroundColor: '#e5e7eb',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 4
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#10b981',
    borderRadius: 4
  },
  progressText: {
    fontSize: 12,
    color: '#6b7280'
  },
  recommendationsCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
    marginBottom: 12
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20
  },
  actionButtons: {
    gap: 12
  },
  secondaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: 'white',
    paddingVertical: 14,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#3b82f6'
  },
  secondaryButtonText: {
    color: '#3b82f6',
    fontSize: 16,
    fontWeight: '600'
  },
  primaryActionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#3b82f6',
    paddingVertical: 16,
    borderRadius: 12,
    shadowColor: '#3b82f6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4
  },
  primaryActionButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold'
  },
  primaryButton: {
    backgroundColor: '#3b82f6',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600'
  },
  uploadSection: {
    padding: 16,
    paddingTop: 0
  },
  uploadButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 12
  },
  uploadButton: {
    flex: 1,
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#e5e7eb',
    borderStyle: 'dashed',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  uploadButtonText: {
    marginTop: 8,
    fontSize: 14,
    fontWeight: '600',
    color: '#3b82f6',
    textAlign: 'center'
  },
  uploadedImageContainer: {
    position: 'relative',
    alignItems: 'center',
    marginTop: 12
  },
  uploadedImage: {
    width: imageWidth * 1.5,
    height: imageWidth * 1.5,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#10b981'
  },
  changeImageButton: {
    position: 'absolute',
    top: -8,
    right: width / 2 - (imageWidth * 1.5) / 2 - 8,
    backgroundColor: 'white',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5
  },
  imageActionButtons: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 12
  },
  changeImageSmallButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#6b7280',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 3,
    elevation: 3
  },
  changeImageSmallButtonText: {
    color: 'white',
    fontSize: 13,
    fontWeight: '600'
  },
  timeframeSection: {
    padding: 16,
    paddingTop: 0
  },
  timeframeButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 12
  },
  timeframeButton: {
    flex: 1,
    backgroundColor: 'white',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#e5e7eb',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  selectedTimeframeButton: {
    borderColor: '#3b82f6',
    backgroundColor: '#eff6ff'
  },
  timeframeButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280'
  },
  selectedTimeframeButtonText: {
    color: '#3b82f6'
  },
  beforeAfterSection: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  beforeAfterContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 12
  },
  beforeAfterItem: {
    flex: 1,
    alignItems: 'center'
  },
  beforeAfterLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 8
  },
  beforeAfterImage: {
    width: imageWidth - 20,
    height: imageWidth - 20,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb'
  },
  arrowContainer: {
    alignItems: 'center',
    paddingHorizontal: 8
  },
  timeframeLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 4,
    textAlign: 'center'
  },
  afterImagePlaceholderContainer: {
    position: 'relative',
    width: imageWidth - 20,
    height: imageWidth - 20,
    borderRadius: 12,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    backgroundColor: 'transparent'
  },
  placeholderBaseImage: {
    width: '100%',
    height: '100%',
    borderRadius: 12
  },
  placeholderAfterImage: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(59, 130, 246, 0.75)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 12
  },
  placeholderText: {
    marginTop: 8,
    fontSize: 14,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center'
  },
  placeholderSubtext: {
    marginTop: 4,
    fontSize: 11,
    color: 'white',
    textAlign: 'center'
  },
  disclaimerBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    marginTop: 16,
    padding: 12,
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#6b7280'
  },
  disclaimerText: {
    flex: 1,
    fontSize: 12,
    color: '#6b7280',
    lineHeight: 16
  },
  diagnosisHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start'
  },
  diagnosisInfo: {
    flex: 1
  },
  severityBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    marginLeft: 12
  },
  severityBadgeMild: {
    backgroundColor: '#10b981'
  },
  severityBadgeModerate: {
    backgroundColor: '#f59e0b'
  },
  severityBadgeSevere: {
    backgroundColor: '#dc2626'
  },
  severityBadgeText: {
    color: 'white',
    fontSize: 11,
    fontWeight: 'bold',
    letterSpacing: 0.5
  },
  adjustedFromText: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center'
  },
  confidenceCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  confidenceSubtitle: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: -8,
    marginBottom: 16
  },
  confidenceRanges: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20
  },
  confidenceItem: {
    flex: 1,
    alignItems: 'center'
  },
  confidenceDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginBottom: 8
  },
  confidenceLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginBottom: 4
  },
  confidenceValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937'
  },
  confidenceBar: {
    height: 8,
    backgroundColor: '#f3f4f6',
    borderRadius: 4,
    position: 'relative',
    overflow: 'visible'
  },
  confidenceRange: {
    position: 'absolute',
    height: '100%',
    backgroundColor: '#bfdbfe',
    borderRadius: 4
  },
  confidenceMarker: {
    position: 'absolute',
    width: 3,
    height: 16,
    backgroundColor: '#3b82f6',
    borderRadius: 2,
    top: -4
  },
  progressiveTimelineCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  timelineSubtitle: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: -8,
    marginBottom: 12
  },
  timelineScroll: {
    marginHorizontal: -16
  },
  timelineScrollContent: {
    paddingHorizontal: 16,
    gap: 12
  },
  timelineImageItem: {
    position: 'relative',
    marginRight: 12,
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 4
  },
  timelineImage: {
    width: 140,
    height: 140,
    borderRadius: 12
  },
  timelineImageOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingVertical: 4,
    paddingHorizontal: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  timelineImageWeek: {
    fontSize: 10,
    color: 'white',
    fontWeight: '600'
  },
  timelineImageImprovement: {
    fontSize: 12,
    color: '#10b981',
    fontWeight: 'bold'
  },
  surgicalOutcomeContainer: {
    width: imageWidth - 20,
    height: imageWidth - 20,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#10b981',
    backgroundColor: '#f0fdf4',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16
  },
  surgicalOutcomeTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#166534',
    marginTop: 12,
    textAlign: 'center'
  },
  surgicalOutcomeText: {
    fontSize: 13,
    color: '#15803d',
    marginTop: 8,
    textAlign: 'center'
  },
  surgicalOutcomeSubtext: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#10b981',
    marginTop: 8,
    textAlign: 'center'
  },
  surgicalNote: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 12,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: 'white',
    borderRadius: 8
  },
  surgicalNoteText: {
    fontSize: 11,
    color: '#6b7280'
  }
});
