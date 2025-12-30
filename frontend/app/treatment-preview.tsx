import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  Dimensions,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  FadeIn,
  FadeInDown,
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  interpolate,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import Slider from '@react-native-community/slider';
import { API_BASE_URL } from '../config';

const { width } = Dimensions.get('window');

interface Treatment {
  id: string;
  name: string;
  description: string;
  timeline: string;
  icon: string;
  color: string;
}

interface Improvement {
  area: string;
  change: string;
  confidence: number;
}

interface PreviewResult {
  original_image: string;
  preview_image: string;
  treatment_type: string;
  treatment_name: string;
  description: string;
  timeline: string;
  expected_improvements: Improvement[];
  confidence: number;
  tips: string[];
  disclaimer: string;
}

const TREATMENTS: Treatment[] = [
  {
    id: 'lesion_removal',
    name: 'Lesion Removal',
    description: 'See how your skin would look after removing a spot',
    timeline: '4-6 weeks healing',
    icon: 'cut',
    color: '#EF4444',
  },
  {
    id: 'sunscreen_use',
    name: 'Sunscreen Protection',
    description: 'Effects of 6 months daily SPF 50 use',
    timeline: '6 months',
    icon: 'sunny',
    color: '#F59E0B',
  },
  {
    id: 'retinol',
    name: 'Retinol Treatment',
    description: 'Anti-aging and texture improvement',
    timeline: '3-6 months',
    icon: 'flask',
    color: '#8B5CF6',
  },
  {
    id: 'vitamin_c',
    name: 'Vitamin C Serum',
    description: 'Brightening and evening skin tone',
    timeline: '2-3 months',
    icon: 'nutrition',
    color: '#F97316',
  },
  {
    id: 'hydration',
    name: 'Intensive Hydration',
    description: 'Deep moisture and plumpness',
    timeline: '2-4 weeks',
    icon: 'water',
    color: '#06B6D4',
  },
  {
    id: 'anti_aging',
    name: 'Anti-Aging Combo',
    description: 'Combined treatment for youthful skin',
    timeline: '6-12 months',
    icon: 'hourglass',
    color: '#EC4899',
  },
  {
    id: 'pigmentation_treatment',
    name: 'Pigmentation Treatment',
    description: 'Reduce dark spots and even skin tone',
    timeline: '3-6 months',
    icon: 'color-fill',
    color: '#84CC16',
  },
  {
    id: 'acne_treatment',
    name: 'Acne Treatment',
    description: 'Clear breakouts and reduce scarring risk',
    timeline: '2-3 months',
    icon: 'medical',
    color: '#14B8A6',
  },
];

export default function TreatmentPreviewScreen() {
  const router = useRouter();
  const [image, setImage] = useState<string | null>(null);
  const [selectedTreatment, setSelectedTreatment] = useState<Treatment | null>(null);
  const [intensity, setIntensity] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PreviewResult | null>(null);
  const [showBefore, setShowBefore] = useState(false);

  // Slider animation for before/after comparison
  const sliderPosition = useSharedValue(0);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please allow access to your photos');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setImage(result.assets[0].uri);
      setResult(null);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please allow access to your camera');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setImage(result.assets[0].uri);
      setResult(null);
    }
  };

  const generatePreview = async () => {
    if (!image || !selectedTreatment) return;

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('image', {
        uri: image,
        type: 'image/jpeg',
        name: 'skin_photo.jpg',
      } as any);
      formData.append('treatment', selectedTreatment.id);
      formData.append('intensity', intensity.toString());

      const response = await fetch(`${API_BASE_URL}/api/treatment-preview/generate`, {
        method: 'POST',
        body: formData,
        // Note: Don't set Content-Type header - fetch will set it automatically with the correct boundary
      });

      if (!response.ok) {
        throw new Error('Preview generation failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Preview error:', error);
      Alert.alert('Error', 'Failed to generate preview. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const renderTreatmentSelector = () => (
    <Animated.View entering={FadeInDown.duration(500)} style={styles.treatmentSection}>
      <Text style={styles.sectionTitle}>Choose a Treatment</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.treatmentScroll}>
        {TREATMENTS.map((treatment) => (
          <TouchableOpacity
            key={treatment.id}
            style={[
              styles.treatmentCard,
              selectedTreatment?.id === treatment.id && styles.treatmentCardSelected,
              { borderColor: treatment.color },
            ]}
            onPress={() => setSelectedTreatment(treatment)}
          >
            <View style={[styles.treatmentIcon, { backgroundColor: treatment.color }]}>
              <Ionicons name={treatment.icon as any} size={24} color="#FFF" />
            </View>
            <Text style={styles.treatmentName}>{treatment.name}</Text>
            <Text style={styles.treatmentTimeline}>{treatment.timeline}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {selectedTreatment && (
        <View style={styles.treatmentDetails}>
          <Text style={styles.treatmentDesc}>{selectedTreatment.description}</Text>

          <View style={styles.intensityContainer}>
            <Text style={styles.intensityLabel}>Effect Intensity</Text>
            <Slider
              style={styles.slider}
              minimumValue={0.1}
              maximumValue={1.0}
              value={intensity}
              onValueChange={setIntensity}
              minimumTrackTintColor={selectedTreatment.color}
              maximumTrackTintColor="#E5E7EB"
              thumbTintColor={selectedTreatment.color}
            />
            <View style={styles.intensityLabels}>
              <Text style={styles.intensityMin}>Subtle</Text>
              <Text style={styles.intensityMax}>Dramatic</Text>
            </View>
          </View>
        </View>
      )}
    </Animated.View>
  );

  const renderResult = () => {
    if (!result) return null;

    return (
      <Animated.View entering={FadeIn.duration(600)} style={styles.resultSection}>
        {/* Before/After Comparison */}
        <View style={styles.comparisonContainer}>
          <Text style={styles.comparisonTitle}>Before & After</Text>

          <View style={styles.imageComparison}>
            <TouchableOpacity
              style={styles.comparisonImageContainer}
              onPressIn={() => setShowBefore(true)}
              onPressOut={() => setShowBefore(false)}
            >
              <Image
                source={{ uri: showBefore
                  ? `data:image/jpeg;base64,${result.original_image}`
                  : `data:image/jpeg;base64,${result.preview_image}`
                }}
                style={styles.comparisonImage}
              />
              <View style={styles.imageLabelContainer}>
                <Text style={styles.imageLabel}>{showBefore ? 'Before' : 'After'}</Text>
              </View>
            </TouchableOpacity>
            <Text style={styles.holdHint}>Hold to see original</Text>
          </View>

          {/* Timeline badge */}
          <View style={[styles.timelineBadge, { backgroundColor: selectedTreatment?.color }]}>
            <Ionicons name="time" size={16} color="#FFF" />
            <Text style={styles.timelineText}>Expected in: {result.timeline}</Text>
          </View>
        </View>

        {/* Expected Improvements */}
        <View style={styles.improvementsSection}>
          <Text style={styles.sectionTitle}>Expected Improvements</Text>
          {result.expected_improvements.map((imp, index) => (
            <View key={index} style={styles.improvementRow}>
              <View style={styles.improvementInfo}>
                <Text style={styles.improvementArea}>{imp.area}</Text>
                <Text style={[styles.improvementChange, { color: selectedTreatment?.color }]}>
                  {imp.change}
                </Text>
              </View>
              <View style={styles.confidenceContainer}>
                <View style={styles.confidenceBarBg}>
                  <View
                    style={[
                      styles.confidenceBarFill,
                      { width: `${imp.confidence * 100}%`, backgroundColor: selectedTreatment?.color },
                    ]}
                  />
                </View>
                <Text style={styles.confidenceText}>{Math.round(imp.confidence * 100)}%</Text>
              </View>
            </View>
          ))}
        </View>

        {/* Tips */}
        <View style={styles.tipsSection}>
          <Text style={styles.sectionTitle}>Pro Tips</Text>
          {result.tips.map((tip, index) => (
            <View key={index} style={styles.tipRow}>
              <Ionicons name="checkmark-circle" size={20} color="#22C55E" />
              <Text style={styles.tipText}>{tip}</Text>
            </View>
          ))}
        </View>

        {/* Disclaimer */}
        <View style={styles.disclaimerBox}>
          <Ionicons name="information-circle" size={20} color="#6B7280" />
          <Text style={styles.disclaimerText}>{result.disclaimer}</Text>
        </View>

        {/* Try Another */}
        <TouchableOpacity
          style={styles.tryAnotherButton}
          onPress={() => {
            setResult(null);
            setSelectedTreatment(null);
          }}
        >
          <Ionicons name="refresh" size={24} color="#8B5CF6" />
          <Text style={styles.tryAnotherText}>Try Another Treatment</Text>
        </TouchableOpacity>
      </Animated.View>
    );
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <LinearGradient colors={['#EC4899', '#8B5CF6']} style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#FFF" />
        </TouchableOpacity>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>Treatment Preview</Text>
          <Text style={styles.headerSubtitle}>See your "what if" results</Text>
        </View>
        <View style={{ width: 40 }} />
      </LinearGradient>

      <View style={styles.content}>
        {/* Image Selection */}
        {!result && (
          <View style={styles.imageSection}>
            {image ? (
              <View style={styles.imagePreviewContainer}>
                <Image source={{ uri: image }} style={styles.imagePreview} />
                <TouchableOpacity style={styles.changeImageBtn} onPress={pickImage}>
                  <Ionicons name="camera" size={20} color="#FFF" />
                </TouchableOpacity>
              </View>
            ) : (
              <TouchableOpacity style={styles.imagePlaceholder} onPress={pickImage}>
                <Ionicons name="image" size={48} color="#9CA3AF" />
                <Text style={styles.placeholderText}>Add a photo of your skin</Text>
                <Text style={styles.placeholderHint}>For best results, use a clear, well-lit photo</Text>
              </TouchableOpacity>
            )}

            {!image && (
              <View style={styles.buttonRow}>
                <TouchableOpacity style={styles.imageButton} onPress={takePhoto}>
                  <Ionicons name="camera" size={24} color="#8B5CF6" />
                  <Text style={styles.imageButtonText}>Camera</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.imageButton} onPress={pickImage}>
                  <Ionicons name="images" size={24} color="#8B5CF6" />
                  <Text style={styles.imageButtonText}>Gallery</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        )}

        {/* Treatment Selection */}
        {image && !result && renderTreatmentSelector()}

        {/* Generate Button */}
        {image && selectedTreatment && !result && (
          <TouchableOpacity
            style={[
              styles.generateButton,
              { backgroundColor: selectedTreatment.color },
              loading && styles.generateButtonDisabled,
            ]}
            onPress={generatePreview}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#FFF" />
            ) : (
              <>
                <Ionicons name="sparkles" size={24} color="#FFF" />
                <Text style={styles.generateButtonText}>Generate Preview</Text>
              </>
            )}
          </TouchableOpacity>
        )}

        {/* Result */}
        {result && renderResult()}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: 60,
    paddingBottom: 30,
    paddingHorizontal: 20,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerContent: {
    flex: 1,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#FFF',
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 4,
  },
  content: {
    padding: 20,
  },
  imageSection: {
    alignItems: 'center',
    marginBottom: 24,
  },
  imagePreviewContainer: {
    position: 'relative',
  },
  imagePreview: {
    width: width - 80,
    height: width - 80,
    borderRadius: 20,
  },
  changeImageBtn: {
    position: 'absolute',
    bottom: 10,
    right: 10,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.6)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  imagePlaceholder: {
    width: width - 80,
    height: 200,
    borderRadius: 20,
    backgroundColor: '#E5E7EB',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: '#D1D5DB',
    borderStyle: 'dashed',
  },
  placeholderText: {
    marginTop: 12,
    color: '#6B7280',
    fontSize: 16,
    fontWeight: '600',
  },
  placeholderHint: {
    marginTop: 4,
    color: '#9CA3AF',
    fontSize: 13,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 16,
    marginTop: 20,
  },
  imageButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
    backgroundColor: '#EDE9FE',
  },
  imageButtonText: {
    color: '#8B5CF6',
    fontWeight: '600',
  },
  treatmentSection: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 16,
  },
  treatmentScroll: {
    marginHorizontal: -20,
    paddingHorizontal: 20,
  },
  treatmentCard: {
    width: 120,
    padding: 16,
    marginRight: 12,
    borderRadius: 16,
    backgroundColor: '#FFF',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  treatmentCardSelected: {
    backgroundColor: '#FFF',
  },
  treatmentIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10,
  },
  treatmentName: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1F2937',
    textAlign: 'center',
    marginBottom: 4,
  },
  treatmentTimeline: {
    fontSize: 11,
    color: '#6B7280',
  },
  treatmentDetails: {
    marginTop: 20,
    padding: 16,
    backgroundColor: '#FFF',
    borderRadius: 16,
  },
  treatmentDesc: {
    fontSize: 14,
    color: '#374151',
    marginBottom: 16,
  },
  intensityContainer: {
    marginTop: 8,
  },
  intensityLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 8,
  },
  slider: {
    width: '100%',
    height: 40,
  },
  intensityLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  intensityMin: {
    fontSize: 12,
    color: '#6B7280',
  },
  intensityMax: {
    fontSize: 12,
    color: '#6B7280',
  },
  generateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 16,
    borderRadius: 16,
    marginBottom: 20,
  },
  generateButtonDisabled: {
    opacity: 0.7,
  },
  generateButtonText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  resultSection: {
    marginTop: 10,
  },
  comparisonContainer: {
    marginBottom: 24,
  },
  comparisonTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 16,
    textAlign: 'center',
  },
  imageComparison: {
    alignItems: 'center',
  },
  comparisonImageContainer: {
    position: 'relative',
  },
  comparisonImage: {
    width: width - 60,
    height: width - 60,
    borderRadius: 20,
  },
  imageLabelContainer: {
    position: 'absolute',
    top: 16,
    left: 16,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  imageLabel: {
    color: '#FFF',
    fontWeight: '600',
  },
  holdHint: {
    marginTop: 10,
    fontSize: 13,
    color: '#6B7280',
    fontStyle: 'italic',
  },
  timelineBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'center',
    gap: 8,
    marginTop: 16,
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  timelineText: {
    color: '#FFF',
    fontWeight: '600',
  },
  improvementsSection: {
    marginBottom: 24,
  },
  improvementRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF',
    padding: 14,
    borderRadius: 12,
    marginBottom: 10,
  },
  improvementInfo: {
    flex: 1,
  },
  improvementArea: {
    fontSize: 14,
    color: '#6B7280',
  },
  improvementChange: {
    fontSize: 16,
    fontWeight: '600',
    marginTop: 2,
  },
  confidenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  confidenceBarBg: {
    width: 60,
    height: 6,
    backgroundColor: '#E5E7EB',
    borderRadius: 3,
  },
  confidenceBarFill: {
    height: '100%',
    borderRadius: 3,
  },
  confidenceText: {
    fontSize: 12,
    color: '#6B7280',
    width: 35,
  },
  tipsSection: {
    marginBottom: 20,
  },
  tipRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: 10,
  },
  tipText: {
    flex: 1,
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
  },
  disclaimerBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    backgroundColor: '#F3F4F6',
    padding: 14,
    borderRadius: 12,
    marginBottom: 20,
  },
  disclaimerText: {
    flex: 1,
    fontSize: 12,
    color: '#6B7280',
    lineHeight: 18,
  },
  tryAnotherButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    backgroundColor: '#EDE9FE',
    paddingVertical: 16,
    borderRadius: 16,
    marginBottom: 40,
  },
  tryAnotherText: {
    color: '#8B5CF6',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
