import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Share,
  Switch,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import Slider from '@react-native-community/slider';
import { API_URL } from '../config';

interface Projection {
  years: number;
  future_age: number;
  scenario: string;
  scenario_name: string;
  image: string;
  description: string;
  skin_age_at_time: number;
}

interface KeyFactor {
  factor: string;
  impact: string;
  severity: string;
  description: string;
}

interface TimeMachineResult {
  original_image: string;
  projections: Projection[];
  current_age: number;
  skin_age: number;
  aging_rate: string;
  key_factors: KeyFactor[];
  recommendations: string[];
  share_text: string;
}

export default function SkinTimeMachineScreen() {
  const router = useRouter();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [result, setResult] = useState<TimeMachineResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedYears, setSelectedYears] = useState(10);
  const [selectedScenario, setSelectedScenario] = useState('no_care');

  // Lifestyle factors
  const [age, setAge] = useState(30);
  const [smoking, setSmoking] = useState(false);
  const [sunExposure, setSunExposure] = useState('moderate');
  const [sleepHours, setSleepHours] = useState(7);
  const [useSunscreen, setUseSunscreen] = useState(false);
  const [useRetinol, setUseRetinol] = useState(false);
  const [stressLevel, setStressLevel] = useState('moderate');

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please grant camera roll permissions');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0].uri);
      setResult(null);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please grant camera permissions');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0].uri);
      setResult(null);
    }
  };

  const generateTimeMachine = async () => {
    if (!selectedImage) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', {
        uri: selectedImage,
        type: 'image/jpeg',
        name: 'photo.jpg',
      } as any);
      formData.append('age', age.toString());
      formData.append('smoking', smoking.toString());
      formData.append('sun_exposure', sunExposure);
      formData.append('sleep_hours', sleepHours.toString());
      formData.append('uses_sunscreen', useSunscreen.toString());
      formData.append('uses_retinol', useRetinol.toString());
      formData.append('stress_level', stressLevel);
      formData.append('years', '10,20,30');

      const response = await fetch(`${API_URL}/api/time-machine/generate`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Generation failed');

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      Alert.alert('Error', 'Failed to generate time machine. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const shareResult = async () => {
    if (!result) return;
    try {
      await Share.share({
        message: result.share_text,
      });
    } catch (error) {
      console.error('Share error:', error);
    }
  };

  const getCurrentProjection = () => {
    if (!result) return null;
    return result.projections.find(
      p => p.years === selectedYears && p.scenario === selectedScenario
    );
  };

  const getAgingRateColor = (rate: string) => {
    switch (rate) {
      case 'accelerated': return '#EF4444';
      case 'normal': return '#F59E0B';
      case 'slowed': return '#22C55E';
      default: return '#6B7280';
    }
  };

  const getImpactColor = (impact: string) => {
    return impact === 'positive' ? '#22C55E' : '#EF4444';
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Text style={styles.backButtonText}>←</Text>
          </TouchableOpacity>
          <Text style={styles.title}>⏰ Skin Time Machine</Text>
          <View style={styles.placeholder} />
        </View>

        <Text style={styles.subtitle}>
          See how your skin will age based on your lifestyle choices
        </Text>

        {/* Image Selection */}
        {!selectedImage ? (
          <View style={styles.imageSelectionContainer}>
            <TouchableOpacity style={styles.imageButton} onPress={takePhoto}>
              <Text style={styles.imageButtonIcon}>📷</Text>
              <Text style={styles.imageButtonText}>Take Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.imageButton} onPress={pickImage}>
              <Text style={styles.imageButtonIcon}>🖼️</Text>
              <Text style={styles.imageButtonText}>Choose Photo</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <View style={styles.selectedImageContainer}>
            <Image source={{ uri: selectedImage }} style={styles.selectedImage} />
            <TouchableOpacity style={styles.changeButton} onPress={pickImage}>
              <Text style={styles.changeButtonText}>Change Photo</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Lifestyle Factors */}
        {selectedImage && !result && (
          <View style={styles.factorsContainer}>
            <Text style={styles.sectionTitle}>Your Lifestyle Factors</Text>

            {/* Age */}
            <View style={styles.factorRow}>
              <Text style={styles.factorLabel}>Your Age: {age}</Text>
              <Slider
                style={styles.slider}
                minimumValue={18}
                maximumValue={80}
                step={1}
                value={age}
                onValueChange={setAge}
                minimumTrackTintColor="#6366F1"
                maximumTrackTintColor="#E5E7EB"
              />
            </View>

            {/* Smoking */}
            <View style={styles.switchRow}>
              <Text style={styles.factorLabel}>🚬 Smoking</Text>
              <Switch
                value={smoking}
                onValueChange={setSmoking}
                trackColor={{ false: '#E5E7EB', true: '#EF4444' }}
              />
            </View>

            {/* Sunscreen */}
            <View style={styles.switchRow}>
              <Text style={styles.factorLabel}>🧴 Daily Sunscreen</Text>
              <Switch
                value={useSunscreen}
                onValueChange={setUseSunscreen}
                trackColor={{ false: '#E5E7EB', true: '#22C55E' }}
              />
            </View>

            {/* Retinol */}
            <View style={styles.switchRow}>
              <Text style={styles.factorLabel}>✨ Use Retinol</Text>
              <Switch
                value={useRetinol}
                onValueChange={setUseRetinol}
                trackColor={{ false: '#E5E7EB', true: '#22C55E' }}
              />
            </View>

            {/* Sun Exposure */}
            <View style={styles.factorRow}>
              <Text style={styles.factorLabel}>☀️ Sun Exposure</Text>
              <View style={styles.optionButtons}>
                {['low', 'moderate', 'high'].map(level => (
                  <TouchableOpacity
                    key={level}
                    style={[
                      styles.optionButton,
                      sunExposure === level && styles.optionButtonActive,
                    ]}
                    onPress={() => setSunExposure(level)}
                  >
                    <Text style={[
                      styles.optionButtonText,
                      sunExposure === level && styles.optionButtonTextActive,
                    ]}>
                      {level.charAt(0).toUpperCase() + level.slice(1)}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Sleep */}
            <View style={styles.factorRow}>
              <Text style={styles.factorLabel}>😴 Sleep: {sleepHours} hours</Text>
              <Slider
                style={styles.slider}
                minimumValue={4}
                maximumValue={10}
                step={0.5}
                value={sleepHours}
                onValueChange={setSleepHours}
                minimumTrackTintColor="#6366F1"
                maximumTrackTintColor="#E5E7EB"
              />
            </View>

            {/* Generate Button */}
            <TouchableOpacity
              style={styles.generateButton}
              onPress={generateTimeMachine}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.generateButtonText}>🔮 See Your Future</Text>
              )}
            </TouchableOpacity>
          </View>
        )}

        {/* Results */}
        {result && (
          <View style={styles.resultsContainer}>
            {/* Skin Age Summary */}
            <View style={styles.summaryCard}>
              <View style={styles.ageComparison}>
                <View style={styles.ageBox}>
                  <Text style={styles.ageLabel}>Real Age</Text>
                  <Text style={styles.ageValue}>{result.current_age}</Text>
                </View>
                <Text style={styles.ageArrow}>→</Text>
                <View style={styles.ageBox}>
                  <Text style={styles.ageLabel}>Skin Age</Text>
                  <Text style={[
                    styles.ageValue,
                    { color: result.skin_age > result.current_age ? '#EF4444' : '#22C55E' }
                  ]}>
                    {result.skin_age}
                  </Text>
                </View>
              </View>
              <View style={[
                styles.rateChip,
                { backgroundColor: getAgingRateColor(result.aging_rate) + '20' }
              ]}>
                <Text style={[
                  styles.rateText,
                  { color: getAgingRateColor(result.aging_rate) }
                ]}>
                  {result.aging_rate.charAt(0).toUpperCase() + result.aging_rate.slice(1)} Aging
                </Text>
              </View>
            </View>

            {/* Year Selector */}
            <View style={styles.yearSelector}>
              <Text style={styles.sectionTitle}>Years Ahead</Text>
              <View style={styles.yearButtons}>
                {[10, 20, 30].map(years => (
                  <TouchableOpacity
                    key={years}
                    style={[
                      styles.yearButton,
                      selectedYears === years && styles.yearButtonActive,
                    ]}
                    onPress={() => setSelectedYears(years)}
                  >
                    <Text style={[
                      styles.yearButtonText,
                      selectedYears === years && styles.yearButtonTextActive,
                    ]}>
                      +{years} Years
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Scenario Selector */}
            <View style={styles.scenarioSelector}>
              <Text style={styles.sectionTitle}>Scenario</Text>
              <View style={styles.scenarioButtons}>
                {[
                  { id: 'no_care', label: 'No Care', icon: '😰' },
                  { id: 'basic_care', label: 'Basic', icon: '🙂' },
                  { id: 'optimal_care', label: 'Optimal', icon: '😊' },
                ].map(scenario => (
                  <TouchableOpacity
                    key={scenario.id}
                    style={[
                      styles.scenarioButton,
                      selectedScenario === scenario.id && styles.scenarioButtonActive,
                    ]}
                    onPress={() => setSelectedScenario(scenario.id)}
                  >
                    <Text style={styles.scenarioIcon}>{scenario.icon}</Text>
                    <Text style={[
                      styles.scenarioButtonText,
                      selectedScenario === scenario.id && styles.scenarioButtonTextActive,
                    ]}>
                      {scenario.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Projection Image */}
            {getCurrentProjection() && (
              <View style={styles.projectionContainer}>
                <View style={styles.projectionHeader}>
                  <Text style={styles.projectionTitle}>
                    You at Age {getCurrentProjection()!.future_age}
                  </Text>
                  <Text style={styles.projectionSubtitle}>
                    Skin Age: {getCurrentProjection()!.skin_age_at_time}
                  </Text>
                </View>
                <Image
                  source={{ uri: `data:image/jpeg;base64,${getCurrentProjection()!.image}` }}
                  style={styles.projectionImage}
                />
                <Text style={styles.projectionDescription}>
                  {getCurrentProjection()!.description}
                </Text>
              </View>
            )}

            {/* Key Factors */}
            <View style={styles.factorsSection}>
              <Text style={styles.sectionTitle}>Key Factors Affecting Your Skin</Text>
              {result.key_factors.map((factor, index) => (
                <View key={index} style={styles.factorCard}>
                  <View style={styles.factorHeader}>
                    <Text style={styles.factorName}>{factor.factor}</Text>
                    <View style={[
                      styles.impactChip,
                      { backgroundColor: getImpactColor(factor.impact) + '20' }
                    ]}>
                      <Text style={[
                        styles.impactText,
                        { color: getImpactColor(factor.impact) }
                      ]}>
                        {factor.impact === 'positive' ? '✓ Good' : '⚠ Bad'}
                      </Text>
                    </View>
                  </View>
                  <Text style={styles.factorDescription}>{factor.description}</Text>
                </View>
              ))}
            </View>

            {/* Recommendations */}
            <View style={styles.recommendationsSection}>
              <Text style={styles.sectionTitle}>Recommendations</Text>
              {result.recommendations.map((rec, index) => (
                <View key={index} style={styles.recommendationItem}>
                  <Text style={styles.recommendationNumber}>{index + 1}</Text>
                  <Text style={styles.recommendationText}>{rec}</Text>
                </View>
              ))}
            </View>

            {/* Share Button */}
            <TouchableOpacity style={styles.shareButton} onPress={shareResult}>
              <Text style={styles.shareButtonText}>📤 Share My Results</Text>
            </TouchableOpacity>

            {/* Try Again */}
            <TouchableOpacity
              style={styles.tryAgainButton}
              onPress={() => {
                setResult(null);
                setSelectedImage(null);
              }}
            >
              <Text style={styles.tryAgainButtonText}>Try Another Photo</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    fontSize: 24,
    color: '#6366F1',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  placeholder: {
    width: 40,
  },
  subtitle: {
    fontSize: 14,
    color: '#6B7280',
    textAlign: 'center',
    marginBottom: 24,
  },
  imageSelectionContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
    marginBottom: 24,
  },
  imageButton: {
    backgroundColor: '#fff',
    padding: 24,
    borderRadius: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    width: 140,
  },
  imageButtonIcon: {
    fontSize: 40,
    marginBottom: 8,
  },
  imageButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
  },
  selectedImageContainer: {
    alignItems: 'center',
    marginBottom: 24,
  },
  selectedImage: {
    width: 200,
    height: 200,
    borderRadius: 16,
    marginBottom: 12,
  },
  changeButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#E5E7EB',
    borderRadius: 8,
  },
  changeButtonText: {
    color: '#374151',
    fontWeight: '500',
  },
  factorsContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 16,
  },
  factorRow: {
    marginBottom: 16,
  },
  factorLabel: {
    fontSize: 16,
    color: '#374151',
    marginBottom: 8,
  },
  slider: {
    width: '100%',
    height: 40,
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  optionButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  optionButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#F3F4F6',
  },
  optionButtonActive: {
    backgroundColor: '#6366F1',
  },
  optionButtonText: {
    fontSize: 14,
    color: '#6B7280',
  },
  optionButtonTextActive: {
    color: '#fff',
  },
  generateButton: {
    backgroundColor: '#6366F1',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 8,
  },
  generateButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  resultsContainer: {
    marginTop: 8,
  },
  summaryCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  ageComparison: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  ageBox: {
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  ageLabel: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 4,
  },
  ageValue: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  ageArrow: {
    fontSize: 24,
    color: '#9CA3AF',
    marginHorizontal: 16,
  },
  rateChip: {
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 20,
  },
  rateText: {
    fontSize: 14,
    fontWeight: '600',
  },
  yearSelector: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  yearButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  yearButton: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: '#F3F4F6',
  },
  yearButtonActive: {
    backgroundColor: '#6366F1',
  },
  yearButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6B7280',
  },
  yearButtonTextActive: {
    color: '#fff',
  },
  scenarioSelector: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  scenarioButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  scenarioButton: {
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: '#F3F4F6',
    minWidth: 90,
  },
  scenarioButtonActive: {
    backgroundColor: '#6366F1',
  },
  scenarioIcon: {
    fontSize: 24,
    marginBottom: 4,
  },
  scenarioButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6B7280',
  },
  scenarioButtonTextActive: {
    color: '#fff',
  },
  projectionContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
    marginBottom: 16,
  },
  projectionHeader: {
    alignItems: 'center',
    marginBottom: 12,
  },
  projectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  projectionSubtitle: {
    fontSize: 14,
    color: '#6B7280',
  },
  projectionImage: {
    width: 280,
    height: 280,
    borderRadius: 16,
    marginBottom: 12,
  },
  projectionDescription: {
    fontSize: 14,
    color: '#6B7280',
    textAlign: 'center',
  },
  factorsSection: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  factorCard: {
    backgroundColor: '#F9FAFB',
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
  },
  factorHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  factorName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
  },
  impactChip: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  impactText: {
    fontSize: 12,
    fontWeight: '600',
  },
  factorDescription: {
    fontSize: 13,
    color: '#6B7280',
    lineHeight: 18,
  },
  recommendationsSection: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  recommendationItem: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  recommendationNumber: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#6366F1',
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
    lineHeight: 24,
    marginRight: 12,
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
  },
  shareButton: {
    backgroundColor: '#10B981',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 12,
  },
  shareButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  tryAgainButton: {
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    backgroundColor: '#F3F4F6',
  },
  tryAgainButtonText: {
    color: '#6B7280',
    fontSize: 16,
    fontWeight: '600',
  },
});
