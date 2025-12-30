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
  Alert,
  Dimensions,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { API_BASE_URL } from '../config';

const { width } = Dimensions.get('window');

interface CategoryScore {
  name: string;
  score: number;
  icon: string;
}

interface Recommendation {
  title: string;
  description: string;
  tips: string[];
  priority: string;
  current_score: number;
}

interface SkinHealthResult {
  overall_score: number;
  skin_age: number;
  chronological_age: number;
  age_difference: number;
  percentile: number;
  category_scores: Record<string, number>;
  recommendations: Recommendation[];
  insights: string[];
  share_text: string;
}

export default function SkinHealthScoreScreen() {
  const router = useRouter();
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SkinHealthResult | null>(null);
  const [userAge, setUserAge] = useState<number>(30);
  const [showAgeInput, setShowAgeInput] = useState(true);

  const categoryIcons: Record<string, string> = {
    hydration: 'water',
    texture: 'layers',
    clarity: 'sunny',
    firmness: 'fitness',
    uv_damage: 'shield',
    pore_size: 'ellipse',
    wrinkles: 'git-branch',
    pigmentation: 'color-palette',
  };

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
      setShowAgeInput(true);
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
      setShowAgeInput(true);
      setResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!image) return;

    setLoading(true);
    setShowAgeInput(false);

    try {
      const apiUrl = `${API_BASE_URL}/api/skin-health/analyze`;
      console.log('[SkinHealthScore] Sending request to:', apiUrl);
      console.log('[SkinHealthScore] Image URI:', image);

      const formData = new FormData();
      formData.append('image', {
        uri: image,
        type: 'image/jpeg',
        name: 'skin_photo.jpg',
      } as any);
      formData.append('age', userAge.toString());
      formData.append('uses_sunscreen', 'true');

      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
        // Note: Don't set Content-Type header - fetch will set it automatically with the correct boundary
      });

      console.log('[SkinHealthScore] Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[SkinHealthScore] Error response:', errorText);
        throw new Error(`Analysis failed: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error: any) {
      console.error('[SkinHealthScore] Analysis error:', error);
      Alert.alert('Error', `Failed to analyze image: ${error.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const shareResults = async () => {
    if (!result) return;

    try {
      await Share.share({
        message: result.share_text,
      });
    } catch (error) {
      console.error('Share error:', error);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return '#22C55E';
    if (score >= 60) return '#84CC16';
    if (score >= 40) return '#F59E0B';
    return '#EF4444';
  };

  const getScoreGradient = (score: number): [string, string] => {
    if (score >= 80) return ['#22C55E', '#16A34A'];
    if (score >= 60) return ['#84CC16', '#65A30D'];
    if (score >= 40) return ['#F59E0B', '#D97706'];
    return ['#EF4444', '#DC2626'];
  };

  const renderScoreCircle = () => {
    if (!result) return null;

    return (
      <Animated.View entering={FadeInUp.duration(800)} style={styles.scoreContainer}>
        <LinearGradient
          colors={getScoreGradient(result.overall_score)}
          style={styles.scoreCircle}
        >
          <Text style={styles.scoreNumber}>{result.overall_score}</Text>
          <Text style={styles.scoreLabel}>Health Score</Text>
        </LinearGradient>

        <View style={styles.ageContainer}>
          <View style={styles.ageBox}>
            <Text style={styles.ageLabel}>Skin Age</Text>
            <Text style={[styles.ageValue, { color: result.age_difference > 0 ? '#22C55E' : '#EF4444' }]}>
              {result.skin_age}
            </Text>
          </View>
          <View style={styles.ageDivider} />
          <View style={styles.ageBox}>
            <Text style={styles.ageLabel}>Your Age</Text>
            <Text style={styles.ageValue}>{result.chronological_age}</Text>
          </View>
          <View style={styles.ageDivider} />
          <View style={styles.ageBox}>
            <Text style={styles.ageLabel}>Difference</Text>
            <Text style={[styles.ageValue, { color: result.age_difference > 0 ? '#22C55E' : '#EF4444' }]}>
              {result.age_difference > 0 ? '+' : ''}{result.age_difference}
            </Text>
          </View>
        </View>

        <View style={styles.percentileContainer}>
          <Ionicons name="trophy" size={20} color="#F59E0B" />
          <Text style={styles.percentileText}>
            Top {100 - result.percentile}% for your age group
          </Text>
        </View>
      </Animated.View>
    );
  };

  const renderCategoryScores = () => {
    if (!result) return null;

    const categories = Object.entries(result.category_scores).map(([key, value]) => ({
      name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      score: value,
      icon: categoryIcons[key] || 'ellipse',
    }));

    return (
      <Animated.View entering={FadeInDown.delay(200).duration(600)} style={styles.categoriesContainer}>
        <Text style={styles.sectionTitle}>Category Breakdown</Text>
        <View style={styles.categoriesGrid}>
          {categories.map((cat, index) => (
            <View key={index} style={styles.categoryCard}>
              <Ionicons name={cat.icon as any} size={24} color={getScoreColor(cat.score)} />
              <Text style={styles.categoryName}>{cat.name}</Text>
              <View style={styles.categoryScoreBar}>
                <View
                  style={[
                    styles.categoryScoreFill,
                    { width: `${cat.score}%`, backgroundColor: getScoreColor(cat.score) },
                  ]}
                />
              </View>
              <Text style={[styles.categoryScore, { color: getScoreColor(cat.score) }]}>
                {cat.score}
              </Text>
            </View>
          ))}
        </View>
      </Animated.View>
    );
  };

  const renderInsights = () => {
    if (!result || !result.insights.length) return null;

    return (
      <Animated.View entering={FadeInDown.delay(400).duration(600)} style={styles.insightsContainer}>
        <Text style={styles.sectionTitle}>Insights</Text>
        {result.insights.map((insight, index) => (
          <View key={index} style={styles.insightItem}>
            <Ionicons name="sparkles" size={16} color="#8B5CF6" />
            <Text style={styles.insightText}>{insight}</Text>
          </View>
        ))}
      </Animated.View>
    );
  };

  const renderRecommendations = () => {
    if (!result || !result.recommendations.length) return null;

    return (
      <Animated.View entering={FadeInDown.delay(600).duration(600)} style={styles.recommendationsContainer}>
        <Text style={styles.sectionTitle}>Recommendations</Text>
        {result.recommendations.map((rec, index) => (
          <View key={index} style={styles.recommendationCard}>
            <View style={styles.recommendationHeader}>
              <Text style={styles.recommendationTitle}>{rec.title}</Text>
              <View style={[
                styles.priorityBadge,
                { backgroundColor: rec.priority === 'high' ? '#FEE2E2' : '#FEF3C7' }
              ]}>
                <Text style={[
                  styles.priorityText,
                  { color: rec.priority === 'high' ? '#EF4444' : '#F59E0B' }
                ]}>
                  {rec.priority}
                </Text>
              </View>
            </View>
            <Text style={styles.recommendationDesc}>{rec.description}</Text>
            {rec.tips.slice(0, 2).map((tip, tipIndex) => (
              <View key={tipIndex} style={styles.tipItem}>
                <Ionicons name="checkmark-circle" size={14} color="#22C55E" />
                <Text style={styles.tipText}>{tip}</Text>
              </View>
            ))}
          </View>
        ))}
      </Animated.View>
    );
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <LinearGradient colors={['#8B5CF6', '#6366F1']} style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#FFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Skin Health Score</Text>
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
                  <Ionicons name="refresh" size={20} color="#FFF" />
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.imagePlaceholder}>
                <Ionicons name="camera" size={48} color="#9CA3AF" />
                <Text style={styles.placeholderText}>Take or select a clear face photo</Text>
              </View>
            )}

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

            {/* Age Input */}
            {image && showAgeInput && (
              <View style={styles.ageInputContainer}>
                <Text style={styles.ageInputLabel}>Enter your age:</Text>
                <View style={styles.ageInputRow}>
                  <TouchableOpacity
                    style={styles.ageBtn}
                    onPress={() => setUserAge(Math.max(18, userAge - 1))}
                  >
                    <Ionicons name="remove" size={24} color="#8B5CF6" />
                  </TouchableOpacity>
                  <Text style={styles.ageInputValue}>{userAge}</Text>
                  <TouchableOpacity
                    style={styles.ageBtn}
                    onPress={() => setUserAge(Math.min(100, userAge + 1))}
                  >
                    <Ionicons name="add" size={24} color="#8B5CF6" />
                  </TouchableOpacity>
                </View>

                <TouchableOpacity
                  style={[styles.analyzeButton, loading && styles.analyzeButtonDisabled]}
                  onPress={analyzeImage}
                  disabled={loading}
                >
                  {loading ? (
                    <ActivityIndicator color="#FFF" />
                  ) : (
                    <>
                      <Ionicons name="analytics" size={24} color="#FFF" />
                      <Text style={styles.analyzeButtonText}>Analyze My Skin</Text>
                    </>
                  )}
                </TouchableOpacity>
              </View>
            )}
          </View>
        )}

        {/* Results */}
        {result && (
          <>
            {renderScoreCircle()}
            {renderCategoryScores()}
            {renderInsights()}
            {renderRecommendations()}

            {/* Action Buttons */}
            <View style={styles.actionButtons}>
              <TouchableOpacity style={styles.shareButton} onPress={shareResults}>
                <Ionicons name="share-social" size={24} color="#FFF" />
                <Text style={styles.shareButtonText}>Share Results</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.retakeButton}
                onPress={() => {
                  setResult(null);
                  setImage(null);
                  setShowAgeInput(true);
                }}
              >
                <Ionicons name="refresh" size={24} color="#8B5CF6" />
                <Text style={styles.retakeButtonText}>New Analysis</Text>
              </TouchableOpacity>
            </View>
          </>
        )}
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
    justifyContent: 'space-between',
    paddingTop: 60,
    paddingBottom: 20,
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
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFF',
  },
  content: {
    padding: 20,
  },
  imageSection: {
    alignItems: 'center',
  },
  imagePreviewContainer: {
    position: 'relative',
    marginBottom: 20,
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
    width: 40,
    height: 40,
    borderRadius: 20,
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
    marginBottom: 20,
  },
  placeholderText: {
    marginTop: 10,
    color: '#6B7280',
    fontSize: 14,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 20,
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
  ageInputContainer: {
    width: '100%',
    alignItems: 'center',
    marginTop: 20,
  },
  ageInputLabel: {
    fontSize: 16,
    color: '#374151',
    marginBottom: 10,
  },
  ageInputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 20,
    marginBottom: 24,
  },
  ageBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#EDE9FE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  ageInputValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#1F2937',
    minWidth: 60,
    textAlign: 'center',
  },
  analyzeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    width: '100%',
    paddingVertical: 16,
    borderRadius: 16,
    backgroundColor: '#8B5CF6',
  },
  analyzeButtonDisabled: {
    opacity: 0.7,
  },
  analyzeButtonText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  scoreContainer: {
    alignItems: 'center',
    marginBottom: 24,
  },
  scoreCircle: {
    width: 160,
    height: 160,
    borderRadius: 80,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 8,
  },
  scoreNumber: {
    fontSize: 56,
    fontWeight: 'bold',
    color: '#FFF',
  },
  scoreLabel: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.9)',
    marginTop: -4,
  },
  ageContainer: {
    flexDirection: 'row',
    backgroundColor: '#FFF',
    borderRadius: 16,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    marginBottom: 12,
  },
  ageBox: {
    flex: 1,
    alignItems: 'center',
  },
  ageDivider: {
    width: 1,
    backgroundColor: '#E5E7EB',
    marginHorizontal: 8,
  },
  ageLabel: {
    fontSize: 12,
    color: '#6B7280',
    marginBottom: 4,
  },
  ageValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  percentileContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#FEF3C7',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  percentileText: {
    color: '#92400E',
    fontWeight: '600',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 16,
  },
  categoriesContainer: {
    marginBottom: 24,
  },
  categoriesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  categoryCard: {
    width: (width - 52) / 2,
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  categoryName: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 8,
    marginBottom: 8,
  },
  categoryScoreBar: {
    width: '100%',
    height: 4,
    backgroundColor: '#E5E7EB',
    borderRadius: 2,
    marginBottom: 6,
  },
  categoryScoreFill: {
    height: '100%',
    borderRadius: 2,
  },
  categoryScore: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  insightsContainer: {
    marginBottom: 24,
  },
  insightItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    backgroundColor: '#FFF',
    padding: 12,
    borderRadius: 12,
    marginBottom: 8,
  },
  insightText: {
    flex: 1,
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
  },
  recommendationsContainer: {
    marginBottom: 24,
  },
  recommendationCard: {
    backgroundColor: '#FFF',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  recommendationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  recommendationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
  },
  priorityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  priorityText: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  recommendationDesc: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 12,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 6,
  },
  tipText: {
    flex: 1,
    fontSize: 13,
    color: '#374151',
  },
  actionButtons: {
    gap: 12,
    marginBottom: 40,
  },
  shareButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    backgroundColor: '#8B5CF6',
    paddingVertical: 16,
    borderRadius: 16,
  },
  shareButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  retakeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    backgroundColor: '#EDE9FE',
    paddingVertical: 16,
    borderRadius: 16,
  },
  retakeButtonText: {
    color: '#8B5CF6',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
