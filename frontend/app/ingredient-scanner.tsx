import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { API_URL } from '../config';

interface IngredientResult {
  name: string;
  category: string;
  rating: string;
  rating_reason: string;
  comedogenic_rating: number;
  irritation_potential: string;
  benefits: string[];
  concerns: string[];
  good_for: string[];
  avoid_if: string[];
}

interface ScanResult {
  product_name: string | null;
  overall_rating: string;
  overall_score: number;
  compatibility_score: number;
  ingredient_count: number;
  ingredients: IngredientResult[];
  summary: string;
  pros: string[];
  cons: string[];
  warnings: string[];
  alternatives: { type: string; suggestion: string; reason: string }[];
  effectiveness_for_goals: Record<string, number>;
}

const SKIN_TYPES = ['normal', 'oily', 'dry', 'combination', 'sensitive'];
const SKIN_CONCERNS = [
  'acne', 'aging', 'hyperpigmentation', 'redness', 'dryness',
  'oiliness', 'sensitivity', 'dullness', 'dark_spots', 'fine_lines',
];

export default function IngredientScannerScreen() {
  const router = useRouter();
  const [ingredientsText, setIngredientsText] = useState('');
  const [productName, setProductName] = useState('');
  const [skinType, setSkinType] = useState('normal');
  const [selectedConcerns, setSelectedConcerns] = useState<string[]>([]);
  const [allergies, setAllergies] = useState('');
  const [result, setResult] = useState<ScanResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [expandedIngredient, setExpandedIngredient] = useState<string | null>(null);

  const toggleConcern = (concern: string) => {
    setSelectedConcerns(prev =>
      prev.includes(concern)
        ? prev.filter(c => c !== concern)
        : [...prev, concern]
    );
  };

  const scanIngredients = async () => {
    if (!ingredientsText.trim()) {
      Alert.alert('Missing ingredients', 'Please paste the ingredient list');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('ingredients', ingredientsText);
      formData.append('skin_type', skinType);
      formData.append('concerns', selectedConcerns.join(','));
      formData.append('allergies', allergies);
      if (productName) {
        formData.append('product_name', productName);
      }

      const response = await fetch(`${API_URL}/api/ingredient-scanner/scan`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Scan failed');

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      Alert.alert('Error', 'Failed to scan ingredients. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getRatingColor = (rating: string) => {
    switch (rating) {
      case 'excellent': return '#22C55E';
      case 'good': return '#10B981';
      case 'neutral': return '#6B7280';
      case 'caution': return '#F59E0B';
      case 'avoid': return '#EF4444';
      default: return '#6B7280';
    }
  };

  const getGradeColor = (grade: string) => {
    switch (grade) {
      case 'A': return '#22C55E';
      case 'B': return '#10B981';
      case 'C': return '#F59E0B';
      case 'D': return '#F97316';
      case 'F': return '#EF4444';
      default: return '#6B7280';
    }
  };

  const getComedogenicLabel = (rating: number) => {
    if (rating === 0) return 'Non-comedogenic';
    if (rating <= 2) return 'Low clog risk';
    if (rating <= 3) return 'Moderate clog risk';
    return 'High clog risk';
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Text style={styles.backButtonText}>←</Text>
          </TouchableOpacity>
          <Text style={styles.title}>🧴 Ingredient Scanner</Text>
          <View style={styles.placeholder} />
        </View>

        <Text style={styles.subtitle}>
          Paste ingredient list to check compatibility with your skin
        </Text>

        {!result ? (
          <>
            {/* Product Name (Optional) */}
            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Product Name (Optional)</Text>
              <TextInput
                style={styles.textInput}
                placeholder="e.g., CeraVe Moisturizing Cream"
                value={productName}
                onChangeText={setProductName}
                placeholderTextColor="#9CA3AF"
              />
            </View>

            {/* Ingredients Text */}
            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Ingredient List *</Text>
              <TextInput
                style={[styles.textInput, styles.textArea]}
                placeholder="Paste ingredient list here (comma-separated)&#10;&#10;e.g., Water, Glycerin, Niacinamide, Hyaluronic Acid, Fragrance..."
                value={ingredientsText}
                onChangeText={setIngredientsText}
                multiline
                numberOfLines={6}
                textAlignVertical="top"
                placeholderTextColor="#9CA3AF"
              />
            </View>

            {/* Skin Type */}
            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Your Skin Type</Text>
              <View style={styles.chipContainer}>
                {SKIN_TYPES.map(type => (
                  <TouchableOpacity
                    key={type}
                    style={[
                      styles.chip,
                      skinType === type && styles.chipActive,
                    ]}
                    onPress={() => setSkinType(type)}
                  >
                    <Text style={[
                      styles.chipText,
                      skinType === type && styles.chipTextActive,
                    ]}>
                      {type.charAt(0).toUpperCase() + type.slice(1)}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Skin Concerns */}
            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Your Skin Concerns</Text>
              <View style={styles.chipContainer}>
                {SKIN_CONCERNS.map(concern => (
                  <TouchableOpacity
                    key={concern}
                    style={[
                      styles.chip,
                      selectedConcerns.includes(concern) && styles.chipActive,
                    ]}
                    onPress={() => toggleConcern(concern)}
                  >
                    <Text style={[
                      styles.chipText,
                      selectedConcerns.includes(concern) && styles.chipTextActive,
                    ]}>
                      {concern.replace('_', ' ').charAt(0).toUpperCase() + concern.replace('_', ' ').slice(1)}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Allergies */}
            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Known Allergies (Optional)</Text>
              <TextInput
                style={styles.textInput}
                placeholder="e.g., fragrance, coconut"
                value={allergies}
                onChangeText={setAllergies}
                placeholderTextColor="#9CA3AF"
              />
            </View>

            {/* Scan Button */}
            <TouchableOpacity
              style={styles.scanButton}
              onPress={scanIngredients}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.scanButtonText}>🔍 Analyze Ingredients</Text>
              )}
            </TouchableOpacity>
          </>
        ) : (
          <>
            {/* Results */}
            <View style={styles.resultsContainer}>
              {/* Overall Score */}
              <View style={styles.scoreCard}>
                <View style={styles.scoreCircle}>
                  <Text style={[
                    styles.scoreGrade,
                    { color: getGradeColor(result.overall_rating) }
                  ]}>
                    {result.overall_rating}
                  </Text>
                  <Text style={styles.scoreNumber}>{result.overall_score}/100</Text>
                </View>
                <View style={styles.scoreDetails}>
                  {result.product_name && (
                    <Text style={styles.productName}>{result.product_name}</Text>
                  )}
                  <Text style={styles.compatibilityText}>
                    Compatibility: {result.compatibility_score}%
                  </Text>
                  <Text style={styles.ingredientCount}>
                    {result.ingredient_count} ingredients analyzed
                  </Text>
                </View>
              </View>

              {/* Summary */}
              <View style={styles.summaryCard}>
                <Text style={styles.summaryText}>{result.summary}</Text>
              </View>

              {/* Warnings */}
              {result.warnings.length > 0 && (
                <View style={styles.warningsCard}>
                  {result.warnings.map((warning, index) => (
                    <Text key={index} style={styles.warningText}>{warning}</Text>
                  ))}
                </View>
              )}

              {/* Pros & Cons */}
              <View style={styles.prosConsContainer}>
                {result.pros.length > 0 && (
                  <View style={styles.prosCard}>
                    <Text style={styles.prosTitle}>✓ Pros</Text>
                    {result.pros.map((pro, index) => (
                      <Text key={index} style={styles.prosItem}>• {pro}</Text>
                    ))}
                  </View>
                )}
                {result.cons.length > 0 && (
                  <View style={styles.consCard}>
                    <Text style={styles.consTitle}>✗ Cons</Text>
                    {result.cons.map((con, index) => (
                      <Text key={index} style={styles.consItem}>• {con}</Text>
                    ))}
                  </View>
                )}
              </View>

              {/* Effectiveness for Goals */}
              {Object.keys(result.effectiveness_for_goals).length > 0 && (
                <View style={styles.effectivenessCard}>
                  <Text style={styles.sectionTitle}>Effectiveness for Your Goals</Text>
                  {Object.entries(result.effectiveness_for_goals).map(([goal, score]) => (
                    <View key={goal} style={styles.effectivenessRow}>
                      <Text style={styles.effectivenessLabel}>
                        {goal.replace('_', ' ').charAt(0).toUpperCase() + goal.replace('_', ' ').slice(1)}
                      </Text>
                      <View style={styles.effectivenessBar}>
                        <View style={[
                          styles.effectivenessFill,
                          { width: `${score}%`, backgroundColor: score >= 70 ? '#22C55E' : score >= 50 ? '#F59E0B' : '#EF4444' }
                        ]} />
                      </View>
                      <Text style={styles.effectivenessScore}>{score}%</Text>
                    </View>
                  ))}
                </View>
              )}

              {/* Ingredient List */}
              <View style={styles.ingredientListCard}>
                <Text style={styles.sectionTitle}>Ingredient Breakdown</Text>
                {result.ingredients.map((ingredient, index) => (
                  <TouchableOpacity
                    key={index}
                    style={styles.ingredientItem}
                    onPress={() => setExpandedIngredient(
                      expandedIngredient === ingredient.name ? null : ingredient.name
                    )}
                  >
                    <View style={styles.ingredientHeader}>
                      <View style={styles.ingredientNameRow}>
                        <View style={[
                          styles.ratingDot,
                          { backgroundColor: getRatingColor(ingredient.rating) }
                        ]} />
                        <Text style={styles.ingredientName}>{ingredient.name}</Text>
                      </View>
                      <View style={[
                        styles.ratingChip,
                        { backgroundColor: getRatingColor(ingredient.rating) + '20' }
                      ]}>
                        <Text style={[
                          styles.ratingChipText,
                          { color: getRatingColor(ingredient.rating) }
                        ]}>
                          {ingredient.rating.charAt(0).toUpperCase() + ingredient.rating.slice(1)}
                        </Text>
                      </View>
                    </View>

                    {expandedIngredient === ingredient.name && (
                      <View style={styles.ingredientDetails}>
                        <Text style={styles.detailCategory}>{ingredient.category}</Text>
                        <Text style={styles.detailReason}>{ingredient.rating_reason}</Text>

                        <View style={styles.detailRow}>
                          <Text style={styles.detailLabel}>Pore-clogging:</Text>
                          <Text style={styles.detailValue}>
                            {ingredient.comedogenic_rating}/5 - {getComedogenicLabel(ingredient.comedogenic_rating)}
                          </Text>
                        </View>

                        <View style={styles.detailRow}>
                          <Text style={styles.detailLabel}>Irritation:</Text>
                          <Text style={styles.detailValue}>
                            {ingredient.irritation_potential.charAt(0).toUpperCase() + ingredient.irritation_potential.slice(1)}
                          </Text>
                        </View>

                        {ingredient.benefits.length > 0 && (
                          <View style={styles.benefitsList}>
                            <Text style={styles.benefitsTitle}>Benefits:</Text>
                            {ingredient.benefits.map((benefit, i) => (
                              <Text key={i} style={styles.benefitItem}>• {benefit}</Text>
                            ))}
                          </View>
                        )}

                        {ingredient.concerns.length > 0 && (
                          <View style={styles.concernsList}>
                            <Text style={styles.concernsTitle}>Concerns:</Text>
                            {ingredient.concerns.map((concern, i) => (
                              <Text key={i} style={styles.concernItem}>• {concern}</Text>
                            ))}
                          </View>
                        )}
                      </View>
                    )}
                  </TouchableOpacity>
                ))}
              </View>

              {/* Alternatives */}
              {result.alternatives.length > 0 && (
                <View style={styles.alternativesCard}>
                  <Text style={styles.sectionTitle}>Suggestions</Text>
                  {result.alternatives.map((alt, index) => (
                    <View key={index} style={styles.alternativeItem}>
                      <Text style={styles.alternativeSuggestion}>{alt.suggestion}</Text>
                      <Text style={styles.alternativeReason}>{alt.reason}</Text>
                    </View>
                  ))}
                </View>
              )}

              {/* Scan Another */}
              <TouchableOpacity
                style={styles.scanAnotherButton}
                onPress={() => {
                  setResult(null);
                  setIngredientsText('');
                  setProductName('');
                }}
              >
                <Text style={styles.scanAnotherButtonText}>Scan Another Product</Text>
              </TouchableOpacity>
            </View>
          </>
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
  inputContainer: {
    marginBottom: 20,
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  textInput: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    color: '#1F2937',
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },
  textArea: {
    minHeight: 120,
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },
  chipActive: {
    backgroundColor: '#6366F1',
    borderColor: '#6366F1',
  },
  chipText: {
    fontSize: 14,
    color: '#6B7280',
  },
  chipTextActive: {
    color: '#fff',
  },
  scanButton: {
    backgroundColor: '#6366F1',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 8,
  },
  scanButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  resultsContainer: {
    marginTop: 8,
  },
  scoreCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  scoreCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#F3F4F6',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  scoreGrade: {
    fontSize: 32,
    fontWeight: 'bold',
  },
  scoreNumber: {
    fontSize: 12,
    color: '#6B7280',
  },
  scoreDetails: {
    flex: 1,
  },
  productName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 4,
  },
  compatibilityText: {
    fontSize: 16,
    color: '#374151',
    marginBottom: 2,
  },
  ingredientCount: {
    fontSize: 14,
    color: '#6B7280',
  },
  summaryCard: {
    backgroundColor: '#EEF2FF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  summaryText: {
    fontSize: 15,
    color: '#4338CA',
    lineHeight: 22,
  },
  warningsCard: {
    backgroundColor: '#FEF2F2',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  warningText: {
    fontSize: 14,
    color: '#DC2626',
    marginBottom: 4,
  },
  prosConsContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  prosCard: {
    flex: 1,
    backgroundColor: '#ECFDF5',
    borderRadius: 12,
    padding: 12,
  },
  prosTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#059669',
    marginBottom: 8,
  },
  prosItem: {
    fontSize: 13,
    color: '#047857',
    marginBottom: 4,
  },
  consCard: {
    flex: 1,
    backgroundColor: '#FEF2F2',
    borderRadius: 12,
    padding: 12,
  },
  consTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#DC2626',
    marginBottom: 8,
  },
  consItem: {
    fontSize: 13,
    color: '#B91C1C',
    marginBottom: 4,
  },
  effectivenessCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 12,
  },
  effectivenessRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  effectivenessLabel: {
    width: 100,
    fontSize: 14,
    color: '#374151',
  },
  effectivenessBar: {
    flex: 1,
    height: 8,
    backgroundColor: '#E5E7EB',
    borderRadius: 4,
    marginHorizontal: 8,
    overflow: 'hidden',
  },
  effectivenessFill: {
    height: '100%',
    borderRadius: 4,
  },
  effectivenessScore: {
    width: 40,
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    textAlign: 'right',
  },
  ingredientListCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  ingredientItem: {
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
    paddingVertical: 12,
  },
  ingredientHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  ingredientNameRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  ratingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  ingredientName: {
    fontSize: 15,
    color: '#1F2937',
    flex: 1,
  },
  ratingChip: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  ratingChipText: {
    fontSize: 12,
    fontWeight: '600',
  },
  ingredientDetails: {
    marginTop: 12,
    paddingLeft: 16,
    backgroundColor: '#F9FAFB',
    borderRadius: 8,
    padding: 12,
  },
  detailCategory: {
    fontSize: 13,
    color: '#6366F1',
    fontWeight: '600',
    marginBottom: 4,
  },
  detailReason: {
    fontSize: 13,
    color: '#6B7280',
    marginBottom: 8,
    fontStyle: 'italic',
  },
  detailRow: {
    flexDirection: 'row',
    marginBottom: 4,
  },
  detailLabel: {
    fontSize: 13,
    color: '#6B7280',
    width: 90,
  },
  detailValue: {
    fontSize: 13,
    color: '#374151',
    flex: 1,
  },
  benefitsList: {
    marginTop: 8,
  },
  benefitsTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#059669',
    marginBottom: 4,
  },
  benefitItem: {
    fontSize: 12,
    color: '#047857',
    marginLeft: 8,
  },
  concernsList: {
    marginTop: 8,
  },
  concernsTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#DC2626',
    marginBottom: 4,
  },
  concernItem: {
    fontSize: 12,
    color: '#B91C1C',
    marginLeft: 8,
  },
  alternativesCard: {
    backgroundColor: '#FEF3C7',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  alternativeItem: {
    marginBottom: 12,
  },
  alternativeSuggestion: {
    fontSize: 14,
    fontWeight: '600',
    color: '#92400E',
    marginBottom: 2,
  },
  alternativeReason: {
    fontSize: 13,
    color: '#B45309',
  },
  scanAnotherButton: {
    backgroundColor: '#6366F1',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  scanAnotherButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
