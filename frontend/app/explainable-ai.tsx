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
  Modal,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import { API_URL } from '../config';

const { width: screenWidth } = Dimensions.get('window');

interface HighlightedRegion {
  id?: number;
  x: number;
  y: number;
  width: number;
  height: number;
  importance_score: number;
  feature_type: string;
  description: string;
}

interface FeatureExplanation {
  feature_name: string;
  category: string;
  importance_score: number;
  description: string;
  clinical_significance: string;
  visual_indicator: string;
}

interface ABCDEScores {
  asymmetry: { score: number; description: string; max_score: number };
  border: { score: number; description: string; max_score: number };
  color: { score: number; description: string; colors_detected: string[]; max_score: number };
  diameter: { score: number; description: string; diameter_mm: number | null; max_score: number };
  evolution: { score: number | null; description: string; max_score: number };
  total_score: number;
  max_total: number;
  risk_level: string;
}

interface ExplainableResult {
  analysis_id: string;
  timestamp: string;
  prediction: {
    class: string;
    confidence: number;
    probabilities: { [key: string]: number };
  };
  visual_explanations: {
    original_image: string;
    grad_cam_heatmap: string;
    grad_cam_plus_heatmap: string | null;
    attention_overlay: string;
    region_highlights: string;
    abcde_annotated: string;
    feature_importance_chart: string;
  };
  important_regions: HighlightedRegion[];
  feature_explanations: FeatureExplanation[];
  feature_importance_scores: { [key: string]: number };
  abcde_analysis: ABCDEScores;
  explanations: {
    summary: string;
    detailed: string;
    clinical_reasoning: string;
  };
  dermatologist_comparison: any | null;
}

type ViewMode = 'original' | 'grad_cam' | 'grad_cam_plus' | 'regions' | 'abcde';

export default function ExplainableAI() {
  const { analysisId } = useLocalSearchParams();
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState<ExplainableResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('grad_cam');
  const [showExplanationModal, setShowExplanationModal] = useState(false);
  const [explanationLevel, setExplanationLevel] = useState<'summary' | 'detailed' | 'clinical'>('detailed');
  const [selectedRegion, setSelectedRegion] = useState<HighlightedRegion | null>(null);
  const [showRegionModal, setShowRegionModal] = useState(false);

  useEffect(() => {
    if (analysisId) {
      fetchExplainableAI();
    }
  }, [analysisId]);

  const fetchExplainableAI = async () => {
    try {
      setLoading(true);
      setError(null);

      const token = await SecureStore.getItemAsync('auth_token');
      if (!token) {
        router.replace('/');
        return;
      }

      const response = await fetch(
        `${API_URL}/analysis/${analysisId}/explainable-ai?include_grad_cam_plus=true`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error('Failed to fetch explainable AI results');
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getViewImage = () => {
    if (!result) return null;

    switch (viewMode) {
      case 'original':
        return result.visual_explanations.original_image;
      case 'grad_cam':
        return result.visual_explanations.grad_cam_heatmap;
      case 'grad_cam_plus':
        return result.visual_explanations.grad_cam_plus_heatmap || result.visual_explanations.grad_cam_heatmap;
      case 'regions':
        return result.visual_explanations.region_highlights;
      case 'abcde':
        return result.visual_explanations.abcde_annotated;
      default:
        return result.visual_explanations.grad_cam_heatmap;
    }
  };

  const getRiskColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high':
        return '#dc3545';
      case 'moderate':
        return '#ffc107';
      case 'low':
        return '#28a745';
      default:
        return '#6c757d';
    }
  };

  const getImportanceColor = (score: number) => {
    if (score > 0.7) return '#dc3545';
    if (score > 0.4) return '#ffc107';
    return '#28a745';
  };

  const renderViewModeSelector = () => (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      style={styles.viewModeContainer}
    >
      {[
        { key: 'original', label: 'Original' },
        { key: 'grad_cam', label: 'Grad-CAM' },
        { key: 'grad_cam_plus', label: 'Grad-CAM++' },
        { key: 'regions', label: 'Regions' },
        { key: 'abcde', label: 'ABCDE' },
      ].map((mode) => (
        <TouchableOpacity
          key={mode.key}
          style={[
            styles.viewModeButton,
            viewMode === mode.key && styles.viewModeButtonActive,
          ]}
          onPress={() => setViewMode(mode.key as ViewMode)}
        >
          <Text
            style={[
              styles.viewModeText,
              viewMode === mode.key && styles.viewModeTextActive,
            ]}
          >
            {mode.label}
          </Text>
        </TouchableOpacity>
      ))}
    </ScrollView>
  );

  const renderPredictionSummary = () => {
    if (!result) return null;

    return (
      <View style={styles.predictionCard}>
        <Text style={styles.predictionLabel}>AI Classification</Text>
        <Text style={styles.predictionClass}>{result.prediction.class}</Text>
        <View style={styles.confidenceBar}>
          <View
            style={[
              styles.confidenceFill,
              { width: `${result.prediction.confidence * 100}%` }
            ]}
          />
        </View>
        <Text style={styles.confidenceText}>
          Confidence: {(result.prediction.confidence * 100).toFixed(1)}%
        </Text>
      </View>
    );
  };

  const renderFeatureImportance = () => {
    if (!result) return null;

    const sortedFeatures = Object.entries(result.feature_importance_scores)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 6);

    return (
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Feature Importance</Text>
        <Text style={styles.sectionSubtitle}>
          Visual features that influenced the AI's decision
        </Text>

        {sortedFeatures.map(([feature, score], index) => (
          <TouchableOpacity
            key={feature}
            style={styles.featureRow}
            onPress={() => {
              const explanation = result.feature_explanations.find(
                f => f.feature_name === feature
              );
              if (explanation) {
                Alert.alert(
                  feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
                  `${explanation.description}\n\nClinical Significance:\n${explanation.clinical_significance}\n\nWhat to look for:\n${explanation.visual_indicator}`
                );
              }
            }}
          >
            <View style={styles.featureInfo}>
              <Text style={styles.featureRank}>#{index + 1}</Text>
              <Text style={styles.featureName}>
                {feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              </Text>
            </View>
            <View style={styles.featureBarContainer}>
              <View
                style={[
                  styles.featureBar,
                  {
                    width: `${score * 100}%`,
                    backgroundColor: getImportanceColor(score)
                  }
                ]}
              />
            </View>
            <Text style={[styles.featureScore, { color: getImportanceColor(score) }]}>
              {(score * 100).toFixed(0)}%
            </Text>
          </TouchableOpacity>
        ))}

        <TouchableOpacity
          style={styles.viewChartButton}
          onPress={() => {
            if (result.visual_explanations.feature_importance_chart) {
              Alert.alert(
                'Feature Importance Chart',
                'View the full feature importance visualization in the image viewer above by selecting the chart view.'
              );
            }
          }}
        >
          <Text style={styles.viewChartText}>Tap any feature for details</Text>
        </TouchableOpacity>
      </View>
    );
  };

  const renderABCDEAnalysis = () => {
    if (!result) return null;

    const { abcde_analysis } = result;

    return (
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>ABCDE Criteria Analysis</Text>
        <Text style={styles.sectionSubtitle}>
          Standard dermatological assessment criteria
        </Text>

        <View style={[styles.riskBadge, { backgroundColor: getRiskColor(abcde_analysis.risk_level) }]}>
          <Text style={styles.riskBadgeText}>
            Risk Level: {abcde_analysis.risk_level.toUpperCase()}
          </Text>
          <Text style={styles.riskScore}>
            Score: {abcde_analysis.total_score.toFixed(1)}/{abcde_analysis.max_total}
          </Text>
        </View>

        {/* Asymmetry */}
        <View style={styles.abcdeItem}>
          <View style={styles.abcdeHeader}>
            <Text style={styles.abcdeLetter}>A</Text>
            <Text style={styles.abcdeTitle}>Asymmetry</Text>
            <Text style={styles.abcdeScore}>
              {abcde_analysis.asymmetry.score}/{abcde_analysis.asymmetry.max_score}
            </Text>
          </View>
          <Text style={styles.abcdeDescription}>{abcde_analysis.asymmetry.description}</Text>
        </View>

        {/* Border */}
        <View style={styles.abcdeItem}>
          <View style={styles.abcdeHeader}>
            <Text style={styles.abcdeLetter}>B</Text>
            <Text style={styles.abcdeTitle}>Border</Text>
            <Text style={styles.abcdeScore}>
              {abcde_analysis.border.score}/{abcde_analysis.border.max_score}
            </Text>
          </View>
          <Text style={styles.abcdeDescription}>{abcde_analysis.border.description}</Text>
        </View>

        {/* Color */}
        <View style={styles.abcdeItem}>
          <View style={styles.abcdeHeader}>
            <Text style={styles.abcdeLetter}>C</Text>
            <Text style={styles.abcdeTitle}>Color</Text>
            <Text style={styles.abcdeScore}>
              {abcde_analysis.color.score}/{abcde_analysis.color.max_score}
            </Text>
          </View>
          <Text style={styles.abcdeDescription}>{abcde_analysis.color.description}</Text>
          {abcde_analysis.color.colors_detected.length > 0 && (
            <View style={styles.colorsContainer}>
              {abcde_analysis.color.colors_detected.map((color, idx) => (
                <View key={idx} style={styles.colorTag}>
                  <Text style={styles.colorTagText}>{color.replace(/_/g, ' ')}</Text>
                </View>
              ))}
            </View>
          )}
        </View>

        {/* Diameter */}
        <View style={styles.abcdeItem}>
          <View style={styles.abcdeHeader}>
            <Text style={styles.abcdeLetter}>D</Text>
            <Text style={styles.abcdeTitle}>Diameter</Text>
            <Text style={styles.abcdeScore}>
              {abcde_analysis.diameter.score}/{abcde_analysis.diameter.max_score}
            </Text>
          </View>
          <Text style={styles.abcdeDescription}>{abcde_analysis.diameter.description}</Text>
          {abcde_analysis.diameter.diameter_mm && (
            <Text style={styles.diameterValue}>
              Measured: {abcde_analysis.diameter.diameter_mm.toFixed(1)}mm
            </Text>
          )}
        </View>

        {/* Evolution */}
        <View style={styles.abcdeItem}>
          <View style={styles.abcdeHeader}>
            <Text style={styles.abcdeLetter}>E</Text>
            <Text style={styles.abcdeTitle}>Evolution</Text>
            <Text style={styles.abcdeScore}>
              {abcde_analysis.evolution.score !== null
                ? `${abcde_analysis.evolution.score}/${abcde_analysis.evolution.max_score}`
                : 'N/A'}
            </Text>
          </View>
          <Text style={styles.abcdeDescription}>{abcde_analysis.evolution.description}</Text>
        </View>
      </View>
    );
  };

  const renderHighlightedRegions = () => {
    if (!result || result.important_regions.length === 0) return null;

    return (
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Highlighted Regions</Text>
        <Text style={styles.sectionSubtitle}>
          Areas the AI focused on for classification
        </Text>

        {result.important_regions.map((region, index) => (
          <TouchableOpacity
            key={index}
            style={styles.regionCard}
            onPress={() => {
              setSelectedRegion(region);
              setShowRegionModal(true);
            }}
          >
            <View style={styles.regionHeader}>
              <View style={[
                styles.regionBadge,
                { backgroundColor: getImportanceColor(region.importance_score) }
              ]}>
                <Text style={styles.regionBadgeText}>#{index + 1}</Text>
              </View>
              <Text style={styles.regionType}>
                {region.feature_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              </Text>
              <Text style={[
                styles.regionScore,
                { color: getImportanceColor(region.importance_score) }
              ]}>
                {(region.importance_score * 100).toFixed(0)}%
              </Text>
            </View>
            <Text style={styles.regionDescription} numberOfLines={2}>
              {region.description}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    );
  };

  const renderExplanationSection = () => {
    if (!result) return null;

    return (
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>AI Explanation</Text>

        <View style={styles.explanationTabs}>
          {(['summary', 'detailed', 'clinical'] as const).map((level) => (
            <TouchableOpacity
              key={level}
              style={[
                styles.explanationTab,
                explanationLevel === level && styles.explanationTabActive,
              ]}
              onPress={() => setExplanationLevel(level)}
            >
              <Text
                style={[
                  styles.explanationTabText,
                  explanationLevel === level && styles.explanationTabTextActive,
                ]}
              >
                {level.charAt(0).toUpperCase() + level.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.explanationContent}>
          <Text style={styles.explanationText}>
            {explanationLevel === 'summary' && result.explanations.summary}
            {explanationLevel === 'detailed' && result.explanations.detailed}
            {explanationLevel === 'clinical' && result.explanations.clinical_reasoning}
          </Text>
        </View>

        <TouchableOpacity
          style={styles.fullExplanationButton}
          onPress={() => setShowExplanationModal(true)}
        >
          <Text style={styles.fullExplanationText}>View Full Explanation</Text>
        </TouchableOpacity>
      </View>
    );
  };

  const renderExplanationModal = () => (
    <Modal
      visible={showExplanationModal}
      animationType="slide"
      onRequestClose={() => setShowExplanationModal(false)}
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <Text style={styles.modalTitle}>AI Clinical Reasoning</Text>
          <TouchableOpacity onPress={() => setShowExplanationModal(false)}>
            <Text style={styles.modalClose}>Close</Text>
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.modalContent}>
          <Text style={styles.modalSectionTitle}>Summary</Text>
          <Text style={styles.modalText}>{result?.explanations.summary}</Text>

          <Text style={styles.modalSectionTitle}>Detailed Analysis</Text>
          <Text style={styles.modalText}>{result?.explanations.detailed}</Text>

          <Text style={styles.modalSectionTitle}>Clinical Reasoning</Text>
          <Text style={styles.modalText}>{result?.explanations.clinical_reasoning}</Text>
        </ScrollView>
      </View>
    </Modal>
  );

  const renderRegionModal = () => (
    <Modal
      visible={showRegionModal}
      animationType="fade"
      transparent
      onRequestClose={() => setShowRegionModal(false)}
    >
      <View style={styles.regionModalOverlay}>
        <View style={styles.regionModalContent}>
          <Text style={styles.regionModalTitle}>
            Region Details
          </Text>

          {selectedRegion && (
            <>
              <View style={styles.regionModalInfo}>
                <Text style={styles.regionModalLabel}>Feature Type:</Text>
                <Text style={styles.regionModalValue}>
                  {selectedRegion.feature_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                </Text>
              </View>

              <View style={styles.regionModalInfo}>
                <Text style={styles.regionModalLabel}>Importance:</Text>
                <Text style={[
                  styles.regionModalValue,
                  { color: getImportanceColor(selectedRegion.importance_score) }
                ]}>
                  {(selectedRegion.importance_score * 100).toFixed(1)}%
                </Text>
              </View>

              <View style={styles.regionModalInfo}>
                <Text style={styles.regionModalLabel}>Location:</Text>
                <Text style={styles.regionModalValue}>
                  ({selectedRegion.x}, {selectedRegion.y}) - {selectedRegion.width}x{selectedRegion.height}px
                </Text>
              </View>

              <Text style={styles.regionModalDescription}>
                {selectedRegion.description}
              </Text>
            </>
          )}

          <TouchableOpacity
            style={styles.regionModalButton}
            onPress={() => setShowRegionModal(false)}
          >
            <Text style={styles.regionModalButtonText}>Close</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Generating AI Explanations...</Text>
        <Text style={styles.loadingSubtext}>
          This may take a moment as we analyze the image
        </Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorTitle}>Error</Text>
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={fetchExplainableAI}>
          <Text style={styles.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Text style={styles.backButton}>Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Explainable AI</Text>
        <View style={{ width: 50 }} />
      </View>

      {renderPredictionSummary()}

      <View style={styles.imageSection}>
        <Text style={styles.imageSectionTitle}>Visual Feature Attribution</Text>
        <Text style={styles.imageSectionSubtitle}>
          See what the AI "saw" when making its decision
        </Text>

        {renderViewModeSelector()}

        <View style={styles.imageContainer}>
          {getViewImage() && (
            <Image
              source={{ uri: getViewImage() }}
              style={styles.analysisImage}
              resizeMode="contain"
            />
          )}
        </View>

        <View style={styles.viewModeDescription}>
          {viewMode === 'original' && (
            <Text style={styles.viewModeDescText}>Original uploaded image</Text>
          )}
          {viewMode === 'grad_cam' && (
            <Text style={styles.viewModeDescText}>
              Grad-CAM heatmap showing regions that influenced the AI's classification.
              Red/yellow areas had the highest influence.
            </Text>
          )}
          {viewMode === 'grad_cam_plus' && (
            <Text style={styles.viewModeDescText}>
              Grad-CAM++ provides improved localization with better focus on the lesion boundaries.
            </Text>
          )}
          {viewMode === 'regions' && (
            <Text style={styles.viewModeDescText}>
              Bounding boxes highlight specific regions of interest identified by the AI.
            </Text>
          )}
          {viewMode === 'abcde' && (
            <Text style={styles.viewModeDescText}>
              ABCDE criteria annotations showing Asymmetry, Border, Color, Diameter analysis.
            </Text>
          )}
        </View>
      </View>

      {renderFeatureImportance()}
      {renderABCDEAnalysis()}
      {renderHighlightedRegions()}
      {renderExplanationSection()}

      <View style={styles.disclaimer}>
        <Text style={styles.disclaimerTitle}>Important Notice</Text>
        <Text style={styles.disclaimerText}>
          AI analysis is provided for educational and informational purposes only.
          It is not a substitute for professional medical diagnosis. Always consult
          a qualified dermatologist for proper evaluation and treatment.
        </Text>
      </View>

      {renderExplanationModal()}
      {renderRegionModal()}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  loadingSubtext: {
    marginTop: 8,
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#dc3545',
    marginBottom: 10,
  },
  errorText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 20,
  },
  retryButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  backButton: {
    fontSize: 16,
    color: '#007AFF',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  predictionCard: {
    backgroundColor: '#fff',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  predictionLabel: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  predictionClass: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  confidenceBar: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    marginBottom: 8,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 4,
  },
  confidenceText: {
    fontSize: 14,
    color: '#666',
  },
  imageSection: {
    backgroundColor: '#fff',
    margin: 16,
    marginTop: 0,
    padding: 16,
    borderRadius: 12,
  },
  imageSectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  imageSectionSubtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 12,
  },
  viewModeContainer: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  viewModeButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    borderRadius: 20,
    backgroundColor: '#f0f0f0',
  },
  viewModeButtonActive: {
    backgroundColor: '#007AFF',
  },
  viewModeText: {
    fontSize: 14,
    color: '#666',
  },
  viewModeTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  imageContainer: {
    backgroundColor: '#000',
    borderRadius: 8,
    overflow: 'hidden',
    minHeight: 250,
  },
  analysisImage: {
    width: '100%',
    height: 250,
  },
  viewModeDescription: {
    marginTop: 12,
    padding: 12,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
  },
  viewModeDescText: {
    fontSize: 13,
    color: '#666',
    lineHeight: 18,
  },
  section: {
    backgroundColor: '#fff',
    margin: 16,
    marginTop: 0,
    padding: 16,
    borderRadius: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
  },
  featureRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  featureInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    width: 120,
  },
  featureRank: {
    fontSize: 12,
    color: '#999',
    width: 24,
  },
  featureName: {
    fontSize: 13,
    color: '#333',
    flex: 1,
  },
  featureBarContainer: {
    flex: 1,
    height: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
    marginHorizontal: 10,
    overflow: 'hidden',
  },
  featureBar: {
    height: '100%',
    borderRadius: 4,
  },
  featureScore: {
    fontSize: 13,
    fontWeight: '600',
    width: 40,
    textAlign: 'right',
  },
  viewChartButton: {
    marginTop: 12,
    alignItems: 'center',
  },
  viewChartText: {
    fontSize: 13,
    color: '#007AFF',
  },
  riskBadge: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  riskBadgeText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  riskScore: {
    fontSize: 14,
    color: '#fff',
  },
  abcdeItem: {
    marginBottom: 16,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  abcdeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  abcdeLetter: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#007AFF',
    width: 30,
  },
  abcdeTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    flex: 1,
  },
  abcdeScore: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  abcdeDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginLeft: 30,
  },
  colorsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
    marginLeft: 30,
  },
  colorTag: {
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    marginRight: 8,
    marginBottom: 4,
  },
  colorTagText: {
    fontSize: 12,
    color: '#666',
  },
  diameterValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#007AFF',
    marginLeft: 30,
    marginTop: 4,
  },
  regionCard: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
  },
  regionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  regionBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  regionBadgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  regionType: {
    flex: 1,
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  regionScore: {
    fontSize: 14,
    fontWeight: '600',
  },
  regionDescription: {
    fontSize: 13,
    color: '#666',
    lineHeight: 18,
  },
  explanationTabs: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  explanationTab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    marginRight: 4,
    borderRadius: 8,
  },
  explanationTabActive: {
    backgroundColor: '#007AFF',
  },
  explanationTabText: {
    fontSize: 14,
    color: '#666',
  },
  explanationTabTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  explanationContent: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    maxHeight: 200,
  },
  explanationText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 22,
  },
  fullExplanationButton: {
    marginTop: 12,
    alignItems: 'center',
  },
  fullExplanationText: {
    fontSize: 14,
    color: '#007AFF',
    fontWeight: '600',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  modalClose: {
    fontSize: 16,
    color: '#007AFF',
  },
  modalContent: {
    flex: 1,
    padding: 16,
  },
  modalSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginTop: 16,
    marginBottom: 8,
  },
  modalText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 22,
  },
  regionModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  regionModalContent: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    width: '100%',
    maxWidth: 350,
  },
  regionModalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 16,
  },
  regionModalInfo: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  regionModalLabel: {
    fontSize: 14,
    color: '#666',
    width: 100,
  },
  regionModalValue: {
    flex: 1,
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  regionModalDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginTop: 8,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  regionModalButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 16,
  },
  regionModalButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  disclaimer: {
    margin: 16,
    marginTop: 0,
    padding: 16,
    backgroundColor: '#fff3cd',
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#ffc107',
  },
  disclaimerTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#856404',
    marginBottom: 4,
  },
  disclaimerText: {
    fontSize: 13,
    color: '#856404',
    lineHeight: 18,
  },
});
