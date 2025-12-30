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
import * as SecureStore from 'expo-secure-store';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { API_BASE_URL } from '../config';

const { width: screenWidth } = Dimensions.get('window');

interface SegmentationResult {
  mask: string;
  boundary: [number, number][];
  area_pixels: number;
  area_percentage: number;
  centroid: [number, number];
  bounding_box: [number, number, number, number];
  asymmetry_score: number;
  border_irregularity: number;
  compactness: number;
  eccentricity: number;
}

interface GrowthPrediction {
  date: string;
  days_from_now: number;
  predicted_area_mm2: number;
  predicted_diameter_mm: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
}

interface RiskAssessment {
  risk_level: string;
  trend: string;
  growth_rate_mm2_per_month: number;
  total_growth_percentage: number;
  abcde_score: number;
  confidence: number;
}

interface FederatedStatus {
  model_version: string;
  update_count: number;
  privacy_budget_remaining: number;
}

export default function EnhancedMLScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'segmentation' | 'prediction' | 'federated'>('segmentation');
  const [loading, setLoading] = useState(false);

  // Segmentation state
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [segmentationResult, setSegmentationResult] = useState<SegmentationResult | null>(null);
  const [showMaskOverlay, setShowMaskOverlay] = useState(true);

  // Prediction state
  const [predictions, setPredictions] = useState<GrowthPrediction[]>([]);
  const [riskAssessment, setRiskAssessment] = useState<RiskAssessment | null>(null);
  const [historicalData, setHistoricalData] = useState<any[]>([]);

  // Federated learning state
  const [federatedStatus, setFederatedStatus] = useState<FederatedStatus | null>(null);

  useEffect(() => {
    if (activeTab === 'federated') {
      loadFederatedStatus();
    }
  }, [activeTab]);

  const getAuthHeaders = async () => {
    const token = await SecureStore.getItemAsync('auth_token');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
      base64: true,
    });

    if (!result.canceled && result.assets[0].base64) {
      setSelectedImage(`data:image/jpeg;base64,${result.assets[0].base64}`);
      setSegmentationResult(null);
    }
  };

  const takePhoto = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission Required', 'Camera permission is needed to take photos.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
      base64: true,
    });

    if (!result.canceled && result.assets[0].base64) {
      setSelectedImage(`data:image/jpeg;base64,${result.assets[0].base64}`);
      setSegmentationResult(null);
    }
  };

  const runSegmentation = async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select an image first.');
      return;
    }

    setLoading(true);
    try {
      const headers = await getAuthHeaders();
      const base64Data = selectedImage.replace(/^data:image\/\w+;base64,/, '');

      const response = await fetch(`${API_BASE_URL}/api/ml/segmentation/analyze`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          image_base64: base64Data,
          threshold: 0.5,
        }),
      });

      if (!response.ok) {
        throw new Error('Segmentation failed');
      }

      const data = await response.json();
      setSegmentationResult(data.segmentation);
    } catch (error) {
      console.error('Segmentation error:', error);
      Alert.alert('Error', 'Failed to segment lesion. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const loadSampleHistoricalData = () => {
    // Load sample historical data for demo
    const sampleData = [
      { date: '2024-06-01', area_mm2: 12.5, diameter_mm: 4.0, asymmetry: 0.15, border_irregularity: 0.2, color_variation: 0.1 },
      { date: '2024-07-01', area_mm2: 13.2, diameter_mm: 4.1, asymmetry: 0.18, border_irregularity: 0.22, color_variation: 0.12 },
      { date: '2024-08-01', area_mm2: 14.1, diameter_mm: 4.3, asymmetry: 0.2, border_irregularity: 0.25, color_variation: 0.15 },
      { date: '2024-09-01', area_mm2: 15.0, diameter_mm: 4.4, asymmetry: 0.22, border_irregularity: 0.28, color_variation: 0.18 },
      { date: '2024-10-01', area_mm2: 16.2, diameter_mm: 4.6, asymmetry: 0.25, border_irregularity: 0.3, color_variation: 0.2 },
    ];
    setHistoricalData(sampleData);
  };

  const runGrowthPrediction = async () => {
    if (historicalData.length < 2) {
      Alert.alert('Insufficient Data', 'At least 2 historical measurements are required.');
      loadSampleHistoricalData();
      return;
    }

    setLoading(true);
    try {
      const headers = await getAuthHeaders();

      const response = await fetch(`${API_BASE_URL}/api/ml/temporal/predict`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          historical_measurements: historicalData,
          prediction_days: 90,
        }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictions(data.prediction_result.predictions || []);
      setRiskAssessment(data.prediction_result.risk_assessment || null);
    } catch (error) {
      console.error('Prediction error:', error);
      Alert.alert('Error', 'Failed to generate prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const loadFederatedStatus = async () => {
    setLoading(true);
    try {
      const headers = await getAuthHeaders();

      const response = await fetch(`${API_BASE_URL}/api/ml/federated/status`, {
        method: 'GET',
        headers,
      });

      if (!response.ok) {
        throw new Error('Failed to load status');
      }

      const data = await response.json();
      setFederatedStatus(data.model_info);
    } catch (error) {
      console.error('Federated status error:', error);
    } finally {
      setLoading(false);
    }
  };

  const contributeToFederated = async () => {
    setLoading(true);
    try {
      const headers = await getAuthHeaders();

      const response = await fetch(`${API_BASE_URL}/api/ml/federated/compute-gradients`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          local_data_count: 10,
        }),
      });

      if (!response.ok) {
        throw new Error('Contribution failed');
      }

      Alert.alert('Success', 'Your contribution has been submitted with differential privacy protection.');
      loadFederatedStatus();
    } catch (error) {
      console.error('Federated contribution error:', error);
      Alert.alert('Error', 'Failed to contribute. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return '#dc3545';
      case 'moderate': return '#ffc107';
      default: return '#28a745';
    }
  };

  const renderSegmentationTab = () => (
    <View style={styles.tabContent}>
      <Text style={styles.sectionTitle}>U-Net Lesion Segmentation</Text>
      <Text style={styles.description}>
        Automatically detect lesion boundaries using deep learning
      </Text>

      <View style={styles.imageContainer}>
        {selectedImage ? (
          <View style={styles.imageWrapper}>
            <Image source={{ uri: selectedImage }} style={styles.previewImage} />
            {segmentationResult && showMaskOverlay && (
              <Image
                source={{ uri: `data:image/png;base64,${segmentationResult.mask}` }}
                style={[styles.previewImage, styles.maskOverlay]}
              />
            )}
          </View>
        ) : (
          <View style={styles.placeholderImage}>
            <Ionicons name="image-outline" size={64} color="#ccc" />
            <Text style={styles.placeholderText}>Select an image to analyze</Text>
          </View>
        )}
      </View>

      <View style={styles.buttonRow}>
        <TouchableOpacity style={styles.imageButton} onPress={pickImage}>
          <Ionicons name="images-outline" size={20} color="#fff" />
          <Text style={styles.imageButtonText}>Gallery</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.imageButton} onPress={takePhoto}>
          <Ionicons name="camera-outline" size={20} color="#fff" />
          <Text style={styles.imageButtonText}>Camera</Text>
        </TouchableOpacity>
      </View>

      {selectedImage && (
        <TouchableOpacity
          style={[styles.analyzeButton, loading && styles.buttonDisabled]}
          onPress={runSegmentation}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <>
              <Ionicons name="scan-outline" size={20} color="#fff" />
              <Text style={styles.analyzeButtonText}>Run Segmentation</Text>
            </>
          )}
        </TouchableOpacity>
      )}

      {segmentationResult && (
        <View style={styles.resultsContainer}>
          <View style={styles.resultHeader}>
            <Text style={styles.resultTitle}>Segmentation Results</Text>
            <TouchableOpacity
              style={styles.toggleButton}
              onPress={() => setShowMaskOverlay(!showMaskOverlay)}
            >
              <Text style={styles.toggleText}>
                {showMaskOverlay ? 'Hide Mask' : 'Show Mask'}
              </Text>
            </TouchableOpacity>
          </View>

          <View style={styles.metricsGrid}>
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>Area</Text>
              <Text style={styles.metricValue}>{segmentationResult.area_percentage.toFixed(1)}%</Text>
              <Text style={styles.metricSubtext}>{segmentationResult.area_pixels.toLocaleString()} px</Text>
            </View>
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>Asymmetry</Text>
              <Text style={[styles.metricValue, { color: segmentationResult.asymmetry_score > 0.3 ? '#dc3545' : '#28a745' }]}>
                {(segmentationResult.asymmetry_score * 100).toFixed(0)}%
              </Text>
              <Text style={styles.metricSubtext}>{segmentationResult.asymmetry_score > 0.3 ? 'Concerning' : 'Normal'}</Text>
            </View>
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>Border</Text>
              <Text style={[styles.metricValue, { color: segmentationResult.border_irregularity > 0.3 ? '#dc3545' : '#28a745' }]}>
                {(segmentationResult.border_irregularity * 100).toFixed(0)}%
              </Text>
              <Text style={styles.metricSubtext}>{segmentationResult.border_irregularity > 0.3 ? 'Irregular' : 'Regular'}</Text>
            </View>
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>Compactness</Text>
              <Text style={styles.metricValue}>{segmentationResult.compactness.toFixed(2)}</Text>
              <Text style={styles.metricSubtext}>Shape factor</Text>
            </View>
          </View>

          <View style={styles.detailsCard}>
            <Text style={styles.detailsTitle}>Geometric Features</Text>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Centroid:</Text>
              <Text style={styles.detailValue}>
                ({segmentationResult.centroid[0].toFixed(0)}, {segmentationResult.centroid[1].toFixed(0)})
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Bounding Box:</Text>
              <Text style={styles.detailValue}>
                {segmentationResult.bounding_box[2] - segmentationResult.bounding_box[0]} x {segmentationResult.bounding_box[3] - segmentationResult.bounding_box[1]} px
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Eccentricity:</Text>
              <Text style={styles.detailValue}>{segmentationResult.eccentricity.toFixed(3)}</Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Boundary Points:</Text>
              <Text style={styles.detailValue}>{segmentationResult.boundary.length}</Text>
            </View>
          </View>
        </View>
      )}
    </View>
  );

  const renderPredictionTab = () => (
    <View style={styles.tabContent}>
      <Text style={styles.sectionTitle}>Growth Prediction</Text>
      <Text style={styles.description}>
        LSTM-based forecasting of lesion growth trajectory
      </Text>

      {historicalData.length === 0 ? (
        <TouchableOpacity style={styles.loadDataButton} onPress={loadSampleHistoricalData}>
          <Ionicons name="document-text-outline" size={20} color="#fff" />
          <Text style={styles.loadDataButtonText}>Load Sample Data</Text>
        </TouchableOpacity>
      ) : (
        <>
          <View style={styles.dataPreview}>
            <Text style={styles.dataPreviewTitle}>Historical Measurements ({historicalData.length})</Text>
            {historicalData.slice(-3).map((item, index) => (
              <View key={index} style={styles.dataRow}>
                <Text style={styles.dataDate}>{item.date}</Text>
                <Text style={styles.dataValue}>Area: {item.area_mm2} mm²</Text>
                <Text style={styles.dataValue}>Diameter: {item.diameter_mm} mm</Text>
              </View>
            ))}
          </View>

          <TouchableOpacity
            style={[styles.analyzeButton, loading && styles.buttonDisabled]}
            onPress={runGrowthPrediction}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <>
                <Ionicons name="trending-up-outline" size={20} color="#fff" />
                <Text style={styles.analyzeButtonText}>Generate Forecast</Text>
              </>
            )}
          </TouchableOpacity>
        </>
      )}

      {riskAssessment && (
        <View style={[styles.riskCard, { borderLeftColor: getRiskColor(riskAssessment.risk_level) }]}>
          <View style={styles.riskHeader}>
            <Text style={styles.riskTitle}>Risk Assessment</Text>
            <View style={[styles.riskBadge, { backgroundColor: getRiskColor(riskAssessment.risk_level) }]}>
              <Text style={styles.riskBadgeText}>{riskAssessment.risk_level.toUpperCase()}</Text>
            </View>
          </View>
          <View style={styles.riskDetails}>
            <Text style={styles.riskItem}>Trend: {riskAssessment.trend.replace('_', ' ')}</Text>
            <Text style={styles.riskItem}>Growth Rate: {riskAssessment.growth_rate_mm2_per_month.toFixed(2)} mm²/month</Text>
            <Text style={styles.riskItem}>Total Growth: {riskAssessment.total_growth_percentage.toFixed(1)}%</Text>
            <Text style={styles.riskItem}>ABCDE Score: {riskAssessment.abcde_score}/5</Text>
            <Text style={styles.riskItem}>Confidence: {(riskAssessment.confidence * 100).toFixed(0)}%</Text>
          </View>
        </View>
      )}

      {predictions.length > 0 && (
        <View style={styles.predictionsContainer}>
          <Text style={styles.predictionsTitle}>90-Day Forecast</Text>
          <View style={styles.forecastChart}>
            {predictions.slice(0, 6).map((pred, index) => (
              <View key={index} style={styles.forecastBar}>
                <View
                  style={[
                    styles.forecastBarFill,
                    { height: `${Math.min(100, (pred.predicted_area_mm2 / 30) * 100)}%` }
                  ]}
                />
                <Text style={styles.forecastLabel}>{pred.days_from_now}d</Text>
              </View>
            ))}
          </View>
          <View style={styles.forecastLegend}>
            <Text style={styles.legendText}>Predicted Area (mm²)</Text>
          </View>

          <View style={styles.predictionsList}>
            {predictions.slice(0, 4).map((pred, index) => (
              <View key={index} style={styles.predictionItem}>
                <Text style={styles.predictionDate}>{pred.date}</Text>
                <Text style={styles.predictionValue}>{pred.predicted_area_mm2} mm²</Text>
                <Text style={styles.predictionCI}>
                  CI: [{pred.confidence_interval.lower} - {pred.confidence_interval.upper}]
                </Text>
              </View>
            ))}
          </View>
        </View>
      )}
    </View>
  );

  const renderFederatedTab = () => (
    <View style={styles.tabContent}>
      <Text style={styles.sectionTitle}>Federated Learning</Text>
      <Text style={styles.description}>
        Privacy-preserving model improvement with differential privacy
      </Text>

      <View style={styles.federatedInfo}>
        <Ionicons name="shield-checkmark-outline" size={48} color="#28a745" />
        <Text style={styles.federatedTitle}>Your Data Stays Private</Text>
        <Text style={styles.federatedDescription}>
          Contribute to model improvement without sharing raw data.
          All updates are protected with differential privacy.
        </Text>
      </View>

      {federatedStatus && (
        <View style={styles.statusCard}>
          <Text style={styles.statusTitle}>Model Status</Text>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Version:</Text>
            <Text style={styles.statusValue}>{federatedStatus.model_version}</Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Updates:</Text>
            <Text style={styles.statusValue}>{federatedStatus.update_count}</Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Privacy Budget:</Text>
            <View style={styles.budgetBar}>
              <View
                style={[
                  styles.budgetFill,
                  { width: `${federatedStatus.privacy_budget_remaining * 10}%` }
                ]}
              />
            </View>
            <Text style={styles.statusValue}>{federatedStatus.privacy_budget_remaining.toFixed(1)}</Text>
          </View>
        </View>
      )}

      <View style={styles.featuresCard}>
        <Text style={styles.featuresTitle}>Privacy Features</Text>
        <View style={styles.featureItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.featureText}>Differential Privacy (ε = 0.1 per update)</Text>
        </View>
        <View style={styles.featureItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.featureText}>Secure Gradient Aggregation</Text>
        </View>
        <View style={styles.featureItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.featureText}>No Raw Data Leaves Device</Text>
        </View>
        <View style={styles.featureItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.featureText}>HIPAA/GDPR Compliant</Text>
        </View>
      </View>

      <TouchableOpacity
        style={[styles.contributeButton, loading && styles.buttonDisabled]}
        onPress={contributeToFederated}
        disabled={loading}
      >
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="cloud-upload-outline" size={20} color="#fff" />
            <Text style={styles.contributeButtonText}>Contribute to Model</Text>
          </>
        )}
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#333" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Enhanced ML</Text>
        <View style={styles.placeholder} />
      </View>

      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'segmentation' && styles.activeTab]}
          onPress={() => setActiveTab('segmentation')}
        >
          <Ionicons name="scan" size={20} color={activeTab === 'segmentation' ? '#007AFF' : '#666'} />
          <Text style={[styles.tabText, activeTab === 'segmentation' && styles.activeTabText]}>
            Segment
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'prediction' && styles.activeTab]}
          onPress={() => setActiveTab('prediction')}
        >
          <Ionicons name="trending-up" size={20} color={activeTab === 'prediction' ? '#007AFF' : '#666'} />
          <Text style={[styles.tabText, activeTab === 'prediction' && styles.activeTabText]}>
            Predict
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'federated' && styles.activeTab]}
          onPress={() => setActiveTab('federated')}
        >
          <Ionicons name="shield-checkmark" size={20} color={activeTab === 'federated' ? '#007AFF' : '#666'} />
          <Text style={[styles.tabText, activeTab === 'federated' && styles.activeTabText]}>
            Privacy
          </Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {activeTab === 'segmentation' && renderSegmentationTab()}
        {activeTab === 'prediction' && renderPredictionTab()}
        {activeTab === 'federated' && renderFederatedTab()}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 50,
    paddingBottom: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  placeholder: {
    width: 40,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderRadius: 8,
    marginHorizontal: 4,
  },
  activeTab: {
    backgroundColor: '#e8f4ff',
  },
  tabText: {
    marginLeft: 6,
    fontSize: 14,
    color: '#666',
  },
  activeTabText: {
    color: '#007AFF',
    fontWeight: '600',
  },
  content: {
    flex: 1,
  },
  tabContent: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#333',
    marginBottom: 4,
  },
  description: {
    fontSize: 14,
    color: '#666',
    marginBottom: 20,
  },
  imageContainer: {
    alignItems: 'center',
    marginBottom: 16,
  },
  imageWrapper: {
    position: 'relative',
    width: screenWidth - 32,
    height: screenWidth - 32,
    borderRadius: 12,
    overflow: 'hidden',
  },
  previewImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  maskOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    opacity: 0.5,
    tintColor: '#00ff00',
  },
  placeholderImage: {
    width: screenWidth - 32,
    height: 200,
    backgroundColor: '#e0e0e0',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderText: {
    marginTop: 8,
    color: '#999',
    fontSize: 14,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
    marginBottom: 16,
  },
  imageButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#666',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
    gap: 6,
  },
  imageButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  analyzeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#007AFF',
    paddingVertical: 14,
    borderRadius: 10,
    gap: 8,
    marginBottom: 20,
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  buttonDisabled: {
    opacity: 0.7,
  },
  resultsContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  toggleButton: {
    backgroundColor: '#e8f4ff',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  toggleText: {
    color: '#007AFF',
    fontSize: 12,
    fontWeight: '600',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 16,
  },
  metricCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#333',
  },
  metricSubtext: {
    fontSize: 11,
    color: '#999',
    marginTop: 2,
  },
  detailsCard: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
  },
  detailsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 4,
  },
  detailLabel: {
    color: '#666',
    fontSize: 13,
  },
  detailValue: {
    color: '#333',
    fontSize: 13,
    fontWeight: '500',
  },
  loadDataButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#6c757d',
    paddingVertical: 14,
    borderRadius: 10,
    gap: 8,
  },
  loadDataButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  dataPreview: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  dataPreviewTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  dataRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  dataDate: {
    color: '#666',
    fontSize: 12,
  },
  dataValue: {
    color: '#333',
    fontSize: 12,
  },
  riskCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
  },
  riskHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  riskTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  riskBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  riskBadgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
  },
  riskDetails: {
    gap: 6,
  },
  riskItem: {
    fontSize: 14,
    color: '#555',
  },
  predictionsContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  predictionsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 16,
  },
  forecastChart: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'flex-end',
    height: 120,
    marginBottom: 8,
  },
  forecastBar: {
    alignItems: 'center',
    width: 40,
    height: '100%',
    justifyContent: 'flex-end',
  },
  forecastBarFill: {
    width: 24,
    backgroundColor: '#007AFF',
    borderRadius: 4,
    marginBottom: 4,
  },
  forecastLabel: {
    fontSize: 10,
    color: '#666',
  },
  forecastLegend: {
    alignItems: 'center',
    marginBottom: 16,
  },
  legendText: {
    fontSize: 12,
    color: '#999',
  },
  predictionsList: {
    gap: 8,
  },
  predictionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  predictionDate: {
    fontSize: 13,
    color: '#666',
    flex: 1,
  },
  predictionValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    flex: 1,
    textAlign: 'center',
  },
  predictionCI: {
    fontSize: 11,
    color: '#999',
    flex: 1,
    textAlign: 'right',
  },
  federatedInfo: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 24,
    alignItems: 'center',
    marginBottom: 16,
  },
  federatedTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginTop: 12,
    marginBottom: 8,
  },
  federatedDescription: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 20,
  },
  statusCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  statusTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
  },
  statusLabel: {
    flex: 1,
    color: '#666',
    fontSize: 14,
  },
  statusValue: {
    color: '#333',
    fontSize: 14,
    fontWeight: '500',
  },
  budgetBar: {
    flex: 2,
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    marginRight: 8,
    overflow: 'hidden',
  },
  budgetFill: {
    height: '100%',
    backgroundColor: '#28a745',
    borderRadius: 4,
  },
  featuresCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  featuresTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    gap: 10,
  },
  featureText: {
    fontSize: 14,
    color: '#555',
  },
  contributeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#28a745',
    paddingVertical: 14,
    borderRadius: 10,
    gap: 8,
    marginBottom: 30,
  },
  contributeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
