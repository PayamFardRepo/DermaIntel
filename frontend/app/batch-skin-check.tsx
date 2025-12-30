/**
 * Batch Skin Check Screen
 *
 * Features:
 * - Full body skin check with multiple images
 * - Body location mapping for each image
 * - Mole map visualization
 * - Lesion trend tracking over time
 * - PDF report generation
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Platform,
  RefreshControl,
  Image,
  Modal,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { API_BASE_URL } from '../config';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Body locations for full body check
const BODY_LOCATIONS = [
  { id: 'face', label: 'Face', icon: 'happy-outline' },
  { id: 'scalp', label: 'Scalp', icon: 'body-outline' },
  { id: 'neck', label: 'Neck', icon: 'body-outline' },
  { id: 'chest', label: 'Chest', icon: 'body-outline' },
  { id: 'abdomen', label: 'Abdomen', icon: 'body-outline' },
  { id: 'back_upper', label: 'Upper Back', icon: 'body-outline' },
  { id: 'back_lower', label: 'Lower Back', icon: 'body-outline' },
  { id: 'left_arm', label: 'Left Arm', icon: 'hand-left-outline' },
  { id: 'right_arm', label: 'Right Arm', icon: 'hand-right-outline' },
  { id: 'left_hand', label: 'Left Hand', icon: 'hand-left-outline' },
  { id: 'right_hand', label: 'Right Hand', icon: 'hand-right-outline' },
  { id: 'left_leg', label: 'Left Leg', icon: 'footsteps-outline' },
  { id: 'right_leg', label: 'Right Leg', icon: 'footsteps-outline' },
  { id: 'left_foot', label: 'Left Foot', icon: 'footsteps-outline' },
  { id: 'right_foot', label: 'Right Foot', icon: 'footsteps-outline' },
];

interface BatchCheck {
  id: number;
  status: string;
  total_images: number;
  processed_images: number;
  total_lesions_found: number;
  high_risk_count: number;
  created_at: string;
  completed_at: string | null;
  notes: string | null;
}

interface UploadedImage {
  body_location: string;
  uri: string;
  uploaded: boolean;
  processing: boolean;
}

interface LesionResult {
  id: number;
  body_location: string;
  classification: string;
  confidence: number;
  risk_level: string;
  image_path: string;
}

interface MoleMapData {
  total_lesions: number;
  by_location: { [key: string]: number };
  high_risk_locations: string[];
  recommendations: string[];
}

interface TrendData {
  date: string;
  total_lesions: number;
  high_risk_count: number;
}

export default function BatchSkinCheckScreen() {
  const { t } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // State
  const [activeTab, setActiveTab] = useState<'history' | 'new' | 'results'>('history');
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // History state
  const [checks, setChecks] = useState<BatchCheck[]>([]);

  // New check state
  const [currentCheckId, setCurrentCheckId] = useState<number | null>(null);
  const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);
  const [selectedLocation, setSelectedLocation] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  // Results state
  const [selectedCheck, setSelectedCheck] = useState<BatchCheck | null>(null);
  const [lesionResults, setLesionResults] = useState<LesionResult[]>([]);
  const [moleMapData, setMoleMapData] = useState<MoleMapData | null>(null);
  const [trendData, setTrendData] = useState<TrendData[]>([]);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  // Modal state
  const [locationModalVisible, setLocationModalVisible] = useState(false);

  const getAuthHeaders = async () => {
    const token = await AsyncStorage.getItem('accessToken');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  };

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
      return;
    }
    loadChecks();
    loadTrends();
  }, [isAuthenticated]);

  const loadChecks = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/batch/checks`, { headers });

      if (response.ok) {
        const data = await response.json();
        setChecks(data.checks || []);
      }
    } catch (error) {
      console.error('Error loading checks:', error);
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const loadTrends = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/batch/lesion-trends`, { headers });

      if (response.ok) {
        const data = await response.json();
        setTrendData(data.trends || []);
      }
    } catch (error) {
      console.error('Error loading trends:', error);
    }
  };

  const startNewCheck = async () => {
    try {
      setIsLoading(true);
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/batch/start-check`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ notes: `Full body check - ${new Date().toLocaleDateString()}` }),
      });

      if (response.ok) {
        const data = await response.json();
        setCurrentCheckId(data.check_id);
        setUploadedImages([]);
        setActiveTab('new');
      } else {
        Alert.alert('Error', 'Failed to start new check');
      }
    } catch (error) {
      console.error('Error starting check:', error);
      Alert.alert('Error', 'Failed to start new check');
    } finally {
      setIsLoading(false);
    }
  };

  const selectImage = async (location: string) => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Required', 'Please grant camera roll permissions');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        const newImage: UploadedImage = {
          body_location: location,
          uri: result.assets[0].uri,
          uploaded: false,
          processing: false,
        };

        setUploadedImages(prev => [...prev.filter(img => img.body_location !== location), newImage]);
        setLocationModalVisible(false);

        // Auto-upload
        await uploadImage(newImage);
      }
    } catch (error) {
      console.error('Error selecting image:', error);
      Alert.alert('Error', 'Failed to select image');
    }
  };

  const takePhoto = async (location: string) => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Required', 'Please grant camera permissions');
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        const newImage: UploadedImage = {
          body_location: location,
          uri: result.assets[0].uri,
          uploaded: false,
          processing: false,
        };

        setUploadedImages(prev => [...prev.filter(img => img.body_location !== location), newImage]);
        setLocationModalVisible(false);

        // Auto-upload
        await uploadImage(newImage);
      }
    } catch (error) {
      console.error('Error taking photo:', error);
      Alert.alert('Error', 'Failed to take photo');
    }
  };

  const uploadImage = async (image: UploadedImage) => {
    if (!currentCheckId) return;

    try {
      setIsUploading(true);
      setUploadedImages(prev =>
        prev.map(img =>
          img.body_location === image.body_location
            ? { ...img, processing: true }
            : img
        )
      );

      const token = await AsyncStorage.getItem('accessToken');
      const formData = new FormData();

      const filename = image.uri.split('/').pop() || 'image.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : 'image/jpeg';

      formData.append('file', {
        uri: image.uri,
        name: filename,
        type,
      } as any);
      formData.append('body_location', image.body_location);

      const response = await fetch(`${API_BASE_URL}/batch/upload/${currentCheckId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        setUploadedImages(prev =>
          prev.map(img =>
            img.body_location === image.body_location
              ? { ...img, uploaded: true, processing: false }
              : img
          )
        );
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      console.error('Error uploading image:', error);
      setUploadedImages(prev =>
        prev.map(img =>
          img.body_location === image.body_location
            ? { ...img, processing: false }
            : img
        )
      );
      Alert.alert('Error', 'Failed to upload image');
    } finally {
      setIsUploading(false);
    }
  };

  const processCheck = async () => {
    if (!currentCheckId) return;

    const uploadedCount = uploadedImages.filter(img => img.uploaded).length;
    if (uploadedCount === 0) {
      Alert.alert('No Images', 'Please upload at least one image before processing');
      return;
    }

    try {
      setIsProcessing(true);
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/batch/process/${currentCheckId}`, {
        method: 'POST',
        headers,
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(
          'Processing Complete',
          `Analyzed ${data.processed_images} images. Found ${data.total_lesions_found} lesions.`,
          [
            {
              text: 'View Results',
              onPress: () => viewCheckResults(currentCheckId),
            },
          ]
        );
        loadChecks();
      } else {
        throw new Error('Processing failed');
      }
    } catch (error) {
      console.error('Error processing check:', error);
      Alert.alert('Error', 'Failed to process images');
    } finally {
      setIsProcessing(false);
    }
  };

  const viewCheckResults = async (checkId: number) => {
    try {
      setIsLoading(true);
      const headers = await getAuthHeaders();

      // Get check details
      const checkResponse = await fetch(`${API_BASE_URL}/batch/check/${checkId}`, { headers });

      if (checkResponse.ok) {
        const checkData = await checkResponse.json();
        setSelectedCheck(checkData.check);
        setLesionResults(checkData.results || []);
      }

      // Get mole map
      const moleMapResponse = await fetch(`${API_BASE_URL}/batch/mole-map/${checkId}`, { headers });

      if (moleMapResponse.ok) {
        const moleMapData = await moleMapResponse.json();
        setMoleMapData(moleMapData);
      }

      setActiveTab('results');
    } catch (error) {
      console.error('Error loading results:', error);
      Alert.alert('Error', 'Failed to load check results');
    } finally {
      setIsLoading(false);
    }
  };

  const generateReport = async () => {
    if (!selectedCheck) return;

    try {
      setIsGeneratingReport(true);
      const headers = await getAuthHeaders();
      const response = await fetch(
        `${API_BASE_URL}/batch/generate-report/${selectedCheck.id}`,
        { method: 'POST', headers }
      );

      if (response.ok) {
        const data = await response.json();

        if (data.pdf_url) {
          // Download and share the PDF
          const fileUri = FileSystem.documentDirectory + `skin_check_report_${selectedCheck.id}.pdf`;
          const token = await AsyncStorage.getItem('accessToken');

          const downloadResult = await FileSystem.downloadAsync(
            `${API_BASE_URL}${data.pdf_url}`,
            fileUri,
            { headers: { 'Authorization': `Bearer ${token}` } }
          );

          if (await Sharing.isAvailableAsync()) {
            await Sharing.shareAsync(downloadResult.uri);
          } else {
            Alert.alert('Success', 'Report generated successfully');
          }
        }
      } else {
        throw new Error('Report generation failed');
      }
    } catch (error) {
      console.error('Error generating report:', error);
      Alert.alert('Error', 'Failed to generate report');
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadChecks();
    loadTrends();
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return '#10b981';
      case 'processing': return '#f59e0b';
      case 'pending': return '#6b7280';
      default: return '#6b7280';
    }
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel?.toLowerCase()) {
      case 'high': return '#ef4444';
      case 'medium': return '#f59e0b';
      case 'low': return '#10b981';
      default: return '#6b7280';
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  // Render History Tab
  const renderHistoryTab = () => (
    <ScrollView
      style={styles.tabContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Trends Summary */}
      {trendData.length > 0 && (
        <View style={styles.trendsCard}>
          <Text style={styles.cardTitle}>Lesion Trends</Text>
          <View style={styles.trendsSummary}>
            <View style={styles.trendItem}>
              <Text style={styles.trendValue}>{trendData[trendData.length - 1]?.total_lesions || 0}</Text>
              <Text style={styles.trendLabel}>Total Lesions</Text>
            </View>
            <View style={styles.trendItem}>
              <Text style={[styles.trendValue, { color: '#ef4444' }]}>
                {trendData[trendData.length - 1]?.high_risk_count || 0}
              </Text>
              <Text style={styles.trendLabel}>High Risk</Text>
            </View>
            <View style={styles.trendItem}>
              <Text style={styles.trendValue}>{checks.length}</Text>
              <Text style={styles.trendLabel}>Total Checks</Text>
            </View>
          </View>
        </View>
      )}

      {/* Start New Check Button */}
      <TouchableOpacity style={styles.newCheckButton} onPress={startNewCheck}>
        <LinearGradient
          colors={['#2563eb', '#1d4ed8']}
          style={styles.newCheckGradient}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
        >
          <Ionicons name="add-circle-outline" size={24} color="#fff" />
          <Text style={styles.newCheckButtonText}>Start Full Body Check</Text>
        </LinearGradient>
      </TouchableOpacity>

      {/* Check History */}
      <Text style={styles.sectionTitle}>Check History</Text>

      {checks.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="body-outline" size={64} color="#9ca3af" />
          <Text style={styles.emptyTitle}>No Checks Yet</Text>
          <Text style={styles.emptyText}>
            Start your first full body skin check to track your moles and lesions over time.
          </Text>
        </View>
      ) : (
        checks.map(check => (
          <TouchableOpacity
            key={check.id}
            style={styles.checkCard}
            onPress={() => viewCheckResults(check.id)}
          >
            <View style={styles.checkHeader}>
              <View style={[styles.statusBadge, { backgroundColor: `${getStatusColor(check.status)}20` }]}>
                <View style={[styles.statusDot, { backgroundColor: getStatusColor(check.status) }]} />
                <Text style={[styles.statusText, { color: getStatusColor(check.status) }]}>
                  {check.status.charAt(0).toUpperCase() + check.status.slice(1)}
                </Text>
              </View>
              <Text style={styles.checkDate}>{formatDate(check.created_at)}</Text>
            </View>

            <View style={styles.checkStats}>
              <View style={styles.statItem}>
                <Ionicons name="images-outline" size={20} color="#6b7280" />
                <Text style={styles.statValue}>{check.total_images}</Text>
                <Text style={styles.statLabel}>Images</Text>
              </View>
              <View style={styles.statItem}>
                <Ionicons name="scan-outline" size={20} color="#6b7280" />
                <Text style={styles.statValue}>{check.total_lesions_found}</Text>
                <Text style={styles.statLabel}>Lesions</Text>
              </View>
              <View style={styles.statItem}>
                <Ionicons name="warning-outline" size={20} color="#ef4444" />
                <Text style={[styles.statValue, { color: '#ef4444' }]}>{check.high_risk_count}</Text>
                <Text style={styles.statLabel}>High Risk</Text>
              </View>
            </View>

            <View style={styles.checkFooter}>
              <Text style={styles.viewResultsText}>View Results</Text>
              <Ionicons name="chevron-forward" size={20} color="#2563eb" />
            </View>
          </TouchableOpacity>
        ))
      )}

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Render New Check Tab
  const renderNewCheckTab = () => (
    <ScrollView style={styles.tabContent}>
      <View style={styles.instructionCard}>
        <Ionicons name="information-circle-outline" size={24} color="#2563eb" />
        <Text style={styles.instructionText}>
          Take photos of each body area to track moles and lesions. Tap on a body location to add an image.
        </Text>
      </View>

      {/* Progress */}
      <View style={styles.progressCard}>
        <Text style={styles.progressTitle}>
          {uploadedImages.filter(img => img.uploaded).length} of {BODY_LOCATIONS.length} areas captured
        </Text>
        <View style={styles.progressBar}>
          <View
            style={[
              styles.progressFill,
              { width: `${(uploadedImages.filter(img => img.uploaded).length / BODY_LOCATIONS.length) * 100}%` }
            ]}
          />
        </View>
      </View>

      {/* Body Locations Grid */}
      <View style={styles.locationsGrid}>
        {BODY_LOCATIONS.map(location => {
          const image = uploadedImages.find(img => img.body_location === location.id);
          const isUploaded = image?.uploaded;
          const isProcessingImage = image?.processing;

          return (
            <TouchableOpacity
              key={location.id}
              style={[
                styles.locationCard,
                isUploaded && styles.locationCardUploaded,
              ]}
              onPress={() => {
                setSelectedLocation(location.id);
                setLocationModalVisible(true);
              }}
              disabled={isProcessingImage}
            >
              {isProcessingImage ? (
                <ActivityIndicator size="small" color="#2563eb" />
              ) : isUploaded ? (
                <View style={styles.locationUploaded}>
                  <Ionicons name="checkmark-circle" size={24} color="#10b981" />
                </View>
              ) : (
                <Ionicons name={location.icon as any} size={24} color="#6b7280" />
              )}
              <Text style={[styles.locationLabel, isUploaded && styles.locationLabelUploaded]}>
                {location.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {/* Process Button */}
      <TouchableOpacity
        style={[
          styles.processButton,
          uploadedImages.filter(img => img.uploaded).length === 0 && styles.processButtonDisabled,
        ]}
        onPress={processCheck}
        disabled={isProcessing || uploadedImages.filter(img => img.uploaded).length === 0}
      >
        {isProcessing ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="analytics-outline" size={24} color="#fff" />
            <Text style={styles.processButtonText}>Process All Images</Text>
          </>
        )}
      </TouchableOpacity>

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Render Results Tab
  const renderResultsTab = () => (
    <ScrollView style={styles.tabContent}>
      {selectedCheck && (
        <>
          {/* Summary Card */}
          <View style={styles.summaryCard}>
            <Text style={styles.summaryTitle}>Check Summary</Text>
            <Text style={styles.summaryDate}>{formatDate(selectedCheck.created_at)}</Text>

            <View style={styles.summaryStats}>
              <View style={styles.summaryStatItem}>
                <Text style={styles.summaryStatValue}>{selectedCheck.total_images}</Text>
                <Text style={styles.summaryStatLabel}>Images</Text>
              </View>
              <View style={styles.summaryStatItem}>
                <Text style={styles.summaryStatValue}>{selectedCheck.total_lesions_found}</Text>
                <Text style={styles.summaryStatLabel}>Lesions</Text>
              </View>
              <View style={styles.summaryStatItem}>
                <Text style={[styles.summaryStatValue, { color: '#ef4444' }]}>
                  {selectedCheck.high_risk_count}
                </Text>
                <Text style={styles.summaryStatLabel}>High Risk</Text>
              </View>
            </View>
          </View>

          {/* Mole Map */}
          {moleMapData && (
            <View style={styles.moleMapCard}>
              <Text style={styles.cardTitle}>Mole Map</Text>

              <View style={styles.moleMapBody}>
                {/* Simple body visualization */}
                <View style={styles.bodyOutline}>
                  {Object.entries(moleMapData.by_location).map(([location, count]) => {
                    const isHighRisk = moleMapData.high_risk_locations.includes(location);
                    return (
                      <View
                        key={location}
                        style={[
                          styles.moleMapLocation,
                          isHighRisk && styles.moleMapLocationHighRisk,
                        ]}
                      >
                        <Text style={styles.moleMapLocationLabel}>
                          {location.replace('_', ' ')}
                        </Text>
                        <Text style={[
                          styles.moleMapLocationCount,
                          isHighRisk && styles.moleMapLocationCountHighRisk,
                        ]}>
                          {count}
                        </Text>
                      </View>
                    );
                  })}
                </View>
              </View>

              {/* Recommendations */}
              {moleMapData.recommendations.length > 0 && (
                <View style={styles.recommendationsSection}>
                  <Text style={styles.recommendationsTitle}>Recommendations</Text>
                  {moleMapData.recommendations.map((rec, index) => (
                    <View key={index} style={styles.recommendationItem}>
                      <Ionicons name="alert-circle-outline" size={16} color="#f59e0b" />
                      <Text style={styles.recommendationText}>{rec}</Text>
                    </View>
                  ))}
                </View>
              )}
            </View>
          )}

          {/* Lesion Results */}
          <Text style={styles.sectionTitle}>Detected Lesions</Text>

          {lesionResults.length === 0 ? (
            <View style={styles.noLesionsCard}>
              <Ionicons name="checkmark-circle-outline" size={48} color="#10b981" />
              <Text style={styles.noLesionsText}>No concerning lesions detected</Text>
            </View>
          ) : (
            lesionResults.map(lesion => (
              <View key={lesion.id} style={styles.lesionCard}>
                <View style={styles.lesionHeader}>
                  <View style={[
                    styles.riskBadge,
                    { backgroundColor: `${getRiskColor(lesion.risk_level)}20` }
                  ]}>
                    <Text style={[styles.riskText, { color: getRiskColor(lesion.risk_level) }]}>
                      {lesion.risk_level} Risk
                    </Text>
                  </View>
                  <Text style={styles.lesionLocation}>
                    {lesion.body_location.replace('_', ' ')}
                  </Text>
                </View>

                <View style={styles.lesionDetails}>
                  <Text style={styles.lesionClassification}>{lesion.classification}</Text>
                  <Text style={styles.lesionConfidence}>
                    {(lesion.confidence * 100).toFixed(1)}% confidence
                  </Text>
                </View>
              </View>
            ))
          )}

          {/* Generate Report Button */}
          <TouchableOpacity
            style={styles.reportButton}
            onPress={generateReport}
            disabled={isGeneratingReport}
          >
            {isGeneratingReport ? (
              <ActivityIndicator color="#2563eb" />
            ) : (
              <>
                <Ionicons name="document-text-outline" size={20} color="#2563eb" />
                <Text style={styles.reportButtonText}>Generate PDF Report</Text>
              </>
            )}
          </TouchableOpacity>
        </>
      )}

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Location Selection Modal
  const renderLocationModal = () => (
    <Modal
      visible={locationModalVisible}
      transparent
      animationType="slide"
      onRequestClose={() => setLocationModalVisible(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>
              {BODY_LOCATIONS.find(l => l.id === selectedLocation)?.label}
            </Text>
            <TouchableOpacity onPress={() => setLocationModalVisible(false)}>
              <Ionicons name="close" size={24} color="#6b7280" />
            </TouchableOpacity>
          </View>

          <Text style={styles.modalSubtitle}>Choose how to capture image</Text>

          <TouchableOpacity
            style={styles.modalOption}
            onPress={() => selectedLocation && takePhoto(selectedLocation)}
          >
            <Ionicons name="camera-outline" size={32} color="#2563eb" />
            <Text style={styles.modalOptionText}>Take Photo</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.modalOption}
            onPress={() => selectedLocation && selectImage(selectedLocation)}
          >
            <Ionicons name="images-outline" size={32} color="#2563eb" />
            <Text style={styles.modalOptionText}>Choose from Library</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  if (isLoading && checks.length === 0) {
    return (
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2563eb" />
          <Text style={styles.loadingText}>Loading...</Text>
        </View>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#2563eb" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Full Body Skin Check</Text>
        <View style={{ width: 40 }} />
      </View>

      {/* Tabs */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'history' && styles.activeTab]}
          onPress={() => setActiveTab('history')}
        >
          <Text style={[styles.tabText, activeTab === 'history' && styles.activeTabText]}>
            History
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'new' && styles.activeTab]}
          onPress={() => activeTab === 'new' ? null : startNewCheck()}
        >
          <Text style={[styles.tabText, activeTab === 'new' && styles.activeTabText]}>
            New Check
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'results' && styles.activeTab]}
          onPress={() => setActiveTab('results')}
          disabled={!selectedCheck}
        >
          <Text style={[
            styles.tabText,
            activeTab === 'results' && styles.activeTabText,
            !selectedCheck && styles.tabTextDisabled,
          ]}>
            Results
          </Text>
        </TouchableOpacity>
      </View>

      {/* Tab Content */}
      {activeTab === 'history' && renderHistoryTab()}
      {activeTab === 'new' && renderNewCheckTab()}
      {activeTab === 'results' && renderResultsTab()}

      {/* Location Modal */}
      {renderLocationModal()}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 20,
    paddingBottom: 15,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  tab: {
    flex: 1,
    paddingVertical: 14,
    alignItems: 'center',
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: '#2563eb',
  },
  tabText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
  },
  activeTabText: {
    color: '#2563eb',
  },
  tabTextDisabled: {
    color: '#d1d5db',
  },
  tabContent: {
    flex: 1,
    padding: 16,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    color: '#6b7280',
    fontSize: 16,
  },
  trendsCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 16,
  },
  trendsSummary: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  trendItem: {
    alignItems: 'center',
  },
  trendValue: {
    fontSize: 28,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  trendLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  newCheckButton: {
    marginBottom: 20,
    borderRadius: 12,
    overflow: 'hidden',
  },
  newCheckGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    gap: 10,
  },
  newCheckButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
    marginTop: 8,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 40,
    backgroundColor: '#fff',
    borderRadius: 16,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  checkCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
  },
  checkHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 6,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
  },
  checkDate: {
    fontSize: 12,
    color: '#6b7280',
  },
  checkStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  statItem: {
    alignItems: 'center',
    gap: 4,
  },
  statValue: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  statLabel: {
    fontSize: 11,
    color: '#6b7280',
  },
  checkFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
    gap: 4,
  },
  viewResultsText: {
    color: '#2563eb',
    fontSize: 14,
    fontWeight: '600',
  },
  instructionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    gap: 12,
  },
  instructionText: {
    flex: 1,
    fontSize: 13,
    color: '#1e40af',
    lineHeight: 20,
  },
  progressCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  progressTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
    marginBottom: 10,
  },
  progressBar: {
    height: 8,
    backgroundColor: '#e5e7eb',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#2563eb',
    borderRadius: 4,
  },
  locationsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  locationCard: {
    width: (SCREEN_WIDTH - 52) / 3,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
  },
  locationCardUploaded: {
    backgroundColor: '#ecfdf5',
    borderWidth: 1,
    borderColor: '#10b981',
  },
  locationUploaded: {
    marginBottom: 4,
  },
  locationLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center',
  },
  locationLabelUploaded: {
    color: '#059669',
    fontWeight: '600',
  },
  processButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    marginTop: 24,
    gap: 10,
  },
  processButtonDisabled: {
    backgroundColor: '#9ca3af',
  },
  processButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  summaryCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  summaryTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  summaryDate: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 4,
  },
  summaryStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 20,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  summaryStatItem: {
    alignItems: 'center',
  },
  summaryStatValue: {
    fontSize: 32,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  summaryStatLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  moleMapCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  moleMapBody: {
    padding: 10,
  },
  bodyOutline: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  moleMapLocation: {
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    padding: 10,
    minWidth: 80,
    alignItems: 'center',
  },
  moleMapLocationHighRisk: {
    backgroundColor: '#fef2f2',
    borderWidth: 1,
    borderColor: '#ef4444',
  },
  moleMapLocationLabel: {
    fontSize: 11,
    color: '#6b7280',
    textTransform: 'capitalize',
  },
  moleMapLocationCount: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
    marginTop: 4,
  },
  moleMapLocationCountHighRisk: {
    color: '#ef4444',
  },
  recommendationsSection: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  recommendationsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
    marginBottom: 10,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    marginBottom: 8,
  },
  recommendationText: {
    flex: 1,
    fontSize: 13,
    color: '#6b7280',
    lineHeight: 18,
  },
  noLesionsCard: {
    backgroundColor: '#ecfdf5',
    borderRadius: 12,
    padding: 32,
    alignItems: 'center',
    marginBottom: 16,
  },
  noLesionsText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#059669',
    marginTop: 12,
  },
  lesionCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
  },
  lesionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  riskBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  riskText: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  lesionLocation: {
    fontSize: 12,
    color: '#6b7280',
    textTransform: 'capitalize',
  },
  lesionDetails: {
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  lesionClassification: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  lesionConfidence: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 4,
  },
  reportButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    marginTop: 8,
    gap: 10,
  },
  reportButtonText: {
    color: '#2563eb',
    fontSize: 16,
    fontWeight: '600',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    paddingBottom: Platform.OS === 'ios' ? 40 : 20,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  modalSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 20,
  },
  modalOption: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 20,
    marginBottom: 12,
    gap: 16,
  },
  modalOptionText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  bottomSpacer: {
    height: 30,
  },
});
