/**
 * Full-Body Skin Check Screen
 *
 * Features:
 * - Upload 20-30 photos of entire body surface
 * - AI automatically tags body location for each lesion
 * - Generate comprehensive full-body report
 * - Flag highest-risk lesions first
 * - Create "mole map" visualization
 * - Track total lesion count over time
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Platform,
  Dimensions,
  Image,
  Modal,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import * as ImagePicker from 'expo-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';
import { Svg, Circle, Ellipse, Rect, Text as SvgText, G, Path } from 'react-native-svg';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');
const BODY_MAP_WIDTH = Math.min(screenWidth - 40, 350);
const BODY_MAP_HEIGHT = BODY_MAP_WIDTH * 1.8;

// Body region coordinates for mole map visualization
const BODY_REGIONS: Record<string, { x: number; y: number; label: string }> = {
  head: { x: 50, y: 5, label: 'Head' },
  face: { x: 50, y: 8, label: 'Face' },
  neck: { x: 50, y: 15, label: 'Neck' },
  chest: { x: 50, y: 28, label: 'Chest' },
  abdomen: { x: 50, y: 40, label: 'Abdomen' },
  back_upper: { x: 50, y: 28, label: 'Upper Back' },
  back_lower: { x: 50, y: 42, label: 'Lower Back' },
  shoulder_left: { x: 25, y: 22, label: 'L Shoulder' },
  shoulder_right: { x: 75, y: 22, label: 'R Shoulder' },
  arm_left_upper: { x: 18, y: 32, label: 'L Upper Arm' },
  arm_right_upper: { x: 82, y: 32, label: 'R Upper Arm' },
  arm_left_lower: { x: 14, y: 44, label: 'L Forearm' },
  arm_right_lower: { x: 86, y: 44, label: 'R Forearm' },
  hand_left: { x: 10, y: 52, label: 'L Hand' },
  hand_right: { x: 90, y: 52, label: 'R Hand' },
  hip_left: { x: 40, y: 48, label: 'L Hip' },
  hip_right: { x: 60, y: 48, label: 'R Hip' },
  thigh_left: { x: 40, y: 60, label: 'L Thigh' },
  thigh_right: { x: 60, y: 60, label: 'R Thigh' },
  knee_left: { x: 40, y: 72, label: 'L Knee' },
  knee_right: { x: 60, y: 72, label: 'R Knee' },
  leg_left_lower: { x: 40, y: 82, label: 'L Calf' },
  leg_right_lower: { x: 60, y: 82, label: 'R Calf' },
  foot_left: { x: 40, y: 94, label: 'L Foot' },
  foot_right: { x: 60, y: 94, label: 'R Foot' },
};

interface UploadedImage {
  uri: string;
  filename: string;
  imageId?: string;
  bodyLocation?: string;
}

interface LesionData {
  lesion_id: string;
  x: number;
  y: number;
  risk_level: string;
  risk_score: number;
  predicted_class: string;
  confidence: number;
}

interface CheckResult {
  check_id: string;
  status: string;
  summary: {
    total_images: number;
    total_lesions: number;
    overall_risk_score: number;
    risk_breakdown: {
      critical: number;
      high: number;
      medium: number;
      low: number;
    };
  };
  body_coverage: Record<string, boolean>;
  mole_map: Record<string, LesionData[]>;
  lesion_count_by_region: Record<string, number>;
  recommendations: string[];
  highest_risk_lesions: any[];
}

export default function FullBodyCheckScreen() {
  const { t } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // State
  const [step, setStep] = useState<'upload' | 'processing' | 'results'>('upload');
  const [checkId, setCheckId] = useState<string | null>(null);
  const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<CheckResult | null>(null);
  const [selectedLesion, setSelectedLesion] = useState<LesionData | null>(null);
  const [showLesionModal, setShowLesionModal] = useState(false);
  const [trends, setTrends] = useState<any>(null);
  const [bodyView, setBodyView] = useState<'front' | 'back'>('front');

  // Auth headers
  const getAuthHeaders = async () => {
    const token = await AsyncStorage.getItem('accessToken');
    return {
      'Authorization': `Bearer ${token}`,
    };
  };

  // Initialize
  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
      return;
    }
    loadTrends();
  }, [isAuthenticated]);

  const loadTrends = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/batch/lesion-trends`, { headers });
      if (response.ok) {
        const data = await response.json();
        setTrends(data);
      }
    } catch (error) {
      console.error('Error loading trends:', error);
    }
  };

  // Start check session
  const startCheck = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/batch/start-check`, {
        method: 'POST',
        headers,
      });

      if (response.ok) {
        const data = await response.json();
        setCheckId(data.check_id);
        return data.check_id;
      }
      throw new Error('Failed to start check');
    } catch (error) {
      console.error('Error starting check:', error);
      Alert.alert('Error', 'Failed to start skin check session');
      return null;
    }
  };

  // Pick images
  const pickImages = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission Required', 'Please grant access to your photo library');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsMultipleSelection: true,
      quality: 0.8,
      selectionLimit: 30,
    });

    if (!result.canceled && result.assets.length > 0) {
      const newImages = result.assets.map((asset, index) => ({
        uri: asset.uri,
        filename: asset.fileName || `image_${Date.now()}_${index}.jpg`,
      }));
      setUploadedImages(prev => [...prev, ...newImages]);
    }
  };

  // Take photo
  const takePhoto = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission Required', 'Please grant camera access');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      quality: 0.8,
    });

    if (!result.canceled && result.assets.length > 0) {
      const asset = result.assets[0];
      setUploadedImages(prev => [...prev, {
        uri: asset.uri,
        filename: asset.fileName || `photo_${Date.now()}.jpg`,
      }]);
    }
  };

  // Remove image
  const removeImage = (index: number) => {
    setUploadedImages(prev => prev.filter((_, i) => i !== index));
  };

  // Upload all images and process
  const processCheck = async () => {
    if (uploadedImages.length < 5) {
      Alert.alert('More Images Needed', 'Please upload at least 5 images for a proper skin check');
      return;
    }

    setIsUploading(true);
    setStep('processing');

    try {
      // Start check session
      let currentCheckId = checkId;
      if (!currentCheckId) {
        currentCheckId = await startCheck();
        if (!currentCheckId) return;
      }

      // Upload each image
      const headers = await getAuthHeaders();
      for (let i = 0; i < uploadedImages.length; i++) {
        const image = uploadedImages[i];

        const formData = new FormData();
        formData.append('file', {
          uri: image.uri,
          type: 'image/jpeg',
          name: image.filename,
        } as any);

        if (image.bodyLocation) {
          formData.append('body_location', image.bodyLocation);
        }

        const uploadResponse = await fetch(`${API_BASE_URL}/batch/upload/${currentCheckId}`, {
          method: 'POST',
          headers,
          body: formData,
        });

        if (!uploadResponse.ok) {
          throw new Error(`Failed to upload image ${i + 1}`);
        }

        const uploadData = await uploadResponse.json();
        setUploadedImages(prev => prev.map((img, idx) =>
          idx === i ? { ...img, imageId: uploadData.image_id } : img
        ));
      }

      setIsUploading(false);
      setIsProcessing(true);

      // Process the batch
      const processResponse = await fetch(`${API_BASE_URL}/batch/process/${currentCheckId}`, {
        method: 'POST',
        headers,
      });

      if (!processResponse.ok) {
        throw new Error('Failed to process images');
      }

      const processData = await processResponse.json();
      setResult(processData);
      setStep('results');
      loadTrends(); // Refresh trends

    } catch (error: any) {
      console.error('Error processing check:', error);
      Alert.alert('Error', error.message || 'Failed to process skin check');
      setStep('upload');
    } finally {
      setIsUploading(false);
      setIsProcessing(false);
    }
  };

  // Get risk color
  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'critical': return '#dc2626';
      case 'high': return '#f97316';
      case 'medium': return '#eab308';
      case 'low': return '#22c55e';
      default: return '#6b7280';
    }
  };

  // Render mole map
  const renderMoleMap = () => {
    if (!result?.mole_map) return null;

    const allLesions: LesionData[] = [];
    Object.entries(result.mole_map).forEach(([region, lesions]) => {
      lesions.forEach(lesion => {
        allLesions.push(lesion);
      });
    });

    // Filter by body view
    const frontRegions = ['face', 'chest', 'abdomen', 'shoulder_left', 'shoulder_right',
      'arm_left_upper', 'arm_right_upper', 'arm_left_lower', 'arm_right_lower',
      'hand_left', 'hand_right', 'thigh_left', 'thigh_right', 'knee_left', 'knee_right',
      'leg_left_lower', 'leg_right_lower', 'foot_left', 'foot_right'];
    const backRegions = ['head', 'neck', 'back_upper', 'back_lower', 'buttock'];

    return (
      <View style={styles.moleMapContainer}>
        {/* View toggle */}
        <View style={styles.viewToggle}>
          <TouchableOpacity
            style={[styles.viewButton, bodyView === 'front' && styles.viewButtonActive]}
            onPress={() => setBodyView('front')}
          >
            <Text style={[styles.viewButtonText, bodyView === 'front' && styles.viewButtonTextActive]}>
              Front
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.viewButton, bodyView === 'back' && styles.viewButtonActive]}
            onPress={() => setBodyView('back')}
          >
            <Text style={[styles.viewButtonText, bodyView === 'back' && styles.viewButtonTextActive]}>
              Back
            </Text>
          </TouchableOpacity>
        </View>

        <Svg width={BODY_MAP_WIDTH} height={BODY_MAP_HEIGHT} style={styles.moleMapSvg}>
          {/* Body outline */}
          <Ellipse
            cx={BODY_MAP_WIDTH / 2}
            cy={BODY_MAP_HEIGHT * 0.07}
            rx={BODY_MAP_WIDTH * 0.12}
            ry={BODY_MAP_HEIGHT * 0.05}
            fill="#f0f0f0"
            stroke="#d0d0d0"
            strokeWidth={1}
          />
          {/* Torso */}
          <Rect
            x={BODY_MAP_WIDTH * 0.3}
            y={BODY_MAP_HEIGHT * 0.12}
            width={BODY_MAP_WIDTH * 0.4}
            height={BODY_MAP_HEIGHT * 0.35}
            rx={BODY_MAP_WIDTH * 0.05}
            fill="#f0f0f0"
            stroke="#d0d0d0"
            strokeWidth={1}
          />
          {/* Left arm */}
          <Rect
            x={BODY_MAP_WIDTH * 0.08}
            y={BODY_MAP_HEIGHT * 0.15}
            width={BODY_MAP_WIDTH * 0.12}
            height={BODY_MAP_HEIGHT * 0.35}
            rx={BODY_MAP_WIDTH * 0.03}
            fill="#f0f0f0"
            stroke="#d0d0d0"
            strokeWidth={1}
          />
          {/* Right arm */}
          <Rect
            x={BODY_MAP_WIDTH * 0.8}
            y={BODY_MAP_HEIGHT * 0.15}
            width={BODY_MAP_WIDTH * 0.12}
            height={BODY_MAP_HEIGHT * 0.35}
            rx={BODY_MAP_WIDTH * 0.03}
            fill="#f0f0f0"
            stroke="#d0d0d0"
            strokeWidth={1}
          />
          {/* Left leg */}
          <Rect
            x={BODY_MAP_WIDTH * 0.32}
            y={BODY_MAP_HEIGHT * 0.48}
            width={BODY_MAP_WIDTH * 0.14}
            height={BODY_MAP_HEIGHT * 0.45}
            rx={BODY_MAP_WIDTH * 0.03}
            fill="#f0f0f0"
            stroke="#d0d0d0"
            strokeWidth={1}
          />
          {/* Right leg */}
          <Rect
            x={BODY_MAP_WIDTH * 0.54}
            y={BODY_MAP_HEIGHT * 0.48}
            width={BODY_MAP_WIDTH * 0.14}
            height={BODY_MAP_HEIGHT * 0.45}
            rx={BODY_MAP_WIDTH * 0.03}
            fill="#f0f0f0"
            stroke="#d0d0d0"
            strokeWidth={1}
          />

          {/* Lesion markers */}
          {allLesions.map((lesion, index) => {
            const x = (lesion.x / 100) * BODY_MAP_WIDTH;
            const y = (lesion.y / 100) * BODY_MAP_HEIGHT;
            const color = getRiskColor(lesion.risk_level);
            const size = lesion.risk_level === 'critical' ? 12 :
                        lesion.risk_level === 'high' ? 10 : 8;

            return (
              <G key={lesion.lesion_id || index}>
                <Circle
                  cx={x}
                  cy={y}
                  r={size}
                  fill={color}
                  fillOpacity={0.8}
                  stroke="#fff"
                  strokeWidth={2}
                  onPress={() => {
                    setSelectedLesion(lesion);
                    setShowLesionModal(true);
                  }}
                />
              </G>
            );
          })}
        </Svg>

        {/* Legend */}
        <View style={styles.legend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#dc2626' }]} />
            <Text style={styles.legendText}>Critical</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#f97316' }]} />
            <Text style={styles.legendText}>High</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#eab308' }]} />
            <Text style={styles.legendText}>Medium</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#22c55e' }]} />
            <Text style={styles.legendText}>Low</Text>
          </View>
        </View>
      </View>
    );
  };

  // Render upload step
  const renderUploadStep = () => (
    <ScrollView style={styles.stepContent}>
      <View style={styles.instructionsCard}>
        <Text style={styles.instructionsTitle}>Full-Body Skin Check</Text>
        <Text style={styles.instructionsText}>
          Upload 20-30 photos covering your entire body surface for a comprehensive skin analysis.
        </Text>
        <View style={styles.instructionsList}>
          <Text style={styles.instructionItem}>- Include face, neck, chest, back, arms, legs</Text>
          <Text style={styles.instructionItem}>- Use good lighting (natural light is best)</Text>
          <Text style={styles.instructionItem}>- Keep camera 12-18 inches from skin</Text>
          <Text style={styles.instructionItem}>- Name files with body location for better detection</Text>
        </View>
      </View>

      {/* Trends card */}
      {trends?.has_history && (
        <View style={styles.trendsCard}>
          <Text style={styles.trendsTitle}>Previous Results</Text>
          <View style={styles.trendsRow}>
            <View style={styles.trendItem}>
              <Text style={styles.trendValue}>{trends.current.total_lesions}</Text>
              <Text style={styles.trendLabel}>Lesions</Text>
            </View>
            <View style={styles.trendItem}>
              <Text style={[styles.trendValue, { color: trends.change.trend === 'increasing' ? '#ef4444' : '#22c55e' }]}>
                {trends.change.absolute > 0 ? '+' : ''}{trends.change.absolute}
              </Text>
              <Text style={styles.trendLabel}>Change</Text>
            </View>
            <View style={styles.trendItem}>
              <Text style={styles.trendValue}>{trends.total_checks}</Text>
              <Text style={styles.trendLabel}>Checks</Text>
            </View>
          </View>
        </View>
      )}

      {/* Upload buttons */}
      <View style={styles.uploadButtons}>
        <TouchableOpacity style={styles.uploadButton} onPress={pickImages}>
          <Ionicons name="images" size={32} color="#2563eb" />
          <Text style={styles.uploadButtonText}>Select Photos</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.uploadButton} onPress={takePhoto}>
          <Ionicons name="camera" size={32} color="#2563eb" />
          <Text style={styles.uploadButtonText}>Take Photo</Text>
        </TouchableOpacity>
      </View>

      {/* Image count */}
      {uploadedImages.length > 0 && (
        <View style={styles.imageCountCard}>
          <Text style={styles.imageCountText}>
            {uploadedImages.length} {uploadedImages.length === 1 ? 'image' : 'images'} selected
          </Text>
          <Text style={styles.imageCountHint}>
            {uploadedImages.length < 20 ? `Add ${20 - uploadedImages.length} more for comprehensive coverage` : 'Great coverage!'}
          </Text>
        </View>
      )}

      {/* Image thumbnails */}
      {uploadedImages.length > 0 && (
        <ScrollView horizontal style={styles.thumbnailScroll} showsHorizontalScrollIndicator={false}>
          {uploadedImages.map((image, index) => (
            <View key={index} style={styles.thumbnailContainer}>
              <Image source={{ uri: image.uri }} style={styles.thumbnail} />
              <TouchableOpacity
                style={styles.removeButton}
                onPress={() => removeImage(index)}
              >
                <Ionicons name="close-circle" size={24} color="#ef4444" />
              </TouchableOpacity>
            </View>
          ))}
        </ScrollView>
      )}

      {/* Process button */}
      {uploadedImages.length >= 5 && (
        <TouchableOpacity style={styles.processButton} onPress={processCheck}>
          <Text style={styles.processButtonText}>Start Analysis</Text>
          <Ionicons name="arrow-forward" size={24} color="#fff" />
        </TouchableOpacity>
      )}
    </ScrollView>
  );

  // Render processing step
  const renderProcessingStep = () => (
    <View style={styles.processingContainer}>
      <ActivityIndicator size="large" color="#2563eb" />
      <Text style={styles.processingText}>
        {isUploading ? 'Uploading images...' : 'Analyzing your skin...'}
      </Text>
      <Text style={styles.processingSubtext}>
        {isUploading
          ? `${uploadedImages.filter(i => i.imageId).length} / ${uploadedImages.length} images uploaded`
          : 'AI is detecting and classifying lesions'}
      </Text>
    </View>
  );

  // Render results step
  const renderResultsStep = () => {
    if (!result) return null;

    return (
      <ScrollView style={styles.stepContent}>
        {/* Summary card */}
        <View style={styles.summaryCard}>
          <Text style={styles.summaryTitle}>Skin Check Complete</Text>
          <View style={styles.summaryStats}>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{result.summary.total_lesions}</Text>
              <Text style={styles.statLabel}>Lesions Found</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={[styles.statValue, { color: getRiskColor(
                result.summary.risk_breakdown.critical > 0 ? 'critical' :
                result.summary.risk_breakdown.high > 0 ? 'high' :
                result.summary.risk_breakdown.medium > 0 ? 'medium' : 'low'
              )}]}>
                {Math.round(result.summary.overall_risk_score)}
              </Text>
              <Text style={styles.statLabel}>Risk Score</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{result.summary.total_images}</Text>
              <Text style={styles.statLabel}>Images</Text>
            </View>
          </View>
        </View>

        {/* Risk breakdown */}
        <View style={styles.riskBreakdown}>
          <Text style={styles.sectionTitle}>Risk Breakdown</Text>
          <View style={styles.riskBars}>
            {result.summary.risk_breakdown.critical > 0 && (
              <View style={styles.riskBar}>
                <View style={[styles.riskBarFill, { backgroundColor: '#dc2626', width: `${Math.min(100, result.summary.risk_breakdown.critical * 20)}%` }]} />
                <Text style={styles.riskBarText}>Critical: {result.summary.risk_breakdown.critical}</Text>
              </View>
            )}
            {result.summary.risk_breakdown.high > 0 && (
              <View style={styles.riskBar}>
                <View style={[styles.riskBarFill, { backgroundColor: '#f97316', width: `${Math.min(100, result.summary.risk_breakdown.high * 20)}%` }]} />
                <Text style={styles.riskBarText}>High: {result.summary.risk_breakdown.high}</Text>
              </View>
            )}
            {result.summary.risk_breakdown.medium > 0 && (
              <View style={styles.riskBar}>
                <View style={[styles.riskBarFill, { backgroundColor: '#eab308', width: `${Math.min(100, result.summary.risk_breakdown.medium * 10)}%` }]} />
                <Text style={styles.riskBarText}>Medium: {result.summary.risk_breakdown.medium}</Text>
              </View>
            )}
            <View style={styles.riskBar}>
              <View style={[styles.riskBarFill, { backgroundColor: '#22c55e', width: `${Math.min(100, result.summary.risk_breakdown.low * 5)}%` }]} />
              <Text style={styles.riskBarText}>Low: {result.summary.risk_breakdown.low}</Text>
            </View>
          </View>
        </View>

        {/* Mole map */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Mole Map</Text>
          {renderMoleMap()}
        </View>

        {/* Highest risk lesions */}
        {result.highest_risk_lesions.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Priority Lesions</Text>
            {result.highest_risk_lesions.slice(0, 5).map((lesion, index) => (
              <View key={lesion.lesion_id || index} style={styles.lesionCard}>
                <View style={styles.lesionHeader}>
                  <View style={[styles.riskBadge, { backgroundColor: getRiskColor(lesion.risk_level) }]}>
                    <Text style={styles.riskBadgeText}>{lesion.risk_level.toUpperCase()}</Text>
                  </View>
                  <Text style={styles.lesionClass}>{lesion.predicted_class}</Text>
                </View>
                <Text style={styles.lesionLocation}>
                  Location: {lesion.body_location?.replace(/_/g, ' ') || 'Unknown'}
                </Text>
                <Text style={styles.lesionScore}>Risk Score: {Math.round(lesion.risk_score)}/100</Text>
              </View>
            ))}
          </View>
        )}

        {/* Recommendations */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Recommendations</Text>
          {result.recommendations.map((rec, index) => (
            <View key={index} style={styles.recommendationItem}>
              <Ionicons
                name={rec.includes('URGENT') ? 'alert-circle' : 'checkmark-circle'}
                size={20}
                color={rec.includes('URGENT') ? '#dc2626' : '#22c55e'}
              />
              <Text style={styles.recommendationText}>{rec}</Text>
            </View>
          ))}
        </View>

        {/* Action buttons */}
        <View style={styles.actionButtons}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => {
              setStep('upload');
              setUploadedImages([]);
              setResult(null);
              setCheckId(null);
            }}
          >
            <Ionicons name="refresh" size={24} color="#fff" />
            <Text style={styles.actionButtonText}>New Check</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.bottomSpacer} />
      </ScrollView>
    );
  };

  // Lesion detail modal
  const renderLesionModal = () => (
    <Modal
      visible={showLesionModal}
      transparent
      animationType="fade"
      onRequestClose={() => setShowLesionModal(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Lesion Details</Text>
            <TouchableOpacity onPress={() => setShowLesionModal(false)}>
              <Ionicons name="close" size={24} color="#333" />
            </TouchableOpacity>
          </View>

          {selectedLesion && (
            <View style={styles.lesionDetails}>
              <View style={[styles.riskBadgeLarge, { backgroundColor: getRiskColor(selectedLesion.risk_level) }]}>
                <Text style={styles.riskBadgeLargeText}>{selectedLesion.risk_level.toUpperCase()}</Text>
              </View>

              <Text style={styles.detailLabel}>Classification</Text>
              <Text style={styles.detailValue}>{selectedLesion.predicted_class}</Text>

              <Text style={styles.detailLabel}>Risk Score</Text>
              <Text style={styles.detailValue}>{Math.round(selectedLesion.risk_score)}/100</Text>

              <Text style={styles.detailLabel}>Confidence</Text>
              <Text style={styles.detailValue}>{Math.round(selectedLesion.confidence * 100)}%</Text>
            </View>
          )}

          <TouchableOpacity
            style={styles.modalButton}
            onPress={() => setShowLesionModal(false)}
          >
            <Text style={styles.modalButtonText}>Close</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  return (
    <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#2563eb" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Full-Body Skin Check</Text>
        <View style={styles.headerSpacer} />
      </View>

      {/* Content */}
      {step === 'upload' && renderUploadStep()}
      {step === 'processing' && renderProcessingStep()}
      {step === 'results' && renderResultsStep()}

      {renderLesionModal()}
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
  headerSpacer: {
    width: 40,
  },
  stepContent: {
    flex: 1,
    padding: 20,
  },
  instructionsCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  instructionsTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 8,
  },
  instructionsText: {
    fontSize: 16,
    color: '#4b5563',
    lineHeight: 24,
    marginBottom: 12,
  },
  instructionsList: {
    marginTop: 8,
  },
  instructionItem: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 4,
  },
  trendsCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  trendsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  trendsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  trendItem: {
    alignItems: 'center',
  },
  trendValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#2563eb',
  },
  trendLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  uploadButtons: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 16,
  },
  uploadButton: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#2563eb',
    borderStyle: 'dashed',
  },
  uploadButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2563eb',
    marginTop: 8,
  },
  imageCountCard: {
    backgroundColor: '#dbeafe',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    alignItems: 'center',
  },
  imageCountText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  imageCountHint: {
    fontSize: 14,
    color: '#3b82f6',
    marginTop: 4,
  },
  thumbnailScroll: {
    marginBottom: 20,
  },
  thumbnailContainer: {
    position: 'relative',
    marginRight: 12,
  },
  thumbnail: {
    width: 80,
    height: 80,
    borderRadius: 8,
  },
  removeButton: {
    position: 'absolute',
    top: -8,
    right: -8,
    backgroundColor: '#fff',
    borderRadius: 12,
  },
  processButton: {
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  processButtonText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  processingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  processingText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1e3a5f',
    marginTop: 20,
  },
  processingSubtext: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center',
  },
  summaryCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  summaryTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
    textAlign: 'center',
    marginBottom: 16,
  },
  summaryStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 32,
    fontWeight: '700',
    color: '#2563eb',
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  riskBreakdown: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  riskBars: {
    gap: 8,
  },
  riskBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    overflow: 'hidden',
  },
  riskBarFill: {
    height: 32,
    minWidth: 4,
  },
  riskBarText: {
    position: 'absolute',
    left: 12,
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  section: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  moleMapContainer: {
    alignItems: 'center',
  },
  viewToggle: {
    flexDirection: 'row',
    marginBottom: 12,
    gap: 8,
  },
  viewButton: {
    paddingVertical: 8,
    paddingHorizontal: 24,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
  },
  viewButtonActive: {
    backgroundColor: '#2563eb',
  },
  viewButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
  },
  viewButtonTextActive: {
    color: '#fff',
  },
  moleMapSvg: {
    backgroundColor: '#fafafa',
    borderRadius: 12,
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
    marginTop: 12,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  legendText: {
    fontSize: 12,
    color: '#6b7280',
  },
  lesionCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
  },
  lesionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  riskBadgeText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#fff',
  },
  lesionClass: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  lesionLocation: {
    fontSize: 14,
    color: '#6b7280',
  },
  lesionScore: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 4,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    marginBottom: 8,
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
  },
  actionButtons: {
    marginTop: 8,
  },
  actionButton: {
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  actionButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  bottomSpacer: {
    height: 40,
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    width: '100%',
    maxWidth: 400,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  lesionDetails: {
    alignItems: 'center',
  },
  riskBadgeLarge: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    marginBottom: 20,
  },
  riskBadgeLargeText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#fff',
  },
  detailLabel: {
    fontSize: 12,
    color: '#6b7280',
    textTransform: 'uppercase',
    marginTop: 12,
  },
  detailValue: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginTop: 4,
  },
  modalButton: {
    backgroundColor: '#2563eb',
    borderRadius: 10,
    padding: 14,
    alignItems: 'center',
    marginTop: 20,
  },
  modalButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
