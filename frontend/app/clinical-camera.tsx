/**
 * Clinical Photography Camera with Real-Time Quality Assessment
 *
 * Features:
 * - Real-time quality feedback overlay
 * - Ruler/color card detection alerts
 * - Lighting, focus, distance guidance
 * - AR-style guidance overlay
 * - Medical photography standards compliance
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Animated,
  Dimensions,
  ActivityIndicator,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as ImageManipulator from 'expo-image-manipulator';
import ImageAnalysisService from '../ImageAnalysisService';

const { width, height } = Dimensions.get('window');

interface QualityFeedback {
  overall_score: number;
  quality_level: 'excellent' | 'good' | 'acceptable' | 'poor' | 'unacceptable';
  meets_medical_standards: boolean;
  dicom_compliant: boolean;
  scores: {
    lighting: number;
    focus: number;
    distance: number;
    angle: number;
    scale: number;
    color_card: number;
  };
  detections: {
    ruler_detected: boolean;
    color_card_detected: boolean;
    has_glare: boolean;
    has_shadows: boolean;
    is_blurry: boolean;
    too_close: boolean;
    too_far: boolean;
  };
  measurements: {
    estimated_distance_cm: number | null;
    pixel_to_mm_ratio: number | null;
    glare_percentage: number;
    shadow_percentage: number;
  };
  feedback: {
    issues: string[];
    suggestions: string[];
    warnings: string[];
  };
  ready_to_capture: boolean;
}

export default function ClinicalCamera() {
  const router = useRouter();
  const cameraRef = useRef<any>(null);
  const [permission, requestPermission] = useCameraPermissions();

  const [facing, setFacing] = useState<'back' | 'front'>('back');
  const [flash, setFlash] = useState<'off' | 'on'>('off');
  const [isCapturing, setIsCapturing] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [qualityFeedback, setQualityFeedback] = useState<QualityFeedback | null>(null);
  const [showGuidance, setShowGuidance] = useState(true);
  const [autoAssessEnabled, setAutoAssessEnabled] = useState(true);

  // Animation values
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const feedbackOpacity = useRef(new Animated.Value(0)).current;

  // Auto-assess quality every 2 seconds
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (autoAssessEnabled && !isCapturing) {
      interval = setInterval(() => {
        capturePreviewForAssessment();
      }, 2000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoAssessEnabled, isCapturing]);

  // Pulse animation for capture button
  useEffect(() => {
    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.1,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    );

    if (qualityFeedback?.ready_to_capture) {
      pulse.start();
    } else {
      pulse.stop();
      pulseAnim.setValue(1);
    }

    return () => pulse.stop();
  }, [qualityFeedback?.ready_to_capture]);

  // Fade in feedback
  useEffect(() => {
    if (qualityFeedback) {
      Animated.timing(feedbackOpacity, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }).start();
    }
  }, [qualityFeedback]);

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need camera permission</Text>
        <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
          <Text style={styles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const capturePreviewForAssessment = async () => {
    if (!cameraRef.current || isAnalyzing || isCapturing) return;

    try {
      setIsAnalyzing(true);

      // Take preview photo
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.5, // Lower quality for preview
        base64: false,
      });

      // Resize for faster processing
      const resized = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 800 } }],
        { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG }
      );

      // Assess quality
      const feedback = await ImageAnalysisService.assessPhotoQuality(resized.uri);
      setQualityFeedback(feedback);

    } catch (error) {
      console.log('Preview assessment error:', error);
      // Don't show error to user - this is background assessment
    } finally {
      setIsAnalyzing(false);
    }
  };

  const takePicture = async () => {
    if (!cameraRef.current || isCapturing) return;

    try {
      setIsCapturing(true);

      const photo = await cameraRef.current.takePictureAsync({
        quality: 1,
        base64: false,
      });

      // Navigate to analysis with photo
      router.push({
        pathname: '/analysis/new',
        params: { photoUri: photo.uri }
      });

    } catch (error) {
      console.error('Capture error:', error);
      Alert.alert('Error', 'Failed to capture photo');
    } finally {
      setIsCapturing(false);
    }
  };

  const toggleFlash = () => {
    setFlash(current => current === 'off' ? 'on' : 'off');
  };

  const toggleCamera = () => {
    setFacing(current => current === 'back' ? 'front' : 'back');
  };

  const getQualityColor = (level: string) => {
    switch (level) {
      case 'excellent':
        return '#10b981';
      case 'good':
        return '#3b82f6';
      case 'acceptable':
        return '#f59e0b';
      case 'poor':
        return '#ef4444';
      case 'unacceptable':
        return '#dc2626';
      default:
        return '#6b7280';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 85) return '#10b981';
    if (score >= 75) return '#3b82f6';
    if (score >= 65) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <View style={styles.container}>
      {/* Camera View */}
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={facing}
        flash={flash}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity
            style={styles.headerButton}
            onPress={() => router.back()}
          >
            <Ionicons name="close" size={28} color="white" />
          </TouchableOpacity>

          <Text style={styles.headerTitle}>Clinical Photography</Text>

          <TouchableOpacity
            style={styles.headerButton}
            onPress={() => setShowGuidance(!showGuidance)}
          >
            <Ionicons
              name={showGuidance ? 'eye' : 'eye-off'}
              size={24}
              color="white"
            />
          </TouchableOpacity>
        </View>

        {/* Quality Feedback Overlay */}
        {showGuidance && qualityFeedback && (
          <Animated.View
            style={[styles.feedbackOverlay, { opacity: feedbackOpacity }]}
          >
            {/* Overall Quality Badge */}
            <View
              style={[
                styles.qualityBadge,
                { backgroundColor: getQualityColor(qualityFeedback.quality_level) }
              ]}
            >
              <Text style={styles.qualityScore}>
                {qualityFeedback.overall_score.toFixed(0)}
              </Text>
              <Text style={styles.qualityLevel}>
                {qualityFeedback.quality_level.toUpperCase()}
              </Text>
            </View>

            {/* Warnings (Critical) */}
            {qualityFeedback.feedback.warnings.length > 0 && (
              <View style={styles.feedbackSection}>
                {qualityFeedback.feedback.warnings.map((warning, index) => (
                  <View key={index} style={[styles.feedbackItem, styles.warningItem]}>
                    <Ionicons name="warning" size={16} color="#dc2626" />
                    <Text style={styles.warningText}>{warning}</Text>
                  </View>
                ))}
              </View>
            )}

            {/* Suggestions */}
            {qualityFeedback.feedback.suggestions.length > 0 && (
              <View style={styles.feedbackSection}>
                {qualityFeedback.feedback.suggestions.slice(0, 2).map((suggestion, index) => (
                  <View key={index} style={[styles.feedbackItem, styles.suggestionItem]}>
                    <Ionicons name="information-circle" size={16} color="#3b82f6" />
                    <Text style={styles.suggestionText}>{suggestion}</Text>
                  </View>
                ))}
              </View>
            )}

            {/* Detection Status Icons */}
            <View style={styles.detectionsRow}>
              <View style={styles.detectionItem}>
                <Ionicons
                  name={qualityFeedback.detections.ruler_detected ? 'checkmark-circle' : 'close-circle'}
                  size={20}
                  color={qualityFeedback.detections.ruler_detected ? '#10b981' : '#ef4444'}
                />
                <Text style={styles.detectionLabel}>Ruler</Text>
              </View>

              <View style={styles.detectionItem}>
                <Ionicons
                  name={qualityFeedback.detections.color_card_detected ? 'checkmark-circle' : 'close-circle'}
                  size={20}
                  color={qualityFeedback.detections.color_card_detected ? '#10b981' : '#ef4444'}
                />
                <Text style={styles.detectionLabel}>Color Card</Text>
              </View>

              <View style={styles.detectionItem}>
                <Ionicons
                  name={!qualityFeedback.detections.has_glare ? 'checkmark-circle' : 'close-circle'}
                  size={20}
                  color={!qualityFeedback.detections.has_glare ? '#10b981' : '#ef4444'}
                />
                <Text style={styles.detectionLabel}>No Glare</Text>
              </View>

              <View style={styles.detectionItem}>
                <Ionicons
                  name={!qualityFeedback.detections.is_blurry ? 'checkmark-circle' : 'close-circle'}
                  size={20}
                  color={!qualityFeedback.detections.is_blurry ? '#10b981' : '#ef4444'}
                />
                <Text style={styles.detectionLabel}>Sharp</Text>
              </View>
            </View>

            {/* Quick Scores */}
            <View style={styles.scoresRow}>
              <View style={styles.scoreItem}>
                <Text style={styles.scoreLabel}>Light</Text>
                <Text style={[styles.scoreValue, { color: getScoreColor(qualityFeedback.scores.lighting) }]}>
                  {qualityFeedback.scores.lighting.toFixed(0)}
                </Text>
              </View>
              <View style={styles.scoreItem}>
                <Text style={styles.scoreLabel}>Focus</Text>
                <Text style={[styles.scoreValue, { color: getScoreColor(qualityFeedback.scores.focus) }]}>
                  {qualityFeedback.scores.focus.toFixed(0)}
                </Text>
              </View>
              <View style={styles.scoreItem}>
                <Text style={styles.scoreLabel}>Distance</Text>
                <Text style={[styles.scoreValue, { color: getScoreColor(qualityFeedback.scores.distance) }]}>
                  {qualityFeedback.scores.distance.toFixed(0)}
                </Text>
              </View>
              <View style={styles.scoreItem}>
                <Text style={styles.scoreLabel}>Angle</Text>
                <Text style={[styles.scoreValue, { color: getScoreColor(qualityFeedback.scores.angle) }]}>
                  {qualityFeedback.scores.angle.toFixed(0)}
                </Text>
              </View>
            </View>

            {/* Standards Compliance */}
            <View style={styles.complianceRow}>
              {qualityFeedback.meets_medical_standards && (
                <View style={styles.complianceBadge}>
                  <Ionicons name="checkmark-circle" size={16} color="#10b981" />
                  <Text style={styles.complianceText}>Medical Standards</Text>
                </View>
              )}
              {qualityFeedback.dicom_compliant && (
                <View style={styles.complianceBadge}>
                  <Ionicons name="checkmark-circle" size={16} color="#10b981" />
                  <Text style={styles.complianceText}>DICOM Compliant</Text>
                </View>
              )}
            </View>
          </Animated.View>
        )}

        {/* AR Guide Lines (Simple Grid) */}
        {showGuidance && (
          <View style={styles.guideLines}>
            <View style={styles.horizontalLine} />
            <View style={styles.verticalLine} />
            <View style={[styles.horizontalLine, { top: '66%' }]} />
            <View style={[styles.verticalLine, { left: '66%' }]} />
          </View>
        )}

        {/* Bottom Controls */}
        <View style={styles.controls}>
          {/* Flash Toggle */}
          <TouchableOpacity style={styles.controlButton} onPress={toggleFlash}>
            <Ionicons
              name={flash === 'on' ? 'flash' : 'flash-off'}
              size={28}
              color="white"
            />
          </TouchableOpacity>

          {/* Capture Button */}
          <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
            <TouchableOpacity
              style={[
                styles.captureButton,
                qualityFeedback?.ready_to_capture && styles.captureButtonReady,
                isCapturing && styles.captureButtonDisabled
              ]}
              onPress={takePicture}
              disabled={isCapturing}
            >
              {isCapturing ? (
                <ActivityIndicator size="large" color="white" />
              ) : (
                <View style={styles.captureButtonInner} />
              )}
            </TouchableOpacity>
          </Animated.View>

          {/* Camera Toggle */}
          <TouchableOpacity style={styles.controlButton} onPress={toggleCamera}>
            <Ionicons name="camera-reverse" size={28} color="white" />
          </TouchableOpacity>
        </View>

        {/* Auto-Assess Toggle */}
        <TouchableOpacity
          style={styles.autoAssessToggle}
          onPress={() => setAutoAssessEnabled(!autoAssessEnabled)}
        >
          <Ionicons
            name={autoAssessEnabled ? 'bulb' : 'bulb-outline'}
            size={20}
            color={autoAssessEnabled ? '#10b981' : 'white'}
          />
          <Text style={styles.autoAssessText}>
            {autoAssessEnabled ? 'AI On' : 'AI Off'}
          </Text>
        </TouchableOpacity>

        {/* Analysis Indicator */}
        {isAnalyzing && (
          <View style={styles.analyzingIndicator}>
            <ActivityIndicator size="small" color="white" />
            <Text style={styles.analyzingText}>Analyzing...</Text>
          </View>
        )}
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  camera: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 50,
    paddingHorizontal: 20,
    paddingBottom: 15,
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  headerButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  feedbackOverlay: {
    position: 'absolute',
    top: 120,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0,0,0,0.75)',
    borderRadius: 12,
    padding: 15,
  },
  qualityBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    marginBottom: 12,
  },
  qualityScore: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    marginRight: 8,
  },
  qualityLevel: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  feedbackSection: {
    marginBottom: 10,
  },
  feedbackItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  warningItem: {
    backgroundColor: 'rgba(220, 38, 38, 0.2)',
    padding: 8,
    borderRadius: 6,
  },
  suggestionItem: {
    backgroundColor: 'rgba(59, 130, 246, 0.2)',
    padding: 8,
    borderRadius: 6,
  },
  warningText: {
    color: '#fca5a5',
    fontSize: 12,
    marginLeft: 6,
    flex: 1,
  },
  suggestionText: {
    color: '#93c5fd',
    fontSize: 12,
    marginLeft: 6,
    flex: 1,
  },
  detectionsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 10,
    marginBottom: 8,
  },
  detectionItem: {
    alignItems: 'center',
  },
  detectionLabel: {
    color: 'white',
    fontSize: 10,
    marginTop: 4,
  },
  scoresRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.2)',
  },
  scoreItem: {
    alignItems: 'center',
  },
  scoreLabel: {
    color: '#9ca3af',
    fontSize: 10,
    marginBottom: 4,
  },
  scoreValue: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  complianceRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 10,
    gap: 10,
  },
  complianceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(16, 185, 129, 0.2)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  complianceText: {
    color: '#6ee7b7',
    fontSize: 10,
    marginLeft: 4,
    fontWeight: '600',
  },
  guideLines: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  horizontalLine: {
    position: 'absolute',
    top: '33%',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.3)',
  },
  verticalLine: {
    position: 'absolute',
    left: '33%',
    top: 0,
    bottom: 0,
    width: 1,
    backgroundColor: 'rgba(255,255,255,0.3)',
  },
  controls: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  controlButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  captureButtonReady: {
    borderColor: '#10b981',
    backgroundColor: '#10b981',
  },
  captureButtonDisabled: {
    opacity: 0.5,
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'white',
  },
  autoAssessToggle: {
    position: 'absolute',
    top: 110,
    right: 20,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  autoAssessText: {
    color: 'white',
    fontSize: 12,
    marginLeft: 6,
    fontWeight: '600',
  },
  analyzingIndicator: {
    position: 'absolute',
    top: 160,
    right: 20,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
  },
  analyzingText: {
    color: 'white',
    fontSize: 11,
    marginLeft: 6,
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
    color: 'white',
    fontSize: 16,
  },
  permissionButton: {
    backgroundColor: '#2563eb',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    alignSelf: 'center',
  },
  permissionButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
