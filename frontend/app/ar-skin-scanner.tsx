import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  Alert,
  ActivityIndicator,
  Modal,
} from 'react-native';
import { Camera, CameraView } from 'expo-camera';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  FadeIn,
  FadeOut,
} from 'react-native-reanimated';
import { API_BASE_URL } from '../config';

const { width, height } = Dimensions.get('window');

interface DetectedRegion {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  risk_level: string;
  confidence: number;
  label: string;
  description: string;
  color: string;
}

interface ScanResult {
  regions: DetectedRegion[];
  scan_time_ms: number;
  frame_quality: string;
  guidance: string;
}

export default function ARSkinScannerScreen() {
  const router = useRouter();
  const cameraRef = useRef<CameraView>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<DetectedRegion | null>(null);
  const [showInfoModal, setShowInfoModal] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  // Animation for scanning effect
  const scanLinePosition = useSharedValue(0);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  useEffect(() => {
    // Animate scan line
    scanLinePosition.value = withRepeat(
      withTiming(1, { duration: 2000 }),
      -1,
      true
    );
  }, []);

  const scanLineStyle = useAnimatedStyle(() => ({
    top: `${scanLinePosition.value * 100}%`,
  }));

  const captureAndAnalyze = async () => {
    if (!cameraRef.current || isScanning || isPaused) return;

    setIsScanning(true);

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.5,
        base64: false,
      });

      if (!photo) {
        throw new Error('Failed to capture photo');
      }

      const formData = new FormData();
      formData.append('image', {
        uri: photo.uri,
        type: 'image/jpeg',
        name: 'scan_frame.jpg',
      } as any);

      const response = await fetch(`${API_BASE_URL}/api/ar-scanner/scan?fast_mode=true`, {
        method: 'POST',
        body: formData,
        // Note: Don't set Content-Type header - fetch will set it automatically with the correct boundary
      });

      if (!response.ok) {
        throw new Error('Scan failed');
      }

      const data: ScanResult = await response.json();
      setScanResult(data);

      // Auto-scan again after delay if not paused
      if (!isPaused) {
        setTimeout(captureAndAnalyze, 1500);
      }
    } catch (error) {
      console.error('Scan error:', error);
      // Retry after error
      if (!isPaused) {
        setTimeout(captureAndAnalyze, 2000);
      }
    } finally {
      setIsScanning(false);
    }
  };

  const startScanning = () => {
    setIsPaused(false);
    captureAndAnalyze();
  };

  const pauseScanning = () => {
    setIsPaused(true);
  };

  const handleRegionPress = (region: DetectedRegion) => {
    setSelectedRegion(region);
    setShowInfoModal(true);
    pauseScanning();
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return '#EF4444';
      case 'moderate': return '#F59E0B';
      case 'low': return '#22C55E';
      default: return '#6B7280';
    }
  };

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case 'high': return 'warning';
      case 'moderate': return 'alert-circle';
      case 'low': return 'checkmark-circle';
      default: return 'help-circle';
    }
  };

  if (hasPermission === null) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#8B5CF6" />
        <Text style={styles.loadingText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.noPermissionContainer}>
        <Ionicons name="camera-outline" size={64} color="#6B7280" />
        <Text style={styles.noPermissionTitle}>Camera Access Required</Text>
        <Text style={styles.noPermissionText}>
          Please enable camera access in your device settings to use the AR Scanner.
        </Text>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Text style={styles.backButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera View */}
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
      >
        {/* AR Overlay */}
        <View style={styles.overlay}>
          {/* Scanning animation */}
          {!isPaused && (
            <Animated.View style={[styles.scanLine, scanLineStyle]} />
          )}

          {/* Detected regions */}
          {scanResult?.regions.map((region) => (
            <TouchableOpacity
              key={region.id}
              style={[
                styles.regionBox,
                {
                  left: `${(region.x - region.width / 2) * 100}%`,
                  top: `${(region.y - region.height / 2) * 100}%`,
                  width: `${region.width * 100}%`,
                  height: `${region.height * 100}%`,
                  borderColor: region.color,
                },
              ]}
              onPress={() => handleRegionPress(region)}
            >
              <View style={[styles.regionLabel, { backgroundColor: region.color }]}>
                <Text style={styles.regionLabelText}>{region.label}</Text>
              </View>
              <View style={styles.regionCorners}>
                <View style={[styles.corner, styles.cornerTL, { borderColor: region.color }]} />
                <View style={[styles.corner, styles.cornerTR, { borderColor: region.color }]} />
                <View style={[styles.corner, styles.cornerBL, { borderColor: region.color }]} />
                <View style={[styles.corner, styles.cornerBR, { borderColor: region.color }]} />
              </View>
            </TouchableOpacity>
          ))}

          {/* Guidance bar */}
          <View style={styles.guidanceBar}>
            <View style={[
              styles.qualityIndicator,
              { backgroundColor: scanResult?.frame_quality === 'good' ? '#22C55E' : '#F59E0B' }
            ]} />
            <Text style={styles.guidanceText}>
              {scanResult?.guidance || 'Point camera at skin to scan'}
            </Text>
          </View>
        </View>

        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity style={styles.headerButton} onPress={() => router.back()}>
            <Ionicons name="close" size={28} color="#FFF" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>AR Skin Scanner</Text>
          <TouchableOpacity style={styles.headerButton} onPress={() => Alert.alert(
            'AR Skin Scanner',
            'Point your camera at your skin. The AI will detect and analyze any lesions in real-time.\n\n' +
            'Colors:\n' +
            '- Green: Low risk\n' +
            '- Yellow: Monitor\n' +
            '- Red: See a doctor\n\n' +
            'Tap on any detected area for more details.'
          )}>
            <Ionicons name="help-circle" size={28} color="#FFF" />
          </TouchableOpacity>
        </View>

        {/* Stats bar */}
        <View style={styles.statsBar}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{scanResult?.regions.length || 0}</Text>
            <Text style={styles.statLabel}>Detected</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{scanResult?.scan_time_ms || '-'}ms</Text>
            <Text style={styles.statLabel}>Scan Time</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <View style={[styles.statusDot, { backgroundColor: isScanning ? '#22C55E' : '#6B7280' }]} />
            <Text style={styles.statLabel}>{isScanning ? 'Scanning' : 'Ready'}</Text>
          </View>
        </View>

        {/* Control buttons */}
        <View style={styles.controls}>
          {isPaused ? (
            <TouchableOpacity style={styles.scanButton} onPress={startScanning}>
              <Ionicons name="play" size={32} color="#FFF" />
              <Text style={styles.scanButtonText}>Resume</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={[styles.scanButton, styles.pauseButton]} onPress={pauseScanning}>
              <Ionicons name="pause" size={32} color="#FFF" />
              <Text style={styles.scanButtonText}>Pause</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Risk legend */}
        <View style={styles.legend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#22C55E' }]} />
            <Text style={styles.legendText}>Low</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#F59E0B' }]} />
            <Text style={styles.legendText}>Monitor</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#EF4444' }]} />
            <Text style={styles.legendText}>High</Text>
          </View>
        </View>
      </CameraView>

      {/* Region Info Modal */}
      <Modal
        visible={showInfoModal}
        transparent
        animationType="slide"
        onRequestClose={() => {
          setShowInfoModal(false);
          startScanning();
        }}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            {selectedRegion && (
              <>
                <View style={[styles.modalHeader, { backgroundColor: getRiskColor(selectedRegion.risk_level) }]}>
                  <Ionicons
                    name={getRiskIcon(selectedRegion.risk_level) as any}
                    size={32}
                    color="#FFF"
                  />
                  <Text style={styles.modalTitle}>{selectedRegion.label}</Text>
                </View>

                <View style={styles.modalBody}>
                  <View style={styles.confidenceRow}>
                    <Text style={styles.confidenceLabel}>Confidence:</Text>
                    <View style={styles.confidenceBar}>
                      <View
                        style={[
                          styles.confidenceFill,
                          {
                            width: `${selectedRegion.confidence * 100}%`,
                            backgroundColor: getRiskColor(selectedRegion.risk_level),
                          },
                        ]}
                      />
                    </View>
                    <Text style={styles.confidenceValue}>
                      {Math.round(selectedRegion.confidence * 100)}%
                    </Text>
                  </View>

                  <Text style={styles.descriptionText}>{selectedRegion.description}</Text>

                  {selectedRegion.risk_level === 'high' && (
                    <View style={styles.actionBox}>
                      <Ionicons name="calendar" size={24} color="#EF4444" />
                      <View style={styles.actionTextContainer}>
                        <Text style={styles.actionTitle}>Schedule an Appointment</Text>
                        <Text style={styles.actionSubtitle}>
                          This area should be examined by a dermatologist
                        </Text>
                      </View>
                    </View>
                  )}

                  {selectedRegion.risk_level === 'moderate' && (
                    <View style={[styles.actionBox, { borderColor: '#F59E0B' }]}>
                      <Ionicons name="eye" size={24} color="#F59E0B" />
                      <View style={styles.actionTextContainer}>
                        <Text style={styles.actionTitle}>Monitor This Spot</Text>
                        <Text style={styles.actionSubtitle}>
                          Take photos monthly to track any changes
                        </Text>
                      </View>
                    </View>
                  )}
                </View>

                <TouchableOpacity
                  style={styles.closeModalButton}
                  onPress={() => {
                    setShowInfoModal(false);
                    startScanning();
                  }}
                >
                  <Text style={styles.closeModalText}>Continue Scanning</Text>
                </TouchableOpacity>
              </>
            )}
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    color: '#FFF',
    marginTop: 16,
  },
  noPermissionContainer: {
    flex: 1,
    backgroundColor: '#1F2937',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
  },
  noPermissionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFF',
    marginTop: 16,
  },
  noPermissionText: {
    fontSize: 14,
    color: '#9CA3AF',
    textAlign: 'center',
    marginTop: 8,
  },
  backButton: {
    marginTop: 24,
    paddingVertical: 12,
    paddingHorizontal: 32,
    backgroundColor: '#8B5CF6',
    borderRadius: 12,
  },
  backButtonText: {
    color: '#FFF',
    fontWeight: '600',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
  },
  scanLine: {
    position: 'absolute',
    left: 20,
    right: 20,
    height: 2,
    backgroundColor: 'rgba(139, 92, 246, 0.8)',
    shadowColor: '#8B5CF6',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 10,
  },
  regionBox: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 8,
  },
  regionLabel: {
    position: 'absolute',
    top: -24,
    left: 0,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  regionLabelText: {
    color: '#FFF',
    fontSize: 10,
    fontWeight: 'bold',
  },
  regionCorners: {
    ...StyleSheet.absoluteFillObject,
  },
  corner: {
    position: 'absolute',
    width: 12,
    height: 12,
    borderWidth: 2,
  },
  cornerTL: {
    top: -1,
    left: -1,
    borderRightWidth: 0,
    borderBottomWidth: 0,
    borderTopLeftRadius: 4,
  },
  cornerTR: {
    top: -1,
    right: -1,
    borderLeftWidth: 0,
    borderBottomWidth: 0,
    borderTopRightRadius: 4,
  },
  cornerBL: {
    bottom: -1,
    left: -1,
    borderRightWidth: 0,
    borderTopWidth: 0,
    borderBottomLeftRadius: 4,
  },
  cornerBR: {
    bottom: -1,
    right: -1,
    borderLeftWidth: 0,
    borderTopWidth: 0,
    borderBottomRightRadius: 4,
  },
  guidanceBar: {
    position: 'absolute',
    bottom: 160,
    left: 20,
    right: 20,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
  },
  qualityIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 10,
  },
  guidanceText: {
    color: '#FFF',
    fontSize: 14,
    flex: 1,
  },
  header: {
    position: 'absolute',
    top: 50,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
  },
  headerButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.5)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  statsBar: {
    position: 'absolute',
    top: 110,
    left: 20,
    right: 20,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingVertical: 10,
    borderRadius: 12,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  statLabel: {
    color: '#9CA3AF',
    fontSize: 11,
    marginTop: 2,
  },
  statDivider: {
    width: 1,
    height: 30,
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginBottom: 4,
  },
  controls: {
    position: 'absolute',
    bottom: 60,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  scanButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#8B5CF6',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 30,
    gap: 10,
  },
  pauseButton: {
    backgroundColor: '#6B7280',
  },
  scanButtonText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  legend: {
    position: 'absolute',
    bottom: 20,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 20,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  legendText: {
    color: '#FFF',
    fontSize: 12,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#FFF',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    overflow: 'hidden',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    gap: 12,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFF',
  },
  modalBody: {
    padding: 20,
  },
  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  confidenceLabel: {
    fontSize: 14,
    color: '#6B7280',
    marginRight: 10,
  },
  confidenceBar: {
    flex: 1,
    height: 8,
    backgroundColor: '#E5E7EB',
    borderRadius: 4,
    marginRight: 10,
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 4,
  },
  confidenceValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  descriptionText: {
    fontSize: 15,
    color: '#374151',
    lineHeight: 22,
    marginBottom: 20,
  },
  actionBox: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#EF4444',
    gap: 12,
  },
  actionTextContainer: {
    flex: 1,
  },
  actionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
  },
  actionSubtitle: {
    fontSize: 13,
    color: '#6B7280',
    marginTop: 2,
  },
  closeModalButton: {
    margin: 20,
    marginTop: 0,
    backgroundColor: '#8B5CF6',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  closeModalText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
