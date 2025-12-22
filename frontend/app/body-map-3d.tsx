import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  Alert,
  ActivityIndicator,
  Modal,
} from 'react-native';
import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { API_BASE_URL } from '../config';
import { Body3DViewer } from '../components/Body3DViewer';
import {
  LesionMarker3D,
  transformLesionsTo3D,
  BODY_PART_3D_CONFIG,
} from '../utils/body3DHelpers';

const { width: screenWidth } = Dimensions.get('window');

interface BodyMapStats {
  total_lesions: number;
  high_risk: number;
  moderate_risk: number;
  low_risk: number;
  most_affected_area: string;
}

type ViewPreset = 'front' | 'back' | 'left' | 'right';

export default function BodyMap3DScreen() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [lesions, setLesions] = useState<LesionMarker3D[]>([]);
  const [stats, setStats] = useState<BodyMapStats | null>(null);
  const [selectedLesionId, setSelectedLesionId] = useState<number | null>(null);
  const [selectedBodyPart, setSelectedBodyPart] = useState<string | null>(null);
  const [viewPreset, setViewPreset] = useState<ViewPreset>('front');
  const [showLegend, setShowLegend] = useState(true);
  const [use3DView, setUse3DView] = useState(true);
  const [glError, setGlError] = useState(false);

  useEffect(() => {
    loadBodyMapData();
  }, []);

  const loadBodyMapData = async () => {
    try {
      const token = await SecureStore.getItemAsync('auth_token');
      const response = await fetch(`${API_BASE_URL}/analysis/history`, {
        headers: { 'Authorization': `Bearer ${token}` },
      });

      if (!response.ok) throw new Error('Failed to load data');

      const data = await response.json();
      const analysesWithLocation = data.filter((a: any) => a.body_location);

      // Transform to 3D markers
      const markers3D = transformLesionsTo3D(analysesWithLocation);
      setLesions(markers3D);
      calculateStats(markers3D);
    } catch (error) {
      console.error('Error loading body map:', error);
      Alert.alert('Error', 'Failed to load body map data');
    } finally {
      setLoading(false);
    }
  };

  const calculateStats = (markers: LesionMarker3D[]) => {
    const high = markers.filter(m => m.riskLevel === 'high').length;
    const moderate = markers.filter(m => m.riskLevel === 'moderate').length;
    const low = markers.filter(m => m.riskLevel === 'low').length;

    // Find most affected area
    const locationCounts: { [key: string]: number } = {};
    markers.forEach(m => {
      locationCounts[m.bodyPart] = (locationCounts[m.bodyPart] || 0) + 1;
    });

    let mostAffected = 'None';
    let maxCount = 0;
    Object.entries(locationCounts).forEach(([loc, count]) => {
      if (count > maxCount) {
        maxCount = count;
        mostAffected = loc.replace(/_/g, ' ');
      }
    });

    setStats({
      total_lesions: markers.length,
      high_risk: high,
      moderate_risk: moderate,
      low_risk: low,
      most_affected_area: mostAffected,
    });
  };

  const handleLesionSelect = useCallback((lesion: LesionMarker3D) => {
    setSelectedLesionId(lesion.id);
    setSelectedBodyPart(lesion.bodyPart);
  }, []);

  const handleBodyPartTap = useCallback((bodyPart: string) => {
    setSelectedBodyPart(bodyPart);
    setSelectedLesionId(null);
  }, []);

  const navigateToLesion = (lesionId: number) => {
    router.push(`/lesion-detail?id=${lesionId}` as any);
  };

  const getSelectedLesions = (): LesionMarker3D[] => {
    if (selectedBodyPart) {
      return lesions.filter(l =>
        l.bodyPart.toLowerCase().includes(selectedBodyPart.toLowerCase()) ||
        selectedBodyPart.toLowerCase().includes(l.bodyPart.toLowerCase())
      );
    }
    return [];
  };

  const getRiskBadgeColor = (risk: string) => {
    switch (risk?.toLowerCase()) {
      case 'high': return '#dc3545';
      case 'moderate':
      case 'medium': return '#ffc107';
      default: return '#28a745';
    }
  };

  const formatBodyPartName = (part: string): string => {
    const config = BODY_PART_3D_CONFIG[part?.toLowerCase()];
    return config?.name || part?.replace(/_/g, ' ') || 'Unknown';
  };

  if (loading) {
    return (
      <LinearGradient colors={['#1a1a2e', '#16213e', '#0f3460']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4cc9f0" />
          <Text style={styles.loadingText}>Loading 3D Body Map...</Text>
        </View>
      </LinearGradient>
    );
  }

  const selectedLesions = getSelectedLesions();
  const selectedLesion = lesions.find(l => l.id === selectedLesionId);

  return (
    <LinearGradient colors={['#1a1a2e', '#16213e', '#0f3460']} style={styles.container}>
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>3D Body Map</Text>
          <TouchableOpacity
            onPress={() => setUse3DView(!use3DView)}
            style={styles.toggleButton}
          >
            <Text style={styles.toggleButtonText}>{use3DView ? '2D' : '3D'}</Text>
          </TouchableOpacity>
        </View>

        {/* Stats Cards */}
        {stats && (
          <View style={styles.statsContainer}>
            <View style={styles.statCard}>
              <Text style={styles.statNumber}>{stats.total_lesions}</Text>
              <Text style={styles.statLabel}>Total</Text>
            </View>
            <View style={[styles.statCard, { backgroundColor: 'rgba(220, 53, 69, 0.2)' }]}>
              <Text style={[styles.statNumber, { color: '#dc3545' }]}>{stats.high_risk}</Text>
              <Text style={styles.statLabel}>High Risk</Text>
            </View>
            <View style={[styles.statCard, { backgroundColor: 'rgba(255, 193, 7, 0.2)' }]}>
              <Text style={[styles.statNumber, { color: '#ffc107' }]}>{stats.moderate_risk}</Text>
              <Text style={styles.statLabel}>Moderate</Text>
            </View>
            <View style={[styles.statCard, { backgroundColor: 'rgba(40, 167, 69, 0.2)' }]}>
              <Text style={[styles.statNumber, { color: '#28a745' }]}>{stats.low_risk}</Text>
              <Text style={styles.statLabel}>Low Risk</Text>
            </View>
          </View>
        )}

        {/* 3D Viewer */}
        {use3DView && !glError ? (
          <View style={styles.viewerContainer}>
            <Body3DViewer
              lesions={lesions}
              onLesionSelect={handleLesionSelect}
              onBodyPartTap={handleBodyPartTap}
              selectedLesionId={selectedLesionId}
              viewPreset={viewPreset}
            />

            {/* Instructions */}
            <View style={styles.instructionsContainer}>
              <Text style={styles.instructionsText}>
                Drag to rotate | Pinch to zoom | Tap to select
              </Text>
            </View>
          </View>
        ) : (
          <View style={styles.fallbackContainer}>
            <Ionicons name="cube-outline" size={64} color="rgba(255,255,255,0.3)" />
            <Text style={styles.fallbackText}>
              {glError ? '3D view not available on this device' : '2D view mode'}
            </Text>
            <TouchableOpacity
              style={styles.enable3DButton}
              onPress={() => {
                setUse3DView(true);
                setGlError(false);
              }}
            >
              <Text style={styles.enable3DButtonText}>Try 3D View</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* View Presets */}
        <View style={styles.presetContainer}>
          {(['front', 'back', 'left', 'right'] as ViewPreset[]).map((preset) => (
            <TouchableOpacity
              key={preset}
              style={[
                styles.presetButton,
                viewPreset === preset && styles.presetButtonActive,
              ]}
              onPress={() => setViewPreset(preset)}
            >
              <Text
                style={[
                  styles.presetButtonText,
                  viewPreset === preset && styles.presetButtonTextActive,
                ]}
              >
                {preset.charAt(0).toUpperCase() + preset.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
          <TouchableOpacity
            style={styles.presetButton}
            onPress={() => setViewPreset('front')}
          >
            <Ionicons name="refresh" size={18} color="white" />
          </TouchableOpacity>
        </View>

        {/* Legend */}
        <TouchableOpacity
          style={styles.legendToggle}
          onPress={() => setShowLegend(!showLegend)}
        >
          <Text style={styles.legendToggleText}>
            {showLegend ? 'Hide Legend' : 'Show Legend'}
          </Text>
          <Ionicons
            name={showLegend ? 'chevron-up' : 'chevron-down'}
            size={16}
            color="white"
          />
        </TouchableOpacity>

        {showLegend && (
          <View style={styles.legendContainer}>
            <View style={styles.legendItem}>
              <View style={[styles.legendDot, { backgroundColor: '#dc3545' }]} />
              <Text style={styles.legendText}>High Risk</Text>
            </View>
            <View style={styles.legendItem}>
              <View style={[styles.legendDot, { backgroundColor: '#ffc107' }]} />
              <Text style={styles.legendText}>Moderate Risk</Text>
            </View>
            <View style={styles.legendItem}>
              <View style={[styles.legendDot, { backgroundColor: '#28a745' }]} />
              <Text style={styles.legendText}>Low Risk</Text>
            </View>
          </View>
        )}

        {/* Selected Lesion Details */}
        {selectedLesion && (
          <View style={styles.selectedCard}>
            <View style={styles.selectedHeader}>
              <Text style={styles.selectedTitle}>Selected Lesion</Text>
              <TouchableOpacity onPress={() => setSelectedLesionId(null)}>
                <Ionicons name="close" size={20} color="white" />
              </TouchableOpacity>
            </View>
            <View style={styles.selectedContent}>
              <View style={styles.selectedRow}>
                <Text style={styles.selectedLabel}>Location:</Text>
                <Text style={styles.selectedValue}>
                  {formatBodyPartName(selectedLesion.bodyPart)}
                </Text>
              </View>
              <View style={styles.selectedRow}>
                <Text style={styles.selectedLabel}>Diagnosis:</Text>
                <Text style={styles.selectedValue}>
                  {selectedLesion.predictedClass || 'Unknown'}
                </Text>
              </View>
              <View style={styles.selectedRow}>
                <Text style={styles.selectedLabel}>Risk Level:</Text>
                <View
                  style={[
                    styles.riskBadge,
                    { backgroundColor: getRiskBadgeColor(selectedLesion.riskLevel) },
                  ]}
                >
                  <Text style={styles.riskBadgeText}>
                    {selectedLesion.riskLevel.toUpperCase()}
                  </Text>
                </View>
              </View>
              <View style={styles.selectedRow}>
                <Text style={styles.selectedLabel}>Date:</Text>
                <Text style={styles.selectedValue}>
                  {new Date(selectedLesion.date).toLocaleDateString()}
                </Text>
              </View>
              <TouchableOpacity
                style={styles.viewDetailsButton}
                onPress={() => navigateToLesion(selectedLesion.id)}
              >
                <Text style={styles.viewDetailsButtonText}>View Full Details</Text>
                <Ionicons name="arrow-forward" size={16} color="white" />
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Body Part Lesions List */}
        {selectedBodyPart && !selectedLesion && (
          <View style={styles.regionCard}>
            <Text style={styles.regionTitle}>
              {formatBodyPartName(selectedBodyPart)}
            </Text>
            <Text style={styles.regionSubtitle}>
              {selectedLesions.length} lesion{selectedLesions.length !== 1 ? 's' : ''} in this area
            </Text>

            {selectedLesions.length > 0 ? (
              selectedLesions.map((lesion) => (
                <TouchableOpacity
                  key={lesion.id}
                  style={styles.lesionItem}
                  onPress={() => navigateToLesion(lesion.id)}
                >
                  <View
                    style={[
                      styles.lesionDot,
                      { backgroundColor: getRiskBadgeColor(lesion.riskLevel) },
                    ]}
                  />
                  <View style={styles.lesionInfo}>
                    <Text style={styles.lesionClass}>
                      {lesion.predictedClass || 'Unknown'}
                    </Text>
                    <Text style={styles.lesionDate}>
                      {new Date(lesion.date).toLocaleDateString()}
                    </Text>
                  </View>
                  <Ionicons name="chevron-forward" size={18} color="rgba(255,255,255,0.5)" />
                </TouchableOpacity>
              ))
            ) : (
              <Text style={styles.noLesionsText}>No lesions recorded in this area</Text>
            )}
          </View>
        )}

        {/* Quick Stats */}
        {stats && stats.most_affected_area !== 'None' && (
          <View style={styles.insightCard}>
            <Ionicons name="analytics" size={24} color="#4cc9f0" />
            <View style={styles.insightContent}>
              <Text style={styles.insightTitle}>Most Affected Area</Text>
              <Text style={styles.insightValue}>{stats.most_affected_area}</Text>
            </View>
          </View>
        )}

        {/* Empty State */}
        {lesions.length === 0 && (
          <View style={styles.emptyState}>
            <Ionicons name="body-outline" size={64} color="rgba(255,255,255,0.3)" />
            <Text style={styles.emptyTitle}>No Lesions Mapped</Text>
            <Text style={styles.emptyText}>
              Analyze skin lesions with body location data to see them mapped here.
            </Text>
            <TouchableOpacity
              style={styles.analyzeButton}
              onPress={() => router.push('/home' as any)}
            >
              <Text style={styles.analyzeButtonText}>Analyze a Lesion</Text>
            </TouchableOpacity>
          </View>
        )}

        <View style={styles.bottomPadding} />
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: 'white',
    marginTop: 16,
    fontSize: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 60,
    paddingBottom: 16,
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
  },
  toggleButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 8,
  },
  toggleButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  statsContainer: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    marginBottom: 16,
    gap: 8,
  },
  statCard: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
  },
  statLabel: {
    fontSize: 10,
    color: 'rgba(255,255,255,0.7)',
    marginTop: 4,
  },
  viewerContainer: {
    paddingHorizontal: 16,
    marginBottom: 16,
  },
  instructionsContainer: {
    marginTop: 8,
    alignItems: 'center',
  },
  instructionsText: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 12,
  },
  fallbackContainer: {
    marginHorizontal: 16,
    marginBottom: 16,
    height: 300,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  fallbackText: {
    color: 'rgba(255,255,255,0.5)',
    marginTop: 16,
    fontSize: 16,
  },
  enable3DButton: {
    marginTop: 16,
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: '#4cc9f0',
    borderRadius: 8,
  },
  enable3DButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  presetContainer: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    marginBottom: 16,
    gap: 8,
  },
  presetButton: {
    flex: 1,
    paddingVertical: 10,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  presetButtonActive: {
    backgroundColor: '#4cc9f0',
  },
  presetButtonText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 12,
    fontWeight: '600',
  },
  presetButtonTextActive: {
    color: 'white',
  },
  legendToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    marginHorizontal: 16,
  },
  legendToggleText: {
    color: 'rgba(255,255,255,0.7)',
    marginRight: 8,
    fontSize: 14,
  },
  legendContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    paddingHorizontal: 16,
    marginBottom: 16,
    gap: 24,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  legendText: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 12,
  },
  selectedCard: {
    marginHorizontal: 16,
    marginBottom: 16,
    backgroundColor: 'rgba(76, 201, 240, 0.15)',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(76, 201, 240, 0.3)',
    overflow: 'hidden',
  },
  selectedHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: 'rgba(76, 201, 240, 0.1)',
  },
  selectedTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
  },
  selectedContent: {
    padding: 16,
  },
  selectedRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  selectedLabel: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 14,
  },
  selectedValue: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  riskBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  riskBadgeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  viewDetailsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#4cc9f0',
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 8,
    gap: 8,
  },
  viewDetailsButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  regionCard: {
    marginHorizontal: 16,
    marginBottom: 16,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 16,
    padding: 16,
  },
  regionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 4,
  },
  regionSubtitle: {
    color: 'rgba(255,255,255,0.6)',
    marginBottom: 16,
  },
  lesionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.1)',
  },
  lesionDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 12,
  },
  lesionInfo: {
    flex: 1,
  },
  lesionClass: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  lesionDate: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 12,
    marginTop: 2,
  },
  noLesionsText: {
    color: 'rgba(255,255,255,0.5)',
    textAlign: 'center',
    paddingVertical: 16,
  },
  insightCard: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: 16,
    marginBottom: 16,
    backgroundColor: 'rgba(76, 201, 240, 0.1)',
    borderRadius: 12,
    padding: 16,
  },
  insightContent: {
    marginLeft: 12,
  },
  insightTitle: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 12,
  },
  insightValue: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    textTransform: 'capitalize',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 48,
    paddingHorizontal: 32,
  },
  emptyTitle: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 16,
  },
  emptyText: {
    color: 'rgba(255,255,255,0.6)',
    textAlign: 'center',
    marginTop: 8,
    lineHeight: 20,
  },
  analyzeButton: {
    marginTop: 24,
    paddingHorizontal: 24,
    paddingVertical: 12,
    backgroundColor: '#4cc9f0',
    borderRadius: 8,
  },
  analyzeButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  bottomPadding: {
    height: 40,
  },
});
