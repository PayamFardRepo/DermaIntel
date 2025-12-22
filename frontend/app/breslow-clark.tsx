/**
 * Breslow/Clark Depth Visualizer
 *
 * 3D visualization of melanoma invasion depth through skin layers.
 * Based on pathology report values with interactive depth slider.
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Platform,
  TextInput,
  Switch,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import * as SecureStore from 'expo-secure-store';
import { API_BASE_URL } from '../config';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface LayerData {
  key: string;
  name: string;
  depth_start_mm: number;
  depth_end_mm: number;
  thickness_mm: number;
  color: string;
  invaded: boolean;
  invasion_percent: number;
  description: string;
}

interface ClarkInfo {
  level: string;
  name: string;
  anatomical_location: string;
  description: string;
  prognosis: string;
  treatment: string;
}

interface VisualizationResult {
  visualization: {
    breslow_thickness_mm: number;
    breslow_category: {
      range: string;
      t_category: string;
      prognosis: string;
      five_year_survival: string;
      excision_margin: string;
      sentinel_node_biopsy: string;
    };
    clark_level: number;
    clark_info: ClarkInfo;
    layers: LayerData[];
    deepest_layer_invaded: string;
  };
  t_category: string;
  risk_factors: string[];
  protective_factors: string[];
  clinical_recommendations: {
    excision_margin: string;
    sentinel_node_biopsy: string;
    prognosis: string;
    five_year_survival: string;
  };
}

// Preset Breslow values for quick selection
const BRESLOW_PRESETS = [
  { value: 0, label: 'In situ' },
  { value: 0.5, label: '0.5 mm' },
  { value: 0.8, label: '0.8 mm' },
  { value: 1.0, label: '1.0 mm' },
  { value: 1.5, label: '1.5 mm' },
  { value: 2.0, label: '2.0 mm' },
  { value: 3.0, label: '3.0 mm' },
  { value: 4.0, label: '4.0 mm' },
  { value: 5.0, label: '5.0 mm' },
];

// Clark level options
const CLARK_OPTIONS = [
  { value: 1, label: 'I - Epidermis only' },
  { value: 2, label: 'II - Papillary dermis (partial)' },
  { value: 3, label: 'III - Papillary dermis (fills)' },
  { value: 4, label: 'IV - Reticular dermis' },
  { value: 5, label: 'V - Subcutaneous fat' },
];

export default function BreslowClarkScreen() {
  const { isAuthenticated } = useAuth();
  const router = useRouter();

  // Input state
  const [breslowThickness, setBreslowThickness] = useState('1.0');
  const [clarkLevel, setClarkLevel] = useState<number | null>(null);
  const [useClarkLevel, setUseClarkLevel] = useState(false);

  // Pathology features
  const [hasUlceration, setHasUlceration] = useState(false);
  const [mitoticRate, setMitoticRate] = useState('');
  const [hasRegression, setHasRegression] = useState(false);
  const [hasLVI, setHasLVI] = useState(false);
  const [hasPNI, setHasPNI] = useState(false);
  const [tilStatus, setTilStatus] = useState<string | null>(null);

  // Results
  const [result, setResult] = useState<VisualizationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showPathologyFeatures, setShowPathologyFeatures] = useState(false);

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    }
  }, [isAuthenticated]);

  const getAuthHeaders = async () => {
    const token = await SecureStore.getItemAsync('auth_token');
    return {
      'Authorization': `Bearer ${token}`,
    };
  };

  const visualize = async () => {
    const thickness = parseFloat(breslowThickness);
    if (isNaN(thickness) || thickness < 0) {
      Alert.alert('Invalid Input', 'Please enter a valid Breslow thickness');
      return;
    }

    setIsLoading(true);
    try {
      const headers = await getAuthHeaders();

      const formData = new FormData();
      formData.append('breslow_thickness', thickness.toString());

      if (useClarkLevel && clarkLevel) {
        formData.append('clark_level', clarkLevel.toString());
      }

      formData.append('ulceration', hasUlceration.toString());

      if (mitoticRate) {
        formData.append('mitotic_rate', mitoticRate);
      }
      if (hasRegression) {
        formData.append('regression', 'true');
      }
      if (hasLVI) {
        formData.append('lymphovascular_invasion', 'true');
      }
      if (hasPNI) {
        formData.append('perineural_invasion', 'true');
      }
      if (tilStatus) {
        formData.append('tumor_infiltrating_lymphocytes', tilStatus);
      }

      const response = await fetch(`${API_BASE_URL}/breslow-clark/visualize`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setResult(data);
      } else {
        const error = await response.text();
        Alert.alert('Error', error || 'Failed to generate visualization');
      }
    } catch (error) {
      console.error('Error visualizing:', error);
      Alert.alert('Error', 'Failed to generate visualization');
    } finally {
      setIsLoading(false);
    }
  };

  const getLayerColor = (layer: LayerData): string => {
    if (layer.invaded) {
      // Blend with red based on invasion
      const invasionOpacity = layer.invasion_percent / 100;
      return layer.color; // Keep original color, we'll add tumor overlay
    }
    return layer.color;
  };

  const getPrognosisColor = (prognosis: string): string => {
    if (prognosis.includes('Excellent')) return '#10b981';
    if (prognosis.includes('Very good')) return '#22c55e';
    if (prognosis.includes('Good')) return '#84cc16';
    if (prognosis.includes('Moderate')) return '#f59e0b';
    if (prognosis.includes('Guarded')) return '#ef4444';
    return '#6b7280';
  };

  const renderSkinVisualization = () => {
    if (!result) return null;

    const { layers, breslow_thickness_mm, clark_level } = result.visualization;
    const maxDepthShown = Math.max(breslow_thickness_mm + 1, 4); // Show at least 4mm or invasion + 1mm

    return (
      <View style={styles.visualizationContainer}>
        <Text style={styles.vizTitle}>Skin Layer Cross-Section</Text>

        {/* Depth scale */}
        <View style={styles.depthScale}>
          <Text style={styles.depthLabel}>0 mm</Text>
          <View style={styles.depthLine} />
          <Text style={styles.depthLabel}>{maxDepthShown.toFixed(1)} mm</Text>
        </View>

        {/* Skin layers */}
        <View style={styles.skinLayers}>
          {layers.map((layer, index) => {
            const layerHeight = Math.min(
              ((layer.depth_end_mm - layer.depth_start_mm) / maxDepthShown) * 200,
              80
            );

            return (
              <View key={layer.key} style={styles.layerContainer}>
                {/* Layer visualization */}
                <View
                  style={[
                    styles.layer,
                    {
                      height: Math.max(layerHeight, 30),
                      backgroundColor: layer.color,
                    }
                  ]}
                >
                  {/* Tumor invasion overlay */}
                  {layer.invaded && (
                    <View
                      style={[
                        styles.tumorOverlay,
                        {
                          width: `${Math.min(layer.invasion_percent, 100)}%`,
                          backgroundColor: 'rgba(139, 69, 19, 0.6)',
                        }
                      ]}
                    />
                  )}

                  {/* Layer label */}
                  <View style={styles.layerLabelContainer}>
                    <Text style={styles.layerName}>{layer.name}</Text>
                    {layer.invaded && (
                      <View style={styles.invadedBadge}>
                        <Text style={styles.invadedText}>
                          {layer.invasion_percent === 100 ? 'INVADED' : `${layer.invasion_percent.toFixed(0)}%`}
                        </Text>
                      </View>
                    )}
                  </View>
                </View>

                {/* Depth markers */}
                <View style={styles.depthMarkers}>
                  <Text style={styles.depthMarker}>{layer.depth_start_mm} mm</Text>
                  <Text style={styles.depthMarker}>{layer.depth_end_mm} mm</Text>
                </View>
              </View>
            );
          })}

          {/* Tumor depth indicator */}
          {breslow_thickness_mm > 0 && (
            <View
              style={[
                styles.tumorDepthIndicator,
                {
                  top: (breslow_thickness_mm / maxDepthShown) * 200,
                }
              ]}
            >
              <View style={styles.tumorDepthLine} />
              <View style={styles.tumorDepthLabel}>
                <Text style={styles.tumorDepthText}>
                  Tumor depth: {breslow_thickness_mm.toFixed(2)} mm
                </Text>
              </View>
            </View>
          )}
        </View>

        {/* Legend */}
        <View style={styles.legend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: 'rgba(139, 69, 19, 0.6)' }]} />
            <Text style={styles.legendText}>Tumor invasion</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#FFDAB9' }]} />
            <Text style={styles.legendText}>Normal tissue</Text>
          </View>
        </View>
      </View>
    );
  };

  return (
    <LinearGradient colors={['#fce7f3', '#fbcfe8', '#f9a8d4']} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#9d174d" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Breslow/Clark Visualizer</Text>
        <View style={styles.headerSpacer} />
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Input Card */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="layers" size={24} color="#9d174d" />
            <Text style={styles.cardTitle}>Pathology Values</Text>
          </View>

          {/* Breslow Thickness Input */}
          <View style={styles.inputSection}>
            <Text style={styles.inputLabel}>Breslow Thickness (mm)</Text>
            <View style={styles.thicknessInputRow}>
              <TextInput
                style={styles.thicknessInput}
                value={breslowThickness}
                onChangeText={setBreslowThickness}
                keyboardType="decimal-pad"
                placeholder="1.0"
                placeholderTextColor="#9ca3af"
              />
              <Text style={styles.unitLabel}>mm</Text>
            </View>

            {/* Quick presets */}
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.presetsScroll}>
              <View style={styles.presetsRow}>
                {BRESLOW_PRESETS.map((preset) => (
                  <TouchableOpacity
                    key={preset.value}
                    style={[
                      styles.presetButton,
                      parseFloat(breslowThickness) === preset.value && styles.presetButtonActive
                    ]}
                    onPress={() => setBreslowThickness(preset.value.toString())}
                  >
                    <Text
                      style={[
                        styles.presetText,
                        parseFloat(breslowThickness) === preset.value && styles.presetTextActive
                      ]}
                    >
                      {preset.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </ScrollView>
          </View>

          {/* Clark Level (Optional) */}
          <View style={styles.switchRow}>
            <Text style={styles.inputLabel}>Specify Clark Level</Text>
            <Switch
              value={useClarkLevel}
              onValueChange={setUseClarkLevel}
              trackColor={{ false: '#d1d5db', true: '#f9a8d4' }}
              thumbColor={useClarkLevel ? '#9d174d' : '#f4f3f4'}
            />
          </View>

          {useClarkLevel && (
            <View style={styles.clarkOptions}>
              {CLARK_OPTIONS.map((option) => (
                <TouchableOpacity
                  key={option.value}
                  style={[
                    styles.clarkOption,
                    clarkLevel === option.value && styles.clarkOptionActive
                  ]}
                  onPress={() => setClarkLevel(option.value)}
                >
                  <View style={[
                    styles.clarkRadio,
                    clarkLevel === option.value && styles.clarkRadioActive
                  ]}>
                    {clarkLevel === option.value && <View style={styles.clarkRadioInner} />}
                  </View>
                  <Text style={[
                    styles.clarkLabel,
                    clarkLevel === option.value && styles.clarkLabelActive
                  ]}>
                    {option.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          )}

          {/* Ulceration */}
          <View style={styles.switchRow}>
            <View>
              <Text style={styles.inputLabel}>Ulceration Present</Text>
              <Text style={styles.inputHint}>Affects T category staging</Text>
            </View>
            <Switch
              value={hasUlceration}
              onValueChange={setHasUlceration}
              trackColor={{ false: '#d1d5db', true: '#f9a8d4' }}
              thumbColor={hasUlceration ? '#9d174d' : '#f4f3f4'}
            />
          </View>
        </View>

        {/* Additional Pathology Features */}
        <TouchableOpacity
          style={styles.expandToggle}
          onPress={() => setShowPathologyFeatures(!showPathologyFeatures)}
        >
          <Ionicons
            name={showPathologyFeatures ? 'chevron-up' : 'chevron-down'}
            size={20}
            color="#9d174d"
          />
          <Text style={styles.expandToggleText}>
            {showPathologyFeatures ? 'Hide' : 'Show'} Additional Pathology Features
          </Text>
        </TouchableOpacity>

        {showPathologyFeatures && (
          <View style={styles.card}>
            <View style={styles.inputRow}>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Mitotic Rate (/mm²)</Text>
                <TextInput
                  style={styles.input}
                  value={mitoticRate}
                  onChangeText={setMitoticRate}
                  keyboardType="decimal-pad"
                  placeholder="e.g., 2"
                  placeholderTextColor="#9ca3af"
                />
              </View>
            </View>

            <View style={styles.switchRow}>
              <Text style={styles.inputLabel}>Regression</Text>
              <Switch
                value={hasRegression}
                onValueChange={setHasRegression}
                trackColor={{ false: '#d1d5db', true: '#f9a8d4' }}
                thumbColor={hasRegression ? '#9d174d' : '#f4f3f4'}
              />
            </View>

            <View style={styles.switchRow}>
              <Text style={styles.inputLabel}>Lymphovascular Invasion (LVI)</Text>
              <Switch
                value={hasLVI}
                onValueChange={setHasLVI}
                trackColor={{ false: '#d1d5db', true: '#f9a8d4' }}
                thumbColor={hasLVI ? '#9d174d' : '#f4f3f4'}
              />
            </View>

            <View style={styles.switchRow}>
              <Text style={styles.inputLabel}>Perineural Invasion (PNI)</Text>
              <Switch
                value={hasPNI}
                onValueChange={setHasPNI}
                trackColor={{ false: '#d1d5db', true: '#f9a8d4' }}
                thumbColor={hasPNI ? '#9d174d' : '#f4f3f4'}
              />
            </View>

            <Text style={styles.inputLabel}>Tumor-Infiltrating Lymphocytes (TILs)</Text>
            <View style={styles.tilOptions}>
              {['brisk', 'non-brisk', 'absent'].map((status) => (
                <TouchableOpacity
                  key={status}
                  style={[
                    styles.tilOption,
                    tilStatus === status && styles.tilOptionActive
                  ]}
                  onPress={() => setTilStatus(tilStatus === status ? null : status)}
                >
                  <Text style={[
                    styles.tilText,
                    tilStatus === status && styles.tilTextActive
                  ]}>
                    {status.charAt(0).toUpperCase() + status.slice(1)}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}

        {/* Visualize Button */}
        <TouchableOpacity
          style={styles.visualizeButton}
          onPress={visualize}
          disabled={isLoading}
        >
          {isLoading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <>
              <Ionicons name="eye" size={24} color="#fff" />
              <Text style={styles.visualizeButtonText}>Visualize Invasion</Text>
            </>
          )}
        </TouchableOpacity>

        {/* Results */}
        {result && (
          <>
            {/* Skin Layer Visualization */}
            {renderSkinVisualization()}

            {/* Clark Level Result */}
            <View style={styles.resultCard}>
              <View style={styles.resultHeader}>
                <View>
                  <Text style={styles.resultLabel}>Clark Level</Text>
                  <Text style={styles.clarkResult}>
                    Level {result.visualization.clark_info.level}
                  </Text>
                </View>
                <View>
                  <Text style={styles.resultLabel}>T Category</Text>
                  <Text style={styles.tCategoryResult}>{result.t_category}</Text>
                </View>
              </View>

              <Text style={styles.anatomicalLocation}>
                {result.visualization.clark_info.anatomical_location}
              </Text>
              <Text style={styles.clarkDescription}>
                {result.visualization.clark_info.description}
              </Text>
            </View>

            {/* Clinical Recommendations */}
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Clinical Recommendations</Text>

              <View style={styles.recommendationRow}>
                <Ionicons name="cut-outline" size={20} color="#9d174d" />
                <View style={styles.recommendationContent}>
                  <Text style={styles.recommendationLabel}>Excision Margin</Text>
                  <Text style={styles.recommendationValue}>
                    {result.clinical_recommendations.excision_margin}
                  </Text>
                </View>
              </View>

              <View style={styles.recommendationRow}>
                <Ionicons name="git-network-outline" size={20} color="#9d174d" />
                <View style={styles.recommendationContent}>
                  <Text style={styles.recommendationLabel}>Sentinel Node Biopsy</Text>
                  <Text style={styles.recommendationValue}>
                    {result.clinical_recommendations.sentinel_node_biopsy}
                  </Text>
                </View>
              </View>

              <View style={styles.prognosisBox}>
                <View style={[
                  styles.prognosisBadge,
                  { backgroundColor: getPrognosisColor(result.clinical_recommendations.prognosis) }
                ]}>
                  <Text style={styles.prognosisText}>
                    {result.clinical_recommendations.prognosis}
                  </Text>
                </View>
                <Text style={styles.survivalText}>
                  5-Year Survival: {result.clinical_recommendations.five_year_survival}
                </Text>
              </View>
            </View>

            {/* Risk Factors */}
            {(result.risk_factors.length > 0 || result.protective_factors.length > 0) && (
              <View style={styles.card}>
                <Text style={styles.sectionTitle}>Prognostic Factors</Text>

                {result.risk_factors.length > 0 && (
                  <>
                    <Text style={styles.factorTitle}>Risk Factors</Text>
                    {result.risk_factors.map((factor, index) => (
                      <View key={index} style={styles.factorRow}>
                        <Ionicons name="warning" size={16} color="#ef4444" />
                        <Text style={styles.riskFactorText}>{factor}</Text>
                      </View>
                    ))}
                  </>
                )}

                {result.protective_factors.length > 0 && (
                  <>
                    <Text style={[styles.factorTitle, { marginTop: 12 }]}>Protective Factors</Text>
                    {result.protective_factors.map((factor, index) => (
                      <View key={index} style={styles.factorRow}>
                        <Ionicons name="shield-checkmark" size={16} color="#10b981" />
                        <Text style={styles.protectiveFactorText}>{factor}</Text>
                      </View>
                    ))}
                  </>
                )}
              </View>
            )}

            {/* Disclaimer */}
            <View style={styles.disclaimerBox}>
              <Ionicons name="information-circle" size={18} color="#9d174d" />
              <Text style={styles.disclaimerText}>
                This visualization is for educational purposes. Pathology interpretation should be performed by qualified pathologists.
              </Text>
            </View>
          </>
        )}

        <View style={styles.bottomSpacer} />
      </ScrollView>
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
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 20,
    paddingBottom: 15,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderBottomWidth: 1,
    borderBottomColor: '#f9a8d4',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: '700',
    color: '#9d174d',
    marginLeft: 12,
  },
  headerSpacer: {
    width: 40,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 10,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
  },
  inputSection: {
    marginBottom: 16,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  inputHint: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: -4,
  },
  thicknessInputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  thicknessInput: {
    flex: 1,
    backgroundColor: '#f9fafb',
    borderWidth: 2,
    borderColor: '#e5e7eb',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 24,
    fontWeight: '700',
    color: '#1f2937',
    textAlign: 'center',
  },
  unitLabel: {
    fontSize: 18,
    fontWeight: '600',
    color: '#6b7280',
  },
  presetsScroll: {
    marginTop: 12,
    marginHorizontal: -16,
    paddingHorizontal: 16,
  },
  presetsRow: {
    flexDirection: 'row',
    gap: 8,
  },
  presetButton: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#f3f4f6',
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  presetButtonActive: {
    backgroundColor: '#9d174d',
    borderColor: '#9d174d',
  },
  presetText: {
    fontSize: 13,
    color: '#4b5563',
    fontWeight: '500',
  },
  presetTextActive: {
    color: '#fff',
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  clarkOptions: {
    marginTop: 8,
    marginBottom: 12,
  },
  clarkOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 10,
    marginBottom: 6,
    backgroundColor: '#f9fafb',
  },
  clarkOptionActive: {
    backgroundColor: '#fce7f3',
  },
  clarkRadio: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#d1d5db',
    marginRight: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  clarkRadioActive: {
    borderColor: '#9d174d',
  },
  clarkRadioInner: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#9d174d',
  },
  clarkLabel: {
    fontSize: 14,
    color: '#4b5563',
  },
  clarkLabelActive: {
    color: '#9d174d',
    fontWeight: '600',
  },
  expandToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    gap: 6,
  },
  expandToggleText: {
    color: '#9d174d',
    fontSize: 14,
    fontWeight: '600',
  },
  inputRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
  },
  inputGroup: {
    flex: 1,
  },
  input: {
    backgroundColor: '#f9fafb',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 15,
    color: '#1f2937',
  },
  tilOptions: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 8,
  },
  tilOption: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 10,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
  },
  tilOptionActive: {
    backgroundColor: '#9d174d',
  },
  tilText: {
    fontSize: 13,
    color: '#4b5563',
    fontWeight: '500',
  },
  tilTextActive: {
    color: '#fff',
  },
  visualizeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#9d174d',
    paddingVertical: 16,
    borderRadius: 14,
    marginBottom: 20,
    gap: 10,
    shadowColor: '#9d174d',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 5,
  },
  visualizeButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
  },
  visualizationContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  vizTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 16,
    textAlign: 'center',
  },
  depthScale: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    paddingHorizontal: 8,
  },
  depthLabel: {
    fontSize: 11,
    color: '#6b7280',
    width: 40,
  },
  depthLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#e5e7eb',
  },
  skinLayers: {
    position: 'relative',
    marginBottom: 16,
  },
  layerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  layer: {
    flex: 1,
    borderRadius: 4,
    marginVertical: 1,
    overflow: 'hidden',
    justifyContent: 'center',
  },
  tumorOverlay: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
  },
  layerLabelContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
  },
  layerName: {
    fontSize: 12,
    fontWeight: '600',
    color: '#1f2937',
  },
  invadedBadge: {
    backgroundColor: 'rgba(139, 69, 19, 0.8)',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  invadedText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  depthMarkers: {
    width: 50,
    alignItems: 'flex-end',
    paddingLeft: 8,
  },
  depthMarker: {
    fontSize: 9,
    color: '#9ca3af',
  },
  tumorDepthIndicator: {
    position: 'absolute',
    left: 0,
    right: 50,
    flexDirection: 'row',
    alignItems: 'center',
  },
  tumorDepthLine: {
    flex: 1,
    height: 2,
    backgroundColor: '#dc2626',
  },
  tumorDepthLabel: {
    backgroundColor: '#dc2626',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    marginLeft: 4,
  },
  tumorDepthText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 20,
    marginTop: 8,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendColor: {
    width: 16,
    height: 16,
    borderRadius: 4,
  },
  legendText: {
    fontSize: 12,
    color: '#6b7280',
  },
  resultCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    borderWidth: 2,
    borderColor: '#9d174d',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  resultLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4,
  },
  clarkResult: {
    fontSize: 28,
    fontWeight: '800',
    color: '#9d174d',
  },
  tCategoryResult: {
    fontSize: 28,
    fontWeight: '800',
    color: '#1f2937',
  },
  anatomicalLocation: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  clarkDescription: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 14,
  },
  recommendationRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
    gap: 12,
  },
  recommendationContent: {
    flex: 1,
  },
  recommendationLabel: {
    fontSize: 13,
    color: '#6b7280',
  },
  recommendationValue: {
    fontSize: 15,
    color: '#1f2937',
    fontWeight: '600',
  },
  prognosisBox: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#f9fafb',
    padding: 14,
    borderRadius: 12,
    marginTop: 8,
  },
  prognosisBadge: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 8,
  },
  prognosisText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '700',
  },
  survivalText: {
    fontSize: 14,
    color: '#374151',
    fontWeight: '600',
  },
  factorTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#4b5563',
    marginBottom: 8,
  },
  factorRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
    gap: 8,
  },
  riskFactorText: {
    flex: 1,
    fontSize: 13,
    color: '#dc2626',
  },
  protectiveFactorText: {
    flex: 1,
    fontSize: 13,
    color: '#059669',
  },
  disclaimerBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#fce7f3',
    padding: 14,
    borderRadius: 12,
    marginBottom: 16,
    gap: 10,
    borderWidth: 1,
    borderColor: '#f9a8d4',
  },
  disclaimerText: {
    flex: 1,
    fontSize: 12,
    color: '#9d174d',
    lineHeight: 18,
  },
  bottomSpacer: {
    height: 40,
  },
});
