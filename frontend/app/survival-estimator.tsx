import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { API_ENDPOINTS } from '../config';

const { width } = Dimensions.get('window');

interface SurvivalPoint {
  year: number;
  survival_percent: number;
}

interface AppliedFactor {
  factor: string;
  status: string;
  hazard_ratio: number;
  effect: 'Adverse' | 'Favorable' | 'Neutral' | 'Uncertain';
}

interface SurvivalResult {
  patient_characteristics: any;
  staging: { stage: string; description: string };
  survival_estimate: {
    five_year_survival: string;
    ten_year_survival: string;
    median_survival_years: number | null;
    risk_category: string;
    risk_color: string;
  };
  hazard_analysis: {
    combined_hazard_ratio: number;
    applied_factors: AppliedFactor[];
    interpretation: string;
  };
  survival_curves: {
    patient_specific: SurvivalPoint[];
    stage_baseline: SurvivalPoint[];
  };
  comparison: {
    difference_from_baseline_5yr: number;
    patient_vs_baseline: string;
  };
  disclaimer: string;
  data_sources: string[];
}

export default function SurvivalEstimatorScreen() {
  const router = useRouter();
  const { token } = useAuth();

  // Required inputs
  const [breslowThickness, setBreslowThickness] = useState('');
  const [ulceration, setUlceration] = useState(false);

  // Optional tumor characteristics
  const [mitoticRate, setMitoticRate] = useState('');
  const [sentinelNode, setSentinelNode] = useState<string | null>(null);
  const [distantMetastasis, setDistantMetastasis] = useState(false);

  // Patient characteristics
  const [age, setAge] = useState('');
  const [sex, setSex] = useState<string | null>(null);
  const [tumorLocation, setTumorLocation] = useState<string | null>(null);

  // Additional pathology features
  const [lvi, setLvi] = useState(false);
  const [regression, setRegression] = useState(false);
  const [microsatellites, setMicrosatellites] = useState(false);
  const [tils, setTils] = useState<string | null>(null);

  // State
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SurvivalResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const calculateSurvival = async () => {
    if (!breslowThickness) {
      setError('Breslow thickness is required');
      return;
    }

    const thickness = parseFloat(breslowThickness);
    if (isNaN(thickness) || thickness < 0) {
      setError('Invalid Breslow thickness value');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('breslow_thickness', thickness.toString());
      formData.append('ulceration', ulceration.toString());
      formData.append('distant_metastasis', distantMetastasis.toString());
      formData.append('lymphovascular_invasion', lvi.toString());
      formData.append('regression', regression.toString());
      formData.append('microsatellites', microsatellites.toString());
      formData.append('years_to_project', '10');

      if (mitoticRate) {
        formData.append('mitotic_rate', mitoticRate);
      }
      if (sentinelNode) {
        formData.append('sentinel_node_status', sentinelNode);
      }
      if (age) {
        formData.append('age', age);
      }
      if (sex) {
        formData.append('sex', sex);
      }
      if (tumorLocation) {
        formData.append('tumor_location', tumorLocation);
      }
      if (tils) {
        formData.append('tils', tils);
      }

      const response = await fetch(`${API_ENDPOINTS.BASE_URL}/clinical/survival/estimate`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to calculate survival');
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error('Survival calculation error:', err);
      setError(err.message || 'Failed to calculate survival estimate');
    } finally {
      setIsLoading(false);
    }
  };

  const renderOptionButton = (
    label: string,
    value: string | null,
    currentValue: string | null,
    onSelect: (val: string | null) => void
  ) => (
    <TouchableOpacity
      style={[
        styles.optionButton,
        currentValue === value && styles.optionButtonSelected,
      ]}
      onPress={() => onSelect(currentValue === value ? null : value)}
    >
      <Text
        style={[
          styles.optionButtonText,
          currentValue === value && styles.optionButtonTextSelected,
        ]}
      >
        {label}
      </Text>
    </TouchableOpacity>
  );

  const renderToggleButton = (
    label: string,
    value: boolean,
    onToggle: (val: boolean) => void
  ) => (
    <TouchableOpacity
      style={[styles.toggleButton, value && styles.toggleButtonActive]}
      onPress={() => onToggle(!value)}
    >
      <Text style={[styles.toggleButtonText, value && styles.toggleButtonTextActive]}>
        {label}
      </Text>
      {value && <Ionicons name="checkmark" size={16} color="#fff" style={{ marginLeft: 4 }} />}
    </TouchableOpacity>
  );

  const renderSurvivalCurve = () => {
    if (!result) return null;

    const { patient_specific, stage_baseline } = result.survival_curves;
    const maxYear = 10;
    const chartHeight = 180;
    const chartWidth = width - 80;

    return (
      <View style={styles.chartContainer}>
        <Text style={styles.chartTitle}>Survival Curves</Text>
        <View style={styles.chart}>
          {/* Y-axis */}
          <View style={styles.yAxis}>
            <Text style={styles.axisLabel}>100%</Text>
            <Text style={styles.axisLabel}>75%</Text>
            <Text style={styles.axisLabel}>50%</Text>
            <Text style={styles.axisLabel}>25%</Text>
            <Text style={styles.axisLabel}>0%</Text>
          </View>

          {/* Chart area */}
          <View style={[styles.chartArea, { height: chartHeight, width: chartWidth }]}>
            {/* Grid lines */}
            {[0, 25, 50, 75, 100].map((pct) => (
              <View
                key={pct}
                style={[
                  styles.gridLine,
                  { bottom: (pct / 100) * chartHeight },
                ]}
              />
            ))}

            {/* Baseline curve (dashed effect with dots) */}
            {stage_baseline.slice(0, maxYear + 1).map((point, i) => (
              <View
                key={`baseline-${i}`}
                style={[
                  styles.curvePoint,
                  styles.baselinePoint,
                  {
                    left: (point.year / maxYear) * chartWidth - 4,
                    bottom: (point.survival_percent / 100) * chartHeight - 4,
                  },
                ]}
              />
            ))}

            {/* Patient curve */}
            {patient_specific.slice(0, maxYear + 1).map((point, i) => (
              <View
                key={`patient-${i}`}
                style={[
                  styles.curvePoint,
                  styles.patientPoint,
                  {
                    left: (point.year / maxYear) * chartWidth - 5,
                    bottom: (point.survival_percent / 100) * chartHeight - 5,
                  },
                ]}
              />
            ))}
          </View>
        </View>

        {/* X-axis */}
        <View style={styles.xAxis}>
          {[0, 2, 4, 6, 8, 10].map((year) => (
            <Text key={year} style={styles.axisLabel}>
              {year}y
            </Text>
          ))}
        </View>

        {/* Legend */}
        <View style={styles.legend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: result.survival_estimate.risk_color }]} />
            <Text style={styles.legendText}>Your Estimate</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#9ca3af', borderStyle: 'dashed' }]} />
            <Text style={styles.legendText}>Stage Baseline</Text>
          </View>
        </View>
      </View>
    );
  };

  const renderFactorCard = (factor: AppliedFactor) => {
    const effectColors = {
      Adverse: '#ef4444',
      Favorable: '#10b981',
      Neutral: '#6b7280',
      Uncertain: '#f59e0b',
    };

    return (
      <View key={factor.factor} style={styles.factorCard}>
        <View style={styles.factorHeader}>
          <Text style={styles.factorName}>{factor.factor}</Text>
          <View
            style={[
              styles.effectBadge,
              { backgroundColor: effectColors[factor.effect] + '20' },
            ]}
          >
            <Text style={[styles.effectText, { color: effectColors[factor.effect] }]}>
              {factor.effect}
            </Text>
          </View>
        </View>
        <View style={styles.factorDetails}>
          <Text style={styles.factorStatus}>{factor.status}</Text>
          <Text style={styles.factorHR}>HR: {factor.hazard_ratio.toFixed(2)}</Text>
        </View>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>Survival Estimator</Text>
          <Text style={styles.headerSubtitle}>
            ML-based survival curves using tumor characteristics
          </Text>
        </View>
      </LinearGradient>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Primary Inputs */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            <Ionicons name="analytics-outline" size={18} color="#0ea5e9" /> Primary Tumor
          </Text>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Breslow Thickness (mm) *</Text>
            <TextInput
              style={styles.textInput}
              value={breslowThickness}
              onChangeText={setBreslowThickness}
              placeholder="e.g., 1.5"
              placeholderTextColor="#6b7280"
              keyboardType="decimal-pad"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Ulceration</Text>
            <View style={styles.toggleRow}>
              {renderToggleButton('Present', ulceration, setUlceration)}
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Mitotic Rate (/mm²)</Text>
            <TextInput
              style={styles.textInput}
              value={mitoticRate}
              onChangeText={setMitoticRate}
              placeholder="Optional"
              placeholderTextColor="#6b7280"
              keyboardType="decimal-pad"
            />
          </View>
        </View>

        {/* Sentinel Node */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            <Ionicons name="git-network-outline" size={18} color="#0ea5e9" /> Lymph Node Status
          </Text>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Sentinel Node Biopsy</Text>
            <View style={styles.optionsRow}>
              {renderOptionButton('Negative', 'negative', sentinelNode, setSentinelNode)}
              {renderOptionButton('Positive', 'positive', sentinelNode, setSentinelNode)}
              {renderOptionButton('Not Done', 'not_done', sentinelNode, setSentinelNode)}
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Distant Metastasis</Text>
            <View style={styles.toggleRow}>
              {renderToggleButton('Present', distantMetastasis, setDistantMetastasis)}
            </View>
          </View>
        </View>

        {/* Patient Characteristics */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            <Ionicons name="person-outline" size={18} color="#0ea5e9" /> Patient Characteristics
          </Text>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Age (years)</Text>
            <TextInput
              style={styles.textInput}
              value={age}
              onChangeText={setAge}
              placeholder="Optional"
              placeholderTextColor="#6b7280"
              keyboardType="number-pad"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Sex</Text>
            <View style={styles.optionsRow}>
              {renderOptionButton('Male', 'male', sex, setSex)}
              {renderOptionButton('Female', 'female', sex, setSex)}
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Tumor Location</Text>
            <View style={styles.optionsRow}>
              {renderOptionButton('Extremity', 'extremity', tumorLocation, setTumorLocation)}
              {renderOptionButton('Trunk', 'trunk', tumorLocation, setTumorLocation)}
              {renderOptionButton('Head/Neck', 'head_neck', tumorLocation, setTumorLocation)}
              {renderOptionButton('Acral', 'acral', tumorLocation, setTumorLocation)}
            </View>
          </View>
        </View>

        {/* Advanced Options */}
        <TouchableOpacity
          style={styles.advancedToggle}
          onPress={() => setShowAdvanced(!showAdvanced)}
        >
          <Text style={styles.advancedToggleText}>
            {showAdvanced ? 'Hide' : 'Show'} Advanced Pathology Features
          </Text>
          <Ionicons
            name={showAdvanced ? 'chevron-up' : 'chevron-down'}
            size={20}
            color="#0ea5e9"
          />
        </TouchableOpacity>

        {showAdvanced && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>
              <Ionicons name="flask-outline" size={18} color="#0ea5e9" /> Pathology Features
            </Text>

            <View style={styles.toggleGrid}>
              {renderToggleButton('LVI', lvi, setLvi)}
              {renderToggleButton('Regression', regression, setRegression)}
              {renderToggleButton('Microsatellites', microsatellites, setMicrosatellites)}
            </View>

            <View style={styles.inputGroup}>
              <Text style={styles.label}>Tumor-Infiltrating Lymphocytes (TILs)</Text>
              <View style={styles.optionsRow}>
                {renderOptionButton('Brisk', 'brisk', tils, setTils)}
                {renderOptionButton('Non-Brisk', 'non_brisk', tils, setTils)}
                {renderOptionButton('Absent', 'absent', tils, setTils)}
              </View>
            </View>
          </View>
        )}

        {/* Calculate Button */}
        <TouchableOpacity
          style={[styles.calculateButton, isLoading && styles.calculateButtonDisabled]}
          onPress={calculateSurvival}
          disabled={isLoading}
        >
          {isLoading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <>
              <Ionicons name="calculator-outline" size={20} color="#fff" />
              <Text style={styles.calculateButtonText}>Calculate Survival Estimate</Text>
            </>
          )}
        </TouchableOpacity>

        {error && (
          <View style={styles.errorContainer}>
            <Ionicons name="alert-circle" size={20} color="#ef4444" />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {/* Results */}
        {result && (
          <View style={styles.resultsContainer}>
            {/* Stage & Risk Summary */}
            <View style={styles.summaryCard}>
              <View style={styles.stageSection}>
                <Text style={styles.stageLabel}>Stage</Text>
                <Text style={styles.stageValue}>{result.staging.stage}</Text>
              </View>
              <View style={[styles.riskSection, { backgroundColor: result.survival_estimate.risk_color + '20' }]}>
                <Text style={styles.riskLabel}>Risk Category</Text>
                <Text style={[styles.riskValue, { color: result.survival_estimate.risk_color }]}>
                  {result.survival_estimate.risk_category}
                </Text>
              </View>
            </View>

            {/* Survival Stats */}
            <View style={styles.statsRow}>
              <View style={styles.statCard}>
                <Text style={styles.statLabel}>5-Year Survival</Text>
                <Text style={[styles.statValue, { color: result.survival_estimate.risk_color }]}>
                  {result.survival_estimate.five_year_survival}
                </Text>
              </View>
              <View style={styles.statCard}>
                <Text style={styles.statLabel}>10-Year Survival</Text>
                <Text style={[styles.statValue, { color: result.survival_estimate.risk_color }]}>
                  {result.survival_estimate.ten_year_survival}
                </Text>
              </View>
            </View>

            {/* Survival Curve */}
            {renderSurvivalCurve()}

            {/* Hazard Analysis */}
            <View style={styles.hazardSection}>
              <Text style={styles.hazardTitle}>Hazard Analysis</Text>
              <View style={styles.combinedHR}>
                <Text style={styles.combinedHRLabel}>Combined Hazard Ratio</Text>
                <Text style={[
                  styles.combinedHRValue,
                  { color: result.hazard_analysis.combined_hazard_ratio > 1 ? '#ef4444' : '#10b981' }
                ]}>
                  {result.hazard_analysis.combined_hazard_ratio.toFixed(2)}
                </Text>
              </View>
              <Text style={styles.hazardInterpretation}>
                {result.comparison.patient_vs_baseline === 'Worse'
                  ? `Your prognosis is ${Math.abs(result.comparison.difference_from_baseline_5yr)}% below the stage baseline`
                  : result.comparison.patient_vs_baseline === 'Better'
                  ? `Your prognosis is ${Math.abs(result.comparison.difference_from_baseline_5yr)}% above the stage baseline`
                  : 'Your prognosis matches the stage baseline'}
              </Text>

              {result.hazard_analysis.applied_factors.length > 0 && (
                <View style={styles.factorsList}>
                  <Text style={styles.factorsTitle}>Applied Prognostic Factors</Text>
                  {result.hazard_analysis.applied_factors.map(renderFactorCard)}
                </View>
              )}
            </View>

            {/* Disclaimer */}
            <View style={styles.disclaimerCard}>
              <Ionicons name="information-circle" size={20} color="#f59e0b" />
              <Text style={styles.disclaimerText}>{result.disclaimer}</Text>
            </View>

            {/* Data Sources */}
            <View style={styles.sourcesSection}>
              <Text style={styles.sourcesTitle}>Data Sources</Text>
              {result.data_sources.map((source, i) => (
                <Text key={i} style={styles.sourceItem}>• {source}</Text>
              ))}
            </View>
          </View>
        )}

        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  header: {
    paddingTop: 50,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  backButton: {
    marginBottom: 15,
  },
  headerContent: {},
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#94a3b8',
    marginTop: 4,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  section: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 16,
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    color: '#94a3b8',
    marginBottom: 8,
  },
  textInput: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 12,
    padding: 14,
    color: '#fff',
    fontSize: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  optionsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  optionButton: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  optionButtonSelected: {
    backgroundColor: '#0ea5e9',
    borderColor: '#0ea5e9',
  },
  optionButtonText: {
    color: '#94a3b8',
    fontSize: 14,
  },
  optionButtonTextSelected: {
    color: '#fff',
    fontWeight: '600',
  },
  toggleRow: {
    flexDirection: 'row',
    gap: 8,
  },
  toggleButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  toggleButtonActive: {
    backgroundColor: '#10b981',
    borderColor: '#10b981',
  },
  toggleButtonText: {
    color: '#94a3b8',
    fontSize: 14,
  },
  toggleButtonTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  toggleGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 16,
  },
  advancedToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    marginBottom: 16,
  },
  advancedToggleText: {
    color: '#0ea5e9',
    fontSize: 14,
    marginRight: 4,
  },
  calculateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0ea5e9',
    padding: 16,
    borderRadius: 16,
    gap: 8,
    marginBottom: 16,
  },
  calculateButtonDisabled: {
    opacity: 0.6,
  },
  calculateButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  errorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(239,68,68,0.1)',
    padding: 12,
    borderRadius: 12,
    gap: 8,
    marginBottom: 16,
  },
  errorText: {
    color: '#ef4444',
    flex: 1,
  },
  resultsContainer: {
    marginTop: 8,
  },
  summaryCard: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  stageSection: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
  },
  stageLabel: {
    color: '#94a3b8',
    fontSize: 14,
    marginBottom: 4,
  },
  stageValue: {
    color: '#fff',
    fontSize: 32,
    fontWeight: 'bold',
  },
  riskSection: {
    flex: 1,
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
  },
  riskLabel: {
    color: '#94a3b8',
    fontSize: 14,
    marginBottom: 4,
  },
  riskValue: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  statCard: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
  },
  statLabel: {
    color: '#94a3b8',
    fontSize: 13,
    marginBottom: 4,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  chartContainer: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  chartTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 16,
    textAlign: 'center',
  },
  chart: {
    flexDirection: 'row',
  },
  yAxis: {
    width: 40,
    justifyContent: 'space-between',
    paddingRight: 8,
  },
  chartArea: {
    position: 'relative',
    backgroundColor: 'rgba(0,0,0,0.2)',
    borderRadius: 8,
  },
  gridLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  curvePoint: {
    position: 'absolute',
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  baselinePoint: {
    backgroundColor: '#6b7280',
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  patientPoint: {
    backgroundColor: '#0ea5e9',
    borderWidth: 2,
    borderColor: '#fff',
  },
  xAxis: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingLeft: 40,
    paddingTop: 8,
  },
  axisLabel: {
    color: '#64748b',
    fontSize: 11,
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 24,
    marginTop: 16,
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
    color: '#94a3b8',
    fontSize: 12,
  },
  hazardSection: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  hazardTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  combinedHR: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.2)',
    padding: 12,
    borderRadius: 12,
    marginBottom: 8,
  },
  combinedHRLabel: {
    color: '#94a3b8',
    fontSize: 14,
  },
  combinedHRValue: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  hazardInterpretation: {
    color: '#94a3b8',
    fontSize: 13,
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 16,
  },
  factorsList: {
    marginTop: 8,
  },
  factorsTitle: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 12,
  },
  factorCard: {
    backgroundColor: 'rgba(0,0,0,0.2)',
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
  },
  factorHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  factorName: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },
  effectBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  effectText: {
    fontSize: 11,
    fontWeight: '600',
  },
  factorDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  factorStatus: {
    color: '#94a3b8',
    fontSize: 13,
  },
  factorHR: {
    color: '#64748b',
    fontSize: 13,
  },
  disclaimerCard: {
    flexDirection: 'row',
    backgroundColor: 'rgba(245,158,11,0.1)',
    borderRadius: 12,
    padding: 12,
    gap: 10,
    marginBottom: 16,
  },
  disclaimerText: {
    color: '#f59e0b',
    fontSize: 12,
    flex: 1,
    lineHeight: 18,
  },
  sourcesSection: {
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderRadius: 12,
    padding: 12,
  },
  sourcesTitle: {
    color: '#64748b',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 8,
  },
  sourceItem: {
    color: '#475569',
    fontSize: 11,
    marginBottom: 4,
  },
});
