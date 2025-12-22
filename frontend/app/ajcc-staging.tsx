/**
 * AJCC Melanoma Staging Calculator
 *
 * Interactive TNM staging for cutaneous melanoma using AJCC 8th Edition criteria.
 * Features automatic stage grouping, prognosis information, and treatment implications.
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
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import * as SecureStore from 'expo-secure-store';
import { API_BASE_URL } from '../config';

interface TNMCategory {
  category: string;
  description: string;
  details: string;
  thickness?: string;
  ulceration?: string;
  nodes?: string;
  satellite_transit?: string;
  site?: string;
  ldh?: string;
}

interface StagingResult {
  stage: string;
  substage: string;
  full_stage: string;
  description: string;
  prognosis: string;
  five_year_survival: string;
  treatment_implications: string[];
}

interface CalculationResult {
  tnm_classification: {
    t: TNMCategory;
    n: TNMCategory;
    m: TNMCategory;
  };
  staging: StagingResult;
  clinical_parameters: {
    breslow_thickness_mm: number | null;
    ulceration: boolean | null;
    mitotic_rate_per_mm2: number | null;
    lymph_nodes_examined: number | null;
    lymph_nodes_positive: number | null;
  };
  suggested_t_category?: {
    category: string;
    note: string;
  };
  ajcc_version: string;
  disclaimer: string;
}

interface StageInfo {
  name: string;
  tnm: string;
  description: string;
  five_year_survival: string;
  ten_year_survival: string;
  characteristics: string[];
  treatment: string[];
  follow_up: string;
}

// T Category options
const T_OPTIONS = [
  { value: 'Tis', label: 'Tis - In situ', short: 'In situ' },
  { value: 'T1a', label: 'T1a - <0.8mm, no ulceration', short: '<0.8mm' },
  { value: 'T1b', label: 'T1b - <0.8mm ulcerated or 0.8-1.0mm', short: '0.8-1.0mm' },
  { value: 'T2a', label: 'T2a - 1.0-2.0mm, no ulceration', short: '1-2mm' },
  { value: 'T2b', label: 'T2b - 1.0-2.0mm, ulcerated', short: '1-2mm ulc' },
  { value: 'T3a', label: 'T3a - 2.0-4.0mm, no ulceration', short: '2-4mm' },
  { value: 'T3b', label: 'T3b - 2.0-4.0mm, ulcerated', short: '2-4mm ulc' },
  { value: 'T4a', label: 'T4a - >4.0mm, no ulceration', short: '>4mm' },
  { value: 'T4b', label: 'T4b - >4.0mm, ulcerated', short: '>4mm ulc' },
];

// N Category options
const N_OPTIONS = [
  { value: 'N0', label: 'N0 - No nodal metastasis', short: 'None' },
  { value: 'N1a', label: 'N1a - 1 occult node', short: '1 occult' },
  { value: 'N1b', label: 'N1b - 1 clinical node', short: '1 clinical' },
  { value: 'N1c', label: 'N1c - Satellite/in-transit only', short: 'Satellite' },
  { value: 'N2a', label: 'N2a - 2-3 occult nodes', short: '2-3 occult' },
  { value: 'N2b', label: 'N2b - 2-3 clinical nodes', short: '2-3 clinical' },
  { value: 'N2c', label: 'N2c - 1 node + satellite', short: '1 + satellite' },
  { value: 'N3a', label: 'N3a - 4+ occult nodes', short: '4+ occult' },
  { value: 'N3b', label: 'N3b - 4+ clinical/matted', short: '4+ matted' },
  { value: 'N3c', label: 'N3c - 2+ nodes + satellite', short: '2+ satellite' },
];

// M Category options
const M_OPTIONS = [
  { value: 'M0', label: 'M0 - No distant metastasis', short: 'None' },
  { value: 'M1a', label: 'M1a - Skin/soft tissue/nodes', short: 'Skin/nodes' },
  { value: 'M1a(1)', label: 'M1a(1) - Above + elevated LDH', short: 'Skin + LDH' },
  { value: 'M1b', label: 'M1b - Lung metastasis', short: 'Lung' },
  { value: 'M1b(1)', label: 'M1b(1) - Lung + elevated LDH', short: 'Lung + LDH' },
  { value: 'M1c', label: 'M1c - Non-CNS visceral', short: 'Visceral' },
  { value: 'M1c(1)', label: 'M1c(1) - Visceral + elevated LDH', short: 'Visc + LDH' },
  { value: 'M1d', label: 'M1d - CNS/Brain metastasis', short: 'Brain' },
  { value: 'M1d(1)', label: 'M1d(1) - CNS + elevated LDH', short: 'Brain + LDH' },
];

export default function AJCCStagingScreen() {
  const { isAuthenticated } = useAuth();
  const router = useRouter();

  // TNM selections
  const [tCategory, setTCategory] = useState('T1a');
  const [nCategory, setNCategory] = useState('N0');
  const [mCategory, setMCategory] = useState('M0');

  // Optional clinical parameters
  const [breslowThickness, setBreslowThickness] = useState('');
  const [hasUlceration, setHasUlceration] = useState(false);
  const [mitoticRate, setMitoticRate] = useState('');
  const [nodesExamined, setNodesExamined] = useState('');
  const [nodesPositive, setNodesPositive] = useState('');

  // Results
  const [result, setResult] = useState<CalculationResult | null>(null);
  const [stageInfo, setStageInfo] = useState<StageInfo | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

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

  const calculateStage = async () => {
    setIsCalculating(true);
    try {
      const headers = await getAuthHeaders();

      const formData = new FormData();
      formData.append('t_category', tCategory);
      formData.append('n_category', nCategory);
      formData.append('m_category', mCategory);

      if (breslowThickness) {
        formData.append('breslow_thickness', breslowThickness);
      }
      formData.append('ulceration', hasUlceration.toString());
      if (mitoticRate) {
        formData.append('mitotic_rate', mitoticRate);
      }
      if (nodesExamined) {
        formData.append('lymph_nodes_examined', nodesExamined);
      }
      if (nodesPositive) {
        formData.append('lymph_nodes_positive', nodesPositive);
      }

      const response = await fetch(`${API_BASE_URL}/ajcc-staging/calculate`, {
        method: 'POST',
        headers: {
          ...headers,
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setResult(data);

        // Fetch detailed stage info
        if (data.staging?.full_stage) {
          fetchStageInfo(data.staging.full_stage);
        }
      } else {
        const error = await response.text();
        Alert.alert('Error', error || 'Failed to calculate stage');
      }
    } catch (error) {
      console.error('Error calculating stage:', error);
      Alert.alert('Error', 'Failed to calculate stage');
    } finally {
      setIsCalculating(false);
    }
  };

  const fetchStageInfo = async (stage: string) => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/ajcc-staging/stage-info/${stage}`, {
        headers,
      });

      if (response.ok) {
        const data = await response.json();
        setStageInfo(data);
      }
    } catch (error) {
      console.error('Error fetching stage info:', error);
    }
  };

  const getStageColor = (stage: string): string => {
    if (stage === '0' || stage === 'IA') return '#10b981';
    if (stage === 'IB' || stage === 'IIA') return '#22c55e';
    if (stage === 'IIB' || stage === 'IIC') return '#f59e0b';
    if (stage.startsWith('III')) return '#f97316';
    if (stage === 'IV') return '#ef4444';
    return '#6b7280';
  };

  const renderCategorySelector = (
    title: string,
    options: typeof T_OPTIONS,
    selected: string,
    onSelect: (value: string) => void,
    color: string
  ) => (
    <View style={styles.categorySection}>
      <Text style={[styles.categoryTitle, { color }]}>{title}</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.optionsScroll}>
        <View style={styles.optionsRow}>
          {options.map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[
                styles.optionButton,
                selected === option.value && [styles.optionButtonSelected, { borderColor: color, backgroundColor: `${color}15` }]
              ]}
              onPress={() => onSelect(option.value)}
            >
              <Text style={[
                styles.optionValue,
                selected === option.value && { color }
              ]}>
                {option.value}
              </Text>
              <Text style={[
                styles.optionLabel,
                selected === option.value && { color }
              ]}>
                {option.short}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>
    </View>
  );

  return (
    <LinearGradient colors={['#fef3c7', '#fde68a', '#fcd34d']} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#92400e" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>AJCC Staging</Text>
        <View style={styles.headerBadge}>
          <Text style={styles.headerBadgeText}>8th Ed</Text>
        </View>
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* TNM Selection */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="analytics" size={24} color="#92400e" />
            <Text style={styles.cardTitle}>TNM Classification</Text>
          </View>

          {renderCategorySelector('T - Primary Tumor', T_OPTIONS, tCategory, setTCategory, '#dc2626')}
          {renderCategorySelector('N - Regional Nodes', N_OPTIONS, nCategory, setNCategory, '#2563eb')}
          {renderCategorySelector('M - Distant Metastasis', M_OPTIONS, mCategory, setMCategory, '#7c3aed')}
        </View>

        {/* Advanced Parameters */}
        <TouchableOpacity
          style={styles.advancedToggle}
          onPress={() => setShowAdvanced(!showAdvanced)}
        >
          <Ionicons name={showAdvanced ? 'chevron-up' : 'chevron-down'} size={20} color="#92400e" />
          <Text style={styles.advancedToggleText}>
            {showAdvanced ? 'Hide' : 'Show'} Clinical Parameters
          </Text>
        </TouchableOpacity>

        {showAdvanced && (
          <View style={styles.card}>
            <View style={styles.inputRow}>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Breslow Thickness (mm)</Text>
                <TextInput
                  style={styles.input}
                  value={breslowThickness}
                  onChangeText={setBreslowThickness}
                  keyboardType="decimal-pad"
                  placeholder="e.g., 1.5"
                  placeholderTextColor="#9ca3af"
                />
              </View>
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
              <Text style={styles.inputLabel}>Ulceration Present</Text>
              <Switch
                value={hasUlceration}
                onValueChange={setHasUlceration}
                trackColor={{ false: '#d1d5db', true: '#fbbf24' }}
                thumbColor={hasUlceration ? '#92400e' : '#f4f3f4'}
              />
            </View>

            <View style={styles.inputRow}>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Nodes Examined</Text>
                <TextInput
                  style={styles.input}
                  value={nodesExamined}
                  onChangeText={setNodesExamined}
                  keyboardType="number-pad"
                  placeholder="e.g., 3"
                  placeholderTextColor="#9ca3af"
                />
              </View>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Nodes Positive</Text>
                <TextInput
                  style={styles.input}
                  value={nodesPositive}
                  onChangeText={setNodesPositive}
                  keyboardType="number-pad"
                  placeholder="e.g., 1"
                  placeholderTextColor="#9ca3af"
                />
              </View>
            </View>
          </View>
        )}

        {/* Calculate Button */}
        <TouchableOpacity
          style={styles.calculateButton}
          onPress={calculateStage}
          disabled={isCalculating}
        >
          {isCalculating ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <>
              <Ionicons name="calculator" size={24} color="#fff" />
              <Text style={styles.calculateButtonText}>Calculate Stage</Text>
            </>
          )}
        </TouchableOpacity>

        {/* Results */}
        {result && (
          <>
            {/* Stage Result */}
            <View style={[styles.resultCard, { borderColor: getStageColor(result.staging.full_stage) }]}>
              <View style={styles.stageHeader}>
                <Text style={styles.stageLabel}>AJCC Stage</Text>
                <View style={[styles.stageBadge, { backgroundColor: getStageColor(result.staging.full_stage) }]}>
                  <Text style={styles.stageValue}>{result.staging.full_stage}</Text>
                </View>
              </View>

              <Text style={styles.stageDescription}>{result.staging.description}</Text>

              <View style={styles.survivalBox}>
                <Ionicons name="stats-chart" size={20} color="#059669" />
                <Text style={styles.survivalText}>
                  5-Year Survival: {result.staging.five_year_survival}
                </Text>
              </View>

              <Text style={styles.prognosisText}>{result.staging.prognosis}</Text>
            </View>

            {/* TNM Details */}
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>TNM Classification Details</Text>

              <View style={styles.tnmDetail}>
                <View style={[styles.tnmBadge, { backgroundColor: '#fee2e2' }]}>
                  <Text style={[styles.tnmBadgeText, { color: '#dc2626' }]}>
                    {result.tnm_classification.t.category}
                  </Text>
                </View>
                <Text style={styles.tnmDescription}>
                  {result.tnm_classification.t.description}
                </Text>
              </View>

              <View style={styles.tnmDetail}>
                <View style={[styles.tnmBadge, { backgroundColor: '#dbeafe' }]}>
                  <Text style={[styles.tnmBadgeText, { color: '#2563eb' }]}>
                    {result.tnm_classification.n.category}
                  </Text>
                </View>
                <Text style={styles.tnmDescription}>
                  {result.tnm_classification.n.description}
                </Text>
              </View>

              <View style={styles.tnmDetail}>
                <View style={[styles.tnmBadge, { backgroundColor: '#ede9fe' }]}>
                  <Text style={[styles.tnmBadgeText, { color: '#7c3aed' }]}>
                    {result.tnm_classification.m.category}
                  </Text>
                </View>
                <Text style={styles.tnmDescription}>
                  {result.tnm_classification.m.description}
                </Text>
              </View>

              {result.suggested_t_category && (
                <View style={styles.suggestionBox}>
                  <Ionicons name="bulb" size={18} color="#f59e0b" />
                  <Text style={styles.suggestionText}>
                    Based on thickness: {result.suggested_t_category.category} - {result.suggested_t_category.note}
                  </Text>
                </View>
              )}
            </View>

            {/* Treatment Implications */}
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Treatment Implications</Text>
              {result.staging.treatment_implications.map((implication, index) => (
                <View key={index} style={styles.implicationRow}>
                  <Ionicons name="checkmark-circle" size={18} color="#10b981" />
                  <Text style={styles.implicationText}>{implication}</Text>
                </View>
              ))}
            </View>

            {/* Stage Details */}
            {stageInfo && (
              <View style={styles.card}>
                <Text style={styles.sectionTitle}>{stageInfo.name}</Text>

                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>TNM:</Text>
                  <Text style={styles.infoValue}>{stageInfo.tnm}</Text>
                </View>

                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>5-Year Survival:</Text>
                  <Text style={styles.infoValue}>{stageInfo.five_year_survival}</Text>
                </View>

                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>10-Year Survival:</Text>
                  <Text style={styles.infoValue}>{stageInfo.ten_year_survival}</Text>
                </View>

                <Text style={styles.subsectionTitle}>Characteristics</Text>
                {stageInfo.characteristics.map((char, index) => (
                  <View key={index} style={styles.bulletRow}>
                    <Text style={styles.bullet}>•</Text>
                    <Text style={styles.bulletText}>{char}</Text>
                  </View>
                ))}

                <Text style={styles.subsectionTitle}>Treatment Approach</Text>
                {stageInfo.treatment.map((tx, index) => (
                  <View key={index} style={styles.bulletRow}>
                    <Text style={styles.bullet}>•</Text>
                    <Text style={styles.bulletText}>{tx}</Text>
                  </View>
                ))}

                <Text style={styles.subsectionTitle}>Follow-up</Text>
                <Text style={styles.followUpText}>{stageInfo.follow_up}</Text>
              </View>
            )}

            {/* Disclaimer */}
            <View style={styles.disclaimerBox}>
              <Ionicons name="warning" size={18} color="#92400e" />
              <Text style={styles.disclaimerText}>{result.disclaimer}</Text>
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
    borderBottomColor: '#fcd34d',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    flex: 1,
    fontSize: 20,
    fontWeight: '700',
    color: '#92400e',
    marginLeft: 12,
  },
  headerBadge: {
    backgroundColor: '#92400e',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  headerBadgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
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
  categorySection: {
    marginBottom: 16,
  },
  categoryTitle: {
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 10,
  },
  optionsScroll: {
    marginHorizontal: -16,
    paddingHorizontal: 16,
  },
  optionsRow: {
    flexDirection: 'row',
    gap: 8,
  },
  optionButton: {
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#e5e7eb',
    backgroundColor: '#f9fafb',
    minWidth: 70,
    alignItems: 'center',
  },
  optionButtonSelected: {
    borderWidth: 2,
  },
  optionValue: {
    fontSize: 14,
    fontWeight: '700',
    color: '#374151',
  },
  optionLabel: {
    fontSize: 10,
    color: '#6b7280',
    marginTop: 2,
  },
  advancedToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    gap: 6,
  },
  advancedToggleText: {
    color: '#92400e',
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
  inputLabel: {
    fontSize: 13,
    color: '#4b5563',
    marginBottom: 6,
    fontWeight: '500',
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
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 8,
    marginBottom: 12,
  },
  calculateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#92400e',
    paddingVertical: 16,
    borderRadius: 14,
    marginBottom: 20,
    gap: 10,
    shadowColor: '#92400e',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 5,
  },
  calculateButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
  },
  resultCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    borderWidth: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 6,
  },
  stageHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  stageLabel: {
    fontSize: 14,
    color: '#6b7280',
    fontWeight: '500',
  },
  stageBadge: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 12,
  },
  stageValue: {
    fontSize: 28,
    fontWeight: '800',
    color: '#fff',
  },
  stageDescription: {
    fontSize: 16,
    color: '#1f2937',
    fontWeight: '600',
    marginBottom: 12,
  },
  survivalBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0fdf4',
    padding: 12,
    borderRadius: 10,
    marginBottom: 12,
    gap: 8,
  },
  survivalText: {
    fontSize: 16,
    color: '#059669',
    fontWeight: '700',
  },
  prognosisText: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 14,
  },
  tnmDetail: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    gap: 12,
  },
  tnmBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    minWidth: 50,
    alignItems: 'center',
  },
  tnmBadgeText: {
    fontSize: 14,
    fontWeight: '700',
  },
  tnmDescription: {
    flex: 1,
    fontSize: 13,
    color: '#4b5563',
  },
  suggestionBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fffbeb',
    padding: 12,
    borderRadius: 10,
    marginTop: 12,
    gap: 8,
  },
  suggestionText: {
    flex: 1,
    fontSize: 13,
    color: '#92400e',
  },
  implicationRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 10,
    gap: 10,
  },
  implicationText: {
    flex: 1,
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  infoLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  infoValue: {
    fontSize: 14,
    color: '#1f2937',
    fontWeight: '600',
  },
  subsectionTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#4b5563',
    marginTop: 16,
    marginBottom: 10,
  },
  bulletRow: {
    flexDirection: 'row',
    marginBottom: 6,
    paddingLeft: 4,
  },
  bullet: {
    fontSize: 14,
    color: '#92400e',
    marginRight: 8,
  },
  bulletText: {
    flex: 1,
    fontSize: 13,
    color: '#4b5563',
    lineHeight: 18,
  },
  followUpText: {
    fontSize: 13,
    color: '#4b5563',
    lineHeight: 18,
    fontStyle: 'italic',
  },
  disclaimerBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#fffbeb',
    padding: 14,
    borderRadius: 12,
    marginBottom: 16,
    gap: 10,
    borderWidth: 1,
    borderColor: '#fcd34d',
  },
  disclaimerText: {
    flex: 1,
    fontSize: 12,
    color: '#92400e',
    lineHeight: 18,
  },
  bottomSpacer: {
    height: 40,
  },
});
