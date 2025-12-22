/**
 * SCORAD/PASI Scoring Screen
 *
 * Features:
 * - SCORAD calculator for eczema/atopic dermatitis severity
 * - PASI calculator for psoriasis severity
 * - Visual body area selection
 * - Score interpretation and recommendations
 * - Score history tracking
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
  Modal,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Intensity scale labels for SCORAD
const INTENSITY_LABELS = ['None', 'Mild', 'Moderate', 'Severe'];

// Severity scale labels for PASI (0-4)
const PASI_SEVERITY_LABELS = ['Clear', 'Almost Clear', 'Mild', 'Moderate', 'Severe'];

// Area involvement labels for PASI
const AREA_LABELS = [
  { value: 0, label: 'None (0%)' },
  { value: 1, label: '<10%' },
  { value: 2, label: '10-29%' },
  { value: 3, label: '30-49%' },
  { value: 4, label: '50-69%' },
  { value: 5, label: '70-89%' },
  { value: 6, label: '90-100%' },
];

interface ScoradResult {
  scorad_score: number;
  severity: string;
  recommendation: string;
  components: {
    extent_score: number;
    intensity_score: number;
    subjective_score: number;
  };
}

interface PasiResult {
  pasi_score: number;
  severity: string;
  recommendation: string;
  regional_scores: {
    head: number;
    upper_extremities: number;
    trunk: number;
    lower_extremities: number;
  };
}

interface ScoreHistory {
  type: 'scorad' | 'pasi';
  score: number;
  severity: string;
  date: string;
}

export default function ScoringScreen() {
  const { t } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // State
  const [activeTab, setActiveTab] = useState<'scorad' | 'pasi'>('scorad');
  const [isLoading, setIsLoading] = useState(false);

  // SCORAD state
  const [scoradExtent, setScoradExtent] = useState(0);
  const [scoradErythema, setScoradErythema] = useState(0);
  const [scoradEdema, setScoradEdema] = useState(0);
  const [scoradOozing, setScoradOozing] = useState(0);
  const [scoradExcoriation, setScoradExcoriation] = useState(0);
  const [scoradLichenification, setScoradLichenification] = useState(0);
  const [scoradDryness, setScoradDryness] = useState(0);
  const [scoradItch, setScoradItch] = useState(0);
  const [scoradSleepLoss, setScoradSleepLoss] = useState(0);
  const [scoradResult, setScoradResult] = useState<ScoradResult | null>(null);

  // PASI state - Head
  const [headInvolvement, setHeadInvolvement] = useState(0);
  const [headErythema, setHeadErythema] = useState(0);
  const [headThickness, setHeadThickness] = useState(0);
  const [headScaling, setHeadScaling] = useState(0);
  // Upper extremities
  const [upperInvolvement, setUpperInvolvement] = useState(0);
  const [upperErythema, setUpperErythema] = useState(0);
  const [upperThickness, setUpperThickness] = useState(0);
  const [upperScaling, setUpperScaling] = useState(0);
  // Trunk
  const [trunkInvolvement, setTrunkInvolvement] = useState(0);
  const [trunkErythema, setTrunkErythema] = useState(0);
  const [trunkThickness, setTrunkThickness] = useState(0);
  const [trunkScaling, setTrunkScaling] = useState(0);
  // Lower extremities
  const [lowerInvolvement, setLowerInvolvement] = useState(0);
  const [lowerErythema, setLowerErythema] = useState(0);
  const [lowerThickness, setLowerThickness] = useState(0);
  const [lowerScaling, setLowerScaling] = useState(0);
  const [pasiResult, setPasiResult] = useState<PasiResult | null>(null);

  // History state
  const [scoreHistory, setScoreHistory] = useState<ScoreHistory[]>([]);
  const [resultModalVisible, setResultModalVisible] = useState(false);

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
      return;
    }
    loadScoreHistory();
  }, [isAuthenticated]);

  const loadScoreHistory = async () => {
    try {
      const historyStr = await AsyncStorage.getItem('scoreHistory');
      if (historyStr) {
        setScoreHistory(JSON.parse(historyStr));
      }
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  const saveScoreToHistory = async (type: 'scorad' | 'pasi', score: number, severity: string) => {
    try {
      const newEntry: ScoreHistory = {
        type,
        score,
        severity,
        date: new Date().toISOString(),
      };
      const newHistory = [newEntry, ...scoreHistory].slice(0, 20); // Keep last 20 entries
      setScoreHistory(newHistory);
      await AsyncStorage.setItem('scoreHistory', JSON.stringify(newHistory));
    } catch (error) {
      console.error('Error saving history:', error);
    }
  };

  const calculateScorad = async () => {
    try {
      setIsLoading(true);
      const token = await AsyncStorage.getItem('accessToken');
      const formData = new FormData();

      formData.append('extent_percentage', scoradExtent.toString());
      formData.append('intensity_erythema', scoradErythema.toString());
      formData.append('intensity_edema', scoradEdema.toString());
      formData.append('intensity_oozing', scoradOozing.toString());
      formData.append('intensity_excoriation', scoradExcoriation.toString());
      formData.append('intensity_lichenification', scoradLichenification.toString());
      formData.append('intensity_dryness', scoradDryness.toString());
      formData.append('subjective_itch', scoradItch.toString());
      formData.append('subjective_sleep_loss', scoradSleepLoss.toString());

      const response = await fetch(`${API_BASE_URL}/calculate-scorad`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setScoradResult(data);
        saveScoreToHistory('scorad', data.scorad_score, data.severity);
        setResultModalVisible(true);
      } else {
        throw new Error('Calculation failed');
      }
    } catch (error) {
      console.error('Error calculating SCORAD:', error);
      Alert.alert('Error', 'Failed to calculate SCORAD score');
    } finally {
      setIsLoading(false);
    }
  };

  const calculatePasi = async () => {
    try {
      setIsLoading(true);
      const token = await AsyncStorage.getItem('accessToken');
      const formData = new FormData();

      // Head
      formData.append('head_involvement', headInvolvement.toString());
      formData.append('head_erythema', headErythema.toString());
      formData.append('head_thickness', headThickness.toString());
      formData.append('head_scaling', headScaling.toString());
      // Upper
      formData.append('upper_involvement', upperInvolvement.toString());
      formData.append('upper_erythema', upperErythema.toString());
      formData.append('upper_thickness', upperThickness.toString());
      formData.append('upper_scaling', upperScaling.toString());
      // Trunk
      formData.append('trunk_involvement', trunkInvolvement.toString());
      formData.append('trunk_erythema', trunkErythema.toString());
      formData.append('trunk_thickness', trunkThickness.toString());
      formData.append('trunk_scaling', trunkScaling.toString());
      // Lower
      formData.append('lower_involvement', lowerInvolvement.toString());
      formData.append('lower_erythema', lowerErythema.toString());
      formData.append('lower_thickness', lowerThickness.toString());
      formData.append('lower_scaling', lowerScaling.toString());

      const response = await fetch(`${API_BASE_URL}/calculate-pasi`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setPasiResult(data);
        saveScoreToHistory('pasi', data.pasi_score, data.severity);
        setResultModalVisible(true);
      } else {
        throw new Error('Calculation failed');
      }
    } catch (error) {
      console.error('Error calculating PASI:', error);
      Alert.alert('Error', 'Failed to calculate PASI score');
    } finally {
      setIsLoading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'mild': return '#10b981';
      case 'moderate': return '#f59e0b';
      case 'severe': return '#ef4444';
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

  // Slider component
  const renderSlider = (
    label: string,
    value: number,
    onChange: (val: number) => void,
    max: number,
    labels?: string[],
    description?: string
  ) => (
    <View style={styles.sliderContainer}>
      <View style={styles.sliderHeader}>
        <Text style={styles.sliderLabel}>{label}</Text>
        <View style={[styles.sliderValueBadge, { backgroundColor: value > 0 ? '#2563eb20' : '#f3f4f6' }]}>
          <Text style={[styles.sliderValue, { color: value > 0 ? '#2563eb' : '#6b7280' }]}>
            {labels ? labels[value] : value}
          </Text>
        </View>
      </View>
      {description && <Text style={styles.sliderDescription}>{description}</Text>}
      <View style={styles.sliderTrack}>
        {Array.from({ length: max + 1 }, (_, i) => (
          <TouchableOpacity
            key={i}
            style={[
              styles.sliderDot,
              i <= value && styles.sliderDotActive,
              i === value && styles.sliderDotCurrent,
            ]}
            onPress={() => onChange(i)}
          >
            {labels && i === value && (
              <View style={styles.sliderTooltip}>
                <Text style={styles.sliderTooltipText}>{labels[i]}</Text>
              </View>
            )}
          </TouchableOpacity>
        ))}
      </View>
      <View style={styles.sliderLabelsRow}>
        <Text style={styles.sliderMinLabel}>0</Text>
        <Text style={styles.sliderMaxLabel}>{max}</Text>
      </View>
    </View>
  );

  // PASI body region component
  const renderPasiRegion = (
    title: string,
    icon: string,
    involvement: number,
    setInvolvement: (val: number) => void,
    erythema: number,
    setErythema: (val: number) => void,
    thickness: number,
    setThickness: (val: number) => void,
    scaling: number,
    setScaling: (val: number) => void,
    weight: string
  ) => (
    <View style={styles.pasiRegionCard}>
      <View style={styles.pasiRegionHeader}>
        <View style={styles.pasiRegionTitle}>
          <Ionicons name={icon as any} size={24} color="#2563eb" />
          <Text style={styles.pasiRegionName}>{title}</Text>
        </View>
        <Text style={styles.pasiWeight}>Weight: {weight}</Text>
      </View>

      {/* Area Involvement */}
      <Text style={styles.pasiSubLabel}>Area Involvement</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        <View style={styles.areaOptions}>
          {AREA_LABELS.map((area) => (
            <TouchableOpacity
              key={area.value}
              style={[
                styles.areaOption,
                involvement === area.value && styles.areaOptionSelected,
              ]}
              onPress={() => setInvolvement(area.value)}
            >
              <Text style={[
                styles.areaOptionText,
                involvement === area.value && styles.areaOptionTextSelected,
              ]}>
                {area.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>

      {/* Severity scores */}
      <View style={styles.pasiSeverityRow}>
        <View style={styles.pasiSeverityItem}>
          <Text style={styles.pasiSeverityLabel}>Erythema</Text>
          <View style={styles.severityButtons}>
            {[0, 1, 2, 3, 4].map((val) => (
              <TouchableOpacity
                key={val}
                style={[
                  styles.severityButton,
                  erythema === val && styles.severityButtonSelected,
                ]}
                onPress={() => setErythema(val)}
              >
                <Text style={[
                  styles.severityButtonText,
                  erythema === val && styles.severityButtonTextSelected,
                ]}>
                  {val}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={styles.pasiSeverityItem}>
          <Text style={styles.pasiSeverityLabel}>Thickness</Text>
          <View style={styles.severityButtons}>
            {[0, 1, 2, 3, 4].map((val) => (
              <TouchableOpacity
                key={val}
                style={[
                  styles.severityButton,
                  thickness === val && styles.severityButtonSelected,
                ]}
                onPress={() => setThickness(val)}
              >
                <Text style={[
                  styles.severityButtonText,
                  thickness === val && styles.severityButtonTextSelected,
                ]}>
                  {val}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={styles.pasiSeverityItem}>
          <Text style={styles.pasiSeverityLabel}>Scaling</Text>
          <View style={styles.severityButtons}>
            {[0, 1, 2, 3, 4].map((val) => (
              <TouchableOpacity
                key={val}
                style={[
                  styles.severityButton,
                  scaling === val && styles.severityButtonSelected,
                ]}
                onPress={() => setScaling(val)}
              >
                <Text style={[
                  styles.severityButtonText,
                  scaling === val && styles.severityButtonTextSelected,
                ]}>
                  {val}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </View>
    </View>
  );

  // Render SCORAD Tab
  const renderScoradTab = () => (
    <ScrollView style={styles.tabContent} showsVerticalScrollIndicator={false}>
      {/* Info Card */}
      <View style={styles.infoCard}>
        <Ionicons name="information-circle-outline" size={24} color="#2563eb" />
        <View style={styles.infoContent}>
          <Text style={styles.infoTitle}>SCORAD Index</Text>
          <Text style={styles.infoText}>
            SCORing Atopic Dermatitis - Used to assess eczema severity based on extent, intensity, and subjective symptoms.
          </Text>
        </View>
      </View>

      {/* Extent Section */}
      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>A. Extent of Disease</Text>
        <Text style={styles.sectionSubtitle}>Percentage of body surface area affected (0-100%)</Text>

        {renderSlider(
          'Body Surface Area',
          scoradExtent,
          setScoradExtent,
          100,
          undefined,
          'Estimate the percentage of skin affected by eczema'
        )}
      </View>

      {/* Intensity Section */}
      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>B. Intensity of Lesions</Text>
        <Text style={styles.sectionSubtitle}>Rate each symptom from 0 (none) to 3 (severe)</Text>

        {renderSlider('Erythema (Redness)', scoradErythema, setScoradErythema, 3, INTENSITY_LABELS)}
        {renderSlider('Edema/Papulation (Swelling)', scoradEdema, setScoradEdema, 3, INTENSITY_LABELS)}
        {renderSlider('Oozing/Crusting', scoradOozing, setScoradOozing, 3, INTENSITY_LABELS)}
        {renderSlider('Excoriation (Scratch marks)', scoradExcoriation, setScoradExcoriation, 3, INTENSITY_LABELS)}
        {renderSlider('Lichenification (Thickening)', scoradLichenification, setScoradLichenification, 3, INTENSITY_LABELS)}
        {renderSlider('Dryness', scoradDryness, setScoradDryness, 3, INTENSITY_LABELS)}
      </View>

      {/* Subjective Symptoms Section */}
      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>C. Subjective Symptoms</Text>
        <Text style={styles.sectionSubtitle}>Rate symptoms over the last 3 days/nights (0-10)</Text>

        {renderSlider('Itching Severity', scoradItch, setScoradItch, 10, undefined, 'How severe was your itching?')}
        {renderSlider('Sleep Loss', scoradSleepLoss, setScoradSleepLoss, 10, undefined, 'How much sleep did you lose due to eczema?')}
      </View>

      {/* Calculate Button */}
      <TouchableOpacity
        style={styles.calculateButton}
        onPress={calculateScorad}
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="calculator-outline" size={24} color="#fff" />
            <Text style={styles.calculateButtonText}>Calculate SCORAD</Text>
          </>
        )}
      </TouchableOpacity>

      {/* Score History */}
      {scoreHistory.filter(s => s.type === 'scorad').length > 0 && (
        <View style={styles.historySection}>
          <Text style={styles.historyTitle}>Recent SCORAD Scores</Text>
          {scoreHistory.filter(s => s.type === 'scorad').slice(0, 5).map((entry, index) => (
            <View key={index} style={styles.historyItem}>
              <View style={styles.historyItemLeft}>
                <Text style={styles.historyScore}>{entry.score}</Text>
                <View style={[styles.historySeverityBadge, { backgroundColor: `${getSeverityColor(entry.severity)}20` }]}>
                  <Text style={[styles.historySeverityText, { color: getSeverityColor(entry.severity) }]}>
                    {entry.severity}
                  </Text>
                </View>
              </View>
              <Text style={styles.historyDate}>{formatDate(entry.date)}</Text>
            </View>
          ))}
        </View>
      )}

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Render PASI Tab
  const renderPasiTab = () => (
    <ScrollView style={styles.tabContent} showsVerticalScrollIndicator={false}>
      {/* Info Card */}
      <View style={styles.infoCard}>
        <Ionicons name="information-circle-outline" size={24} color="#8b5cf6" />
        <View style={styles.infoContent}>
          <Text style={styles.infoTitle}>PASI Score</Text>
          <Text style={styles.infoText}>
            Psoriasis Area and Severity Index - Assesses psoriasis severity by body region, considering erythema, thickness, and scaling.
          </Text>
        </View>
      </View>

      {/* Severity Scale Reference */}
      <View style={styles.scaleReference}>
        <Text style={styles.scaleTitle}>Severity Scale (0-4)</Text>
        <View style={styles.scaleItems}>
          {PASI_SEVERITY_LABELS.map((label, index) => (
            <View key={index} style={styles.scaleItem}>
              <View style={[styles.scaleNumber, index === 0 && { backgroundColor: '#e5e7eb' }]}>
                <Text style={styles.scaleNumberText}>{index}</Text>
              </View>
              <Text style={styles.scaleLabel}>{label}</Text>
            </View>
          ))}
        </View>
      </View>

      {/* Body Regions */}
      {renderPasiRegion(
        'Head & Neck',
        'happy-outline',
        headInvolvement,
        setHeadInvolvement,
        headErythema,
        setHeadErythema,
        headThickness,
        setHeadThickness,
        headScaling,
        setHeadScaling,
        '10%'
      )}

      {renderPasiRegion(
        'Upper Extremities',
        'hand-left-outline',
        upperInvolvement,
        setUpperInvolvement,
        upperErythema,
        setUpperErythema,
        upperThickness,
        setUpperThickness,
        upperScaling,
        setUpperScaling,
        '20%'
      )}

      {renderPasiRegion(
        'Trunk',
        'body-outline',
        trunkInvolvement,
        setTrunkInvolvement,
        trunkErythema,
        setTrunkErythema,
        trunkThickness,
        setTrunkThickness,
        trunkScaling,
        setTrunkScaling,
        '30%'
      )}

      {renderPasiRegion(
        'Lower Extremities',
        'footsteps-outline',
        lowerInvolvement,
        setLowerInvolvement,
        lowerErythema,
        setLowerErythema,
        lowerThickness,
        setLowerThickness,
        lowerScaling,
        setLowerScaling,
        '40%'
      )}

      {/* Calculate Button */}
      <TouchableOpacity
        style={[styles.calculateButton, { backgroundColor: '#8b5cf6' }]}
        onPress={calculatePasi}
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="calculator-outline" size={24} color="#fff" />
            <Text style={styles.calculateButtonText}>Calculate PASI</Text>
          </>
        )}
      </TouchableOpacity>

      {/* Score History */}
      {scoreHistory.filter(s => s.type === 'pasi').length > 0 && (
        <View style={styles.historySection}>
          <Text style={styles.historyTitle}>Recent PASI Scores</Text>
          {scoreHistory.filter(s => s.type === 'pasi').slice(0, 5).map((entry, index) => (
            <View key={index} style={styles.historyItem}>
              <View style={styles.historyItemLeft}>
                <Text style={styles.historyScore}>{entry.score}</Text>
                <View style={[styles.historySeverityBadge, { backgroundColor: `${getSeverityColor(entry.severity)}20` }]}>
                  <Text style={[styles.historySeverityText, { color: getSeverityColor(entry.severity) }]}>
                    {entry.severity}
                  </Text>
                </View>
              </View>
              <Text style={styles.historyDate}>{formatDate(entry.date)}</Text>
            </View>
          ))}
        </View>
      )}

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );

  // Result Modal
  const renderResultModal = () => (
    <Modal
      visible={resultModalVisible}
      transparent
      animationType="slide"
      onRequestClose={() => setResultModalVisible(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.resultModalContent}>
          <TouchableOpacity
            style={styles.closeModalButton}
            onPress={() => setResultModalVisible(false)}
          >
            <Ionicons name="close" size={24} color="#6b7280" />
          </TouchableOpacity>

          {activeTab === 'scorad' && scoradResult ? (
            <>
              <View style={styles.resultHeader}>
                <Text style={styles.resultType}>SCORAD Score</Text>
                <View style={[
                  styles.resultScoreCircle,
                  { borderColor: getSeverityColor(scoradResult.severity) }
                ]}>
                  <Text style={[styles.resultScore, { color: getSeverityColor(scoradResult.severity) }]}>
                    {scoradResult.scorad_score}
                  </Text>
                </View>
                <View style={[
                  styles.resultSeverityBadge,
                  { backgroundColor: `${getSeverityColor(scoradResult.severity)}20` }
                ]}>
                  <Text style={[styles.resultSeverityText, { color: getSeverityColor(scoradResult.severity) }]}>
                    {scoradResult.severity.charAt(0).toUpperCase() + scoradResult.severity.slice(1)} Eczema
                  </Text>
                </View>
              </View>

              {/* Score Breakdown */}
              <View style={styles.breakdownCard}>
                <Text style={styles.breakdownTitle}>Score Breakdown</Text>
                <View style={styles.breakdownRow}>
                  <Text style={styles.breakdownLabel}>Extent (A/5)</Text>
                  <Text style={styles.breakdownValue}>{scoradResult.components.extent_score}</Text>
                </View>
                <View style={styles.breakdownRow}>
                  <Text style={styles.breakdownLabel}>Intensity (7B/2)</Text>
                  <Text style={styles.breakdownValue}>{scoradResult.components.intensity_score}</Text>
                </View>
                <View style={styles.breakdownRow}>
                  <Text style={styles.breakdownLabel}>Subjective (C)</Text>
                  <Text style={styles.breakdownValue}>{scoradResult.components.subjective_score}</Text>
                </View>
              </View>

              {/* Scale Reference */}
              <View style={styles.scaleCard}>
                <View style={styles.scaleBar}>
                  <View style={[styles.scaleSegment, { backgroundColor: '#10b981', flex: 25 }]} />
                  <View style={[styles.scaleSegment, { backgroundColor: '#f59e0b', flex: 25 }]} />
                  <View style={[styles.scaleSegment, { backgroundColor: '#ef4444', flex: 53 }]} />
                </View>
                <View style={styles.scaleMarkers}>
                  <Text style={styles.scaleMarker}>0</Text>
                  <Text style={styles.scaleMarker}>25</Text>
                  <Text style={styles.scaleMarker}>50</Text>
                  <Text style={styles.scaleMarker}>103</Text>
                </View>
                <View style={styles.scaleLegend}>
                  <View style={styles.legendItem}>
                    <View style={[styles.legendDot, { backgroundColor: '#10b981' }]} />
                    <Text style={styles.legendText}>Mild (&lt;25)</Text>
                  </View>
                  <View style={styles.legendItem}>
                    <View style={[styles.legendDot, { backgroundColor: '#f59e0b' }]} />
                    <Text style={styles.legendText}>Moderate (25-50)</Text>
                  </View>
                  <View style={styles.legendItem}>
                    <View style={[styles.legendDot, { backgroundColor: '#ef4444' }]} />
                    <Text style={styles.legendText}>Severe (&gt;50)</Text>
                  </View>
                </View>
              </View>

              {/* Recommendation */}
              <View style={styles.recommendationCard}>
                <Ionicons name="medical-outline" size={24} color="#2563eb" />
                <Text style={styles.recommendationText}>{scoradResult.recommendation}</Text>
              </View>
            </>
          ) : activeTab === 'pasi' && pasiResult ? (
            <>
              <View style={styles.resultHeader}>
                <Text style={styles.resultType}>PASI Score</Text>
                <View style={[
                  styles.resultScoreCircle,
                  { borderColor: getSeverityColor(pasiResult.severity) }
                ]}>
                  <Text style={[styles.resultScore, { color: getSeverityColor(pasiResult.severity) }]}>
                    {pasiResult.pasi_score}
                  </Text>
                </View>
                <View style={[
                  styles.resultSeverityBadge,
                  { backgroundColor: `${getSeverityColor(pasiResult.severity)}20` }
                ]}>
                  <Text style={[styles.resultSeverityText, { color: getSeverityColor(pasiResult.severity) }]}>
                    {pasiResult.severity.charAt(0).toUpperCase() + pasiResult.severity.slice(1)} Psoriasis
                  </Text>
                </View>
              </View>

              {/* Regional Scores */}
              <View style={styles.breakdownCard}>
                <Text style={styles.breakdownTitle}>Regional Scores</Text>
                <View style={styles.breakdownRow}>
                  <View style={styles.breakdownLabelRow}>
                    <Ionicons name="happy-outline" size={16} color="#6b7280" />
                    <Text style={styles.breakdownLabel}>Head & Neck</Text>
                  </View>
                  <Text style={styles.breakdownValue}>{pasiResult.regional_scores.head}</Text>
                </View>
                <View style={styles.breakdownRow}>
                  <View style={styles.breakdownLabelRow}>
                    <Ionicons name="hand-left-outline" size={16} color="#6b7280" />
                    <Text style={styles.breakdownLabel}>Upper Extremities</Text>
                  </View>
                  <Text style={styles.breakdownValue}>{pasiResult.regional_scores.upper_extremities}</Text>
                </View>
                <View style={styles.breakdownRow}>
                  <View style={styles.breakdownLabelRow}>
                    <Ionicons name="body-outline" size={16} color="#6b7280" />
                    <Text style={styles.breakdownLabel}>Trunk</Text>
                  </View>
                  <Text style={styles.breakdownValue}>{pasiResult.regional_scores.trunk}</Text>
                </View>
                <View style={styles.breakdownRow}>
                  <View style={styles.breakdownLabelRow}>
                    <Ionicons name="footsteps-outline" size={16} color="#6b7280" />
                    <Text style={styles.breakdownLabel}>Lower Extremities</Text>
                  </View>
                  <Text style={styles.breakdownValue}>{pasiResult.regional_scores.lower_extremities}</Text>
                </View>
              </View>

              {/* Scale Reference */}
              <View style={styles.scaleCard}>
                <View style={styles.scaleBar}>
                  <View style={[styles.scaleSegment, { backgroundColor: '#10b981', flex: 10 }]} />
                  <View style={[styles.scaleSegment, { backgroundColor: '#f59e0b', flex: 10 }]} />
                  <View style={[styles.scaleSegment, { backgroundColor: '#ef4444', flex: 52 }]} />
                </View>
                <View style={styles.scaleMarkers}>
                  <Text style={styles.scaleMarker}>0</Text>
                  <Text style={styles.scaleMarker}>10</Text>
                  <Text style={styles.scaleMarker}>20</Text>
                  <Text style={styles.scaleMarker}>72</Text>
                </View>
                <View style={styles.scaleLegend}>
                  <View style={styles.legendItem}>
                    <View style={[styles.legendDot, { backgroundColor: '#10b981' }]} />
                    <Text style={styles.legendText}>Mild (&lt;10)</Text>
                  </View>
                  <View style={styles.legendItem}>
                    <View style={[styles.legendDot, { backgroundColor: '#f59e0b' }]} />
                    <Text style={styles.legendText}>Moderate (10-20)</Text>
                  </View>
                  <View style={styles.legendItem}>
                    <View style={[styles.legendDot, { backgroundColor: '#ef4444' }]} />
                    <Text style={styles.legendText}>Severe (&gt;20)</Text>
                  </View>
                </View>
              </View>

              {/* Recommendation */}
              <View style={styles.recommendationCard}>
                <Ionicons name="medical-outline" size={24} color="#8b5cf6" />
                <Text style={styles.recommendationText}>{pasiResult.recommendation}</Text>
              </View>
            </>
          ) : null}

          <TouchableOpacity
            style={styles.doneButton}
            onPress={() => setResultModalVisible(false)}
          >
            <Text style={styles.doneButtonText}>Done</Text>
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
        <Text style={styles.headerTitle}>Clinical Scoring</Text>
        <View style={{ width: 40 }} />
      </View>

      {/* Tabs */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'scorad' && styles.activeTab]}
          onPress={() => setActiveTab('scorad')}
        >
          <View style={[styles.tabIcon, activeTab === 'scorad' && { backgroundColor: '#2563eb20' }]}>
            <Ionicons
              name="analytics-outline"
              size={20}
              color={activeTab === 'scorad' ? '#2563eb' : '#6b7280'}
            />
          </View>
          <Text style={[styles.tabText, activeTab === 'scorad' && styles.activeTabText]}>
            SCORAD
          </Text>
          <Text style={styles.tabSubtext}>Eczema</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'pasi' && styles.activeTabPasi]}
          onPress={() => setActiveTab('pasi')}
        >
          <View style={[styles.tabIcon, activeTab === 'pasi' && { backgroundColor: '#8b5cf620' }]}>
            <Ionicons
              name="grid-outline"
              size={20}
              color={activeTab === 'pasi' ? '#8b5cf6' : '#6b7280'}
            />
          </View>
          <Text style={[styles.tabText, activeTab === 'pasi' && styles.activeTabTextPasi]}>
            PASI
          </Text>
          <Text style={styles.tabSubtext}>Psoriasis</Text>
        </TouchableOpacity>
      </View>

      {/* Tab Content */}
      {activeTab === 'scorad' && renderScoradTab()}
      {activeTab === 'pasi' && renderPasiTab()}

      {/* Result Modal */}
      {renderResultModal()}
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
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
    gap: 12,
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 14,
    borderWidth: 2,
    borderColor: '#e5e7eb',
  },
  activeTab: {
    borderColor: '#2563eb',
    backgroundColor: '#eff6ff',
  },
  activeTabPasi: {
    borderColor: '#8b5cf6',
    backgroundColor: '#f5f3ff',
  },
  tabIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f3f4f6',
    marginBottom: 6,
  },
  tabText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#6b7280',
  },
  activeTabText: {
    color: '#2563eb',
  },
  activeTabTextPasi: {
    color: '#8b5cf6',
  },
  tabSubtext: {
    fontSize: 12,
    color: '#9ca3af',
    marginTop: 2,
  },
  tabContent: {
    flex: 1,
    padding: 16,
  },
  infoCard: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    gap: 12,
  },
  infoContent: {
    flex: 1,
  },
  infoTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  infoText: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 4,
    lineHeight: 20,
  },
  sectionCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 4,
  },
  sectionSubtitle: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 16,
  },
  sliderContainer: {
    marginBottom: 20,
  },
  sliderHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  sliderLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
  },
  sliderValueBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 8,
  },
  sliderValue: {
    fontSize: 13,
    fontWeight: '700',
  },
  sliderDescription: {
    fontSize: 12,
    color: '#9ca3af',
    marginBottom: 12,
  },
  sliderTrack: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: 40,
    backgroundColor: '#f3f4f6',
    borderRadius: 20,
    paddingHorizontal: 8,
  },
  sliderDot: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#e5e7eb',
  },
  sliderDotActive: {
    backgroundColor: '#93c5fd',
  },
  sliderDotCurrent: {
    backgroundColor: '#2563eb',
    transform: [{ scale: 1.2 }],
  },
  sliderTooltip: {
    position: 'absolute',
    top: -30,
    backgroundColor: '#1e3a5f',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  sliderTooltipText: {
    fontSize: 11,
    color: '#fff',
    fontWeight: '600',
  },
  sliderLabelsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  sliderMinLabel: {
    fontSize: 12,
    color: '#9ca3af',
  },
  sliderMaxLabel: {
    fontSize: 12,
    color: '#9ca3af',
  },
  calculateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 18,
    gap: 10,
  },
  calculateButtonText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '700',
  },
  historySection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginTop: 20,
  },
  historyTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  historyItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  historyItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  historyScore: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  historySeverityBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },
  historySeverityText: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  historyDate: {
    fontSize: 12,
    color: '#9ca3af',
  },
  scaleReference: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  scaleTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  scaleItems: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  scaleItem: {
    alignItems: 'center',
  },
  scaleNumber: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#2563eb',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 4,
  },
  scaleNumberText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
  },
  scaleLabel: {
    fontSize: 10,
    color: '#6b7280',
    textAlign: 'center',
  },
  pasiRegionCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  pasiRegionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  pasiRegionTitle: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  pasiRegionName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  pasiWeight: {
    fontSize: 12,
    color: '#8b5cf6',
    fontWeight: '600',
    backgroundColor: '#f5f3ff',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },
  pasiSubLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 10,
  },
  areaOptions: {
    flexDirection: 'row',
    gap: 8,
    paddingVertical: 4,
  },
  areaOption: {
    backgroundColor: '#f3f4f6',
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 10,
  },
  areaOptionSelected: {
    backgroundColor: '#8b5cf6',
  },
  areaOptionText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
  },
  areaOptionTextSelected: {
    color: '#fff',
  },
  pasiSeverityRow: {
    flexDirection: 'row',
    marginTop: 16,
    gap: 10,
  },
  pasiSeverityItem: {
    flex: 1,
  },
  pasiSeverityLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 8,
    textAlign: 'center',
  },
  severityButtons: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 4,
  },
  severityButton: {
    width: 28,
    height: 28,
    borderRadius: 6,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  severityButtonSelected: {
    backgroundColor: '#8b5cf6',
  },
  severityButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
  },
  severityButtonTextSelected: {
    color: '#fff',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  resultModalContent: {
    backgroundColor: '#fff',
    borderRadius: 24,
    padding: 24,
    width: '100%',
    maxWidth: 400,
  },
  closeModalButton: {
    position: 'absolute',
    top: 16,
    right: 16,
    zIndex: 1,
  },
  resultHeader: {
    alignItems: 'center',
    marginBottom: 20,
  },
  resultType: {
    fontSize: 14,
    color: '#6b7280',
    fontWeight: '600',
    marginBottom: 12,
  },
  resultScoreCircle: {
    width: 100,
    height: 100,
    borderRadius: 50,
    borderWidth: 4,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  resultScore: {
    fontSize: 36,
    fontWeight: '700',
  },
  resultSeverityBadge: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  resultSeverityText: {
    fontSize: 16,
    fontWeight: '700',
  },
  breakdownCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  breakdownTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  breakdownRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  breakdownLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  breakdownLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  breakdownValue: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  scaleCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  scaleBar: {
    flexDirection: 'row',
    height: 12,
    borderRadius: 6,
    overflow: 'hidden',
  },
  scaleSegment: {
    height: '100%',
  },
  scaleMarkers: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 6,
  },
  scaleMarker: {
    fontSize: 10,
    color: '#9ca3af',
  },
  scaleLegend: {
    flexDirection: 'row',
    justifyContent: 'center',
    flexWrap: 'wrap',
    marginTop: 12,
    gap: 16,
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
    fontSize: 11,
    color: '#6b7280',
  },
  recommendationCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    gap: 12,
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    color: '#1e40af',
    lineHeight: 20,
  },
  doneButton: {
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 16,
  },
  doneButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  bottomSpacer: {
    height: 30,
  },
});
