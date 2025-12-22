import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

/**
 * ABCDE Feature Analysis Display Component
 *
 * Displays quantitative metrics for the ABCDE criteria:
 * - A: Asymmetry (shape and color asymmetry scores)
 * - B: Border (irregularity, notching, blur scores)
 * - C: Color (number of colors, variance, concerning features)
 * - D: Diameter (estimated size in mm if calibration available)
 * - E: Evolution (requires longitudinal data)
 *
 * Designed to give dermatologists the specific features driving AI decisions.
 */

interface FeatureAnalysis {
  overall_score: number;
  risk_level: 'low' | 'moderate' | 'high' | 'very_high';
  description: string;
  clinical_interpretation: string;
}

interface AsymmetryAnalysis extends FeatureAnalysis {
  horizontal_asymmetry: number;
  vertical_asymmetry: number;
  shape_asymmetry: number;
  color_asymmetry: number;
}

interface BorderAnalysis extends FeatureAnalysis {
  irregularity_index: number;
  notching_score: number;
  blur_score: number;
  radial_variance: number;
  num_border_colors: number;
}

interface ColorAnalysis extends FeatureAnalysis {
  num_colors: number;
  colors_detected: string[];
  color_variance: number;
  has_blue_white_veil: boolean;
  has_regression: boolean;
  dominant_color: string;
  color_distribution: Record<string, number>;
}

interface DiameterAnalysis extends FeatureAnalysis {
  estimated_diameter_mm: number | null;
  pixel_diameter: number;
  area_pixels: number;
  is_above_6mm: boolean | null;
  calibration_available: boolean;
}

interface EvolutionAnalysis {
  has_comparison: boolean;
  change_detected: boolean | null;
  change_description: string | null;
  risk_level: string;
  description: string;
  clinical_interpretation: string;
}

export interface ABCDEAnalysis {
  asymmetry: AsymmetryAnalysis;
  border: BorderAnalysis;
  color: ColorAnalysis;
  diameter: DiameterAnalysis;
  evolution: EvolutionAnalysis;
  total_score: number;
  risk_level: 'low' | 'moderate' | 'high' | 'very_high';
  summary: string;
  key_concerns: string[];
  recommendation: string;
  methodology_notes: string[];
}

interface ABCDEFeatureDisplayProps {
  analysis: ABCDEAnalysis;
  showMethodology?: boolean;
}

const RISK_COLORS = {
  low: { bg: '#ecfdf5', border: '#a7f3d0', text: '#047857', icon: 'checkmark-circle' },
  moderate: { bg: '#fffbeb', border: '#fcd34d', text: '#b45309', icon: 'alert-circle' },
  high: { bg: '#fef2f2', border: '#fecaca', text: '#dc2626', icon: 'warning' },
  very_high: { bg: '#fef2f2', border: '#f87171', text: '#b91c1c', icon: 'warning' },
};

const ScoreMeter: React.FC<{ score: number; label: string; maxScore?: number }> = ({
  score,
  label,
  maxScore = 1,
}) => {
  const percentage = Math.min((score / maxScore) * 100, 100);
  const color = percentage > 60 ? '#dc2626' : percentage > 40 ? '#f59e0b' : '#10b981';

  return (
    <View style={styles.meterContainer}>
      <View style={styles.meterHeader}>
        <Text style={styles.meterLabel}>{label}</Text>
        <Text style={[styles.meterValue, { color }]}>{(score * 100).toFixed(0)}%</Text>
      </View>
      <View style={styles.meterTrack}>
        <View style={[styles.meterFill, { width: `${percentage}%`, backgroundColor: color }]} />
      </View>
    </View>
  );
};

const FeatureCard: React.FC<{
  title: string;
  letter: string;
  analysis: FeatureAnalysis;
  children: React.ReactNode;
  expanded: boolean;
  onToggle: () => void;
}> = ({ title, letter, analysis, children, expanded, onToggle }) => {
  const riskStyle = RISK_COLORS[analysis.risk_level] || RISK_COLORS.moderate;

  return (
    <View style={[styles.featureCard, { borderLeftColor: riskStyle.text }]}>
      <Pressable style={styles.featureHeader} onPress={onToggle}>
        <View style={styles.featureHeaderLeft}>
          <View style={[styles.letterBadge, { backgroundColor: riskStyle.bg, borderColor: riskStyle.border }]}>
            <Text style={[styles.letterText, { color: riskStyle.text }]}>{letter}</Text>
          </View>
          <View style={styles.featureTitleContainer}>
            <Text style={styles.featureTitle}>{title}</Text>
            <Text style={[styles.riskBadge, { color: riskStyle.text, backgroundColor: riskStyle.bg }]}>
              {analysis.risk_level.replace('_', ' ').toUpperCase()}
            </Text>
          </View>
        </View>
        <View style={styles.featureHeaderRight}>
          <Text style={styles.overallScore}>{(analysis.overall_score * 100).toFixed(0)}%</Text>
          <Ionicons
            name={expanded ? 'chevron-up' : 'chevron-down'}
            size={20}
            color="#6b7280"
          />
        </View>
      </Pressable>

      <Text style={styles.featureDescription}>{analysis.description}</Text>

      {expanded && (
        <View style={styles.featureDetails}>
          {children}
          <View style={styles.interpretationBox}>
            <Ionicons name="medkit-outline" size={16} color="#6366f1" />
            <Text style={styles.interpretationText}>{analysis.clinical_interpretation}</Text>
          </View>
        </View>
      )}
    </View>
  );
};

export const ABCDEFeatureDisplay: React.FC<ABCDEFeatureDisplayProps> = ({
  analysis,
  showMethodology = false,
}) => {
  const [expandedSection, setExpandedSection] = useState<string | null>('asymmetry');
  const [showNotes, setShowNotes] = useState(false);

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  const riskStyle = RISK_COLORS[analysis.risk_level] || RISK_COLORS.moderate;

  return (
    <View style={styles.container}>
      {/* Header with Total Score */}
      <View style={[styles.headerCard, { backgroundColor: riskStyle.bg, borderColor: riskStyle.border }]}>
        <View style={styles.headerTop}>
          <View>
            <Text style={styles.headerTitle}>ABCDE Feature Analysis</Text>
            <Text style={styles.headerSubtitle}>Quantitative lesion assessment</Text>
          </View>
          <View style={styles.totalScoreContainer}>
            <Text style={styles.totalScoreLabel}>Total Score</Text>
            <Text style={[styles.totalScoreValue, { color: riskStyle.text }]}>
              {analysis.total_score.toFixed(1)}
            </Text>
            <Text style={styles.totalScoreMax}>/10</Text>
          </View>
        </View>

        <View style={[styles.riskLevelBanner, { backgroundColor: riskStyle.text }]}>
          <Ionicons name={riskStyle.icon as any} size={18} color="white" />
          <Text style={styles.riskLevelText}>
            {analysis.risk_level.replace('_', ' ').toUpperCase()} RISK
          </Text>
        </View>

        <Text style={styles.summaryText}>{analysis.summary}</Text>

        {analysis.key_concerns.length > 0 && (
          <View style={styles.concernsContainer}>
            <Text style={styles.concernsTitle}>Key Concerns:</Text>
            {analysis.key_concerns.map((concern, index) => (
              <View key={index} style={styles.concernItem}>
                <Ionicons name="alert-circle" size={14} color={riskStyle.text} />
                <Text style={[styles.concernText, { color: riskStyle.text }]}>{concern}</Text>
              </View>
            ))}
          </View>
        )}
      </View>

      {/* Individual Feature Cards */}
      <FeatureCard
        title="Asymmetry"
        letter="A"
        analysis={analysis.asymmetry}
        expanded={expandedSection === 'asymmetry'}
        onToggle={() => toggleSection('asymmetry')}
      >
        <View style={styles.metersGrid}>
          <ScoreMeter score={analysis.asymmetry.horizontal_asymmetry} label="Left-Right" />
          <ScoreMeter score={analysis.asymmetry.vertical_asymmetry} label="Top-Bottom" />
          <ScoreMeter score={analysis.asymmetry.shape_asymmetry} label="Shape" />
          <ScoreMeter score={analysis.asymmetry.color_asymmetry} label="Color Distribution" />
        </View>
      </FeatureCard>

      <FeatureCard
        title="Border"
        letter="B"
        analysis={analysis.border}
        expanded={expandedSection === 'border'}
        onToggle={() => toggleSection('border')}
      >
        <View style={styles.metersGrid}>
          <ScoreMeter score={analysis.border.irregularity_index} label="Irregularity Index" />
          <ScoreMeter score={analysis.border.notching_score} label="Notching/Scalloping" />
          <ScoreMeter score={analysis.border.blur_score} label="Border Blur" />
          <ScoreMeter score={analysis.border.radial_variance} label="Radial Variance" />
        </View>
        <View style={styles.statRow}>
          <Text style={styles.statLabel}>Border color transitions:</Text>
          <Text style={styles.statValue}>{analysis.border.num_border_colors}</Text>
        </View>
      </FeatureCard>

      <FeatureCard
        title="Color"
        letter="C"
        analysis={analysis.color}
        expanded={expandedSection === 'color'}
        onToggle={() => toggleSection('color')}
      >
        <View style={styles.colorSection}>
          <View style={styles.colorStatsRow}>
            <View style={styles.colorStat}>
              <Text style={styles.colorStatValue}>{analysis.color.num_colors}</Text>
              <Text style={styles.colorStatLabel}>Colors</Text>
            </View>
            <View style={styles.colorStat}>
              <Text style={styles.colorStatValue}>{(analysis.color.color_variance * 100).toFixed(0)}%</Text>
              <Text style={styles.colorStatLabel}>Variance</Text>
            </View>
            <View style={styles.colorStat}>
              <Text style={styles.colorStatValue}>{analysis.color.dominant_color.replace('_', ' ')}</Text>
              <Text style={styles.colorStatLabel}>Dominant</Text>
            </View>
          </View>

          {analysis.color.colors_detected.length > 0 && (
            <View style={styles.colorsDetected}>
              <Text style={styles.colorsLabel}>Colors detected:</Text>
              <View style={styles.colorChips}>
                {analysis.color.colors_detected.map((color, index) => (
                  <View key={index} style={styles.colorChip}>
                    <Text style={styles.colorChipText}>{color.replace('_', ' ')}</Text>
                  </View>
                ))}
              </View>
            </View>
          )}

          {/* Concerning Features */}
          <View style={styles.concerningFeatures}>
            <View style={[styles.featureFlag, analysis.color.has_blue_white_veil && styles.featureFlagActive]}>
              <Ionicons
                name={analysis.color.has_blue_white_veil ? 'alert-circle' : 'checkmark-circle'}
                size={16}
                color={analysis.color.has_blue_white_veil ? '#dc2626' : '#10b981'}
              />
              <Text style={[styles.featureFlagText, analysis.color.has_blue_white_veil && styles.featureFlagTextActive]}>
                Blue-white veil {analysis.color.has_blue_white_veil ? 'PRESENT' : 'not detected'}
              </Text>
            </View>
            <View style={[styles.featureFlag, analysis.color.has_regression && styles.featureFlagActive]}>
              <Ionicons
                name={analysis.color.has_regression ? 'alert-circle' : 'checkmark-circle'}
                size={16}
                color={analysis.color.has_regression ? '#dc2626' : '#10b981'}
              />
              <Text style={[styles.featureFlagText, analysis.color.has_regression && styles.featureFlagTextActive]}>
                Regression areas {analysis.color.has_regression ? 'PRESENT' : 'not detected'}
              </Text>
            </View>
          </View>
        </View>
      </FeatureCard>

      <FeatureCard
        title="Diameter"
        letter="D"
        analysis={analysis.diameter}
        expanded={expandedSection === 'diameter'}
        onToggle={() => toggleSection('diameter')}
      >
        <View style={styles.diameterSection}>
          {analysis.diameter.calibration_available ? (
            <View style={styles.diameterMeasurement}>
              <Text style={styles.diameterValue}>
                {analysis.diameter.estimated_diameter_mm?.toFixed(1) || '?'}
              </Text>
              <Text style={styles.diameterUnit}>mm</Text>
              {analysis.diameter.is_above_6mm && (
                <View style={styles.thresholdWarning}>
                  <Ionicons name="warning" size={16} color="#dc2626" />
                  <Text style={styles.thresholdWarningText}>Exceeds 6mm threshold</Text>
                </View>
              )}
            </View>
          ) : (
            <View style={styles.noCalibration}>
              <Ionicons name="information-circle" size={20} color="#6b7280" />
              <Text style={styles.noCalibrationText}>
                No calibration available - diameter in pixels: {analysis.diameter.pixel_diameter}px
              </Text>
            </View>
          )}
          <View style={styles.statRow}>
            <Text style={styles.statLabel}>Area:</Text>
            <Text style={styles.statValue}>{analysis.diameter.area_pixels.toLocaleString()} px²</Text>
          </View>
        </View>
      </FeatureCard>

      {/* Evolution Card (special - usually no comparison data) */}
      <View style={[styles.featureCard, { borderLeftColor: '#6366f1' }]}>
        <View style={styles.featureHeader}>
          <View style={styles.featureHeaderLeft}>
            <View style={[styles.letterBadge, { backgroundColor: '#eef2ff', borderColor: '#c7d2fe' }]}>
              <Text style={[styles.letterText, { color: '#6366f1' }]}>E</Text>
            </View>
            <View style={styles.featureTitleContainer}>
              <Text style={styles.featureTitle}>Evolution</Text>
              {!analysis.evolution.has_comparison && (
                <Text style={[styles.riskBadge, { color: '#6366f1', backgroundColor: '#eef2ff' }]}>
                  NO COMPARISON DATA
                </Text>
              )}
            </View>
          </View>
        </View>
        <Text style={styles.featureDescription}>{analysis.evolution.description}</Text>
        <View style={styles.evolutionNote}>
          <Ionicons name="time-outline" size={16} color="#6366f1" />
          <Text style={styles.evolutionNoteText}>{analysis.evolution.clinical_interpretation}</Text>
        </View>
      </View>

      {/* Recommendation */}
      <View style={styles.recommendationCard}>
        <View style={styles.recommendationHeader}>
          <Ionicons name="clipboard-outline" size={20} color="#1e40af" />
          <Text style={styles.recommendationTitle}>Recommendation</Text>
        </View>
        <Text style={styles.recommendationText}>{analysis.recommendation}</Text>
      </View>

      {/* Methodology Notes (collapsible) */}
      {showMethodology && (
        <View style={styles.methodologySection}>
          <Pressable style={styles.methodologyHeader} onPress={() => setShowNotes(!showNotes)}>
            <Text style={styles.methodologyTitle}>Methodology Notes</Text>
            <Ionicons name={showNotes ? 'chevron-up' : 'chevron-down'} size={18} color="#6b7280" />
          </Pressable>
          {showNotes && (
            <View style={styles.methodologyContent}>
              {analysis.methodology_notes.map((note, index) => (
                <Text key={index} style={styles.methodologyNote}>• {note}</Text>
              ))}
            </View>
          )}
        </View>
      )}

      {/* Disclaimer */}
      <View style={styles.disclaimer}>
        <Text style={styles.disclaimerText}>
          This analysis supplements clinical dermoscopy examination. Quantitative metrics are computed
          from image analysis and may not capture all clinically relevant features visible under magnification.
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
  },
  headerCard: {
    borderRadius: 16,
    borderWidth: 2,
    padding: 16,
    marginBottom: 16,
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
  },
  headerSubtitle: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  totalScoreContainer: {
    alignItems: 'center',
  },
  totalScoreLabel: {
    fontSize: 11,
    color: '#6b7280',
    textTransform: 'uppercase',
  },
  totalScoreValue: {
    fontSize: 32,
    fontWeight: '800',
  },
  totalScoreMax: {
    fontSize: 12,
    color: '#9ca3af',
  },
  riskLevelBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 8,
    borderRadius: 8,
    marginBottom: 12,
    gap: 6,
  },
  riskLevelText: {
    color: 'white',
    fontWeight: '700',
    fontSize: 14,
  },
  summaryText: {
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
    marginBottom: 12,
  },
  concernsContainer: {
    marginTop: 8,
  },
  concernsTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 6,
  },
  concernItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 6,
    marginBottom: 4,
  },
  concernText: {
    flex: 1,
    fontSize: 13,
    lineHeight: 18,
  },
  featureCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderLeftWidth: 4,
    padding: 14,
    marginBottom: 12,
  },
  featureHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  featureHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  featureHeaderRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  letterBadge: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 2,
    justifyContent: 'center',
    alignItems: 'center',
  },
  letterText: {
    fontSize: 18,
    fontWeight: '800',
  },
  featureTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  featureTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  riskBadge: {
    fontSize: 10,
    fontWeight: '700',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  overallScore: {
    fontSize: 16,
    fontWeight: '700',
    color: '#374151',
  },
  featureDescription: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 8,
  },
  featureDetails: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  metersGrid: {
    gap: 10,
  },
  meterContainer: {
    marginBottom: 8,
  },
  meterHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  meterLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
  meterValue: {
    fontSize: 12,
    fontWeight: '600',
  },
  meterTrack: {
    height: 6,
    backgroundColor: '#e5e7eb',
    borderRadius: 3,
    overflow: 'hidden',
  },
  meterFill: {
    height: '100%',
    borderRadius: 3,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 6,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 13,
    color: '#6b7280',
  },
  statValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#374151',
  },
  interpretationBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    backgroundColor: '#eef2ff',
    padding: 10,
    borderRadius: 8,
    marginTop: 12,
  },
  interpretationText: {
    flex: 1,
    fontSize: 12,
    color: '#4338ca',
    lineHeight: 18,
  },
  colorSection: {},
  colorStatsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 12,
  },
  colorStat: {
    alignItems: 'center',
  },
  colorStatValue: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1f2937',
  },
  colorStatLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 2,
  },
  colorsDetected: {
    marginBottom: 12,
  },
  colorsLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 6,
  },
  colorChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  colorChip: {
    backgroundColor: '#f3f4f6',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  colorChipText: {
    fontSize: 12,
    color: '#374151',
    textTransform: 'capitalize',
  },
  concerningFeatures: {
    gap: 8,
  },
  featureFlag: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 8,
    backgroundColor: '#f9fafb',
    borderRadius: 8,
  },
  featureFlagActive: {
    backgroundColor: '#fef2f2',
  },
  featureFlagText: {
    fontSize: 13,
    color: '#374151',
  },
  featureFlagTextActive: {
    color: '#dc2626',
    fontWeight: '600',
  },
  diameterSection: {},
  diameterMeasurement: {
    flexDirection: 'row',
    alignItems: 'baseline',
    justifyContent: 'center',
    marginBottom: 12,
  },
  diameterValue: {
    fontSize: 36,
    fontWeight: '800',
    color: '#1f2937',
  },
  diameterUnit: {
    fontSize: 18,
    color: '#6b7280',
    marginLeft: 4,
  },
  thresholdWarning: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginLeft: 12,
    backgroundColor: '#fef2f2',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  thresholdWarningText: {
    fontSize: 11,
    color: '#dc2626',
    fontWeight: '600',
  },
  noCalibration: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#f9fafb',
    padding: 10,
    borderRadius: 8,
    marginBottom: 8,
  },
  noCalibrationText: {
    flex: 1,
    fontSize: 12,
    color: '#6b7280',
  },
  evolutionNote: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    backgroundColor: '#eef2ff',
    padding: 10,
    borderRadius: 8,
    marginTop: 8,
  },
  evolutionNoteText: {
    flex: 1,
    fontSize: 12,
    color: '#4338ca',
    lineHeight: 18,
  },
  recommendationCard: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#bfdbfe',
  },
  recommendationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  recommendationTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e40af',
  },
  recommendationText: {
    fontSize: 14,
    color: '#1e40af',
    lineHeight: 20,
  },
  methodologySection: {
    backgroundColor: '#f9fafb',
    borderRadius: 10,
    marginBottom: 12,
    overflow: 'hidden',
  },
  methodologyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
  },
  methodologyTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
  },
  methodologyContent: {
    paddingHorizontal: 12,
    paddingBottom: 12,
  },
  methodologyNote: {
    fontSize: 11,
    color: '#6b7280',
    lineHeight: 16,
    marginBottom: 4,
  },
  disclaimer: {
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#9ca3af',
  },
  disclaimerText: {
    fontSize: 11,
    color: '#6b7280',
    lineHeight: 16,
    fontStyle: 'italic',
  },
});

export default ABCDEFeatureDisplay;
