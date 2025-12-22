import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

/**
 * Calibrated Uncertainty Display Component
 *
 * Displays clinical-grade uncertainty categories instead of raw confidence percentages.
 * This addresses the problem that "99.95% confidence" is model confidence, not diagnostic accuracy.
 *
 * Instead of showing misleading percentages, we show:
 * - "High concern - urgent evaluation needed"
 * - "Moderate concern - schedule dermatology appointment"
 * - "Low concern - monitor for changes"
 * - "Uncertain - clinical evaluation recommended"
 */

export interface CalibratedUncertainty {
  concern_level: 'high_concern' | 'moderate_concern' | 'low_concern' | 'uncertain' | 'insufficient_data';
  concern_label: string;
  concern_description: string;
  action_recommendation: string;
  model_confidence: number;
  calibrated_confidence: number;
  uncertainty_score: number;
  clinical_impression: string;
  clinical_caveats: string[];
  calibration_applied: string;
  factors_considered: string[];
  is_high_concern: boolean;
  is_uncertain: boolean;
  requires_evaluation: boolean;
  show_confidence_percentage: boolean;
}

interface CalibratedResultsDisplayProps {
  calibratedUncertainty: CalibratedUncertainty;
  predictedClass?: string;
  showDetails?: boolean;
}

const CONCERN_COLORS = {
  high_concern: {
    primary: '#dc2626',      // Red
    background: '#fef2f2',
    border: '#fecaca',
    icon: 'warning',
  },
  moderate_concern: {
    primary: '#f59e0b',      // Amber
    background: '#fffbeb',
    border: '#fed7aa',
    icon: 'alert-circle',
  },
  low_concern: {
    primary: '#10b981',      // Green
    background: '#ecfdf5',
    border: '#a7f3d0',
    icon: 'checkmark-circle',
  },
  uncertain: {
    primary: '#6366f1',      // Indigo
    background: '#eef2ff',
    border: '#c7d2fe',
    icon: 'help-circle',
  },
  insufficient_data: {
    primary: '#6b7280',      // Gray
    background: '#f9fafb',
    border: '#e5e7eb',
    icon: 'information-circle',
  },
};

export const CalibratedResultsDisplay: React.FC<CalibratedResultsDisplayProps> = ({
  calibratedUncertainty,
  predictedClass,
  showDetails = true,
}) => {
  const [expandedCaveats, setExpandedCaveats] = React.useState(false);
  const [expandedFactors, setExpandedFactors] = React.useState(false);

  const colors = CONCERN_COLORS[calibratedUncertainty.concern_level] || CONCERN_COLORS.uncertain;

  // Format the predicted class for display
  const formatClassName = (className: string) => {
    return className
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };

  return (
    <View style={styles.container}>
      {/* Main Concern Level Card */}
      <View style={[styles.concernCard, { backgroundColor: colors.background, borderColor: colors.border }]}>
        <View style={styles.concernHeader}>
          <View style={[styles.iconContainer, { backgroundColor: colors.primary }]}>
            <Ionicons name={colors.icon as any} size={28} color="white" />
          </View>
          <View style={styles.concernTitleContainer}>
            <Text style={[styles.concernLabel, { color: colors.primary }]}>
              {calibratedUncertainty.concern_label}
            </Text>
          </View>
        </View>

        {/* Clinical Impression - The main message */}
        <View style={styles.impressionContainer}>
          <Text style={styles.impressionText}>
            {calibratedUncertainty.clinical_impression}
          </Text>
        </View>

        {/* Action Recommendation */}
        <View style={[styles.actionContainer, { backgroundColor: `${colors.primary}15` }]}>
          <Ionicons name="arrow-forward-circle" size={20} color={colors.primary} />
          <Text style={[styles.actionText, { color: colors.primary }]}>
            {calibratedUncertainty.action_recommendation}
          </Text>
        </View>
      </View>

      {/* Description Card */}
      <View style={styles.descriptionCard}>
        <Text style={styles.descriptionTitle}>What This Means</Text>
        <Text style={styles.descriptionText}>
          {calibratedUncertainty.concern_description}
        </Text>
      </View>

      {/* Important Caveats - Critical Information */}
      {calibratedUncertainty.clinical_caveats && calibratedUncertainty.clinical_caveats.length > 0 && (
        <View style={styles.caveatsCard}>
          <Pressable
            style={styles.caveatsHeader}
            onPress={() => setExpandedCaveats(!expandedCaveats)}
          >
            <View style={styles.caveatsHeaderLeft}>
              <Ionicons name="alert-circle" size={20} color="#b45309" />
              <Text style={styles.caveatsTitle}>Important Information</Text>
            </View>
            <Ionicons
              name={expandedCaveats ? "chevron-up" : "chevron-down"}
              size={20}
              color="#78716c"
            />
          </Pressable>

          {expandedCaveats && (
            <View style={styles.caveatsList}>
              {calibratedUncertainty.clinical_caveats.map((caveat, index) => (
                <View key={index} style={styles.caveatItem}>
                  <Text style={styles.caveatBullet}>•</Text>
                  <Text style={styles.caveatText}>{caveat}</Text>
                </View>
              ))}
            </View>
          )}
        </View>
      )}

      {/* Technical Details - Collapsible */}
      {showDetails && (
        <View style={styles.detailsCard}>
          <Pressable
            style={styles.detailsHeader}
            onPress={() => setExpandedFactors(!expandedFactors)}
          >
            <View style={styles.detailsHeaderLeft}>
              <Ionicons name="analytics" size={20} color="#6b7280" />
              <Text style={styles.detailsTitle}>Analysis Details</Text>
            </View>
            <Ionicons
              name={expandedFactors ? "chevron-up" : "chevron-down"}
              size={20}
              color="#9ca3af"
            />
          </Pressable>

          {expandedFactors && (
            <View style={styles.detailsContent}>
              {/* Uncertainty Meter */}
              <View style={styles.meterContainer}>
                <Text style={styles.meterLabel}>AI Certainty Level</Text>
                <View style={styles.meterTrack}>
                  <View
                    style={[
                      styles.meterFill,
                      {
                        width: `${(1 - calibratedUncertainty.uncertainty_score) * 100}%`,
                        backgroundColor: calibratedUncertainty.uncertainty_score > 0.5 ? '#f59e0b' : '#10b981'
                      }
                    ]}
                  />
                </View>
                <Text style={styles.meterValue}>
                  {calibratedUncertainty.uncertainty_score > 0.5
                    ? 'Model shows significant uncertainty'
                    : 'Model shows reasonable certainty'}
                </Text>
              </View>

              {/* Factors Considered */}
              {calibratedUncertainty.factors_considered && calibratedUncertainty.factors_considered.length > 0 && (
                <View style={styles.factorsList}>
                  <Text style={styles.factorsLabel}>Factors Considered:</Text>
                  {calibratedUncertainty.factors_considered.map((factor, index) => (
                    <Text key={index} style={styles.factorItem}>• {factor}</Text>
                  ))}
                </View>
              )}

              {/* Calibration Method */}
              <Text style={styles.calibrationNote}>
                {calibratedUncertainty.calibration_applied}
              </Text>

              {/* Note about not showing raw percentages */}
              <View style={styles.noPercentageNote}>
                <Ionicons name="information-circle-outline" size={16} color="#6b7280" />
                <Text style={styles.noPercentageText}>
                  Raw confidence percentages are not displayed because they do not represent
                  diagnostic accuracy. Clinical categories provide more meaningful guidance.
                </Text>
              </View>
            </View>
          )}
        </View>
      )}

      {/* Disclaimer */}
      <View style={styles.disclaimerContainer}>
        <Text style={styles.disclaimerText}>
          This AI tool is for informational purposes only and does not provide medical diagnosis.
          Always consult a qualified healthcare provider for medical advice.
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
  },
  concernCard: {
    borderRadius: 16,
    borderWidth: 2,
    padding: 20,
    marginBottom: 16,
  },
  concernHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  concernTitleContainer: {
    flex: 1,
  },
  concernLabel: {
    fontSize: 18,
    fontWeight: '700',
    lineHeight: 24,
  },
  impressionContainer: {
    marginBottom: 16,
  },
  impressionText: {
    fontSize: 16,
    color: '#374151',
    lineHeight: 24,
  },
  actionContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: 12,
    borderRadius: 10,
    gap: 10,
  },
  actionText: {
    flex: 1,
    fontSize: 14,
    fontWeight: '600',
    lineHeight: 20,
  },
  descriptionCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  descriptionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  descriptionText: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 22,
  },
  caveatsCard: {
    backgroundColor: '#fffbeb',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#fcd34d',
    marginBottom: 16,
    overflow: 'hidden',
  },
  caveatsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 14,
  },
  caveatsHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  caveatsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#b45309',
  },
  caveatsList: {
    paddingHorizontal: 14,
    paddingBottom: 14,
    borderTopWidth: 1,
    borderTopColor: '#fcd34d',
    paddingTop: 14,
  },
  caveatItem: {
    flexDirection: 'row',
    marginBottom: 10,
  },
  caveatBullet: {
    fontSize: 14,
    color: '#b45309',
    marginRight: 8,
    marginTop: 2,
  },
  caveatText: {
    flex: 1,
    fontSize: 13,
    color: '#92400e',
    lineHeight: 20,
  },
  detailsCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    marginBottom: 16,
    overflow: 'hidden',
  },
  detailsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 14,
  },
  detailsHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  detailsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#4b5563',
  },
  detailsContent: {
    paddingHorizontal: 14,
    paddingBottom: 14,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
    paddingTop: 14,
  },
  meterContainer: {
    marginBottom: 16,
  },
  meterLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 6,
  },
  meterTrack: {
    height: 8,
    backgroundColor: '#e5e7eb',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 4,
  },
  meterFill: {
    height: '100%',
    borderRadius: 4,
  },
  meterValue: {
    fontSize: 11,
    color: '#9ca3af',
    fontStyle: 'italic',
  },
  factorsList: {
    marginBottom: 12,
  },
  factorsLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 6,
    fontWeight: '500',
  },
  factorItem: {
    fontSize: 12,
    color: '#6b7280',
    marginLeft: 4,
    marginBottom: 2,
  },
  calibrationNote: {
    fontSize: 11,
    color: '#9ca3af',
    fontStyle: 'italic',
    marginBottom: 12,
  },
  noPercentageNote: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 6,
    backgroundColor: '#f3f4f6',
    padding: 10,
    borderRadius: 8,
  },
  noPercentageText: {
    flex: 1,
    fontSize: 11,
    color: '#6b7280',
    lineHeight: 16,
  },
  disclaimerContainer: {
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

export default CalibratedResultsDisplay;
