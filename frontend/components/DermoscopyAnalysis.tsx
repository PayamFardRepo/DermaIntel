import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

interface DermoscopyData {
  is_dermoscopy_image: boolean;
  detection_confidence: number;
  image_type: string;
  detection_features: string[];
  dermoscopic_structures?: {
    pigment_network?: {
      present: boolean;
      density: string;
      score: number;
    };
    dots_globules?: {
      present: boolean;
      count: number;
      pattern: string;
    };
    blue_white_veil?: {
      present: boolean;
      coverage: number;
    };
    streaks?: {
      present: boolean;
      count: number;
      pattern: string;
    };
    vascular_patterns?: {
      present: boolean;
      types: string[];
    };
    regression_structures?: {
      present: boolean;
      percentage: number;
    };
  };
}

interface Props {
  dermoscopyData: DermoscopyData;
}

/**
 * Dermoscopy Analysis Component
 * Displays specialized analysis for dermatoscope images
 */
export default function DermoscopyAnalysis({ dermoscopyData }: Props) {
  const structures = dermoscopyData.dermoscopic_structures || {};

  const getStructureIcon = (structureName: string) => {
    const icons: Record<string, string> = {
      'pigment_network': 'grid-outline',
      'dots_globules': 'ellipse-outline',
      'blue_white_veil': 'water-outline',
      'streaks': 'trending-up-outline',
      'vascular_patterns': 'git-network-outline',
      'regression_structures': 'analytics-outline'
    };
    return icons[structureName] || 'information-circle-outline';
  };

  const getStructureDescription = (structureName: string) => {
    const descriptions: Record<string, string> = {
      'pigment_network': 'A net-like pigmented pattern indicating melanocyte distribution',
      'dots_globules': 'Small dark circular structures indicating melanin clusters',
      'blue_white_veil': 'Blue-whitish areas indicating deeper melanin deposits',
      'streaks': 'Linear pigmented structures at lesion periphery',
      'vascular_patterns': 'Blood vessel patterns visible through dermoscopy',
      'regression_structures': 'Areas of melanin regression indicating healing or changes'
    };
    return descriptions[structureName] || 'Dermoscopic feature';
  };

  const getClinicalSignificance = (structureName: string, data: any) => {
    switch (structureName) {
      case 'pigment_network':
        if (data.density === 'irregular') {
          return { level: 'warning', text: 'Irregular network may indicate atypical melanocytic lesion' };
        }
        return { level: 'info', text: 'Regular network is common in benign nevi' };

      case 'dots_globules':
        if (data.pattern === 'clustered' || data.count > 20) {
          return { level: 'warning', text: 'Multiple/clustered globules warrant dermatologist evaluation' };
        }
        return { level: 'info', text: 'Few scattered globules are often benign' };

      case 'blue_white_veil':
        return { level: 'alert', text: 'Blue-white veil is a melanoma-associated feature - requires evaluation' };

      case 'streaks':
        if (data.pattern === 'irregular') {
          return { level: 'warning', text: 'Irregular streaks may indicate melanoma' };
        }
        return { level: 'info', text: 'Radial streaming pattern detected' };

      case 'vascular_patterns':
        return { level: 'info', text: 'Vascular patterns help differentiate lesion types' };

      case 'regression_structures':
        if (data.percentage > 30) {
          return { level: 'warning', text: 'Significant regression may indicate melanoma' };
        }
        return { level: 'info', text: 'Regression structures detected' };

      default:
        return { level: 'info', text: 'Dermoscopic feature detected' };
    }
  };

  const getSignificanceColor = (level: string) => {
    switch (level) {
      case 'alert': return '#dc2626';
      case 'warning': return '#f59e0b';
      case 'info': return '#3b82f6';
      default: return '#6b7280';
    }
  };

  if (!dermoscopyData.is_dermoscopy_image) {
    return (
      <View style={styles.container}>
        <View style={styles.nonDermoscopyCard}>
          <Ionicons name="camera-outline" size={48} color="#9ca3af" />
          <Text style={styles.nonDermoscopyTitle}>Clinical Photo Detected</Text>
          <Text style={styles.nonDermoscopyText}>
            This appears to be a standard clinical photograph rather than a dermatoscope image.
            Dermoscopy mode provides enhanced analysis for images captured with a dermatoscope device.
          </Text>
          <View style={styles.infoBox}>
            <Ionicons name="information-circle" size={20} color="#3b82f6" />
            <Text style={styles.infoText}>
              For dermoscopy analysis, use a dermatoscope or dermoscopy attachment for your smartphone camera.
            </Text>
          </View>
        </View>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {/* Detection Confidence */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="checkmark-circle" size={24} color="#10b981" />
          <Text style={styles.cardTitle}>Dermoscopy Image Detected</Text>
        </View>
        <Text style={styles.confidenceText}>
          Confidence: {(dermoscopyData.detection_confidence * 100).toFixed(0)}%
        </Text>

        {dermoscopyData.detection_features.length > 0 && (
          <View style={styles.featuresContainer}>
            <Text style={styles.featuresTitle}>Detected Features:</Text>
            {dermoscopyData.detection_features.map((feature, index) => (
              <View key={index} style={styles.featureItem}>
                <Ionicons name="checkmark" size={16} color="#10b981" />
                <Text style={styles.featureText}>{feature}</Text>
              </View>
            ))}
          </View>
        )}
      </View>

      {/* Dermoscopic Structures */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="medical" size={24} color="#3b82f6" />
          <Text style={styles.cardTitle}>Dermoscopic Structures Analysis</Text>
        </View>
        <Text style={styles.cardSubtitle}>
          Pattern recognition using 7-point checklist and ABCD criteria
        </Text>

        {Object.keys(structures).length === 0 ? (
          <View style={styles.noStructures}>
            <Ionicons name="scan-outline" size={32} color="#9ca3af" />
            <Text style={styles.noStructuresText}>No specific dermoscopic structures detected</Text>
          </View>
        ) : (
          Object.entries(structures).map(([key, value]: [string, any]) => {
            if (!value.present) return null;

            const significance = getClinicalSignificance(key, value);
            const significanceColor = getSignificanceColor(significance.level);

            return (
              <View key={key} style={styles.structureCard}>
                <View style={styles.structureHeader}>
                  <View style={styles.structureHeaderLeft}>
                    <Ionicons
                      name={getStructureIcon(key) as any}
                      size={24}
                      color="#3b82f6"
                    />
                    <Text style={styles.structureName}>
                      {key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                    </Text>
                  </View>
                  <View style={[styles.presentBadge, { backgroundColor: significanceColor }]}>
                    <Text style={styles.presentBadgeText}>Present</Text>
                  </View>
                </View>

                <Text style={styles.structureDescription}>
                  {getStructureDescription(key)}
                </Text>

                {/* Structure-specific details */}
                <View style={styles.detailsContainer}>
                  {value.density && (
                    <View style={styles.detailItem}>
                      <Text style={styles.detailLabel}>Pattern:</Text>
                      <Text style={styles.detailValue}>{value.density}</Text>
                    </View>
                  )}
                  {value.count !== undefined && (
                    <View style={styles.detailItem}>
                      <Text style={styles.detailLabel}>Count:</Text>
                      <Text style={styles.detailValue}>{value.count}</Text>
                    </View>
                  )}
                  {value.pattern && (
                    <View style={styles.detailItem}>
                      <Text style={styles.detailLabel}>Distribution:</Text>
                      <Text style={styles.detailValue}>{value.pattern}</Text>
                    </View>
                  )}
                  {value.coverage !== undefined && (
                    <View style={styles.detailItem}>
                      <Text style={styles.detailLabel}>Coverage:</Text>
                      <Text style={styles.detailValue}>{(value.coverage * 100).toFixed(1)}%</Text>
                    </View>
                  )}
                  {value.percentage !== undefined && (
                    <View style={styles.detailItem}>
                      <Text style={styles.detailLabel}>Percentage:</Text>
                      <Text style={styles.detailValue}>{value.percentage.toFixed(1)}%</Text>
                    </View>
                  )}
                  {value.types && (
                    <View style={styles.detailItem}>
                      <Text style={styles.detailLabel}>Types:</Text>
                      <Text style={styles.detailValue}>{value.types.join(', ')}</Text>
                    </View>
                  )}
                  {value.score !== undefined && (
                    <View style={styles.detailItem}>
                      <Text style={styles.detailLabel}>Score:</Text>
                      <Text style={styles.detailValue}>{(value.score * 100).toFixed(0)}%</Text>
                    </View>
                  )}
                </View>

                {/* Clinical Significance */}
                <View style={[styles.significanceBox, { borderLeftColor: significanceColor }]}>
                  <Ionicons
                    name={significance.level === 'alert' ? 'alert-circle' : 'information-circle'}
                    size={18}
                    color={significanceColor}
                  />
                  <Text style={[styles.significanceText, { color: significanceColor }]}>
                    {significance.text}
                  </Text>
                </View>
              </View>
            );
          })
        )}
      </View>

      {/* Educational Information */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="school-outline" size={24} color="#8b5cf6" />
          <Text style={styles.cardTitle}>About Dermoscopy</Text>
        </View>

        <Text style={styles.educationText}>
          Dermoscopy (also called dermatoscopy or epiluminescence microscopy) is a non-invasive technique
          that allows visualization of subsurface skin structures not visible to the naked eye.
        </Text>

        <View style={styles.benefitsList}>
          <View style={styles.benefitItem}>
            <Ionicons name="checkmark-circle" size={18} color="#10b981" />
            <Text style={styles.benefitText}>
              Improves melanoma detection accuracy by up to 30%
            </Text>
          </View>
          <View style={styles.benefitItem}>
            <Ionicons name="checkmark-circle" size={18} color="#10b981" />
            <Text style={styles.benefitText}>
              Reduces unnecessary biopsies of benign lesions
            </Text>
          </View>
          <View style={styles.benefitItem}>
            <Ionicons name="checkmark-circle" size={18} color="#10b981" />
            <Text style={styles.benefitText}>
              Enables tracking of lesion changes over time
            </Text>
          </View>
          <View style={styles.benefitItem}>
            <Ionicons name="checkmark-circle" size={18} color="#10b981" />
            <Text style={styles.benefitText}>
              Helps differentiate melanocytic from non-melanocytic lesions
            </Text>
          </View>
        </View>
      </View>

      {/* Recommendation */}
      <View style={styles.recommendationCard}>
        <LinearGradient
          colors={['#3b82f6', '#2563eb']}
          style={styles.recommendationGradient}
        >
          <Ionicons name="medical" size={32} color="#fff" />
          <Text style={styles.recommendationTitle}>Professional Evaluation</Text>
          <Text style={styles.recommendationText}>
            Dermoscopy findings should always be interpreted by a qualified dermatologist.
            AI analysis is a screening tool and not a substitute for expert clinical assessment.
          </Text>
        </LinearGradient>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    margin: 16,
    marginBottom: 0,
    marginTop: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
    gap: 10,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
  },
  cardSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 16,
  },
  confidenceText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#10b981',
    marginBottom: 12,
  },
  featuresContainer: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  featuresTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#4b5563',
    marginBottom: 8,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
  },
  featureText: {
    fontSize: 14,
    color: '#6b7280',
  },
  noStructures: {
    alignItems: 'center',
    paddingVertical: 24,
  },
  noStructuresText: {
    marginTop: 8,
    fontSize: 14,
    color: '#9ca3af',
  },
  structureCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  structureHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  structureHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    flex: 1,
  },
  structureName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  presentBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  presentBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#fff',
  },
  structureDescription: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 12,
    lineHeight: 18,
  },
  detailsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 12,
  },
  detailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  detailLabel: {
    fontSize: 13,
    color: '#6b7280',
  },
  detailValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1f2937',
  },
  significanceBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    backgroundColor: '#fff',
    padding: 12,
    borderRadius: 8,
    borderLeftWidth: 3,
  },
  significanceText: {
    flex: 1,
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '500',
  },
  educationText: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
    marginBottom: 16,
  },
  benefitsList: {
    gap: 10,
  },
  benefitItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
  },
  benefitText: {
    flex: 1,
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
  },
  recommendationCard: {
    margin: 16,
    marginTop: 16,
    borderRadius: 16,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  recommendationGradient: {
    padding: 24,
    alignItems: 'center',
  },
  recommendationTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#fff',
    marginTop: 12,
    marginBottom: 8,
  },
  recommendationText: {
    fontSize: 14,
    color: '#fff',
    textAlign: 'center',
    lineHeight: 20,
    opacity: 0.95,
  },
  nonDermoscopyCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    margin: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  nonDermoscopyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
    marginTop: 16,
    marginBottom: 8,
  },
  nonDermoscopyText: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 16,
  },
  infoBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    backgroundColor: '#eff6ff',
    padding: 16,
    borderRadius: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#3b82f6',
  },
  infoText: {
    flex: 1,
    fontSize: 13,
    color: '#1e40af',
    lineHeight: 18,
  },
});
