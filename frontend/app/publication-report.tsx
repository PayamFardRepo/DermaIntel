/**
 * Publication Report Builder Screen
 *
 * Allows users to generate publication-ready PDF case reports
 * from their skin analysis data.
 *
 * Features:
 * - Select analysis to include in report
 * - Configure report options (images, dermoscopy, etc.)
 * - Preview de-identified data
 * - Generate and download PDF
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  Switch,
  ActivityIndicator,
  Alert,
  Platform,
  Modal,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import * as FileSystem from 'expo-file-system/legacy';
import * as Sharing from 'expo-sharing';

import { API_BASE_URL } from '../config';
import AuthService from '../services/AuthService';

// Types
interface Analysis {
  id: number;
  image_url: string;
  predicted_class: string;
  confidence: number;
  risk_level: string;
  body_location: string | null;
  created_at: string;
  has_dermoscopy: boolean;
  has_biopsy: boolean;
  has_abcde: boolean;
}

interface ReportPreview {
  case_id: string;
  analysis_date: string;
  demographics: {
    age_range: string;
    gender: string;
    skin_type: string;
  };
  clinical_presentation: {
    body_location: string;
    symptom_duration: string | null;
  };
  diagnosis: {
    predicted_class: string;
    confidence: number;
    risk_level: string;
  };
  has_images: boolean;
  has_dermoscopy: boolean;
  has_biopsy: boolean;
  has_abcde: boolean;
}

interface ReportOptions {
  include_images: boolean;
  include_dermoscopy: boolean;
  include_heatmap: boolean;
  include_biopsy: boolean;
}

export default function PublicationReportScreen() {
  const router = useRouter();

  // State
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState<Analysis | null>(null);
  const [preview, setPreview] = useState<ReportPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [showPreviewModal, setShowPreviewModal] = useState(false);

  // Report options
  const [options, setOptions] = useState<ReportOptions>({
    include_images: true,
    include_dermoscopy: true,
    include_heatmap: true,
    include_biopsy: true,
  });

  // Load analyses on mount
  useEffect(() => {
    loadAnalyses();
  }, []);

  const loadAnalyses = async () => {
    try {
      setLoading(true);
      const token = AuthService.getToken();

      // Use AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout

      const response = await fetch(`${API_BASE_URL}/reports/analyses?limit=50`, {
        headers: { Authorization: `Bearer ${token}` },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        setAnalyses(data.analyses || []);
      } else {
        console.error('Error response:', response.status);
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.error('Request timed out');
        Alert.alert('Connection Timeout', 'Could not connect to server. Please check your network connection.');
      } else {
        console.error('Error loading analyses:', error);
        Alert.alert('Error', 'Failed to load analyses. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const loadPreview = async (analysisId: number) => {
    try {
      setPreviewLoading(true);
      const token = AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/reports/preview/${analysisId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (response.ok) {
        const data = await response.json();
        setPreview(data);
        setShowPreviewModal(true);
      }
    } catch (error) {
      console.error('Error loading preview:', error);
    } finally {
      setPreviewLoading(false);
    }
  };

  const generateReport = async () => {
    if (!selectedAnalysis) {
      Alert.alert('Select Analysis', 'Please select an analysis to generate a report.');
      return;
    }

    try {
      setGenerating(true);
      const token = AuthService.getToken();

      // Start report generation
      const response = await fetch(`${API_BASE_URL}/reports/case-report`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysis_id: selectedAnalysis.id,
          ...options,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start report generation');
      }

      const { report_id } = await response.json();

      // Poll for completion
      let attempts = 0;
      const maxAttempts = 30;

      while (attempts < maxAttempts) {
        await new Promise((resolve) => setTimeout(resolve, 1000));

        const statusResponse = await fetch(`${API_BASE_URL}/reports/${report_id}/status`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (statusResponse.ok) {
          const status = await statusResponse.json();

          if (status.status === 'ready') {
            // Download the report
            await downloadReport(report_id, status.case_id);
            return;
          } else if (status.status === 'failed') {
            throw new Error(status.error || 'Report generation failed');
          }
        }

        attempts++;
      }

      throw new Error('Report generation timed out');
    } catch (error: any) {
      console.error('Error generating report:', error);
      Alert.alert('Error', error.message || 'Failed to generate report');
    } finally {
      setGenerating(false);
    }
  };

  const downloadReport = async (reportId: string, caseId: string) => {
    try {
      const token = AuthService.getToken();
      const downloadUrl = `${API_BASE_URL}/reports/${reportId}/download`;

      if (Platform.OS === 'web') {
        // Web: Open in new tab or trigger download
        const response = await fetch(downloadUrl, {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (response.ok) {
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `case_report_${caseId}.pdf`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);

          Alert.alert('Success', 'Report downloaded successfully!');
        }
      } else {
        // Mobile: Use FileSystem.downloadAsync for direct download
        const filePath = `${FileSystem.documentDirectory}case_report_${caseId}.pdf`;

        const downloadResult = await FileSystem.downloadAsync(
          downloadUrl,
          filePath,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );

        if (downloadResult.status === 200) {
          if (await Sharing.isAvailableAsync()) {
            await Sharing.shareAsync(downloadResult.uri, {
              mimeType: 'application/pdf',
              dialogTitle: 'Share Case Report',
            });
          } else {
            Alert.alert('Success', `Report saved to: ${filePath}`);
          }
        } else {
          throw new Error(`Download failed with status ${downloadResult.status}`);
        }
      }
    } catch (error) {
      console.error('Error downloading report:', error);
      Alert.alert('Error', 'Failed to download report');
    }
  };

  const getRiskColor = (risk: string): string => {
    switch (risk?.toLowerCase()) {
      case 'high':
      case 'very_high':
        return '#dc2626';
      case 'medium':
        return '#f59e0b';
      default:
        return '#10b981';
    }
  };

  const renderAnalysisCard = (analysis: Analysis) => {
    const isSelected = selectedAnalysis?.id === analysis.id;

    return (
      <TouchableOpacity
        key={analysis.id}
        style={[styles.analysisCard, isSelected && styles.selectedCard]}
        onPress={() => setSelectedAnalysis(analysis)}
      >
        <View style={styles.cardContent}>
          {analysis.image_url && (
            <Image
              source={{ uri: `${API_BASE_URL}${analysis.image_url}` }}
              style={styles.thumbnail}
            />
          )}
          <View style={styles.cardInfo}>
            <Text style={styles.diagnosisText}>{analysis.predicted_class}</Text>
            <Text style={styles.confidenceText}>
              {((analysis.confidence || 0) * 100).toFixed(0)}% confidence
            </Text>
            <View style={styles.badgeRow}>
              <View style={[styles.riskBadge, { backgroundColor: getRiskColor(analysis.risk_level) }]}>
                <Text style={styles.riskText}>{analysis.risk_level?.toUpperCase()}</Text>
              </View>
              {analysis.has_dermoscopy && (
                <View style={styles.featureBadge}>
                  <Ionicons name="eye" size={12} color="#6366f1" />
                  <Text style={styles.featureText}>Dermoscopy</Text>
                </View>
              )}
              {analysis.has_biopsy && (
                <View style={styles.featureBadge}>
                  <Ionicons name="flask" size={12} color="#6366f1" />
                  <Text style={styles.featureText}>Biopsy</Text>
                </View>
              )}
            </View>
            <Text style={styles.dateText}>
              {new Date(analysis.created_at).toLocaleDateString()}
            </Text>
          </View>
          {isSelected && (
            <View style={styles.checkmark}>
              <Ionicons name="checkmark-circle" size={24} color="#10b981" />
            </View>
          )}
        </View>
      </TouchableOpacity>
    );
  };

  const renderPreviewModal = () => (
    <Modal
      visible={showPreviewModal}
      animationType="slide"
      transparent={true}
      onRequestClose={() => setShowPreviewModal(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Report Preview</Text>
            <TouchableOpacity onPress={() => setShowPreviewModal(false)}>
              <Ionicons name="close" size={24} color="#fff" />
            </TouchableOpacity>
          </View>

          {preview && (
            <ScrollView style={styles.previewScroll}>
              <View style={styles.previewSection}>
                <Text style={styles.previewLabel}>Case ID</Text>
                <Text style={styles.previewValue}>{preview.case_id}</Text>
              </View>

              <View style={styles.previewSection}>
                <Text style={styles.previewLabel}>Analysis Date</Text>
                <Text style={styles.previewValue}>{preview.analysis_date}</Text>
              </View>

              <View style={styles.previewSection}>
                <Text style={styles.previewSectionTitle}>De-identified Demographics</Text>
                <Text style={styles.previewValue}>
                  Age: {preview.demographics.age_range}
                </Text>
                <Text style={styles.previewValue}>
                  Gender: {preview.demographics.gender}
                </Text>
                <Text style={styles.previewValue}>
                  Skin Type: {preview.demographics.skin_type}
                </Text>
              </View>

              <View style={styles.previewSection}>
                <Text style={styles.previewSectionTitle}>Clinical Presentation</Text>
                <Text style={styles.previewValue}>
                  Location: {preview.clinical_presentation.body_location}
                </Text>
                {preview.clinical_presentation.symptom_duration && (
                  <Text style={styles.previewValue}>
                    Duration: {preview.clinical_presentation.symptom_duration}
                  </Text>
                )}
              </View>

              <View style={styles.previewSection}>
                <Text style={styles.previewSectionTitle}>Diagnosis</Text>
                <Text style={styles.previewValue}>
                  {preview.diagnosis.predicted_class} ({(preview.diagnosis.confidence * 100).toFixed(0)}%)
                </Text>
                <Text style={styles.previewValue}>
                  Risk Level: {preview.diagnosis.risk_level}
                </Text>
              </View>

              <View style={styles.previewSection}>
                <Text style={styles.previewSectionTitle}>Available Data</Text>
                <View style={styles.dataAvailability}>
                  <DataBadge label="Images" available={preview.has_images} />
                  <DataBadge label="Dermoscopy" available={preview.has_dermoscopy} />
                  <DataBadge label="ABCDE" available={preview.has_abcde} />
                  <DataBadge label="Biopsy" available={preview.has_biopsy} />
                </View>
              </View>

              <View style={styles.disclaimerBox}>
                <Ionicons name="shield-checkmark" size={20} color="#10b981" />
                <Text style={styles.disclaimerText}>
                  All personal identifiers will be removed from the report.
                  Names, dates of birth, and contact information are not included.
                </Text>
              </View>
            </ScrollView>
          )}

          <TouchableOpacity
            style={styles.closePreviewButton}
            onPress={() => setShowPreviewModal(false)}
          >
            <Text style={styles.closePreviewText}>Close Preview</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  return (
    <View style={styles.container}>
      <LinearGradient colors={['#0f172a', '#1e293b']} style={styles.gradient}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="#fff" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Publication Report</Text>
          <View style={styles.placeholder} />
        </View>

        <ScrollView style={styles.content}>
          {/* Instructions */}
          <View style={styles.instructionBox}>
            <Ionicons name="document-text" size={24} color="#6366f1" />
            <View style={styles.instructionText}>
              <Text style={styles.instructionTitle}>Generate Case Report</Text>
              <Text style={styles.instructionDesc}>
                Create a publication-ready PDF with de-identified patient data,
                clinical images, and AI analysis results.
              </Text>
            </View>
          </View>

          {/* Analysis Selection */}
          <Text style={styles.sectionTitle}>Select Analysis</Text>
          {loading ? (
            <ActivityIndicator size="large" color="#6366f1" style={styles.loader} />
          ) : analyses.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="images-outline" size={48} color="rgba(255,255,255,0.5)" />
              <Text style={styles.emptyText}>No analyses available</Text>
              <Text style={styles.emptySubtext}>
                Run a skin analysis first to generate reports.
              </Text>
            </View>
          ) : (
            <View style={styles.analysisList}>
              {analyses.map(renderAnalysisCard)}
            </View>
          )}

          {/* Report Options */}
          {selectedAnalysis && (
            <>
              <Text style={styles.sectionTitle}>Report Options</Text>
              <View style={styles.optionsCard}>
                <OptionRow
                  label="Include Clinical Images"
                  value={options.include_images}
                  onToggle={(v) => setOptions({ ...options, include_images: v })}
                />
                <OptionRow
                  label="Include Dermoscopy Analysis"
                  value={options.include_dermoscopy}
                  onToggle={(v) => setOptions({ ...options, include_dermoscopy: v })}
                  disabled={!selectedAnalysis.has_dermoscopy}
                />
                <OptionRow
                  label="Include AI Attention Heatmap"
                  value={options.include_heatmap}
                  onToggle={(v) => setOptions({ ...options, include_heatmap: v })}
                />
                <OptionRow
                  label="Include Biopsy Correlation"
                  value={options.include_biopsy}
                  onToggle={(v) => setOptions({ ...options, include_biopsy: v })}
                  disabled={!selectedAnalysis.has_biopsy}
                />
              </View>

              {/* Preview Button */}
              <TouchableOpacity
                style={styles.previewButton}
                onPress={() => loadPreview(selectedAnalysis.id)}
                disabled={previewLoading}
              >
                {previewLoading ? (
                  <ActivityIndicator size="small" color="#6366f1" />
                ) : (
                  <>
                    <Ionicons name="eye-outline" size={20} color="#6366f1" />
                    <Text style={styles.previewButtonText}>Preview Report Data</Text>
                  </>
                )}
              </TouchableOpacity>

              {/* Generate Button */}
              <TouchableOpacity
                style={[styles.generateButton, generating && styles.generatingButton]}
                onPress={generateReport}
                disabled={generating}
              >
                {generating ? (
                  <>
                    <ActivityIndicator size="small" color="#fff" />
                    <Text style={styles.generateText}>Generating Report...</Text>
                  </>
                ) : (
                  <>
                    <Ionicons name="download-outline" size={20} color="#fff" />
                    <Text style={styles.generateText}>Generate PDF Report</Text>
                  </>
                )}
              </TouchableOpacity>
            </>
          )}
        </ScrollView>

        {renderPreviewModal()}
      </LinearGradient>
    </View>
  );
}

// Helper Components
const OptionRow = ({
  label,
  value,
  onToggle,
  disabled = false,
}: {
  label: string;
  value: boolean;
  onToggle: (value: boolean) => void;
  disabled?: boolean;
}) => (
  <View style={[styles.optionRow, disabled && styles.optionDisabled]}>
    <Text style={[styles.optionLabel, disabled && styles.optionLabelDisabled]}>{label}</Text>
    <Switch
      value={value && !disabled}
      onValueChange={onToggle}
      disabled={disabled}
      trackColor={{ false: '#374151', true: '#6366f1' }}
      thumbColor={value && !disabled ? '#fff' : '#9ca3af'}
    />
  </View>
);

const DataBadge = ({ label, available }: { label: string; available: boolean }) => (
  <View style={[styles.dataBadge, available ? styles.dataBadgeAvailable : styles.dataBadgeUnavailable]}>
    <Ionicons
      name={available ? 'checkmark-circle' : 'close-circle'}
      size={14}
      color={available ? '#10b981' : '#6b7280'}
    />
    <Text style={[styles.dataBadgeText, available && styles.dataBadgeTextAvailable]}>{label}</Text>
  </View>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  gradient: {
    flex: 1,
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
    color: '#fff',
  },
  placeholder: {
    width: 40,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  instructionBox: {
    flexDirection: 'row',
    backgroundColor: 'rgba(99, 102, 241, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: 'rgba(99, 102, 241, 0.3)',
  },
  instructionText: {
    flex: 1,
    marginLeft: 12,
  },
  instructionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 4,
  },
  instructionDesc: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.7)',
    lineHeight: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 12,
    marginTop: 8,
  },
  loader: {
    marginVertical: 40,
  },
  emptyState: {
    alignItems: 'center',
    padding: 40,
  },
  emptyText: {
    fontSize: 16,
    color: '#fff',
    marginTop: 12,
  },
  emptySubtext: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.5)',
    marginTop: 4,
  },
  analysisList: {
    marginBottom: 16,
  },
  analysisCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  selectedCard: {
    borderColor: '#10b981',
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
  },
  cardContent: {
    flexDirection: 'row',
    padding: 12,
  },
  thumbnail: {
    width: 80,
    height: 80,
    borderRadius: 8,
    backgroundColor: '#1f2937',
  },
  cardInfo: {
    flex: 1,
    marginLeft: 12,
  },
  diagnosisText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  confidenceText: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.7)',
    marginTop: 2,
  },
  badgeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
    gap: 6,
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  riskText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#fff',
  },
  featureBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(99, 102, 241, 0.2)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  featureText: {
    fontSize: 10,
    color: '#6366f1',
    marginLeft: 4,
  },
  dateText: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.5)',
    marginTop: 4,
  },
  checkmark: {
    justifyContent: 'center',
  },
  optionsCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 4,
    marginBottom: 16,
  },
  optionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.1)',
  },
  optionDisabled: {
    opacity: 0.5,
  },
  optionLabel: {
    fontSize: 15,
    color: '#fff',
  },
  optionLabelDisabled: {
    color: 'rgba(255,255,255,0.5)',
  },
  previewButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(99, 102, 241, 0.1)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(99, 102, 241, 0.3)',
  },
  previewButtonText: {
    fontSize: 16,
    color: '#6366f1',
    fontWeight: '600',
    marginLeft: 8,
  },
  generateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#6366f1',
    borderRadius: 12,
    padding: 16,
    marginBottom: 40,
  },
  generatingButton: {
    backgroundColor: '#4f46e5',
  },
  generateText: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '600',
    marginLeft: 8,
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.8)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#1e293b',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '80%',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.1)',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  previewScroll: {
    padding: 16,
  },
  previewSection: {
    marginBottom: 16,
  },
  previewSectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6366f1',
    marginBottom: 8,
  },
  previewLabel: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.5)',
    marginBottom: 2,
  },
  previewValue: {
    fontSize: 15,
    color: '#fff',
    marginBottom: 4,
  },
  dataAvailability: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  dataBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
    backgroundColor: 'rgba(255,255,255,0.05)',
  },
  dataBadgeAvailable: {
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
  },
  dataBadgeUnavailable: {
    backgroundColor: 'rgba(107, 114, 128, 0.1)',
  },
  dataBadgeText: {
    fontSize: 12,
    color: '#6b7280',
    marginLeft: 4,
  },
  dataBadgeTextAvailable: {
    color: '#10b981',
  },
  disclaimerBox: {
    flexDirection: 'row',
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
  },
  disclaimerText: {
    flex: 1,
    fontSize: 13,
    color: 'rgba(255,255,255,0.8)',
    marginLeft: 10,
    lineHeight: 18,
  },
  closePreviewButton: {
    alignItems: 'center',
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.1)',
  },
  closePreviewText: {
    fontSize: 16,
    color: '#6366f1',
    fontWeight: '600',
  },
});
