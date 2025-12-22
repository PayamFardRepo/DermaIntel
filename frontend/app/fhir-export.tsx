import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  RefreshControl,
  Modal,
  Image,
  Share,
  Platform,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { API_URL } from '../config';

interface Analysis {
  id: number;
  predicted_class: string;
  lesion_confidence: number;
  image_url?: string;
  created_at: string;
  risk_level?: string;
  inflammatory_condition?: string;
}

interface FHIRResource {
  resourceType: string;
  id: string;
  status: string;
  category?: any[];
  code?: any;
  subject?: any;
  effectiveDateTime?: string;
  conclusion?: string;
  result?: any[];
  extension?: any[];
}

export default function FHIRExportScreen() {
  const router = useRouter();
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [exporting, setExporting] = useState<number | null>(null);

  // Preview modal
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [previewData, setPreviewData] = useState<FHIRResource | null>(null);
  const [previewAnalysis, setPreviewAnalysis] = useState<Analysis | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);

  // Export history
  const [exportHistory, setExportHistory] = useState<{ [key: number]: string }>({});

  useEffect(() => {
    fetchAnalyses();
    loadExportHistory();
  }, []);

  const fetchAnalyses = async () => {
    try {
      const token = await AsyncStorage.getItem('userToken');
      const response = await fetch(`${API_URL}/history`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setAnalyses(data.history || []);
      }
    } catch (error) {
      console.error('Error fetching analyses:', error);
      Alert.alert('Error', 'Failed to fetch analyses');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const loadExportHistory = async () => {
    try {
      const history = await AsyncStorage.getItem('fhirExportHistory');
      if (history) {
        setExportHistory(JSON.parse(history));
      }
    } catch (error) {
      console.error('Error loading export history:', error);
    }
  };

  const saveExportHistory = async (analysisId: number) => {
    try {
      const newHistory = {
        ...exportHistory,
        [analysisId]: new Date().toISOString(),
      };
      setExportHistory(newHistory);
      await AsyncStorage.setItem('fhirExportHistory', JSON.stringify(newHistory));
    } catch (error) {
      console.error('Error saving export history:', error);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchAnalyses();
  }, []);

  const fetchFHIRData = async (analysisId: number): Promise<FHIRResource | null> => {
    try {
      const token = await AsyncStorage.getItem('userToken');
      const response = await fetch(`${API_URL}/analysis/export/fhir/${analysisId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        return await response.json();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to fetch FHIR data');
        return null;
      }
    } catch (error) {
      console.error('Error fetching FHIR data:', error);
      Alert.alert('Error', 'Failed to fetch FHIR data');
      return null;
    }
  };

  const handlePreview = async (analysis: Analysis) => {
    setPreviewAnalysis(analysis);
    setShowPreviewModal(true);
    setLoadingPreview(true);

    const fhirData = await fetchFHIRData(analysis.id);
    setPreviewData(fhirData);
    setLoadingPreview(false);
  };

  const handleExport = async (analysis: Analysis) => {
    setExporting(analysis.id);

    try {
      const fhirData = await fetchFHIRData(analysis.id);
      if (!fhirData) {
        setExporting(null);
        return;
      }

      const fileName = `fhir-report-${analysis.id}.json`;
      const jsonString = JSON.stringify(fhirData, null, 2);

      if (Platform.OS === 'web') {
        // Web download
        const blob = new Blob([jsonString], { type: 'application/fhir+json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        a.click();
        URL.revokeObjectURL(url);
      } else {
        // Mobile - save to file and share
        const filePath = `${FileSystem.documentDirectory}${fileName}`;
        await FileSystem.writeAsStringAsync(filePath, jsonString);

        if (await Sharing.isAvailableAsync()) {
          await Sharing.shareAsync(filePath, {
            mimeType: 'application/json',
            dialogTitle: 'Export FHIR Report',
          });
        } else {
          Alert.alert('Success', `FHIR report saved to ${fileName}`);
        }
      }

      saveExportHistory(analysis.id);
      Alert.alert('Success', 'FHIR report exported successfully');
    } catch (error) {
      console.error('Error exporting FHIR:', error);
      Alert.alert('Error', 'Failed to export FHIR report');
    } finally {
      setExporting(null);
    }
  };

  const handleShareFHIR = async () => {
    if (!previewData || !previewAnalysis) return;

    try {
      const jsonString = JSON.stringify(previewData, null, 2);

      if (Platform.OS === 'web') {
        await navigator.clipboard.writeText(jsonString);
        Alert.alert('Copied', 'FHIR data copied to clipboard');
      } else {
        await Share.share({
          message: jsonString,
          title: `FHIR Report - Analysis ${previewAnalysis.id}`,
        });
      }
    } catch (error) {
      console.error('Error sharing:', error);
    }
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return '';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const formatDateTime = (dateString: string) => {
    if (!dateString) return '';
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getRiskColor = (risk?: string) => {
    switch (risk?.toLowerCase()) {
      case 'low': return '#4CAF50';
      case 'medium': return '#FF9800';
      case 'high': return '#F44336';
      case 'critical': return '#9C27B0';
      default: return '#9E9E9E';
    }
  };

  const renderAnalysisCard = (analysis: Analysis) => {
    const wasExported = exportHistory[analysis.id];
    const isExporting = exporting === analysis.id;

    return (
      <View key={analysis.id} style={styles.analysisCard}>
        <View style={styles.cardHeader}>
          {analysis.image_url ? (
            <Image
              source={{ uri: `${API_URL}${analysis.image_url}` }}
              style={styles.thumbnail}
            />
          ) : (
            <View style={styles.thumbnailPlaceholder}>
              <Ionicons name="image-outline" size={24} color="#ccc" />
            </View>
          )}
          <View style={styles.cardInfo}>
            <Text style={styles.diagnosisText}>{analysis.predicted_class || 'Unknown'}</Text>
            <View style={styles.metaRow}>
              <View style={styles.confidenceBadge}>
                <Text style={styles.confidenceText}>
                  {((analysis.lesion_confidence || 0) * 100).toFixed(0)}% confidence
                </Text>
              </View>
              {analysis.risk_level && (
                <View style={[styles.riskBadge, { backgroundColor: getRiskColor(analysis.risk_level) }]}>
                  <Text style={styles.riskText}>{analysis.risk_level}</Text>
                </View>
              )}
            </View>
            <Text style={styles.dateText}>{formatDate(analysis.created_at)}</Text>
          </View>
        </View>

        {wasExported && (
          <View style={styles.exportedBadge}>
            <Ionicons name="checkmark-circle" size={14} color="#4CAF50" />
            <Text style={styles.exportedText}>
              Exported {formatDateTime(wasExported)}
            </Text>
          </View>
        )}

        <View style={styles.cardActions}>
          <TouchableOpacity
            style={styles.previewButton}
            onPress={() => handlePreview(analysis)}
          >
            <Ionicons name="eye-outline" size={18} color="#667eea" />
            <Text style={styles.previewButtonText}>Preview</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.exportButton, isExporting && styles.exportButtonDisabled]}
            onPress={() => handleExport(analysis)}
            disabled={isExporting}
          >
            {isExporting ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <>
                <Ionicons name="download-outline" size={18} color="#fff" />
                <Text style={styles.exportButtonText}>Export FHIR</Text>
              </>
            )}
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  const renderFHIRSection = (title: string, icon: string, content: React.ReactNode) => (
    <View style={styles.fhirSection}>
      <View style={styles.fhirSectionHeader}>
        <Ionicons name={icon as any} size={18} color="#667eea" />
        <Text style={styles.fhirSectionTitle}>{title}</Text>
      </View>
      {content}
    </View>
  );

  const renderPreviewModal = () => (
    <Modal visible={showPreviewModal} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setShowPreviewModal(false)}>
            <Ionicons name="close" size={24} color="#333" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>FHIR Preview</Text>
          <TouchableOpacity onPress={handleShareFHIR} disabled={!previewData}>
            <Ionicons name="share-outline" size={24} color={previewData ? '#667eea' : '#ccc'} />
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.modalScroll}>
          {loadingPreview ? (
            <View style={styles.loadingPreview}>
              <ActivityIndicator size="large" color="#667eea" />
              <Text style={styles.loadingPreviewText}>Generating FHIR report...</Text>
            </View>
          ) : previewData ? (
            <>
              {/* FHIR Header */}
              <View style={styles.fhirHeader}>
                <View style={styles.fhirBadge}>
                  <Text style={styles.fhirBadgeText}>HL7 FHIR R4</Text>
                </View>
                <Text style={styles.fhirResourceType}>{previewData.resourceType}</Text>
                <Text style={styles.fhirId}>ID: {previewData.id}</Text>
              </View>

              {/* Status */}
              {renderFHIRSection('Status', 'checkmark-circle', (
                <View style={styles.statusContainer}>
                  <View style={[
                    styles.statusIndicator,
                    { backgroundColor: previewData.status === 'final' ? '#4CAF50' : '#FF9800' }
                  ]} />
                  <Text style={styles.statusText}>{previewData.status?.toUpperCase()}</Text>
                </View>
              ))}

              {/* Category */}
              {previewData.category && previewData.category.length > 0 && renderFHIRSection('Category', 'folder', (
                <View>
                  {previewData.category.map((cat: any, index: number) => (
                    <View key={index} style={styles.codingItem}>
                      {cat.coding?.map((c: any, cIndex: number) => (
                        <View key={cIndex}>
                          <Text style={styles.codingDisplay}>{c.display}</Text>
                          <Text style={styles.codingCode}>Code: {c.code}</Text>
                          <Text style={styles.codingSystem}>{c.system}</Text>
                        </View>
                      ))}
                    </View>
                  ))}
                </View>
              ))}

              {/* Code (Diagnosis) */}
              {previewData.code && renderFHIRSection('Diagnosis Code', 'medical', (
                <View style={styles.codingItem}>
                  {previewData.code.coding?.map((c: any, index: number) => (
                    <View key={index}>
                      <Text style={styles.codingDisplay}>{c.display}</Text>
                      <Text style={styles.codingCode}>SNOMED CT: {c.code}</Text>
                    </View>
                  ))}
                </View>
              ))}

              {/* Subject */}
              {previewData.subject && renderFHIRSection('Subject', 'person', (
                <View style={styles.referenceItem}>
                  <Text style={styles.referenceText}>{previewData.subject.reference}</Text>
                  {previewData.subject.display && (
                    <Text style={styles.referenceDisplay}>{previewData.subject.display}</Text>
                  )}
                </View>
              ))}

              {/* Effective Date */}
              {previewData.effectiveDateTime && renderFHIRSection('Effective Date', 'calendar', (
                <Text style={styles.dateValue}>
                  {formatDateTime(previewData.effectiveDateTime)}
                </Text>
              ))}

              {/* Conclusion */}
              {previewData.conclusion && renderFHIRSection('Conclusion', 'document-text', (
                <Text style={styles.conclusionText}>{previewData.conclusion}</Text>
              ))}

              {/* Results */}
              {previewData.result && previewData.result.length > 0 && renderFHIRSection('Results', 'analytics', (
                <View>
                  {previewData.result.map((r: any, index: number) => (
                    <View key={index} style={styles.resultItem}>
                      <Ionicons name="arrow-forward" size={14} color="#667eea" />
                      <Text style={styles.resultReference}>{r.reference}</Text>
                    </View>
                  ))}
                </View>
              ))}

              {/* Extensions */}
              {previewData.extension && previewData.extension.length > 0 && renderFHIRSection('Extensions', 'extension-puzzle', (
                <View>
                  {previewData.extension.map((ext: any, index: number) => (
                    <View key={index} style={styles.extensionItem}>
                      <Text style={styles.extensionUrl}>{ext.url?.split('/').pop()}</Text>
                      <Text style={styles.extensionValue}>
                        {ext.valueDecimal ?? ext.valueString ?? ext.valueBoolean?.toString() ?? 'N/A'}
                      </Text>
                    </View>
                  ))}
                </View>
              ))}

              {/* Raw JSON Preview */}
              <View style={styles.rawJsonSection}>
                <TouchableOpacity
                  style={styles.rawJsonHeader}
                  onPress={() => {}}
                >
                  <Ionicons name="code-slash" size={18} color="#666" />
                  <Text style={styles.rawJsonTitle}>Raw JSON</Text>
                </TouchableOpacity>
                <ScrollView horizontal style={styles.rawJsonScroll}>
                  <Text style={styles.rawJsonText}>
                    {JSON.stringify(previewData, null, 2)}
                  </Text>
                </ScrollView>
              </View>

              {/* Export Button */}
              {previewAnalysis && (
                <TouchableOpacity
                  style={styles.modalExportButton}
                  onPress={() => {
                    setShowPreviewModal(false);
                    handleExport(previewAnalysis);
                  }}
                >
                  <Ionicons name="download-outline" size={20} color="#fff" />
                  <Text style={styles.modalExportButtonText}>Export FHIR Report</Text>
                </TouchableOpacity>
              )}
            </>
          ) : (
            <View style={styles.errorPreview}>
              <Ionicons name="alert-circle" size={48} color="#F44336" />
              <Text style={styles.errorPreviewText}>Failed to load FHIR data</Text>
            </View>
          )}
        </ScrollView>
      </View>
    </Modal>
  );

  if (loading) {
    return (
      <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#fff" />
          <Text style={styles.loadingText}>Loading analyses...</Text>
        </View>
      </LinearGradient>
    );
  }

  const exportedCount = Object.keys(exportHistory).length;

  return (
    <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>FHIR Export</Text>
        <View style={styles.headerRight} />
      </View>

      {/* Info Banner */}
      <View style={styles.infoBanner}>
        <View style={styles.infoBannerIcon}>
          <Ionicons name="medical" size={24} color="#667eea" />
        </View>
        <View style={styles.infoBannerContent}>
          <Text style={styles.infoBannerTitle}>HL7 FHIR R4 Format</Text>
          <Text style={styles.infoBannerText}>
            Export analyses in standardized healthcare format for EMR integration
          </Text>
        </View>
      </View>

      {/* Stats */}
      <View style={styles.statsRow}>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{analyses.length}</Text>
          <Text style={styles.statLabel}>Total Analyses</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{exportedCount}</Text>
          <Text style={styles.statLabel}>Exported</Text>
        </View>
      </View>

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#fff" />
        }
      >
        <Text style={styles.sectionTitle}>Available for Export</Text>

        {analyses.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons name="document-text-outline" size={64} color="rgba(255,255,255,0.5)" />
            <Text style={styles.emptyText}>No analyses available</Text>
            <Text style={styles.emptySubtext}>
              Complete a skin analysis first to export FHIR reports
            </Text>
          </View>
        ) : (
          analyses.map(renderAnalysisCard)
        )}

        {/* FHIR Info */}
        <View style={styles.fhirInfoCard}>
          <Text style={styles.fhirInfoTitle}>About FHIR Export</Text>
          <View style={styles.fhirInfoItem}>
            <Ionicons name="shield-checkmark" size={20} color="#4CAF50" />
            <Text style={styles.fhirInfoText}>
              HL7 FHIR R4 compliant DiagnosticReport format
            </Text>
          </View>
          <View style={styles.fhirInfoItem}>
            <Ionicons name="git-branch" size={20} color="#2196F3" />
            <Text style={styles.fhirInfoText}>
              SNOMED CT coded diagnoses for interoperability
            </Text>
          </View>
          <View style={styles.fhirInfoItem}>
            <Ionicons name="cloud-upload" size={20} color="#FF9800" />
            <Text style={styles.fhirInfoText}>
              Ready for EMR/EHR system integration
            </Text>
          </View>
          <View style={styles.fhirInfoItem}>
            <Ionicons name="analytics" size={20} color="#9C27B0" />
            <Text style={styles.fhirInfoText}>
              Includes AI confidence and risk assessments
            </Text>
          </View>
        </View>
      </ScrollView>

      {renderPreviewModal()}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#fff',
    marginTop: 10,
    fontSize: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 20,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  headerRight: {
    width: 40,
  },
  infoBanner: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    marginHorizontal: 20,
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
    alignItems: 'center',
  },
  infoBannerIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#E8EAF6',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  infoBannerContent: {
    flex: 1,
  },
  infoBannerTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  infoBannerText: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  statsRow: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.15)',
    marginHorizontal: 20,
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  statLabel: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 4,
  },
  statDivider: {
    width: 1,
    backgroundColor: 'rgba(255,255,255,0.3)',
    marginHorizontal: 20,
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 12,
  },
  analysisCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  thumbnail: {
    width: 60,
    height: 60,
    borderRadius: 8,
  },
  thumbnailPlaceholder: {
    width: 60,
    height: 60,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardInfo: {
    flex: 1,
    marginLeft: 12,
  },
  diagnosisText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 6,
  },
  metaRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 4,
  },
  confidenceBadge: {
    backgroundColor: '#E8EAF6',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  confidenceText: {
    fontSize: 11,
    color: '#667eea',
    fontWeight: '500',
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  riskText: {
    fontSize: 11,
    color: '#fff',
    fontWeight: '500',
    textTransform: 'capitalize',
  },
  dateText: {
    fontSize: 12,
    color: '#999',
  },
  exportedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#E8F5E9',
    padding: 8,
    borderRadius: 6,
    marginTop: 12,
    gap: 6,
  },
  exportedText: {
    fontSize: 12,
    color: '#4CAF50',
  },
  cardActions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  previewButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f0f0f0',
    padding: 10,
    borderRadius: 8,
    gap: 6,
  },
  previewButtonText: {
    fontSize: 14,
    color: '#667eea',
    fontWeight: '500',
  },
  exportButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#667eea',
    padding: 10,
    borderRadius: 8,
    gap: 6,
  },
  exportButtonDisabled: {
    opacity: 0.7,
  },
  exportButtonText: {
    fontSize: 14,
    color: '#fff',
    fontWeight: '500',
  },
  emptyState: {
    alignItems: 'center',
    paddingTop: 40,
    paddingBottom: 40,
  },
  emptyText: {
    color: 'rgba(255,255,255,0.9)',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
  },
  emptySubtext: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
  fhirInfoCard: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 12,
    padding: 16,
    marginTop: 10,
    marginBottom: 30,
  },
  fhirInfoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 12,
  },
  fhirInfoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 10,
  },
  fhirInfoText: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.9)',
    flex: 1,
  },
  // Modal Styles
  modalContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 15,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  modalScroll: {
    flex: 1,
    padding: 20,
  },
  loadingPreview: {
    alignItems: 'center',
    paddingTop: 60,
  },
  loadingPreviewText: {
    fontSize: 14,
    color: '#666',
    marginTop: 12,
  },
  errorPreview: {
    alignItems: 'center',
    paddingTop: 60,
  },
  errorPreviewText: {
    fontSize: 16,
    color: '#F44336',
    marginTop: 12,
  },
  fhirHeader: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    marginBottom: 16,
  },
  fhirBadge: {
    backgroundColor: '#E8EAF6',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
    marginBottom: 12,
  },
  fhirBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#667eea',
  },
  fhirResourceType: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
  },
  fhirId: {
    fontSize: 12,
    color: '#999',
    marginTop: 4,
  },
  fhirSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  fhirSectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  fhirSectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  statusIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  statusText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  codingItem: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  codingDisplay: {
    fontSize: 15,
    fontWeight: '600',
    color: '#333',
  },
  codingCode: {
    fontSize: 13,
    color: '#667eea',
    marginTop: 4,
  },
  codingSystem: {
    fontSize: 11,
    color: '#999',
    marginTop: 2,
  },
  referenceItem: {
    padding: 8,
  },
  referenceText: {
    fontSize: 14,
    color: '#333',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  referenceDisplay: {
    fontSize: 13,
    color: '#666',
    marginTop: 4,
  },
  dateValue: {
    fontSize: 15,
    color: '#333',
  },
  conclusionText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 22,
  },
  resultItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 6,
  },
  resultReference: {
    fontSize: 13,
    color: '#333',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  extensionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  extensionUrl: {
    fontSize: 13,
    color: '#666',
  },
  extensionValue: {
    fontSize: 13,
    fontWeight: '500',
    color: '#333',
  },
  rawJsonSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    marginBottom: 16,
    overflow: 'hidden',
  },
  rawJsonHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  rawJsonTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  rawJsonScroll: {
    maxHeight: 200,
    backgroundColor: '#1e1e1e',
  },
  rawJsonText: {
    fontSize: 11,
    color: '#d4d4d4',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    padding: 12,
  },
  modalExportButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#667eea',
    padding: 16,
    borderRadius: 12,
    marginBottom: 30,
    gap: 8,
  },
  modalExportButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
});
