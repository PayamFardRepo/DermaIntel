import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  Alert,
  Platform,
  ActivityIndicator,
  TextInput,
  Modal
} from 'react-native';
import { router } from 'expo-router';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';

interface GeneticTestResult {
  id: number;
  test_id: string;
  test_date: string;
  lab_name: string;
  test_type: string;
  total_variants_analyzed: number;
  pathogenic_variants: number;
  likely_pathogenic_variants: number;
  vus_variants: number;
  melanoma_risk_level: string;
  nmsc_risk_level: string;
  screening_recommendations: string[];
  created_at: string;
}

interface GeneticVariant {
  id: number;
  gene_symbol: string;
  rsid: string;
  hgvs_p: string;
  acmg_classification: string;
  zygosity: string;
  clinical_significance: string;
  associated_conditions: string[];
}

interface ReferenceGene {
  gene_symbol: string;
  gene_name: string;
  category: string;
  chromosome?: string;
  inheritance: string;
  penetrance?: string;
  risk_increase: string;
  screening: string;
  acmg_actionable: boolean;
  melanoma_risk_multiplier: number;
  bcc_risk_multiplier?: number;
  clinvar_pathogenic_count?: number;
  associated_conditions?: string[];
  key_variants?: Array<{
    variation_id?: string;
    title?: string;
    clinical_significance?: string;
  }>;
  sources?: string[];
}

interface RiskSummary {
  has_genetic_data: boolean;
  melanoma_risk: {
    level: string;
    multiplier: number;
  };
  nmsc_risk: {
    level: string;
    multiplier: number;
  };
  high_risk_genes: string[];
  moderate_risk_genes: string[];
  pharmacogenomic_alerts: string[];
  recommendations: string[];
}

export default function GeneticTestingScreen() {
  const { t } = useTranslation();
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'results' | 'upload' | 'genes' | 'summary'>('results');
  const [geneticTests, setGeneticTests] = useState<GeneticTestResult[]>([]);
  const [referenceGenes, setReferenceGenes] = useState<ReferenceGene[]>([]);
  const [riskSummary, setRiskSummary] = useState<RiskSummary | null>(null);
  const [selectedTest, setSelectedTest] = useState<GeneticTestResult | null>(null);
  const [testVariants, setTestVariants] = useState<GeneticVariant[]>([]);
  const [showVariantsModal, setShowVariantsModal] = useState(false);

  // Upload form state
  const [selectedFile, setSelectedFile] = useState<any>(null);
  const [labName, setLabName] = useState('');
  const [testType, setTestType] = useState('panel');
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    fetchGeneticTests();
    fetchReferenceGenes();
    fetchRiskSummary();
  }, []);

  const fetchGeneticTests = async () => {
    try {
      setIsLoading(true);
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert('Error', 'Please login again');
        router.push('/');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/genetics/test-results`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setGeneticTests(data.test_results || []);
      }
    } catch (error) {
      console.error('Error fetching genetic tests:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchReferenceGenes = async () => {
    try {
      const token = AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/genetics/reference-genes`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setReferenceGenes(data.genes || []);
      }
    } catch (error) {
      console.error('Error fetching reference genes:', error);
    }
  };

  const fetchRiskSummary = async () => {
    try {
      const token = AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/genetics/risk-summary`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setRiskSummary(data);
      }
    } catch (error) {
      console.error('Error fetching risk summary:', error);
    }
  };

  const pickVCFFile = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['*/*'],
        copyToCacheDirectory: true,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const file = result.assets[0];
        // Check if it's a VCF file
        if (file.name?.toLowerCase().endsWith('.vcf') || file.name?.toLowerCase().endsWith('.vcf.gz')) {
          setSelectedFile(file);
        } else {
          Alert.alert('Invalid File', 'Please select a VCF file (.vcf or .vcf.gz)');
        }
      }
    } catch (error) {
      console.error('Error picking file:', error);
      Alert.alert('Error', 'Failed to select file');
    }
  };

  const uploadVCFFile = async () => {
    if (!selectedFile) {
      Alert.alert('Error', 'Please select a VCF file first');
      return;
    }

    try {
      setIsUploading(true);
      const token = AuthService.getToken();

      const formData = new FormData();
      const fileData = {
        uri: selectedFile.uri,
        type: 'text/plain',
        name: selectedFile.name || 'genetic_test.vcf',
      } as any;
      formData.append('file', fileData);
      formData.append('test_name', labName ? `${labName} - ${testType}` : `VCF Upload - ${testType}`);

      const response = await fetch(`${API_BASE_URL}/genetics/upload-vcf`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(
          'Success',
          `VCF file uploaded successfully!\n\nVariants analyzed: ${data.total_variants_in_file}\nDermatology-relevant: ${data.dermatology_relevant_variants}\nPathogenic: ${data.pathogenic_found}`,
          [{ text: 'OK', onPress: () => {
            setSelectedFile(null);
            setLabName('');
            fetchGeneticTests();
            fetchRiskSummary();
            setActiveTab('results');
          }}]
        );
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to upload VCF file');
      }
    } catch (error) {
      console.error('Error uploading VCF:', error);
      Alert.alert('Error', 'Failed to upload VCF file');
    } finally {
      setIsUploading(false);
    }
  };

  const viewTestVariants = async (test: GeneticTestResult) => {
    setSelectedTest(test);
    // In a real implementation, fetch variants for this test
    // For now, we'll show a placeholder
    setShowVariantsModal(true);
  };

  const getRiskLevelColor = (level: string | undefined | null) => {
    if (!level || typeof level !== 'string') {
      return '#6b7280';
    }
    switch (level.toLowerCase()) {
      case 'very_high':
      case 'high':
        return '#dc2626';
      case 'elevated':
      case 'moderate':
        return '#f59e0b';
      case 'average':
      case 'low':
        return '#10b981';
      default:
        return '#6b7280';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'melanoma':
        return '#dc2626';
      case 'nmsc':
        return '#f59e0b';
      case 'photosensitivity':
        return '#8b5cf6';
      case 'pharmacogenomics':
        return '#3b82f6';
      case 'pigmentation':
        return '#ec4899';
      default:
        return '#6b7280';
    }
  };

  const renderResultsTab = () => (
    <View style={styles.tabContent}>
      {geneticTests.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="flask-outline" size={64} color="#d1d5db" />
          <Text style={styles.emptyTitle}>No Genetic Tests Yet</Text>
          <Text style={styles.emptySubtitle}>
            Upload a VCF file from your genetic testing provider to see your results here.
          </Text>
          <Pressable
            style={styles.emptyButton}
            onPress={() => setActiveTab('upload')}
          >
            <Text style={styles.emptyButtonText}>Upload VCF File</Text>
          </Pressable>
        </View>
      ) : (
        (geneticTests || []).map((test, index) => (
          <Pressable
            key={`test-${index}-${test.id || test.test_id}`}
            style={styles.testCard}
            onPress={() => viewTestVariants(test)}
          >
            <View style={styles.testHeader}>
              <View>
                <Text style={styles.testId}>{test.test_id}</Text>
                <Text style={styles.testDate}>
                  {new Date(test.test_date || test.created_at).toLocaleDateString()}
                </Text>
              </View>
              <Ionicons name="chevron-forward" size={24} color="#9ca3af" />
            </View>

            {test.lab_name && (
              <Text style={styles.labName}>Lab: {test.lab_name}</Text>
            )}

            <View style={styles.variantSummary}>
              {[
                { id: 'total', count: test.total_variants_analyzed || 0, label: 'Total', style: styles.variantBadge, countStyle: styles.variantCount, labelStyle: styles.variantLabel },
                test.pathogenic_variants > 0 ? { id: 'pathogenic', count: test.pathogenic_variants, label: 'Pathogenic', style: [styles.variantBadge, styles.pathogenicBadge], countStyle: [styles.variantCount, styles.pathogenicText], labelStyle: [styles.variantLabel, styles.pathogenicText] } : null,
                test.likely_pathogenic_variants > 0 ? { id: 'likely-pathogenic', count: test.likely_pathogenic_variants, label: 'Likely Path.', style: [styles.variantBadge, styles.likelyPathogenicBadge], countStyle: [styles.variantCount, styles.likelyPathogenicText], labelStyle: [styles.variantLabel, styles.likelyPathogenicText] } : null,
                test.vus_variants > 0 ? { id: 'vus', count: test.vus_variants, label: 'VUS', style: [styles.variantBadge, styles.vusBadge], countStyle: [styles.variantCount, styles.vusText], labelStyle: [styles.variantLabel, styles.vusText] } : null,
              ].filter(Boolean).map((badge: any) => (
                <View key={badge.id} style={badge.style}>
                  <Text style={badge.countStyle}>{badge.count}</Text>
                  <Text style={badge.labelStyle}>{badge.label}</Text>
                </View>
              ))}
            </View>

            {test.melanoma_risk_level && typeof test.melanoma_risk_level === 'string' && (
              <View style={styles.riskRow}>
                <Text style={styles.riskLabel}>Melanoma Risk:</Text>
                <Text style={[styles.riskValue, { color: getRiskLevelColor(test.melanoma_risk_level) }]}>
                  {test.melanoma_risk_level.replace('_', ' ').toUpperCase()}
                </Text>
              </View>
            )}
          </Pressable>
        ))
      )}
    </View>
  );

  const renderUploadTab = () => (
    <View style={styles.tabContent}>
      <View style={styles.uploadCard}>
        <View style={styles.uploadHeader}>
          <Ionicons name="cloud-upload" size={48} color="#8b5cf6" />
          <Text style={styles.uploadTitle}>Upload VCF File</Text>
          <Text style={styles.uploadSubtitle}>
            Upload your genetic testing results in VCF format to analyze dermatology-relevant variants.
          </Text>
        </View>

        <Pressable style={styles.filePickerButton} onPress={pickVCFFile}>
          {selectedFile ? (
            <View style={styles.selectedFileInfo}>
              <Ionicons name="document" size={24} color="#8b5cf6" />
              <Text style={styles.selectedFileName} numberOfLines={1}>
                {selectedFile.name}
              </Text>
              <Pressable onPress={() => setSelectedFile(null)}>
                <Ionicons name="close-circle" size={24} color="#dc2626" />
              </Pressable>
            </View>
          ) : (
            <View style={styles.filePickerContent}>
              <Ionicons name="add-circle-outline" size={32} color="#8b5cf6" />
              <Text style={styles.filePickerText}>Select VCF File</Text>
              <Text style={styles.filePickerHint}>.vcf or .vcf.gz</Text>
            </View>
          )}
        </Pressable>

        <View style={styles.formField}>
          <Text style={styles.fieldLabel}>Laboratory Name (Optional)</Text>
          <TextInput
            style={styles.textInput}
            value={labName}
            onChangeText={setLabName}
            placeholder="e.g., 23andMe, Color Genomics"
            placeholderTextColor="#9ca3af"
          />
        </View>

        <View style={styles.formField}>
          <Text style={styles.fieldLabel}>Test Type</Text>
          <View style={styles.testTypeOptions}>
            {['panel', 'wes', 'wgs'].map((type) => (
              <Pressable
                key={type}
                style={[styles.testTypeOption, testType === type && styles.testTypeSelected]}
                onPress={() => setTestType(type)}
              >
                <Text style={[styles.testTypeText, testType === type && styles.testTypeTextSelected]}>
                  {type === 'panel' ? 'Gene Panel' : type === 'wes' ? 'Whole Exome' : 'Whole Genome'}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        <Pressable
          style={[styles.uploadButton, (!selectedFile || isUploading) && styles.uploadButtonDisabled]}
          onPress={uploadVCFFile}
          disabled={!selectedFile || isUploading}
        >
          {isUploading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <>
              <Ionicons name="cloud-upload" size={20} color="#fff" />
              <Text style={styles.uploadButtonText}>Upload & Analyze</Text>
            </>
          )}
        </Pressable>

        <View style={styles.infoBox}>
          <Ionicons name="information-circle" size={20} color="#3b82f6" />
          <Text style={styles.infoText}>
            Your VCF file will be analyzed for variants in dermatology-relevant genes including CDKN2A, MC1R, BRAF, PTCH1, and more.
          </Text>
        </View>
      </View>
    </View>
  );

  const renderGenesTab = () => (
    <View style={styles.tabContent}>
      <Text style={styles.sectionTitle}>Dermatology-Relevant Genes</Text>
      <Text style={styles.sectionSubtitle}>
        Gene data from ClinVar/NCBI - automatically updated with latest research
      </Text>

      {(referenceGenes || []).map((gene, index) => (
        <View key={`gene-${index}-${gene.gene_symbol || 'unknown'}`} style={styles.geneCard}>
          <View style={styles.geneHeader}>
            <View style={styles.geneSymbolContainer}>
              <Text style={styles.geneSymbol}>{gene.gene_symbol}</Text>
              {gene.chromosome && (
                <Text style={styles.chromosomeText}>Chr {gene.chromosome}</Text>
              )}
            </View>
            <View style={[styles.categoryBadge, { backgroundColor: getCategoryColor(gene.category) + '20' }]}>
              <Text style={[styles.categoryText, { color: getCategoryColor(gene.category) }]}>
                {gene.category?.replace('_', ' ') || 'other'}
              </Text>
            </View>
          </View>

          <Text style={styles.geneName}>{gene.gene_name}</Text>

          {/* ClinVar Data */}
          {gene.clinvar_pathogenic_count > 0 && (
            <View style={styles.clinvarBadge}>
              <Ionicons name="document-text" size={14} color="#3b82f6" />
              <Text style={styles.clinvarText}>
                {gene.clinvar_pathogenic_count} pathogenic variants in ClinVar
              </Text>
            </View>
          )}

          <View style={styles.geneDetails}>
            {[
              gene.inheritance ? { id: 'inheritance', icon: 'git-branch', color: '#6b7280', text: `Inheritance: ${gene.inheritance}` } : null,
              gene.penetrance ? { id: 'penetrance', icon: 'pulse', color: '#6b7280', text: `Penetrance: ${gene.penetrance}` } : null,
              gene.risk_increase ? { id: 'risk', icon: 'warning', color: '#f59e0b', text: gene.risk_increase } : null,
              gene.melanoma_risk_multiplier > 1 ? { id: 'melanoma', icon: 'trending-up', color: '#dc2626', text: `Melanoma risk: ${gene.melanoma_risk_multiplier}x baseline` } : null,
              gene.bcc_risk_multiplier && gene.bcc_risk_multiplier > 1 ? { id: 'bcc', icon: 'trending-up', color: '#f59e0b', text: `BCC risk: ${gene.bcc_risk_multiplier}x baseline` } : null,
            ].filter(Boolean).map((detail: any) => (
              <View key={detail.id} style={styles.geneDetailRow}>
                <Ionicons name={detail.icon} size={16} color={detail.color} />
                <Text style={styles.geneDetailText}>{detail.text}</Text>
              </View>
            ))}
          </View>

          {/* Associated Conditions */}
          {gene.associated_conditions && gene.associated_conditions.length > 0 && (
            <View style={styles.conditionsContainer}>
              <Text style={styles.conditionsLabel}>Associated conditions:</Text>
              <Text style={styles.conditionsText} numberOfLines={2}>
                {gene.associated_conditions.slice(0, 3).join(', ')}
                {gene.associated_conditions.length > 3 && ` +${gene.associated_conditions.length - 3} more`}
              </Text>
            </View>
          )}

          {gene.acmg_actionable && (
            <View style={styles.acmgBadge}>
              <Ionicons name="checkmark-circle" size={16} color="#10b981" />
              <Text style={styles.acmgText}>ACMG Actionable Gene</Text>
            </View>
          )}

          {gene.screening && (
            <Text style={styles.screeningRecommendation}>
              <Text style={styles.screeningLabel}>Screening: </Text>
              {gene.screening}
            </Text>
          )}

          {/* Data Sources */}
          {gene.sources && gene.sources.length > 0 && (
            <View style={styles.sourcesContainer}>
              <Ionicons name="globe-outline" size={12} color="#9ca3af" />
              <Text style={styles.sourcesText}>
                Sources: {gene.sources.join(', ')}
              </Text>
            </View>
          )}
        </View>
      ))}
    </View>
  );

  const renderSummaryTab = () => (
    <View style={styles.tabContent}>
      {!riskSummary?.has_genetic_data ? (
        <View style={styles.emptyState}>
          <Ionicons name="analytics-outline" size={64} color="#d1d5db" />
          <Text style={styles.emptyTitle}>No Risk Summary Available</Text>
          <Text style={styles.emptySubtitle}>
            Upload genetic test results to see your personalized risk summary.
          </Text>
        </View>
      ) : (
        <>
          {/* Risk Levels */}
          <View style={styles.summaryCard}>
            <Text style={styles.summaryTitle}>Your Genetic Risk Profile</Text>

            <View key="melanoma-risk" style={styles.riskLevelCard}>
              <Text style={styles.riskLevelTitle}>Melanoma Risk</Text>
              <View style={styles.riskLevelContent}>
                <Text style={[styles.riskLevelValue, { color: getRiskLevelColor(riskSummary.melanoma_risk?.level) }]}>
                  {riskSummary.melanoma_risk?.level?.replace('_', ' ').toUpperCase() || 'Unknown'}
                </Text>
                {riskSummary.melanoma_risk?.multiplier > 1 && (
                  <Text style={styles.riskMultiplier}>
                    {riskSummary.melanoma_risk.multiplier}x baseline risk
                  </Text>
                )}
              </View>
            </View>

            <View key="nmsc-risk" style={styles.riskLevelCard}>
              <Text style={styles.riskLevelTitle}>NMSC Risk</Text>
              <View style={styles.riskLevelContent}>
                <Text style={[styles.riskLevelValue, { color: getRiskLevelColor(riskSummary.nmsc_risk?.level) }]}>
                  {riskSummary.nmsc_risk?.level?.replace('_', ' ').toUpperCase() || 'Average'}
                </Text>
              </View>
            </View>
          </View>

          {/* High Risk Genes */}
          {riskSummary.high_risk_genes?.length > 0 && (
            <View style={styles.summaryCard}>
              <Text style={styles.summaryTitle}>High-Risk Genes Detected</Text>
              <View style={styles.geneChips}>
                {riskSummary.high_risk_genes.map((gene, index) => (
                  <View key={`high-risk-${index}-${gene}`} style={styles.highRiskChip}>
                    <Ionicons name="warning" size={14} color="#fff" />
                    <Text style={styles.highRiskChipText}>{gene}</Text>
                  </View>
                ))}
              </View>
            </View>
          )}

          {/* Moderate Risk Genes */}
          {riskSummary.moderate_risk_genes?.length > 0 && (
            <View style={styles.summaryCard}>
              <Text style={styles.summaryTitle}>Moderate-Risk Genes</Text>
              <View style={styles.geneChips}>
                {riskSummary.moderate_risk_genes.map((gene, index) => (
                  <View key={`moderate-risk-${index}-${gene}`} style={styles.moderateRiskChip}>
                    <Text style={styles.moderateRiskChipText}>{gene}</Text>
                  </View>
                ))}
              </View>
            </View>
          )}

          {/* Pharmacogenomic Alerts */}
          {riskSummary.pharmacogenomic_alerts?.length > 0 && (
            <View style={styles.alertCard}>
              <View style={styles.alertHeader}>
                <Ionicons name="medical" size={24} color="#3b82f6" />
                <Text style={styles.alertTitle}>Pharmacogenomic Alerts</Text>
              </View>
              {riskSummary.pharmacogenomic_alerts.map((alert, index) => (
                <View key={`pharma-alert-${index}`} style={styles.alertItem}>
                  <Ionicons name="information-circle" size={16} color="#3b82f6" />
                  <Text style={styles.alertItemText}>{alert}</Text>
                </View>
              ))}
            </View>
          )}

          {/* Recommendations */}
          {riskSummary.recommendations?.length > 0 && (
            <View style={styles.summaryCard}>
              <Text style={styles.summaryTitle}>Personalized Recommendations</Text>
              {riskSummary.recommendations.map((rec, index) => (
                <View key={`recommendation-${index}`} style={styles.recommendationItem}>
                  <Ionicons name="checkmark-circle" size={18} color="#10b981" />
                  <Text style={styles.recommendationText}>{rec}</Text>
                </View>
              ))}
            </View>
          )}
        </>
      )}
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Header */}
      <LinearGradient
        colors={['#8b5cf6', '#7c3aed']}
        style={styles.header}
      >
        <Pressable onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </Pressable>
        <Text style={styles.title}>Genetic Testing</Text>
        <View style={{ width: 40 }} />
      </LinearGradient>

      {/* Tabs */}
      <View style={styles.tabBar}>
        {[
          { key: 'results', label: 'Results', icon: 'list' },
          { key: 'upload', label: 'Upload', icon: 'cloud-upload' },
          { key: 'genes', label: 'Genes', icon: 'flask' },
          { key: 'summary', label: 'Summary', icon: 'analytics' },
        ].map((tab) => (
          <Pressable
            key={tab.key}
            style={[styles.tab, activeTab === tab.key && styles.tabActive]}
            onPress={() => setActiveTab(tab.key as any)}
          >
            <Ionicons
              name={tab.icon as any}
              size={20}
              color={activeTab === tab.key ? '#8b5cf6' : '#9ca3af'}
            />
            <Text style={[styles.tabLabel, activeTab === tab.key && styles.tabLabelActive]}>
              {tab.label}
            </Text>
          </Pressable>
        ))}
      </View>

      {/* Content */}
      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {isLoading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#8b5cf6" />
            <Text style={styles.loadingText}>Loading...</Text>
          </View>
        ) : (
          <>
            {activeTab === 'results' && renderResultsTab()}
            {activeTab === 'upload' && renderUploadTab()}
            {activeTab === 'genes' && renderGenesTab()}
            {activeTab === 'summary' && renderSummaryTab()}
          </>
        )}

        {/* Link to Risk Calculator */}
        <Pressable
          style={styles.riskCalculatorButton}
          onPress={() => router.push('/risk-calculator' as any)}
        >
          <Ionicons name="calculator" size={20} color="#fff" />
          <Text style={styles.riskCalculatorButtonText}>
            View Full Risk Assessment
          </Text>
        </Pressable>

        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 16,
    paddingHorizontal: 16,
  },
  backButton: {
    padding: 8,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 12,
  },
  tabActive: {
    borderBottomWidth: 2,
    borderBottomColor: '#8b5cf6',
  },
  tabLabel: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 4,
  },
  tabLabelActive: {
    color: '#8b5cf6',
    fontWeight: '600',
  },
  content: {
    flex: 1,
  },
  tabContent: {
    padding: 16,
  },
  loadingContainer: {
    padding: 60,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
    color: '#6b7280',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#374151',
    marginTop: 16,
  },
  emptySubtitle: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginTop: 8,
    paddingHorizontal: 32,
  },
  emptyButton: {
    marginTop: 24,
    paddingHorizontal: 24,
    paddingVertical: 12,
    backgroundColor: '#8b5cf6',
    borderRadius: 8,
  },
  emptyButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  testCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  testHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  testId: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  testDate: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  labName: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 12,
  },
  variantSummary: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 12,
  },
  variantBadge: {
    backgroundColor: '#f3f4f6',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    alignItems: 'center',
  },
  pathogenicBadge: {
    backgroundColor: '#fee2e2',
  },
  likelyPathogenicBadge: {
    backgroundColor: '#fef3c7',
  },
  vusBadge: {
    backgroundColor: '#e0e7ff',
  },
  variantCount: {
    fontSize: 16,
    fontWeight: '700',
    color: '#374151',
  },
  pathogenicText: {
    color: '#dc2626',
  },
  likelyPathogenicText: {
    color: '#d97706',
  },
  vusText: {
    color: '#4f46e5',
  },
  variantLabel: {
    fontSize: 10,
    color: '#6b7280',
    marginTop: 2,
  },
  riskRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  riskLabel: {
    fontSize: 13,
    color: '#6b7280',
    marginRight: 8,
  },
  riskValue: {
    fontSize: 13,
    fontWeight: '600',
  },
  uploadCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 3,
  },
  uploadHeader: {
    alignItems: 'center',
    marginBottom: 24,
  },
  uploadTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1f2937',
    marginTop: 12,
  },
  uploadSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginTop: 8,
  },
  filePickerButton: {
    borderWidth: 2,
    borderColor: '#e5e7eb',
    borderStyle: 'dashed',
    borderRadius: 12,
    padding: 24,
    marginBottom: 20,
  },
  filePickerContent: {
    alignItems: 'center',
  },
  filePickerText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#8b5cf6',
    marginTop: 8,
  },
  filePickerHint: {
    fontSize: 12,
    color: '#9ca3af',
    marginTop: 4,
  },
  selectedFileInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  selectedFileName: {
    flex: 1,
    fontSize: 14,
    color: '#374151',
  },
  formField: {
    marginBottom: 16,
  },
  fieldLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    color: '#1f2937',
  },
  testTypeOptions: {
    flexDirection: 'row',
    gap: 8,
  },
  testTypeOption: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    alignItems: 'center',
  },
  testTypeSelected: {
    borderColor: '#8b5cf6',
    backgroundColor: '#f5f3ff',
  },
  testTypeText: {
    fontSize: 12,
    color: '#6b7280',
  },
  testTypeTextSelected: {
    color: '#8b5cf6',
    fontWeight: '600',
  },
  uploadButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#8b5cf6',
    padding: 16,
    borderRadius: 12,
    gap: 8,
    marginTop: 8,
  },
  uploadButtonDisabled: {
    backgroundColor: '#d1d5db',
  },
  uploadButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  infoBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#eff6ff',
    padding: 12,
    borderRadius: 8,
    marginTop: 16,
    gap: 8,
  },
  infoText: {
    flex: 1,
    fontSize: 12,
    color: '#1e40af',
    lineHeight: 18,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 4,
  },
  sectionSubtitle: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 16,
  },
  geneCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  geneHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  geneSymbol: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
  },
  geneSymbolContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  chromosomeText: {
    fontSize: 11,
    color: '#9ca3af',
    backgroundColor: '#f3f4f6',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  clinvarBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#dbeafe',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
    alignSelf: 'flex-start',
    marginBottom: 12,
  },
  clinvarText: {
    fontSize: 12,
    fontWeight: '500',
    color: '#2563eb',
  },
  conditionsContainer: {
    marginBottom: 12,
  },
  conditionsLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 4,
  },
  conditionsText: {
    fontSize: 12,
    color: '#6b7280',
    lineHeight: 16,
  },
  sourcesContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  sourcesText: {
    fontSize: 10,
    color: '#9ca3af',
  },
  categoryBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  categoryText: {
    fontSize: 11,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  geneName: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 12,
  },
  geneDetails: {
    gap: 6,
    marginBottom: 12,
  },
  geneDetailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  geneDetailText: {
    fontSize: 13,
    color: '#374151',
  },
  acmgBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#d1fae5',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
    alignSelf: 'flex-start',
    marginBottom: 12,
  },
  acmgText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#059669',
  },
  screeningRecommendation: {
    fontSize: 12,
    color: '#6b7280',
    lineHeight: 18,
  },
  screeningLabel: {
    fontWeight: '600',
    color: '#374151',
  },
  summaryCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  summaryTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 16,
  },
  riskLevelCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  riskLevelTitle: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 4,
  },
  riskLevelContent: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 12,
  },
  riskLevelValue: {
    fontSize: 20,
    fontWeight: '700',
  },
  riskMultiplier: {
    fontSize: 12,
    color: '#6b7280',
  },
  geneChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  highRiskChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#dc2626',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    gap: 6,
  },
  highRiskChipText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },
  moderateRiskChip: {
    backgroundColor: '#fef3c7',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  moderateRiskChipText: {
    color: '#d97706',
    fontSize: 13,
    fontWeight: '600',
  },
  alertCard: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
  },
  alertHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 12,
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e40af',
  },
  alertItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    marginBottom: 8,
  },
  alertItemText: {
    flex: 1,
    fontSize: 13,
    color: '#1e40af',
    lineHeight: 18,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: 12,
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
  },
  riskCalculatorButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#7c3aed',
    marginHorizontal: 16,
    marginTop: 8,
    padding: 16,
    borderRadius: 12,
    gap: 8,
  },
  riskCalculatorButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
