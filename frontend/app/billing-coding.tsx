import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  ActivityIndicator,
  TextInput,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';

interface BillingCode {
  code_type: string; // 'CPT', 'ICD-10', 'HCPCS'
  code: string;
  description: string;
  category: string;
  typical_reimbursement?: number;
}

interface BillingRecord {
  id: number;
  analysis_id: number;
  patient_name?: string;
  diagnosis: string;
  procedure_date: string;
  cpt_codes: BillingCode[];
  icd10_codes: BillingCode[];
  hcpcs_codes?: BillingCode[];
  total_charges: number;
  estimated_reimbursement: number;
  status: string; // 'draft', 'submitted', 'approved', 'rejected'
  claim_number?: string;
  created_at: string;
}

export default function BillingCodingScreen() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const { t } = useTranslation();

  const [billingRecords, setBillingRecords] = useState<BillingRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [showCodeLookup, setShowCodeLookup] = useState(false);
  const [lookupQuery, setLookupQuery] = useState('');
  const [lookupResults, setLookupResults] = useState<BillingCode[]>([]);

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    } else {
      loadBillingRecords();
    }
  }, [isAuthenticated, filterStatus]);

  const loadBillingRecords = async () => {
    setIsLoading(true);

    if (!user?.token) {
      console.error('No authentication token available');
      setIsLoading(false);
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/billing/records?status=${filterStatus}`,
        {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${user.token}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setBillingRecords(data.records || []);
      } else {
        console.log('Failed to load billing records. Status:', response.status);
        setBillingRecords([]);
      }
    } catch (error) {
      console.error('Error loading billing records:', error);
      setBillingRecords([]);
    } finally {
      setIsLoading(false);
    }
  };

  const searchCodes = async () => {
    if (!lookupQuery.trim() || !user?.token) return;

    try {
      const response = await fetch(
        `${API_BASE_URL}/billing/codes/search?query=${encodeURIComponent(lookupQuery)}`,
        {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${user.token}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setLookupResults(data.codes || []);
      }
    } catch (error) {
      console.error('Error searching codes:', error);
    }
  };

  const generateBillingForAnalysis = async (analysisId: number) => {
    if (!user?.token) return;

    try {
      const response = await fetch(
        `${API_BASE_URL}/billing/generate/${analysisId}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${user.token}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (response.ok) {
        Alert.alert(
          t('billingCoding.success'),
          t('billingCoding.billingGenerated')
        );
        loadBillingRecords();
      } else {
        Alert.alert(
          t('common.error'),
          t('billingCoding.generationFailed')
        );
      }
    } catch (error) {
      console.error('Error generating billing:', error);
      Alert.alert(t('common.error'), t('billingCoding.generationFailed'));
    }
  };

  const exportToCMS1500 = async (recordId: number) => {
    if (!user?.token) return;

    try {
      const response = await fetch(
        `${API_BASE_URL}/billing/export/cms1500/${recordId}`,
        {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${user.token}`,
          },
        }
      );

      if (response.ok) {
        Alert.alert(
          t('billingCoding.success'),
          t('billingCoding.exportSuccess')
        );
      }
    } catch (error) {
      console.error('Error exporting to CMS-1500:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved':
        return '#28a745';
      case 'submitted':
        return '#17a2b8';
      case 'rejected':
        return '#dc3545';
      default:
        return '#6c757d';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'approved':
        return 'checkmark-circle';
      case 'submitted':
        return 'paper-plane';
      case 'rejected':
        return 'close-circle';
      default:
        return 'document-text';
    }
  };

  const renderCodeLookup = () => {
    if (!showCodeLookup) return null;

    return (
      <View style={styles.lookupContainer}>
        <View style={styles.lookupHeader}>
          <Text style={styles.lookupTitle}>{t('billingCoding.codeLookup')}</Text>
          <Pressable onPress={() => setShowCodeLookup(false)}>
            <Ionicons name="close" size={24} color="#2c5282" />
          </Pressable>
        </View>

        <View style={styles.searchContainer}>
          <TextInput
            style={styles.searchInput}
            placeholder={t('billingCoding.searchPlaceholder')}
            value={lookupQuery}
            onChangeText={setLookupQuery}
            onSubmitEditing={searchCodes}
          />
          <Pressable style={styles.searchButton} onPress={searchCodes}>
            <Ionicons name="search" size={20} color="#fff" />
          </Pressable>
        </View>

        <ScrollView style={styles.lookupResults}>
          {lookupResults.map((code, index) => (
            <View key={index} style={styles.codeCard}>
              <View style={styles.codeHeader}>
                <Text style={styles.codeType}>{code.code_type}</Text>
                <Text style={styles.codeNumber}>{code.code}</Text>
              </View>
              <Text style={styles.codeDescription}>{code.description}</Text>
              <Text style={styles.codeCategory}>{code.category}</Text>
              {code.typical_reimbursement && (
                <Text style={styles.codeReimbursement}>
                  {t('billingCoding.typicalReimbursement')}: ${code.typical_reimbursement.toFixed(2)}
                </Text>
              )}
            </View>
          ))}
        </ScrollView>
      </View>
    );
  };

  const renderBillingRecord = (record: BillingRecord) => {
    return (
      <View key={record.id} style={styles.recordCard}>
        <View style={styles.recordHeader}>
          <View style={styles.recordTitleContainer}>
            <Text style={styles.recordDiagnosis}>{record.diagnosis}</Text>
            {record.claim_number && (
              <Text style={styles.claimNumber}>
                {t('billingCoding.claim')}: {record.claim_number}
              </Text>
            )}
          </View>
          <View style={[styles.statusBadge, { backgroundColor: getStatusColor(record.status) }]}>
            <Ionicons
              name={getStatusIcon(record.status) as any}
              size={16}
              color="#fff"
              style={styles.statusIcon}
            />
            <Text style={styles.statusText}>
              {t(`billingCoding.status.${record.status}`)}
            </Text>
          </View>
        </View>

        <View style={styles.recordDetails}>
          <View style={styles.detailRow}>
            <Ionicons name="calendar-outline" size={16} color="#4a5568" />
            <Text style={styles.detailText}>
              {new Date(record.procedure_date).toLocaleDateString()}
            </Text>
          </View>
          {record.patient_name && (
            <View style={styles.detailRow}>
              <Ionicons name="person-outline" size={16} color="#4a5568" />
              <Text style={styles.detailText}>{record.patient_name}</Text>
            </View>
          )}
        </View>

        <View style={styles.codesSection}>
          <Text style={styles.codesSectionTitle}>{t('billingCoding.codes')}:</Text>

          <View style={styles.codeGroup}>
            <Text style={styles.codeGroupLabel}>CPT:</Text>
            {record.cpt_codes.map((code, idx) => (
              <View key={idx} style={styles.codeTag}>
                <Text style={styles.codeTagText}>{code.code}</Text>
              </View>
            ))}
          </View>

          <View style={styles.codeGroup}>
            <Text style={styles.codeGroupLabel}>ICD-10:</Text>
            {record.icd10_codes.map((code, idx) => (
              <View key={idx} style={styles.codeTag}>
                <Text style={styles.codeTagText}>{code.code}</Text>
              </View>
            ))}
          </View>
        </View>

        <View style={styles.financials}>
          <View style={styles.financialRow}>
            <Text style={styles.financialLabel}>{t('billingCoding.totalCharges')}:</Text>
            <Text style={styles.financialValue}>${record.total_charges.toFixed(2)}</Text>
          </View>
          <View style={styles.financialRow}>
            <Text style={styles.financialLabel}>{t('billingCoding.estimatedReimbursement')}:</Text>
            <Text style={[styles.financialValue, styles.reimbursementValue]}>
              ${record.estimated_reimbursement.toFixed(2)}
            </Text>
          </View>
        </View>

        <View style={styles.recordActions}>
          <Pressable
            style={styles.actionButton}
            onPress={() => exportToCMS1500(record.id)}
          >
            <Ionicons name="download-outline" size={18} color="#4299e1" />
            <Text style={styles.actionButtonText}>{t('billingCoding.exportCMS1500')}</Text>
          </Pressable>
          <Pressable
            style={styles.actionButton}
            onPress={() => router.push(`/analysis/${record.analysis_id}` as any)}
          >
            <Ionicons name="eye-outline" size={18} color="#4299e1" />
            <Text style={styles.actionButtonText}>{t('billingCoding.viewAnalysis')}</Text>
          </Pressable>
        </View>
      </View>
    );
  };

  const filteredRecords = billingRecords.filter(record =>
    record.diagnosis.toLowerCase().includes(searchQuery.toLowerCase()) ||
    record.claim_number?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.background}
      />

      {/* Header */}
      <View style={styles.header}>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Text style={styles.backButtonText}>← {t('common.back')}</Text>
        </Pressable>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>💳 {t('billingCoding.title')}</Text>
          <Text style={styles.headerSubtitle}>{t('billingCoding.subtitle')}</Text>
        </View>
      </View>

      <ScrollView style={styles.content}>
        {/* Quick Actions */}
        <View style={styles.quickActions}>
          <Pressable
            style={styles.quickActionButton}
            onPress={() => setShowCodeLookup(!showCodeLookup)}
          >
            <Ionicons name="search" size={24} color="#4299e1" />
            <Text style={styles.quickActionText}>{t('billingCoding.codeLookup')}</Text>
          </Pressable>
          <Pressable
            style={styles.quickActionButton}
            onPress={() => router.push('/history')}
          >
            <Ionicons name="add-circle" size={24} color="#4299e1" />
            <Text style={styles.quickActionText}>{t('billingCoding.newBilling')}</Text>
          </Pressable>
        </View>

        {/* Code Lookup Modal */}
        {renderCodeLookup()}

        {/* Search and Filter */}
        <View style={styles.searchFilterContainer}>
          <View style={styles.searchBox}>
            <Ionicons name="search" size={20} color="#4a5568" />
            <TextInput
              style={styles.searchBoxInput}
              placeholder={t('billingCoding.searchRecords')}
              value={searchQuery}
              onChangeText={setSearchQuery}
            />
          </View>

          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterButtons}>
            {['all', 'draft', 'submitted', 'approved', 'rejected'].map((status) => (
              <Pressable
                key={status}
                style={[
                  styles.filterButton,
                  filterStatus === status && styles.filterButtonActive,
                ]}
                onPress={() => setFilterStatus(status)}
              >
                <Text
                  style={[
                    styles.filterButtonText,
                    filterStatus === status && styles.filterButtonTextActive,
                  ]}
                >
                  {t(`billingCoding.filter.${status}`)}
                </Text>
              </Pressable>
            ))}
          </ScrollView>
        </View>

        {/* Billing Records */}
        {isLoading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#4299e1" />
            <Text style={styles.loadingText}>{t('billingCoding.loading')}</Text>
          </View>
        ) : filteredRecords.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons name="receipt-outline" size={64} color="#cbd5e0" />
            <Text style={styles.emptyStateTitle}>{t('billingCoding.noRecords')}</Text>
            <Text style={styles.emptyStateSubtext}>{t('billingCoding.noRecordsSubtext')}</Text>
          </View>
        ) : (
          <View style={styles.recordsList}>
            {filteredRecords.map(renderBillingRecord)}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  background: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.2)',
  },
  backButton: {
    backgroundColor: 'rgba(66, 153, 225, 0.9)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 15,
  },
  backButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  headerContent: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#4a5568',
    marginTop: 4,
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  quickActions: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  quickActionButton: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  quickActionText: {
    marginTop: 8,
    fontSize: 13,
    color: '#2c5282',
    fontWeight: '600',
  },
  searchFilterContainer: {
    marginBottom: 20,
  },
  searchBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginBottom: 12,
  },
  searchBoxInput: {
    flex: 1,
    marginLeft: 8,
    fontSize: 15,
    color: '#2d3748',
  },
  filterButtons: {
    flexDirection: 'row',
  },
  filterButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  filterButtonActive: {
    backgroundColor: '#4299e1',
    borderColor: '#4299e1',
  },
  filterButtonText: {
    fontSize: 13,
    color: '#4a5568',
    fontWeight: '600',
  },
  filterButtonTextActive: {
    color: '#fff',
  },
  loadingContainer: {
    paddingVertical: 60,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#4a5568',
  },
  emptyState: {
    paddingVertical: 80,
    alignItems: 'center',
  },
  emptyStateTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#4a5568',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#718096',
    textAlign: 'center',
    paddingHorizontal: 40,
  },
  recordsList: {
    paddingBottom: 20,
  },
  recordCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  recordHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  recordTitleContainer: {
    flex: 1,
    marginRight: 12,
  },
  recordDiagnosis: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2d3748',
    marginBottom: 4,
  },
  claimNumber: {
    fontSize: 12,
    color: '#718096',
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
  },
  statusIcon: {
    marginRight: 4,
  },
  statusText: {
    fontSize: 11,
    color: '#fff',
    fontWeight: 'bold',
  },
  recordDetails: {
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  detailText: {
    fontSize: 13,
    color: '#4a5568',
    marginLeft: 8,
  },
  codesSection: {
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    paddingTop: 12,
    marginBottom: 12,
  },
  codesSectionTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#2c5282',
    marginBottom: 8,
  },
  codeGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    marginBottom: 8,
  },
  codeGroupLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#4a5568',
    marginRight: 8,
    minWidth: 50,
  },
  codeTag: {
    backgroundColor: '#e6f2ff',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    marginRight: 6,
    marginBottom: 4,
  },
  codeTagText: {
    fontSize: 11,
    color: '#2c5282',
    fontWeight: '600',
  },
  financials: {
    backgroundColor: '#f7fafc',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  financialRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  financialLabel: {
    fontSize: 13,
    color: '#4a5568',
  },
  financialValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2d3748',
  },
  reimbursementValue: {
    color: '#28a745',
  },
  recordActions: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#e6f2ff',
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  actionButtonText: {
    fontSize: 13,
    color: '#4299e1',
    fontWeight: '600',
    marginLeft: 6,
  },
  lookupContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.98)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 5,
  },
  lookupHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  lookupTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  searchContainer: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  searchInput: {
    flex: 1,
    backgroundColor: '#f7fafc',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
    marginRight: 8,
  },
  searchButton: {
    backgroundColor: '#4299e1',
    borderRadius: 8,
    width: 44,
    justifyContent: 'center',
    alignItems: 'center',
  },
  lookupResults: {
    maxHeight: 300,
  },
  codeCard: {
    backgroundColor: '#f7fafc',
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
  },
  codeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  codeType: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#4299e1',
    backgroundColor: '#e6f2ff',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  codeNumber: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#2d3748',
  },
  codeDescription: {
    fontSize: 13,
    color: '#2d3748',
    marginBottom: 6,
  },
  codeCategory: {
    fontSize: 11,
    color: '#718096',
    fontStyle: 'italic',
    marginBottom: 4,
  },
  codeReimbursement: {
    fontSize: 12,
    color: '#28a745',
    fontWeight: '600',
  },
});
