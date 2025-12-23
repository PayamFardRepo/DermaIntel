import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Dimensions,
  Switch,
  Clipboard,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { API_BASE_URL } from '../config';

const { width } = Dimensions.get('window');

interface ICD10Code {
  code: string;
  description: string;
  category: string;
  confidence?: string;
  location_match?: boolean;
}

interface CPTCode {
  code: string;
  description: string;
  category: string;
  rvu?: number;
  confidence?: string;
  recommendation?: string;
  note?: string;
}

interface CodingSuggestion {
  diagnosis_input: string;
  matched_condition: string | null;
  icd10_suggestions: ICD10Code[];
  cpt_suggestions: CPTCode[];
  documentation_tips: string[];
  coding_summary: {
    primary_icd10: string | null;
    primary_cpt: string | null;
    total_rvu: number;
    estimated_medicare_reimbursement: string;
  };
  disclaimer: string;
}

type TabType = 'suggest' | 'search' | 'history';

export default function AutoCodingScreen() {
  const router = useRouter();
  const { user } = useAuth();

  const [activeTab, setActiveTab] = useState<TabType>('suggest');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Suggestion form state
  const [diagnosis, setDiagnosis] = useState('');
  const [clinicalNotes, setClinicalNotes] = useState('');
  const [bodyLocation, setBodyLocation] = useState('');
  const [lesionSize, setLesionSize] = useState('');
  const [isNewPatient, setIsNewPatient] = useState(false);
  const [procedure, setProcedure] = useState('');
  const [suggestion, setSuggestion] = useState<CodingSuggestion | null>(null);

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchType, setSearchType] = useState<'icd10' | 'cpt'>('icd10');
  const [searchResults, setSearchResults] = useState<any[]>([]);

  // History state
  const [codingHistory, setCodingHistory] = useState<CodingSuggestion[]>([]);

  const commonDiagnoses = [
    'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma',
    'Nevus', 'Actinic Keratosis', 'Seborrheic Keratosis',
    'Eczema', 'Psoriasis', 'Acne', 'Rosacea'
  ];

  const commonProcedures = [
    'Punch biopsy', 'Shave biopsy', 'Excision',
    'Cryotherapy', 'Dermoscopy', 'None'
  ];

  const bodyLocations = [
    'Face', 'Scalp', 'Neck', 'Chest', 'Back',
    'Arm', 'Hand', 'Leg', 'Foot', 'Trunk'
  ];

  const getSuggestions = async () => {
    if (!diagnosis.trim()) {
      setError('Please enter a diagnosis');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('diagnosis', diagnosis);
      if (clinicalNotes) formData.append('clinical_notes', clinicalNotes);
      if (bodyLocation) formData.append('body_location', bodyLocation);
      if (lesionSize) formData.append('lesion_size_cm', lesionSize);
      formData.append('is_new_patient', isNewPatient.toString());
      if (procedure && procedure !== 'None') formData.append('procedure_performed', procedure);

      const response = await fetch(`${API_BASE_URL}/auto-coding/suggest`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${user?.token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to get suggestions');
      }

      const data = await response.json();
      setSuggestion(data);

      // Add to history
      setCodingHistory(prev => [data, ...prev.slice(0, 9)]);
    } catch (err: any) {
      setError(err.message || 'Failed to get code suggestions');
    } finally {
      setIsLoading(false);
    }
  };

  const searchCodes = async () => {
    if (!searchQuery.trim() || searchQuery.length < 2) {
      setError('Enter at least 2 characters to search');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const endpoint = searchType === 'icd10' ? 'icd10-search' : 'cpt-search';
      const response = await fetch(
        `${API_BASE_URL}/auto-coding/${endpoint}?query=${encodeURIComponent(searchQuery)}`,
        {
          headers: {
            'Authorization': `Bearer ${user?.token}`,
          },
        }
      );

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Search failed');
      }

      const data = await response.json();
      setSearchResults(data.results);
    } catch (err: any) {
      setError(err.message || 'Search failed');
    } finally {
      setIsLoading(false);
    }
  };

  const copyCode = (code: string) => {
    Clipboard.setString(code);
    Alert.alert('Copied', `Code ${code} copied to clipboard`);
  };

  const getConfidenceColor = (confidence?: string) => {
    switch (confidence) {
      case 'High': return '#10b981';
      case 'Medium': return '#f59e0b';
      case 'Low': return '#ef4444';
      default: return '#64748b';
    }
  };

  const renderSuggestTab = () => (
    <View>
      {/* Diagnosis Input */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Diagnosis</Text>
        <TextInput
          style={styles.textInput}
          placeholder="Enter diagnosis (e.g., Melanoma, BCC, Psoriasis)"
          placeholderTextColor="#64748b"
          value={diagnosis}
          onChangeText={setDiagnosis}
        />
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.quickPicks}>
          {commonDiagnoses.map((dx) => (
            <TouchableOpacity
              key={dx}
              style={[styles.quickPickButton, diagnosis === dx && styles.quickPickActive]}
              onPress={() => setDiagnosis(dx)}
            >
              <Text style={[styles.quickPickText, diagnosis === dx && styles.quickPickTextActive]}>
                {dx}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      {/* Body Location */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Body Location (Optional)</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.quickPicks}>
          {bodyLocations.map((loc) => (
            <TouchableOpacity
              key={loc}
              style={[styles.quickPickButton, bodyLocation === loc && styles.quickPickActive]}
              onPress={() => setBodyLocation(bodyLocation === loc ? '' : loc)}
            >
              <Text style={[styles.quickPickText, bodyLocation === loc && styles.quickPickTextActive]}>
                {loc}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      {/* Procedure */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Procedure Performed (Optional)</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.quickPicks}>
          {commonProcedures.map((proc) => (
            <TouchableOpacity
              key={proc}
              style={[styles.quickPickButton, procedure === proc && styles.quickPickActive]}
              onPress={() => setProcedure(procedure === proc ? '' : proc)}
            >
              <Text style={[styles.quickPickText, procedure === proc && styles.quickPickTextActive]}>
                {proc}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      {/* Lesion Size & Patient Type */}
      <View style={styles.row}>
        <View style={[styles.section, { flex: 1, marginRight: 8 }]}>
          <Text style={styles.sectionTitle}>Lesion Size (cm)</Text>
          <TextInput
            style={styles.textInput}
            placeholder="e.g., 1.5"
            placeholderTextColor="#64748b"
            value={lesionSize}
            onChangeText={setLesionSize}
            keyboardType="decimal-pad"
          />
        </View>
        <View style={[styles.section, { flex: 1, marginLeft: 8 }]}>
          <Text style={styles.sectionTitle}>New Patient?</Text>
          <View style={styles.switchRow}>
            <Text style={styles.switchLabel}>{isNewPatient ? 'Yes' : 'No'}</Text>
            <Switch
              value={isNewPatient}
              onValueChange={setIsNewPatient}
              trackColor={{ false: '#334155', true: '#0ea5e9' }}
              thumbColor="#fff"
            />
          </View>
        </View>
      </View>

      {/* Clinical Notes */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Clinical Notes (Optional)</Text>
        <TextInput
          style={[styles.textInput, styles.textArea]}
          placeholder="Additional clinical context..."
          placeholderTextColor="#64748b"
          value={clinicalNotes}
          onChangeText={setClinicalNotes}
          multiline
          numberOfLines={3}
        />
      </View>

      {/* Submit Button */}
      <TouchableOpacity
        style={[styles.submitButton, isLoading && styles.submitButtonDisabled]}
        onPress={getSuggestions}
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="code-working" size={20} color="#fff" />
            <Text style={styles.submitButtonText}>Get Code Suggestions</Text>
          </>
        )}
      </TouchableOpacity>

      {/* Error */}
      {error && (
        <View style={styles.errorBox}>
          <Ionicons name="alert-circle" size={20} color="#ef4444" />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {/* Results */}
      {suggestion && (
        <View style={styles.results}>
          {/* Summary Card */}
          <View style={styles.summaryCard}>
            <Text style={styles.summaryTitle}>Coding Summary</Text>
            <View style={styles.summaryGrid}>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>Primary ICD-10</Text>
                <TouchableOpacity onPress={() => copyCode(suggestion.coding_summary.primary_icd10 || '')}>
                  <Text style={styles.summaryCode}>{suggestion.coding_summary.primary_icd10 || 'N/A'}</Text>
                </TouchableOpacity>
              </View>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>Primary CPT</Text>
                <TouchableOpacity onPress={() => copyCode(suggestion.coding_summary.primary_cpt || '')}>
                  <Text style={styles.summaryCode}>{suggestion.coding_summary.primary_cpt || 'N/A'}</Text>
                </TouchableOpacity>
              </View>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>Total RVU</Text>
                <Text style={styles.summaryValue}>{suggestion.coding_summary.total_rvu}</Text>
              </View>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>Est. Medicare</Text>
                <Text style={styles.summaryValue}>{suggestion.coding_summary.estimated_medicare_reimbursement}</Text>
              </View>
            </View>
          </View>

          {/* ICD-10 Suggestions */}
          <View style={styles.codeSection}>
            <Text style={styles.codeSectionTitle}>
              <Ionicons name="medical" size={16} color="#0ea5e9" /> ICD-10 Codes
            </Text>
            {suggestion.icd10_suggestions.map((code, i) => (
              <TouchableOpacity
                key={i}
                style={styles.codeCard}
                onPress={() => copyCode(code.code)}
              >
                <View style={styles.codeHeader}>
                  <Text style={styles.codeText}>{code.code}</Text>
                  <View style={[styles.confidenceBadge, { backgroundColor: getConfidenceColor(code.confidence) + '20' }]}>
                    <Text style={[styles.confidenceText, { color: getConfidenceColor(code.confidence) }]}>
                      {code.confidence || 'N/A'}
                    </Text>
                  </View>
                </View>
                <Text style={styles.codeDescription}>{code.description}</Text>
                <View style={styles.codeFooter}>
                  <Text style={styles.codeCategory}>{code.category}</Text>
                  <Ionicons name="copy-outline" size={16} color="#64748b" />
                </View>
              </TouchableOpacity>
            ))}
          </View>

          {/* CPT Suggestions */}
          <View style={styles.codeSection}>
            <Text style={styles.codeSectionTitle}>
              <Ionicons name="construct" size={16} color="#10b981" /> CPT Codes
            </Text>
            {suggestion.cpt_suggestions.map((code, i) => (
              <TouchableOpacity
                key={i}
                style={styles.codeCard}
                onPress={() => copyCode(code.code)}
              >
                <View style={styles.codeHeader}>
                  <Text style={styles.codeText}>{code.code}</Text>
                  {code.rvu && (
                    <Text style={styles.rvuText}>{code.rvu} RVU</Text>
                  )}
                </View>
                <Text style={styles.codeDescription}>{code.description}</Text>
                {code.recommendation && (
                  <Text style={styles.codeNote}>{code.recommendation}</Text>
                )}
                {code.note && (
                  <Text style={styles.codeNote}>{code.note}</Text>
                )}
                <View style={styles.codeFooter}>
                  <Text style={styles.codeCategory}>{code.category}</Text>
                  <Ionicons name="copy-outline" size={16} color="#64748b" />
                </View>
              </TouchableOpacity>
            ))}
          </View>

          {/* Documentation Tips */}
          {suggestion.documentation_tips.length > 0 && (
            <View style={styles.tipsSection}>
              <Text style={styles.tipsSectionTitle}>
                <Ionicons name="bulb" size={16} color="#f59e0b" /> Documentation Tips
              </Text>
              {suggestion.documentation_tips.map((tip, i) => (
                <View key={i} style={styles.tipItem}>
                  <Ionicons name="checkmark-circle" size={16} color="#f59e0b" />
                  <Text style={styles.tipText}>{tip}</Text>
                </View>
              ))}
            </View>
          )}

          {/* Disclaimer */}
          <View style={styles.disclaimer}>
            <Ionicons name="information-circle" size={16} color="#64748b" />
            <Text style={styles.disclaimerText}>{suggestion.disclaimer}</Text>
          </View>
        </View>
      )}
    </View>
  );

  const renderSearchTab = () => (
    <View>
      {/* Search Type Toggle */}
      <View style={styles.toggleContainer}>
        <TouchableOpacity
          style={[styles.toggleButton, searchType === 'icd10' && styles.toggleActive]}
          onPress={() => { setSearchType('icd10'); setSearchResults([]); }}
        >
          <Text style={[styles.toggleText, searchType === 'icd10' && styles.toggleTextActive]}>
            ICD-10
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.toggleButton, searchType === 'cpt' && styles.toggleActive]}
          onPress={() => { setSearchType('cpt'); setSearchResults([]); }}
        >
          <Text style={[styles.toggleText, searchType === 'cpt' && styles.toggleTextActive]}>
            CPT
          </Text>
        </TouchableOpacity>
      </View>

      {/* Search Input */}
      <View style={styles.searchContainer}>
        <TextInput
          style={styles.searchInput}
          placeholder={`Search ${searchType === 'icd10' ? 'ICD-10' : 'CPT'} codes...`}
          placeholderTextColor="#64748b"
          value={searchQuery}
          onChangeText={setSearchQuery}
          onSubmitEditing={searchCodes}
        />
        <TouchableOpacity style={styles.searchButton} onPress={searchCodes}>
          <Ionicons name="search" size={20} color="#fff" />
        </TouchableOpacity>
      </View>

      {/* Search Results */}
      {isLoading ? (
        <ActivityIndicator size="large" color="#0ea5e9" style={{ marginTop: 40 }} />
      ) : (
        <View style={styles.searchResults}>
          {searchResults.map((result, i) => (
            <TouchableOpacity
              key={i}
              style={styles.searchResultCard}
              onPress={() => copyCode(result.code)}
            >
              <View style={styles.codeHeader}>
                <Text style={styles.codeText}>{result.code}</Text>
                {result.rvu && <Text style={styles.rvuText}>{result.rvu} RVU</Text>}
              </View>
              <Text style={styles.codeDescription}>{result.description}</Text>
              <Text style={styles.codeCategory}>{result.category}</Text>
            </TouchableOpacity>
          ))}
          {searchResults.length === 0 && searchQuery.length >= 2 && !isLoading && (
            <Text style={styles.noResults}>No results found</Text>
          )}
        </View>
      )}
    </View>
  );

  const renderHistoryTab = () => (
    <View>
      {codingHistory.length === 0 ? (
        <View style={styles.emptyHistory}>
          <Ionicons name="time-outline" size={48} color="#64748b" />
          <Text style={styles.emptyHistoryText}>No coding history yet</Text>
          <Text style={styles.emptyHistorySubtext}>
            Your recent code suggestions will appear here
          </Text>
        </View>
      ) : (
        codingHistory.map((item, i) => (
          <View key={i} style={styles.historyCard}>
            <View style={styles.historyHeader}>
              <Text style={styles.historyDiagnosis}>{item.diagnosis_input}</Text>
              <Text style={styles.historyCondition}>{item.matched_condition || 'Unmatched'}</Text>
            </View>
            <View style={styles.historyCodesRow}>
              <View style={styles.historyCodeItem}>
                <Text style={styles.historyCodeLabel}>ICD-10</Text>
                <Text style={styles.historyCodeValue}>{item.coding_summary.primary_icd10 || '-'}</Text>
              </View>
              <View style={styles.historyCodeItem}>
                <Text style={styles.historyCodeLabel}>CPT</Text>
                <Text style={styles.historyCodeValue}>{item.coding_summary.primary_cpt || '-'}</Text>
              </View>
              <View style={styles.historyCodeItem}>
                <Text style={styles.historyCodeLabel}>Est.</Text>
                <Text style={styles.historyCodeValue}>{item.coding_summary.estimated_medicare_reimbursement}</Text>
              </View>
            </View>
          </View>
        ))
      )}
    </View>
  );

  return (
    <View style={styles.container}>
      <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>Auto-Coding Engine</Text>
          <Text style={styles.headerSubtitle}>
            ICD-10 & CPT code suggestions from AI diagnosis
          </Text>
        </View>
      </LinearGradient>

      {/* Tabs */}
      <View style={styles.tabBar}>
        {[
          { id: 'suggest' as TabType, label: 'Suggest', icon: 'flash' },
          { id: 'search' as TabType, label: 'Search', icon: 'search' },
          { id: 'history' as TabType, label: 'History', icon: 'time' },
        ].map((tab) => (
          <TouchableOpacity
            key={tab.id}
            style={[styles.tab, activeTab === tab.id && styles.tabActive]}
            onPress={() => setActiveTab(tab.id)}
          >
            <Ionicons
              name={tab.icon as any}
              size={18}
              color={activeTab === tab.id ? '#0ea5e9' : '#64748b'}
            />
            <Text style={[styles.tabText, activeTab === tab.id && styles.tabTextActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {activeTab === 'suggest' && renderSuggestTab()}
        {activeTab === 'search' && renderSearchTab()}
        {activeTab === 'history' && renderHistoryTab()}
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
  tabBar: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.05)',
    marginHorizontal: 20,
    marginTop: -10,
    borderRadius: 12,
    padding: 4,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderRadius: 8,
    gap: 6,
  },
  tabActive: {
    backgroundColor: 'rgba(14,165,233,0.2)',
  },
  tabText: {
    color: '#64748b',
    fontSize: 14,
    fontWeight: '500',
  },
  tabTextActive: {
    color: '#0ea5e9',
  },
  content: {
    flex: 1,
    padding: 20,
  },
  section: {
    marginBottom: 16,
  },
  sectionTitle: {
    color: '#94a3b8',
    fontSize: 13,
    fontWeight: '600',
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  textInput: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 10,
    padding: 14,
    color: '#fff',
    fontSize: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  textArea: {
    height: 80,
    textAlignVertical: 'top',
  },
  quickPicks: {
    marginTop: 8,
  },
  quickPickButton: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  quickPickActive: {
    backgroundColor: 'rgba(14,165,233,0.2)',
    borderColor: '#0ea5e9',
  },
  quickPickText: {
    color: '#94a3b8',
    fontSize: 13,
  },
  quickPickTextActive: {
    color: '#0ea5e9',
  },
  row: {
    flexDirection: 'row',
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 10,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  switchLabel: {
    color: '#fff',
    fontSize: 16,
  },
  submitButton: {
    backgroundColor: '#0ea5e9',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    borderRadius: 12,
    gap: 8,
    marginTop: 8,
  },
  submitButtonDisabled: {
    opacity: 0.6,
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  errorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(239,68,68,0.1)',
    padding: 12,
    borderRadius: 10,
    marginTop: 12,
    gap: 8,
  },
  errorText: {
    color: '#ef4444',
    flex: 1,
  },
  results: {
    marginTop: 24,
  },
  summaryCard: {
    backgroundColor: 'rgba(14,165,233,0.1)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(14,165,233,0.3)',
  },
  summaryTitle: {
    color: '#0ea5e9',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  summaryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  summaryItem: {
    width: '50%',
    marginBottom: 12,
  },
  summaryLabel: {
    color: '#64748b',
    fontSize: 11,
    marginBottom: 2,
  },
  summaryCode: {
    color: '#0ea5e9',
    fontSize: 18,
    fontWeight: 'bold',
  },
  summaryValue: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  codeSection: {
    marginBottom: 16,
  },
  codeSectionTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  codeCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  codeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  codeText: {
    color: '#0ea5e9',
    fontSize: 18,
    fontWeight: 'bold',
  },
  confidenceBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  confidenceText: {
    fontSize: 12,
    fontWeight: '600',
  },
  rvuText: {
    color: '#10b981',
    fontSize: 13,
    fontWeight: '500',
  },
  codeDescription: {
    color: '#e2e8f0',
    fontSize: 14,
    marginBottom: 8,
  },
  codeNote: {
    color: '#f59e0b',
    fontSize: 12,
    fontStyle: 'italic',
    marginBottom: 6,
  },
  codeFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  codeCategory: {
    color: '#64748b',
    fontSize: 12,
  },
  tipsSection: {
    backgroundColor: 'rgba(245,158,11,0.1)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
  },
  tipsSectionTitle: {
    color: '#f59e0b',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 10,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
    gap: 8,
  },
  tipText: {
    color: '#fcd34d',
    fontSize: 13,
    flex: 1,
  },
  disclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    padding: 12,
    backgroundColor: 'rgba(100,116,139,0.1)',
    borderRadius: 10,
  },
  disclaimerText: {
    color: '#64748b',
    fontSize: 12,
    flex: 1,
    lineHeight: 18,
  },
  toggleContainer: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 10,
    padding: 4,
    marginBottom: 16,
  },
  toggleButton: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
  },
  toggleActive: {
    backgroundColor: '#0ea5e9',
  },
  toggleText: {
    color: '#64748b',
    fontWeight: '600',
  },
  toggleTextActive: {
    color: '#fff',
  },
  searchContainer: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 20,
  },
  searchInput: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 10,
    padding: 14,
    color: '#fff',
    fontSize: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  searchButton: {
    backgroundColor: '#0ea5e9',
    width: 50,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  searchResults: {},
  searchResultCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
  },
  noResults: {
    color: '#64748b',
    textAlign: 'center',
    marginTop: 40,
  },
  emptyHistory: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyHistoryText: {
    color: '#94a3b8',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
  },
  emptyHistorySubtext: {
    color: '#64748b',
    fontSize: 14,
    marginTop: 8,
  },
  historyCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  historyDiagnosis: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  historyCondition: {
    color: '#0ea5e9',
    fontSize: 12,
  },
  historyCodesRow: {
    flexDirection: 'row',
  },
  historyCodeItem: {
    flex: 1,
    alignItems: 'center',
  },
  historyCodeLabel: {
    color: '#64748b',
    fontSize: 11,
    marginBottom: 4,
  },
  historyCodeValue: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
});
