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
  Modal,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { API_BASE_URL } from '../config';

const { width } = Dimensions.get('window');

interface LymphNodeBasin {
  id: string;
  name: string;
  description: string;
  location: { x: number; y: number };
  drainage_from: string[];
  is_primary_drainage?: boolean;
  priority?: number;
}

interface BiopsyRecord {
  biopsy_id: string;
  primary_site: string;
  biopsy_date: string;
  basin: string;
  node_id: string;
  result: string;
  result_description: string;
  tumor_deposit_mm?: number;
  nodes_examined: number;
  nodes_positive: number;
  extracapsular_extension: boolean;
  n_category: string;
  result_color: string;
  implications: string[];
}

interface ResultCategory {
  id: string;
  code: string;
  description: string;
  color: string;
  implications: string[];
}

type TabType = 'map' | 'record' | 'history';

export default function SentinelNodeMapperScreen() {
  const router = useRouter();
  const { user } = useAuth();

  const [activeTab, setActiveTab] = useState<TabType>('map');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Mapping state
  const [primarySite, setPrimarySite] = useState('');
  const [laterality, setLaterality] = useState<string | null>(null);
  const [mappingResult, setMappingResult] = useState<any>(null);

  // Recording state
  const [resultCategories, setResultCategories] = useState<ResultCategory[]>([]);
  const [selectedBasin, setSelectedBasin] = useState('');
  const [nodeId, setNodeId] = useState('');
  const [biopsyDate, setBiopsyDate] = useState(new Date().toISOString().split('T')[0]);
  const [selectedResult, setSelectedResult] = useState<string | null>(null);
  const [tumorDeposit, setTumorDeposit] = useState('');
  const [nodesExamined, setNodesExamined] = useState('1');
  const [nodesPositive, setNodesPositive] = useState('0');
  const [extracapsular, setExtracapsular] = useState(false);
  const [ihcMarkers, setIhcMarkers] = useState('');
  const [notes, setNotes] = useState('');
  const [showResultModal, setShowResultModal] = useState(false);

  // History state
  const [biopsyHistory, setBiopsyHistory] = useState<BiopsyRecord[]>([]);
  const [historySummary, setHistorySummary] = useState<any>(null);

  // Common body sites for quick selection
  const commonSites = [
    { label: 'Scalp', value: 'scalp' },
    { label: 'Face', value: 'face' },
    { label: 'Neck', value: 'neck' },
    { label: 'Arm', value: 'arm' },
    { label: 'Forearm', value: 'forearm' },
    { label: 'Hand', value: 'hand' },
    { label: 'Chest', value: 'chest' },
    { label: 'Back', value: 'back' },
    { label: 'Abdomen', value: 'abdomen' },
    { label: 'Thigh', value: 'thigh' },
    { label: 'Leg', value: 'leg' },
    { label: 'Foot', value: 'foot' },
  ];

  useEffect(() => {
    loadResultCategories();
    if (activeTab === 'history') {
      loadBiopsyHistory();
    }
  }, [activeTab]);

  const loadResultCategories = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/sentinel-node/result-categories`, {
        headers: { 'Authorization': `Bearer ${user?.token}` },
      });
      if (response.ok) {
        const data = await response.json();
        setResultCategories(data.categories);
      }
    } catch (err) {
      console.error('Error loading result categories:', err);
    }
  };

  const loadBiopsyHistory = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/sentinel-node/biopsies`, {
        headers: { 'Authorization': `Bearer ${user?.token}` },
      });
      if (response.ok) {
        const data = await response.json();
        setBiopsyHistory(data.biopsies);
        setHistorySummary(data.summary);
      }
    } catch (err) {
      console.error('Error loading biopsy history:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const mapDrainage = async () => {
    if (!primarySite) {
      setError('Please enter or select a primary tumor site');
      return;
    }

    setIsLoading(true);
    setError(null);
    setMappingResult(null);

    try {
      const formData = new FormData();
      formData.append('primary_site', primarySite);
      if (laterality) {
        formData.append('laterality', laterality);
      }

      const response = await fetch(`${API_BASE_URL}/sentinel-node/map-drainage`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user?.token}` },
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to map drainage');
      }

      const data = await response.json();
      setMappingResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to map lymphatic drainage');
    } finally {
      setIsLoading(false);
    }
  };

  const recordBiopsy = async () => {
    if (!primarySite || !selectedBasin || !nodeId || !selectedResult) {
      setError('Please fill in all required fields');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('primary_site', primarySite);
      formData.append('biopsy_date', biopsyDate);
      formData.append('basin', selectedBasin);
      formData.append('node_id', nodeId);
      formData.append('result', selectedResult);
      formData.append('nodes_examined', nodesExamined);
      formData.append('nodes_positive', nodesPositive);
      formData.append('extracapsular', extracapsular.toString());

      if (tumorDeposit) {
        formData.append('tumor_deposit_mm', tumorDeposit);
      }
      if (ihcMarkers) {
        formData.append('immunohistochemistry', ihcMarkers);
      }
      if (notes) {
        formData.append('notes', notes);
      }

      const response = await fetch(`${API_BASE_URL}/sentinel-node/record-biopsy`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user?.token}` },
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to record biopsy');
      }

      const data = await response.json();
      setShowResultModal(true);

      // Reset form
      setSelectedBasin('');
      setNodeId('');
      setSelectedResult(null);
      setTumorDeposit('');
      setNodesExamined('1');
      setNodesPositive('0');
      setExtracapsular(false);
      setIhcMarkers('');
      setNotes('');

      // Reload history
      loadBiopsyHistory();
    } catch (err: any) {
      setError(err.message || 'Failed to record biopsy');
    } finally {
      setIsLoading(false);
    }
  };

  const renderNodeMap = () => {
    if (!mappingResult) return null;

    const basins = mappingResult.expected_basins as LymphNodeBasin[];

    return (
      <View style={styles.mapContainer}>
        <Text style={styles.mapTitle}>{mappingResult.region_name} Lymphatic Basins</Text>

        {/* Visual Node Map */}
        <View style={styles.bodyMap}>
          {basins.map((basin) => (
            <TouchableOpacity
              key={basin.id}
              style={[
                styles.nodeMarker,
                {
                  left: `${basin.location.x}%`,
                  top: `${basin.location.y}%`,
                  backgroundColor: basin.is_primary_drainage ? '#0ea5e9' : '#475569',
                  transform: [{ translateX: -12 }, { translateY: -12 }],
                },
              ]}
              onPress={() => {
                setSelectedBasin(basin.id);
                setActiveTab('record');
              }}
            >
              {basin.is_primary_drainage && (
                <View style={styles.primaryIndicator} />
              )}
            </TouchableOpacity>
          ))}
        </View>

        {/* Basin List */}
        <View style={styles.basinList}>
          {basins.map((basin) => (
            <View
              key={basin.id}
              style={[
                styles.basinCard,
                basin.is_primary_drainage && styles.primaryBasinCard,
              ]}
            >
              <View style={styles.basinHeader}>
                <View style={styles.basinNameRow}>
                  {basin.is_primary_drainage && (
                    <Ionicons name="star" size={14} color="#f59e0b" style={{ marginRight: 4 }} />
                  )}
                  <Text style={styles.basinName}>{basin.name}</Text>
                </View>
                <TouchableOpacity
                  style={styles.recordButton}
                  onPress={() => {
                    setSelectedBasin(basin.id);
                    setActiveTab('record');
                  }}
                >
                  <Ionicons name="add-circle" size={20} color="#0ea5e9" />
                </TouchableOpacity>
              </View>
              <Text style={styles.basinDescription}>{basin.description}</Text>
              {basin.is_primary_drainage && (
                <Text style={styles.primaryLabel}>Primary drainage basin</Text>
              )}
            </View>
          ))}
        </View>

        {/* Watershed Info */}
        {mappingResult.watershed_info && (
          <View style={styles.watershedInfo}>
            <Ionicons name="information-circle" size={18} color="#0ea5e9" />
            <Text style={styles.watershedText}>{mappingResult.watershed_info.description}</Text>
          </View>
        )}

        {/* Clinical Notes */}
        <View style={styles.clinicalNotes}>
          <Text style={styles.notesTitle}>Clinical Notes</Text>
          {mappingResult.clinical_notes.map((note: string, i: number) => (
            <Text key={i} style={styles.noteItem}>• {note}</Text>
          ))}
        </View>
      </View>
    );
  };

  const renderMappingTab = () => (
    <View>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>
          <Ionicons name="location-outline" size={18} color="#0ea5e9" /> Primary Tumor Site
        </Text>

        <TextInput
          style={styles.textInput}
          value={primarySite}
          onChangeText={setPrimarySite}
          placeholder="Enter tumor location (e.g., right upper arm)"
          placeholderTextColor="#6b7280"
        />

        <Text style={styles.quickSelectLabel}>Quick Select:</Text>
        <View style={styles.quickSelectGrid}>
          {commonSites.map((site) => (
            <TouchableOpacity
              key={site.value}
              style={[
                styles.quickSelectButton,
                primarySite.toLowerCase().includes(site.value) && styles.quickSelectActive,
              ]}
              onPress={() => setPrimarySite(site.label)}
            >
              <Text
                style={[
                  styles.quickSelectText,
                  primarySite.toLowerCase().includes(site.value) && styles.quickSelectTextActive,
                ]}
              >
                {site.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <Text style={styles.label}>Laterality</Text>
        <View style={styles.lateralityRow}>
          {['left', 'right', 'midline'].map((lat) => (
            <TouchableOpacity
              key={lat}
              style={[
                styles.lateralityButton,
                laterality === lat && styles.lateralityButtonActive,
              ]}
              onPress={() => setLaterality(laterality === lat ? null : lat)}
            >
              <Text
                style={[
                  styles.lateralityText,
                  laterality === lat && styles.lateralityTextActive,
                ]}
              >
                {lat.charAt(0).toUpperCase() + lat.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <TouchableOpacity
        style={[styles.mapButton, isLoading && styles.buttonDisabled]}
        onPress={mapDrainage}
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="git-network-outline" size={20} color="#fff" />
            <Text style={styles.mapButtonText}>Map Lymphatic Drainage</Text>
          </>
        )}
      </TouchableOpacity>

      {error && (
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle" size={20} color="#ef4444" />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {renderNodeMap()}
    </View>
  );

  const renderRecordTab = () => (
    <View>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>
          <Ionicons name="document-text-outline" size={18} color="#0ea5e9" /> Record SLNB Result
        </Text>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Primary Site *</Text>
          <TextInput
            style={styles.textInput}
            value={primarySite}
            onChangeText={setPrimarySite}
            placeholder="Enter primary tumor location"
            placeholderTextColor="#6b7280"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Biopsy Date *</Text>
          <TextInput
            style={styles.textInput}
            value={biopsyDate}
            onChangeText={setBiopsyDate}
            placeholder="YYYY-MM-DD"
            placeholderTextColor="#6b7280"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Lymph Node Basin *</Text>
          <TextInput
            style={styles.textInput}
            value={selectedBasin}
            onChangeText={setSelectedBasin}
            placeholder="e.g., axillary_level_i"
            placeholderTextColor="#6b7280"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Node Identifier *</Text>
          <TextInput
            style={styles.textInput}
            value={nodeId}
            onChangeText={setNodeId}
            placeholder="e.g., SLN-1, Hot node"
            placeholderTextColor="#6b7280"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Result *</Text>
          <View style={styles.resultGrid}>
            {resultCategories.map((cat) => (
              <TouchableOpacity
                key={cat.id}
                style={[
                  styles.resultCard,
                  selectedResult === cat.id && { borderColor: cat.color, borderWidth: 2 },
                ]}
                onPress={() => setSelectedResult(cat.id)}
              >
                <View style={[styles.resultDot, { backgroundColor: cat.color }]} />
                <Text style={styles.resultLabel}>{cat.id.replace(/_/g, ' ')}</Text>
                <Text style={styles.resultCode}>{cat.code}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={styles.rowInputs}>
          <View style={[styles.inputGroup, { flex: 1 }]}>
            <Text style={styles.label}>Nodes Examined</Text>
            <TextInput
              style={styles.textInput}
              value={nodesExamined}
              onChangeText={setNodesExamined}
              keyboardType="number-pad"
              placeholderTextColor="#6b7280"
            />
          </View>
          <View style={[styles.inputGroup, { flex: 1, marginLeft: 12 }]}>
            <Text style={styles.label}>Nodes Positive</Text>
            <TextInput
              style={styles.textInput}
              value={nodesPositive}
              onChangeText={setNodesPositive}
              keyboardType="number-pad"
              placeholderTextColor="#6b7280"
            />
          </View>
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Tumor Deposit Size (mm)</Text>
          <TextInput
            style={styles.textInput}
            value={tumorDeposit}
            onChangeText={setTumorDeposit}
            placeholder="Optional"
            placeholderTextColor="#6b7280"
            keyboardType="decimal-pad"
          />
        </View>

        <TouchableOpacity
          style={[styles.toggleRow, extracapsular && styles.toggleRowActive]}
          onPress={() => setExtracapsular(!extracapsular)}
        >
          <Text style={styles.toggleLabel}>Extracapsular Extension</Text>
          <View style={[styles.checkbox, extracapsular && styles.checkboxActive]}>
            {extracapsular && <Ionicons name="checkmark" size={14} color="#fff" />}
          </View>
        </TouchableOpacity>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>IHC Markers</Text>
          <TextInput
            style={styles.textInput}
            value={ihcMarkers}
            onChangeText={setIhcMarkers}
            placeholder="e.g., S100, HMB45, Melan-A"
            placeholderTextColor="#6b7280"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Notes</Text>
          <TextInput
            style={[styles.textInput, styles.textArea]}
            value={notes}
            onChangeText={setNotes}
            placeholder="Additional pathology notes"
            placeholderTextColor="#6b7280"
            multiline
            numberOfLines={3}
          />
        </View>
      </View>

      <TouchableOpacity
        style={[styles.recordBiopsyButton, isLoading && styles.buttonDisabled]}
        onPress={recordBiopsy}
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="save-outline" size={20} color="#fff" />
            <Text style={styles.recordBiopsyButtonText}>Record Biopsy Result</Text>
          </>
        )}
      </TouchableOpacity>

      {error && (
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle" size={20} color="#ef4444" />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}
    </View>
  );

  const renderHistoryTab = () => (
    <View>
      {historySummary && (
        <View style={styles.summaryCards}>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryValue}>{historySummary.total_procedures}</Text>
            <Text style={styles.summaryLabel}>Procedures</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryValue}>{historySummary.total_nodes_examined}</Text>
            <Text style={styles.summaryLabel}>Nodes</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={[styles.summaryValue, { color: historySummary.total_nodes_positive > 0 ? '#ef4444' : '#10b981' }]}>
              {historySummary.positivity_rate}
            </Text>
            <Text style={styles.summaryLabel}>Positive</Text>
          </View>
        </View>
      )}

      {isLoading ? (
        <ActivityIndicator color="#0ea5e9" style={{ marginTop: 40 }} />
      ) : biopsyHistory.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="document-outline" size={48} color="#475569" />
          <Text style={styles.emptyText}>No biopsy records yet</Text>
          <Text style={styles.emptySubtext}>Record your first SLNB result in the Record tab</Text>
        </View>
      ) : (
        <View style={styles.historyList}>
          {biopsyHistory.map((biopsy) => (
            <View key={biopsy.biopsy_id} style={styles.historyCard}>
              <View style={styles.historyHeader}>
                <View>
                  <Text style={styles.historyDate}>{biopsy.biopsy_date}</Text>
                  <Text style={styles.historySite}>{biopsy.primary_site}</Text>
                </View>
                <View style={[styles.resultBadge, { backgroundColor: biopsy.result_color + '20' }]}>
                  <Text style={[styles.resultBadgeText, { color: biopsy.result_color }]}>
                    {biopsy.n_category}
                  </Text>
                </View>
              </View>

              <View style={styles.historyDetails}>
                <Text style={styles.historyBasin}>
                  <Ionicons name="location" size={12} color="#64748b" /> {biopsy.basin.replace(/_/g, ' ')}
                </Text>
                <Text style={styles.historyNodes}>
                  {biopsy.nodes_positive}/{biopsy.nodes_examined} nodes positive
                </Text>
              </View>

              <Text style={styles.historyResult}>{biopsy.result_description}</Text>

              {biopsy.extracapsular_extension && (
                <View style={styles.warningBadge}>
                  <Ionicons name="warning" size={12} color="#ef4444" />
                  <Text style={styles.warningText}>Extracapsular extension</Text>
                </View>
              )}
            </View>
          ))}
        </View>
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
          <Text style={styles.headerTitle}>Sentinel Node Mapper</Text>
          <Text style={styles.headerSubtitle}>
            Visual lymph node basin mapping & biopsy tracking
          </Text>
        </View>
      </LinearGradient>

      {/* Tabs */}
      <View style={styles.tabBar}>
        {[
          { id: 'map' as TabType, label: 'Map', icon: 'git-network-outline' },
          { id: 'record' as TabType, label: 'Record', icon: 'document-text-outline' },
          { id: 'history' as TabType, label: 'History', icon: 'time-outline' },
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
        {activeTab === 'map' && renderMappingTab()}
        {activeTab === 'record' && renderRecordTab()}
        {activeTab === 'history' && renderHistoryTab()}
        <View style={{ height: 40 }} />
      </ScrollView>

      {/* Success Modal */}
      <Modal visible={showResultModal} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.successIcon}>
              <Ionicons name="checkmark-circle" size={48} color="#10b981" />
            </View>
            <Text style={styles.modalTitle}>Biopsy Recorded</Text>
            <Text style={styles.modalText}>
              The sentinel node biopsy result has been saved successfully.
            </Text>
            <TouchableOpacity
              style={styles.modalButton}
              onPress={() => {
                setShowResultModal(false);
                setActiveTab('history');
              }}
            >
              <Text style={styles.modalButtonText}>View History</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
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
    fontSize: 13,
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
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 16,
  },
  textInput: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 12,
    padding: 14,
    color: '#fff',
    fontSize: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  textArea: {
    minHeight: 80,
    textAlignVertical: 'top',
  },
  quickSelectLabel: {
    color: '#94a3b8',
    fontSize: 13,
    marginTop: 12,
    marginBottom: 8,
  },
  quickSelectGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  quickSelectButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  quickSelectActive: {
    backgroundColor: '#0ea5e9',
  },
  quickSelectText: {
    color: '#94a3b8',
    fontSize: 13,
  },
  quickSelectTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  label: {
    color: '#94a3b8',
    fontSize: 14,
    marginBottom: 8,
    marginTop: 12,
  },
  lateralityRow: {
    flexDirection: 'row',
    gap: 8,
  },
  lateralityButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.1)',
    alignItems: 'center',
  },
  lateralityButtonActive: {
    backgroundColor: '#0ea5e9',
  },
  lateralityText: {
    color: '#94a3b8',
    fontSize: 14,
  },
  lateralityTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  mapButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0ea5e9',
    padding: 16,
    borderRadius: 16,
    gap: 8,
    marginBottom: 16,
  },
  mapButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  errorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(239,68,68,0.1)',
    padding: 12,
    borderRadius: 12,
    gap: 8,
    marginBottom: 16,
  },
  errorText: {
    color: '#ef4444',
    flex: 1,
  },
  mapContainer: {
    marginTop: 8,
  },
  mapTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
    textAlign: 'center',
  },
  bodyMap: {
    height: 200,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    marginBottom: 16,
    position: 'relative',
  },
  nodeMarker: {
    position: 'absolute',
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#f59e0b',
  },
  basinList: {
    gap: 8,
  },
  basinCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 12,
  },
  primaryBasinCard: {
    borderLeftWidth: 3,
    borderLeftColor: '#0ea5e9',
  },
  basinHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  basinNameRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  basinName: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '500',
  },
  recordButton: {
    padding: 4,
  },
  basinDescription: {
    color: '#64748b',
    fontSize: 13,
    marginTop: 4,
  },
  primaryLabel: {
    color: '#0ea5e9',
    fontSize: 11,
    marginTop: 4,
    fontWeight: '500',
  },
  watershedInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(14,165,233,0.1)',
    padding: 12,
    borderRadius: 12,
    gap: 8,
    marginTop: 16,
  },
  watershedText: {
    color: '#0ea5e9',
    fontSize: 13,
    flex: 1,
  },
  clinicalNotes: {
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderRadius: 12,
    padding: 12,
    marginTop: 16,
  },
  notesTitle: {
    color: '#64748b',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 8,
  },
  noteItem: {
    color: '#94a3b8',
    fontSize: 12,
    marginBottom: 4,
  },
  inputGroup: {
    marginBottom: 8,
  },
  resultGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  resultCard: {
    width: (width - 72) / 2,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  resultDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginBottom: 6,
  },
  resultLabel: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
    textTransform: 'capitalize',
  },
  resultCode: {
    color: '#64748b',
    fontSize: 11,
    marginTop: 2,
  },
  rowInputs: {
    flexDirection: 'row',
  },
  toggleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.05)',
    padding: 14,
    borderRadius: 12,
    marginBottom: 12,
  },
  toggleRowActive: {
    backgroundColor: 'rgba(239,68,68,0.1)',
  },
  toggleLabel: {
    color: '#fff',
    fontSize: 14,
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: '#475569',
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxActive: {
    backgroundColor: '#ef4444',
    borderColor: '#ef4444',
  },
  recordBiopsyButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#10b981',
    padding: 16,
    borderRadius: 16,
    gap: 8,
    marginBottom: 16,
  },
  recordBiopsyButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  summaryCards: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  summaryCard: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  summaryValue: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  summaryLabel: {
    color: '#64748b',
    fontSize: 12,
    marginTop: 4,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    color: '#94a3b8',
    fontSize: 16,
    marginTop: 16,
  },
  emptySubtext: {
    color: '#64748b',
    fontSize: 13,
    marginTop: 4,
  },
  historyList: {
    gap: 12,
  },
  historyCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 14,
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  historyDate: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '500',
  },
  historySite: {
    color: '#94a3b8',
    fontSize: 13,
    marginTop: 2,
  },
  resultBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  resultBadgeText: {
    fontSize: 13,
    fontWeight: '600',
  },
  historyDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.1)',
  },
  historyBasin: {
    color: '#64748b',
    fontSize: 12,
  },
  historyNodes: {
    color: '#64748b',
    fontSize: 12,
  },
  historyResult: {
    color: '#94a3b8',
    fontSize: 13,
    marginTop: 8,
  },
  warningBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(239,68,68,0.1)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
    alignSelf: 'flex-start',
    marginTop: 8,
    gap: 4,
  },
  warningText: {
    color: '#ef4444',
    fontSize: 11,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalContent: {
    backgroundColor: '#1e293b',
    borderRadius: 20,
    padding: 24,
    width: width - 60,
    alignItems: 'center',
  },
  successIcon: {
    marginBottom: 16,
  },
  modalTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  modalText: {
    color: '#94a3b8',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 20,
  },
  modalButton: {
    backgroundColor: '#0ea5e9',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
  },
  modalButtonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
});
