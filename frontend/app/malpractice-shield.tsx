import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  TextInput,
  Switch,
  Alert,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import * as SecureStore from 'expo-secure-store';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

// Helper to get token (SecureStore on native, AsyncStorage on web)
const getToken = async () => {
  if (Platform.OS === 'web') {
    return AsyncStorage.getItem('auth_token');
  }
  return SecureStore.getItemAsync('auth_token');
};

type TabType = 'risk' | 'insurance' | 'checklist' | 'statistics';

export default function MalpracticeShield() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<TabType>('risk');
  const [loading, setLoading] = useState(false);

  // Risk Analysis State
  const [diagnosis, setDiagnosis] = useState('');
  const [confidenceLevel, setConfidenceLevel] = useState('0.85');
  const [riskAnalysis, setRiskAnalysis] = useState<any>(null);

  // Documentation checkboxes
  const [docItems, setDocItems] = useState({
    photo_documentation: false,
    detailed_history: false,
    physical_exam_documented: false,
    differential_diagnosis: false,
    treatment_rationale: false,
    informed_consent: false,
    follow_up_plan: false,
    patient_education: false,
  });

  // Insurance State
  const [practiceType, setPracticeType] = useState('solo_practice');
  const [patientVolume, setPatientVolume] = useState('2000');
  const [highRiskProcedures, setHighRiskProcedures] = useState(false);
  const [insuranceRecs, setInsuranceRecs] = useState<any>(null);

  // Checklist State
  const [conditionType, setConditionType] = useState('general');
  const [checklist, setChecklist] = useState<any>(null);

  // Statistics State
  const [statistics, setStatistics] = useState<any>(null);
  const [mitigationTips, setMitigationTips] = useState<any>(null);

  const getAuthHeaders = async () => {
    const token = await getToken();
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    };
  };

  const analyzeRisk = async () => {
    if (!diagnosis.trim()) {
      Alert.alert('Error', 'Please enter a diagnosis');
      return;
    }

    setLoading(true);
    try {
      const headers = await getAuthHeaders();
      const formData = new URLSearchParams();
      formData.append('diagnosis', diagnosis);
      formData.append('confidence_level', confidenceLevel);
      formData.append('documentation_completeness', JSON.stringify(docItems));

      const response = await fetch(`${API_BASE_URL}/malpractice/analyze-risk`, {
        method: 'POST',
        headers,
        body: formData.toString(),
      });

      if (response.ok) {
        const data = await response.json();
        setRiskAnalysis(data);
      } else {
        Alert.alert('Error', 'Failed to analyze risk');
      }
    } catch (error) {
      Alert.alert('Error', 'Network error');
    } finally {
      setLoading(false);
    }
  };

  const getInsuranceRecommendations = async () => {
    setLoading(true);
    try {
      const token = await getToken();
      const response = await fetch(
        `${API_BASE_URL}/malpractice/insurance-recommendations?practice_type=${practiceType}&annual_patient_volume=${patientVolume}&high_risk_procedures=${highRiskProcedures}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setInsuranceRecs(data);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  const getChecklist = async () => {
    setLoading(true);
    try {
      const token = await getToken();
      const response = await fetch(
        `${API_BASE_URL}/malpractice/documentation-checklist?condition_type=${conditionType}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setChecklist(data);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to get checklist');
    } finally {
      setLoading(false);
    }
  };

  const getStatistics = async () => {
    setLoading(true);
    try {
      const token = await getToken();
      const [statsRes, tipsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/malpractice/claim-statistics`, {
          headers: { 'Authorization': `Bearer ${token}` },
        }),
        fetch(`${API_BASE_URL}/malpractice/risk-mitigation-tips`, {
          headers: { 'Authorization': `Bearer ${token}` },
        }),
      ]);

      if (statsRes.ok) {
        setStatistics(await statsRes.json());
      }
      if (tipsRes.ok) {
        setMitigationTips(await tipsRes.json());
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to get statistics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'statistics' && !statistics) {
      getStatistics();
    }
  }, [activeTab]);

  const renderTabs = () => (
    <View style={styles.tabContainer}>
      {[
        { key: 'risk', label: 'Risk Analysis' },
        { key: 'insurance', label: 'Insurance' },
        { key: 'checklist', label: 'Checklist' },
        { key: 'statistics', label: 'Statistics' },
      ].map((tab) => (
        <TouchableOpacity
          key={tab.key}
          style={[styles.tab, activeTab === tab.key && styles.activeTab]}
          onPress={() => setActiveTab(tab.key as TabType)}
        >
          <Text style={[styles.tabText, activeTab === tab.key && styles.activeTabText]}>
            {tab.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  const renderRiskAnalysis = () => (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>Liability Risk Analysis</Text>

      <Text style={styles.inputLabel}>Diagnosis</Text>
      <TextInput
        style={styles.input}
        value={diagnosis}
        onChangeText={setDiagnosis}
        placeholder="e.g., Melanoma, Basal Cell Carcinoma"
        placeholderTextColor="#999"
      />

      <Text style={styles.inputLabel}>AI Confidence Level</Text>
      <TextInput
        style={styles.input}
        value={confidenceLevel}
        onChangeText={setConfidenceLevel}
        placeholder="0.0 - 1.0"
        keyboardType="decimal-pad"
        placeholderTextColor="#999"
      />

      <Text style={styles.inputLabel}>Documentation Completed</Text>
      <View style={styles.checkboxContainer}>
        {Object.entries(docItems).map(([key, value]) => (
          <TouchableOpacity
            key={key}
            style={styles.checkboxRow}
            onPress={() => setDocItems({ ...docItems, [key]: !value })}
          >
            <View style={[styles.checkbox, value && styles.checkboxChecked]}>
              {value && <Text style={styles.checkmark}>✓</Text>}
            </View>
            <Text style={styles.checkboxLabel}>
              {key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <TouchableOpacity style={styles.analyzeButton} onPress={analyzeRisk}>
        <Text style={styles.analyzeButtonText}>Analyze Risk</Text>
      </TouchableOpacity>

      {riskAnalysis && (
        <View style={styles.resultsContainer}>
          <View style={[styles.riskBadge, { backgroundColor: riskAnalysis.risk_assessment.risk_color }]}>
            <Text style={styles.riskBadgeText}>
              {riskAnalysis.risk_assessment.risk_category.toUpperCase()} RISK
            </Text>
            <Text style={styles.riskScore}>
              Score: {riskAnalysis.risk_assessment.adjusted_risk_score}/100
            </Text>
          </View>

          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>Liability Exposure</Text>
            <Text style={styles.resultValue}>
              Average Settlement: {riskAnalysis.liability_exposure.average_settlement_formatted}
            </Text>
            <Text style={styles.resultValue}>
              Median Settlement: {riskAnalysis.liability_exposure.median_settlement_formatted}
            </Text>
            <Text style={styles.resultSubtitle}>Common Claims:</Text>
            {riskAnalysis.liability_exposure.common_claims.map((claim: string, i: number) => (
              <Text key={i} style={styles.bulletItem}>• {claim}</Text>
            ))}
          </View>

          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>Documentation Score</Text>
            <Text style={styles.gradeText}>
              Grade: {riskAnalysis.documentation_analysis.grade} ({riskAnalysis.documentation_analysis.score_percentage})
            </Text>
            {riskAnalysis.documentation_analysis.missing_items.length > 0 && (
              <>
                <Text style={styles.resultSubtitle}>Missing Items:</Text>
                {riskAnalysis.documentation_analysis.missing_items.slice(0, 3).map((item: any, i: number) => (
                  <Text key={i} style={[styles.bulletItem, item.priority === 'high' && styles.highPriority]}>
                    • {item.item} ({item.priority} priority)
                  </Text>
                ))}
              </>
            )}
          </View>

          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>Top Mitigation Strategies</Text>
            {riskAnalysis.mitigation_strategies.slice(0, 3).map((strategy: any, i: number) => (
              <View key={i} style={styles.strategyItem}>
                <Text style={styles.strategyName}>{strategy.strategy}</Text>
                <Text style={styles.strategyDetail}>
                  Premium Reduction: {strategy.premium_reduction}
                </Text>
              </View>
            ))}
          </View>
        </View>
      )}
    </View>
  );

  const renderInsurance = () => (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>Insurance Coverage Recommendations</Text>

      <Text style={styles.inputLabel}>Practice Type</Text>
      <View style={styles.pickerContainer}>
        {['solo_practice', 'group_practice', 'academic_medical_center'].map((type) => (
          <TouchableOpacity
            key={type}
            style={[styles.pickerOption, practiceType === type && styles.pickerOptionSelected]}
            onPress={() => setPracticeType(type)}
          >
            <Text style={[styles.pickerText, practiceType === type && styles.pickerTextSelected]}>
              {type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <Text style={styles.inputLabel}>Annual Patient Volume</Text>
      <TextInput
        style={styles.input}
        value={patientVolume}
        onChangeText={setPatientVolume}
        keyboardType="number-pad"
        placeholderTextColor="#999"
      />

      <View style={styles.switchRow}>
        <Text style={styles.switchLabel}>High-Risk Procedures</Text>
        <Switch
          value={highRiskProcedures}
          onValueChange={setHighRiskProcedures}
          trackColor={{ false: '#ddd', true: '#007AFF' }}
        />
      </View>

      <TouchableOpacity style={styles.analyzeButton} onPress={getInsuranceRecommendations}>
        <Text style={styles.analyzeButtonText}>Get Recommendations</Text>
      </TouchableOpacity>

      {insuranceRecs && (
        <View style={styles.resultsContainer}>
          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>Minimum Coverage</Text>
            <Text style={styles.coverageNotation}>{insuranceRecs.minimum_coverage.notation}</Text>
            <Text style={styles.resultValue}>
              Per Occurrence: {insuranceRecs.minimum_coverage.per_occurrence_formatted}
            </Text>
            <Text style={styles.resultValue}>
              Aggregate: {insuranceRecs.minimum_coverage.aggregate_formatted}
            </Text>
          </View>

          <View style={[styles.resultCard, styles.recommendedCard]}>
            <Text style={styles.resultCardTitle}>Recommended Coverage</Text>
            <Text style={styles.coverageNotation}>{insuranceRecs.recommended_coverage.notation}</Text>
            <Text style={styles.resultValue}>
              Per Occurrence: {insuranceRecs.recommended_coverage.per_occurrence_formatted}
            </Text>
            <Text style={styles.resultValue}>
              Aggregate: {insuranceRecs.recommended_coverage.aggregate_formatted}
            </Text>
          </View>

          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>Estimated Annual Premium</Text>
            <Text style={styles.resultValue}>
              Minimum: {insuranceRecs.estimated_annual_premium.minimum_coverage_range}
            </Text>
            <Text style={styles.resultValue}>
              Recommended: {insuranceRecs.estimated_annual_premium.recommended_coverage_range}
            </Text>
          </View>

          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>Policy Recommendation</Text>
            <Text style={styles.policyType}>
              {insuranceRecs.recommendation === 'claims_made' ? 'Claims-Made Policy' : 'Occurrence Policy'}
            </Text>
            <Text style={styles.resultValue}>{insuranceRecs.recommendation_rationale}</Text>
          </View>

          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>Additional Coverage</Text>
            {insuranceRecs.additional_coverage_considerations.map((item: any, i: number) => (
              <View key={i} style={styles.additionalItem}>
                <Text style={styles.additionalType}>{item.type}</Text>
                <Text style={styles.additionalReason}>{item.reason}</Text>
                {item.recommended_limit && (
                  <Text style={styles.additionalLimit}>Limit: {item.recommended_limit}</Text>
                )}
              </View>
            ))}
          </View>
        </View>
      )}
    </View>
  );

  const renderChecklist = () => (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>Documentation Checklist</Text>

      <Text style={styles.inputLabel}>Condition Type</Text>
      <View style={styles.pickerContainer}>
        {['general', 'melanoma', 'skin_cancer', 'inflammatory'].map((type) => (
          <TouchableOpacity
            key={type}
            style={[styles.pickerOption, conditionType === type && styles.pickerOptionSelected]}
            onPress={() => setConditionType(type)}
          >
            <Text style={[styles.pickerText, conditionType === type && styles.pickerTextSelected]}>
              {type.charAt(0).toUpperCase() + type.slice(1).replace('_', ' ')}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <TouchableOpacity style={styles.analyzeButton} onPress={getChecklist}>
        <Text style={styles.analyzeButtonText}>Get Checklist</Text>
      </TouchableOpacity>

      {checklist && (
        <View style={styles.resultsContainer}>
          <View style={styles.checklistSummary}>
            <Text style={styles.checklistCount}>
              {checklist.required_items} required / {checklist.total_items} total items
            </Text>
          </View>

          {Object.entries(checklist.categories).map(([category, items]: [string, any]) => (
            items.length > 0 && (
              <View key={category} style={styles.categoryCard}>
                <Text style={styles.categoryTitle}>{category}</Text>
                {items.map((item: any, i: number) => (
                  <View key={i} style={styles.checklistItem}>
                    <View style={[styles.requiredDot, !item.required && styles.optionalDot]} />
                    <Text style={styles.checklistItemText}>{item.item}</Text>
                  </View>
                ))}
              </View>
            )
          ))}

          <View style={styles.tipsCard}>
            <Text style={styles.resultCardTitle}>Documentation Tips</Text>
            {checklist.tips.map((tip: string, i: number) => (
              <Text key={i} style={styles.bulletItem}>• {tip}</Text>
            ))}
          </View>
        </View>
      )}
    </View>
  );

  const renderStatistics = () => (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>Claim Statistics & Trends</Text>

      {statistics && (
        <View style={styles.resultsContainer}>
          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>Overview</Text>
            <Text style={styles.statText}>{statistics.overview.specialty_ranking}</Text>
            <Text style={styles.statHighlight}>
              Annual Claim Rate: {statistics.overview.annual_claim_rate}
            </Text>
            <Text style={styles.statText}>
              Claims Resulting in Payment: {statistics.overview.claims_resulting_in_payment}
            </Text>
          </View>

          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>Claim Breakdown by Type</Text>
            {statistics.claim_breakdown_by_type.map((claim: any, i: number) => (
              <View key={i} style={styles.claimRow}>
                <View style={styles.claimInfo}>
                  <Text style={styles.claimType}>{claim.type}</Text>
                  <Text style={styles.claimPayout}>Avg: ${claim.avg_payout.toLocaleString()}</Text>
                </View>
                <View style={styles.percentageBar}>
                  <View style={[styles.percentageFill, { width: `${claim.percentage}%` }]} />
                </View>
                <Text style={styles.percentageText}>{claim.percentage}%</Text>
              </View>
            ))}
          </View>

          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>Defense Success Factors</Text>
            {statistics.defense_success_factors.map((factor: any, i: number) => (
              <View key={i} style={styles.factorRow}>
                <Text style={styles.factorName}>{factor.factor}</Text>
                <Text style={styles.factorImpact}>{factor.impact}</Text>
              </View>
            ))}
          </View>

          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>High-Risk Scenarios</Text>
            {statistics.high_risk_scenarios.map((scenario: any, i: number) => (
              <View key={i} style={styles.scenarioItem}>
                <View style={[
                  styles.riskIndicator,
                  scenario.risk_level === 'very_high' ? styles.veryHighRisk : styles.highRisk
                ]} />
                <View style={styles.scenarioContent}>
                  <Text style={styles.scenarioText}>{scenario.scenario}</Text>
                  <Text style={styles.scenarioRec}>{scenario.recommendation}</Text>
                </View>
              </View>
            ))}
          </View>
        </View>
      )}

      {mitigationTips && (
        <View style={styles.resultsContainer}>
          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>AI-Specific Guidance</Text>
            {mitigationTips.ai_specific_guidance.map((item: any, i: number) => (
              <View key={i} style={styles.guidanceItem}>
                <Text style={styles.guidancePractice}>{item.practice}</Text>
                <Text style={styles.guidanceRationale}>{item.rationale}</Text>
              </View>
            ))}
          </View>

          <View style={styles.resultCard}>
            <Text style={styles.resultCardTitle}>When to Refer</Text>
            {mitigationTips.when_to_refer.map((item: string, i: number) => (
              <Text key={i} style={styles.bulletItem}>• {item}</Text>
            ))}
          </View>
        </View>
      )}
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient
        colors={['#1a1a2e', '#16213e', '#0f3460']}
        style={styles.gradient}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Malpractice Shield</Text>
        </View>

        {renderTabs()}

        <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
          {loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#00ff88" />
              <Text style={styles.loadingText}>Loading...</Text>
            </View>
          ) : (
            <>
              {activeTab === 'risk' && renderRiskAnalysis()}
              {activeTab === 'insurance' && renderInsurance()}
              {activeTab === 'checklist' && renderChecklist()}
              {activeTab === 'statistics' && renderStatistics()}
            </>
          )}
        </ScrollView>
      </LinearGradient>
    </SafeAreaView>
  );
}

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
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  backButton: {
    marginRight: 15,
  },
  backButtonText: {
    color: '#00ff88',
    fontSize: 16,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  tabContainer: {
    flexDirection: 'row',
    paddingHorizontal: 10,
    marginBottom: 10,
  },
  tab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
    marginHorizontal: 4,
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  activeTab: {
    backgroundColor: '#00ff88',
  },
  tabText: {
    color: '#aaa',
    fontSize: 12,
    fontWeight: '600',
  },
  activeTabText: {
    color: '#1a1a2e',
  },
  content: {
    flex: 1,
    paddingHorizontal: 15,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 100,
  },
  loadingText: {
    color: '#fff',
    marginTop: 10,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    color: '#00ff88',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  inputLabel: {
    color: '#aaa',
    fontSize: 14,
    marginBottom: 5,
    marginTop: 10,
  },
  input: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 10,
    padding: 15,
    color: '#fff',
    fontSize: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  checkboxContainer: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 10,
    padding: 10,
  },
  checkboxRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
  },
  checkbox: {
    width: 24,
    height: 24,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: '#00ff88',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  checkboxChecked: {
    backgroundColor: '#00ff88',
  },
  checkmark: {
    color: '#1a1a2e',
    fontWeight: 'bold',
  },
  checkboxLabel: {
    color: '#fff',
    fontSize: 14,
  },
  analyzeButton: {
    backgroundColor: '#00ff88',
    borderRadius: 10,
    padding: 15,
    alignItems: 'center',
    marginTop: 20,
  },
  analyzeButtonText: {
    color: '#1a1a2e',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultsContainer: {
    marginTop: 20,
  },
  riskBadge: {
    borderRadius: 15,
    padding: 20,
    alignItems: 'center',
    marginBottom: 15,
  },
  riskBadgeText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  riskScore: {
    color: '#fff',
    fontSize: 16,
    marginTop: 5,
  },
  resultCard: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 15,
    padding: 15,
    marginBottom: 15,
  },
  recommendedCard: {
    borderWidth: 2,
    borderColor: '#00ff88',
  },
  resultCardTitle: {
    color: '#00ff88',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  resultValue: {
    color: '#fff',
    fontSize: 14,
    marginBottom: 5,
  },
  resultSubtitle: {
    color: '#aaa',
    fontSize: 14,
    marginTop: 10,
    marginBottom: 5,
  },
  bulletItem: {
    color: '#ddd',
    fontSize: 14,
    marginLeft: 10,
    marginBottom: 3,
  },
  highPriority: {
    color: '#ff6b6b',
  },
  gradeText: {
    color: '#00ff88',
    fontSize: 18,
    fontWeight: 'bold',
  },
  strategyItem: {
    backgroundColor: 'rgba(0,255,136,0.1)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
  },
  strategyName: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  strategyDetail: {
    color: '#00ff88',
    fontSize: 12,
    marginTop: 3,
  },
  pickerContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  pickerOption: {
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.1)',
    marginBottom: 5,
  },
  pickerOptionSelected: {
    backgroundColor: '#00ff88',
  },
  pickerText: {
    color: '#aaa',
    fontSize: 12,
  },
  pickerTextSelected: {
    color: '#1a1a2e',
    fontWeight: 'bold',
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 15,
    backgroundColor: 'rgba(255,255,255,0.05)',
    padding: 15,
    borderRadius: 10,
  },
  switchLabel: {
    color: '#fff',
    fontSize: 14,
  },
  coverageNotation: {
    color: '#00ff88',
    fontSize: 28,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 10,
  },
  policyType: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  additionalItem: {
    backgroundColor: 'rgba(0,255,136,0.1)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
  },
  additionalType: {
    color: '#00ff88',
    fontSize: 14,
    fontWeight: 'bold',
  },
  additionalReason: {
    color: '#ddd',
    fontSize: 12,
    marginTop: 3,
  },
  additionalLimit: {
    color: '#fff',
    fontSize: 12,
    marginTop: 3,
    fontWeight: '600',
  },
  checklistSummary: {
    backgroundColor: 'rgba(0,255,136,0.2)',
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
    alignItems: 'center',
  },
  checklistCount: {
    color: '#00ff88',
    fontSize: 16,
    fontWeight: 'bold',
  },
  categoryCard: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 15,
    padding: 15,
    marginBottom: 15,
  },
  categoryTitle: {
    color: '#00ff88',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  checklistItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 5,
  },
  requiredDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#ff6b6b',
    marginRight: 10,
  },
  optionalDot: {
    backgroundColor: '#888',
  },
  checklistItemText: {
    color: '#fff',
    fontSize: 14,
    flex: 1,
  },
  tipsCard: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 15,
    padding: 15,
    marginTop: 10,
  },
  statText: {
    color: '#ddd',
    fontSize: 14,
    marginBottom: 8,
  },
  statHighlight: {
    color: '#00ff88',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  claimRow: {
    marginBottom: 12,
  },
  claimInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  claimType: {
    color: '#fff',
    fontSize: 13,
    flex: 1,
  },
  claimPayout: {
    color: '#aaa',
    fontSize: 12,
  },
  percentageBar: {
    height: 8,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 4,
    marginBottom: 3,
  },
  percentageFill: {
    height: '100%',
    backgroundColor: '#00ff88',
    borderRadius: 4,
  },
  percentageText: {
    color: '#aaa',
    fontSize: 12,
    textAlign: 'right',
  },
  factorRow: {
    backgroundColor: 'rgba(0,255,136,0.1)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
  },
  factorName: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  factorImpact: {
    color: '#00ff88',
    fontSize: 12,
    marginTop: 3,
  },
  scenarioItem: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  riskIndicator: {
    width: 4,
    borderRadius: 2,
    marginRight: 10,
  },
  veryHighRisk: {
    backgroundColor: '#ff4757',
  },
  highRisk: {
    backgroundColor: '#ffa502',
  },
  scenarioContent: {
    flex: 1,
  },
  scenarioText: {
    color: '#fff',
    fontSize: 14,
    marginBottom: 3,
  },
  scenarioRec: {
    color: '#aaa',
    fontSize: 12,
    fontStyle: 'italic',
  },
  guidanceItem: {
    backgroundColor: 'rgba(0,255,136,0.1)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
  },
  guidancePractice: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  guidanceRationale: {
    color: '#aaa',
    fontSize: 12,
    marginTop: 3,
  },
});
