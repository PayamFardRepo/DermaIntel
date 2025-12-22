import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  TextInput,
  Platform
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';

interface RiskAssessment {
  assessment_id: string;
  overall_risk_score: number;
  risk_category: string;
  risk_category_description: string;
  melanoma_risk: {
    relative_risk: number;
    lifetime_risk_percent: number;
    interpretation: string;
  };
  bcc_risk: {
    relative_risk: number;
    interpretation: string;
  };
  scc_risk: {
    relative_risk: number;
    interpretation: string;
  };
  component_scores: {
    genetic: number;
    phenotype: number;
    sun_exposure: number;
    behavioral: number;
    medical_history: number;
    ai_findings: number;
  };
  risk_factors: Array<{
    factor: string;
    category: string;
    impact: string;
    risk_multiplier: number;
  }>;
  recommendations: string[];
  screening_recommendations: {
    self_exam_frequency: string;
    professional_exam_frequency: string;
    urgent_referral: boolean;
  };
  comparison_to_previous: {
    has_previous: boolean;
    risk_change: number | null;
    trend: string | null;
  };
  confidence_score: number;
  created_at: string;
}

interface AssessmentInput {
  age: string;
  gender: string;
  fitzpatrick_type: number;
  natural_hair_color: string;
  natural_eye_color: string;
  freckles: string;
  total_mole_count: string;
  sun_exposure_level: number;
  childhood_severe_sunburns: string;
  childhood_mild_sunburns: string;
  adult_severe_sunburns: string;
  adult_mild_sunburns: string;
  has_family_history: boolean;
  tanning_bed_use: boolean;
  outdoor_occupation: boolean;
  immunosuppressed: boolean;
  include_ai_findings: boolean;
}

const FITZPATRICK_TYPES = [
  { value: 1, label: 'Type I - Very fair, always burns', color: '#fce4ec' },
  { value: 2, label: 'Type II - Fair, usually burns', color: '#f8bbd0' },
  { value: 3, label: 'Type III - Medium, sometimes burns', color: '#e1bee7' },
  { value: 4, label: 'Type IV - Olive, rarely burns', color: '#d7ccc8' },
  { value: 5, label: 'Type V - Brown, very rarely burns', color: '#a1887f' },
  { value: 6, label: 'Type VI - Dark brown/black, never burns', color: '#6d4c41' },
];

const HAIR_COLORS = ['red', 'blonde', 'light_brown', 'dark_brown', 'black'];
const EYE_COLORS = ['blue', 'green', 'hazel', 'brown'];
const FRECKLE_OPTIONS = ['none', 'few', 'some', 'many'];
const MOLE_COUNT_OPTIONS = ['none', 'few', 'some', 'many', 'very_many'];

export default function RiskCalculatorScreen() {
  const { t } = useTranslation();
  const { isAuthenticated, logout } = useAuth();
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [loadingLatest, setLoadingLatest] = useState(true);
  const [currentStep, setCurrentStep] = useState(0);
  const [latestAssessment, setLatestAssessment] = useState<RiskAssessment | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState<RiskAssessment | null>(null);

  const [input, setInput] = useState<AssessmentInput>({
    age: '40',
    gender: 'male',
    fitzpatrick_type: 3,
    natural_hair_color: 'dark_brown',
    natural_eye_color: 'brown',
    freckles: 'few',
    total_mole_count: 'some',
    sun_exposure_level: 3,
    childhood_severe_sunburns: '0',
    childhood_mild_sunburns: '2',
    adult_severe_sunburns: '0',
    adult_mild_sunburns: '2',
    has_family_history: false,
    tanning_bed_use: false,
    outdoor_occupation: false,
    immunosuppressed: false,
    include_ai_findings: true,
  });

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    } else {
      loadLatestAssessment();
    }
  }, [isAuthenticated]);

  const loadLatestAssessment = async () => {
    try {
      const token = AuthService.getToken();
      if (!token) return;

      const response = await fetch(`${API_BASE_URL}/risk-calculator/latest`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        if (data.has_assessment) {
          setLatestAssessment(data);
        }
      }
    } catch (error) {
      console.error('Error loading latest assessment:', error);
    } finally {
      setLoadingLatest(false);
    }
  };

  const calculateRisk = async () => {
    setLoading(true);
    try {
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert(t('riskCalculator.error'), t('riskCalculator.pleaseLoginAgain'));
        logout();
        return;
      }

      const requestData = {
        age: parseInt(input.age) || 40,
        gender: input.gender,
        fitzpatrick_type: input.fitzpatrick_type,
        natural_hair_color: input.natural_hair_color,
        natural_eye_color: input.natural_eye_color,
        freckles: input.freckles,
        total_mole_count: input.total_mole_count,
        sun_exposure_level: input.sun_exposure_level,
        sunburn_history: {
          childhood_severe: parseInt(input.childhood_severe_sunburns) || 0,
          childhood_mild: parseInt(input.childhood_mild_sunburns) || 0,
          adult_severe: parseInt(input.adult_severe_sunburns) || 0,
          adult_mild: parseInt(input.adult_mild_sunburns) || 0,
        },
        family_history: input.has_family_history ? [
          { relationship: 'parent', cancer_type: 'skin_cancer', age_at_diagnosis: 60 }
        ] : [],
        personal_history: [],
        tanning_bed_use: input.tanning_bed_use,
        outdoor_occupation: input.outdoor_occupation,
        immunosuppressed: input.immunosuppressed,
        include_ai_findings: input.include_ai_findings,
      };

      const response = await fetch(`${API_BASE_URL}/risk-calculator/calculate`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data);
        setShowResults(true);
      } else if (response.status === 401) {
        Alert.alert(t('riskCalculator.sessionExpired'), t('riskCalculator.pleaseLoginAgain'));
        logout();
      } else {
        const errorData = await response.json();
        Alert.alert(t('riskCalculator.error'), errorData.detail || t('riskCalculator.calculationError'));
      }
    } catch (error) {
      console.error('Error calculating risk:', error);
      Alert.alert(t('riskCalculator.error'), t('riskCalculator.networkError'));
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevelColor = (category: string) => {
    switch (category) {
      case 'very_high': return '#dc2626';
      case 'high': return '#f59e0b';
      case 'moderate': return '#eab308';
      case 'low': return '#10b981';
      case 'very_low': return '#06b6d4';
      default: return '#6b7280';
    }
  };

  const getTrendIcon = (trend: string | null) => {
    if (trend === 'improving') return 'trending-down';
    if (trend === 'worsening') return 'trending-up';
    return 'remove';
  };

  const getTrendColor = (trend: string | null) => {
    if (trend === 'improving') return '#10b981';
    if (trend === 'worsening') return '#dc2626';
    return '#6b7280';
  };

  const renderStepIndicator = () => (
    <View style={styles.stepIndicator}>
      {[0, 1, 2, 3].map((step) => (
        <View
          key={step}
          style={[
            styles.stepDot,
            currentStep >= step && styles.stepDotActive
          ]}
        />
      ))}
    </View>
  );

  const renderStep0 = () => (
    <View style={styles.stepContainer}>
      <Text style={styles.stepTitle}>{t('riskCalculator.basicInfo')}</Text>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>{t('riskCalculator.age')}</Text>
        <TextInput
          style={styles.textInput}
          value={input.age}
          onChangeText={(text) => setInput({ ...input, age: text })}
          keyboardType="numeric"
          placeholder="40"
        />
      </View>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>{t('riskCalculator.gender')}</Text>
        <View style={styles.buttonGroup}>
          {['male', 'female'].map((g) => (
            <TouchableOpacity
              key={g}
              style={[styles.optionButton, input.gender === g && styles.optionButtonActive]}
              onPress={() => setInput({ ...input, gender: g })}
            >
              <Text style={[styles.optionText, input.gender === g && styles.optionTextActive]}>
                {t(`riskCalculator.${g}`)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>{t('riskCalculator.skinType')}</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <View style={styles.fitzpatrickContainer}>
            {FITZPATRICK_TYPES.map((type) => (
              <TouchableOpacity
                key={type.value}
                style={[
                  styles.fitzpatrickButton,
                  { backgroundColor: type.color },
                  input.fitzpatrick_type === type.value && styles.fitzpatrickButtonActive
                ]}
                onPress={() => setInput({ ...input, fitzpatrick_type: type.value })}
              >
                <Text style={[
                  styles.fitzpatrickText,
                  type.value > 3 && { color: 'white' },
                  input.fitzpatrick_type === type.value && styles.fitzpatrickTextActive
                ]}>
                  {type.value}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </ScrollView>
        <Text style={styles.fitzpatrickDescription}>
          {FITZPATRICK_TYPES.find(t => t.value === input.fitzpatrick_type)?.label}
        </Text>
      </View>
    </View>
  );

  const renderStep1 = () => (
    <View style={styles.stepContainer}>
      <Text style={styles.stepTitle}>{t('riskCalculator.phenotype')}</Text>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>{t('riskCalculator.hairColor')}</Text>
        <View style={styles.buttonGroup}>
          {HAIR_COLORS.map((color) => (
            <TouchableOpacity
              key={color}
              style={[styles.optionButton, styles.smallOption, input.natural_hair_color === color && styles.optionButtonActive]}
              onPress={() => setInput({ ...input, natural_hair_color: color })}
            >
              <Text style={[styles.optionText, styles.smallText, input.natural_hair_color === color && styles.optionTextActive]}>
                {t(`riskCalculator.hairColors.${color}`)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>{t('riskCalculator.eyeColor')}</Text>
        <View style={styles.buttonGroup}>
          {EYE_COLORS.map((color) => (
            <TouchableOpacity
              key={color}
              style={[styles.optionButton, styles.smallOption, input.natural_eye_color === color && styles.optionButtonActive]}
              onPress={() => setInput({ ...input, natural_eye_color: color })}
            >
              <Text style={[styles.optionText, styles.smallText, input.natural_eye_color === color && styles.optionTextActive]}>
                {t(`riskCalculator.eyeColors.${color}`)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>{t('riskCalculator.freckles')}</Text>
        <View style={styles.buttonGroup}>
          {FRECKLE_OPTIONS.map((option) => (
            <TouchableOpacity
              key={option}
              style={[styles.optionButton, styles.smallOption, input.freckles === option && styles.optionButtonActive]}
              onPress={() => setInput({ ...input, freckles: option })}
            >
              <Text style={[styles.optionText, styles.smallText, input.freckles === option && styles.optionTextActive]}>
                {t(`riskCalculator.freckleOptions.${option}`)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>{t('riskCalculator.moleCount')}</Text>
        <View style={styles.buttonGroup}>
          {MOLE_COUNT_OPTIONS.map((option) => (
            <TouchableOpacity
              key={option}
              style={[styles.optionButton, styles.smallOption, input.total_mole_count === option && styles.optionButtonActive]}
              onPress={() => setInput({ ...input, total_mole_count: option })}
            >
              <Text style={[styles.optionText, styles.smallText, input.total_mole_count === option && styles.optionTextActive]}>
                {t(`riskCalculator.moleOptions.${option}`)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    </View>
  );

  const renderStep2 = () => (
    <View style={styles.stepContainer}>
      <Text style={styles.stepTitle}>{t('riskCalculator.sunExposure')}</Text>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>{t('riskCalculator.exposureLevel')}</Text>
        <View style={styles.sliderContainer}>
          {[1, 2, 3, 4, 5].map((level) => (
            <TouchableOpacity
              key={level}
              style={[
                styles.levelButton,
                input.sun_exposure_level === level && styles.levelButtonActive
              ]}
              onPress={() => setInput({ ...input, sun_exposure_level: level })}
            >
              <Text style={[
                styles.levelText,
                input.sun_exposure_level === level && styles.levelTextActive
              ]}>
                {level}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
        <View style={styles.levelLabels}>
          <Text style={styles.levelLabelText}>{t('riskCalculator.minimal')}</Text>
          <Text style={styles.levelLabelText}>{t('riskCalculator.veryHigh')}</Text>
        </View>
      </View>

      <View style={styles.sunburnSection}>
        <Text style={styles.sunburnTitle}>{t('riskCalculator.sunburnHistory')}</Text>

        <View style={styles.sunburnRow}>
          <Text style={styles.sunburnLabel}>{t('riskCalculator.childhoodSevere')}</Text>
          <TextInput
            style={styles.sunburnInput}
            value={input.childhood_severe_sunburns}
            onChangeText={(text) => setInput({ ...input, childhood_severe_sunburns: text })}
            keyboardType="numeric"
            placeholder="0"
          />
        </View>

        <View style={styles.sunburnRow}>
          <Text style={styles.sunburnLabel}>{t('riskCalculator.childhoodMild')}</Text>
          <TextInput
            style={styles.sunburnInput}
            value={input.childhood_mild_sunburns}
            onChangeText={(text) => setInput({ ...input, childhood_mild_sunburns: text })}
            keyboardType="numeric"
            placeholder="0"
          />
        </View>

        <View style={styles.sunburnRow}>
          <Text style={styles.sunburnLabel}>{t('riskCalculator.adultSevere')}</Text>
          <TextInput
            style={styles.sunburnInput}
            value={input.adult_severe_sunburns}
            onChangeText={(text) => setInput({ ...input, adult_severe_sunburns: text })}
            keyboardType="numeric"
            placeholder="0"
          />
        </View>

        <View style={styles.sunburnRow}>
          <Text style={styles.sunburnLabel}>{t('riskCalculator.adultMild')}</Text>
          <TextInput
            style={styles.sunburnInput}
            value={input.adult_mild_sunburns}
            onChangeText={(text) => setInput({ ...input, adult_mild_sunburns: text })}
            keyboardType="numeric"
            placeholder="0"
          />
        </View>
      </View>
    </View>
  );

  const renderStep3 = () => (
    <View style={styles.stepContainer}>
      <Text style={styles.stepTitle}>{t('riskCalculator.additionalFactors')}</Text>

      <TouchableOpacity
        style={styles.toggleRow}
        onPress={() => setInput({ ...input, has_family_history: !input.has_family_history })}
      >
        <View style={styles.toggleInfo}>
          <Ionicons name="people" size={24} color="#8b5cf6" />
          <Text style={styles.toggleLabel}>{t('riskCalculator.familyHistory')}</Text>
        </View>
        <View style={[styles.toggle, input.has_family_history && styles.toggleActive]}>
          <View style={[styles.toggleThumb, input.has_family_history && styles.toggleThumbActive]} />
        </View>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.toggleRow}
        onPress={() => setInput({ ...input, tanning_bed_use: !input.tanning_bed_use })}
      >
        <View style={styles.toggleInfo}>
          <Ionicons name="sunny" size={24} color="#f59e0b" />
          <Text style={styles.toggleLabel}>{t('riskCalculator.tanningBed')}</Text>
        </View>
        <View style={[styles.toggle, input.tanning_bed_use && styles.toggleActive]}>
          <View style={[styles.toggleThumb, input.tanning_bed_use && styles.toggleThumbActive]} />
        </View>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.toggleRow}
        onPress={() => setInput({ ...input, outdoor_occupation: !input.outdoor_occupation })}
      >
        <View style={styles.toggleInfo}>
          <Ionicons name="briefcase" size={24} color="#10b981" />
          <Text style={styles.toggleLabel}>{t('riskCalculator.outdoorWork')}</Text>
        </View>
        <View style={[styles.toggle, input.outdoor_occupation && styles.toggleActive]}>
          <View style={[styles.toggleThumb, input.outdoor_occupation && styles.toggleThumbActive]} />
        </View>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.toggleRow}
        onPress={() => setInput({ ...input, immunosuppressed: !input.immunosuppressed })}
      >
        <View style={styles.toggleInfo}>
          <Ionicons name="medical" size={24} color="#dc2626" />
          <Text style={styles.toggleLabel}>{t('riskCalculator.immunosuppressed')}</Text>
        </View>
        <View style={[styles.toggle, input.immunosuppressed && styles.toggleActive]}>
          <View style={[styles.toggleThumb, input.immunosuppressed && styles.toggleThumbActive]} />
        </View>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.toggleRow}
        onPress={() => setInput({ ...input, include_ai_findings: !input.include_ai_findings })}
      >
        <View style={styles.toggleInfo}>
          <Ionicons name="analytics" size={24} color="#6366f1" />
          <Text style={styles.toggleLabel}>{t('riskCalculator.includeAIFindings')}</Text>
        </View>
        <View style={[styles.toggle, input.include_ai_findings && styles.toggleActive]}>
          <View style={[styles.toggleThumb, input.include_ai_findings && styles.toggleThumbActive]} />
        </View>
      </TouchableOpacity>
    </View>
  );

  const renderResults = () => {
    if (!results) return null;

    return (
      <ScrollView style={styles.resultsContainer}>
        {/* Overall Risk Score */}
        <View style={styles.overallRiskCard}>
          <Text style={styles.overallRiskTitle}>{t('riskCalculator.yourRiskScore')}</Text>
          <View style={styles.scoreCircle}>
            <Text style={[styles.scoreNumber, { color: getRiskLevelColor(results.risk_category) }]}>
              {results.overall_risk_score.toFixed(0)}
            </Text>
            <Text style={styles.scoreMax}>/100</Text>
          </View>
          <Text style={[styles.riskCategory, { color: getRiskLevelColor(results.risk_category) }]}>
            {t(`riskCalculator.categories.${results.risk_category}`)}
          </Text>
          <Text style={styles.riskDescription}>{results.risk_category_description}</Text>

          {results.comparison_to_previous.has_previous && results.comparison_to_previous.trend && (
            <View style={styles.trendContainer}>
              <Ionicons
                name={getTrendIcon(results.comparison_to_previous.trend)}
                size={20}
                color={getTrendColor(results.comparison_to_previous.trend)}
              />
              <Text style={[styles.trendText, { color: getTrendColor(results.comparison_to_previous.trend) }]}>
                {results.comparison_to_previous.risk_change !== null &&
                  `${results.comparison_to_previous.risk_change > 0 ? '+' : ''}${results.comparison_to_previous.risk_change.toFixed(1)} `}
                {t(`riskCalculator.trends.${results.comparison_to_previous.trend}`)}
              </Text>
            </View>
          )}
        </View>

        {/* Melanoma Risk */}
        <View style={styles.cancerRiskCard}>
          <Text style={styles.cancerRiskTitle}>{t('riskCalculator.melanomaRisk')}</Text>
          <View style={styles.cancerRiskRow}>
            <Text style={styles.cancerRiskLabel}>{t('riskCalculator.relativeRisk')}</Text>
            <Text style={styles.cancerRiskValue}>{results.melanoma_risk.relative_risk.toFixed(2)}x</Text>
          </View>
          <View style={styles.cancerRiskRow}>
            <Text style={styles.cancerRiskLabel}>{t('riskCalculator.lifetimeRisk')}</Text>
            <Text style={styles.cancerRiskValue}>{results.melanoma_risk.lifetime_risk_percent.toFixed(1)}%</Text>
          </View>
          <Text style={styles.interpretation}>{results.melanoma_risk.interpretation}</Text>
        </View>

        {/* Component Scores */}
        <View style={styles.componentScoresCard}>
          <Text style={styles.componentTitle}>{t('riskCalculator.componentScores')}</Text>
          {Object.entries(results.component_scores).map(([key, value]) => (
            <View key={key} style={styles.componentRow}>
              <Text style={styles.componentLabel}>{t(`riskCalculator.components.${key}`)}</Text>
              <View style={styles.componentBarContainer}>
                <View style={[styles.componentBar, { width: `${value}%` }]} />
              </View>
              <Text style={styles.componentValue}>{value?.toFixed(0) || 0}</Text>
            </View>
          ))}
        </View>

        {/* Risk Factors */}
        {results.risk_factors && results.risk_factors.length > 0 && (
          <View style={styles.riskFactorsCard}>
            <Text style={styles.riskFactorsTitle}>{t('riskCalculator.identifiedRiskFactors')}</Text>
            {results.risk_factors.map((factor, index) => (
              <View key={index} style={styles.riskFactorItem}>
                <Ionicons
                  name={factor.impact === 'high' ? 'warning' : 'information-circle'}
                  size={20}
                  color={factor.impact === 'high' ? '#dc2626' : '#f59e0b'}
                />
                <View style={styles.riskFactorInfo}>
                  <Text style={styles.riskFactorText}>{factor.factor}</Text>
                  <Text style={styles.riskFactorMultiplier}>
                    {factor.risk_multiplier}x {t('riskCalculator.riskMultiplier')}
                  </Text>
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Recommendations */}
        {results.recommendations && results.recommendations.length > 0 && (
          <View style={styles.recommendationsCard}>
            <Text style={styles.recommendationsTitle}>{t('riskCalculator.recommendations')}</Text>
            {results.recommendations.map((rec, index) => (
              <View key={index} style={styles.recommendationItem}>
                <Ionicons name="checkmark-circle" size={20} color="#10b981" />
                <Text style={styles.recommendationText}>{rec}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Screening Schedule */}
        <View style={styles.screeningCard}>
          <Text style={styles.screeningTitle}>{t('riskCalculator.screeningSchedule')}</Text>
          <View style={styles.screeningRow}>
            <Ionicons name="eye" size={24} color="#8b5cf6" />
            <View style={styles.screeningInfo}>
              <Text style={styles.screeningLabel}>{t('riskCalculator.selfExam')}</Text>
              <Text style={styles.screeningValue}>
                {t(`riskCalculator.frequencies.${results.screening_recommendations.self_exam_frequency}`)}
              </Text>
            </View>
          </View>
          <View style={styles.screeningRow}>
            <Ionicons name="medical" size={24} color="#6366f1" />
            <View style={styles.screeningInfo}>
              <Text style={styles.screeningLabel}>{t('riskCalculator.professionalExam')}</Text>
              <Text style={styles.screeningValue}>
                {t(`riskCalculator.frequencies.${results.screening_recommendations.professional_exam_frequency}`)}
              </Text>
            </View>
          </View>
          {results.screening_recommendations.urgent_referral && (
            <View style={styles.urgentReferral}>
              <Ionicons name="alert-circle" size={24} color="#dc2626" />
              <Text style={styles.urgentReferralText}>{t('riskCalculator.urgentReferral')}</Text>
            </View>
          )}
        </View>

        {/* New Assessment Button */}
        <TouchableOpacity
          style={styles.newAssessmentButton}
          onPress={() => {
            setShowResults(false);
            setCurrentStep(0);
          }}
        >
          <Text style={styles.newAssessmentText}>{t('riskCalculator.newAssessment')}</Text>
        </TouchableOpacity>
      </ScrollView>
    );
  };

  const renderLatestAssessmentSummary = () => {
    if (loadingLatest) {
      return (
        <View style={styles.latestCard}>
          <ActivityIndicator size="small" color="#8b5cf6" />
        </View>
      );
    }

    if (!latestAssessment) {
      return (
        <View style={styles.latestCard}>
          <Ionicons name="information-circle" size={24} color="#6b7280" />
          <Text style={styles.noAssessmentText}>{t('riskCalculator.noAssessment')}</Text>
        </View>
      );
    }

    return (
      <TouchableOpacity
        style={styles.latestCard}
        onPress={() => {
          setResults(latestAssessment);
          setShowResults(true);
        }}
      >
        <View style={styles.latestHeader}>
          <Text style={styles.latestTitle}>{t('riskCalculator.latestAssessment')}</Text>
          <Ionicons name="chevron-forward" size={20} color="#6b7280" />
        </View>
        <View style={styles.latestContent}>
          <View style={styles.latestScoreContainer}>
            <Text style={[styles.latestScore, { color: getRiskLevelColor(latestAssessment.risk_category) }]}>
              {latestAssessment.overall_risk_score.toFixed(0)}
            </Text>
            <Text style={styles.latestScoreMax}>/100</Text>
          </View>
          <Text style={[styles.latestCategory, { color: getRiskLevelColor(latestAssessment.risk_category) }]}>
            {t(`riskCalculator.categories.${latestAssessment.risk_category}`)}
          </Text>
        </View>
      </TouchableOpacity>
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#8b5cf6" />
        <Text style={styles.loadingText}>{t('riskCalculator.calculating')}</Text>
      </View>
    );
  }

  return (
    <LinearGradient colors={['#7c3aed', '#8b5cf6', '#a78bfa']} style={styles.container}>
      <ScrollView style={styles.scrollView}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.title}>{t('riskCalculator.title')}</Text>
          <View style={styles.placeholder} />
        </View>

        {showResults ? (
          renderResults()
        ) : (
          <>
            {/* Latest Assessment Summary */}
            {renderLatestAssessmentSummary()}

            {/* Assessment Form */}
            <View style={styles.formCard}>
              <Text style={styles.formTitle}>{t('riskCalculator.newAssessment')}</Text>
              {renderStepIndicator()}

              {currentStep === 0 && renderStep0()}
              {currentStep === 1 && renderStep1()}
              {currentStep === 2 && renderStep2()}
              {currentStep === 3 && renderStep3()}

              {/* Navigation Buttons */}
              <View style={styles.navButtons}>
                {currentStep > 0 && (
                  <TouchableOpacity
                    style={styles.prevButton}
                    onPress={() => setCurrentStep(currentStep - 1)}
                  >
                    <Ionicons name="arrow-back" size={20} color="#8b5cf6" />
                    <Text style={styles.prevButtonText}>{t('riskCalculator.previous')}</Text>
                  </TouchableOpacity>
                )}

                {currentStep < 3 ? (
                  <TouchableOpacity
                    style={styles.nextButton}
                    onPress={() => setCurrentStep(currentStep + 1)}
                  >
                    <Text style={styles.nextButtonText}>{t('riskCalculator.next')}</Text>
                    <Ionicons name="arrow-forward" size={20} color="white" />
                  </TouchableOpacity>
                ) : (
                  <TouchableOpacity
                    style={styles.calculateButton}
                    onPress={calculateRisk}
                  >
                    <Ionicons name="calculator" size={20} color="white" />
                    <Text style={styles.calculateButtonText}>{t('riskCalculator.calculateRisk')}</Text>
                  </TouchableOpacity>
                )}
              </View>
            </View>
          </>
        )}
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#7c3aed',
  },
  loadingText: {
    marginTop: 16,
    color: 'white',
    fontSize: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 20,
  },
  backButton: {
    padding: 8,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
  },
  placeholder: {
    width: 40,
  },
  latestCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    marginHorizontal: 20,
    marginBottom: 16,
  },
  latestHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  latestTitle: {
    fontSize: 14,
    color: '#6b7280',
  },
  latestContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  latestScoreContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginRight: 16,
  },
  latestScore: {
    fontSize: 32,
    fontWeight: 'bold',
  },
  latestScoreMax: {
    fontSize: 16,
    color: '#6b7280',
  },
  latestCategory: {
    fontSize: 16,
    fontWeight: '600',
  },
  noAssessmentText: {
    marginLeft: 8,
    color: '#6b7280',
  },
  formCard: {
    backgroundColor: 'white',
    borderRadius: 24,
    padding: 24,
    marginHorizontal: 20,
    marginBottom: 40,
  },
  formTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
    textAlign: 'center',
  },
  stepIndicator: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 24,
  },
  stepDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#e5e7eb',
    marginHorizontal: 4,
  },
  stepDotActive: {
    backgroundColor: '#8b5cf6',
  },
  stepContainer: {
    marginBottom: 24,
  },
  stepTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 16,
  },
  inputGroup: {
    marginBottom: 20,
  },
  inputLabel: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 8,
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
  },
  buttonGroup: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  optionButton: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    backgroundColor: 'white',
  },
  smallOption: {
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  optionButtonActive: {
    backgroundColor: '#8b5cf6',
    borderColor: '#8b5cf6',
  },
  optionText: {
    color: '#374151',
  },
  smallText: {
    fontSize: 12,
  },
  optionTextActive: {
    color: 'white',
  },
  fitzpatrickContainer: {
    flexDirection: 'row',
    gap: 8,
  },
  fitzpatrickButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  fitzpatrickButtonActive: {
    borderColor: '#8b5cf6',
  },
  fitzpatrickText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  fitzpatrickTextActive: {
    fontWeight: 'bold',
  },
  fitzpatrickDescription: {
    marginTop: 8,
    fontSize: 12,
    color: '#6b7280',
    textAlign: 'center',
  },
  sliderContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  levelButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  levelButtonActive: {
    backgroundColor: '#8b5cf6',
    borderColor: '#8b5cf6',
  },
  levelText: {
    fontSize: 16,
    color: '#374151',
  },
  levelTextActive: {
    color: 'white',
  },
  levelLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  levelLabelText: {
    fontSize: 12,
    color: '#6b7280',
  },
  sunburnSection: {
    marginTop: 16,
  },
  sunburnTitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 12,
  },
  sunburnRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  sunburnLabel: {
    flex: 1,
    fontSize: 14,
    color: '#374151',
  },
  sunburnInput: {
    width: 60,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 8,
    padding: 8,
    textAlign: 'center',
  },
  toggleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  toggleInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  toggleLabel: {
    marginLeft: 12,
    fontSize: 14,
    color: '#374151',
  },
  toggle: {
    width: 48,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#e5e7eb',
    justifyContent: 'center',
    padding: 2,
  },
  toggleActive: {
    backgroundColor: '#8b5cf6',
  },
  toggleThumb: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: 'white',
  },
  toggleThumbActive: {
    alignSelf: 'flex-end',
  },
  navButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 24,
  },
  prevButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  prevButtonText: {
    marginLeft: 8,
    color: '#8b5cf6',
    fontSize: 16,
  },
  nextButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#8b5cf6',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    marginLeft: 'auto',
  },
  nextButtonText: {
    color: 'white',
    fontSize: 16,
    marginRight: 8,
  },
  calculateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#10b981',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    marginLeft: 'auto',
  },
  calculateButtonText: {
    color: 'white',
    fontSize: 16,
    marginLeft: 8,
  },
  resultsContainer: {
    flex: 1,
    paddingHorizontal: 20,
  },
  overallRiskCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    marginBottom: 16,
  },
  overallRiskTitle: {
    fontSize: 16,
    color: '#6b7280',
    marginBottom: 16,
  },
  scoreCircle: {
    flexDirection: 'row',
    alignItems: 'baseline',
  },
  scoreNumber: {
    fontSize: 64,
    fontWeight: 'bold',
  },
  scoreMax: {
    fontSize: 24,
    color: '#6b7280',
  },
  riskCategory: {
    fontSize: 20,
    fontWeight: '600',
    marginTop: 8,
  },
  riskDescription: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginTop: 8,
  },
  trendContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
  },
  trendText: {
    marginLeft: 8,
    fontSize: 14,
  },
  cancerRiskCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  cancerRiskTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  cancerRiskRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  cancerRiskLabel: {
    color: '#6b7280',
  },
  cancerRiskValue: {
    fontWeight: '600',
    color: '#1f2937',
  },
  interpretation: {
    fontSize: 12,
    color: '#8b5cf6',
    fontStyle: 'italic',
    marginTop: 8,
  },
  componentScoresCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  componentTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  componentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  componentLabel: {
    width: 100,
    fontSize: 12,
    color: '#6b7280',
  },
  componentBarContainer: {
    flex: 1,
    height: 8,
    backgroundColor: '#e5e7eb',
    borderRadius: 4,
    marginHorizontal: 8,
  },
  componentBar: {
    height: '100%',
    backgroundColor: '#8b5cf6',
    borderRadius: 4,
  },
  componentValue: {
    width: 30,
    textAlign: 'right',
    fontSize: 12,
    color: '#374151',
  },
  riskFactorsCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  riskFactorsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  riskFactorItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  riskFactorInfo: {
    marginLeft: 12,
    flex: 1,
  },
  riskFactorText: {
    fontSize: 14,
    color: '#374151',
  },
  riskFactorMultiplier: {
    fontSize: 12,
    color: '#6b7280',
  },
  recommendationsCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  recommendationsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  recommendationText: {
    marginLeft: 12,
    flex: 1,
    fontSize: 14,
    color: '#374151',
  },
  screeningCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  screeningTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  screeningRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  screeningInfo: {
    marginLeft: 12,
  },
  screeningLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
  screeningValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
  },
  urgentReferral: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fef2f2',
    padding: 12,
    borderRadius: 8,
    marginTop: 8,
  },
  urgentReferralText: {
    marginLeft: 12,
    color: '#dc2626',
    fontWeight: '600',
  },
  newAssessmentButton: {
    backgroundColor: '#8b5cf6',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 40,
  },
  newAssessmentText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});
