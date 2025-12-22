import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Platform,
  Modal,
  TextInput,
  RefreshControl,
  Share,
  Linking
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';

interface Location {
  facility: string | null;
  city: string | null;
  state: string | null;
  country: string | null;
  zip: string | null;
}

interface Trial {
  id: number;
  nct_id: string;
  title: string;
  brief_summary: string | null;
  phase: string | null;
  status: string;
  conditions: string[];
  interventions: { type: string; name: string; description: string }[];
  eligibility_criteria: string | null;
  min_age: number | null;
  max_age: number | null;
  gender: string | null;
  locations: Location[];
  contact_name: string | null;
  contact_email: string | null;
  contact_phone: string | null;
  sponsor: string | null;
  url: string | null;
  // Biomarker/genetic requirements
  required_biomarkers?: string[];
  excluded_biomarkers?: string[];
  genetic_requirements?: any[];
  biomarker_keywords?: string[];
  requires_genetic_testing?: boolean;
  targeted_therapy_trial?: boolean;
}

interface TrialMatch {
  trial: Trial;
  trial_id: number;
  match_score: number;
  match_reasons: string[];
  unmet_criteria: string[];
  matched_conditions: string[];
  diagnosis_score: number;
  distance_miles: number | null;
  nearest_location: Location | null;
  age_eligible: boolean;
  gender_eligible: boolean;
  // Genetic matching fields
  genetic_score?: number;
  matched_biomarkers?: string[];
  missing_biomarkers?: string[];
  excluded_biomarkers_found?: string[];
  genetic_eligible?: boolean;
  genetic_match_type?: string;
}

interface TrialInterest {
  id: number;
  trial_id: number;
  trial: Trial;
  interest_level: string;
  preferred_contact: string;
  notes: string | null;
  status: string;
  expressed_at: string;
}

interface NewMatchAlert {
  new_matches_count: number;
  new_trials: Trial[];
  last_checked: string;
}

interface EligibilityQuestion {
  id: string;
  question: string;
  type: 'boolean' | 'number' | 'select' | 'text';
  options?: string[];
  required: boolean;
  category: string;
}

interface EligibilityResult {
  eligible: boolean;
  score: number;
  met_criteria: string[];
  unmet_criteria: string[];
  uncertain_criteria: string[];
  recommendation: string;
}

interface MapLocation {
  facility: string;
  city: string;
  state: string;
  country: string;
  latitude: number;
  longitude: number;
  distance_miles: number | null;
}

type TabType = 'matches' | 'all' | 'interests';

export default function ClinicalTrialsScreen() {
  const { t } = useTranslation();
  const { isAuthenticated, logout } = useAuth();
  const router = useRouter();

  const [activeTab, setActiveTab] = useState<TabType>('matches');
  const [matches, setMatches] = useState<TrialMatch[]>([]);
  const [allTrials, setAllTrials] = useState<Trial[]>([]);
  const [totalTrials, setTotalTrials] = useState<number>(0);
  const [interests, setInterests] = useState<TrialInterest[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Detail modal
  const [selectedTrial, setSelectedTrial] = useState<Trial | null>(null);
  const [selectedMatch, setSelectedMatch] = useState<TrialMatch | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  // Interest form modal
  const [showInterestModal, setShowInterestModal] = useState(false);
  const [interestTrialId, setInterestTrialId] = useState<number | null>(null);
  const [interestLevel, setInterestLevel] = useState('exploring');
  const [preferredContact, setPreferredContact] = useState('email');
  const [interestNotes, setInterestNotes] = useState('');
  const [submittingInterest, setSubmittingInterest] = useState(false);

  // New trial alerts
  const [newMatchAlert, setNewMatchAlert] = useState<NewMatchAlert | null>(null);
  const [alertsDismissed, setAlertsDismissed] = useState(false);

  // Eligibility checker modal
  const [showEligibilityModal, setShowEligibilityModal] = useState(false);
  const [eligibilityQuestions, setEligibilityQuestions] = useState<EligibilityQuestion[]>([]);
  const [eligibilityAnswers, setEligibilityAnswers] = useState<Record<string, any>>({});
  const [eligibilityResult, setEligibilityResult] = useState<EligibilityResult | null>(null);
  const [checkingEligibility, setCheckingEligibility] = useState(false);
  const [loadingQuestions, setLoadingQuestions] = useState(false);

  // Map modal
  const [showMapModal, setShowMapModal] = useState(false);
  const [mapLocations, setMapLocations] = useState<MapLocation[]>([]);
  const [loadingMap, setLoadingMap] = useState(false);

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    } else {
      loadData();
    }
  }, [isAuthenticated]);

  const loadData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadMatches(),
        loadAllTrials(),
        loadInterests(),
        checkNewMatchAlerts()
      ]);
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  const loadMatches = async () => {
    try {
      const token = AuthService.getToken();
      if (!token) return;

      const response = await fetch(`${API_BASE_URL}/clinical-trials/matches`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setMatches(data.matches || []);
      } else if (response.status === 401) {
        logout();
      }
    } catch (error) {
      console.error('Error loading matches:', error);
    }
  };

  const loadAllTrials = async () => {
    try {
      const token = AuthService.getToken();
      if (!token) return;

      const response = await fetch(`${API_BASE_URL}/clinical-trials?limit=50`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setAllTrials(data.trials || []);
        setTotalTrials(data.total || 0);
      }
    } catch (error) {
      console.error('Error loading trials:', error);
    }
  };

  const loadInterests = async () => {
    try {
      const token = AuthService.getToken();
      if (!token) return;

      const response = await fetch(`${API_BASE_URL}/clinical-trials/interests`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setInterests(data.interests || []);
      }
    } catch (error) {
      console.error('Error loading interests:', error);
    }
  };

  const checkNewMatchAlerts = async () => {
    try {
      const token = AuthService.getToken();
      if (!token) return;

      const response = await fetch(`${API_BASE_URL}/clinical-trials/alerts/new-matches`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        if (data.new_matches_count > 0) {
          setNewMatchAlert(data);
          setAlertsDismissed(false);
        }
      }
    } catch (error) {
      console.error('Error checking alerts:', error);
    }
  };

  const shareTrialWithDoctor = async (trial: Trial) => {
    try {
      const token = AuthService.getToken();
      if (!token) return;

      const response = await fetch(`${API_BASE_URL}/clinical-trials/${trial.id}/share?format=text`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        await Share.share({
          message: data.content,
          title: `Clinical Trial: ${trial.title}`
        });
      } else {
        Alert.alert('Error', 'Failed to generate share content');
      }
    } catch (error) {
      console.error('Error sharing trial:', error);
      Alert.alert('Error', 'Failed to share trial');
    }
  };

  const loadEligibilityQuestions = async (trialId: number) => {
    setLoadingQuestions(true);
    setEligibilityResult(null);
    setEligibilityAnswers({});
    try {
      const token = AuthService.getToken();
      if (!token) return;

      const response = await fetch(`${API_BASE_URL}/clinical-trials/${trialId}/eligibility-questions`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setEligibilityQuestions(data.questions || []);
        setShowEligibilityModal(true);
      } else {
        Alert.alert('Error', 'Failed to load eligibility questions');
      }
    } catch (error) {
      console.error('Error loading eligibility questions:', error);
      Alert.alert('Error', 'Failed to load eligibility questions');
    } finally {
      setLoadingQuestions(false);
    }
  };

  const checkEligibility = async () => {
    if (!selectedTrial) return;

    setCheckingEligibility(true);
    try {
      const token = AuthService.getToken();
      if (!token) return;

      const response = await fetch(`${API_BASE_URL}/clinical-trials/${selectedTrial.id}/check-eligibility`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ answers: eligibilityAnswers })
      });

      if (response.ok) {
        const data = await response.json();
        setEligibilityResult(data);
      } else {
        Alert.alert('Error', 'Failed to check eligibility');
      }
    } catch (error) {
      console.error('Error checking eligibility:', error);
      Alert.alert('Error', 'Failed to check eligibility');
    } finally {
      setCheckingEligibility(false);
    }
  };

  const loadMapLocations = async (trialId: number) => {
    setLoadingMap(true);
    try {
      const token = AuthService.getToken();
      if (!token) return;

      const response = await fetch(`${API_BASE_URL}/clinical-trials/${trialId}/locations`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        const locationsWithCoords = data.locations.filter(
          (loc: MapLocation) => loc.latitude && loc.longitude
        );
        if (locationsWithCoords.length > 0) {
          setMapLocations(locationsWithCoords);
          setShowMapModal(true);
        } else {
          Alert.alert('No Map Data', 'Location coordinates are not available for this trial.');
        }
      } else {
        Alert.alert('Error', 'Failed to load location data');
      }
    } catch (error) {
      console.error('Error loading map locations:', error);
      Alert.alert('Error', 'Failed to load location data');
    } finally {
      setLoadingMap(false);
    }
  };

  const openTrialDetail = (trial: Trial, match?: TrialMatch) => {
    setSelectedTrial(trial);
    setSelectedMatch(match || null);
    setShowDetailModal(true);
  };

  const openInterestForm = (trialId: number) => {
    setInterestTrialId(trialId);
    setInterestLevel('exploring');
    setPreferredContact('email');
    setInterestNotes('');
    setShowInterestModal(true);
    setShowDetailModal(false);
  };

  const submitInterest = async () => {
    if (!interestTrialId) return;

    setSubmittingInterest(true);
    try {
      const token = AuthService.getToken();
      const response = await fetch(`${API_BASE_URL}/clinical-trials/${interestTrialId}/interest`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          interest_level: interestLevel,
          preferred_contact: preferredContact,
          notes: interestNotes || null
        })
      });

      if (response.ok) {
        Alert.alert('Success', 'Your interest has been recorded. The study coordinator may contact you.');
        setShowInterestModal(false);
        loadInterests();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to submit interest');
      }
    } catch (error) {
      Alert.alert('Error', 'Network error. Please try again.');
    } finally {
      setSubmittingInterest(false);
    }
  };

  const withdrawInterest = async (interestId: number) => {
    Alert.alert(
      'Withdraw Interest',
      'Are you sure you want to withdraw your interest in this trial?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Withdraw',
          style: 'destructive',
          onPress: async () => {
            try {
              const token = AuthService.getToken();
              const response = await fetch(`${API_BASE_URL}/clinical-trials/interests/${interestId}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` }
              });

              if (response.ok) {
                Alert.alert('Success', 'Interest withdrawn');
                loadInterests();
              } else {
                Alert.alert('Error', 'Failed to withdraw interest');
              }
            } catch (error) {
              Alert.alert('Error', 'Network error');
            }
          }
        }
      ]
    );
  };

  const getScoreColor = (score: number) => {
    if (score >= 70) return '#10b981';
    if (score >= 50) return '#f59e0b';
    if (score >= 30) return '#f97316';
    return '#6b7280';
  };

  const getStatusColor = (status: string) => {
    if (status.toLowerCase().includes('recruiting')) return '#10b981';
    if (status.toLowerCase().includes('active')) return '#3b82f6';
    return '#6b7280';
  };

  const getPhaseLabel = (phase: string | null) => {
    if (!phase) return 'N/A';
    return phase;
  };

  const renderMatchReasons = (reasons: string[]) => {
    const reasonLabels: Record<string, { label: string; icon: string; color: string }> = {
      'diagnosis_exact_match': { label: 'Diagnosis Match', icon: 'checkmark-circle', color: '#10b981' },
      'diagnosis_related_match': { label: 'Related Condition', icon: 'git-branch', color: '#3b82f6' },
      'location_nearby': { label: 'Near You', icon: 'location', color: '#8b5cf6' },
      'age_eligible': { label: 'Age Eligible', icon: 'person', color: '#06b6d4' },
      'gender_eligible': { label: 'Gender Eligible', icon: 'people', color: '#06b6d4' },
      'preferred_phase': { label: 'Phase 2/3', icon: 'flask', color: '#f59e0b' },
      // Genetic matching reasons
      'genetic_exact_match': { label: 'Genetic Match', icon: 'fitness', color: '#ec4899' },
      'genetic_partial_match': { label: 'Partial Genetic', icon: 'fitness-outline', color: '#f97316' },
    };

    return (
      <View style={styles.reasonsContainer}>
        {reasons.map((reason, index) => {
          const info = reasonLabels[reason] || { label: reason, icon: 'checkmark', color: '#6b7280' };
          return (
            <View key={index} style={[styles.reasonBadge, { backgroundColor: info.color + '20' }]}>
              <Ionicons name={info.icon as any} size={12} color={info.color} />
              <Text style={[styles.reasonText, { color: info.color }]}>{info.label}</Text>
            </View>
          );
        })}
      </View>
    );
  };

  const renderTrialCard = (trial: Trial, match?: TrialMatch, interest?: TrialInterest) => {
    const hasInterest = interests.some(i => i.trial_id === trial.id);

    return (
      <TouchableOpacity
        key={trial.id}
        style={styles.trialCard}
        onPress={() => openTrialDetail(trial, match)}
        activeOpacity={0.7}
      >
        <View style={styles.trialHeader}>
          <View style={styles.trialBadges}>
            <View style={[styles.phaseBadge, { backgroundColor: '#8b5cf6' }]}>
              <Text style={styles.phaseBadgeText}>{getPhaseLabel(trial.phase)}</Text>
            </View>
            <View style={[styles.statusBadge, { backgroundColor: getStatusColor(trial.status) }]}>
              <Text style={styles.statusBadgeText}>{trial.status}</Text>
            </View>
          </View>
          {match && (
            <View style={[styles.scoreBadge, { backgroundColor: getScoreColor(match.match_score) }]}>
              <Text style={styles.scoreBadgeText}>{match.match_score.toFixed(0)}%</Text>
            </View>
          )}
        </View>

        <Text style={styles.trialTitle} numberOfLines={2}>{trial.title}</Text>

        {trial.conditions && trial.conditions.length > 0 && (
          <View style={styles.conditionsRow}>
            <Ionicons name="medical" size={14} color="#6b7280" />
            <Text style={styles.conditionsText} numberOfLines={1}>
              {trial.conditions.slice(0, 3).join(', ')}
              {trial.conditions.length > 3 && ` +${trial.conditions.length - 3} more`}
            </Text>
          </View>
        )}

        {match && match.match_reasons && (
          renderMatchReasons(match.match_reasons)
        )}

        {match && match.distance_miles && match.nearest_location && (
          <View style={styles.locationRow}>
            <Ionicons name="location" size={14} color="#8b5cf6" />
            <Text style={styles.locationText}>
              {match.distance_miles.toFixed(1)} mi - {match.nearest_location.city}, {match.nearest_location.state}
            </Text>
          </View>
        )}

        {/* Genetic Match Info */}
        {match && match.matched_biomarkers && match.matched_biomarkers.length > 0 && (
          <View style={styles.geneticMatchRow}>
            <Ionicons name="fitness" size={14} color="#ec4899" />
            <Text style={styles.geneticMatchText}>
              Biomarkers: {match.matched_biomarkers.slice(0, 2).join(', ')}
              {match.matched_biomarkers.length > 2 && ` +${match.matched_biomarkers.length - 2}`}
            </Text>
          </View>
        )}

        {/* Targeted Therapy Badge */}
        {trial.targeted_therapy_trial && (
          <View style={styles.targetedBadge}>
            <Ionicons name="medical" size={12} color="#ec4899" />
            <Text style={styles.targetedBadgeText}>Targeted Therapy</Text>
          </View>
        )}

        {interest && (
          <View style={styles.interestRow}>
            <Ionicons name="heart" size={14} color="#ec4899" />
            <Text style={styles.interestText}>
              Interest: {interest.interest_level} ({interest.status})
            </Text>
          </View>
        )}

        <View style={styles.trialFooter}>
          <Text style={styles.nctId}>{trial.nct_id}</Text>
          {!hasInterest && (
            <TouchableOpacity
              style={styles.expressInterestButton}
              onPress={(e) => {
                e.stopPropagation();
                openInterestForm(trial.id);
              }}
            >
              <Ionicons name="heart-outline" size={16} color="#8b5cf6" />
              <Text style={styles.expressInterestText}>Interested</Text>
            </TouchableOpacity>
          )}
        </View>
      </TouchableOpacity>
    );
  };

  const renderMatches = () => {
    if (matches.length === 0) {
      return (
        <View style={styles.emptyState}>
          <Ionicons name="flask-outline" size={64} color="rgba(255,255,255,0.5)" />
          <Text style={styles.emptyStateTitle}>No Matches Yet</Text>
          <Text style={styles.emptyStateText}>
            Complete some skin analyses to get personalized trial matches based on your diagnosis history.
          </Text>
        </View>
      );
    }

    return matches.map(match => renderTrialCard(match.trial, match));
  };

  const renderAllTrials = () => {
    if (allTrials.length === 0) {
      return (
        <View style={styles.emptyState}>
          <Ionicons name="search-outline" size={64} color="rgba(255,255,255,0.5)" />
          <Text style={styles.emptyStateTitle}>No Trials Found</Text>
          <Text style={styles.emptyStateText}>
            There are no clinical trials available at this time. Check back later.
          </Text>
        </View>
      );
    }

    return allTrials.map(trial => renderTrialCard(trial));
  };

  const renderInterests = () => {
    if (interests.length === 0) {
      return (
        <View style={styles.emptyState}>
          <Ionicons name="heart-outline" size={64} color="rgba(255,255,255,0.5)" />
          <Text style={styles.emptyStateTitle}>No Interests Yet</Text>
          <Text style={styles.emptyStateText}>
            Browse trials and express interest to track them here.
          </Text>
        </View>
      );
    }

    return interests
      .filter(interest => interest.trial)  // Filter out interests without valid trials
      .map(interest => (
        <View key={interest.id}>
          {renderTrialCard(interest.trial, undefined, interest)}
          <TouchableOpacity
            style={styles.withdrawButton}
            onPress={() => withdrawInterest(interest.id)}
          >
            <Ionicons name="close-circle" size={16} color="#dc2626" />
            <Text style={styles.withdrawButtonText}>Withdraw Interest</Text>
          </TouchableOpacity>
        </View>
      ));
  };

  const renderNewMatchAlert = () => {
    if (!newMatchAlert || alertsDismissed || newMatchAlert.new_matches_count === 0) {
      return null;
    }

    return (
      <View style={styles.alertBanner}>
        <View style={styles.alertContent}>
          <Ionicons name="notifications" size={24} color="#f59e0b" />
          <View style={styles.alertTextContainer}>
            <Text style={styles.alertTitle}>
              {newMatchAlert.new_matches_count} New Trial{newMatchAlert.new_matches_count > 1 ? 's' : ''} Match Your Profile!
            </Text>
            <Text style={styles.alertSubtext}>
              Tap to view matching trials based on your diagnosis history
            </Text>
          </View>
          <TouchableOpacity
            style={styles.alertDismiss}
            onPress={() => setAlertsDismissed(true)}
          >
            <Ionicons name="close" size={20} color="#6b7280" />
          </TouchableOpacity>
        </View>
        <TouchableOpacity
          style={styles.alertButton}
          onPress={() => {
            setActiveTab('matches');
            setAlertsDismissed(true);
          }}
        >
          <Text style={styles.alertButtonText}>View Matches</Text>
          <Ionicons name="arrow-forward" size={16} color="white" />
        </TouchableOpacity>
      </View>
    );
  };

  const renderDetailModal = () => {
    if (!selectedTrial) return null;

    return (
      <Modal
        visible={showDetailModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowDetailModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <ScrollView style={styles.modalScroll}>
              <View style={styles.modalHeader}>
                <TouchableOpacity onPress={() => setShowDetailModal(false)}>
                  <Ionicons name="close" size={24} color="#1f2937" />
                </TouchableOpacity>
                <Text style={styles.modalTitle}>Trial Details</Text>
                <TouchableOpacity onPress={() => shareTrialWithDoctor(selectedTrial)}>
                  <Ionicons name="share-outline" size={24} color="#8b5cf6" />
                </TouchableOpacity>
              </View>

              {selectedMatch && (
                <View style={[styles.matchScoreSection, { backgroundColor: getScoreColor(selectedMatch.match_score) + '20' }]}>
                  <Text style={styles.matchScoreLabel}>Match Score</Text>
                  <Text style={[styles.matchScoreValue, { color: getScoreColor(selectedMatch.match_score) }]}>
                    {selectedMatch.match_score.toFixed(0)}%
                  </Text>
                  {selectedMatch.matched_conditions && selectedMatch.matched_conditions.length > 0 && (
                    <Text style={styles.matchedConditions}>
                      Matched: {selectedMatch.matched_conditions.join(', ')}
                    </Text>
                  )}
                </View>
              )}

              {/* Genetic Match Details */}
              {selectedMatch && selectedMatch.genetic_score !== undefined && selectedMatch.genetic_score > 0 && (
                <View style={styles.geneticMatchSection}>
                  <View style={styles.geneticMatchHeader}>
                    <Ionicons name="fitness" size={20} color="#ec4899" />
                    <Text style={styles.geneticMatchTitle}>Genetic Match</Text>
                    <View style={[styles.geneticScoreBadge, {
                      backgroundColor: selectedMatch.genetic_match_type === 'exact' ? '#dcfce7' :
                                       selectedMatch.genetic_match_type === 'partial' ? '#fef3c7' : '#f3f4f6'
                    }]}>
                      <Text style={[styles.geneticScoreText, {
                        color: selectedMatch.genetic_match_type === 'exact' ? '#10b981' :
                               selectedMatch.genetic_match_type === 'partial' ? '#f59e0b' : '#6b7280'
                      }]}>
                        {selectedMatch.genetic_match_type === 'exact' ? 'Full Match' :
                         selectedMatch.genetic_match_type === 'partial' ? 'Partial Match' : 'No Match'}
                      </Text>
                    </View>
                  </View>

                  {selectedMatch.matched_biomarkers && selectedMatch.matched_biomarkers.length > 0 && (
                    <View style={styles.biomarkerList}>
                      <Text style={styles.biomarkerLabel}>Your Matching Biomarkers:</Text>
                      <View style={styles.biomarkerTags}>
                        {selectedMatch.matched_biomarkers.map((biomarker, idx) => (
                          <View key={idx} style={styles.biomarkerTag}>
                            <Ionicons name="checkmark" size={12} color="#10b981" />
                            <Text style={styles.biomarkerTagText}>{biomarker}</Text>
                          </View>
                        ))}
                      </View>
                    </View>
                  )}

                  {selectedMatch.missing_biomarkers && selectedMatch.missing_biomarkers.length > 0 && (
                    <View style={styles.biomarkerList}>
                      <Text style={styles.biomarkerLabelMissing}>Required (not found):</Text>
                      <View style={styles.biomarkerTags}>
                        {selectedMatch.missing_biomarkers.map((biomarker, idx) => (
                          <View key={idx} style={[styles.biomarkerTag, styles.biomarkerTagMissing]}>
                            <Ionicons name="help" size={12} color="#f59e0b" />
                            <Text style={styles.biomarkerTagTextMissing}>{biomarker}</Text>
                          </View>
                        ))}
                      </View>
                    </View>
                  )}
                </View>
              )}

              {/* Trial Biomarker Requirements */}
              {selectedTrial.required_biomarkers && selectedTrial.required_biomarkers.length > 0 && (
                <View style={styles.detailSection}>
                  <Text style={styles.detailSectionTitle}>
                    <Ionicons name="flask" size={16} color="#ec4899" /> Biomarker Requirements
                  </Text>
                  <View style={styles.biomarkerRequirements}>
                    {selectedTrial.required_biomarkers.map((biomarker, idx) => (
                      <View key={idx} style={styles.biomarkerRequirement}>
                        <Ionicons name="add-circle" size={14} color="#10b981" />
                        <Text style={styles.biomarkerRequirementText}>{biomarker} required</Text>
                      </View>
                    ))}
                    {selectedTrial.excluded_biomarkers && selectedTrial.excluded_biomarkers.map((biomarker, idx) => (
                      <View key={`ex-${idx}`} style={styles.biomarkerRequirement}>
                        <Ionicons name="remove-circle" size={14} color="#ef4444" />
                        <Text style={styles.biomarkerRequirementText}>{biomarker} excluded</Text>
                      </View>
                    ))}
                  </View>
                  {selectedTrial.targeted_therapy_trial && (
                    <View style={styles.targetedTherapyBanner}>
                      <Ionicons name="medical" size={16} color="#ec4899" />
                      <Text style={styles.targetedTherapyText}>This is a targeted therapy trial</Text>
                    </View>
                  )}
                </View>
              )}

              <Text style={styles.detailTrialTitle}>{selectedTrial.title}</Text>

              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>NCT ID</Text>
                <Text style={styles.detailValue}>{selectedTrial.nct_id}</Text>
              </View>

              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Phase</Text>
                <Text style={styles.detailValue}>{selectedTrial.phase || 'Not specified'}</Text>
              </View>

              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Status</Text>
                <Text style={[styles.detailValue, { color: getStatusColor(selectedTrial.status) }]}>
                  {selectedTrial.status}
                </Text>
              </View>

              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Sponsor</Text>
                <Text style={styles.detailValue}>{selectedTrial.sponsor || 'Not specified'}</Text>
              </View>

              {selectedTrial.brief_summary && (
                <View style={styles.detailSection}>
                  <Text style={styles.detailSectionTitle}>Summary</Text>
                  <Text style={styles.detailSectionText}>{selectedTrial.brief_summary}</Text>
                </View>
              )}

              {selectedTrial.conditions && selectedTrial.conditions.length > 0 && (
                <View style={styles.detailSection}>
                  <Text style={styles.detailSectionTitle}>Conditions</Text>
                  <View style={styles.conditionTags}>
                    {selectedTrial.conditions.map((condition, idx) => (
                      <View key={idx} style={styles.conditionTag}>
                        <Text style={styles.conditionTagText}>{condition}</Text>
                      </View>
                    ))}
                  </View>
                </View>
              )}

              {selectedTrial.eligibility_criteria && (
                <View style={styles.detailSection}>
                  <Text style={styles.detailSectionTitle}>Eligibility Criteria</Text>
                  <Text style={styles.detailSectionText} numberOfLines={10}>
                    {selectedTrial.eligibility_criteria}
                  </Text>
                </View>
              )}

              <View style={styles.detailSection}>
                <Text style={styles.detailSectionTitle}>Age Requirements</Text>
                <Text style={styles.detailSectionText}>
                  {selectedTrial.min_age || 'No minimum'} - {selectedTrial.max_age || 'No maximum'} years
                </Text>
              </View>

              {selectedTrial.locations && selectedTrial.locations.length > 0 && (
                <View style={styles.detailSection}>
                  <View style={styles.locationHeader}>
                    <Text style={styles.detailSectionTitle}>Locations ({selectedTrial.locations.length})</Text>
                    <TouchableOpacity
                      style={styles.mapButton}
                      onPress={() => loadMapLocations(selectedTrial.id)}
                      disabled={loadingMap}
                    >
                      {loadingMap ? (
                        <ActivityIndicator size="small" color="#8b5cf6" />
                      ) : (
                        <>
                          <Ionicons name="map" size={16} color="#8b5cf6" />
                          <Text style={styles.mapButtonText}>View Map</Text>
                        </>
                      )}
                    </TouchableOpacity>
                  </View>
                  {selectedTrial.locations.slice(0, 5).map((loc, idx) => (
                    <View key={idx} style={styles.locationItem}>
                      <Ionicons name="location" size={16} color="#8b5cf6" />
                      <Text style={styles.locationItemText}>
                        {[loc.facility, loc.city, loc.state, loc.country].filter(Boolean).join(', ')}
                      </Text>
                    </View>
                  ))}
                  {selectedTrial.locations.length > 5 && (
                    <Text style={styles.moreLocations}>
                      +{selectedTrial.locations.length - 5} more locations
                    </Text>
                  )}
                </View>
              )}

              {(selectedTrial.contact_name || selectedTrial.contact_email || selectedTrial.contact_phone) && (
                <View style={styles.detailSection}>
                  <Text style={styles.detailSectionTitle}>Contact</Text>
                  {selectedTrial.contact_name && (
                    <Text style={styles.detailSectionText}>{selectedTrial.contact_name}</Text>
                  )}
                  {selectedTrial.contact_email && (
                    <Text style={styles.contactEmail}>{selectedTrial.contact_email}</Text>
                  )}
                  {selectedTrial.contact_phone && (
                    <Text style={styles.detailSectionText}>{selectedTrial.contact_phone}</Text>
                  )}
                </View>
              )}

              {/* Action Buttons */}
              <View style={styles.actionButtonsContainer}>
                <TouchableOpacity
                  style={styles.eligibilityButton}
                  onPress={() => loadEligibilityQuestions(selectedTrial.id)}
                  disabled={loadingQuestions}
                >
                  {loadingQuestions ? (
                    <ActivityIndicator size="small" color="#10b981" />
                  ) : (
                    <>
                      <Ionicons name="checkmark-circle-outline" size={20} color="#10b981" />
                      <Text style={styles.eligibilityButtonText}>Check Eligibility</Text>
                    </>
                  )}
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.shareButton}
                  onPress={() => shareTrialWithDoctor(selectedTrial)}
                >
                  <Ionicons name="share-social-outline" size={20} color="#3b82f6" />
                  <Text style={styles.shareButtonText}>Share with Doctor</Text>
                </TouchableOpacity>
              </View>

              <TouchableOpacity
                style={styles.expressInterestFullButton}
                onPress={() => openInterestForm(selectedTrial.id)}
              >
                <Ionicons name="heart" size={20} color="white" />
                <Text style={styles.expressInterestFullText}>Express Interest</Text>
              </TouchableOpacity>

              {selectedTrial.url && (
                <TouchableOpacity
                  style={styles.viewOnCtGovButton}
                  onPress={() => Linking.openURL(selectedTrial.url!)}
                >
                  <Ionicons name="open-outline" size={20} color="#8b5cf6" />
                  <Text style={styles.viewOnCtGovText}>View on ClinicalTrials.gov</Text>
                </TouchableOpacity>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>
    );
  };

  const renderInterestModal = () => {
    return (
      <Modal
        visible={showInterestModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowInterestModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.interestModalContent}>
            <View style={styles.modalHeader}>
              <TouchableOpacity onPress={() => setShowInterestModal(false)}>
                <Ionicons name="close" size={24} color="#1f2937" />
              </TouchableOpacity>
              <Text style={styles.modalTitle}>Express Interest</Text>
              <View style={{ width: 24 }} />
            </View>

            <Text style={styles.interestLabel}>Interest Level</Text>
            <View style={styles.interestOptions}>
              {['exploring', 'medium', 'high'].map(level => (
                <TouchableOpacity
                  key={level}
                  style={[
                    styles.interestOption,
                    interestLevel === level && styles.interestOptionSelected
                  ]}
                  onPress={() => setInterestLevel(level)}
                >
                  <Text style={[
                    styles.interestOptionText,
                    interestLevel === level && styles.interestOptionTextSelected
                  ]}>
                    {level.charAt(0).toUpperCase() + level.slice(1)}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.interestLabel}>Preferred Contact Method</Text>
            <View style={styles.interestOptions}>
              {['email', 'phone'].map(method => (
                <TouchableOpacity
                  key={method}
                  style={[
                    styles.interestOption,
                    preferredContact === method && styles.interestOptionSelected
                  ]}
                  onPress={() => setPreferredContact(method)}
                >
                  <Ionicons
                    name={method === 'email' ? 'mail' : 'call'}
                    size={16}
                    color={preferredContact === method ? 'white' : '#8b5cf6'}
                  />
                  <Text style={[
                    styles.interestOptionText,
                    preferredContact === method && styles.interestOptionTextSelected
                  ]}>
                    {method.charAt(0).toUpperCase() + method.slice(1)}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.interestLabel}>Notes (Optional)</Text>
            <TextInput
              style={styles.notesInput}
              placeholder="Any questions or comments for the study team..."
              placeholderTextColor="#9ca3af"
              value={interestNotes}
              onChangeText={setInterestNotes}
              multiline
              numberOfLines={4}
            />

            <TouchableOpacity
              style={[styles.submitInterestButton, submittingInterest && styles.buttonDisabled]}
              onPress={submitInterest}
              disabled={submittingInterest}
            >
              {submittingInterest ? (
                <ActivityIndicator color="white" />
              ) : (
                <>
                  <Ionicons name="heart" size={20} color="white" />
                  <Text style={styles.submitInterestText}>Submit Interest</Text>
                </>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    );
  };

  const renderEligibilityModal = () => {
    return (
      <Modal
        visible={showEligibilityModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowEligibilityModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.eligibilityModalContent}>
            <ScrollView>
              <View style={styles.modalHeader}>
                <TouchableOpacity onPress={() => setShowEligibilityModal(false)}>
                  <Ionicons name="close" size={24} color="#1f2937" />
                </TouchableOpacity>
                <Text style={styles.modalTitle}>Eligibility Checker</Text>
                <View style={{ width: 24 }} />
              </View>

              {eligibilityResult ? (
                // Show results
                <View style={styles.eligibilityResults}>
                  <View style={[
                    styles.eligibilityResultHeader,
                    { backgroundColor: eligibilityResult.eligible ? '#dcfce7' : '#fee2e2' }
                  ]}>
                    <Ionicons
                      name={eligibilityResult.eligible ? 'checkmark-circle' : 'close-circle'}
                      size={48}
                      color={eligibilityResult.eligible ? '#10b981' : '#ef4444'}
                    />
                    <Text style={[
                      styles.eligibilityResultTitle,
                      { color: eligibilityResult.eligible ? '#10b981' : '#ef4444' }
                    ]}>
                      {eligibilityResult.eligible ? 'Likely Eligible' : 'May Not Be Eligible'}
                    </Text>
                    <Text style={styles.eligibilityScore}>
                      Eligibility Score: {eligibilityResult.score}%
                    </Text>
                  </View>

                  <Text style={styles.eligibilityRecommendation}>
                    {eligibilityResult.recommendation}
                  </Text>

                  {eligibilityResult.met_criteria.length > 0 && (
                    <View style={styles.criteriaSection}>
                      <Text style={styles.criteriaSectionTitle}>
                        <Ionicons name="checkmark" size={16} color="#10b981" /> Met Criteria
                      </Text>
                      {eligibilityResult.met_criteria.map((criteria, idx) => (
                        <Text key={idx} style={styles.criteriaItem}>• {criteria}</Text>
                      ))}
                    </View>
                  )}

                  {eligibilityResult.unmet_criteria.length > 0 && (
                    <View style={styles.criteriaSection}>
                      <Text style={[styles.criteriaSectionTitle, { color: '#ef4444' }]}>
                        <Ionicons name="close" size={16} color="#ef4444" /> Unmet Criteria
                      </Text>
                      {eligibilityResult.unmet_criteria.map((criteria, idx) => (
                        <Text key={idx} style={styles.criteriaItem}>• {criteria}</Text>
                      ))}
                    </View>
                  )}

                  {eligibilityResult.uncertain_criteria.length > 0 && (
                    <View style={styles.criteriaSection}>
                      <Text style={[styles.criteriaSectionTitle, { color: '#f59e0b' }]}>
                        <Ionicons name="help" size={16} color="#f59e0b" /> Needs Verification
                      </Text>
                      {eligibilityResult.uncertain_criteria.map((criteria, idx) => (
                        <Text key={idx} style={styles.criteriaItem}>• {criteria}</Text>
                      ))}
                    </View>
                  )}

                  <TouchableOpacity
                    style={styles.retakeButton}
                    onPress={() => setEligibilityResult(null)}
                  >
                    <Text style={styles.retakeButtonText}>Retake Questionnaire</Text>
                  </TouchableOpacity>
                </View>
              ) : (
                // Show questions
                <View style={styles.eligibilityQuestions}>
                  <Text style={styles.eligibilityIntro}>
                    Answer these questions to check if you may be eligible for this trial.
                    This is not a final determination - the study team will verify all criteria.
                  </Text>

                  {eligibilityQuestions.map((question, idx) => (
                    <View key={question.id} style={styles.questionContainer}>
                      <Text style={styles.questionText}>
                        {idx + 1}. {question.question}
                        {question.required && <Text style={styles.requiredMark}> *</Text>}
                      </Text>

                      {question.type === 'boolean' && (
                        <View style={styles.booleanOptions}>
                          <TouchableOpacity
                            style={[
                              styles.booleanOption,
                              eligibilityAnswers[question.id] === true && styles.booleanOptionSelected
                            ]}
                            onPress={() => setEligibilityAnswers({
                              ...eligibilityAnswers,
                              [question.id]: true
                            })}
                          >
                            <Text style={[
                              styles.booleanOptionText,
                              eligibilityAnswers[question.id] === true && styles.booleanOptionTextSelected
                            ]}>Yes</Text>
                          </TouchableOpacity>
                          <TouchableOpacity
                            style={[
                              styles.booleanOption,
                              eligibilityAnswers[question.id] === false && styles.booleanOptionSelected
                            ]}
                            onPress={() => setEligibilityAnswers({
                              ...eligibilityAnswers,
                              [question.id]: false
                            })}
                          >
                            <Text style={[
                              styles.booleanOptionText,
                              eligibilityAnswers[question.id] === false && styles.booleanOptionTextSelected
                            ]}>No</Text>
                          </TouchableOpacity>
                        </View>
                      )}

                      {question.type === 'number' && (
                        <TextInput
                          style={styles.numberInput}
                          placeholder="Enter a number"
                          placeholderTextColor="#9ca3af"
                          keyboardType="numeric"
                          value={eligibilityAnswers[question.id]?.toString() || ''}
                          onChangeText={(text) => setEligibilityAnswers({
                            ...eligibilityAnswers,
                            [question.id]: text ? parseInt(text) : undefined
                          })}
                        />
                      )}

                      {question.type === 'select' && question.options && (
                        <View style={styles.selectOptions}>
                          {question.options.map((option) => (
                            <TouchableOpacity
                              key={option}
                              style={[
                                styles.selectOption,
                                eligibilityAnswers[question.id] === option && styles.selectOptionSelected
                              ]}
                              onPress={() => setEligibilityAnswers({
                                ...eligibilityAnswers,
                                [question.id]: option
                              })}
                            >
                              <Text style={[
                                styles.selectOptionText,
                                eligibilityAnswers[question.id] === option && styles.selectOptionTextSelected
                              ]}>{option}</Text>
                            </TouchableOpacity>
                          ))}
                        </View>
                      )}

                      {question.type === 'text' && (
                        <TextInput
                          style={styles.textInput}
                          placeholder="Enter your answer"
                          placeholderTextColor="#9ca3af"
                          value={eligibilityAnswers[question.id] || ''}
                          onChangeText={(text) => setEligibilityAnswers({
                            ...eligibilityAnswers,
                            [question.id]: text
                          })}
                        />
                      )}
                    </View>
                  ))}

                  <TouchableOpacity
                    style={[styles.checkEligibilityButton, checkingEligibility && styles.buttonDisabled]}
                    onPress={checkEligibility}
                    disabled={checkingEligibility}
                  >
                    {checkingEligibility ? (
                      <ActivityIndicator color="white" />
                    ) : (
                      <>
                        <Ionicons name="checkmark-circle" size={20} color="white" />
                        <Text style={styles.checkEligibilityButtonText}>Check My Eligibility</Text>
                      </>
                    )}
                  </TouchableOpacity>
                </View>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>
    );
  };

  const openInMaps = (location: MapLocation) => {
    const address = `${location.facility}, ${location.city}, ${location.state}, ${location.country}`;
    const encodedAddress = encodeURIComponent(address);

    if (location.latitude && location.longitude) {
      // Use coordinates if available
      const url = Platform.select({
        ios: `maps:0,0?q=${location.latitude},${location.longitude}(${encodeURIComponent(location.facility)})`,
        android: `geo:${location.latitude},${location.longitude}?q=${location.latitude},${location.longitude}(${encodeURIComponent(location.facility)})`,
        default: `https://www.google.com/maps/search/?api=1&query=${location.latitude},${location.longitude}`
      });
      Linking.openURL(url as string);
    } else {
      // Use address search
      const url = Platform.select({
        ios: `maps:0,0?q=${encodedAddress}`,
        android: `geo:0,0?q=${encodedAddress}`,
        default: `https://www.google.com/maps/search/?api=1&query=${encodedAddress}`
      });
      Linking.openURL(url as string);
    }
  };

  const renderMapModal = () => {
    return (
      <Modal
        visible={showMapModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowMapModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.mapModalContent}>
            <View style={styles.modalHeader}>
              <TouchableOpacity onPress={() => setShowMapModal(false)}>
                <Ionicons name="close" size={24} color="#1f2937" />
              </TouchableOpacity>
              <Text style={styles.modalTitle}>Trial Locations</Text>
              <View style={{ width: 24 }} />
            </View>

            <View style={styles.mapInfoBanner}>
              <Ionicons name="information-circle" size={20} color="#3b82f6" />
              <Text style={styles.mapInfoText}>
                Tap any location to open in Maps
              </Text>
            </View>

            <ScrollView style={styles.mapLocationsList}>
              {mapLocations.map((location, idx) => (
                <TouchableOpacity
                  key={idx}
                  style={styles.mapLocationItem}
                  onPress={() => openInMaps(location)}
                  activeOpacity={0.7}
                >
                  <View style={styles.mapLocationIcon}>
                    <Ionicons name="location" size={24} color="#8b5cf6" />
                  </View>
                  <View style={styles.mapLocationInfo}>
                    <Text style={styles.mapLocationFacility}>{location.facility}</Text>
                    <Text style={styles.mapLocationAddress}>
                      {location.city}, {location.state}, {location.country}
                    </Text>
                    {location.distance_miles && (
                      <View style={styles.distanceBadge}>
                        <Ionicons name="navigate" size={12} color="#8b5cf6" />
                        <Text style={styles.mapLocationDistance}>
                          {location.distance_miles.toFixed(1)} miles away
                        </Text>
                      </View>
                    )}
                  </View>
                  <Ionicons name="open-outline" size={20} color="#9ca3af" />
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </View>
      </Modal>
    );
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#8b5cf6" />
        <Text style={styles.loadingText}>Loading clinical trials...</Text>
      </View>
    );
  }

  return (
    <LinearGradient
      colors={['#7c3aed', '#8b5cf6', '#a78bfa']}
      style={styles.container}
    >
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="white" />
        </TouchableOpacity>
        <Text style={styles.title}>Clinical Trials</Text>
        <TouchableOpacity onPress={onRefresh} style={styles.refreshButton}>
          <Ionicons name="refresh" size={24} color="white" />
        </TouchableOpacity>
      </View>

      {/* Tabs */}
      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'matches' && styles.activeTab]}
          onPress={() => setActiveTab('matches')}
        >
          <Ionicons
            name="sparkles"
            size={18}
            color={activeTab === 'matches' ? '#8b5cf6' : 'rgba(255,255,255,0.7)'}
          />
          <Text style={[styles.tabText, activeTab === 'matches' && styles.activeTabText]}>
            Matches ({matches.length})
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'all' && styles.activeTab]}
          onPress={() => setActiveTab('all')}
        >
          <Ionicons
            name="list"
            size={18}
            color={activeTab === 'all' ? '#8b5cf6' : 'rgba(255,255,255,0.7)'}
          />
          <Text style={[styles.tabText, activeTab === 'all' && styles.activeTabText]}>
            All Trials ({totalTrials})
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'interests' && styles.activeTab]}
          onPress={() => setActiveTab('interests')}
        >
          <Ionicons
            name="heart"
            size={18}
            color={activeTab === 'interests' ? '#8b5cf6' : 'rgba(255,255,255,0.7)'}
          />
          <Text style={[styles.tabText, activeTab === 'interests' && styles.activeTabText]}>
            My Interests ({interests.length})
          </Text>
        </TouchableOpacity>
      </View>

      {/* New Match Alert Banner */}
      {renderNewMatchAlert()}

      {/* Content */}
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="white" />
        }
      >
        {activeTab === 'matches' && renderMatches()}
        {activeTab === 'all' && renderAllTrials()}
        {activeTab === 'interests' && renderInterests()}
      </ScrollView>

      {renderDetailModal()}
      {renderInterestModal()}
      {renderEligibilityModal()}
      {renderMapModal()}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#7c3aed',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: 'white',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 16,
    paddingBottom: 16,
  },
  backButton: {
    padding: 8,
  },
  refreshButton: {
    padding: 8,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    flex: 1,
    textAlign: 'center',
  },
  tabContainer: {
    flexDirection: 'row',
    marginHorizontal: 16,
    marginBottom: 16,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 12,
    padding: 4,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderRadius: 10,
    gap: 4,
  },
  activeTab: {
    backgroundColor: 'white',
  },
  tabText: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.9)',
    fontWeight: '600',
  },
  activeTabText: {
    color: '#8b5cf6',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 16,
    paddingBottom: 32,
  },
  trialCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  trialHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  trialBadges: {
    flexDirection: 'row',
    gap: 8,
  },
  phaseBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 6,
  },
  phaseBadgeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 6,
  },
  statusBadgeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  scoreBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  scoreBadgeText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  trialTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
    lineHeight: 22,
  },
  conditionsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
  },
  conditionsText: {
    fontSize: 13,
    color: '#6b7280',
    flex: 1,
  },
  reasonsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginBottom: 8,
  },
  reasonBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    gap: 4,
  },
  reasonText: {
    fontSize: 11,
    fontWeight: '600',
  },
  locationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
  },
  locationText: {
    fontSize: 13,
    color: '#8b5cf6',
  },
  interestRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
  },
  interestText: {
    fontSize: 13,
    color: '#ec4899',
    fontWeight: '500',
  },
  trialFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  nctId: {
    fontSize: 12,
    color: '#9ca3af',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  expressInterestButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    backgroundColor: '#f3e8ff',
  },
  expressInterestText: {
    fontSize: 12,
    color: '#8b5cf6',
    fontWeight: '600',
  },
  withdrawButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    marginTop: -8,
    marginBottom: 12,
    paddingVertical: 8,
    backgroundColor: '#fee2e2',
    borderBottomLeftRadius: 16,
    borderBottomRightRadius: 16,
  },
  withdrawButtonText: {
    fontSize: 12,
    color: '#dc2626',
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 48,
  },
  emptyStateTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyStateText: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.8)',
    textAlign: 'center',
    paddingHorizontal: 32,
    lineHeight: 20,
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: 'white',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    maxHeight: '90%',
  },
  modalScroll: {
    padding: 20,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  matchScoreSection: {
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 20,
  },
  matchScoreLabel: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 4,
  },
  matchScoreValue: {
    fontSize: 36,
    fontWeight: 'bold',
  },
  matchedConditions: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center',
  },
  detailTrialTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 20,
    lineHeight: 26,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  detailLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  detailValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  detailSection: {
    marginTop: 20,
  },
  detailSectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 10,
  },
  detailSectionText: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 22,
  },
  conditionTags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  conditionTag: {
    backgroundColor: '#f3e8ff',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  conditionTagText: {
    fontSize: 12,
    color: '#8b5cf6',
    fontWeight: '500',
  },
  locationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  locationItemText: {
    fontSize: 13,
    color: '#4b5563',
    flex: 1,
  },
  moreLocations: {
    fontSize: 12,
    color: '#8b5cf6',
    marginTop: 8,
    fontWeight: '500',
  },
  contactEmail: {
    fontSize: 14,
    color: '#3b82f6',
    marginVertical: 4,
  },
  expressInterestFullButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#8b5cf6',
    paddingVertical: 16,
    borderRadius: 12,
    marginTop: 24,
  },
  expressInterestFullText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
  },
  viewOnCtGovButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 16,
    marginTop: 12,
    marginBottom: 20,
  },
  viewOnCtGovText: {
    fontSize: 14,
    color: '#8b5cf6',
    fontWeight: '600',
  },
  // Interest modal styles
  interestModalContent: {
    backgroundColor: 'white',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    padding: 20,
  },
  interestLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginTop: 16,
    marginBottom: 8,
  },
  interestOptions: {
    flexDirection: 'row',
    gap: 8,
  },
  interestOption: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 12,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#e5e7eb',
  },
  interestOptionSelected: {
    backgroundColor: '#8b5cf6',
    borderColor: '#8b5cf6',
  },
  interestOptionText: {
    fontSize: 14,
    color: '#8b5cf6',
    fontWeight: '600',
  },
  interestOptionTextSelected: {
    color: 'white',
  },
  notesInput: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 10,
    padding: 12,
    fontSize: 14,
    color: '#1f2937',
    minHeight: 100,
    textAlignVertical: 'top',
  },
  submitInterestButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#8b5cf6',
    paddingVertical: 16,
    borderRadius: 12,
    marginTop: 24,
    marginBottom: Platform.OS === 'ios' ? 24 : 16,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  submitInterestText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
  },
  // Alert banner styles
  alertBanner: {
    backgroundColor: 'white',
    marginHorizontal: 16,
    marginBottom: 12,
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  alertContent: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
  },
  alertTextContainer: {
    flex: 1,
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4,
  },
  alertSubtext: {
    fontSize: 13,
    color: '#6b7280',
  },
  alertDismiss: {
    padding: 4,
  },
  alertButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    backgroundColor: '#8b5cf6',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    marginTop: 12,
  },
  alertButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  // Location header with map button
  locationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  mapButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#f3e8ff',
    borderRadius: 6,
  },
  mapButtonText: {
    fontSize: 12,
    color: '#8b5cf6',
    fontWeight: '600',
  },
  // Action buttons container
  actionButtonsContainer: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 20,
  },
  eligibilityButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 12,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#10b981',
    backgroundColor: '#dcfce7',
  },
  eligibilityButtonText: {
    fontSize: 13,
    color: '#10b981',
    fontWeight: '600',
  },
  shareButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 12,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#3b82f6',
    backgroundColor: '#dbeafe',
  },
  shareButtonText: {
    fontSize: 13,
    color: '#3b82f6',
    fontWeight: '600',
  },
  // Eligibility modal styles
  eligibilityModalContent: {
    backgroundColor: 'white',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    maxHeight: '90%',
    padding: 20,
  },
  eligibilityResults: {
    paddingBottom: 20,
  },
  eligibilityResultHeader: {
    alignItems: 'center',
    padding: 24,
    borderRadius: 16,
    marginBottom: 20,
  },
  eligibilityResultTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 12,
  },
  eligibilityScore: {
    fontSize: 16,
    color: '#6b7280',
    marginTop: 8,
  },
  eligibilityRecommendation: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 22,
    marginBottom: 20,
    textAlign: 'center',
  },
  criteriaSection: {
    marginBottom: 16,
  },
  criteriaSectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#10b981',
    marginBottom: 8,
  },
  criteriaItem: {
    fontSize: 13,
    color: '#4b5563',
    marginBottom: 4,
    paddingLeft: 8,
  },
  retakeButton: {
    alignItems: 'center',
    paddingVertical: 12,
    marginTop: 16,
  },
  retakeButtonText: {
    fontSize: 14,
    color: '#8b5cf6',
    fontWeight: '600',
  },
  eligibilityQuestions: {
    paddingBottom: 20,
  },
  eligibilityIntro: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
    marginBottom: 20,
    padding: 12,
    backgroundColor: '#f3f4f6',
    borderRadius: 10,
  },
  questionContainer: {
    marginBottom: 24,
  },
  questionText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#1f2937',
    marginBottom: 12,
    lineHeight: 20,
  },
  requiredMark: {
    color: '#ef4444',
  },
  booleanOptions: {
    flexDirection: 'row',
    gap: 12,
  },
  booleanOption: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 12,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#e5e7eb',
  },
  booleanOptionSelected: {
    backgroundColor: '#8b5cf6',
    borderColor: '#8b5cf6',
  },
  booleanOptionText: {
    fontSize: 14,
    color: '#4b5563',
    fontWeight: '600',
  },
  booleanOptionTextSelected: {
    color: 'white',
  },
  numberInput: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 10,
    padding: 12,
    fontSize: 14,
    color: '#1f2937',
  },
  selectOptions: {
    gap: 8,
  },
  selectOption: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#e5e7eb',
  },
  selectOptionSelected: {
    backgroundColor: '#8b5cf6',
    borderColor: '#8b5cf6',
  },
  selectOptionText: {
    fontSize: 14,
    color: '#4b5563',
  },
  selectOptionTextSelected: {
    color: 'white',
    fontWeight: '600',
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 10,
    padding: 12,
    fontSize: 14,
    color: '#1f2937',
  },
  checkEligibilityButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#10b981',
    paddingVertical: 16,
    borderRadius: 12,
    marginTop: 20,
  },
  checkEligibilityButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
  },
  // Map modal styles
  mapModalContent: {
    backgroundColor: 'white',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    maxHeight: '80%',
    paddingTop: 20,
  },
  mapInfoBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginHorizontal: 20,
    padding: 12,
    backgroundColor: '#dbeafe',
    borderRadius: 10,
    marginBottom: 12,
  },
  mapInfoText: {
    fontSize: 13,
    color: '#3b82f6',
    flex: 1,
  },
  mapLocationsList: {
    paddingHorizontal: 20,
    paddingBottom: 24,
  },
  mapLocationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  mapLocationIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#f3e8ff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  mapLocationInfo: {
    flex: 1,
  },
  mapLocationFacility: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  mapLocationAddress: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 4,
  },
  distanceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  mapLocationDistance: {
    fontSize: 12,
    color: '#8b5cf6',
    fontWeight: '500',
  },
  // Genetic match styles
  geneticMatchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
  },
  geneticMatchText: {
    fontSize: 13,
    color: '#ec4899',
    flex: 1,
  },
  targetedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#fdf2f8',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    alignSelf: 'flex-start',
    marginBottom: 8,
  },
  targetedBadgeText: {
    fontSize: 11,
    color: '#ec4899',
    fontWeight: '600',
  },
  geneticMatchSection: {
    backgroundColor: '#fdf2f8',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
  },
  geneticMatchHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  geneticMatchTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#831843',
    flex: 1,
  },
  geneticScoreBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 6,
  },
  geneticScoreText: {
    fontSize: 12,
    fontWeight: '600',
  },
  biomarkerList: {
    marginTop: 8,
  },
  biomarkerLabel: {
    fontSize: 12,
    color: '#10b981',
    fontWeight: '600',
    marginBottom: 6,
  },
  biomarkerLabelMissing: {
    fontSize: 12,
    color: '#f59e0b',
    fontWeight: '600',
    marginBottom: 6,
  },
  biomarkerTags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  biomarkerTag: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#dcfce7',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
  },
  biomarkerTagText: {
    fontSize: 12,
    color: '#10b981',
    fontWeight: '500',
  },
  biomarkerTagMissing: {
    backgroundColor: '#fef3c7',
  },
  biomarkerTagTextMissing: {
    fontSize: 12,
    color: '#f59e0b',
    fontWeight: '500',
  },
  biomarkerRequirements: {
    gap: 8,
  },
  biomarkerRequirement: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  biomarkerRequirementText: {
    fontSize: 14,
    color: '#4b5563',
  },
  targetedTherapyBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#fdf2f8',
    padding: 12,
    borderRadius: 8,
    marginTop: 12,
  },
  targetedTherapyText: {
    fontSize: 13,
    color: '#831843',
    fontWeight: '500',
  },
});
