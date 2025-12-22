import React, { useState, useEffect, useRef } from 'react';
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
  Modal,
  Dimensions,
  KeyboardAvoidingView,
  TouchableWithoutFeedback,
  Keyboard
} from 'react-native';
import { router } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface TriageCase {
  id: string;
  priority: string;
  priority_score: number;
  estimated_wait_time: string;
  risk_factors: string[];
  routing_recommendations: string[];
  status: string;
  created_at: string;
}

interface ConsensusCase {
  case_id: string;
  status: string;
  urgency: string;
  created_at: string;
  deadline: string;
  consensus_diagnosis: string | null;
  agreement_ratio: number | null;
  opinions_received: number;
  opinions_expected: number;
}

interface VideoSession {
  session_id: string;
  status: string;
  specialist_id: number;
  annotations: any[];
}

export default function AdvancedTeledermScreen() {
  const [activeTab, setActiveTab] = useState<'triage' | 'video' | 'consensus' | 'store-forward'>('triage');
  const [isLoading, setIsLoading] = useState(false);

  // Triage state
  const [triageResult, setTriageResult] = useState<TriageCase | null>(null);
  const [chiefComplaint, setChiefComplaint] = useState('');
  const [symptomDuration, setSymptomDuration] = useState('');
  const [painLevel, setPainLevel] = useState(0);
  const [isSpreading, setIsSpreading] = useState(false);
  const [hasFever, setHasFever] = useState(false);

  // Video state
  const [videoSession, setVideoSession] = useState<VideoSession | null>(null);
  const [showAnnotationTools, setShowAnnotationTools] = useState(false);
  const [selectedTool, setSelectedTool] = useState<string>('draw');
  const [annotations, setAnnotations] = useState<any[]>([]);
  const [isCreatingVideo, setIsCreatingVideo] = useState(false);

  // Consensus state
  const [consensusCases, setConsensusCases] = useState<ConsensusCase[]>([]);
  const [showCreateConsensus, setShowCreateConsensus] = useState(false);
  const [consensusSummary, setConsensusSummary] = useState('');
  const [consensusUrgency, setConsensusUrgency] = useState('standard');
  const [isCreatingConsensus, setIsCreatingConsensus] = useState(false);
  const [isLoadingCases, setIsLoadingCases] = useState(false);
  const [isSpecialist, setIsSpecialist] = useState(false);

  // Store-forward state
  const [storeForwardCases, setStoreForwardCases] = useState<any[]>([]);

  // Fetch consensus cases when tab changes
  useEffect(() => {
    if (activeTab === 'consensus') {
      fetchConsensusCases();
    }
  }, [activeTab]);

  const fetchConsensusCases = async () => {
    try {
      setIsLoadingCases(true);
      const token = AuthService.getToken();
      if (!token) return;

      const response = await fetch(`${API_BASE_URL}/teledermatology/advanced/consensus/user/cases`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setConsensusCases(data.cases || []);
      }
    } catch (error) {
      console.error('Error fetching consensus cases:', error);
    } finally {
      setIsLoadingCases(false);
    }
  };

  const submitTriage = async () => {
    if (!chiefComplaint.trim()) {
      Alert.alert('Error', 'Please describe your chief complaint');
      return;
    }

    try {
      setIsLoading(true);
      const token = AuthService.getToken();

      const response = await fetch(`${API_BASE_URL}/teledermatology/advanced/triage/assess`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chief_complaint: chiefComplaint,
          symptom_duration: symptomDuration,
          pain_level: painLevel,
          is_spreading: isSpreading,
          has_fever: hasFever,
          ai_risk_score: 0.5, // Would come from actual analysis
          ai_predicted_conditions: []
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setTriageResult(data);
        Alert.alert(
          'Triage Complete',
          `Priority: ${data.priority.toUpperCase()}\nEstimated wait: ${data.estimated_wait_time}`
        );
      } else {
        Alert.alert('Error', 'Failed to submit triage');
      }
    } catch (error) {
      console.error('Triage error:', error);
      Alert.alert('Error', 'Failed to connect to server');
    } finally {
      setIsLoading(false);
    }
  };

  const createVideoSession = async () => {
    if (isCreatingVideo) return; // Prevent double-tap
    setIsCreatingVideo(true);

    try {
      const token = AuthService.getToken();

      if (!token) {
        Alert.alert('Error', 'Please login to start a video session');
        setIsCreatingVideo(false);
        return;
      }

      const response = await fetch(`${API_BASE_URL}/teledermatology/advanced/video/create-session?specialist_id=1&session_type=consultation`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      const responseText = await response.text();
      let data: any = {};

      try {
        data = JSON.parse(responseText);
      } catch (parseError) {
        console.error('Failed to parse response:', responseText);
        Alert.alert('Error', 'Invalid response from server');
        setIsCreatingVideo(false);
        return;
      }

      if (response.ok && data) {
        const sessionData: VideoSession = {
          session_id: data.session_id || `VID-${Date.now()}`,
          status: data.status || 'waiting',
          specialist_id: data.specialist_id || 1,
          annotations: data.annotations || []
        };
        setVideoSession(sessionData);
        Alert.alert(
          'Session Created',
          `Your video consultation session is ready.\n\nSession ID: ${sessionData.session_id}\nStatus: ${sessionData.status}\n\nNote: In a production environment, this would connect you to an available dermatologist via WebRTC video.`
        );
      } else {
        Alert.alert('Error', data?.detail || 'Failed to create video session');
      }
    } catch (error: any) {
      console.error('Video session error:', error);
      Alert.alert('Connection Error', error?.message || 'Failed to connect to the server.');
    } finally {
      setIsCreatingVideo(false);
    }
  };

  const createConsensusCase = async () => {
    if (!consensusSummary.trim()) {
      Alert.alert('Error', 'Please provide a case summary');
      return;
    }

    try {
      setIsCreatingConsensus(true);
      const token = AuthService.getToken();

      if (!token) {
        Alert.alert('Error', 'Please login to request a consensus review');
        router.push('/');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/teledermatology/advanced/consensus/create`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          case_id: `CON-${Date.now().toString(36).toUpperCase()}`,
          specialist_ids: [1, 2, 3], // Default specialists
          case_summary: consensusSummary,
          images: [], // Would include uploaded images
          urgency: consensusUrgency,
          deadline_hours: consensusUrgency === 'urgent' ? 24 : 72
        }),
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(
          'Consensus Request Submitted',
          `Your case has been submitted for multi-specialist review.\n\nCase ID: ${data.case_id}\nSpecialists: ${data.specialist_count}\nDeadline: ${new Date(data.deadline).toLocaleString()}\n\nYou will be notified when opinions are submitted.`,
          [{ text: 'OK', onPress: () => {
            setShowCreateConsensus(false);
            setConsensusSummary('');
            setConsensusUrgency('standard');
            fetchConsensusCases(); // Refresh the list
          }}]
        );
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        Alert.alert('Error', errorData.detail || 'Failed to create consensus case');
      }
    } catch (error) {
      console.error('Consensus error:', error);
      Alert.alert('Connection Error', 'Failed to connect to the server. Please try again.');
    } finally {
      setIsCreatingConsensus(false);
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'emergency': return '#dc2626';
      case 'urgent': return '#f59e0b';
      case 'standard': return '#3b82f6';
      case 'low': return '#10b981';
      default: return '#6b7280';
    }
  };

  const renderTriageTab = () => (
    <View style={styles.tabContent}>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Automated Triage Assessment</Text>
        <Text style={styles.cardSubtitle}>
          Answer a few questions to determine the urgency of your consultation
        </Text>

        <View style={styles.formField}>
          <Text style={styles.label}>Chief Complaint *</Text>
          <TextInput
            style={styles.textArea}
            value={chiefComplaint}
            onChangeText={setChiefComplaint}
            placeholder="Describe your main concern..."
            placeholderTextColor="#9ca3af"
            multiline
            numberOfLines={3}
          />
        </View>

        <View style={styles.formField}>
          <Text style={styles.label}>How long have you had this?</Text>
          <View style={styles.optionRow}>
            {['< 1 week', '1-4 weeks', '1-3 months', '> 3 months'].map((option) => (
              <Pressable
                key={option}
                style={[styles.optionChip, symptomDuration === option && styles.optionChipSelected]}
                onPress={() => setSymptomDuration(option)}
              >
                <Text style={[styles.optionText, symptomDuration === option && styles.optionTextSelected]}>
                  {option}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        <View style={styles.formField}>
          <Text style={styles.label}>Pain Level: {painLevel}/10</Text>
          <View style={styles.painSlider}>
            {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((level) => (
              <Pressable
                key={level}
                style={[
                  styles.painDot,
                  painLevel >= level && styles.painDotActive,
                  level >= 7 && painLevel >= level && styles.painDotHigh
                ]}
                onPress={() => setPainLevel(level)}
              >
                <Text style={styles.painDotText}>{level}</Text>
              </Pressable>
            ))}
          </View>
        </View>

        <View style={styles.toggleRow}>
          <Pressable
            style={[styles.toggleButton, isSpreading && styles.toggleButtonActive]}
            onPress={() => setIsSpreading(!isSpreading)}
          >
            <Ionicons
              name={isSpreading ? 'checkbox' : 'square-outline'}
              size={20}
              color={isSpreading ? '#fff' : '#6b7280'}
            />
            <Text style={[styles.toggleText, isSpreading && styles.toggleTextActive]}>
              Spreading/Growing
            </Text>
          </Pressable>

          <Pressable
            style={[styles.toggleButton, hasFever && styles.toggleButtonActive]}
            onPress={() => setHasFever(!hasFever)}
          >
            <Ionicons
              name={hasFever ? 'checkbox' : 'square-outline'}
              size={20}
              color={hasFever ? '#fff' : '#6b7280'}
            />
            <Text style={[styles.toggleText, hasFever && styles.toggleTextActive]}>
              Fever Present
            </Text>
          </Pressable>
        </View>

        <Pressable
          style={[styles.submitButton, isLoading && styles.submitButtonDisabled]}
          onPress={submitTriage}
          disabled={isLoading}
        >
          {isLoading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <>
              <Ionicons name="medical" size={20} color="#fff" />
              <Text style={styles.submitButtonText}>Assess Urgency</Text>
            </>
          )}
        </Pressable>
      </View>

      {triageResult && (
        <View style={[styles.resultCard, { borderLeftColor: getPriorityColor(triageResult.priority) }]}>
          <View style={styles.resultHeader}>
            <View style={[styles.priorityBadge, { backgroundColor: getPriorityColor(triageResult.priority) }]}>
              <Text style={styles.priorityText}>{triageResult.priority.toUpperCase()}</Text>
            </View>
            <Text style={styles.priorityScore}>Score: {triageResult.priority_score}/100</Text>
          </View>

          <Text style={styles.waitTime}>
            Estimated Wait: {triageResult.estimated_wait_time}
          </Text>

          {triageResult.risk_factors.length > 0 && (
            <View style={styles.riskFactors}>
              <Text style={styles.sectionLabel}>Risk Factors Identified:</Text>
              {triageResult.risk_factors.map((factor, index) => (
                <View key={index} style={styles.riskFactorItem}>
                  <Ionicons name="warning" size={14} color="#f59e0b" />
                  <Text style={styles.riskFactorText}>{factor}</Text>
                </View>
              ))}
            </View>
          )}

          <View style={styles.recommendations}>
            <Text style={styles.sectionLabel}>Recommendations:</Text>
            {triageResult.routing_recommendations.map((rec, index) => (
              <View key={index} style={styles.recItem}>
                <Ionicons name="checkmark-circle" size={14} color="#10b981" />
                <Text style={styles.recText}>{rec}</Text>
              </View>
            ))}
          </View>
        </View>
      )}
    </View>
  );

  const renderVideoTab = () => (
    <View style={styles.tabContent}>
      <View style={styles.card}>
        <View style={styles.videoHeader}>
          <Ionicons name="videocam" size={48} color="#8b5cf6" />
          <Text style={styles.cardTitle}>Live Video Consultation</Text>
          <Text style={styles.cardSubtitle}>
            Connect with a dermatologist via secure video with real-time annotation tools
          </Text>
        </View>

        {!videoSession ? (
          <Pressable
            style={[styles.startVideoButton, isCreatingVideo && styles.startVideoButtonDisabled]}
            onPress={createVideoSession}
            disabled={isCreatingVideo}
          >
            {isCreatingVideo ? (
              <ActivityIndicator color="#fff" size="small" />
            ) : (
              <Ionicons name="videocam" size={24} color="#fff" />
            )}
            <Text style={styles.startVideoText}>
              {isCreatingVideo ? 'Creating Session...' : 'Start Video Session'}
            </Text>
          </Pressable>
        ) : (
          <View style={styles.videoContainer}>
            <View style={styles.videoPlaceholder}>
              <Ionicons name="person-circle" size={80} color="#d1d5db" />
              <Text style={styles.videoStatus}>
                Session: {videoSession.session_id}
              </Text>
              <Text style={styles.videoStatusSub}>
                Status: {videoSession.status}
              </Text>
            </View>

            {/* Annotation Toolbar */}
            <View style={styles.annotationToolbar}>
              <Text style={styles.toolbarTitle}>Annotation Tools</Text>
              <View style={styles.toolButtons}>
                {[
                  { id: 'draw', icon: 'brush', label: 'Draw' },
                  { id: 'arrow', icon: 'arrow-forward', label: 'Arrow' },
                  { id: 'circle', icon: 'ellipse-outline', label: 'Circle' },
                  { id: 'text', icon: 'text', label: 'Text' },
                  { id: 'measure', icon: 'resize', label: 'Measure' },
                ].map((tool) => (
                  <Pressable
                    key={tool.id}
                    style={[styles.toolButton, selectedTool === tool.id && styles.toolButtonActive]}
                    onPress={() => setSelectedTool(tool.id)}
                  >
                    <Ionicons
                      name={tool.icon as any}
                      size={20}
                      color={selectedTool === tool.id ? '#fff' : '#6b7280'}
                    />
                    <Text style={[styles.toolLabel, selectedTool === tool.id && styles.toolLabelActive]}>
                      {tool.label}
                    </Text>
                  </Pressable>
                ))}
              </View>

              <View style={styles.colorPicker}>
                {['#dc2626', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'].map((color) => (
                  <Pressable
                    key={color}
                    style={[styles.colorDot, { backgroundColor: color }]}
                  />
                ))}
              </View>
            </View>

            <View style={styles.videoActions}>
              <Pressable style={styles.videoActionButton}>
                <Ionicons name="mic" size={24} color="#374151" />
              </Pressable>
              <Pressable style={styles.videoActionButton}>
                <Ionicons name="camera-reverse" size={24} color="#374151" />
              </Pressable>
              <Pressable style={[styles.videoActionButton, styles.endCallButton]}>
                <Ionicons name="call" size={24} color="#fff" />
              </Pressable>
              <Pressable style={styles.videoActionButton}>
                <Ionicons name="image" size={24} color="#374151" />
              </Pressable>
              <Pressable style={styles.videoActionButton}>
                <Ionicons name="chatbubble" size={24} color="#374151" />
              </Pressable>
            </View>
          </View>
        )}

        <View style={styles.featureList}>
          <Text style={styles.featureTitle}>Video Features:</Text>
          {[
            'HD video with WebRTC technology',
            'Real-time drawing & annotations',
            'Screen sharing for images',
            'Session recording (with consent)',
            'Secure end-to-end encryption'
          ].map((feature, index) => (
            <View key={index} style={styles.featureItem}>
              <Ionicons name="checkmark-circle" size={16} color="#10b981" />
              <Text style={styles.featureText}>{feature}</Text>
            </View>
          ))}
        </View>
      </View>
    </View>
  );

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return '#f59e0b';
      case 'in_review': return '#3b82f6';
      case 'consensus_reached': return '#10b981';
      case 'disagreement': return '#ef4444';
      case 'escalated': return '#8b5cf6';
      default: return '#6b7280';
    }
  };

  const renderConsensusTab = () => (
    <View style={styles.tabContent}>
      {/* Specialist Access Card */}
      <Pressable
        style={styles.specialistCard}
        onPress={() => router.push('/specialist-review' as any)}
      >
        <View style={styles.specialistCardContent}>
          <Ionicons name="medical" size={32} color="#8b5cf6" />
          <View style={styles.specialistCardText}>
            <Text style={styles.specialistCardTitle}>Specialist Dashboard</Text>
            <Text style={styles.specialistCardSubtitle}>Review and submit opinions on assigned cases</Text>
          </View>
        </View>
        <Ionicons name="arrow-forward" size={20} color="#8b5cf6" />
      </Pressable>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Multi-Specialist Consensus</Text>
        <Text style={styles.cardSubtitle}>
          Get opinions from multiple dermatologists for complex cases
        </Text>

        <Pressable
          style={styles.createConsensusButton}
          onPress={() => setShowCreateConsensus(true)}
        >
          <Ionicons name="people" size={20} color="#fff" />
          <Text style={styles.createConsensusText}>Request Consensus Review</Text>
        </Pressable>

        <View style={styles.workflowSteps}>
          <Text style={styles.workflowTitle}>How It Works:</Text>
          {[
            { step: 1, title: 'Submit Case', desc: 'Upload images and clinical details' },
            { step: 2, title: 'Expert Review', desc: '2-3 specialists review independently' },
            { step: 3, title: 'Consensus', desc: 'Opinions are aggregated and analyzed' },
            { step: 4, title: 'Final Report', desc: 'Receive comprehensive diagnosis' },
          ].map((item) => (
            <View key={item.step} style={styles.workflowStep}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>{item.step}</Text>
              </View>
              <View style={styles.stepContent}>
                <Text style={styles.stepTitle}>{item.title}</Text>
                <Text style={styles.stepDesc}>{item.desc}</Text>
              </View>
            </View>
          ))}
        </View>
      </View>

      {/* User's Cases */}
      <View style={styles.card}>
        <View style={styles.casesHeader}>
          <Text style={styles.casesTitle}>Your Cases</Text>
          {isLoadingCases && <ActivityIndicator size="small" color="#8b5cf6" />}
        </View>

        {consensusCases.length === 0 ? (
          <View style={styles.emptyCases}>
            <Ionicons name="folder-open-outline" size={48} color="#d1d5db" />
            <Text style={styles.emptyCasesText}>No consensus cases yet</Text>
            <Text style={styles.emptyCasesSubtext}>Request a consensus review to get started</Text>
          </View>
        ) : (
          consensusCases.map((c) => (
            <Pressable
              key={c.case_id}
              style={styles.caseCard}
              onPress={() => Alert.alert(
                c.case_id,
                `Status: ${c.status.replace('_', ' ')}\nOpinions: ${c.opinions_received}/${c.opinions_expected}\n${c.consensus_diagnosis ? `Diagnosis: ${c.consensus_diagnosis}` : 'Awaiting consensus...'}`
              )}
            >
              <View style={styles.caseCardHeader}>
                <Text style={styles.caseId}>{c.case_id}</Text>
                <View style={[styles.statusBadge, { backgroundColor: getStatusColor(c.status) }]}>
                  <Text style={styles.statusBadgeText}>{c.status.replace('_', ' ')}</Text>
                </View>
              </View>

              <View style={styles.caseCardDetails}>
                <View style={styles.caseDetail}>
                  <Ionicons name="people-outline" size={14} color="#6b7280" />
                  <Text style={styles.caseDetailText}>
                    {c.opinions_received}/{c.opinions_expected} opinions
                  </Text>
                </View>
                <View style={styles.caseDetail}>
                  <Ionicons name="time-outline" size={14} color="#6b7280" />
                  <Text style={styles.caseDetailText}>
                    {new Date(c.created_at).toLocaleDateString()}
                  </Text>
                </View>
              </View>

              {c.consensus_diagnosis && (
                <View style={styles.diagnosisRow}>
                  <Ionicons name="checkmark-circle" size={16} color="#10b981" />
                  <Text style={styles.diagnosisText}>{c.consensus_diagnosis}</Text>
                  {c.agreement_ratio && (
                    <Text style={styles.agreementText}>
                      ({Math.round(c.agreement_ratio * 100)}% agreement)
                    </Text>
                  )}
                </View>
              )}
            </Pressable>
          ))
        )}
      </View>
    </View>
  );

  const renderStoreForwardTab = () => (
    <View style={styles.tabContent}>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Store-and-Forward</Text>
        <Text style={styles.cardSubtitle}>
          Submit your case for asynchronous specialist review
        </Text>

        <View style={styles.sfBenefits}>
          {[
            { icon: 'time', title: 'Convenient', desc: 'No scheduling needed' },
            { icon: 'document-text', title: 'Detailed', desc: 'Comprehensive written response' },
            { icon: 'cash', title: 'Cost-Effective', desc: 'Lower cost than video' },
          ].map((benefit, index) => (
            <View key={index} style={styles.benefitCard}>
              <Ionicons name={benefit.icon as any} size={32} color="#8b5cf6" />
              <Text style={styles.benefitTitle}>{benefit.title}</Text>
              <Text style={styles.benefitDesc}>{benefit.desc}</Text>
            </View>
          ))}
        </View>

        <Pressable
          style={styles.createSFButton}
          onPress={() => router.push('/consultations' as any)}
        >
          <Ionicons name="create" size={20} color="#fff" />
          <Text style={styles.createSFText}>Create New Case</Text>
        </Pressable>

        <View style={styles.responseTime}>
          <Ionicons name="timer" size={24} color="#f59e0b" />
          <View>
            <Text style={styles.responseLabel}>Typical Response Time</Text>
            <Text style={styles.responseValue}>24-48 hours for urgent cases</Text>
          </View>
        </View>
      </View>
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
        <Text style={styles.title}>Advanced Teledermatology</Text>
        <View style={{ width: 40 }} />
      </LinearGradient>

      {/* Tabs */}
      <View style={styles.tabBar}>
        {[
          { key: 'triage', label: 'Triage', icon: 'medical' },
          { key: 'video', label: 'Video', icon: 'videocam' },
          { key: 'consensus', label: 'Consensus', icon: 'people' },
          { key: 'store-forward', label: 'S&F', icon: 'document-text' },
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
        {activeTab === 'triage' && renderTriageTab()}
        {activeTab === 'video' && renderVideoTab()}
        {activeTab === 'consensus' && renderConsensusTab()}
        {activeTab === 'store-forward' && renderStoreForwardTab()}
        <View style={{ height: 40 }} />
      </ScrollView>

      {/* Consensus Request Modal */}
      <Modal
        visible={showCreateConsensus}
        animationType="slide"
        transparent={true}
        onRequestClose={() => {
          Keyboard.dismiss();
          setShowCreateConsensus(false);
        }}
      >
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.modalOverlay}
        >
          <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
            <View style={styles.modalOverlay}>
              <View style={styles.modalContent}>
                <View style={styles.modalHeader}>
                  <Text style={styles.modalTitle}>Request Consensus Review</Text>
                  <Pressable onPress={() => {
                    Keyboard.dismiss();
                    setShowCreateConsensus(false);
                  }}>
                    <Ionicons name="close" size={24} color="#6b7280" />
                  </Pressable>
                </View>

                <ScrollView
                  style={styles.modalScroll}
                  showsVerticalScrollIndicator={false}
                  keyboardShouldPersistTaps="handled"
                >
                  <Text style={styles.modalSubtitle}>
                    Submit your case for review by multiple dermatology specialists
                  </Text>

                  <View style={styles.formField}>
                    <Text style={styles.label}>Case Summary *</Text>
                    <TextInput
                      style={[styles.textArea, styles.modalTextArea]}
                      value={consensusSummary}
                      onChangeText={setConsensusSummary}
                      placeholder="Describe the case, clinical findings, and any relevant history..."
                      placeholderTextColor="#9ca3af"
                      multiline
                      numberOfLines={5}
                      textAlignVertical="top"
                    />
                  </View>

                  <View style={styles.formField}>
                    <Text style={styles.label}>Urgency Level</Text>
                    <View style={styles.optionRow}>
                      {[
                        { value: 'standard', label: 'Standard (72h)' },
                        { value: 'urgent', label: 'Urgent (24h)' },
                      ].map((option) => (
                        <Pressable
                          key={option.value}
                          style={[styles.optionChip, consensusUrgency === option.value && styles.optionChipSelected]}
                          onPress={() => setConsensusUrgency(option.value)}
                        >
                          <Text style={[styles.optionText, consensusUrgency === option.value && styles.optionTextSelected]}>
                            {option.label}
                          </Text>
                        </Pressable>
                      ))}
                    </View>
                  </View>

                  <View style={styles.infoBox}>
                    <Ionicons name="information-circle" size={20} color="#3b82f6" />
                    <Text style={styles.infoText}>
                      Your case will be reviewed by 3 board-certified dermatologists. You'll receive a comprehensive report once consensus is reached.
                    </Text>
                  </View>

                  <View style={styles.modalActions}>
                    <Pressable
                      style={styles.cancelButton}
                      onPress={() => {
                        Keyboard.dismiss();
                        setShowCreateConsensus(false);
                      }}
                    >
                      <Text style={styles.cancelButtonText}>Cancel</Text>
                    </Pressable>
                    <Pressable
                      style={[styles.submitButton, isCreatingConsensus && styles.submitButtonDisabled]}
                      onPress={() => {
                        Keyboard.dismiss();
                        createConsensusCase();
                      }}
                      disabled={isCreatingConsensus}
                    >
                      {isCreatingConsensus ? (
                        <ActivityIndicator color="#fff" size="small" />
                      ) : (
                        <>
                          <Ionicons name="send" size={18} color="#fff" />
                          <Text style={styles.submitButtonText}>Submit Request</Text>
                        </>
                      )}
                    </Pressable>
                  </View>
                  <View style={{ height: 20 }} />
                </ScrollView>
              </View>
            </View>
          </TouchableWithoutFeedback>
        </KeyboardAvoidingView>
      </Modal>
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
    fontSize: 18,
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
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 3,
    marginBottom: 16,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 4,
  },
  cardSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 20,
  },
  formField: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  textArea: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    color: '#1f2937',
    minHeight: 80,
    textAlignVertical: 'top',
  },
  optionRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  optionChip: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    backgroundColor: '#fff',
  },
  optionChipSelected: {
    backgroundColor: '#8b5cf6',
    borderColor: '#8b5cf6',
  },
  optionText: {
    fontSize: 12,
    color: '#6b7280',
  },
  optionTextSelected: {
    color: '#fff',
    fontWeight: '600',
  },
  painSlider: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  painDot: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  painDotActive: {
    backgroundColor: '#8b5cf6',
  },
  painDotHigh: {
    backgroundColor: '#dc2626',
  },
  painDotText: {
    fontSize: 10,
    color: '#6b7280',
    fontWeight: '600',
  },
  toggleRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  toggleButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  toggleButtonActive: {
    backgroundColor: '#8b5cf6',
    borderColor: '#8b5cf6',
  },
  toggleText: {
    fontSize: 13,
    color: '#6b7280',
  },
  toggleTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  submitButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#8b5cf6',
    padding: 16,
    borderRadius: 12,
    gap: 8,
  },
  submitButtonDisabled: {
    backgroundColor: '#d1d5db',
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  resultCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    marginTop: 16,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  priorityBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  priorityText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
  },
  priorityScore: {
    fontSize: 14,
    color: '#6b7280',
    fontWeight: '600',
  },
  waitTime: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 16,
  },
  sectionLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  riskFactors: {
    marginBottom: 16,
  },
  riskFactorItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  riskFactorText: {
    fontSize: 13,
    color: '#6b7280',
  },
  recommendations: {},
  recItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  recText: {
    fontSize: 13,
    color: '#374151',
  },
  videoHeader: {
    alignItems: 'center',
    marginBottom: 24,
  },
  startVideoButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#10b981',
    padding: 20,
    borderRadius: 12,
    gap: 12,
  },
  startVideoText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
  },
  videoContainer: {
    marginBottom: 20,
  },
  videoPlaceholder: {
    height: 200,
    backgroundColor: '#1f2937',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  videoStatus: {
    color: '#fff',
    fontSize: 14,
    marginTop: 8,
  },
  videoStatusSub: {
    color: '#9ca3af',
    fontSize: 12,
    marginTop: 4,
  },
  annotationToolbar: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  toolbarTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 12,
  },
  toolButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  toolButton: {
    alignItems: 'center',
    padding: 10,
    borderRadius: 8,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    minWidth: 56,
  },
  toolButtonActive: {
    backgroundColor: '#8b5cf6',
    borderColor: '#8b5cf6',
  },
  toolLabel: {
    fontSize: 10,
    color: '#6b7280',
    marginTop: 4,
  },
  toolLabelActive: {
    color: '#fff',
  },
  colorPicker: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
  },
  colorDot: {
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 2,
    borderColor: '#fff',
  },
  videoActions: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
  },
  videoActionButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  endCallButton: {
    backgroundColor: '#dc2626',
    transform: [{ rotate: '135deg' }],
  },
  featureList: {
    marginTop: 20,
  },
  featureTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 12,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 8,
  },
  featureText: {
    fontSize: 13,
    color: '#6b7280',
  },
  createConsensusButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#8b5cf6',
    padding: 16,
    borderRadius: 12,
    gap: 8,
    marginBottom: 24,
  },
  createConsensusText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  workflowSteps: {
    marginBottom: 20,
  },
  workflowTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 16,
  },
  workflowStep: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  stepNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#8b5cf6',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  stepNumberText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '700',
  },
  stepContent: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 2,
  },
  stepDesc: {
    fontSize: 12,
    color: '#6b7280',
  },
  casesList: {
    marginTop: 20,
  },
  casesTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 12,
  },
  caseCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  caseId: {
    fontSize: 13,
    fontWeight: '600',
    color: '#374151',
  },
  caseStatus: {
    fontSize: 12,
    color: '#6b7280',
  },
  sfBenefits: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  benefitCard: {
    flex: 1,
    alignItems: 'center',
    padding: 12,
  },
  benefitTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1f2937',
    marginTop: 8,
  },
  benefitDesc: {
    fontSize: 11,
    color: '#6b7280',
    textAlign: 'center',
    marginTop: 4,
  },
  createSFButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#3b82f6',
    padding: 16,
    borderRadius: 12,
    gap: 8,
    marginBottom: 16,
  },
  createSFText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  responseTime: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: '#fef3c7',
    padding: 16,
    borderRadius: 12,
  },
  responseLabel: {
    fontSize: 12,
    color: '#92400e',
  },
  responseValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#78350f',
  },
  startVideoButtonDisabled: {
    backgroundColor: '#9ca3af',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingTop: 24,
    paddingHorizontal: 24,
    maxHeight: '80%',
  },
  modalScroll: {
    flexGrow: 0,
  },
  modalTextArea: {
    minHeight: 120,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1f2937',
  },
  modalSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 24,
  },
  infoBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#eff6ff',
    padding: 12,
    borderRadius: 8,
    gap: 10,
    marginBottom: 24,
  },
  infoText: {
    flex: 1,
    fontSize: 13,
    color: '#1e40af',
    lineHeight: 18,
  },
  modalActions: {
    flexDirection: 'row',
    gap: 12,
  },
  cancelButton: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    alignItems: 'center',
  },
  cancelButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6b7280',
  },
  specialistCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#f3e8ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e9d5ff',
  },
  specialistCardContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
  },
  specialistCardText: {
    flex: 1,
  },
  specialistCardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#7c3aed',
  },
  specialistCardSubtitle: {
    fontSize: 12,
    color: '#8b5cf6',
    marginTop: 2,
  },
  casesHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  emptyCases: {
    alignItems: 'center',
    paddingVertical: 32,
  },
  emptyCasesText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#6b7280',
    marginTop: 12,
  },
  emptyCasesSubtext: {
    fontSize: 13,
    color: '#9ca3af',
    marginTop: 4,
  },
  caseCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusBadgeText: {
    fontSize: 11,
    fontWeight: '500',
    color: '#fff',
    textTransform: 'capitalize',
  },
  caseCardDetails: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 8,
  },
  caseDetail: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  caseDetailText: {
    fontSize: 12,
    color: '#6b7280',
  },
  diagnosisRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  diagnosisText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#1f2937',
  },
  agreementText: {
    fontSize: 12,
    color: '#6b7280',
  },
});
