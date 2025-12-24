import React, {useState, useRef, useEffect, useMemo, memo} from 'react';
import { View, Text, StyleSheet, TextInput, Button, Pressable, Image, ActivityIndicator, ScrollView, Dimensions, Animated, Modal } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { StatusBar } from 'expo-status-bar';
import { useRouter } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import { Alert } from "react-native";
import { Linking } from 'react-native';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';
import { API_ENDPOINTS, REQUEST_TIMEOUT } from '../config';
import { BinaryClassificationResponse, FullClassificationResponse, ImagePickerAsset } from '../types';
import ImageAnalysisService from '../ImageAnalysisService';
import { useAuth } from '../contexts/AuthContext';
import BodyMapSelector from '../components/BodyMapSelector';
import { HelpTooltip, InlineHelp, HelpBadge } from '../components/HelpTooltip';
import AlertsBanner from '../components/AlertsBanner';
import { ClinicalContextForm, ClinicalContext } from '../components/ClinicalContextForm';
import { CalibratedResultsDisplay, CalibratedUncertainty } from '../components/CalibratedResultsDisplay';
import { ABCDEFeatureDisplay, ABCDEAnalysis } from '../components/ABCDEFeatureDisplay';
import { useUserSettings } from '../contexts/UserSettingsContext';
import { DisplayModeToggle } from '../components/DisplayModeToggle';

// Memoized image component to prevent re-renders during progress updates
const MemoizedImage = memo(({ uri, style, resizeMode }: { uri: string, style: any, resizeMode: string }) => {
  console.log('MemoizedImage rendering');
  return <Image source={{uri}} style={style} resizeMode={resizeMode} />;
});

export default function PhotoScreen() {
  const { t } = useTranslation();
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [modelPreds, setModelPreds] = useState<string>("Waiting...");
  const { user, logout, isAuthenticated } = useAuth();
  const { settings: userSettings } = useUserSettings();
  const router = useRouter();

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    }
  }, [isAuthenticated]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isClassifying, setIsClassifying] = useState<boolean>(false);
  const [progressMessage, setProgressMessage] = useState<string>("");
  const [progressPercentage, setProgressPercentage] = useState<number>(0);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [showResults, setShowResults] = useState<boolean>(false);
  const [isRunningDetailedAnalysis, setIsRunningDetailedAnalysis] = useState<boolean>(false);
  const [analysisStep, setAnalysisStep] = useState<number>(0);
  const [professionalData, setProfessionalData] = useState<any>(null);
  const [imageQuality, setImageQuality] = useState<any>(null);
  const [showQualityCheck, setShowQualityCheck] = useState<boolean>(false);
  const [qualityCheckPassed, setQualityCheckPassed] = useState<boolean>(false);
  const [isExportingPDF, setIsExportingPDF] = useState<boolean>(false);
  const [showBodyMapSelector, setShowBodyMapSelector] = useState<boolean>(false);
  const [bodyMapData, setBodyMapData] = useState<any>(null);
  const [showMenu, setShowMenu] = useState<boolean>(false);
  const [showClinicalContext, setShowClinicalContext] = useState<boolean>(false);
  const [clinicalContext, setClinicalContext] = useState<ClinicalContext | null>(null);
  const [analysisType, setAnalysisType] = useState<'lesion' | 'infectious'>('lesion');

  // Menu category expansion state
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['Patient Monitoring']) // Default expanded category
  );

  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => {
      const newSet = new Set(prev);
      if (newSet.has(category)) {
        newSet.delete(category);
      } else {
        newSet.add(category);
      }
      return newSet;
    });
  };

  // Menu category component
  const MenuCategory = ({
    title,
    icon,
    children,
    count
  }: {
    title: string;
    icon: string;
    children: React.ReactNode;
    count: number;
  }) => {
    const isExpanded = expandedCategories.has(title);
    return (
      <View style={styles.menuCategory}>
        <Pressable
          style={styles.menuCategoryHeader}
          onPress={() => toggleCategory(title)}
        >
          <Text style={styles.menuCategoryIcon}>{icon}</Text>
          <View style={styles.menuCategoryTitleContainer}>
            <Text style={styles.menuCategoryTitle}>{title}</Text>
            <Text style={styles.menuCategoryCount}>{count} items</Text>
          </View>
          <Text style={styles.menuCategoryArrow}>
            {isExpanded ? '▼' : '▶'}
          </Text>
        </Pressable>
        {isExpanded && (
          <View style={styles.menuCategoryContent}>
            {children}
          </View>
        )}
      </View>
    );
  };

  // AI Explanation state
  const [aiExplanation, setAiExplanation] = useState<string | null>(null);
  const [isLoadingAiExplanation, setIsLoadingAiExplanation] = useState<boolean>(false);
  const [showAiExplanation, setShowAiExplanation] = useState<boolean>(false);
  const [aiExplanationError, setAiExplanationError] = useState<string | null>(null);

  // Differential Reasoning state
  const [differentialReasoning, setDifferentialReasoning] = useState<string | null>(null);
  const [isLoadingReasoning, setIsLoadingReasoning] = useState<boolean>(false);
  const [showReasoning, setShowReasoning] = useState<boolean>(false);
  const [reasoningError, setReasoningError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const progressAnim = useRef(new Animated.Value(0)).current;
  const spinAnim = useRef(new Animated.Value(0)).current;

  // Logout handler
  const handleLogout = async () => {
    Alert.alert(
      `🚪 ${t('auth.logout')}`,
      `Are you sure you want to logout, ${user?.username || 'User'}?\n\nYou'll need to login again to access the skin analysis features.`,
      [
        {
          text: t('home.cancel'),
          style: "cancel"
        },
        {
          text: `${t('common.yes')}, ${t('auth.logout')}`,
          style: "destructive",
          onPress: async () => {
            try {
              // Set loading state instead of alert
              setIsLoading(true);

              // Clear any ongoing operations immediately
              setIsClassifying(false);
              setShowResults(false);
              setImageUri(null);
              setAnalysisResult(null);

              // Perform logout
              await logout();

              // Navigate back to login
              router.replace('/');
            } catch (error) {
              console.error('Logout error:', error);
              setIsLoading(false);
              Alert.alert(t('common.error'), "Failed to logout. Please try again.");
            }
          }
        }
      ]
    );
  };

  //Choose method of uploading picture
  const uploadPhotoPress = () => {
    Alert.alert(
      t('home.uploadPhoto'),
      t('home.uploadSource'),
      [
        {text: t('home.camera'), onPress: cameraPhoto},
        {text: t('home.library'), onPress: libraryPhoto},
        {text: t('home.cancel'), style: "cancel"},
      ]
    );
  };

  const handleError = (error: Error, context: string) => {
    console.error(`${context}:`, error);

    // Check if it's an authentication error
    if (error.message.includes('401') || error.message.includes('Authentication') || error.message.includes('Unauthorized')) {
      Alert.alert(
        t('homeScreen.errors.sessionExpired'),
        t('homeScreen.errors.sessionExpiredMessage'),
        [
          {
            text: t('common.ok'),
            onPress: () => logout()
          }
        ]
      );
    } else {
      Alert.alert(t('common.error'), `${context} ${t('common.error').toLowerCase()}: ${error.message}`);
    }

    setIsLoading(false);
    setIsClassifying(false);
  };

  //Passes data to backend and receives classification with improved timeout and retry logic
  const handleLocationSelect = (location: any) => {
    setBodyMapData(location);
  };

  const proceedToClassify = () => {
    setShowBodyMapSelector(false);
    // Show clinical context form next
    setShowClinicalContext(true);
  };

  const handleClinicalContextSubmit = (context: ClinicalContext) => {
    setClinicalContext(context);
    setShowClinicalContext(false);
    performClassification(context);
  };

  const skipClinicalContext = () => {
    setClinicalContext(null);
    setShowClinicalContext(false);
    performClassification(null);
  };

  const classifyPhoto = async () => {
    if (!imageUri) {
      Alert.alert(t('common.error'), t('homeScreen.errors.uploadPhotoFirst'));
      return;
    }

    // Show body map selector first
    setShowBodyMapSelector(true);
  };

  const performClassification = async (contextData: ClinicalContext | null = null) => {
    setIsClassifying(true);
    setProgressMessage(t('homeScreen.progress.starting'));
    setProgressPercentage(0);
    setModelPreds(t('homeScreen.progress.waiting'));
    setShowResults(false);
    setAnalysisResult(null);

    try {
      const result = await ImageAnalysisService.analyzeWithRetry(
        imageUri,
        (message, percentage) => {
          setProgressMessage(message);
          if (percentage !== undefined) {
            setProgressPercentage(percentage);
          }
        },
        bodyMapData,
        analysisType,
        contextData  // Pass clinical context to analysis
      );

      console.log("Analysis result:", result);
      console.log("Burn result available:", !!result.burnResult);
      console.log("Formatted burn result available:", !!result.formattedBurnResult);
      console.log("Dermoscopy result available:", !!result.dermoscopyResult);

      setAnalysisResult(result);
      setShowResults(true);

      // Update the display text and professional data based on result type
      if (result.isLesion && result.formattedResult) {
        const professionalResult = ImageAnalysisService.formatAnalysisResult(result.fullResult);

        // Get primary condition type for filtering
        const primaryCondition = result.fullResult?.primary_condition_type;

        // Add burn data to professional result ONLY if burn is the primary condition
        if (result.formattedBurnResult && primaryCondition === 'burn') {
          console.log("Adding burn data to professional result (burn is primary condition)");
          professionalResult.burnData = result.formattedBurnResult;
        } else if (result.formattedBurnResult) {
          console.log(`Skipping burn data - primary condition is ${primaryCondition}, not burn`);
          professionalResult.burnData = null;
        } else {
          console.log("No burn data to add");
        }

        // Add dermoscopy data to professional result if available
        // SKIP dermoscopy for burns, inflammatory, and infectious conditions - melanoma screening not relevant
        const skipDermoscopy = primaryCondition === 'burn' ||
                               primaryCondition === 'inflammatory' ||
                               primaryCondition === 'infectious';

        if (result.dermoscopyResult && !skipDermoscopy) {
          console.log("Adding dermoscopy data to professional result");
          professionalResult.dermoscopyData = result.dermoscopyResult;
        } else if (skipDermoscopy) {
          console.log(`Skipping dermoscopy data - not relevant for ${primaryCondition} condition`);
          professionalResult.dermoscopyData = null;
        } else {
          console.log("No dermoscopy data to add");
        }

        // Add calibrated uncertainty data (clinical-grade uncertainty categories)
        if (result.fullResult?.calibrated_uncertainty) {
          console.log("Adding calibrated uncertainty to professional result");
          professionalResult.calibratedUncertainty = result.fullResult.calibrated_uncertainty;
        } else {
          console.log("No calibrated uncertainty data available");
          professionalResult.calibratedUncertainty = null;
        }

        // Add ABCDE feature analysis data (quantitative explainability for dermatologists)
        if (result.fullResult?.abcde_analysis) {
          console.log("Adding ABCDE analysis to professional result");
          professionalResult.abcdeAnalysis = result.fullResult.abcde_analysis;
        } else {
          console.log("No ABCDE analysis data available");
          professionalResult.abcdeAnalysis = null;
        }

        // Add multimodal analysis data (combined image + labs + history)
        if (result.fullResult?.multimodal_analysis) {
          console.log("Adding multimodal analysis to professional result");
          professionalResult.multimodalAnalysis = result.fullResult.multimodal_analysis;
        } else {
          console.log("No multimodal analysis data available");
          professionalResult.multimodalAnalysis = null;
        }

        console.log("Professional result with all data:", professionalResult);
        setProfessionalData(professionalResult);
        setModelPreds(professionalResult.formattedText);
      } else if (!result.isLesion) {
        setModelPreds(t('homeScreen.progress.nonLesionDetected'));
        setProfessionalData(null);
      }

    } catch (error) {
      if (error instanceof Error) {
        handleError(error, t('homeScreen.errors.imageFailed'));
        setModelPreds(t('homeScreen.errors.analysisFailed'));
      } else {
        handleError(new Error('Unknown error'), t('homeScreen.errors.imageFailed'));
        setModelPreds(t('homeScreen.errors.analysisFailed'));
      }
    } finally {
      setIsClassifying(false);
      setProgressMessage("");
      setProgressPercentage(0);
    }
  };

  // Function to run detailed analysis on non-lesion images
  const runDetailedAnalysisOnNonLesion = async () => {
    if (!imageUri || !analysisResult) return;

    setIsRunningDetailedAnalysis(true);
    setProgressMessage(t('homeScreen.progress.runningDetailed'));
    setProgressPercentage(0);

    try {
      const detailedResult = await ImageAnalysisService.runFullAnalysisOnNonLesion(
        imageUri,
        (message, percentage) => {
          setProgressMessage(message);
          if (percentage !== undefined) {
            setProgressPercentage(percentage);
          }
        }
      );

      const professionalResult = ImageAnalysisService.formatAnalysisResult(detailedResult.fullResult);
      setProfessionalData(professionalResult);
      setModelPreds(professionalResult.formattedText);
      setAnalysisResult({
        ...analysisResult,
        fullResult: detailedResult.fullResult,
        formattedResult: professionalResult.formattedText
      });

    } catch (error) {
      if (error instanceof Error) {
        handleError(error, t('homeScreen.errors.detailedFailed'));
      } else {
        handleError(new Error('Unknown error'), t('homeScreen.errors.detailedFailed'));
      }
    } finally {
      setIsRunningDetailedAnalysis(false);
      setProgressMessage("");
      setProgressPercentage(0);
    }
  };

  // Function to start over with new image
  const startOver = () => {
    setImageUri(null);
    setModelPreds(t('homeScreen.progress.waiting'));
    setAnalysisResult(null);
    setShowResults(false);
    setProgressMessage("");
    setProgressPercentage(0);
    setAnalysisStep(0);
    setProfessionalData(null);
    setImageQuality(null);
    setShowQualityCheck(false);
    setQualityCheckPassed(false);
    setIsExportingPDF(false);
    // Reset AI explanation state
    setAiExplanation(null);
    setShowAiExplanation(false);
    setAiExplanationError(null);
    // Reset differential reasoning state
    setDifferentialReasoning(null);
    setShowReasoning(false);
    setReasoningError(null);
  };

  // Function to fetch AI explanation for a condition
  const fetchAIExplanation = async (condition: string, severity?: string) => {
    if (!condition || condition === 'Unknown') {
      setAiExplanationError('No condition to explain');
      return;
    }

    setIsLoadingAiExplanation(true);
    setAiExplanationError(null);
    setShowAiExplanation(true);

    try {
      const token = await require('expo-secure-store').getItemAsync('auth_token');
      const { API_BASE_URL } = require('../config');

      const formData = new FormData();
      formData.append('condition', condition);
      if (severity) {
        formData.append('severity', severity);
      }

      const response = await fetch(`${API_BASE_URL}/ai-chat/explain-condition`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get AI explanation');
      }

      const data = await response.json();
      setAiExplanation(data.explanation);
    } catch (error: any) {
      console.error('AI explanation error:', error);
      setAiExplanationError(error.message || 'Failed to load explanation. Please try again.');
    } finally {
      setIsLoadingAiExplanation(false);
    }
  };

  // Function to fetch differential diagnosis reasoning (chain-of-thought)
  const fetchDifferentialReasoning = async () => {
    const diagnosis = professionalData?.diagnosis || analysisResult?.fullResult?.predicted_class;
    if (!diagnosis) {
      setReasoningError('No diagnosis available for reasoning');
      return;
    }

    setIsLoadingReasoning(true);
    setReasoningError(null);
    setShowReasoning(true);

    try {
      const token = await require('expo-secure-store').getItemAsync('auth_token');
      const { API_BASE_URL } = require('../config');

      const formData = new FormData();
      formData.append('primary_diagnosis', diagnosis);

      // Add confidence
      const confidence = professionalData?.confidence || analysisResult?.fullResult?.confidence;
      if (confidence) {
        formData.append('confidence', confidence.toString());
      }

      // Add risk level
      const riskLevel = professionalData?.riskLevel || analysisResult?.fullResult?.risk_level;
      if (riskLevel) {
        formData.append('risk_level', riskLevel);
      }

      // Add differential diagnoses
      const differentials = professionalData?.differentialDiagnoses ||
                           analysisResult?.fullResult?.differential_diagnoses?.lesion ||
                           analysisResult?.fullResult?.differential_diagnoses;
      if (differentials && Array.isArray(differentials)) {
        formData.append('differential_diagnoses', JSON.stringify(differentials));
      }

      // Add clinical context if available
      if (clinicalContext) {
        formData.append('clinical_context', JSON.stringify(clinicalContext));
      }

      const response = await fetch(`${API_BASE_URL}/ai-chat/differential-reasoning`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get reasoning');
      }

      const data = await response.json();
      setDifferentialReasoning(data.reasoning);
    } catch (error: any) {
      console.error('Differential reasoning error:', error);
      setReasoningError(error.message || 'Failed to load reasoning. Please try again.');
    } finally {
      setIsLoadingReasoning(false);
    }
  };

  // Function to handle PDF export
  const handlePDFExport = async (analysisData: any) => {
    if (!analysisData || !imageUri) {
      Alert.alert(t('common.error'), t('homeScreen.errors.noAnalysisData'));
      return;
    }

    setIsExportingPDF(true);

    try {
      await ImageAnalysisService.exportAndSharePDF(analysisData, imageUri);
      Alert.alert(t('common.success'), t('homeScreen.alerts.pdfSuccess'));
    } catch (error) {
      console.error('PDF export error:', error);
      Alert.alert(t('common.error'), t('homeScreen.alerts.pdfError'));
    } finally {
      setIsExportingPDF(false);
    }
  };

  // Professional Loading Animations
  React.useEffect(() => {
    if (isClassifying || isRunningDetailedAnalysis) {
      // Start pulse animation
      const pulseAnimation = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.2,
            duration: 1000,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 1000,
            useNativeDriver: true,
          }),
        ])
      );

      // Start spin animation
      const spinAnimation = Animated.loop(
        Animated.timing(spinAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        })
      );

      pulseAnimation.start();
      spinAnimation.start();

      return () => {
        pulseAnimation.stop();
        spinAnimation.stop();
      };
    } else {
      pulseAnim.setValue(1);
      spinAnim.setValue(0);
      progressAnim.setValue(0);
    }
  }, [isClassifying, isRunningDetailedAnalysis]);

  // Update analysis step based on progress message
  React.useEffect(() => {
    if (progressMessage.includes('Optimizing image')) setAnalysisStep(1);
    else if (progressMessage.includes('Connecting')) setAnalysisStep(2);
    else if (progressMessage.includes('Lesion detected')) setAnalysisStep(3);
    else if (progressMessage.includes('detailed analysis')) setAnalysisStep(4);
    else if (progressMessage.includes('Starting')) setAnalysisStep(0);
  }, [progressMessage]);

  // Medical Loading Spinner Component
  const MedicalLoadingSpinner = ({ size = 60, color = '#4299e1' }) => {
    const spin = spinAnim.interpolate({
      inputRange: [0, 1],
      outputRange: ['0deg', '360deg'],
    });

    return (
      <View style={styles.loadingSpinnerContainer}>
        <Animated.View
          style={[
            styles.loadingSpinner,
            {
              width: size,
              height: size,
              borderColor: color,
              transform: [{ rotate: spin }]
            }
          ]}
        />
        <View style={[styles.loadingSpinnerCenter, {
          width: size * 0.3,
          height: size * 0.3,
          backgroundColor: color
        }]} />
      </View>
    );
  };

  // Progress Bar Component
  const AnalysisProgressBar = ({ progress }) => {
    React.useEffect(() => {
      Animated.timing(progressAnim, {
        toValue: progress,
        duration: 500,
        useNativeDriver: false,
      }).start();
    }, [progress]);

    const progressWidth = progressAnim.interpolate({
      inputRange: [0, 1],
      outputRange: ['0%', '100%'],
    });

    const steps = [
      t('homeScreen.steps.preparing'),
      t('homeScreen.steps.optimizingImage'),
      t('homeScreen.steps.initialAnalysis'),
      t('homeScreen.steps.detailedClassification'),
      t('homeScreen.steps.complete')
    ];

    return (
      <View style={styles.progressBarContainer}>
        <Text style={styles.progressTitle}>{t('homeScreen.progressBar.title')}</Text>
        <View style={styles.progressBar}>
          <Animated.View
            style={[
              styles.progressBarFill,
              { width: progressWidth }
            ]}
          />
        </View>
        <View style={styles.progressSteps}>
          {steps.map((step, index) => (
            <View key={index} style={styles.progressStep}>
              <View style={[
                styles.progressStepDot,
                index <= Math.floor(progress * (steps.length - 1)) ? styles.progressStepActive : {}
              ]} />
              <Text style={[
                styles.progressStepText,
                index <= Math.floor(progress * (steps.length - 1)) ? styles.progressStepTextActive : {}
              ]}>
                {step}
              </Text>
            </View>
          ))}
        </View>
      </View>
    );
  };

  // Image Quality Validation Display Component (Scrollable)
  const ImageQualityDisplay = ({ qualityData }) => {
    if (!qualityData) return null;

    const assessment = ImageAnalysisService.getQualityAssessment(qualityData.score);

    return (
      <View style={styles.qualityFullScreenContainer}>
        <ScrollView
          style={styles.qualityScrollView}
          contentContainerStyle={styles.qualityScrollContent}
          showsVerticalScrollIndicator={true}
        >
          {/* Display the image being checked */}
          {qualityData.metadata && qualityData.metadata.uri && (
            <View style={styles.qualityImageContainer}>
              <Image
                source={{uri: qualityData.metadata.uri}}
                style={styles.qualityImage}
                resizeMode="contain"
              />
            </View>
          )}

          <View style={styles.qualityContainer}>
            <View style={[styles.qualityCard, qualityData.passed ? styles.qualityPassedCard : styles.qualityFailedCard]}>
              <View style={styles.qualityHeader}>
                <Text style={styles.qualityTitle}>📸 {t('homeScreen.quality.title')}</Text>
                <View style={[styles.qualityBadge, { backgroundColor: assessment.color }]}>
                  <Text style={styles.qualityBadgeText}>{assessment.emoji} {assessment.level}</Text>
                </View>
              </View>

              <View style={styles.qualityScore}>
                <Text style={styles.qualityScoreText}>{t('homeScreen.quality.score', { score: Math.round(qualityData.score * 100) })}</Text>
              </View>

              {qualityData.metadata && (
                <View style={styles.qualityMetadata}>
                  <Text style={styles.qualityMetadataText}>
                    {t('homeScreen.quality.resolution', { width: qualityData.metadata.width, height: qualityData.metadata.height, size: qualityData.metadata.fileSizeKB })}
                  </Text>
                </View>
              )}

              {qualityData.issues && qualityData.issues.length > 0 && (
                <View style={styles.qualityIssues}>
                  <Text style={styles.qualityIssuesTitle}>{t('homeScreen.quality.issuesDetected')}</Text>
                  {qualityData.issues.map((issue, index) => (
                    <View key={index} style={[styles.qualityIssueItem,
                      { borderLeftColor: issue.severity === 'high' ? '#ef4444' : issue.severity === 'medium' ? '#f59e0b' : '#6b7280' }]}>
                      <Text style={[styles.qualityIssueMessage,
                        { color: issue.severity === 'high' ? '#dc2626' : issue.severity === 'medium' ? '#d97706' : '#4b5563' }]}>
                        {issue.message}
                      </Text>
                      <Text style={styles.qualityIssueDescription}>{issue.description}</Text>
                    </View>
                  ))}
                </View>
              )}

              {qualityData.recommendations && qualityData.recommendations.length > 0 && (
                <View style={styles.qualityRecommendations}>
                  <Text style={styles.qualityRecommendationsTitle}>{t('homeScreen.quality.recommendations')}</Text>
                  {qualityData.recommendations.map((rec, index) => (
                    <Text key={index} style={styles.qualityRecommendationItem}>• {rec}</Text>
                  ))}
                </View>
              )}

              {qualityData.passed && (
                <View style={styles.qualitySuccessSection}>
                  <Text style={styles.qualityPassedText}>✅ {t('homeScreen.quality.acceptable')}</Text>
                </View>
              )}

              {/* Spacer to ensure buttons are always accessible */}
              <View style={styles.qualityButtonSpacer} />
            </View>
          </View>
        </ScrollView>

        {/* Fixed buttons at bottom of screen */}
        <View style={styles.qualityFixedButtons}>
          {!qualityData.passed && (
            <>
              <Pressable style={[styles.button, styles.retryButton, styles.qualityFixedButton]} onPress={uploadPhotoPress}>
                <Text style={styles.buttonText}>📷 {t('homeScreen.quality.tryDifferent')}</Text>
              </Pressable>
              <Pressable
                style={[styles.button, styles.continueAnywayButton, styles.qualityFixedButton]}
                onPress={() => {
                  setImageUri(qualityData.metadata.uri);
                  setShowQualityCheck(false);
                  setQualityCheckPassed(true);
                }}
              >
                <Text style={styles.buttonText}>⚠️ {t('homeScreen.quality.continueAnyway')}</Text>
              </Pressable>
            </>
          )}
          {qualityData.passed && (
            <Pressable
              style={[styles.button, styles.primaryButton, styles.qualityFixedButton]}
              onPress={() => {
                setShowQualityCheck(false);
                setQualityCheckPassed(true);
              }}
            >
              <Text style={styles.buttonText}>✅ {t('homeScreen.quality.continueToAnalysis')}</Text>
            </Pressable>
          )}
        </View>
      </View>
    );
  };

  // Professional Photo Tips Component
  const PhotoTipsComponent = () => (
    <View style={styles.photoTipsContainer}>
      <Text style={styles.photoTipsTitle}>📋 {t('homeScreen.photoTips.title')}</Text>
      <View style={styles.photoTipsList}>
        <View style={styles.photoTip}>
          <Text style={styles.photoTipEmoji}>🔍</Text>
          <Text style={styles.photoTipText}>{t('homeScreen.photoTips.fillFrame')}</Text>
        </View>
        <View style={styles.photoTip}>
          <Text style={styles.photoTipEmoji}>💡</Text>
          <Text style={styles.photoTipText}>{t('homeScreen.photoTips.lighting')}</Text>
        </View>
        <View style={styles.photoTip}>
          <Text style={styles.photoTipEmoji}>📱</Text>
          <Text style={styles.photoTipText}>{t('homeScreen.photoTips.steady')}</Text>
        </View>
        <View style={styles.photoTip}>
          <Text style={styles.photoTipEmoji}>📏</Text>
          <Text style={styles.photoTipText}>{t('homeScreen.photoTips.distance')}</Text>
        </View>
      </View>
    </View>
  );

  // Burn Results Display Component
  const BurnResultsDisplay = ({ analysisData }) => {
    console.log("BurnResultsDisplay called with data:", analysisData);
    if (!analysisData) {
      console.log("BurnResultsDisplay: No analysis data, returning null");
      return null;
    }

    console.log("BurnResultsDisplay: Rendering burn results");
    return (
      <View style={styles.professionalResultsContainer}>
        {/* Burn Severity Card */}
        <View style={styles.primaryDiagnosisCard}>
          {/* Header with title and timestamp stacked vertically */}
          <View style={{ alignItems: 'center', marginBottom: 15, borderBottomWidth: 1, borderBottomColor: '#e2e8f0', paddingBottom: 10 }}>
            <Text style={styles.diagnosisTitle}>🔥 Burn Severity Classification</Text>
            <Text style={[styles.diagnosisTimestamp, { marginTop: 6 }]}>
              {new Date(analysisData.timestamp).toLocaleString()}
            </Text>
          </View>

          <View style={styles.primaryResult}>
            <Text style={styles.predictedClass}>{analysisData.severityClass || 'Burn Detected'}</Text>
            <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
              <View style={styles.confidenceContainer}>
                <Text style={styles.confidenceLabel}>Confidence:</Text>
                <Text style={[styles.confidenceValue, { color: analysisData.confidenceLevel?.color || '#666' }]}>
                  {analysisData.confidence || 'N/A'}% ({analysisData.confidenceLevel?.level || 'N/A'})
                </Text>
              </View>
            </View>
            <Text style={styles.confidenceDescription}>
              {analysisData.confidenceLevel?.description || ''}
            </Text>
          </View>
        </View>

        {/* Urgency Assessment Card */}
        <View style={[styles.riskAssessmentCard, { borderLeftColor: analysisData.riskLevel?.color || '#f59e0b' }]}>
          <View style={styles.riskHeader}>
            <Text style={styles.riskTitle}>⚠️ Urgency Assessment</Text>
            <View style={[styles.riskBadge, { backgroundColor: analysisData.riskLevel?.color || '#f59e0b' }]}>
              <Text style={styles.riskLevel}>{analysisData.riskLevel?.level || 'Unknown'}</Text>
            </View>
          </View>
          <Text style={styles.riskRecommendation}>
            {analysisData.urgency || 'Seek medical evaluation'}
          </Text>
        </View>

        {/* Treatment Advice Card */}
        <View style={styles.treatmentAdviceCard}>
          <Text style={styles.treatmentAdviceTitle}>💊 Treatment Recommendations</Text>
          <Text style={styles.treatmentAdviceText}>
            {analysisData.treatmentAdvice}
          </Text>
          {analysisData.medicalAttentionRequired && (
            <View style={styles.medicalAttentionWarning}>
              <Text style={styles.medicalAttentionText}>
                ⚕️ MEDICAL ATTENTION REQUIRED
              </Text>
            </View>
          )}
        </View>

        {/* Detailed Probabilities Card - only render if probabilities exist */}
        {analysisData.probabilities && analysisData.probabilities.length > 0 && (
          <View style={styles.probabilitiesCard}>
            <Text style={styles.probabilitiesTitle}>📊 Burn Severity Probabilities</Text>
            {analysisData.probabilities.map((prob, index) => (
              <View key={prob.severity || index} style={styles.probabilityRow}>
                <View style={styles.probabilityInfo}>
                  <Text style={[
                    styles.probabilityLabel,
                    index === 0 ? styles.topProbabilityLabel : {}
                  ]}>
                    {prob.severity}
                  </Text>
                  <Text style={[
                    styles.probabilityValue,
                    index === 0 ? styles.topProbabilityValue : {}
                  ]}>
                    {prob.percentage}%
                  </Text>
                </View>
                <View style={styles.probabilityBarContainer}>
                  <View
                    style={[
                      styles.probabilityBar,
                      {
                        width: `${(prob.probability || 0) * 100}%`,
                        backgroundColor: index === 0 ? analysisData.confidenceLevel?.color || '#3b82f6' : '#e2e8f0'
                      }
                    ]}
                  />
                </View>
              </View>
            ))}
          </View>
        )}
      </View>
    );
  };

  // Dermoscopy Results Display Component
  // NOTE: This component should NOT be rendered for burns, inflammatory, or infectious conditions
  const DermoscopyResultsDisplay = ({ analysisData }) => {
    console.log("DermoscopyResultsDisplay called with data:", analysisData);
    if (!analysisData) {
      console.log("DermoscopyResultsDisplay: No analysis data, returning null");
      return null;
    }

    // Double-check: skip if dermoscopy indicates it's not applicable
    if (analysisData?.analysis_applicable === false || analysisData?.skipped_reason) {
      console.log("DermoscopyResultsDisplay: Analysis not applicable, returning null");
      return null;
    }

    console.log("DermoscopyResultsDisplay: Rendering dermoscopy results");

    const getRiskColor = (riskLevel) => {
      switch (riskLevel?.toUpperCase()) {
        case 'HIGH':
        case 'CRITICAL':
          return '#dc2626';
        case 'MODERATE':
          return '#f59e0b';
        case 'LOW-MODERATE':
          return '#fbbf24';
        case 'LOW':
          return '#22c55e';
        default:
          return '#94a3b8';
      }
    };

    const getUrgencyColor = (urgency) => {
      if (urgency?.toLowerCase().includes('urgent') || urgency?.toLowerCase().includes('emergency')) {
        return '#dc2626';
      } else if (urgency?.toLowerCase().includes('routine')) {
        return '#22c55e';
      }
      return '#f59e0b';
    };

    return (
      <View style={styles.dermoscopyContainer}>
        {/* Header */}
        <View style={styles.dermoscopyHeader}>
          <Text style={styles.dermoscopyHeaderTitle}>🔬 Dermoscopic Analysis</Text>
          <Text style={styles.dermoscopyHeaderSubtitle}>
            Professional melanoma screening using 7-Point Checklist & ABCD scoring
          </Text>
        </View>

        {/* Clinical Scores Row */}
        <View style={styles.dermoscopyScoresRow}>
          {/* 7-Point Checklist Score */}
          <View style={[
            styles.dermoscopyScoreCard,
            { borderLeftColor: getUrgencyColor(analysisData.seven_point_checklist?.urgency) }
          ]}>
            <Text style={styles.scoreCardLabel}>7-Point Checklist</Text>
            <Text style={styles.scoreCardValue}>
              {analysisData.seven_point_checklist?.score || 0}/{analysisData.seven_point_checklist?.max_score || 9}
            </Text>
            <Text style={[
              styles.scoreCardInterpretation,
              { color: getUrgencyColor(analysisData.seven_point_checklist?.urgency) }
            ]}>
              {analysisData.seven_point_checklist?.urgency || 'Unknown'}
            </Text>
          </View>

          {/* ABCD Score */}
          <View style={[
            styles.dermoscopyScoreCard,
            { borderLeftColor: getRiskColor(analysisData.abcd_score?.classification) }
          ]}>
            <Text style={styles.scoreCardLabel}>ABCD Score</Text>
            <Text style={styles.scoreCardValue}>
              {analysisData.abcd_score?.total_score?.toFixed(2) || '0.00'}
            </Text>
            <Text style={[
              styles.scoreCardClassification,
              { color: getRiskColor(analysisData.abcd_score?.classification) }
            ]}>
              {analysisData.abcd_score?.classification || 'Unknown'}
            </Text>
          </View>
        </View>

        {/* Clinical Interpretation */}
        {analysisData.seven_point_checklist?.interpretation && (
          <View style={styles.dermoscopyInterpretationCard}>
            <Text style={styles.interpretationTitle}>Clinical Interpretation:</Text>
            <Text style={styles.interpretationText}>
              {analysisData.seven_point_checklist.interpretation}
            </Text>
          </View>
        )}

        {/* ABCD Recommendation */}
        {analysisData.abcd_score?.recommendation && (
          <View style={styles.dermoscopyRecommendationCard}>
            <Text style={styles.recommendationTitle}>ABCD Recommendation:</Text>
            <Text style={styles.recommendationText}>
              {analysisData.abcd_score.recommendation}
            </Text>
          </View>
        )}

        {/* Detected Features Grid */}
        <Text style={styles.dermoscopyFeaturesTitle}>Detected Dermoscopic Features</Text>
        <View style={styles.dermoscopyFeaturesGrid}>
          {/* Pigment Network */}
          {analysisData.pigment_network?.detected && (
            <View style={[
              styles.featureCard,
              { borderLeftColor: getRiskColor(analysisData.pigment_network.risk_level) }
            ]}>
              <Text style={styles.featureIcon}>🕸️</Text>
              <Text style={styles.featureTitle}>Pigment Network</Text>
              <Text style={styles.featureType}>{analysisData.pigment_network.type}</Text>
              <Text style={[
                styles.featureRisk,
                { color: getRiskColor(analysisData.pigment_network.risk_level) }
              ]}>
                {analysisData.pigment_network.risk_level} risk
              </Text>
              <Text style={styles.featureDescription}>
                {analysisData.pigment_network.description}
              </Text>
            </View>
          )}

          {/* Globules */}
          {analysisData.globules?.detected && (
            <View style={[
              styles.featureCard,
              { borderLeftColor: getRiskColor(analysisData.globules.risk_level) }
            ]}>
              <Text style={styles.featureIcon}>⚫</Text>
              <Text style={styles.featureTitle}>Globules</Text>
              <Text style={styles.featureCount}>Count: {analysisData.globules.count}</Text>
              <Text style={styles.featureType}>{analysisData.globules.type}</Text>
              <Text style={[
                styles.featureRisk,
                { color: getRiskColor(analysisData.globules.risk_level) }
              ]}>
                {analysisData.globules.risk_level} risk
              </Text>
            </View>
          )}

          {/* Streaks */}
          {analysisData.streaks?.detected && (
            <View style={[
              styles.featureCard,
              { borderLeftColor: getRiskColor(analysisData.streaks.risk_level) }
            ]}>
              <Text style={styles.featureIcon}>📏</Text>
              <Text style={styles.featureTitle}>Streaks</Text>
              <Text style={styles.featureCount}>Count: {analysisData.streaks.count}</Text>
              <Text style={styles.featureType}>{analysisData.streaks.type}</Text>
              <Text style={[
                styles.featureRisk,
                { color: getRiskColor(analysisData.streaks.risk_level) }
              ]}>
                {analysisData.streaks.risk_level} risk
              </Text>
            </View>
          )}

          {/* Blue-White Veil - HIGH PRIORITY */}
          {analysisData.blue_white_veil?.detected && (
            <View style={[
              styles.featureCard,
              styles.highRiskFeatureCard,
              { borderLeftColor: '#dc2626' }
            ]}>
              <Text style={styles.featureIcon}>⚠️</Text>
              <Text style={styles.featureTitle}>Blue-White Veil</Text>
              <Text style={styles.highRiskBadge}>HIGH RISK</Text>
              <Text style={styles.featureCoverage}>
                Coverage: {analysisData.blue_white_veil.coverage_percentage?.toFixed(1)}%
              </Text>
              <Text style={styles.featureIntensity}>
                Intensity: {analysisData.blue_white_veil.intensity}
              </Text>
              <Text style={styles.highRiskDescription}>
                {analysisData.blue_white_veil.description}
              </Text>
            </View>
          )}

          {/* Vascular Patterns */}
          {analysisData.vascular_patterns?.detected && (
            <View style={[
              styles.featureCard,
              { borderLeftColor: getRiskColor(analysisData.vascular_patterns.risk_level) }
            ]}>
              <Text style={styles.featureIcon}>🩸</Text>
              <Text style={styles.featureTitle}>Vascular Patterns</Text>
              <Text style={styles.featureType}>{analysisData.vascular_patterns.type}</Text>
              <Text style={styles.featureCount}>
                Vessels: {analysisData.vascular_patterns.vessel_count}
              </Text>
              <Text style={[
                styles.featureRisk,
                { color: getRiskColor(analysisData.vascular_patterns.risk_level) }
              ]}>
                {analysisData.vascular_patterns.risk_level} risk
              </Text>
            </View>
          )}

          {/* Regression */}
          {analysisData.regression?.detected && (
            <View style={[
              styles.featureCard,
              { borderLeftColor: getRiskColor(analysisData.regression.risk_level) }
            ]}>
              <Text style={styles.featureIcon}>◻️</Text>
              <Text style={styles.featureTitle}>Regression</Text>
              <Text style={styles.featureSeverity}>{analysisData.regression.severity}</Text>
              <Text style={styles.featureCoverage}>
                Coverage: {analysisData.regression.coverage_percentage?.toFixed(1)}%
              </Text>
              <Text style={[
                styles.featureRisk,
                { color: getRiskColor(analysisData.regression.risk_level) }
              ]}>
                {analysisData.regression.risk_level} risk
              </Text>
            </View>
          )}
        </View>

        {/* Color Analysis */}
        {analysisData.color_analysis && (
          <View style={styles.colorAnalysisCard}>
            <Text style={styles.colorAnalysisTitle}>🎨 Color Analysis</Text>
            <View style={styles.colorAnalysisRow}>
              <View style={styles.colorMetric}>
                <Text style={styles.colorMetricLabel}>Distinct Colors</Text>
                <Text style={styles.colorMetricValue}>
                  {analysisData.color_analysis.distinct_colors}
                </Text>
              </View>
              <View style={styles.colorMetric}>
                <Text style={styles.colorMetricLabel}>Variety</Text>
                <Text style={styles.colorMetricValue}>
                  {analysisData.color_analysis.variety}
                </Text>
              </View>
              <View style={styles.colorMetric}>
                <Text style={styles.colorMetricLabel}>Risk</Text>
                <Text style={[
                  styles.colorMetricValue,
                  { color: getRiskColor(analysisData.color_analysis.risk_level) }
                ]}>
                  {analysisData.color_analysis.risk_level}
                </Text>
              </View>
            </View>
            {analysisData.color_analysis.distinct_colors >= 3 && (
              <Text style={styles.colorWarning}>
                ⚠️ Multiple colors detected (melanoma typically has 3+ colors)
              </Text>
            )}
          </View>
        )}

        {/* Symmetry Analysis */}
        {analysisData.symmetry_analysis && (
          <View style={styles.symmetryAnalysisCard}>
            <Text style={styles.symmetryAnalysisTitle}>⚖️ Symmetry Analysis</Text>
            <View style={styles.symmetryRow}>
              <View style={styles.symmetryMetric}>
                <Text style={styles.symmetryMetricLabel}>Asymmetry Score</Text>
                <Text style={styles.symmetryMetricValue}>
                  {(analysisData.symmetry_analysis.asymmetry_score * 100).toFixed(1)}%
                </Text>
              </View>
              <View style={styles.symmetryMetric}>
                <Text style={styles.symmetryMetricLabel}>Classification</Text>
                <Text style={styles.symmetryMetricValue}>
                  {analysisData.symmetry_analysis.classification}
                </Text>
              </View>
            </View>
          </View>
        )}

        {/* Visual Overlay */}
        {analysisData.overlays?.combined && (
          <View style={styles.dermoscopyOverlaySection}>
            <Text style={styles.overlayTitle}>📊 Feature Visualization</Text>
            <Text style={styles.overlaySubtitle}>
              Detected patterns highlighted on image
            </Text>
            <Image
              source={{ uri: `data:image/png;base64,${analysisData.overlays.combined}` }}
              style={styles.dermoscopyOverlayImage}
              resizeMode="contain"
            />
            <Text style={styles.overlayLegend}>
              Green: Pigment Network | Red: Globules | Blue: Streaks | Yellow: Blue-White Veil
            </Text>
          </View>
        )}

        {/* Overall Risk Assessment */}
        {analysisData.risk_assessment && (
          <View style={[
            styles.dermoscopyRiskCard,
            { borderLeftColor: getRiskColor(analysisData.risk_assessment.risk_level) }
          ]}>
            <Text style={styles.riskTitle}>Overall Dermoscopic Risk Assessment</Text>

            <View style={[
              styles.riskBadge,
              { backgroundColor: getRiskColor(analysisData.risk_assessment.risk_level) }
            ]}>
              <Text style={styles.riskBadgeText}>
                {analysisData.risk_assessment.risk_level} RISK
              </Text>
            </View>

            <Text style={styles.riskScore}>
              Risk Score: {analysisData.risk_assessment.risk_score}
            </Text>

            <Text style={styles.riskRecommendation}>
              {analysisData.risk_assessment.recommendation}
            </Text>

            {analysisData.risk_assessment.risk_factors?.length > 0 && (
              <View style={styles.riskFactorsSection}>
                <Text style={styles.riskFactorsTitle}>Identified Risk Factors:</Text>
                {analysisData.risk_assessment.risk_factors.map((factor, index) => (
                  <View key={index} style={styles.riskFactorRow}>
                    <Text style={styles.riskFactorBullet}>•</Text>
                    <Text style={styles.riskFactorText}>{factor}</Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}

        {/* 7-Point Criteria Met */}
        {analysisData.seven_point_checklist?.criteria_met?.length > 0 && (
          <View style={styles.criteriaMetCard}>
            <Text style={styles.criteriaMetTitle}>7-Point Checklist - Criteria Met:</Text>
            {analysisData.seven_point_checklist.criteria_met.map((criterion, index) => (
              <View key={index} style={styles.criterionRow}>
                <Text style={styles.criterionBullet}>✓</Text>
                <Text style={styles.criterionText}>{criterion}</Text>
              </View>
            ))}
          </View>
        )}
      </View>
    );
  };

  // Professional Results Display Component
  const ProfessionalResultsDisplay = ({ analysisData, burnData, dermoscopyData }) => {
    if (!analysisData) return null;

    return (
      <View style={styles.professionalResultsContainer}>
        {/* Primary Diagnosis Card */}
        <View style={styles.primaryDiagnosisCard}>
          <View style={styles.diagnosisHeader}>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Text style={styles.diagnosisTitle}>🔬 {t('homeScreen.diagnosis.title')}</Text>
              <HelpTooltip
                title={t('homeScreen.diagnosis.aiDiagnosisTitle')}
                content={t('homeScreen.diagnosis.aiDiagnosisHelp')}
                size={18}
                color="#2c5282"
              />
            </View>
            <Text style={styles.diagnosisTimestamp}>
              {new Date(analysisData.timestamp).toLocaleString()}
            </Text>
          </View>

          <View style={styles.primaryResult}>
            <Text style={styles.predictedClass}>{analysisData.predictedClass || 'Analysis Pending'}</Text>
            <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
              <View style={styles.confidenceContainer}>
                <Text style={styles.confidenceLabel}>{t('homeScreen.diagnosis.confidence')}</Text>
                <Text style={[styles.confidenceValue, { color: analysisData.confidenceLevel?.color || '#666' }]}>
                  {analysisData.confidence || 'N/A'}% ({analysisData.confidenceLevel?.level || 'N/A'})
                </Text>
              </View>
              <HelpTooltip
                title={t('homeScreen.diagnosis.confidenceLevelTitle')}
                content={t('homeScreen.diagnosis.confidenceLevelHelp')}
                size={16}
                color={analysisData.confidenceLevel?.color || '#666'}
              />
            </View>
            <Text style={styles.confidenceDescription}>
              {analysisData.confidenceLevel?.description || ''}
            </Text>
          </View>
        </View>

        {/* Risk Assessment Card */}
        <View style={[styles.riskAssessmentCard, { borderLeftColor: analysisData.riskLevel?.color || '#666' }]}>
          <View style={styles.riskHeader}>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Text style={styles.riskTitle}>⚠️ {t('homeScreen.risk.title')}</Text>
              <HelpTooltip
                title={t('homeScreen.risk.title')}
                content={t('homeScreen.risk.helpContent')}
                size={18}
                color="#2c5282"
              />
            </View>
            <View style={[styles.riskBadge, { backgroundColor: analysisData.riskLevel?.color || '#666' }]}>
              <Text style={styles.riskLevel}>{analysisData.riskLevel?.level || 'Unknown'}</Text>
            </View>
          </View>
          <Text style={styles.riskRecommendation}>
            {analysisData.riskLevel?.recommendation || 'Consult a healthcare professional'}
          </Text>
        </View>

        {/* Detailed Probabilities Card - only render if probabilities exist */}
        {analysisData.probabilities && analysisData.probabilities.length > 0 && (
          <View style={styles.probabilitiesCard}>
            <Text style={styles.probabilitiesTitle}>📊 {t('homeScreen.detailedAnalysis.title')}</Text>
            {analysisData.probabilities.map((prob, index) => (
              <View key={prob.key || index} style={styles.probabilityRow}>
                <View style={styles.probabilityInfo}>
                  <Text style={[
                    styles.probabilityLabel,
                    index === 0 ? styles.topProbabilityLabel : {}
                  ]}>
                    {prob.label}
                  </Text>
                  <Text style={[
                    styles.probabilityValue,
                    index === 0 ? styles.topProbabilityValue : {}
                  ]}>
                    {prob.percentage}%
                  </Text>
                </View>
                <View style={styles.probabilityBarContainer}>
                  <View
                    style={[
                      styles.probabilityBar,
                      {
                        width: `${(prob.probability || 0) * 100}%`,
                        backgroundColor: index === 0 ? analysisData.confidenceLevel?.color || '#3b82f6' : '#e2e8f0'
                      }
                    ]}
                  />
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Inflammatory Conditions Card */}
        {analysisData.inflammatoryCondition && (
          <View style={styles.inflammatoryCard}>
            <Text style={styles.inflammatoryTitle}>🔥 {t('homeScreen.inflammatory.title')}</Text>

            <View style={styles.inflammatoryResult}>
              <Text style={styles.inflammatoryCondition}>{analysisData.inflammatoryCondition}</Text>
              {analysisData.inflammatoryConfidence && (
                <View style={styles.confidenceContainer}>
                  <Text style={styles.confidenceLabel}>{t('homeScreen.inflammatory.confidence')}</Text>
                  <Text style={[styles.confidenceValue, {
                    color: parseFloat(analysisData.inflammatoryConfidence) > 70 ? '#28a745' :
                           parseFloat(analysisData.inflammatoryConfidence) > 50 ? '#ffc107' : '#dc3545'
                  }]}>
                    {analysisData.inflammatoryConfidence}%
                  </Text>
                </View>
              )}
            </View>

            {/* Inflammatory Probabilities */}
            {Object.keys(analysisData.inflammatoryProbabilities).length > 0 && (
              <View style={styles.inflammatoryProbabilities}>
                <Text style={styles.probabilitiesSubtitle}>{t('homeScreen.inflammatory.conditionProbabilities')}</Text>
                {Object.entries(analysisData.inflammatoryProbabilities)
                  .sort(([,a], [,b]) => b - a)
                  .slice(0, 3)
                  .map(([condition, probability]) => (
                    <View key={condition} style={styles.probabilityRow}>
                      <View style={styles.probabilityInfo}>
                        <Text style={styles.probabilityLabel}>{condition}</Text>
                        <Text style={styles.probabilityValue}>{(probability * 100).toFixed(1)}%</Text>
                      </View>
                      <View style={styles.probabilityBarContainer}>
                        <View
                          style={[
                            styles.probabilityBar,
                            {
                              width: `${probability * 100}%`,
                              backgroundColor: '#ff6b6b'
                            }
                          ]}
                        />
                      </View>
                    </View>
                  ))}
              </View>
            )}
          </View>
        )}

        {/* Infectious Disease Card */}
        {analysisData.infectiousDisease && (
          <View style={styles.infectiousCard}>
            <Text style={styles.infectiousTitle}>🦠 {t('homeScreen.infectious.title')}</Text>

            <View style={styles.infectiousResult}>
              <Text style={styles.infectiousDisease}>{analysisData.infectiousDisease}</Text>
              {analysisData.infectiousConfidence && (
                <View style={styles.confidenceContainer}>
                  <Text style={styles.confidenceLabel}>{t('homeScreen.infectious.confidence')}</Text>
                  <Text style={[styles.confidenceValue, {
                    color: parseFloat(analysisData.infectiousConfidence) > 70 ? '#28a745' :
                           parseFloat(analysisData.infectiousConfidence) > 50 ? '#ffc107' : '#dc3545'
                  }]}>
                    {analysisData.infectiousConfidence}%
                  </Text>
                </View>
              )}
            </View>

            {/* Infection Details */}
            {(analysisData.infectionType || analysisData.infectiousSeverity || analysisData.contagious !== undefined) && (
              <View style={styles.infectionDetails}>
                {analysisData.infectionType && (
                  <View style={styles.infectionDetailRow}>
                    <Text style={styles.infectionDetailLabel}>{t('homeScreen.infectious.type')}</Text>
                    <Text style={styles.infectionDetailValue}>{analysisData.infectionType}</Text>
                  </View>
                )}
                {analysisData.infectiousSeverity && (
                  <View style={styles.infectionDetailRow}>
                    <Text style={styles.infectionDetailLabel}>{t('homeScreen.infectious.severity')}</Text>
                    <Text style={[styles.infectionDetailValue, {
                      color: analysisData.infectiousSeverity === 'severe' ? '#dc3545' :
                             analysisData.infectiousSeverity === 'moderate' ? '#ffc107' : '#28a745'
                    }]}>
                      {analysisData.infectiousSeverity}
                    </Text>
                  </View>
                )}
                {analysisData.contagious !== undefined && (
                  <View style={styles.infectionDetailRow}>
                    <Text style={styles.infectionDetailLabel}>{t('homeScreen.infectious.contagious')}</Text>
                    <Text style={[styles.infectionDetailValue, {
                      color: analysisData.contagious ? '#dc3545' : '#28a745'
                    }]}>
                      {analysisData.contagious ? t('homeScreen.infectious.yes') : t('homeScreen.infectious.no')}
                    </Text>
                  </View>
                )}
                {analysisData.transmissionRisk && (
                  <View style={styles.infectionDetailRow}>
                    <Text style={styles.infectionDetailLabel}>{t('homeScreen.infectious.transmissionRisk')}</Text>
                    <Text style={[styles.infectionDetailValue, {
                      color: analysisData.transmissionRisk === 'high' ? '#dc3545' :
                             analysisData.transmissionRisk === 'medium' ? '#ffc107' : '#28a745'
                    }]}>
                      {analysisData.transmissionRisk}
                    </Text>
                  </View>
                )}
              </View>
            )}

            {/* Infectious Probabilities */}
            {Object.keys(analysisData.infectiousProbabilities || {}).length > 0 && (
              <View style={styles.infectiousProbabilities}>
                <Text style={styles.probabilitiesSubtitle}>{t('homeScreen.infectious.diseaseProbabilities')}</Text>
                {Object.entries(analysisData.infectiousProbabilities)
                  .sort(([,a], [,b]) => b - a)
                  .slice(0, 3)
                  .map(([disease, probability]) => (
                    <View key={disease} style={styles.probabilityRow}>
                      <View style={styles.probabilityInfo}>
                        <Text style={styles.probabilityLabel}>{disease.replace(/_/g, ' ')}</Text>
                        <Text style={styles.probabilityValue}>{(probability * 100).toFixed(1)}%</Text>
                      </View>
                      <View style={styles.probabilityBarContainer}>
                        <View
                          style={[
                            styles.probabilityBar,
                            {
                              width: `${probability * 100}%`,
                              backgroundColor: '#9333ea'
                            }
                          ]}
                        />
                      </View>
                    </View>
                  ))}
              </View>
            )}
          </View>
        )}

        {/* Burn Analysis Results - ONLY show when burn is the primary condition */}
        {burnData && analysisData?.primaryConditionType === 'burn' && (
          <BurnResultsDisplay analysisData={burnData} />
        )}

        {/* Dermoscopy Analysis Results - ONLY show for neoplastic/lesion conditions */}
        {/* Skip for burns, inflammatory, and infectious conditions where melanoma screening is not relevant */}
        {dermoscopyData &&
         analysisData?.primaryConditionType !== 'burn' &&
         analysisData?.primaryConditionType !== 'inflammatory' &&
         analysisData?.primaryConditionType !== 'infectious' && (
          <DermoscopyResultsDisplay analysisData={dermoscopyData} />
        )}

        {/* Differential Diagnoses Card */}
        {analysisData.differentialDiagnoses && (
          (analysisData.differentialDiagnoses.lesion?.length > 0 || analysisData.differentialDiagnoses.inflammatory?.length > 0 || analysisData.differentialDiagnoses.infectious?.length > 0) && (
            <View style={styles.differentialCard}>
              <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginBottom: 8 }}>
                <Text style={styles.differentialTitle}>🩺 {t('homeScreen.differential.title')}</Text>
                <HelpTooltip
                  title={t('homeScreen.differential.title')}
                  content={t('homeScreen.differential.helpContent')}
                  size={18}
                  color="#2c5282"
                />
              </View>
              <Text style={styles.differentialSubtext}>
                {t('homeScreen.differential.rankedByProbability')}
              </Text>

              {/* Lesion Differential Diagnoses */}
              {analysisData.differentialDiagnoses.lesion?.length > 0 && (
                <View style={styles.differentialSubsection}>
                  <Text style={styles.differentialCategory}>{t('homeScreen.differential.lesionClassification')}</Text>
                  {analysisData.differentialDiagnoses.lesion.slice(0, 3).map((diagnosis, index) => (
                    <View key={index} style={styles.diagnosisCard}>
                      <View style={styles.diagnosisHeader}>
                        <Text style={styles.diagnosisRank}>#{index + 1}</Text>
                        <Text style={styles.diagnosisCondition}>{diagnosis.condition}</Text>
                        <Text style={styles.diagnosisProbability}>
                          {Math.round(diagnosis.probability * 100)}%
                        </Text>
                      </View>
                      <View style={[
                        styles.severityBadge,
                        {
                          backgroundColor:
                            diagnosis.severity === 'critical' ? '#dc3545' :
                            diagnosis.severity === 'high' ? '#fd7e14' :
                            diagnosis.severity === 'medium' ? '#ffc107' :
                            '#28a745'
                        }
                      ]}>
                        <Text style={styles.severityText}>{diagnosis.severity.toUpperCase()}</Text>
                      </View>
                      <Text style={styles.diagnosisUrgency}>⏱️ {diagnosis.urgency}</Text>
                      <Text style={styles.diagnosisDescription}>{diagnosis.description}</Text>
                    </View>
                  ))}
                </View>
              )}

              {/* Inflammatory Differential Diagnoses */}
              {analysisData.differentialDiagnoses.inflammatory?.length > 0 && (
                <View style={styles.differentialSubsection}>
                  <Text style={styles.differentialCategory}>{t('homeScreen.differential.inflammatoryConditions')}</Text>
                  {analysisData.differentialDiagnoses.inflammatory.slice(0, 3).map((diagnosis, index) => (
                    <View key={index} style={styles.diagnosisCard}>
                      <View style={styles.diagnosisHeader}>
                        <Text style={styles.diagnosisRank}>#{index + 1}</Text>
                        <Text style={styles.diagnosisCondition}>{diagnosis.condition}</Text>
                        <Text style={styles.diagnosisProbability}>
                          {Math.round(diagnosis.probability * 100)}%
                        </Text>
                      </View>
                      <View style={[
                        styles.severityBadge,
                        {
                          backgroundColor:
                            diagnosis.severity === 'high' ? '#fd7e14' :
                            diagnosis.severity === 'medium' ? '#ffc107' :
                            '#28a745'
                        }
                      ]}>
                        <Text style={styles.severityText}>{diagnosis.severity.toUpperCase()}</Text>
                      </View>
                      <Text style={styles.diagnosisUrgency}>⏱️ {diagnosis.urgency}</Text>
                      <Text style={styles.diagnosisDescription}>{diagnosis.description}</Text>
                    </View>
                  ))}
                </View>
              )}

              {/* Infectious Disease Differential Diagnoses */}
              {analysisData.differentialDiagnoses.infectious?.length > 0 && (
                <View style={styles.differentialSubsection}>
                  <Text style={styles.differentialCategory}>{t('homeScreen.differential.infectiousDiseases')}</Text>
                  {analysisData.differentialDiagnoses.infectious.slice(0, 3).map((diagnosis, index) => (
                    <View key={index} style={styles.diagnosisCard}>
                      <View style={styles.diagnosisHeader}>
                        <Text style={styles.diagnosisRank}>#{index + 1}</Text>
                        <Text style={styles.diagnosisCondition}>{diagnosis.condition}</Text>
                        <Text style={styles.diagnosisProbability}>
                          {Math.round(diagnosis.probability * 100)}%
                        </Text>
                      </View>
                      <View style={[
                        styles.severityBadge,
                        {
                          backgroundColor:
                            diagnosis.severity === 'severe' ? '#dc3545' :
                            diagnosis.severity === 'moderate' ? '#ffc107' :
                            '#28a745'
                        }
                      ]}>
                        <Text style={styles.severityText}>{diagnosis.severity.toUpperCase()}</Text>
                      </View>
                      {diagnosis.contagious && (
                        <Text style={styles.contagiousWarning}>
                          ⚠️ Contagious - {diagnosis.transmission_risk} transmission risk
                        </Text>
                      )}
                      <Text style={styles.diagnosisUrgency}>⏱️ {diagnosis.urgency}</Text>
                      <Text style={styles.diagnosisDescription}>{diagnosis.description}</Text>
                    </View>
                  ))}
                </View>
              )}
            </View>
          )
        )}

        {/* Uncertainty Quantification Card - ONLY for lesion analysis, not burns/inflammatory/infectious */}
        {analysisData.uncertaintyMetrics &&
         Object.keys(analysisData.uncertaintyMetrics).length > 0 &&
         analysisData.primaryConditionType !== 'burn' &&
         analysisData.primaryConditionType !== 'inflammatory' &&
         analysisData.primaryConditionType !== 'infectious' && (
          <View style={styles.uncertaintyCard}>
            <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginBottom: 15 }}>
              <Text style={styles.uncertaintyTitle}>🎯 {t('homeScreen.uncertainty.title')}</Text>
              <HelpTooltip
                title={t('homeScreen.uncertainty.quantificationTitle')}
                content={t('homeScreen.uncertainty.quantificationHelp')}
                size={18}
                color="#4338ca"
              />
            </View>

            <View style={styles.uncertaintyMetrics}>
              {/* Reliability Score */}
              {analysisData.uncertaintyMetrics.reliability_score !== undefined && (
                <View style={styles.uncertaintyRow}>
                  <Text style={styles.uncertaintyLabel}>Model Confidence:</Text>
                  <View style={styles.uncertaintyValueContainer}>
                    <Text style={[styles.uncertaintyValue, {
                      color: analysisData.uncertaintyMetrics.reliability_score > 0.8 ? '#28a745' :
                             analysisData.uncertaintyMetrics.reliability_score > 0.6 ? '#ffc107' : '#dc3545'
                    }]}>
                      {(analysisData.uncertaintyMetrics.reliability_score * 100).toFixed(1)}%
                    </Text>
                    <View style={[styles.reliabilityBar, {
                      backgroundColor: analysisData.uncertaintyMetrics.reliability_score > 0.8 ? '#28a745' :
                                      analysisData.uncertaintyMetrics.reliability_score > 0.6 ? '#ffc107' : '#dc3545',
                      width: `${analysisData.uncertaintyMetrics.reliability_score * 100}%`
                    }]} />
                  </View>
                </View>
              )}

              {/* Uncertainty Breakdown */}
              <View style={styles.uncertaintyBreakdown}>
                <Text style={styles.uncertaintySubtitle}>Uncertainty Analysis:</Text>

                {analysisData.uncertaintyMetrics.epistemic_uncertainty !== undefined && (
                  <View style={styles.uncertaintyMetric}>
                    <Text style={styles.metricLabel}>Model Uncertainty:</Text>
                    <Text style={styles.metricValue}>
                      {analysisData.uncertaintyMetrics.epistemic_uncertainty.toFixed(3)}
                    </Text>
                  </View>
                )}

                {analysisData.uncertaintyMetrics.aleatoric_uncertainty !== undefined && (
                  <View style={styles.uncertaintyMetric}>
                    <Text style={styles.metricLabel}>Data Uncertainty:</Text>
                    <Text style={styles.metricValue}>
                      {analysisData.uncertaintyMetrics.aleatoric_uncertainty.toFixed(3)}
                    </Text>
                  </View>
                )}

                {analysisData.uncertaintyMetrics.predictive_entropy !== undefined && (
                  <View style={styles.uncertaintyMetric}>
                    <Text style={styles.metricLabel}>Prediction Entropy:</Text>
                    <Text style={styles.metricValue}>
                      {analysisData.uncertaintyMetrics.predictive_entropy.toFixed(3)}
                    </Text>
                  </View>
                )}
              </View>

              {/* Clinical Recommendation based on uncertainty */}
              <View style={styles.clinicalRecommendation}>
                <Text style={styles.recommendationTitle}>Clinical Guidance:</Text>
                <Text style={[styles.recommendationText, {
                  color: analysisData.uncertaintyMetrics.reliability_score > 0.8 ? '#28a745' :
                         analysisData.uncertaintyMetrics.reliability_score > 0.6 ? '#f59e0b' : '#dc2626'
                }]}>
                  {analysisData.uncertaintyMetrics.reliability_score > 0.8
                    ? "High confidence prediction - suitable for preliminary assessment"
                    : analysisData.uncertaintyMetrics.reliability_score > 0.6
                    ? "Moderate confidence - consider additional imaging or expert review"
                    : "Low confidence prediction - dermatologist consultation strongly recommended"}
                </Text>
              </View>

              {/* MC Dropout Info */}
              {analysisData.mcSamplesUsed > 0 && (
                <View style={styles.mcInfo}>
                  <Text style={styles.mcInfoText}>
                    Analysis based on {analysisData.mcSamplesUsed} Monte Carlo samples
                  </Text>
                </View>
              )}
            </View>
          </View>
        )}

        {/* Analysis Metadata */}
        <View style={styles.metadataCard}>
          <Text style={styles.metadataTitle}>📋 Analysis Details</Text>
          <View style={styles.metadataRow}>
            <Text style={styles.metadataLabel}>Analysis Type:</Text>
            <Text style={styles.metadataValue}>{analysisData.analysisType}</Text>
          </View>
          <View style={styles.metadataRow}>
            <Text style={styles.metadataLabel}>Processing Time:</Text>
            <Text style={styles.metadataValue}>AI-Powered Deep Learning</Text>
          </View>
          <View style={styles.metadataRow}>
            <Text style={styles.metadataLabel}>Date & Time:</Text>
            <Text style={styles.metadataValue}>
              {new Date(analysisData.timestamp).toLocaleDateString()} at {new Date(analysisData.timestamp).toLocaleTimeString()}
            </Text>
          </View>
        </View>

        {/* Medical Disclaimer */}
        <View style={styles.disclaimerCard}>
          <Text style={styles.disclaimerTitle}>⚠️ Medical Disclaimer</Text>
          <Text style={styles.disclaimerText}>
            This AI analysis is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for definitive diagnosis and treatment decisions.
          </Text>
        </View>

        {/* PDF Export Button */}
        <View style={styles.exportButtonContainer}>
          <Pressable
            style={[styles.button, styles.exportButton, isExportingPDF && styles.buttonDisabled]}
            onPress={() => handlePDFExport(analysisData)}
            disabled={isExportingPDF}
          >
            <Text style={styles.buttonText}>
              {isExportingPDF ? "📄 " + t('common.loading') : "📄 " + t('homeScreen.buttons.exportPDF')}
            </Text>
            {isExportingPDF && <MedicalLoadingSpinner size={20} color="white" />}
          </Pressable>
        </View>
      </View>
    );
  };

  // Validate image quality after selection
  const validateAndSetImage = async (uri: string) => {
    setIsLoading(true);
    setShowQualityCheck(true);
    setImageQuality(null);
    setQualityCheckPassed(false);

    try {
      console.log('Validating image quality for:', uri);
      const qualityResult = await ImageAnalysisService.validateImageQuality(uri);
      setImageQuality(qualityResult);
      setQualityCheckPassed(qualityResult.passed);

      if (qualityResult.passed) {
        setImageUri(uri);
        setModelPreds(t('homeScreen.progress.waiting'));
        console.log('Image quality validation passed:', qualityResult);
      } else {
        console.log('Image quality validation failed:', qualityResult);
        // Don't set imageUri if validation fails - user needs to fix issues first
      }
    } catch (error) {
      console.error('Image quality validation error:', error);
      // On validation error, just proceed with the image
      setImageUri(uri);
      setModelPreds(t('homeScreen.progress.waiting'));
      setImageQuality(null);
      setShowQualityCheck(false);
      setQualityCheckPassed(true);
    }

    setIsLoading(false);
  };

  //Gets camera permission and updates imageUri
  const cameraPhoto = async () => {
    setIsLoading(true);
    const { granted } = await ImagePicker.requestCameraPermissionsAsync();

    if (!granted) {
      Alert.alert("Permission Denied", "Camera permission is required to take photos.");
      setIsLoading(false);
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled) {
      const photo: ImagePickerAsset = result.assets[0];
      await validateAndSetImage(photo.uri);
      console.log("Camera photo:", photo);
    }

    setIsLoading(false);
  };

  //Gets library permission and updates imageUri
  const libraryPhoto = async () => {
    setIsLoading(true);
    const { granted } = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!granted) {
      Alert.alert("Permission Denied", "Photo library permission is required to select photos.");
      setIsLoading(false);
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled) {
      const photo: ImagePickerAsset = result.assets[0];
      await validateAndSetImage(photo.uri);
      console.log("Library photo:", photo);
    }

    setIsLoading(false);
  };

  // Note: runFullClassify is now handled within the ImageAnalysisService
   
  // Medical Background Component
  const MedicalBackground = () => (
    <View style={styles.backgroundContainer}>
      <LinearGradient
        colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.backgroundContainer}
      />
      {/* Subtle medical cross pattern overlay */}
      <View style={styles.overlayPattern}>
        <View style={styles.medicalPattern} />
      </View>
    </View>
  );

  // Results Screen Component
  const ResultsScreen = () => {
    const screenData = Dimensions.get('window');
    const imageHeight = Math.min(screenData.height * 0.4, 300); // Max 40% of screen height or 300px

    return (
      <View style={styles.container}>
        <MedicalBackground />

        {/* User Header for Results Screen */}
        <View style={styles.userHeader}>
          <View style={styles.userInfo}>
            <Text style={styles.welcomeText}>Welcome, {user?.username || 'User'}</Text>
            {user?.full_name && <Text style={styles.fullNameText}>{user.full_name}</Text>}
          </View>
          <Pressable
            style={[styles.logoutButton, isLoading && styles.buttonDisabled]}
            onPress={handleLogout}
            disabled={isLoading}
            android_ripple={{ color: 'rgba(255,255,255,0.3)' }}
          >
            <Text style={styles.logoutButtonText}>
              {isLoading ? "🔄 Logging out..." : "🚪 Logout"}
            </Text>
          </Pressable>
        </View>

        <ScrollView
          style={styles.scrollContainer}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={true}
        >
          {imageUri && (
            <Image
              source={{uri: imageUri}}
              style={[styles.imageSmall, { height: imageHeight }]}
              resizeMode="contain"
            />
          )}

          {analysisResult && !analysisResult.isLesion && (
            <View style={styles.nonLesionCard}>
              <Text style={styles.resultTitle}>Analysis Complete</Text>
              <Text style={styles.nonLesionText}>
                This image appears to be a non-lesion image.
              </Text>
              {analysisResult.binaryData?.lesion_probability !== undefined && (
                <Text style={styles.probabilityText}>
                  Lesion confidence: {(analysisResult.binaryData.lesion_probability * 100).toFixed(1)}%
                  {analysisResult.binaryData.lesion_probability > 0.4 && analysisResult.binaryData.lesion_probability <= 0.5 &&
                    " (borderline - consider detailed analysis)"
                  }
                </Text>
              )}

              {!analysisResult.formattedResult && (
                <Animated.View style={{ transform: [{ scale: isRunningDetailedAnalysis ? pulseAnim : 1 }] }}>
                  <Pressable
                    style={[styles.button, styles.secondaryButton, isRunningDetailedAnalysis && styles.buttonDisabled]}
                    onPress={runDetailedAnalysisOnNonLesion}
                    disabled={isRunningDetailedAnalysis}
                  >
                    <Text style={styles.buttonText}>
                      {isRunningDetailedAnalysis ? "🧬 Analyzing..." : "🔬 Run Detailed Analysis Anyway"}
                    </Text>
                    {isRunningDetailedAnalysis && <MedicalLoadingSpinner size={20} color="white" />}
                  </Pressable>
                </Animated.View>
              )}
            </View>
          )}

          {analysisResult && analysisResult.isLesion && (
            <View style={styles.lesionCard}>
              <Text style={styles.resultTitle}>Lesion Detected</Text>
              <Text style={styles.lesionText}>
                Lesion detected - detailed analysis completed automatically.
              </Text>
            </View>
          )}

          {/* Display Mode Toggle - Compact version in results area */}
          {professionalData && (
            <View style={styles.displayModeToggleContainer}>
              <DisplayModeToggle compact={true} />
            </View>
          )}

          {/* Calibrated Uncertainty Display - Clinical-grade risk categories */}
          {/* Shows meaningful concern levels instead of misleading confidence percentages */}
          {/* Always shown in simple mode, with more details in professional mode */}
          {professionalData?.calibratedUncertainty && (
            <View style={styles.calibratedResultsCard}>
              <CalibratedResultsDisplay
                calibratedUncertainty={professionalData.calibratedUncertainty}
                predictedClass={professionalData.diagnosis}
                showDetails={userSettings.displayMode === 'professional'}
              />
            </View>
          )}

          {/* AI Explanation Section - Learn more about the diagnosed condition */}
          {(professionalData?.diagnosis || analysisResult?.fullResult?.predicted_class) && (
            <View style={styles.aiExplanationContainer}>
              <Pressable
                style={[styles.learnMoreButton, showAiExplanation && styles.learnMoreButtonActive]}
                onPress={() => {
                  if (showAiExplanation) {
                    setShowAiExplanation(false);
                  } else {
                    const condition = professionalData?.diagnosis || analysisResult?.fullResult?.predicted_class || 'Unknown';
                    const severity = professionalData?.riskLevel || analysisResult?.fullResult?.risk_level;
                    fetchAIExplanation(condition, severity);
                  }
                }}
              >
                <Ionicons
                  name={showAiExplanation ? "chevron-up-circle" : "information-circle"}
                  size={20}
                  color="#667eea"
                  style={{ marginRight: 8 }}
                />
                <Text style={styles.learnMoreButtonText}>
                  {showAiExplanation ? 'Hide Explanation' : 'Learn More About This Condition'}
                </Text>
                {isLoadingAiExplanation && (
                  <ActivityIndicator size="small" color="#667eea" style={{ marginLeft: 8 }} />
                )}
              </Pressable>

              {showAiExplanation && (
                <View style={styles.aiExplanationContent}>
                  {isLoadingAiExplanation ? (
                    <View style={styles.aiExplanationLoading}>
                      <ActivityIndicator size="large" color="#667eea" />
                      <Text style={styles.aiExplanationLoadingText}>
                        Getting AI explanation...
                      </Text>
                    </View>
                  ) : aiExplanationError ? (
                    <View style={styles.aiExplanationError}>
                      <Ionicons name="alert-circle" size={24} color="#e53e3e" />
                      <Text style={styles.aiExplanationErrorText}>{aiExplanationError}</Text>
                      <Pressable
                        style={styles.retryButton}
                        onPress={() => {
                          const condition = professionalData?.diagnosis || analysisResult?.fullResult?.predicted_class;
                          const severity = professionalData?.riskLevel || analysisResult?.fullResult?.risk_level;
                          fetchAIExplanation(condition, severity);
                        }}
                      >
                        <Text style={styles.retryButtonText}>Try Again</Text>
                      </Pressable>
                    </View>
                  ) : aiExplanation ? (
                    <View style={styles.aiExplanationText}>
                      <View style={styles.aiExplanationHeader}>
                        <Ionicons name="sparkles" size={18} color="#667eea" />
                        <Text style={styles.aiExplanationTitle}>
                          About {professionalData?.diagnosis || analysisResult?.fullResult?.predicted_class}
                        </Text>
                      </View>
                      <Text style={styles.aiExplanationBody}>{aiExplanation}</Text>
                      <View style={styles.aiExplanationDisclaimer}>
                        <Ionicons name="information-circle-outline" size={14} color="#718096" />
                        <Text style={styles.aiExplanationDisclaimerText}>
                          This AI explanation is for educational purposes only. Please consult a healthcare provider for medical advice.
                        </Text>
                      </View>
                    </View>
                  ) : null}
                </View>
              )}
            </View>
          )}

          {/* Professional Results Display - Only show in professional mode */}
          {userSettings.displayMode === 'professional' && professionalData && (
            <ProfessionalResultsDisplay
              analysisData={professionalData}
              burnData={professionalData.burnData}
              dermoscopyData={professionalData.dermoscopyData}
            />
          )}

          {/* ABCDE Feature Analysis - Only show in professional mode */}
          {/* Quantitative explainability for dermatologists */}
          {/* Shows asymmetry scores, border irregularity, color variance, diameter measurements */}
          {userSettings.displayMode === 'professional' && professionalData?.abcdeAnalysis && (
            <View style={styles.abcdeAnalysisCard}>
              <ABCDEFeatureDisplay
                analysis={professionalData.abcdeAnalysis}
                showMethodology={true}
              />
            </View>
          )}

          {analysisResult && analysisResult.formattedResult && !professionalData && (
            <View style={styles.resultsTextContainer}>
              <Text style={styles.preds}>{analysisResult.formattedResult}</Text>
            </View>
          )}

          {/* Multimodal Analysis Indicator */}
          {(professionalData?.multimodalAnalysis?.enabled || analysisResult?.fullResult?.multimodal_analysis?.enabled) && (
            <View style={styles.multimodalCard}>
              <View style={styles.multimodalHeader}>
                <Text style={styles.multimodalTitle}>Multimodal Analysis</Text>
                <View style={styles.multimodalBadge}>
                  <Text style={styles.multimodalBadgeText}>ENHANCED</Text>
                </View>
              </View>

              {/* Data Sources Used */}
              <View style={styles.multimodalSection}>
                <Text style={styles.multimodalSectionTitle}>Data Sources Combined:</Text>
                <View style={styles.dataSourcesContainer}>
                  {(professionalData?.multimodalAnalysis?.data_sources || analysisResult?.fullResult?.multimodal_analysis?.data_sources || []).map((source: string, index: number) => (
                    <View key={index} style={styles.dataSourceChip}>
                      <Text style={styles.dataSourceIcon}>
                        {source === 'image' ? '📷' : source === 'clinical_history' ? '📋' : source === 'labs' ? '🧪' : source === 'lesion_tracking' ? '📊' : '✓'}
                      </Text>
                      <Text style={styles.dataSourceText}>
                        {source === 'image' ? 'Image' : source === 'clinical_history' ? 'History' : source === 'labs' ? 'Labs' : source === 'lesion_tracking' ? 'Tracking' : source}
                      </Text>
                    </View>
                  ))}
                </View>
              </View>

              {/* Confidence Breakdown */}
              {(professionalData?.multimodalAnalysis?.confidence_breakdown || analysisResult?.fullResult?.multimodal_analysis?.confidence_breakdown) && (
                <View style={styles.multimodalSection}>
                  <Text style={styles.multimodalSectionTitle}>Confidence Breakdown:</Text>
                  {(() => {
                    const breakdown = professionalData?.multimodalAnalysis?.confidence_breakdown || analysisResult?.fullResult?.multimodal_analysis?.confidence_breakdown;
                    return (
                      <View style={styles.confidenceBreakdown}>
                        {breakdown?.image_model !== undefined && (
                          <View style={styles.confidenceRow}>
                            <Text style={styles.confidenceLabel}>Image Model:</Text>
                            <Text style={styles.confidenceValue}>{(breakdown.image_model * 100).toFixed(1)}%</Text>
                          </View>
                        )}
                        {breakdown?.clinical_adjustment !== undefined && breakdown.clinical_adjustment !== 0 && (
                          <View style={styles.confidenceRow}>
                            <Text style={styles.confidenceLabel}>Clinical Adjustment:</Text>
                            <Text style={[styles.confidenceValue, { color: breakdown.clinical_adjustment > 0 ? '#38a169' : '#e53e3e' }]}>
                              {breakdown.clinical_adjustment > 0 ? '+' : ''}{(breakdown.clinical_adjustment * 100).toFixed(1)}%
                            </Text>
                          </View>
                        )}
                        {breakdown?.lab_adjustment !== undefined && breakdown.lab_adjustment !== 0 && (
                          <View style={styles.confidenceRow}>
                            <Text style={styles.confidenceLabel}>Lab Adjustment:</Text>
                            <Text style={[styles.confidenceValue, { color: breakdown.lab_adjustment > 0 ? '#38a169' : '#e53e3e' }]}>
                              {breakdown.lab_adjustment > 0 ? '+' : ''}{(breakdown.lab_adjustment * 100).toFixed(1)}%
                            </Text>
                          </View>
                        )}
                        {breakdown?.total !== undefined && (
                          <View style={[styles.confidenceRow, styles.confidenceTotalRow]}>
                            <Text style={styles.confidenceTotalLabel}>Final Confidence:</Text>
                            <Text style={styles.confidenceTotalValue}>{(breakdown.total * 100).toFixed(1)}%</Text>
                          </View>
                        )}
                      </View>
                    );
                  })()}
                </View>
              )}

              {/* Clinical Factors */}
              <View style={styles.multimodalSection}>
                <Text style={styles.multimodalSectionTitle}>Clinical Factors:</Text>
                {(professionalData?.multimodalAnalysis?.clinical_adjustments?.factors?.length > 0 ||
                  analysisResult?.fullResult?.multimodal_analysis?.clinical_adjustments?.factors?.length > 0) ? (
                  (professionalData?.multimodalAnalysis?.clinical_adjustments?.factors ||
                    analysisResult?.fullResult?.multimodal_analysis?.clinical_adjustments?.factors || []).slice(0, 3).map((factor: any, index: number) => (
                    <View key={index} style={styles.factorRow}>
                      <Text style={styles.factorName}>{factor.factor?.replace(/_/g, ' ')}</Text>
                      <Text style={styles.factorMultiplier}>x{factor.multiplier?.toFixed(1)}</Text>
                    </View>
                  ))
                ) : (
                  <View style={styles.noFactorsContainer}>
                    <Text style={styles.noFactorsText}>No clinical factors applied</Text>
                    <Text style={styles.noFactorsHint}>
                      Add your skin type, age, and medical history in Profile to enable personalized risk adjustments
                    </Text>
                  </View>
                )}
              </View>
            </View>
          )}

          {/* Treatment Outcome Simulator Button */}
          {(professionalData || analysisResult?.formattedResult) && (
            <View style={styles.exportButtonContainer}>
              <Pressable
                style={[styles.button, styles.treatmentSimulatorButton]}
                onPress={() => {
                  console.log('[HOME] professionalData:', professionalData);
                  console.log('[HOME] analysisResult:', analysisResult);
                  console.log('[HOME] analysisResult.fullResult:', analysisResult?.fullResult);
                  console.log('[HOME] top_prediction:', analysisResult?.fullResult?.top_prediction);
                  console.log('[HOME] formattedResult type:', typeof analysisResult?.formattedResult);

                  // Try multiple ways to get the diagnosis
                  let diagnosis = 'Unknown';

                  if (professionalData?.diagnosis) {
                    diagnosis = professionalData.diagnosis;
                  } else if (analysisResult?.fullResult?.top_prediction) {
                    diagnosis = analysisResult.fullResult.top_prediction;
                  } else if (analysisResult?.fullResult?.predicted_class) {
                    diagnosis = analysisResult.fullResult.predicted_class;
                  } else if (typeof analysisResult?.formattedResult === 'string') {
                    const firstLine = analysisResult.formattedResult.split('\n')[0];
                    diagnosis = firstLine.replace('Prediction: ', '').trim();
                  } else if (analysisResult?.formattedResult?.predicted_class) {
                    diagnosis = analysisResult.formattedResult.predicted_class;
                  }

                  console.log('[HOME] Final diagnosis:', diagnosis);
                  router.push(`/ar-treatment-simulator?diagnosis=${encodeURIComponent(diagnosis)}&imageUrl=${encodeURIComponent(imageUri || '')}`);
                }}
              >
                <Ionicons name="images-outline" size={20} color="white" style={{ marginRight: 8 }} />
                <Text style={styles.buttonText}>
                  Treatment Outcome Simulator
                </Text>
              </Pressable>

              {/* AI Chat Button */}
              <Pressable
                style={[styles.button, styles.aiChatButton]}
                onPress={() => {
                  let diagnosis = 'Unknown';
                  let confidence = '';
                  let analysisIdParam = '';

                  if (professionalData?.diagnosis) {
                    diagnosis = professionalData.diagnosis;
                  } else if (analysisResult?.fullResult?.top_prediction) {
                    diagnosis = analysisResult.fullResult.top_prediction;
                  } else if (analysisResult?.fullResult?.predicted_class) {
                    diagnosis = analysisResult.fullResult.predicted_class;
                  }

                  if (professionalData?.confidence) {
                    confidence = professionalData.confidence.toString();
                  } else if (analysisResult?.fullResult?.confidence) {
                    confidence = (analysisResult.fullResult.confidence * 100).toFixed(0);
                  }

                  if (analysisResult?.fullResult?.analysis_id) {
                    analysisIdParam = `&analysisId=${analysisResult.fullResult.analysis_id}`;
                  }

                  router.push(`/ai-chat?diagnosis=${encodeURIComponent(diagnosis)}&confidence=${encodeURIComponent(confidence)}${analysisIdParam}` as any);
                }}
              >
                <Ionicons name="chatbubbles-outline" size={20} color="white" style={{ marginRight: 8 }} />
                <Text style={styles.buttonText}>
                  Ask AI About This
                </Text>
              </Pressable>

              {/* Show Reasoning Button */}
              <Pressable
                style={[styles.button, styles.reasoningButton]}
                onPress={() => {
                  if (showReasoning) {
                    setShowReasoning(false);
                  } else {
                    fetchDifferentialReasoning();
                  }
                }}
              >
                <Ionicons
                  name={showReasoning ? "chevron-up-circle" : "bulb-outline"}
                  size={20}
                  color="white"
                  style={{ marginRight: 8 }}
                />
                <Text style={styles.buttonText}>
                  {showReasoning ? 'Hide Reasoning' : 'Show Diagnostic Reasoning'}
                </Text>
                {isLoadingReasoning && (
                  <ActivityIndicator size="small" color="white" style={{ marginLeft: 8 }} />
                )}
              </Pressable>
            </View>
          )}

          {/* Differential Reasoning Display */}
          {showReasoning && (
            <View style={styles.reasoningContainer}>
              <View style={styles.reasoningHeader}>
                <Ionicons name="bulb" size={22} color="#8b5cf6" />
                <Text style={styles.reasoningTitle}>Diagnostic Reasoning</Text>
              </View>

              {isLoadingReasoning ? (
                <View style={styles.reasoningLoading}>
                  <ActivityIndicator size="large" color="#8b5cf6" />
                  <Text style={styles.reasoningLoadingText}>
                    Generating chain-of-thought explanation...
                  </Text>
                </View>
              ) : reasoningError ? (
                <View style={styles.reasoningError}>
                  <Ionicons name="alert-circle" size={24} color="#e53e3e" />
                  <Text style={styles.reasoningErrorText}>{reasoningError}</Text>
                  <Pressable
                    style={styles.reasoningRetryButton}
                    onPress={fetchDifferentialReasoning}
                  >
                    <Text style={styles.reasoningRetryText}>Try Again</Text>
                  </Pressable>
                </View>
              ) : differentialReasoning ? (
                <View style={styles.reasoningContent}>
                  <Text style={styles.reasoningBody}>{differentialReasoning}</Text>
                  <View style={styles.reasoningDisclaimer}>
                    <Ionicons name="information-circle-outline" size={14} color="#718096" />
                    <Text style={styles.reasoningDisclaimerText}>
                      This AI-generated reasoning is for educational purposes only. It does not constitute medical advice.
                    </Text>
                  </View>
                </View>
              ) : null}
            </View>
          )}

          {isRunningDetailedAnalysis && progressMessage && (
            <Text style={styles.progressText}>
              {progressMessage}
            </Text>
          )}

          {/* Spacer to ensure button is always visible */}
          <View style={styles.spacer} />
        </ScrollView>

        {/* Fixed button at bottom */}
        <View style={styles.fixedButtonContainer}>
          <Pressable
            style={[styles.button, styles.primaryButton, styles.fixedButton]}
            onPress={startOver}
          >
            <Text style={styles.buttonText}>Analyze New Image</Text>
          </Pressable>
        </View>
      </View>
    );
  };

  // Main Upload Screen Component
  const UploadScreen = () => (
    <View style={styles.container}>
      <MedicalBackground />

      {/* Show quality check as full screen when active */}
      {showQualityCheck && imageQuality ? (
        <ImageQualityDisplay qualityData={imageQuality} />
      ) : (
        /* Normal upload content */
        <ScrollView
          style={styles.contentOverlay}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={true}
        >
          {/* User Header */}
          <View style={styles.userHeader}>
            <View style={styles.userInfo}>
              <Text style={styles.welcomeText}>Welcome, {user?.username || 'User'}</Text>
              {user?.full_name && <Text style={styles.fullNameText}>{user.full_name}</Text>}
            </View>
            <Pressable
              style={styles.menuButton}
              onPress={() => setShowMenu(true)}
              android_ripple={{ color: 'rgba(255,255,255,0.3)' }}
            >
              <Text style={styles.menuButtonText}>☰</Text>
            </Pressable>
          </View>

          <View style={styles.headerSection}>
            <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
              <Text style={styles.appTitle}>{t('homeScreen.title')}</Text>
              <HelpTooltip
                title={t('homeScreen.title')}
                content={t('home.comprehensiveAnalysis')}
                size={24}
                color="#2c5282"
              />
            </View>
            <Text style={styles.appSubtitle}>{t('homeScreen.subtitle')}</Text>
          </View>

          {/* Alerts Banner */}
          <AlertsBanner />

          {imageUri && (
            <MemoizedImage uri={imageUri} style={styles.image} resizeMode="contain" />
          )}

          {!imageUri && (
            <View style={styles.placeholderContainer}>
              <View style={styles.uploadIcon}>
                <Text style={styles.uploadIconText}>📸</Text>
              </View>
              <Text style={styles.placeholderText}>{t('homeScreen.selectImage')}</Text>
              <PhotoTipsComponent />
            </View>
          )}

          {/* Info about comprehensive analysis */}
          <View style={styles.analysisInfoContainer}>
            <Text style={styles.analysisInfoText} numberOfLines={3}>
              📋 {t('home.comprehensiveAnalysis')}
            </Text>
          </View>

          <View style={styles.buttonSection}>
            <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center', width: '100%' }}>
              <Pressable
                style={[styles.button, styles.uploadButton, isLoading && styles.buttonDisabled]}
                onPress={uploadPhotoPress}
                disabled={isLoading || isClassifying}
              >
                <Text style={styles.buttonText}>
                  {isLoading ? t('common.loading') : `📷 ${t('homeScreen.uploadPhoto')}`}
                </Text>
                {isLoading && <MedicalLoadingSpinner size={20} color="white" />}
              </Pressable>
              <HelpTooltip
                title={t('homeScreen.uploadPhoto')}
                content={t('homeScreen.userGuide.photoTipsGuide.title')}
                size={20}
                color="#4299e1"
              />
            </View>

            <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center', width: '100%' }}>
              <Animated.View style={{ transform: [{ scale: isClassifying ? pulseAnim : 1 }], flex: 1 }}>
                <Pressable
                  style={[styles.button, styles.analyzeButton, (isClassifying || !imageUri) && styles.buttonDisabled]}
                  onPress={classifyPhoto}
                  disabled={isLoading || isClassifying || !imageUri}
                >
                  <Text style={styles.buttonText}>
                    {isClassifying ? `🧬 ${t('home.analyzing')}... ${Math.round(progressPercentage)}%` : `🔍 ${t('homeScreen.runAnalysis')}`}
                  </Text>
                  {isClassifying && <MedicalLoadingSpinner size={20} color="white" />}
                </Pressable>
              </Animated.View>
              <HelpTooltip
                title={t('homeScreen.runAnalysis')}
                content={t('homeScreen.userGuide.howToUse.step2Content')}
                size={20}
                color="#38a169"
              />
            </View>
          </View>

          {isClassifying && (
            <View style={styles.analysisContainer}>
              <AnalysisProgressBar progress={progressPercentage / 100} />
              <View style={styles.progressContainer}>
                <Text style={styles.progressText}>
                  {progressMessage || "Initializing analysis..."}
                </Text>
                <Text style={styles.progressSubtext}>
                  This may take up to 2 minutes
                </Text>
              </View>
            </View>
          )}

          {!showResults && modelPreds !== "Waiting..." && (
            <Text style={styles.preds}>{modelPreds}</Text>
          )}
        </ScrollView>
      )}
    </View>
  );

  return (
    <>
      {showResults ? <ResultsScreen /> : <UploadScreen />}

      {/* Body Map Selector Modal */}
      <Modal
        visible={showBodyMapSelector}
        animationType="slide"
        transparent={false}
        onRequestClose={() => setShowBodyMapSelector(false)}
      >
        <View style={styles.modalContainer}>
          <LinearGradient
            colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.modalBackground}
          />
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Mark Lesion Location</Text>
            <Pressable
              style={styles.modalCloseButton}
              onPress={() => setShowBodyMapSelector(false)}
            >
              <Text style={styles.modalCloseButtonText}>✕</Text>
            </Pressable>
          </View>

          <ScrollView style={styles.modalScrollView}>
            <BodyMapSelector
              onLocationSelect={handleLocationSelect}
              selectedLocation={bodyMapData}
            />
          </ScrollView>

          <View style={styles.modalFooter}>
            <Pressable
              style={[styles.button, styles.skipButton]}
              onPress={() => {
                setBodyMapData(null);
                proceedToClassify();
              }}
            >
              <Text style={styles.skipButtonText}>Skip</Text>
            </Pressable>
            <Pressable
              style={[styles.button, styles.proceedButton, !bodyMapData && styles.buttonDisabled]}
              onPress={proceedToClassify}
              disabled={!bodyMapData}
            >
              <Text style={styles.proceedButtonText}>
                {bodyMapData ? '✓ Proceed to Analyze' : 'Select Location First'}
              </Text>
            </Pressable>
          </View>
        </View>
      </Modal>

      {/* Clinical Context Form Modal */}
      <ClinicalContextForm
        visible={showClinicalContext}
        onClose={skipClinicalContext}
        onSubmit={handleClinicalContextSubmit}
        initialContext={clinicalContext || {}}
      />

      {/* Navigation Menu Modal */}
      <Modal
        visible={showMenu}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowMenu(false)}
      >
        <Pressable
          style={styles.menuOverlay}
          onPress={() => setShowMenu(false)}
        >
          <View style={styles.menuContainer}>
            <View style={styles.menuHeader}>
              <Text style={styles.menuTitle}>{t('homeScreen.menu.title')}</Text>
              <Pressable onPress={() => setShowMenu(false)}>
                <Text style={styles.menuCloseButton}>✕</Text>
              </Pressable>
            </View>

            <ScrollView style={styles.menuContent}>
              {/* 1. Patient Monitoring */}
              <MenuCategory title="Patient Monitoring" icon="🔍" count={6}>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/lesion-tracking'); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🔍</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('homeScreen.navigation.lesionTracking')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('history.menu.trackLesions.subtitle')}</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/progression-timeline'); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>📊</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('progressionTimeline.title')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('progressionTimeline.subtitle')}</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/body-map-3d' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🧍</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>3D Body Map</Text>
                    <Text style={styles.menuItemSubtext}>Visual body mapping with AR integration</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/sun-exposure' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>☀️</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('homeScreen.navigation.sunExposure')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('history.menu.sunExposure.subtitle')}</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/wearables' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>⌚</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Wearable Devices</Text>
                    <Text style={styles.menuItemSubtext}>Track UV exposure from smartwatch</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/history'); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>📋</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('homeScreen.navigation.history')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('homeScreen.userGuide.features.history')}</Text>
                  </View>
                </Pressable>
              </MenuCategory>

              {/* 2. Health Profile */}
              <MenuCategory title="Health Profile" icon="👤" count={4}>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/profile' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>👤</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Profile</Text>
                    <Text style={styles.menuItemSubtext}>Age, skin type, medical history</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/family-history' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🧬</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('homeScreen.navigation.familyHistory')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('history.menu.familyHistory.subtitle')}</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/genetic-testing' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🧪</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Genetic Testing</Text>
                    <Text style={styles.menuItemSubtext}>Upload VCF files & view genetic risk</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/lab-results' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🔬</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Lab Results</Text>
                    <Text style={styles.menuItemSubtext}>Blood, urine & stool analysis</Text>
                  </View>
                </Pressable>
              </MenuCategory>

              {/* 3. Consult & Diagnosis */}
              <MenuCategory title="Consult & Diagnosis" icon="👨‍⚕️" count={4}>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/advanced-telederm' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>📹</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Advanced Teledermatology</Text>
                    <Text style={styles.menuItemSubtext}>Video consults, triage & consensus</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/dermatologist-integration' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>👨‍⚕️</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('homeScreen.navigation.dermatologist')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('history.menu.dermatologist.subtitle')}</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/ai-chat' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🤖</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>AI Assistant</Text>
                    <Text style={styles.menuItemSubtext}>Ask questions about skin health</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/patient-communities' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>👥</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Patient Communities</Text>
                    <Text style={styles.menuItemSubtext}>Connect with others on Inspire & HealthUnlocked</Text>
                  </View>
                </Pressable>
              </MenuCategory>

              {/* 4. Staging & Prognosis */}
              <MenuCategory title="Staging & Prognosis" icon="📊" count={5}>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/biopsy' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🧫</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Biopsy Tracking</Text>
                    <Text style={styles.menuItemSubtext}>Track biopsies & analyze histopathology slides</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/ajcc-staging' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>📊</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>AJCC Staging</Text>
                    <Text style={styles.menuItemSubtext}>Interactive TNM melanoma staging calculator</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/breslow-clark' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🔬</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Breslow/Clark Visualizer</Text>
                    <Text style={styles.menuItemSubtext}>3D visualization of invasion depth</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/survival-estimator' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>📉</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Survival Estimator</Text>
                    <Text style={styles.menuItemSubtext}>ML-based survival curves from tumor data</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/sentinel-node-mapper' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🔗</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Sentinel Node Mapper</Text>
                    <Text style={styles.menuItemSubtext}>Lymph node basin mapping & biopsy tracking</Text>
                  </View>
                </Pressable>
              </MenuCategory>

              {/* 5. Treatment */}
              <MenuCategory title="Treatment" icon="💊" count={2}>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/treatment-monitoring' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>💊</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('homeScreen.navigation.treatment')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('history.menu.treatmentMonitoring.subtitle')}</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/clinical-trials' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🔬</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Clinical Trials</Text>
                    <Text style={styles.menuItemSubtext}>Find matching research studies</Text>
                  </View>
                </Pressable>
              </MenuCategory>

              {/* 6. Billing & Documentation */}
              <MenuCategory title="Billing & Documentation" icon="💳" count={6}>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/billing-coding'); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>💳</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('billingCoding.title')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('billingCoding.subtitle')}</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/billing-insurance'); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🏥</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Insurance & Appeals</Text>
                    <Text style={styles.menuItemSubtext}>Pre-auth, claims & appeal letters</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/cost-transparency'); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>💰</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Cost Transparency</Text>
                    <Text style={styles.menuItemSubtext}>Prices, provider comparison & Rx savings</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/auto-coding' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🏷️</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Auto-Coding Engine</Text>
                    <Text style={styles.menuItemSubtext}>ICD-10/CPT codes from AI diagnosis</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/publication-report' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>📄</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Publication Report</Text>
                    <Text style={styles.menuItemSubtext}>Generate publication-ready case reports</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/malpractice-shield' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🛡️</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Malpractice Shield</Text>
                    <Text style={styles.menuItemSubtext}>Liability analysis & insurance coverage</Text>
                  </View>
                </Pressable>
              </MenuCategory>

              {/* 7. Analytics & AI */}
              <MenuCategory title="Analytics & AI" icon="📈" count={5}>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/analytics' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>📈</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('homeScreen.navigation.analytics')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('history.menu.analytics.subtitle')}</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/population-health'); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>📊</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('populationHealth.title')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('populationHealth.subtitle')}</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/ai-accuracy' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🎯</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>AI Accuracy</Text>
                    <Text style={styles.menuItemSubtext}>Diagnostic accuracy improving over time</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/enhanced-ml' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🤖</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Enhanced ML</Text>
                    <Text style={styles.menuItemSubtext}>Segmentation, growth prediction & privacy</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/data-augmentation' as any); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>🔄</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>Data Augmentation</Text>
                    <Text style={styles.menuItemSubtext}>Generate synthetic training data</Text>
                  </View>
                </Pressable>
              </MenuCategory>

              {/* 8. Account */}
              <MenuCategory title="Account" icon="⚙️" count={2}>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/help'); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>📚</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('homeScreen.navigation.help')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('history.menu.help.subtitle')}</Text>
                  </View>
                </Pressable>
                <Pressable
                  style={styles.menuItemCompact}
                  onPress={() => { setShowMenu(false); router.push('/settings'); }}
                  disabled={isLoading || isClassifying}
                >
                  <Text style={styles.menuItemIconSmall}>⚙️</Text>
                  <View style={styles.menuItemTextContainer}>
                    <Text style={styles.menuItemText}>{t('homeScreen.navigation.settings')}</Text>
                    <Text style={styles.menuItemSubtext}>{t('settings.language')} and preferences</Text>
                  </View>
                </Pressable>
              </MenuCategory>

              <View style={styles.menuDivider} />

              <Pressable
                style={[styles.menuItem, styles.logoutMenuItem]}
                onPress={() => {
                  setShowMenu(false);
                  handleLogout();
                }}
                disabled={isLoading}
              >
                <Text style={styles.menuItemIcon}>🚪</Text>
                <View style={styles.menuItemTextContainer}>
                  <Text style={[styles.menuItemText, styles.logoutMenuText]}>
                    {isLoading ? t('common.loading') : t('homeScreen.navigation.logout')}
                  </Text>
                  <Text style={styles.menuItemSubtext}>{t('history.menu.logout.subtitle')}</Text>
                </View>
              </Pressable>
            </ScrollView>
          </View>
        </Pressable>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  backgroundContainer: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  overlayPattern: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    opacity: 0.03,
    backgroundColor: 'transparent',
  },
  medicalPattern: {
    flex: 1,
    backgroundColor: 'repeating-linear-gradient(45deg, transparent, transparent 35px, rgba(79, 172, 254, 0.1) 35px, rgba(79, 172, 254, 0.1) 70px)',
  },
  contentOverlay: {
    flex: 1,
    width: '100%',
  },
  scrollContent: {
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingBottom: 40,
    minHeight: '100%',
  },
  headerSection: {
    alignItems: 'center',
    marginBottom: 8,
    paddingTop: 20,
  },
  appTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 4,
    textAlign: 'center',
    paddingHorizontal: 16,
  },
  appSubtitle: {
    fontSize: 14,
    color: '#4a5568',
    textAlign: 'center',
    fontWeight: '300',
  },
  placeholderContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 6,
    padding: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#e2e8f0',
    borderStyle: 'dashed',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 3,
  },
  uploadIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#ebf8ff',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  uploadIconText: {
    fontSize: 32,
  },
  placeholderText: {
    fontSize: 16,
    color: '#4a5568',
    textAlign: 'center',
    fontWeight: '500',
  },
  buttonSection: {
    width: '100%',
    alignItems: 'center',
    marginTop: 15,
  },
  uploadButton: {
    backgroundColor: '#4299e1',
    shadowColor: '#4299e1',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  analyzeButton: {
    backgroundColor: '#38a169',
    shadowColor: '#38a169',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  historyButton: {
    backgroundColor: '#6f42c1',
    shadowColor: '#6f42c1',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  progressContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: 12,
    padding: 16,
    marginTop: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  progressSubtext: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 4,
    fontStyle: 'italic',
  },
  analysisContainer: {
    width: '100%',
    alignItems: 'center',
    marginTop: 20,
  },
  loadingSpinnerContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 8,
  },
  loadingSpinner: {
    borderWidth: 3,
    borderTopColor: 'transparent',
    borderRightColor: 'transparent',
    borderBottomColor: 'currentColor',
    borderLeftColor: 'currentColor',
    borderRadius: 50,
  },
  loadingSpinnerCenter: {
    position: 'absolute',
    borderRadius: 50,
    opacity: 0.3,
  },
  progressBarContainer: {
    width: '90%',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: 12,
    padding: 20,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  progressTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    textAlign: 'center',
    marginBottom: 15,
  },
  progressBar: {
    height: 8,
    backgroundColor: '#e2e8f0',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 15,
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: '#4299e1',
    borderRadius: 4,
  },
  progressSteps: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  progressStep: {
    alignItems: 'center',
    flex: 1,
  },
  progressStepDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#e2e8f0',
    marginBottom: 5,
  },
  progressStepActive: {
    backgroundColor: '#4299e1',
  },
  progressStepText: {
    fontSize: 10,
    color: '#666',
    textAlign: 'center',
  },
  progressStepTextActive: {
    color: '#2c5282',
    fontWeight: 'bold',
  },
  // Professional Results Styling
  professionalResultsContainer: {
    width: '100%',
    paddingHorizontal: 15,
  },
  primaryDiagnosisCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 6,
    borderLeftWidth: 5,
    borderLeftColor: '#4299e1',
  },
  diagnosisHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
    paddingBottom: 10,
  },
  diagnosisTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  diagnosisTimestamp: {
    fontSize: 12,
    color: '#666',
    fontStyle: 'italic',
  },
  primaryResult: {
    alignItems: 'center',
  },
  predictedClass: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1a202c',
    textAlign: 'center',
    marginBottom: 15,
  },
  confidenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  confidenceLabel: {
    fontSize: 16,
    color: '#4a5568',
    marginRight: 8,
  },
  confidenceValue: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  confidenceDescription: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    fontStyle: 'italic',
  },
  riskAssessmentCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 6,
    borderLeftWidth: 5,
  },
  riskHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  riskTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  riskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  riskLevel: {
    color: '#ffffff',
    fontWeight: 'bold',
    fontSize: 12,
  },
  riskRecommendation: {
    fontSize: 16,
    color: '#4a5568',
    lineHeight: 22,
  },
  probabilitiesCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 6,
  },
  probabilitiesTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
    paddingBottom: 10,
  },
  probabilityRow: {
    marginBottom: 15,
  },
  probabilityInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  probabilityLabel: {
    fontSize: 14,
    color: '#4a5568',
    flex: 1,
  },
  topProbabilityLabel: {
    fontWeight: 'bold',
    color: '#1a202c',
    fontSize: 16,
  },
  probabilityValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2d3748',
  },
  topProbabilityValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1a202c',
  },
  probabilityBarContainer: {
    height: 8,
    backgroundColor: '#f7fafc',
    borderRadius: 4,
    overflow: 'hidden',
  },
  probabilityBar: {
    height: '100%',
    borderRadius: 4,
  },
  metadataCard: {
    backgroundColor: '#f8f9fa',
    borderRadius: 16,
    padding: 20,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  metadataTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 15,
  },
  metadataRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  metadataLabel: {
    fontSize: 14,
    color: '#666',
    flex: 1,
  },
  metadataValue: {
    fontSize: 14,
    color: '#2d3748',
    fontWeight: '500',
    flex: 2,
    textAlign: 'right',
  },
  disclaimerCard: {
    backgroundColor: '#fef5e7',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#f6e05e',
    borderLeftWidth: 5,
    borderLeftColor: '#d69e2e',
  },
  disclaimerTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#744210',
    marginBottom: 10,
  },
  disclaimerText: {
    fontSize: 14,
    color: '#744210',
    lineHeight: 20,
    textAlign: 'justify',
  },
  // Image Quality Validation Styles
  fullScreenOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: '#f8fafb',
    zIndex: 1000,
  },
  qualityFullScreenContainer: {
    flex: 1,
    width: '100%',
  },
  qualityScrollView: {
    flex: 1,
    width: '100%',
  },
  qualityScrollContent: {
    paddingBottom: 120, // Extra space for fixed buttons
    paddingTop: 20,
  },
  qualityImageContainer: {
    alignItems: 'center',
    marginBottom: 20,
    paddingHorizontal: 15,
  },
  qualityImage: {
    width: '100%',
    maxWidth: 300,
    height: 250,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#e2e8f0',
  },
  qualityContainer: {
    width: '100%',
    paddingHorizontal: 15,
    marginVertical: 10,
  },
  qualityCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 6,
    borderLeftWidth: 5,
  },
  qualityPassedCard: {
    borderLeftColor: '#22c55e',
  },
  qualityFailedCard: {
    borderLeftColor: '#ef4444',
  },
  qualityHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  qualityTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  qualityBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  qualityBadgeText: {
    color: '#ffffff',
    fontWeight: 'bold',
    fontSize: 12,
  },
  qualityScore: {
    alignItems: 'center',
    marginBottom: 15,
  },
  qualityScoreText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2d3748',
  },
  qualityMetadata: {
    backgroundColor: '#f7fafc',
    borderRadius: 8,
    padding: 10,
    marginBottom: 15,
  },
  qualityMetadataText: {
    fontSize: 14,
    color: '#4a5568',
    textAlign: 'center',
  },
  qualityIssues: {
    marginBottom: 15,
  },
  qualityIssuesTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 10,
  },
  qualityIssueItem: {
    borderLeftWidth: 3,
    paddingLeft: 12,
    marginBottom: 10,
    paddingVertical: 5,
  },
  qualityIssueMessage: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 2,
  },
  qualityIssueDescription: {
    fontSize: 13,
    color: '#666',
    fontStyle: 'italic',
  },
  qualityRecommendations: {
    backgroundColor: '#f0f9ff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 15,
  },
  qualityRecommendationsTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#1e40af',
    marginBottom: 8,
  },
  qualityRecommendationItem: {
    fontSize: 14,
    color: '#1e40af',
    marginBottom: 4,
    lineHeight: 18,
  },
  qualityActions: {
    flexDirection: 'column',
    gap: 10,
  },
  retryButton: {
    backgroundColor: '#3b82f6',
  },
  continueAnywayButton: {
    backgroundColor: '#f59e0b',
  },
  qualityPassedText: {
    fontSize: 16,
    color: '#22c55e',
    textAlign: 'center',
    fontWeight: '600',
  },
  qualitySuccessSection: {
    backgroundColor: '#f0fdf4',
    borderRadius: 8,
    padding: 15,
    marginTop: 10,
  },
  qualityButtonSpacer: {
    height: 20,
  },
  qualityFixedButtons: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#ffffff',
    paddingTop: 15,
    paddingBottom: 30,
    paddingHorizontal: 20,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: -2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
    gap: 10,
  },
  qualityFixedButton: {
    width: '100%',
    marginBottom: 0,
  },
  // Photo Tips Styles
  photoTipsContainer: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 15,
    marginTop: 15,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  photoTipsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
    textAlign: 'center',
  },
  photoTipsList: {
    gap: 8,
  },
  photoTip: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  photoTipEmoji: {
    fontSize: 16,
    marginRight: 8,
    marginTop: 2,
  },
  photoTipText: {
    fontSize: 14,
    color: '#4a5568',
    flex: 1,
    lineHeight: 18,
  },
  scrollContainer: {
    flex: 1,
    width: '100%',
  },
  scrollContent: {
    alignItems: 'center',
    paddingTop: 20,
    paddingBottom: 100, // Extra space for fixed button
  },
  button: {
    backgroundColor: "#8d8f92ff",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 44, // Better touch target
  },
  buttonText: {
    color: "white",
    fontSize: 18,
  },
  image: {
    width: 280,
    height: 280,
    marginTop: 8,
    marginBottom: 8,
  },
  imageSmall: {
    width: '90%',
    maxWidth: 320,
    marginBottom: 20,
  },
  preds: {
    fontSize: 16,
    textAlign: 'center',
    paddingHorizontal: 20,
    lineHeight: 22,
  },
  resultsTextContainer: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 20,
    marginHorizontal: 20,
    marginVertical: 10,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  progressText: {
    fontSize: 14,
    color: '#666',
    marginTop: 8,
    textAlign: 'center',
    paddingHorizontal: 20,
  },
  progressPercentageText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#3b82f6',
    marginTop: 4,
    textAlign: 'center',
  },
  resultsContainer: {
    flex: 1,
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  nonLesionCard: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 20,
    marginHorizontal: 20,
    marginVertical: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#ffc107',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  lesionCard: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 20,
    marginHorizontal: 20,
    marginVertical: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#dc3545',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  calibratedResultsCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    marginHorizontal: 16,
    marginVertical: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 3,
    },
    shadowOpacity: 0.12,
    shadowRadius: 6,
    elevation: 6,
    overflow: 'hidden',
  },
  displayModeToggleContainer: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    paddingHorizontal: 16,
    paddingTop: 8,
    marginBottom: -4,
  },
  abcdeAnalysisCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    marginHorizontal: 16,
    marginVertical: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 3,
    },
    shadowOpacity: 0.12,
    shadowRadius: 6,
    elevation: 6,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
    color: '#333',
  },
  nonLesionText: {
    fontSize: 16,
    color: '#856404',
    textAlign: 'center',
    marginBottom: 10,
  },
  probabilityText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 15,
    fontStyle: 'italic',
  },
  lesionText: {
    fontSize: 16,
    color: '#721c24',
    textAlign: 'center',
    marginBottom: 10,
  },
  confidenceText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 15,
  },
  buttonContainer: {
    width: '100%',
    alignItems: 'center',
    marginTop: 20,
  },
  fixedButtonContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#fff',
    paddingTop: 10,
    paddingBottom: 30,
    paddingHorizontal: 20,
    borderTopWidth: 1,
    borderTopColor: '#e9ecef',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: -2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  fixedButton: {
    width: '100%',
    marginBottom: 0,
  },
  spacer: {
    height: 20,
  },
  primaryButton: {
    backgroundColor: '#007bff',
  },
  secondaryButton: {
    backgroundColor: '#6c757d',
    marginTop: 10,
  },
  exportButtonContainer: {
    width: '100%',
    alignItems: 'center',
    marginTop: 15,
  },
  exportButton: {
    backgroundColor: '#17a2b8',
    shadowColor: '#17a2b8',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  treatmentSimulatorButton: {
    backgroundColor: '#667eea',
    shadowColor: '#667eea',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  aiChatButton: {
    backgroundColor: '#0ea5e9',
    shadowColor: '#0ea5e9',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
  },

  // AI Explanation Styles
  aiExplanationContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 16,
    marginHorizontal: 20,
    marginVertical: 10,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  learnMoreButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: 20,
    backgroundColor: 'rgba(102, 126, 234, 0.08)',
    borderBottomWidth: 0,
  },
  learnMoreButtonActive: {
    backgroundColor: 'rgba(102, 126, 234, 0.15)',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(102, 126, 234, 0.2)',
  },
  learnMoreButtonText: {
    color: '#667eea',
    fontSize: 15,
    fontWeight: '600',
  },
  aiExplanationContent: {
    padding: 16,
  },
  aiExplanationLoading: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 30,
  },
  aiExplanationLoadingText: {
    marginTop: 12,
    color: '#667eea',
    fontSize: 14,
  },
  aiExplanationError: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  aiExplanationErrorText: {
    color: '#e53e3e',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 12,
    paddingVertical: 8,
    paddingHorizontal: 20,
    backgroundColor: '#667eea',
    borderRadius: 8,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  aiExplanationText: {
    // Container for the explanation text
  },
  aiExplanationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  aiExplanationTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#2d3748',
    marginLeft: 8,
  },
  aiExplanationBody: {
    fontSize: 14,
    lineHeight: 22,
    color: '#4a5568',
  },
  aiExplanationDisclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginTop: 16,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0, 0, 0, 0.08)',
  },
  aiExplanationDisclaimerText: {
    flex: 1,
    marginLeft: 6,
    fontSize: 11,
    color: '#718096',
    lineHeight: 16,
  },

  // Differential Reasoning Styles
  reasoningButton: {
    backgroundColor: '#8b5cf6',
    shadowColor: '#8b5cf6',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
  },
  reasoningContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 16,
    marginHorizontal: 20,
    marginVertical: 10,
    padding: 16,
    shadowColor: '#8b5cf6',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 10,
    elevation: 4,
    borderLeftWidth: 4,
    borderLeftColor: '#8b5cf6',
  },
  reasoningHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(139, 92, 246, 0.2)',
  },
  reasoningTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#2d3748',
    marginLeft: 10,
  },
  reasoningLoading: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 40,
  },
  reasoningLoadingText: {
    marginTop: 16,
    color: '#8b5cf6',
    fontSize: 14,
    fontWeight: '500',
  },
  reasoningError: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  reasoningErrorText: {
    color: '#e53e3e',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
  reasoningRetryButton: {
    marginTop: 12,
    paddingVertical: 10,
    paddingHorizontal: 24,
    backgroundColor: '#8b5cf6',
    borderRadius: 8,
  },
  reasoningRetryText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  reasoningContent: {
    // Container for reasoning content
  },
  reasoningBody: {
    fontSize: 14,
    lineHeight: 24,
    color: '#4a5568',
  },
  reasoningDisclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginTop: 20,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: 'rgba(139, 92, 246, 0.15)',
  },
  reasoningDisclaimerText: {
    flex: 1,
    marginLeft: 8,
    fontSize: 11,
    color: '#718096',
    lineHeight: 16,
  },

  userHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 50,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.2)',
    marginBottom: 20,
  },
  userInfo: {
    flex: 1,
  },
  welcomeText: {
    color: '#1e3a5f',
    fontSize: 16,
    fontWeight: '600',
  },
  fullNameText: {
    color: '#4b5563',
    fontSize: 14,
    marginTop: 2,
  },
  logoutButton: {
    backgroundColor: 'rgba(220, 53, 69, 0.9)', // Strong red background
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#dc3545', // Solid red border
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
  },
  logoutButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold', // Bold text for better visibility
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: {width: 1, height: 1},
    textShadowRadius: 2,
  },
  headerButtonsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  helpButtonHeader: {
    backgroundColor: 'rgba(34, 139, 230, 0.9)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#228be6',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  historyButtonHeader: {
    backgroundColor: 'rgba(107, 66, 193, 0.9)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#6f42c1',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  trackingButtonHeader: {
    backgroundColor: 'rgba(16, 185, 129, 0.9)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#10b981',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  familyHistoryButtonHeader: {
    backgroundColor: 'rgba(139, 92, 246, 0.9)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#8b5cf6',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  analyticsButtonHeader: {
    backgroundColor: 'rgba(16, 185, 129, 0.9)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#10b981',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  headerButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: {width: 1, height: 1},
    textShadowRadius: 2,
  },
  menuButton: {
    backgroundColor: 'rgba(59, 130, 246, 0.9)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#3b82f6',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  menuButtonText: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
  },
  menuOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  menuContainer: {
    backgroundColor: '#ffffff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '80%',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: -4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 10,
  },
  menuHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  menuTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  menuCloseButton: {
    fontSize: 28,
    color: '#6b7280',
    fontWeight: '300',
  },
  menuContent: {
    padding: 10,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    marginVertical: 4,
    marginHorizontal: 10,
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  menuItemIcon: {
    fontSize: 28,
    marginRight: 16,
  },
  menuItemTextContainer: {
    flex: 1,
  },
  menuItemText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  menuItemSubtext: {
    fontSize: 13,
    color: '#6b7280',
  },
  menuDivider: {
    height: 1,
    backgroundColor: '#e5e7eb',
    marginVertical: 10,
    marginHorizontal: 10,
  },
  // Collapsible menu category styles
  menuCategory: {
    marginBottom: 8,
    marginHorizontal: 10,
  },
  menuCategoryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#bae6fd',
  },
  menuCategoryIcon: {
    fontSize: 22,
    marginRight: 12,
  },
  menuCategoryTitleContainer: {
    flex: 1,
  },
  menuCategoryTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0369a1',
  },
  menuCategoryCount: {
    fontSize: 12,
    color: '#0891b2',
    marginTop: 2,
  },
  menuCategoryArrow: {
    fontSize: 12,
    color: '#0369a1',
    marginLeft: 8,
  },
  menuCategoryContent: {
    marginTop: 4,
    marginLeft: 8,
    borderLeftWidth: 2,
    borderLeftColor: '#e0f2fe',
    paddingLeft: 8,
  },
  menuItemCompact: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    marginVertical: 2,
    backgroundColor: '#ffffff',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#f3f4f6',
  },
  menuItemIconSmall: {
    fontSize: 20,
    marginRight: 12,
  },
  logoutMenuItem: {
    backgroundColor: '#fef2f2',
    borderColor: '#fecaca',
  },
  logoutMenuText: {
    color: '#dc2626',
  },
  // Inflammatory Conditions Styling
  inflammatoryCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 6,
    borderLeftWidth: 5,
    borderLeftColor: '#ff6b6b',
  },
  inflammatoryTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#c53030',
    marginBottom: 15,
    textAlign: 'center',
  },
  inflammatoryResult: {
    alignItems: 'center',
    marginBottom: 15,
  },
  inflammatoryCondition: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2d3748',
    textAlign: 'center',
    marginBottom: 10,
  },
  inflammatoryProbabilities: {
    marginTop: 10,
  },
  probabilitiesSubtitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#4a5568',
    marginBottom: 10,
  },
  // Treatment Advice Card Styling (for burns)
  treatmentAdviceCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  treatmentAdviceTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
  },
  treatmentAdviceText: {
    fontSize: 14,
    lineHeight: 22,
    color: '#4a5568',
    marginBottom: 12,
  },
  medicalAttentionWarning: {
    backgroundColor: '#fef2f2',
    borderLeftWidth: 4,
    borderLeftColor: '#dc2626',
    padding: 12,
    borderRadius: 8,
    marginTop: 8,
  },
  medicalAttentionText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#dc2626',
    textAlign: 'center',
  },
  // Infectious Disease Styling
  infectiousCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 6,
    borderLeftWidth: 5,
    borderLeftColor: '#9333ea',
  },
  infectiousTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#7c3aed',
    marginBottom: 15,
    textAlign: 'center',
  },
  infectiousResult: {
    alignItems: 'center',
    marginBottom: 15,
  },
  infectiousDisease: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2d3748',
    textAlign: 'center',
    marginBottom: 10,
  },
  infectionDetails: {
    backgroundColor: '#faf5ff',
    borderRadius: 12,
    padding: 12,
    marginBottom: 15,
  },
  infectionDetailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 6,
  },
  infectionDetailLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
  },
  infectionDetailValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2d3748',
  },
  infectiousProbabilities: {
    marginTop: 10,
  },
  // Differential Diagnoses Styling
  differentialCard: {
    backgroundColor: '#f8f9ff',
    borderRadius: 16,
    padding: 18,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    elevation: 4,
    borderLeftWidth: 5,
    borderLeftColor: '#4299e1',
  },
  differentialTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 8,
    textAlign: 'center',
  },
  differentialSubtext: {
    fontSize: 12,
    color: '#6c757d',
    fontStyle: 'italic',
    textAlign: 'center',
    marginBottom: 15,
  },
  differentialSubsection: {
    marginBottom: 15,
  },
  differentialCategory: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 10,
  },
  diagnosisCard: {
    backgroundColor: 'white',
    padding: 12,
    borderRadius: 10,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  diagnosisHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  diagnosisRank: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#4299e1',
    marginRight: 8,
    minWidth: 25,
  },
  diagnosisCondition: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2d3748',
    flex: 1,
  },
  diagnosisProbability: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#4299e1',
  },
  severityBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    alignSelf: 'flex-start',
    marginBottom: 6,
  },
  severityText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
  },
  diagnosisUrgency: {
    fontSize: 12,
    color: '#e67e22',
    fontWeight: '600',
    marginBottom: 4,
  },
  contagiousWarning: {
    fontSize: 11,
    color: '#dc3545',
    fontWeight: '700',
    backgroundColor: '#ffe6e6',
    padding: 6,
    borderRadius: 6,
    marginBottom: 6,
    marginTop: 4,
  },
  diagnosisDescription: {
    fontSize: 12,
    color: '#4a5568',
    lineHeight: 16,
  },
  // Uncertainty Quantification Styling
  uncertaintyCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 6,
    borderLeftWidth: 5,
    borderLeftColor: '#6366f1',
  },
  uncertaintyTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#4338ca',
    marginBottom: 15,
    textAlign: 'center',
  },
  uncertaintyMetrics: {
    width: '100%',
  },
  uncertaintyRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  uncertaintyLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    flex: 1,
  },
  uncertaintyValueContainer: {
    flex: 1,
    alignItems: 'flex-end',
  },
  uncertaintyValue: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  reliabilityBar: {
    height: 6,
    borderRadius: 3,
    width: '100%',
    maxWidth: 120,
  },
  uncertaintyBreakdown: {
    backgroundColor: '#f8fafc',
    borderRadius: 8,
    padding: 15,
    marginBottom: 15,
  },
  uncertaintySubtitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#4a5568',
    marginBottom: 10,
  },
  uncertaintyMetric: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  metricLabel: {
    fontSize: 13,
    color: '#6b7280',
    flex: 1,
  },
  metricValue: {
    fontSize: 13,
    fontWeight: '500',
    color: '#374151',
    fontFamily: 'monospace',
  },
  clinicalRecommendation: {
    backgroundColor: '#f0f9ff',
    borderRadius: 8,
    padding: 15,
    marginBottom: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#0ea5e9',
  },
  recommendationTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#0c4a6e',
    marginBottom: 5,
  },
  recommendationText: {
    fontSize: 13,
    lineHeight: 18,
    fontStyle: 'italic',
  },
  mcInfo: {
    alignItems: 'center',
    marginTop: 10,
  },
  mcInfoText: {
    fontSize: 11,
    color: '#9ca3af',
    fontStyle: 'italic',
  },
  // Body Map Modal Styles
  modalContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  modalBackground: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingTop: 50,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2d3748',
  },
  modalCloseButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#f1f5f9',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalCloseButtonText: {
    fontSize: 20,
    color: '#64748b',
  },
  modalScrollView: {
    flex: 1,
  },
  modalFooter: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    backgroundColor: '#fff',
  },
  skipButton: {
    flex: 1,
    backgroundColor: '#f1f5f9',
    borderWidth: 1,
    borderColor: '#cbd5e0',
  },
  skipButtonText: {
    color: '#64748b',
    fontWeight: '600',
  },
  proceedButton: {
    flex: 2,
    backgroundColor: '#4299e1',
  },
  proceedButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  // Analysis Type Selector Styles
  analysisTypeContainer: {
    width: '100%',
    marginVertical: 12,
    paddingHorizontal: 8,
  },
  analysisTypeTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2c5282',
    marginBottom: 8,
    textAlign: 'center',
  },
  analysisTypeButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 8,
  },
  analysisTypeButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 8,
    borderRadius: 8,
    backgroundColor: '#f7fafc',
    borderWidth: 2,
    borderColor: '#e2e8f0',
    alignItems: 'center',
    justifyContent: 'center',
  },
  analysisTypeButtonActive: {
    backgroundColor: '#ebf8ff',
    borderColor: '#4299e1',
  },
  analysisTypeButtonText: {
    fontSize: 12,
    fontWeight: '500',
    color: '#4a5568',
  },
  analysisTypeButtonTextActive: {
    color: '#2c5282',
    fontWeight: '700',
  },
  // Analysis Info Banner Styles
  analysisInfoContainer: {
    backgroundColor: '#ebf8ff',
    borderRadius: 10,
    padding: 10,
    marginHorizontal: 20,
    marginVertical: 6,
    borderLeftWidth: 3,
    borderLeftColor: '#4299e1',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.08,
    shadowRadius: 3,
    elevation: 2,
  },
  analysisInfoText: {
    fontSize: 11,
    fontWeight: '500',
    color: '#2c5282',
    textAlign: 'center',
    lineHeight: 16,
    paddingHorizontal: 6,
  },
  // AR Treatment Simulator Styles
  arSimulatorContainer: {
    marginHorizontal: 16,
    marginTop: 20,
    marginBottom: 10,
  },
  arSimulatorButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    backgroundColor: '#667eea',
    padding: 16,
    borderRadius: 16,
    shadowColor: '#667eea',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  arSimulatorIcon: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 14,
  },
  arSimulatorIconText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
  arSimulatorTextContainer: {
    flex: 1,
  },
  arSimulatorTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 4,
  },
  arSimulatorSubtitle: {
    fontSize: 13,
    color: 'rgba(255, 255, 255, 0.9)',
  },
  arSimulatorArrow: {
    fontSize: 24,
    color: 'white',
    fontWeight: 'bold',
  },

  // Dermoscopy Results Styling
  dermoscopyContainer: {
    width: '100%',
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    marginTop: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 6,
    borderLeftWidth: 5,
    borderLeftColor: '#8b5cf6',
  },
  dermoscopyHeader: {
    marginBottom: 20,
  },
  dermoscopyHeaderTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 6,
  },
  dermoscopyHeaderSubtitle: {
    fontSize: 13,
    color: '#666',
    fontStyle: 'italic',
    lineHeight: 18,
  },
  dermoscopyHeaderIcon: {
    fontSize: 28,
    marginRight: 10,
  },
  dermoscopyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  dermoscopySubtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 20,
    fontStyle: 'italic',
  },
  dermoscopyScoresRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    gap: 12,
  },
  dermoscopyScoreCard: {
    flex: 1,
    backgroundColor: '#f8fafb',
    borderRadius: 12,
    padding: 14,
    alignItems: 'center',
    borderLeftWidth: 4,
    borderLeftColor: '#8b5cf6',
  },
  scoreCardLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 8,
    textAlign: 'center',
    fontWeight: '500',
  },
  scoreCardValue: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 6,
  },
  scoreCardInterpretation: {
    fontSize: 11,
    fontWeight: '600',
    textAlign: 'center',
    textTransform: 'uppercase',
  },
  scoreCardClassification: {
    fontSize: 11,
    fontWeight: '600',
    textAlign: 'center',
    textTransform: 'uppercase',
  },
  scoreCard: {
    flex: 1,
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 3,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  scoreTitle: {
    fontSize: 13,
    color: '#666',
    marginBottom: 8,
    textAlign: 'center',
  },
  scoreValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 4,
  },
  scoreInterpretation: {
    fontSize: 12,
    color: '#4a5568',
    textAlign: 'center',
    fontWeight: '600',
  },
  scoreClassification: {
    fontSize: 12,
    color: '#4a5568',
    textAlign: 'center',
    fontWeight: '600',
  },
  clinicalInterpretationCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 3,
  },
  clinicalSectionTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 8,
  },
  clinicalText: {
    fontSize: 14,
    color: '#4a5568',
    lineHeight: 20,
  },
  urgencyBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    alignSelf: 'flex-start',
    marginTop: 8,
  },
  urgencyText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  featuresGridContainer: {
    marginBottom: 20,
  },
  featuresSectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
  },
  featuresGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    justifyContent: 'space-between',
  },
  featureCard: {
    width: '48%',
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 3,
    borderLeftWidth: 4,
  },
  featureCardLow: {
    borderLeftColor: '#48bb78',
  },
  featureCardModerate: {
    borderLeftColor: '#ed8936',
  },
  featureCardHigh: {
    borderLeftColor: '#f56565',
  },
  featureTitle: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 6,
  },
  featureType: {
    fontSize: 12,
    color: '#4a5568',
    marginBottom: 4,
    textTransform: 'capitalize',
  },
  featureCount: {
    fontSize: 12,
    color: '#4a5568',
    marginBottom: 4,
  },
  featureCoverage: {
    fontSize: 12,
    color: '#4a5568',
    marginBottom: 4,
  },
  featureRisk: {
    fontSize: 11,
    fontWeight: '600',
    marginTop: 4,
  },
  riskLow: {
    color: '#48bb78',
  },
  riskModerate: {
    color: '#ed8936',
  },
  riskHigh: {
    color: '#f56565',
  },
  highRiskCard: {
    width: '100%',
    backgroundColor: '#fff5f5',
    borderLeftColor: '#f56565',
  },
  highRiskText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#f56565',
    marginTop: 4,
  },
  colorAnalysisCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 3,
    borderLeftWidth: 4,
  },
  symmetryAnalysisCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 3,
    borderLeftWidth: 4,
  },
  analysisCardTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 8,
  },
  analysisCardText: {
    fontSize: 13,
    color: '#4a5568',
    marginBottom: 4,
  },
  overlaySection: {
    marginBottom: 20,
  },
  overlayTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
  },
  overlayImage: {
    width: '100%',
    height: 250,
    borderRadius: 12,
    backgroundColor: '#f7fafc',
  },
  riskCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 18,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
    borderLeftWidth: 6,
  },
  riskLevel: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  riskRecommendation: {
    fontSize: 14,
    color: '#4a5568',
    lineHeight: 20,
    marginBottom: 12,
  },
  riskFactors: {
    marginTop: 8,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
  },
  riskFactorsTitle: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 8,
  },
  riskFactor: {
    fontSize: 13,
    color: '#4a5568',
    marginBottom: 4,
    lineHeight: 18,
  },
  criteriaMetCard: {
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
    borderLeftWidth: 4,
    borderLeftColor: '#4299e1',
  },
  criteriaMetTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 10,
  },
  criteriaItem: {
    fontSize: 13,
    color: '#4a5568',
    marginBottom: 6,
    lineHeight: 18,
  },

  // Additional Dermoscopy Specific Styles
  dermoscopyInterpretationCard: {
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    padding: 14,
    marginBottom: 15,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
  },
  interpretationTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 6,
  },
  interpretationText: {
    fontSize: 13,
    color: '#4a5568',
    lineHeight: 19,
  },
  dermoscopyRecommendationCard: {
    backgroundColor: '#fef3c7',
    borderRadius: 12,
    padding: 14,
    marginBottom: 15,
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  dermoscopyFeaturesTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginTop: 10,
    marginBottom: 12,
  },
  dermoscopyFeaturesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 20,
  },
  featureIcon: {
    fontSize: 24,
    marginBottom: 6,
    textAlign: 'center',
  },
  featureDescription: {
    fontSize: 11,
    color: '#666',
    marginTop: 6,
    lineHeight: 15,
  },
  featureSeverity: {
    fontSize: 12,
    color: '#4a5568',
    marginBottom: 4,
    fontWeight: '500',
  },
  featureIntensity: {
    fontSize: 12,
    color: '#4a5568',
    marginBottom: 4,
  },
  highRiskFeatureCard: {
    width: '100%',
    backgroundColor: '#fef2f2',
  },
  highRiskBadge: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#dc2626',
    backgroundColor: '#fee2e2',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
    alignSelf: 'flex-start',
    marginTop: 4,
    marginBottom: 6,
  },
  highRiskDescription: {
    fontSize: 12,
    color: '#991b1b',
    marginTop: 6,
    lineHeight: 16,
    fontWeight: '500',
  },
  colorAnalysisCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 3,
    borderLeftWidth: 4,
    borderLeftColor: '#ec4899',
  },
  colorAnalysisTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
  },
  colorAnalysisRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 10,
  },
  colorMetric: {
    alignItems: 'center',
  },
  colorMetricLabel: {
    fontSize: 11,
    color: '#666',
    marginBottom: 4,
  },
  colorMetricValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  colorWarning: {
    fontSize: 12,
    color: '#dc2626',
    backgroundColor: '#fef2f2',
    padding: 8,
    borderRadius: 8,
    marginTop: 8,
    textAlign: 'center',
  },
  symmetryAnalysisCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 3,
    borderLeftWidth: 4,
    borderLeftColor: '#14b8a6',
  },
  symmetryAnalysisTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
  },
  symmetryRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  symmetryMetric: {
    alignItems: 'center',
  },
  symmetryMetricLabel: {
    fontSize: 11,
    color: '#666',
    marginBottom: 4,
  },
  symmetryMetricValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  dermoscopyOverlaySection: {
    marginBottom: 20,
  },
  overlaySubtitle: {
    fontSize: 12,
    color: '#666',
    marginBottom: 12,
    fontStyle: 'italic',
  },
  dermoscopyOverlayImage: {
    width: '100%',
    height: 250,
    borderRadius: 12,
    backgroundColor: '#f7fafc',
    marginBottom: 10,
  },
  overlayLegend: {
    fontSize: 11,
    color: '#666',
    textAlign: 'center',
    fontStyle: 'italic',
  },
  dermoscopyRiskCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
    borderLeftWidth: 6,
  },
  riskHeader: {
    marginBottom: 12,
  },
  riskTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 10,
  },
  riskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    alignSelf: 'flex-start',
    marginBottom: 12,
  },
  riskScore: {
    fontSize: 14,
    color: '#4a5568',
    marginBottom: 10,
    fontWeight: '500',
  },
  riskBadgeText: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  riskFactorsSection: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
  },
  riskFactorRow: {
    flexDirection: 'row',
    marginBottom: 6,
    alignItems: 'flex-start',
  },
  riskFactorBullet: {
    fontSize: 14,
    color: '#4a5568',
    marginRight: 8,
    lineHeight: 18,
  },
  riskFactorText: {
    fontSize: 13,
    color: '#4a5568',
    flex: 1,
    lineHeight: 18,
  },
  criterionRow: {
    flexDirection: 'row',
    marginBottom: 6,
    alignItems: 'flex-start',
  },
  criterionBullet: {
    fontSize: 14,
    color: '#22c55e',
    marginRight: 8,
    fontWeight: 'bold',
    lineHeight: 18,
  },
  criterionText: {
    fontSize: 13,
    color: '#4a5568',
    flex: 1,
    lineHeight: 18,
  },

  // Multimodal Analysis Card Styles
  multimodalCard: {
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    padding: 16,
    marginHorizontal: 20,
    marginVertical: 12,
    borderWidth: 1,
    borderColor: '#0ea5e9',
    shadowColor: '#0ea5e9',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  multimodalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  multimodalTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0369a1',
  },
  multimodalBadge: {
    backgroundColor: '#0ea5e9',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  multimodalBadgeText: {
    fontSize: 10,
    fontWeight: '700',
    color: 'white',
    letterSpacing: 0.5,
  },
  multimodalSection: {
    marginBottom: 12,
  },
  multimodalSectionTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#0c4a6e',
    marginBottom: 8,
  },
  dataSourcesContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  dataSourceChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#7dd3fc',
  },
  dataSourceIcon: {
    fontSize: 14,
    marginRight: 4,
  },
  dataSourceText: {
    fontSize: 12,
    color: '#0369a1',
    fontWeight: '500',
  },
  confidenceBreakdown: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: '#e0f2fe',
  },
  confidenceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
  },
  confidenceLabel: {
    fontSize: 12,
    color: '#64748b',
  },
  confidenceValue: {
    fontSize: 12,
    fontWeight: '600',
    color: '#334155',
  },
  confidenceTotalRow: {
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    marginTop: 6,
    paddingTop: 8,
  },
  confidenceTotalLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#0369a1',
  },
  confidenceTotalValue: {
    fontSize: 14,
    fontWeight: '700',
    color: '#0ea5e9',
  },
  factorRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
    marginBottom: 4,
    borderWidth: 1,
    borderColor: '#e0f2fe',
  },
  factorName: {
    fontSize: 12,
    color: '#475569',
    textTransform: 'capitalize',
    flex: 1,
  },
  factorMultiplier: {
    fontSize: 12,
    fontWeight: '600',
    color: '#0ea5e9',
  },
  noFactorsContainer: {
    backgroundColor: '#fefce8',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: '#fef08a',
  },
  noFactorsText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#a16207',
    marginBottom: 4,
  },
  noFactorsHint: {
    fontSize: 11,
    color: '#ca8a04',
    lineHeight: 16,
  },
});