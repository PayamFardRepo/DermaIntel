import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  ScrollView,
  Pressable,
  Alert,
  ActivityIndicator,
  Image,
  Platform
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter, useLocalSearchParams } from 'expo-router';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';

export default function CreateLesionGroupScreen() {
  const { isAuthenticated, logout } = useAuth();
  const router = useRouter();
  const { analysis_id } = useLocalSearchParams();
  const { t } = useTranslation();

  const [lesionName, setLesionName] = useState('');
  const [lesionDescription, setLesionDescription] = useState('');
  const [monitoringFrequency, setMonitoringFrequency] = useState('monthly');
  const [firstNoticedDate, setFirstNoticedDate] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [analysisData, setAnalysisData] = useState<any>(null);

  // Load analysis data when component mounts
  useEffect(() => {
    if (analysis_id) {
      console.log('Component mounted with analysis_id:', analysis_id);
      loadAnalysisData();
    } else {
      console.log('No analysis_id provided - creating standalone lesion group');
      setIsLoading(false);
    }
  }, [analysis_id]);

  const loadAnalysisData = async () => {
    try {
      setIsLoading(true);

      // Get token from AuthService singleton
      const token = AuthService.getToken();
      console.log('Token exists:', !!token);
      console.log('Token length:', token?.length);

      if (!token) {
        console.error('No token found in AuthService');
        Alert.alert(t('createLesionGroup.errors.authError'), t('createLesionGroup.errors.noToken'));
        setTimeout(() => router.replace('/'), 100);
        return;
      }

      console.log('Analysis ID from params:', analysis_id);
      console.log('API Base URL:', API_BASE_URL);
      console.log('Full API URL:', `${API_BASE_URL}/analysis/history/${analysis_id}`);

      const response = await fetch(`${API_BASE_URL}/analysis/history/${analysis_id}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      console.log('Response status:', response.status);
      console.log('Response ok:', response.ok);

      if (!response.ok) {
        const responseText = await response.text();
        console.error('Response error body:', responseText);

        if (response.status === 401) {
          Alert.alert(t('createLesionGroup.errors.sessionExpired'), t('createLesionGroup.errors.pleaseLogin'));
          setTimeout(() => logout(), 100);
          return;
        }

        if (response.status === 404) {
          Alert.alert(t('createLesionGroup.errors.notFound'), t('createLesionGroup.errors.analysisNotFound', { id: analysis_id }));
          setTimeout(() => router.back(), 100);
          return;
        }

        throw new Error(`HTTP ${response.status}: ${responseText}`);
      }

      const data = await response.json();
      console.log('Analysis data loaded successfully:', JSON.stringify(data, null, 2));
      setAnalysisData(data);

      // Pre-populate lesion name with predicted class if available
      if (data.predicted_class) {
        setLesionName(`${data.predicted_class} - ${data.body_location || 'Unknown location'}`);
      }
    } catch (error: any) {
      console.error('Failed to load analysis - Full error:', error);
      console.error('Error message:', error.message);
      console.error('Error stack:', error.stack);
      Alert.alert(
        t('createLesionGroup.errors.loadError'),
        t('createLesionGroup.errors.loadErrorDetails', { message: error.message }),
        [
          { text: t('createLesionGroup.errors.goBack'), onPress: () => router.back() },
          { text: t('createLesionGroup.errors.retry'), onPress: () => loadAnalysisData() }
        ]
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateLesionGroup = async () => {
    // Validation
    if (!lesionName.trim()) {
      Alert.alert(t('createLesionGroup.errors.requiredField'), t('createLesionGroup.errors.enterName'));
      return;
    }

    try {
      setIsSubmitting(true);
      const token = AuthService.getToken();

      const formData = new FormData();
      formData.append('lesion_name', lesionName.trim());
      formData.append('monitoring_frequency', monitoringFrequency);

      if (analysis_id) {
        formData.append('analysis_id', String(analysis_id));
      }

      if (lesionDescription.trim()) {
        formData.append('lesion_description', lesionDescription.trim());
      }

      if (firstNoticedDate.trim()) {
        formData.append('first_noticed_date', firstNoticedDate.trim());
      }

      const response = await fetch(`${API_BASE_URL}/lesion_groups/`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create lesion group');
      }

      const result = await response.json();

      Alert.alert(
        t('createLesionGroup.alerts.success'),
        t('createLesionGroup.alerts.successMessage', { frequency: monitoringFrequency }),
        [
          {
            text: t('createLesionGroup.alerts.viewTracked'),
            onPress: () => router.push('/lesion-tracking')
          },
          {
            text: t('createLesionGroup.alerts.viewDetails'),
            onPress: () => router.push(`/lesion-detail?id=${result.id}` as any)
          }
        ]
      );
    } catch (error: any) {
      console.error('Failed to create lesion group:', error);

      if (error.message.includes('401') || error.message.includes('Authentication')) {
        Alert.alert(
          t('createLesionGroup.errors.sessionExpired'),
          t('createLesionGroup.errors.pleaseLogin'),
          [{ text: 'OK', onPress: () => logout() }]
        );
      } else {
        Alert.alert(t('createLesionGroup.alerts.error'), error.message || t('createLesionGroup.alerts.failedToCreate'));
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const frequencyOptions = [
    { value: 'weekly', label: t('createLesionGroup.monitoringFrequency.weekly.label'), description: t('createLesionGroup.monitoringFrequency.weekly.description') },
    { value: 'monthly', label: t('createLesionGroup.monitoringFrequency.monthly.label'), description: t('createLesionGroup.monitoringFrequency.monthly.description') },
    { value: 'quarterly', label: t('createLesionGroup.monitoringFrequency.quarterly.label'), description: t('createLesionGroup.monitoringFrequency.quarterly.description') },
    { value: 'biannual', label: t('createLesionGroup.monitoringFrequency.biannual.label'), description: t('createLesionGroup.monitoringFrequency.biannual.description') },
    { value: 'annual', label: t('createLesionGroup.monitoringFrequency.annual.label'), description: t('createLesionGroup.monitoringFrequency.annual.description') }
  ];

  if (isLoading) {
    return (
      <View style={styles.container}>
        <LinearGradient
          colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.background}
        />
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#10b981" />
          <Text style={styles.loadingText}>{t('createLesionGroup.loading')}</Text>
        </View>
      </View>
    );
  }

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
        <Pressable
          style={styles.backButton}
          onPress={() => router.back()}
        >
          <Text style={styles.backButtonText}>← {t('createLesionGroup.back')}</Text>
        </Pressable>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>🔍 {t('createLesionGroup.title')}</Text>
          <Text style={styles.headerSubtitle}>
            {analysisData ? t('createLesionGroup.subtitleWithAnalysis') : t('createLesionGroup.subtitleWithoutAnalysis')}
          </Text>
        </View>
      </View>

      <ScrollView
        style={styles.scrollContainer}
        contentContainerStyle={styles.scrollContent}
      >
        {/* Analysis Preview */}
        {analysisData && (
          <View style={styles.previewCard}>
            <Text style={styles.previewTitle}>{t('createLesionGroup.analysisPreview.title')}</Text>
            {analysisData.image_url && (
              <Image
                source={{ uri: `${API_BASE_URL}${analysisData.image_url}` }}
                style={styles.previewImage}
                resizeMode="cover"
              />
            )}
            <View style={styles.previewInfo}>
              <Text style={styles.previewLabel}>{t('createLesionGroup.analysisPreview.diagnosis')}</Text>
              <Text style={styles.previewValue}>{analysisData.predicted_class || 'N/A'}</Text>
            </View>
            <View style={styles.previewInfo}>
              <Text style={styles.previewLabel}>{t('createLesionGroup.analysisPreview.location')}</Text>
              <Text style={styles.previewValue}>
                {analysisData.body_location
                  ? `${analysisData.body_location}${analysisData.body_side ? ` (${analysisData.body_side})` : ''}`
                  : t('createLesionGroup.analysisPreview.notSpecified')}
              </Text>
            </View>
            <View style={styles.previewInfo}>
              <Text style={styles.previewLabel}>{t('createLesionGroup.analysisPreview.date')}</Text>
              <Text style={styles.previewValue}>
                {new Date(analysisData.created_at).toLocaleDateString()}
              </Text>
            </View>
          </View>
        )}

        {/* Help text when no analysis data */}
        {!analysisData && (
          <View style={styles.infoBox}>
            <Text style={styles.infoTitle}>{t('createLesionGroup.noAnalysisInfo.title')}</Text>
            <Text style={styles.infoText}>
              {t('createLesionGroup.noAnalysisInfo.description')}
            </Text>
          </View>
        )}

        {/* Lesion Name Input */}
        <View style={styles.inputSection}>
          <Text style={styles.inputLabel}>{t('createLesionGroup.lesionName.label')}</Text>
          <Text style={styles.inputHelper}>
            {t('createLesionGroup.lesionName.helper')}
          </Text>
          <TextInput
            style={styles.textInput}
            value={lesionName}
            onChangeText={setLesionName}
            placeholder={t('createLesionGroup.lesionName.placeholder')}
            placeholderTextColor="#a0aec0"
            maxLength={100}
          />
        </View>

        {/* Description Input */}
        <View style={styles.inputSection}>
          <Text style={styles.inputLabel}>{t('createLesionGroup.description.label')}</Text>
          <Text style={styles.inputHelper}>
            {t('createLesionGroup.description.helper')}
          </Text>
          <TextInput
            style={[styles.textInput, styles.textAreaInput]}
            value={lesionDescription}
            onChangeText={setLesionDescription}
            placeholder={t('createLesionGroup.description.placeholder')}
            placeholderTextColor="#a0aec0"
            multiline
            numberOfLines={4}
            maxLength={500}
          />
        </View>

        {/* Monitoring Frequency Selection */}
        <View style={styles.inputSection}>
          <Text style={styles.inputLabel}>{t('createLesionGroup.monitoringFrequency.label')}</Text>
          <Text style={styles.inputHelper}>
            {t('createLesionGroup.monitoringFrequency.helper')}
          </Text>
          <View style={styles.frequencyOptions}>
            {frequencyOptions.map((option) => (
              <Pressable
                key={option.value}
                style={[
                  styles.frequencyOption,
                  monitoringFrequency === option.value && styles.frequencyOptionSelected
                ]}
                onPress={() => setMonitoringFrequency(option.value)}
              >
                <View style={styles.frequencyRadio}>
                  {monitoringFrequency === option.value && (
                    <View style={styles.frequencyRadioInner} />
                  )}
                </View>
                <View style={styles.frequencyTextContainer}>
                  <Text style={[
                    styles.frequencyLabel,
                    monitoringFrequency === option.value && styles.frequencyLabelSelected
                  ]}>
                    {option.label}
                  </Text>
                  <Text style={styles.frequencyDescription}>{option.description}</Text>
                </View>
              </Pressable>
            ))}
          </View>
        </View>

        {/* First Noticed Date */}
        <View style={styles.inputSection}>
          <Text style={styles.inputLabel}>{t('createLesionGroup.firstNoticed.label')}</Text>
          <Text style={styles.inputHelper}>
            {t('createLesionGroup.firstNoticed.helper')}
          </Text>
          <TextInput
            style={styles.textInput}
            value={firstNoticedDate}
            onChangeText={setFirstNoticedDate}
            placeholder={t('createLesionGroup.firstNoticed.placeholder')}
            placeholderTextColor="#a0aec0"
            maxLength={10}
          />
        </View>

        {/* Info Box */}
        <View style={styles.infoBox}>
          <Text style={styles.infoTitle}>{t('createLesionGroup.nextSteps.title')}</Text>
          <Text style={styles.infoText}>
            {t('createLesionGroup.nextSteps.description')}
          </Text>
        </View>

        {/* Spacer for fixed button */}
        <View style={styles.spacer} />
      </ScrollView>

      {/* Fixed Create Button */}
      <View style={styles.fixedButtonContainer}>
        <Pressable
          style={[styles.createButton, (isSubmitting || !lesionName.trim()) && styles.buttonDisabled]}
          onPress={handleCreateLesionGroup}
          disabled={isSubmitting || !lesionName.trim()}
        >
          <Text style={styles.createButtonText}>
            {isSubmitting ? t('createLesionGroup.button.creating') : t('createLesionGroup.button.create')}
          </Text>
        </Pressable>
      </View>
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#4a5568',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 16,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.2)',
  },
  backButton: {
    backgroundColor: 'rgba(66, 153, 225, 0.9)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#4299e1',
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
  scrollContainer: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingVertical: 20,
    paddingBottom: 100,
  },
  previewCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    borderLeftWidth: 4,
    borderLeftColor: '#10b981',
  },
  previewTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 16,
  },
  previewImage: {
    width: '100%',
    height: 200,
    borderRadius: 12,
    marginBottom: 16,
    backgroundColor: '#e2e8f0',
  },
  previewInfo: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  previewLabel: {
    fontSize: 14,
    color: '#4a5568',
    fontWeight: '600',
    width: 100,
  },
  previewValue: {
    fontSize: 14,
    color: '#2d3748',
    flex: 1,
  },
  inputSection: {
    marginBottom: 24,
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 6,
  },
  inputHelper: {
    fontSize: 13,
    color: '#718096',
    marginBottom: 10,
    lineHeight: 18,
  },
  textInput: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    color: '#2d3748',
    borderWidth: 2,
    borderColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  textAreaInput: {
    minHeight: 100,
    textAlignVertical: 'top',
  },
  frequencyOptions: {
    gap: 12,
  },
  frequencyOption: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    borderWidth: 2,
    borderColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  frequencyOptionSelected: {
    borderColor: '#10b981',
    backgroundColor: '#f0fdf4',
  },
  frequencyRadio: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#cbd5e0',
    marginRight: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  frequencyRadioInner: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#10b981',
  },
  frequencyTextContainer: {
    flex: 1,
  },
  frequencyLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2d3748',
    marginBottom: 2,
  },
  frequencyLabelSelected: {
    color: '#10b981',
  },
  frequencyDescription: {
    fontSize: 13,
    color: '#718096',
  },
  infoBox: {
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#0ea5e9',
    marginTop: 8,
  },
  infoTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#0c4a6e',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 14,
    color: '#0c4a6e',
    lineHeight: 20,
  },
  spacer: {
    height: 20,
  },
  fixedButtonContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#fff',
    padding: 20,
    paddingBottom: 30,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 5,
  },
  createButton: {
    backgroundColor: '#10b981',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#10b981',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  createButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
});
