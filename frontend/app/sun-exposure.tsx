import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  TextInput,
  Alert,
  ActivityIndicator,
  Switch,
  Modal,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';

export default function SunExposureScreen() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const { t } = useTranslation();

  // Tab state
  const [activeTab, setActiveTab] = useState<'log' | 'history' | 'statistics'>('log');

  // Form state for logging new exposure
  const [exposureDate, setExposureDate] = useState(new Date().toISOString().split('T')[0]);
  const [durationMinutes, setDurationMinutes] = useState('');
  const [timeOfDay, setTimeOfDay] = useState('midday');
  const [location, setLocation] = useState('');
  const [activity, setActivity] = useState('');
  const [uvIndex, setUvIndex] = useState('');
  const [weatherConditions, setWeatherConditions] = useState('sunny');

  // Protection state
  const [sunProtectionUsed, setSunProtectionUsed] = useState(false);
  const [sunscreenApplied, setSunscreenApplied] = useState(false);
  const [spfLevel, setSpfLevel] = useState('');
  const [sunscreenReapplied, setSunscreenReapplied] = useState(false);
  const [protectiveClothing, setProtectiveClothing] = useState(false);
  const [hatWorn, setHatWorn] = useState(false);
  const [sunglassesWorn, setSunglassesWorn] = useState(false);
  const [shadeSought, setShadeSought] = useState(false);

  // Body areas exposed
  const [exposedBodyAreas, setExposedBodyAreas] = useState<string[]>([]);
  const bodyAreaOptions = ['face', 'arms', 'legs', 'back', 'chest', 'shoulders', 'hands', 'feet'];

  // Skin reaction
  const [skinReaction, setSkinReaction] = useState('none');
  const [reactionSeverity, setReactionSeverity] = useState(0);
  const [peelingOccurred, setPeelingOccurred] = useState(false);
  const [intentionalTanning, setIntentionalTanning] = useState(false);
  const [indoorTanning, setIndoorTanning] = useState(false);
  const [notes, setNotes] = useState('');

  // History and statistics
  const [exposures, setExposures] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [correlations, setCorrelations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    }
  }, [isAuthenticated]);

  // Load data when tab changes
  useEffect(() => {
    if (isAuthenticated) {
      if (activeTab === 'history') {
        loadExposureHistory();
      } else if (activeTab === 'statistics') {
        loadStatistics();
        loadCorrelations();
      }
    }
  }, [activeTab, isAuthenticated]);

  const loadExposureHistory = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/sun-exposure?limit=50`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${user?.token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        setExposures(data.exposures || []);
      } else {
        // Just set empty data, no error alerts
        console.log('No exposure history data available (status:', response.status, ')');
        setExposures([]);
      }
    } catch (error) {
      // Just set empty data, no error alerts
      console.log('Could not load exposure history:', error);
      setExposures([]);
    } finally {
      setIsLoading(false);
    }
  };

  const loadStatistics = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/sun-exposure/statistics/summary?period_days=30`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${user?.token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        setStatistics(data);
      } else {
        // Just set empty data, no error alerts
        console.log('No statistics data available (status:', response.status, ')');
        setStatistics({
          total_exposures: 0,
          total_exposure_hours: 0,
          average_uv_index: 0,
          max_uv_index: 0,
          sunburn_events: 0,
          protection_rate: 0,
          average_spf: 0,
          high_risk_exposures: 0,
          recommendations: []
        });
      }
    } catch (error) {
      // Just set empty data, no error alerts
      console.log('Could not load statistics:', error);
      setStatistics({
        total_exposures: 0,
        total_exposure_hours: 0,
        average_uv_index: 0,
        max_uv_index: 0,
        sunburn_events: 0,
        protection_rate: 0,
        average_spf: 0,
        high_risk_exposures: 0,
        recommendations: []
      });
    } finally {
      setIsLoading(false);
    }
  };

  const loadCorrelations = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/sun-exposure/correlations`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${user?.token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        setCorrelations(data.correlations || []);
      } else {
        // Set empty array on error, don't show alert
        setCorrelations([]);
      }
    } catch (error) {
      console.error('Error loading correlations:', error);
      // Set empty array on error, don't show alert
      setCorrelations([]);
    }
  };

  const toggleBodyArea = (area: string) => {
    if (exposedBodyAreas.includes(area)) {
      setExposedBodyAreas(exposedBodyAreas.filter(a => a !== area));
    } else {
      setExposedBodyAreas([...exposedBodyAreas, area]);
    }
  };

  const handleSubmitExposure = async () => {
    // Validation
    if (!durationMinutes || parseInt(durationMinutes) <= 0) {
      Alert.alert(t('sunExposure.validation.title', 'Validation Error'), t('sunExposure.validation.durationError'));
      return;
    }

    if (!location || !activity) {
      Alert.alert(t('sunExposure.validation.title', 'Validation Error'), t('sunExposure.validation.locationActivityError'));
      return;
    }

    if (sunscreenApplied && (!spfLevel || parseInt(spfLevel) <= 0)) {
      Alert.alert(t('sunExposure.validation.title', 'Validation Error'), t('sunExposure.validation.spfError'));
      return;
    }

    setIsSubmitting(true);

    try {
      const exposureDatetime = new Date(exposureDate).toISOString();

      const requestBody = {
        exposure_date: exposureDatetime,
        duration_minutes: parseInt(durationMinutes),
        time_of_day: timeOfDay,
        location,
        activity,
        uv_index: uvIndex ? parseFloat(uvIndex) : null,
        uv_index_source: 'manual',
        weather_conditions: weatherConditions,
        sun_protection_used: sunProtectionUsed,
        sunscreen_applied: sunscreenApplied,
        spf_level: spfLevel ? parseInt(spfLevel) : null,
        sunscreen_reapplied: sunscreenReapplied,
        protective_clothing: protectiveClothing,
        hat_worn: hatWorn,
        sunglasses_worn: sunglassesWorn,
        shade_sought: shadeSought,
        exposed_body_areas: exposedBodyAreas,
        skin_reaction: skinReaction,
        reaction_severity: reactionSeverity,
        peeling_occurred: peelingOccurred,
        intentional_tanning: intentionalTanning,
        indoor_tanning: indoorTanning,
        notes: notes || null,
      };

      const response = await fetch(`${API_BASE_URL}/sun-exposure`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${user?.token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(
          t('sunExposure.success.title'),
          t('sunExposure.success.message', {
            uvDose: data.uv_dose?.toFixed(1) || '0',
            riskScore: data.risk_score?.toFixed(0) || '0'
          })
        );

        // Reset form
        resetForm();
        setActiveTab('history');
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || t('sunExposure.error.failed'));
      }
    } catch (error: any) {
      console.error('Error logging exposure:', error);
      Alert.alert(t('sunExposure.error.title'), error.message || t('sunExposure.error.failed'));
    } finally {
      setIsSubmitting(false);
    }
  };

  const resetForm = () => {
    setExposureDate(new Date().toISOString().split('T')[0]);
    setDurationMinutes('');
    setTimeOfDay('midday');
    setLocation('');
    setActivity('');
    setUvIndex('');
    setWeatherConditions('sunny');
    setSunProtectionUsed(false);
    setSunscreenApplied(false);
    setSpfLevel('');
    setSunscreenReapplied(false);
    setProtectiveClothing(false);
    setHatWorn(false);
    setSunglassesWorn(false);
    setShadeSought(false);
    setExposedBodyAreas([]);
    setSkinReaction('none');
    setReactionSeverity(0);
    setPeelingOccurred(false);
    setIntentionalTanning(false);
    setIndoorTanning(false);
    setNotes('');
  };

  const getRiskBadgeStyle = (riskScore: number) => {
    if (riskScore >= 70) {
      return { ...styles.riskBadge, backgroundColor: '#dc3545' };
    } else if (riskScore >= 40) {
      return { ...styles.riskBadge, backgroundColor: '#ffc107', color: '#000' };
    } else {
      return { ...styles.riskBadge, backgroundColor: '#28a745' };
    }
  };

  const renderLogTab = () => (
    <ScrollView style={styles.tabContent}>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('sunExposure.log.whenWhere')}</Text>

        <Text style={styles.label}>{t('sunExposure.log.date')}</Text>
        <TextInput
          style={styles.input}
          value={exposureDate}
          onChangeText={setExposureDate}
          placeholder="YYYY-MM-DD"
        />

        <Text style={styles.label}>{t('sunExposure.log.duration')}</Text>
        <TextInput
          style={styles.input}
          value={durationMinutes}
          onChangeText={setDurationMinutes}
          placeholder={t('sunExposure.log.durationPlaceholder')}
          keyboardType="numeric"
        />

        <Text style={styles.label}>{t('sunExposure.log.timeOfDay')}</Text>
        <View style={styles.buttonGroup}>
          {['morning', 'midday', 'afternoon', 'evening'].map(time => (
            <Pressable
              key={time}
              style={[styles.optionButton, timeOfDay === time && styles.optionButtonActive]}
              onPress={() => setTimeOfDay(time)}
            >
              <Text style={[styles.optionButtonText, timeOfDay === time && styles.optionButtonTextActive]}>
                {t(`sunExposure.log.${time}`)}
              </Text>
            </Pressable>
          ))}
        </View>

        <Text style={styles.label}>{t('sunExposure.log.location')}</Text>
        <TextInput
          style={styles.input}
          value={location}
          onChangeText={setLocation}
          placeholder={t('sunExposure.log.locationPlaceholder')}
        />

        <Text style={styles.label}>{t('sunExposure.log.activity')}</Text>
        <TextInput
          style={styles.input}
          value={activity}
          onChangeText={setActivity}
          placeholder={t('sunExposure.log.activityPlaceholder')}
        />
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('sunExposure.log.uvConditions')}</Text>

        <Text style={styles.label}>{t('sunExposure.log.uvIndex')}</Text>
        <TextInput
          style={styles.input}
          value={uvIndex}
          onChangeText={setUvIndex}
          placeholder={t('sunExposure.log.uvIndexPlaceholder')}
          keyboardType="decimal-pad"
        />

        <Text style={styles.label}>{t('sunExposure.log.weatherConditions')}</Text>
        <View style={styles.buttonGroup}>
          {['sunny', 'partly_cloudy', 'cloudy', 'overcast'].map(weather => (
            <Pressable
              key={weather}
              style={[styles.optionButton, weatherConditions === weather && styles.optionButtonActive]}
              onPress={() => setWeatherConditions(weather)}
            >
              <Text style={[styles.optionButtonText, weatherConditions === weather && styles.optionButtonTextActive]}>
                {t(`sunExposure.log.${weather === 'partly_cloudy' ? 'partlyCloudy' : weather}`)}
              </Text>
            </Pressable>
          ))}
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('sunExposure.log.sunProtection')}</Text>

        <View style={styles.switchRow}>
          <Text style={styles.switchLabel}>{t('sunExposure.log.sunProtectionUsed')}</Text>
          <Switch value={sunProtectionUsed} onValueChange={setSunProtectionUsed} />
        </View>

        {sunProtectionUsed && (
          <>
            <View style={styles.switchRow}>
              <Text style={styles.switchLabel}>{t('sunExposure.log.sunscreenApplied')}</Text>
              <Switch value={sunscreenApplied} onValueChange={setSunscreenApplied} />
            </View>

            {sunscreenApplied && (
              <>
                <Text style={styles.label}>{t('sunExposure.log.spfLevel')}</Text>
                <TextInput
                  style={styles.input}
                  value={spfLevel}
                  onChangeText={setSpfLevel}
                  placeholder={t('sunExposure.log.spfPlaceholder')}
                  keyboardType="numeric"
                />

                <View style={styles.switchRow}>
                  <Text style={styles.switchLabel}>{t('sunExposure.log.sunscreenReapplied')}</Text>
                  <Switch value={sunscreenReapplied} onValueChange={setSunscreenReapplied} />
                </View>
              </>
            )}

            <View style={styles.switchRow}>
              <Text style={styles.switchLabel}>{t('sunExposure.log.protectiveClothing')}</Text>
              <Switch value={protectiveClothing} onValueChange={setProtectiveClothing} />
            </View>

            <View style={styles.switchRow}>
              <Text style={styles.switchLabel}>{t('sunExposure.log.hatWorn')}</Text>
              <Switch value={hatWorn} onValueChange={setHatWorn} />
            </View>

            <View style={styles.switchRow}>
              <Text style={styles.switchLabel}>{t('sunExposure.log.sunglassesWorn')}</Text>
              <Switch value={sunglassesWorn} onValueChange={setSunglassesWorn} />
            </View>

            <View style={styles.switchRow}>
              <Text style={styles.switchLabel}>{t('sunExposure.log.shadeSought')}</Text>
              <Switch value={shadeSought} onValueChange={setShadeSought} />
            </View>
          </>
        )}
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('sunExposure.log.exposedBodyAreas')}</Text>
        <View style={styles.bodyAreasGrid}>
          {bodyAreaOptions.map(area => (
            <Pressable
              key={area}
              style={[styles.bodyAreaButton, exposedBodyAreas.includes(area) && styles.bodyAreaButtonActive]}
              onPress={() => toggleBodyArea(area)}
            >
              <Text style={[styles.bodyAreaText, exposedBodyAreas.includes(area) && styles.bodyAreaTextActive]}>
                {t(`sunExposure.log.bodyAreas.${area}`)}
              </Text>
            </Pressable>
          ))}
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('sunExposure.log.skinReaction')}</Text>

        <Text style={styles.label}>{t('sunExposure.log.reactionType')}</Text>
        <View style={styles.buttonGroup}>
          {['none', 'mild_redness', 'moderate_burn', 'severe_burn', 'tanning'].map(reaction => (
            <Pressable
              key={reaction}
              style={[styles.optionButton, skinReaction === reaction && styles.optionButtonActive]}
              onPress={() => setSkinReaction(reaction)}
            >
              <Text style={[styles.optionButtonText, skinReaction === reaction && styles.optionButtonTextActive]}>
                {t(`sunExposure.log.${reaction === 'mild_redness' ? 'mildRedness' : reaction === 'moderate_burn' ? 'moderateBurn' : reaction === 'severe_burn' ? 'severeBurn' : reaction}`)}
              </Text>
            </Pressable>
          ))}
        </View>

        <View style={styles.switchRow}>
          <Text style={styles.switchLabel}>{t('sunExposure.log.peelingOccurred')}</Text>
          <Switch value={peelingOccurred} onValueChange={setPeelingOccurred} />
        </View>

        <View style={styles.switchRow}>
          <Text style={styles.switchLabel}>{t('sunExposure.log.intentionalTanning')}</Text>
          <Switch value={intentionalTanning} onValueChange={setIntentionalTanning} />
        </View>

        <View style={styles.switchRow}>
          <Text style={styles.switchLabel}>{t('sunExposure.log.indoorTanning')}</Text>
          <Switch value={indoorTanning} onValueChange={setIndoorTanning} />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('sunExposure.log.notes')}</Text>
        <TextInput
          style={[styles.input, styles.notesInput]}
          value={notes}
          onChangeText={setNotes}
          placeholder={t('sunExposure.log.notesPlaceholder')}
          multiline
          numberOfLines={4}
        />
      </View>

      <Pressable
        style={[styles.submitButton, isSubmitting && styles.submitButtonDisabled]}
        onPress={handleSubmitExposure}
        disabled={isSubmitting}
      >
        {isSubmitting ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.submitButtonText}>{t('sunExposure.log.logExposure')}</Text>
        )}
      </Pressable>

      <View style={{ height: 40 }} />
    </ScrollView>
  );

  const renderHistoryTab = () => (
    <ScrollView style={styles.tabContent}>
      {isLoading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4299e1" />
          <Text style={styles.loadingText}>{t('sunExposure.history.loading')}</Text>
        </View>
      ) : exposures.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateText}>{t('sunExposure.history.emptyTitle')}</Text>
          <Text style={styles.emptyStateSubtext}>{t('sunExposure.history.emptySubtext')}</Text>
        </View>
      ) : (
        exposures.map((exposure: any) => (
          <View key={exposure.id} style={styles.exposureCard}>
            <View style={styles.exposureHeader}>
              <Text style={styles.exposureDate}>
                {new Date(exposure.exposure_date).toLocaleDateString()}
              </Text>
              <View style={getRiskBadgeStyle(exposure.risk_score || 0)}>
                <Text style={styles.riskBadgeText}>
                  {t('sunExposure.history.risk')} {Math.round(exposure.risk_score || 0)}
                </Text>
              </View>
            </View>

            <View style={styles.exposureDetails}>
              <Text style={styles.exposureDetailText}>
                ⏱️ {exposure.duration_minutes} min • 🌤️ UV {exposure.uv_index || 'N/A'}
              </Text>
              <Text style={styles.exposureDetailText}>
                📍 {exposure.location} • 🏃 {exposure.activity}
              </Text>
              <Text style={styles.exposureDetailText}>
                🕐 {exposure.time_of_day.charAt(0).toUpperCase() + exposure.time_of_day.slice(1)}
              </Text>

              {exposure.sun_protection_used && (
                <View style={styles.protectionBadge}>
                  <Text style={styles.protectionBadgeText}>
                    🛡️ {t('sunExposure.history.protected')} {exposure.spf_level ? `(SPF ${exposure.spf_level})` : ''}
                  </Text>
                </View>
              )}

              {exposure.skin_reaction !== 'none' && (
                <Text style={[styles.exposureDetailText, { color: '#dc3545', marginTop: 8 }]}>
                  ⚠️ {exposure.skin_reaction.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                </Text>
              )}

              {exposure.calculated_uv_dose > 0 && (
                <Text style={styles.exposureDetailText}>
                  {t('sunExposure.history.uvDose')} {exposure.calculated_uv_dose.toFixed(1)}
                </Text>
              )}
            </View>
          </View>
        ))
      )}
      <View style={{ height: 20 }} />
    </ScrollView>
  );

  const renderStatisticsTab = () => (
    <ScrollView style={styles.tabContent}>
      {isLoading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4299e1" />
          <Text style={styles.loadingText}>{t('sunExposure.statistics.loading')}</Text>
        </View>
      ) : statistics ? (
        <>
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>{t('sunExposure.statistics.summary')}</Text>

            <View style={styles.statsGrid}>
              <View style={styles.statCard}>
                <Text style={styles.statNumber}>{statistics.total_exposures}</Text>
                <Text style={styles.statLabel}>{t('sunExposure.statistics.exposures')}</Text>
              </View>
              <View style={styles.statCard}>
                <Text style={styles.statNumber}>{statistics.total_exposure_hours}</Text>
                <Text style={styles.statLabel}>{t('sunExposure.statistics.totalHours')}</Text>
              </View>
              <View style={styles.statCard}>
                <Text style={styles.statNumber}>{statistics.average_uv_index}</Text>
                <Text style={styles.statLabel}>{t('sunExposure.statistics.avgUVIndex')}</Text>
              </View>
              <View style={styles.statCard}>
                <Text style={styles.statNumber}>{statistics.max_uv_index}</Text>
                <Text style={styles.statLabel}>{t('sunExposure.statistics.maxUVIndex')}</Text>
              </View>
            </View>

            <View style={styles.statsGrid}>
              <View style={styles.statCard}>
                <Text style={styles.statNumber}>{statistics.sunburn_events}</Text>
                <Text style={styles.statLabel}>{t('sunExposure.statistics.sunburns')}</Text>
              </View>
              <View style={styles.statCard}>
                <Text style={styles.statNumber}>{(statistics.protection_rate * 100).toFixed(0)}%</Text>
                <Text style={styles.statLabel}>{t('sunExposure.statistics.protectionRate')}</Text>
              </View>
              <View style={styles.statCard}>
                <Text style={styles.statNumber}>{statistics.average_spf}</Text>
                <Text style={styles.statLabel}>{t('sunExposure.statistics.avgSPF')}</Text>
              </View>
              <View style={styles.statCard}>
                <Text style={styles.statNumber}>{statistics.high_risk_exposures}</Text>
                <Text style={styles.statLabel}>{t('sunExposure.statistics.highRisk')}</Text>
              </View>
            </View>

            {statistics.most_common_location && (
              <Text style={styles.statsInfo}>
                {t('sunExposure.statistics.mostCommonLocation')} {statistics.most_common_location}
              </Text>
            )}
            {statistics.most_common_activity && (
              <Text style={styles.statsInfo}>
                {t('sunExposure.statistics.mostCommonActivity')} {statistics.most_common_activity}
              </Text>
            )}
          </View>

          {statistics.recommendations && statistics.recommendations.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>{t('sunExposure.statistics.recommendations')}</Text>
              {statistics.recommendations.map((rec: string, index: number) => (
                <View key={index} style={styles.recommendationItem}>
                  <Text style={styles.recommendationText}>• {rec}</Text>
                </View>
              ))}
            </View>
          )}

          {correlations && correlations.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>{t('sunExposure.statistics.lesionCorrelations')}</Text>
              {correlations.map((corr: any) => (
                <View key={corr.id} style={styles.correlationCard}>
                  <Text style={styles.correlationTitle}>{corr.lesion_name}</Text>
                  <Text style={styles.correlationSubtitle}>
                    {corr.lesion_body_area} • {corr.total_exposure_hours}{t('sunExposure.statistics.hoursExposure')}
                  </Text>

                  <View style={styles.correlationRow}>
                    <Text style={styles.correlationLabel}>{t('sunExposure.statistics.correlation')}</Text>
                    <View style={[
                      styles.correlationBadge,
                      { backgroundColor: corr.correlation_type === 'strong_positive' ? '#dc3545' :
                          corr.correlation_type === 'moderate_positive' ? '#ffc107' : '#28a745' }
                    ]}>
                      <Text style={styles.correlationBadgeText}>
                        {corr.correlation_type.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                      </Text>
                    </View>
                  </View>

                  <Text style={styles.correlationScore}>
                    {t('sunExposure.statistics.score')} {(corr.correlation_score * 100).toFixed(0)}% • {t('sunExposure.statistics.confidence')} {(corr.correlation_confidence * 100).toFixed(0)}%
                  </Text>

                  {corr.sunburn_events_count > 0 && (
                    <Text style={styles.correlationWarning}>
                      ⚠️ {corr.sunburn_events_count} {t('sunExposure.statistics.sunburnEvents')}
                    </Text>
                  )}

                  <Text style={styles.correlationUrgency}>
                    {t('sunExposure.statistics.screening')} {corr.screening_urgency.toUpperCase()}
                  </Text>
                </View>
              ))}
            </View>
          )}
        </>
      ) : (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateText}>{t('sunExposure.statistics.noData')}</Text>
        </View>
      )}
      <View style={{ height: 20 }} />
    </ScrollView>
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
          <Text style={styles.headerTitle}>☀️ {t('sunExposure.title')}</Text>
          <Text style={styles.headerSubtitle}>{t('sunExposure.subtitle')}</Text>
        </View>
      </View>

      {/* Tabs */}
      <View style={styles.tabBar}>
        <Pressable
          style={[styles.tab, activeTab === 'log' && styles.tabActive]}
          onPress={() => setActiveTab('log')}
        >
          <Text style={[styles.tabText, activeTab === 'log' && styles.tabTextActive]}>
            {t('sunExposure.tabs.logExposure')}
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tab, activeTab === 'history' && styles.tabActive]}
          onPress={() => setActiveTab('history')}
        >
          <Text style={[styles.tabText, activeTab === 'history' && styles.tabTextActive]}>
            {t('sunExposure.tabs.history')}
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tab, activeTab === 'statistics' && styles.tabActive]}
          onPress={() => setActiveTab('statistics')}
        >
          <Text style={[styles.tabText, activeTab === 'statistics' && styles.tabTextActive]}>
            {t('sunExposure.tabs.statistics')}
          </Text>
        </Pressable>
      </View>

      {/* Tab Content */}
      {activeTab === 'log' && renderLogTab()}
      {activeTab === 'history' && renderHistoryTab()}
      {activeTab === 'statistics' && renderStatisticsTab()}
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
  tabBar: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  tab: {
    flex: 1,
    paddingVertical: 16,
    alignItems: 'center',
    borderBottomWidth: 2,
    borderBottomColor: 'transparent',
  },
  tabActive: {
    borderBottomColor: '#4299e1',
  },
  tabText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#718096',
  },
  tabTextActive: {
    color: '#4299e1',
  },
  tabContent: {
    flex: 1,
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  section: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#4a5568',
    marginBottom: 8,
    marginTop: 12,
  },
  input: {
    backgroundColor: '#f7fafc',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  notesInput: {
    height: 100,
    textAlignVertical: 'top',
  },
  buttonGroup: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 8,
  },
  optionButton: {
    backgroundColor: '#f7fafc',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  optionButtonActive: {
    backgroundColor: '#4299e1',
    borderColor: '#4299e1',
  },
  optionButtonText: {
    fontSize: 13,
    color: '#4a5568',
    fontWeight: '500',
  },
  optionButtonTextActive: {
    color: '#fff',
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  switchLabel: {
    fontSize: 14,
    color: '#2d3748',
  },
  bodyAreasGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  bodyAreaButton: {
    backgroundColor: '#f7fafc',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  bodyAreaButtonActive: {
    backgroundColor: '#10b981',
    borderColor: '#10b981',
  },
  bodyAreaText: {
    fontSize: 13,
    color: '#4a5568',
  },
  bodyAreaTextActive: {
    color: '#fff',
  },
  submitButton: {
    backgroundColor: '#4299e1',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 20,
  },
  submitButtonDisabled: {
    opacity: 0.6,
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  loadingContainer: {
    paddingVertical: 40,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#4a5568',
  },
  emptyState: {
    paddingVertical: 60,
    alignItems: 'center',
  },
  emptyStateText: {
    fontSize: 20,
    color: '#4a5568',
    marginBottom: 8,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#718096',
    textAlign: 'center',
  },
  exposureCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  exposureHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  exposureDate: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2d3748',
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  riskBadgeText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: 'bold',
  },
  exposureDetails: {
    gap: 6,
  },
  exposureDetailText: {
    fontSize: 13,
    color: '#4a5568',
  },
  protectionBadge: {
    backgroundColor: '#e6fffa',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
    alignSelf: 'flex-start',
    marginTop: 8,
  },
  protectionBadgeText: {
    fontSize: 12,
    color: '#0f766e',
    fontWeight: '600',
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 16,
  },
  statCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: '#f7fafc',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#4299e1',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#4a5568',
    textAlign: 'center',
  },
  statsInfo: {
    fontSize: 14,
    color: '#2d3748',
    marginBottom: 8,
  },
  recommendationItem: {
    marginBottom: 12,
  },
  recommendationText: {
    fontSize: 14,
    color: '#2d3748',
    lineHeight: 20,
  },
  correlationCard: {
    backgroundColor: '#f7fafc',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#4299e1',
  },
  correlationTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2d3748',
    marginBottom: 4,
  },
  correlationSubtitle: {
    fontSize: 13,
    color: '#64748b',
    marginBottom: 12,
  },
  correlationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  correlationLabel: {
    fontSize: 13,
    color: '#4a5568',
    marginRight: 8,
  },
  correlationBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  correlationBadgeText: {
    fontSize: 11,
    color: '#fff',
    fontWeight: '600',
  },
  correlationScore: {
    fontSize: 12,
    color: '#64748b',
    marginBottom: 8,
  },
  correlationWarning: {
    fontSize: 13,
    color: '#dc2626',
    marginBottom: 8,
  },
  correlationUrgency: {
    fontSize: 13,
    fontWeight: '600',
    color: '#2d3748',
  },
});
