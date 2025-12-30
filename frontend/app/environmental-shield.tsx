import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import * as Location from 'expo-location';
import { API_URL } from '../config';

interface RoutineStep {
  step?: number;
  time?: string;
  action: string;
  reason: string;
}

interface ProductNeeded {
  type: string;
  priority: string;
  reason: string;
}

interface EnvironmentalAlert {
  type: string;
  priority: string;
  title: string;
  message: string;
  icon: string;
  action: string;
  expires_at: string;
}

interface ProtectionPlan {
  morning_routine: RoutineStep[];
  midday_actions: RoutineStep[];
  evening_routine: RoutineStep[];
  products_needed: ProductNeeded[];
  lifestyle_tips: string[];
}

interface EnvironmentalData {
  uv_index: number;
  uv_level: string;
  pollution_aqi: number;
  pollution_level: string;
  humidity: number;
  humidity_impact: string;
  pollen_count: number;
  pollen_level: string;
  temperature: number;
  temperature_impact: string;
  wind_speed: number;
  overall_skin_risk: string;
  alerts: EnvironmentalAlert[];
  protection_plan: ProtectionPlan;
  next_sunscreen_reminder: string;
  timestamp: string;
  location_name?: string;
  weather_description?: string;
}

const SKIN_TYPES = ['normal', 'oily', 'dry', 'combination', 'sensitive'];

export default function EnvironmentalShieldScreen() {
  const router = useRouter();
  const [data, setData] = useState<EnvironmentalData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [skinType, setSkinType] = useState('normal');
  const [conditions, setConditions] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'routine' | 'products'>('overview');
  const [location, setLocation] = useState<{ latitude: number; longitude: number } | null>(null);

  useEffect(() => {
    getLocationAndData();
  }, []);

  const getLocationAndData = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Location needed', 'Please enable location for local environmental data');
        // Use default location
        setLocation({ latitude: 40.7128, longitude: -74.0060 }); // NYC
        fetchEnvironmentalData(40.7128, -74.0060);
        return;
      }

      const loc = await Location.getCurrentPositionAsync({});
      setLocation({
        latitude: loc.coords.latitude,
        longitude: loc.coords.longitude,
      });
      fetchEnvironmentalData(loc.coords.latitude, loc.coords.longitude);
    } catch (error) {
      console.error('Location error:', error);
      // Use default location
      setLocation({ latitude: 40.7128, longitude: -74.0060 });
      fetchEnvironmentalData(40.7128, -74.0060);
    }
  };

  const fetchEnvironmentalData = async (lat: number, lon: number) => {
    try {
      const params = new URLSearchParams({
        latitude: lat.toString(),
        longitude: lon.toString(),
        skin_type: skinType,
        conditions: conditions.join(','),
      });

      const response = await fetch(
        `${API_URL}/api/environmental-shield/status?${params}`
      );

      if (!response.ok) throw new Error('Failed to fetch data');

      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error('Error:', error);
      Alert.alert('Error', 'Failed to fetch environmental data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    if (location) {
      fetchEnvironmentalData(location.latitude, location.longitude);
    } else {
      getLocationAndData();
    }
  };

  const getUVColor = (index: number) => {
    if (index < 3) return '#22C55E';
    if (index < 6) return '#F59E0B';
    if (index < 8) return '#F97316';
    return '#EF4444';
  };

  const getAQIColor = (aqi: number) => {
    if (aqi <= 50) return '#22C55E';
    if (aqi <= 100) return '#F59E0B';
    if (aqi <= 150) return '#F97316';
    return '#EF4444';
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Minimal': return '#22C55E';
      case 'Low': return '#84CC16';
      case 'Moderate': return '#F59E0B';
      case 'High': return '#EF4444';
      default: return '#6B7280';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return '#EF4444';
      case 'high': return '#F97316';
      case 'medium': return '#F59E0B';
      case 'low': return '#22C55E';
      default: return '#6B7280';
    }
  };

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#6366F1" />
          <Text style={styles.loadingText}>Getting your location...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Text style={styles.backButtonText}>←</Text>
          </TouchableOpacity>
          <Text style={styles.title}>🛡️ Environmental Shield</Text>
          <TouchableOpacity onPress={onRefresh}>
            <Text style={styles.refreshIcon}>🔄</Text>
          </TouchableOpacity>
        </View>

        {/* Location Info */}
        {data?.location_name && (
          <View style={styles.locationCard}>
            <Text style={styles.locationIcon}>📍</Text>
            <View style={styles.locationInfo}>
              <Text style={styles.locationName}>{data.location_name}</Text>
              {data.weather_description && (
                <Text style={styles.weatherDesc}>{data.weather_description}</Text>
              )}
            </View>
          </View>
        )}

        {data && (
          <>
            {/* Overall Risk Card */}
            <View style={[
              styles.riskCard,
              { borderColor: getRiskColor(data.overall_skin_risk) }
            ]}>
              <Text style={styles.riskLabel}>Skin Risk Level</Text>
              <Text style={[
                styles.riskValue,
                { color: getRiskColor(data.overall_skin_risk) }
              ]}>
                {data.overall_skin_risk}
              </Text>
              {data.next_sunscreen_reminder !== 'No reminder needed - low UV' && (
                <View style={styles.reminderChip}>
                  <Text style={styles.reminderText}>
                    🧴 Next SPF: {data.next_sunscreen_reminder}
                  </Text>
                </View>
              )}
            </View>

            {/* Environmental Metrics */}
            <View style={styles.metricsGrid}>
              {/* UV Index */}
              <View style={styles.metricCard}>
                <Text style={styles.metricIcon}>☀️</Text>
                <Text style={styles.metricLabel}>UV Index</Text>
                <Text style={[styles.metricValue, { color: getUVColor(data.uv_index) }]}>
                  {data.uv_index}
                </Text>
                <Text style={styles.metricSublabel}>{data.uv_level}</Text>
              </View>

              {/* Air Quality */}
              <View style={styles.metricCard}>
                <Text style={styles.metricIcon}>🏭</Text>
                <Text style={styles.metricLabel}>Air Quality</Text>
                <Text style={[styles.metricValue, { color: getAQIColor(data.pollution_aqi) }]}>
                  {data.pollution_aqi}
                </Text>
                <Text style={styles.metricSublabel}>{data.pollution_level}</Text>
              </View>

              {/* Humidity */}
              <View style={styles.metricCard}>
                <Text style={styles.metricIcon}>💧</Text>
                <Text style={styles.metricLabel}>Humidity</Text>
                <Text style={styles.metricValue}>{data.humidity}%</Text>
                <Text style={styles.metricSublabel}>
                  {data.humidity < 40 ? 'Dry' : data.humidity > 70 ? 'Humid' : 'Normal'}
                </Text>
              </View>

              {/* Pollen */}
              <View style={styles.metricCard}>
                <Text style={styles.metricIcon}>🌸</Text>
                <Text style={styles.metricLabel}>Pollen</Text>
                <Text style={styles.metricValue}>{data.pollen_count}</Text>
                <Text style={styles.metricSublabel}>{data.pollen_level}</Text>
              </View>

              {/* Temperature */}
              <View style={styles.metricCard}>
                <Text style={styles.metricIcon}>🌡️</Text>
                <Text style={styles.metricLabel}>Temp</Text>
                <Text style={styles.metricValue}>{data.temperature}°F</Text>
                <Text style={styles.metricSublabel}>
                  {data.temperature < 50 ? 'Cold' : data.temperature > 85 ? 'Hot' : 'Mild'}
                </Text>
              </View>

              {/* Wind */}
              <View style={styles.metricCard}>
                <Text style={styles.metricIcon}>💨</Text>
                <Text style={styles.metricLabel}>Wind</Text>
                <Text style={styles.metricValue}>{data.wind_speed}</Text>
                <Text style={styles.metricSublabel}>mph</Text>
              </View>
            </View>

            {/* Alerts */}
            {data.alerts.length > 0 && (
              <View style={styles.alertsContainer}>
                <Text style={styles.sectionTitle}>Active Alerts</Text>
                {data.alerts.map((alert, index) => (
                  <View
                    key={index}
                    style={[
                      styles.alertCard,
                      { borderLeftColor: getPriorityColor(alert.priority) }
                    ]}
                  >
                    <View style={styles.alertHeader}>
                      <Text style={styles.alertIcon}>{alert.icon}</Text>
                      <View style={styles.alertTitleContainer}>
                        <Text style={styles.alertTitle}>{alert.title}</Text>
                        <View style={[
                          styles.priorityChip,
                          { backgroundColor: getPriorityColor(alert.priority) + '20' }
                        ]}>
                          <Text style={[
                            styles.priorityText,
                            { color: getPriorityColor(alert.priority) }
                          ]}>
                            {alert.priority.toUpperCase()}
                          </Text>
                        </View>
                      </View>
                    </View>
                    <Text style={styles.alertMessage}>{alert.message}</Text>
                    <View style={styles.alertAction}>
                      <Text style={styles.alertActionText}>→ {alert.action}</Text>
                    </View>
                  </View>
                ))}
              </View>
            )}

            {/* Tab Navigation */}
            <View style={styles.tabContainer}>
              {(['overview', 'routine', 'products'] as const).map(tab => (
                <TouchableOpacity
                  key={tab}
                  style={[styles.tab, activeTab === tab && styles.tabActive]}
                  onPress={() => setActiveTab(tab)}
                >
                  <Text style={[
                    styles.tabText,
                    activeTab === tab && styles.tabTextActive
                  ]}>
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {/* Tab Content */}
            {activeTab === 'overview' && (
              <View style={styles.tabContent}>
                <Text style={styles.sectionTitle}>Environmental Impact</Text>

                <View style={styles.impactCard}>
                  <Text style={styles.impactTitle}>💧 Humidity</Text>
                  <Text style={styles.impactText}>{data.humidity_impact}</Text>
                </View>

                <View style={styles.impactCard}>
                  <Text style={styles.impactTitle}>🌡️ Temperature</Text>
                  <Text style={styles.impactText}>{data.temperature_impact}</Text>
                </View>

                {/* Lifestyle Tips */}
                <Text style={[styles.sectionTitle, { marginTop: 16 }]}>Today's Tips</Text>
                {data.protection_plan.lifestyle_tips.map((tip, index) => (
                  <View key={index} style={styles.tipItem}>
                    <Text style={styles.tipBullet}>💡</Text>
                    <Text style={styles.tipText}>{tip}</Text>
                  </View>
                ))}
              </View>
            )}

            {activeTab === 'routine' && (
              <View style={styles.tabContent}>
                {/* Morning Routine */}
                <Text style={styles.sectionTitle}>🌅 Morning Routine</Text>
                {data.protection_plan.morning_routine.map((step, index) => (
                  <View key={index} style={styles.routineStep}>
                    <View style={styles.stepNumber}>
                      <Text style={styles.stepNumberText}>{step.step || index + 1}</Text>
                    </View>
                    <View style={styles.stepContent}>
                      <Text style={styles.stepAction}>{step.action}</Text>
                      <Text style={styles.stepReason}>{step.reason}</Text>
                    </View>
                  </View>
                ))}

                {/* Midday Actions */}
                {data.protection_plan.midday_actions.length > 0 && (
                  <>
                    <Text style={[styles.sectionTitle, { marginTop: 20 }]}>☀️ Midday Actions</Text>
                    {data.protection_plan.midday_actions.map((action, index) => (
                      <View key={index} style={styles.middayAction}>
                        <Text style={styles.middayTime}>{action.time}</Text>
                        <Text style={styles.middayActionText}>{action.action}</Text>
                        <Text style={styles.middayReason}>{action.reason}</Text>
                      </View>
                    ))}
                  </>
                )}

                {/* Evening Routine */}
                <Text style={[styles.sectionTitle, { marginTop: 20 }]}>🌙 Evening Routine</Text>
                {data.protection_plan.evening_routine.map((step, index) => (
                  <View key={index} style={styles.routineStep}>
                    <View style={styles.stepNumber}>
                      <Text style={styles.stepNumberText}>{step.step || index + 1}</Text>
                    </View>
                    <View style={styles.stepContent}>
                      <Text style={styles.stepAction}>{step.action}</Text>
                      <Text style={styles.stepReason}>{step.reason}</Text>
                    </View>
                  </View>
                ))}
              </View>
            )}

            {activeTab === 'products' && (
              <View style={styles.tabContent}>
                <Text style={styles.sectionTitle}>Products You Need Today</Text>
                {data.protection_plan.products_needed.length === 0 ? (
                  <Text style={styles.noProducts}>
                    No special products needed today. Your regular routine should suffice!
                  </Text>
                ) : (
                  data.protection_plan.products_needed.map((product, index) => (
                    <View key={index} style={styles.productCard}>
                      <View style={styles.productHeader}>
                        <Text style={styles.productType}>{product.type}</Text>
                        <View style={[
                          styles.productPriority,
                          {
                            backgroundColor:
                              product.priority === 'essential' ? '#EF4444' :
                              product.priority === 'high' ? '#F97316' : '#F59E0B'
                          }
                        ]}>
                          <Text style={styles.productPriorityText}>
                            {product.priority.toUpperCase()}
                          </Text>
                        </View>
                      </View>
                      <Text style={styles.productReason}>{product.reason}</Text>
                    </View>
                  ))
                )}
              </View>
            )}

            {/* Skin Type Selector */}
            <View style={styles.skinTypeContainer}>
              <Text style={styles.skinTypeLabel}>Your Skin Type:</Text>
              <View style={styles.skinTypeButtons}>
                {SKIN_TYPES.map(type => (
                  <TouchableOpacity
                    key={type}
                    style={[
                      styles.skinTypeButton,
                      skinType === type && styles.skinTypeButtonActive,
                    ]}
                    onPress={() => {
                      setSkinType(type);
                      if (location) {
                        fetchEnvironmentalData(location.latitude, location.longitude);
                      }
                    }}
                  >
                    <Text style={[
                      styles.skinTypeButtonText,
                      skinType === type && styles.skinTypeButtonTextActive,
                    ]}>
                      {type.charAt(0).toUpperCase() + type.slice(1)}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            <Text style={styles.lastUpdated}>
              Last updated: {new Date(data.timestamp).toLocaleTimeString()}
            </Text>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  scrollContent: {
    padding: 20,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#6B7280',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    fontSize: 24,
    color: '#6366F1',
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  refreshIcon: {
    fontSize: 24,
  },
  locationCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#EEF2FF',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
  },
  locationIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  locationInfo: {
    flex: 1,
  },
  locationName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
  },
  weatherDesc: {
    fontSize: 14,
    color: '#6366F1',
    marginTop: 2,
  },
  riskCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    marginBottom: 16,
    borderWidth: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  riskLabel: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 4,
  },
  riskValue: {
    fontSize: 32,
    fontWeight: 'bold',
  },
  reminderChip: {
    marginTop: 12,
    backgroundColor: '#FEF3C7',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  reminderText: {
    fontSize: 14,
    color: '#92400E',
    fontWeight: '600',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },
  metricCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
    width: '31%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  metricIcon: {
    fontSize: 24,
    marginBottom: 4,
  },
  metricLabel: {
    fontSize: 11,
    color: '#6B7280',
    marginBottom: 2,
  },
  metricValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  metricSublabel: {
    fontSize: 10,
    color: '#9CA3AF',
  },
  alertsContainer: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 12,
  },
  alertCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
    borderLeftWidth: 4,
  },
  alertHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  alertIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  alertTitleContainer: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
    flex: 1,
  },
  priorityChip: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 8,
  },
  priorityText: {
    fontSize: 10,
    fontWeight: 'bold',
  },
  alertMessage: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 8,
    marginLeft: 36,
  },
  alertAction: {
    marginLeft: 36,
    backgroundColor: '#EEF2FF',
    padding: 8,
    borderRadius: 8,
  },
  alertActionText: {
    fontSize: 13,
    color: '#4338CA',
    fontWeight: '500',
  },
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: '#E5E7EB',
    borderRadius: 12,
    padding: 4,
    marginBottom: 16,
  },
  tab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
  },
  tabActive: {
    backgroundColor: '#fff',
  },
  tabText: {
    fontSize: 14,
    color: '#6B7280',
    fontWeight: '500',
  },
  tabTextActive: {
    color: '#6366F1',
    fontWeight: '600',
  },
  tabContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  impactCard: {
    backgroundColor: '#F9FAFB',
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
  },
  impactTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 4,
  },
  impactText: {
    fontSize: 14,
    color: '#6B7280',
    lineHeight: 20,
  },
  tipItem: {
    flexDirection: 'row',
    marginBottom: 10,
    alignItems: 'flex-start',
  },
  tipBullet: {
    fontSize: 16,
    marginRight: 8,
  },
  tipText: {
    flex: 1,
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
  },
  routineStep: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  stepNumber: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#6366F1',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  stepNumberText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  stepContent: {
    flex: 1,
  },
  stepAction: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 2,
  },
  stepReason: {
    fontSize: 13,
    color: '#6B7280',
  },
  middayAction: {
    backgroundColor: '#FEF3C7',
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
  },
  middayTime: {
    fontSize: 12,
    fontWeight: '600',
    color: '#92400E',
    marginBottom: 4,
  },
  middayActionText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#78350F',
    marginBottom: 2,
  },
  middayReason: {
    fontSize: 13,
    color: '#B45309',
  },
  productCard: {
    backgroundColor: '#F9FAFB',
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
  },
  productHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  productType: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
  },
  productPriority: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 8,
  },
  productPriorityText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#fff',
  },
  productReason: {
    fontSize: 13,
    color: '#6B7280',
  },
  noProducts: {
    fontSize: 14,
    color: '#6B7280',
    fontStyle: 'italic',
    textAlign: 'center',
    padding: 20,
  },
  skinTypeContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  skinTypeLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 12,
  },
  skinTypeButtons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  skinTypeButton: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#F3F4F6',
  },
  skinTypeButtonActive: {
    backgroundColor: '#6366F1',
  },
  skinTypeButtonText: {
    fontSize: 13,
    color: '#6B7280',
  },
  skinTypeButtonTextActive: {
    color: '#fff',
  },
  lastUpdated: {
    fontSize: 12,
    color: '#9CA3AF',
    textAlign: 'center',
    marginTop: 8,
  },
});
