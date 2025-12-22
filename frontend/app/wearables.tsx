import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  Alert,
  Switch,
  ActivityIndicator,
  Linking,
  Image,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import config from '../config';
import HealthKitService, { HealthKitStatus } from '../services/HealthKitService';

interface WearableDevice {
  id: number;
  device_type: string;
  device_model: string | null;
  device_name: string;
  is_connected: boolean;
  connection_status: string;
  last_sync_at: string | null;
  has_uv_sensor: boolean;
  total_uv_readings: number;
  auto_sync_enabled: boolean;
  uv_alert_threshold: number;
  connected_at: string | null;
}

interface SupportedDevice {
  type: string;
  name: string;
  logo: string;
  capabilities: {
    uv_sensor: boolean;
    location: boolean;
    activity: boolean;
    heart_rate: boolean;
  };
  oauth_required: boolean;
  notes: string;
}

interface TodayUV {
  date: string;
  has_data: boolean;
  total_outdoor_minutes: number;
  average_uv_index: number | null;
  max_uv_index: number | null;
  risk_category: string;
  daily_risk_score: number | null;
}

export default function WearablesScreen() {
  const router = useRouter();
  const { user } = useAuth();
  const token = user?.token;
  const [devices, setDevices] = useState<WearableDevice[]>([]);
  const [supportedDevices, setSupportedDevices] = useState<SupportedDevice[]>([]);
  const [todayUV, setTodayUV] = useState<TodayUV | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [syncing, setSyncing] = useState<number | null>(null);
  const [showAddDevice, setShowAddDevice] = useState(false);
  const [healthKitStatus, setHealthKitStatus] = useState<HealthKitStatus | null>(null);
  const [connectingHealthKit, setConnectingHealthKit] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      // Fetch supported devices (doesn't require auth)
      const supportedRes = await fetch(`${config.API_URL}/wearables/supported-devices`);
      if (supportedRes.ok) {
        const data = await supportedRes.json();
        console.log('Supported devices loaded:', data.devices?.length);
        setSupportedDevices(data.devices || []);
      } else {
        console.error('Failed to fetch supported devices:', supportedRes.status);
      }

      // Only fetch user-specific data if authenticated
      if (token) {
        const [devicesRes, uvRes] = await Promise.all([
          fetch(`${config.API_URL}/wearables/devices`, {
            headers: { Authorization: `Bearer ${token}` },
          }),
          fetch(`${config.API_URL}/wearables/uv-exposure/today`, {
            headers: { Authorization: `Bearer ${token}` },
          }),
        ]);

        if (devicesRes.ok) {
          const data = await devicesRes.json();
          setDevices(data.devices || []);
        }

        if (uvRes.ok) {
          const data = await uvRes.json();
          setTodayUV(data);
        }
      }
    } catch (error) {
      console.error('Error fetching wearable data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [token]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Check HealthKit availability on mount (iOS only)
  useEffect(() => {
    const checkHealthKit = async () => {
      if (Platform.OS === 'ios') {
        const status = await HealthKitService.checkAvailability();
        setHealthKitStatus(status);
        console.log('[Wearables] HealthKit status:', status);
      }
    };
    checkHealthKit();
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    fetchData();
  };

  // Connect to Apple Health using HealthKit
  const connectAppleHealth = async () => {
    if (!token) {
      Alert.alert(
        'Login Required',
        'Please log in to connect Apple Health.',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Log In', onPress: () => router.push('/') },
        ]
      );
      return;
    }

    if (Platform.OS !== 'ios') {
      Alert.alert('Not Available', 'Apple Health is only available on iOS devices.');
      return;
    }

    setConnectingHealthKit(true);

    try {
      // Check if HealthKit is available
      const status = await HealthKitService.checkAvailability();

      if (!status.isAvailable) {
        if (status.needsRebuild) {
          Alert.alert(
            'Rebuild Required',
            'To use Apple Health, you need to rebuild the app with HealthKit support.\n\nRun this command:\neas build --profile development --platform ios\n\nThen install the new build on your iPhone.',
            [{ text: 'OK' }]
          );
        } else {
          Alert.alert(
            'HealthKit Not Available',
            status.error || 'HealthKit is not available on this device.',
            [{ text: 'OK' }]
          );
        }
        setConnectingHealthKit(false);
        return;
      }

      // Request authorization
      const authorized = await HealthKitService.requestAuthorization();

      if (!authorized) {
        Alert.alert(
          'Authorization Denied',
          'Please grant access to Health data in Settings > Privacy > Health > SkinLesionDetection',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Open Settings', onPress: () => Linking.openSettings() },
          ]
        );
        setConnectingHealthKit(false);
        return;
      }

      // Register the device with the backend
      const response = await fetch(`${config.API_URL}/wearables/connect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          device_type: 'apple_health',
          device_name: 'Apple Health',
          connection_type: 'healthkit_direct',
        }),
      });

      if (response.ok) {
        // Sync initial data
        const syncResult = await HealthKitService.syncToBackend(config.API_URL, token, 7);

        Alert.alert(
          'Apple Health Connected!',
          syncResult.success
            ? `Successfully connected and synced data: ${syncResult.message}`
            : 'Connected, but initial sync failed. You can sync manually later.',
          [{ text: 'OK' }]
        );

        setShowAddDevice(false);
        setHealthKitStatus({ ...status, isAuthorized: true });
        fetchData();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to register device');
      }
    } catch (error: any) {
      console.error('[Wearables] HealthKit connection error:', error);
      Alert.alert('Error', error.message || 'Failed to connect to Apple Health');
    } finally {
      setConnectingHealthKit(false);
    }
  };

  // Sync HealthKit data manually
  const syncHealthKitData = async (deviceId: number) => {
    if (!token || Platform.OS !== 'ios') return;

    setSyncing(deviceId);

    try {
      const result = await HealthKitService.syncToBackend(config.API_URL, token, 7);

      if (result.success) {
        Alert.alert('Sync Complete', result.message);
        fetchData();
      } else {
        Alert.alert('Sync Failed', result.message);
      }
    } catch (error: any) {
      Alert.alert('Error', error.message || 'Failed to sync health data');
    } finally {
      setSyncing(null);
    }
  };

  const connectDevice = async (deviceType: string) => {
    // Handle Apple Health specially
    if (deviceType === 'apple_health' || deviceType === 'apple_watch') {
      await connectAppleHealth();
      return;
    }

    if (!token) {
      Alert.alert(
        'Login Required',
        'Please log in to connect a wearable device.',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Log In', onPress: () => router.push('/') },
        ]
      );
      return;
    }

    // For other devices, show OAuth not implemented message
    Alert.alert(
      'Coming Soon',
      `${deviceType.replace('_', ' ')} integration requires OAuth setup. Currently, only Apple Health is supported with direct integration.`,
      [{ text: 'OK' }]
    );
  };

  const disconnectDevice = async (deviceId: number, deviceName: string) => {
    Alert.alert(
      'Disconnect Device',
      `Are you sure you want to disconnect ${deviceName}? Your UV data will be preserved.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Disconnect',
          style: 'destructive',
          onPress: async () => {
            try {
              const response = await fetch(`${config.API_URL}/wearables/devices/${deviceId}`, {
                method: 'DELETE',
                headers: { Authorization: `Bearer ${token}` },
              });

              if (response.ok) {
                Alert.alert('Success', 'Device disconnected');
                fetchData();
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to disconnect device');
            }
          },
        },
      ]
    );
  };

  const syncDevice = async (deviceId: number, deviceType?: string) => {
    if (!token) return;

    // Use HealthKit for Apple Health devices
    if (deviceType === 'apple_health' && Platform.OS === 'ios') {
      await syncHealthKitData(deviceId);
      return;
    }

    setSyncing(deviceId);
    try {
      const response = await fetch(`${config.API_URL}/wearables/devices/${deviceId}/sync`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });

      if (response.ok) {
        Alert.alert('Sync Started', 'Your device is syncing in the background');
        setTimeout(fetchData, 3000);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to sync device');
    } finally {
      setSyncing(null);
    }
  };

  const toggleAutoSync = async (deviceId: number, currentValue: boolean) => {
    try {
      await fetch(`${config.API_URL}/wearables/devices/${deviceId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ auto_sync_enabled: !currentValue }),
      });
      fetchData();
    } catch (error) {
      console.error('Error toggling auto sync:', error);
    }
  };

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'apple_watch':
        return 'watch-outline';
      case 'fitbit':
        return 'fitness-outline';
      case 'garmin':
        return 'navigate-outline';
      case 'samsung':
        return 'watch';
      default:
        return 'watch-outline';
    }
  };

  const getRiskColor = (category: string) => {
    switch (category) {
      case 'low':
        return '#10b981';
      case 'moderate':
        return '#f59e0b';
      case 'high':
        return '#f97316';
      case 'very_high':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const renderTodayUVCard = () => {
    if (!todayUV) return null;

    return (
      <View style={styles.uvCard}>
        <View style={styles.uvCardHeader}>
          <View style={styles.uvIconContainer}>
            <Ionicons name="sunny" size={28} color="#f59e0b" />
          </View>
          <View style={styles.uvCardTitle}>
            <Text style={styles.uvCardTitleText}>Today's UV Exposure</Text>
            <Text style={styles.uvCardDate}>{todayUV.date}</Text>
          </View>
        </View>

        {todayUV.has_data ? (
          <>
            <View style={styles.uvStats}>
              <View style={styles.uvStatItem}>
                <Text style={styles.uvStatValue}>{todayUV.total_outdoor_minutes}</Text>
                <Text style={styles.uvStatLabel}>Minutes Outdoors</Text>
              </View>
              <View style={styles.uvStatDivider} />
              <View style={styles.uvStatItem}>
                <Text style={styles.uvStatValue}>
                  {todayUV.average_uv_index?.toFixed(1) || '-'}
                </Text>
                <Text style={styles.uvStatLabel}>Avg UV Index</Text>
              </View>
              <View style={styles.uvStatDivider} />
              <View style={styles.uvStatItem}>
                <Text style={styles.uvStatValue}>{todayUV.max_uv_index || '-'}</Text>
                <Text style={styles.uvStatLabel}>Max UV Index</Text>
              </View>
            </View>

            <View style={[styles.riskBadge, { backgroundColor: getRiskColor(todayUV.risk_category) + '20' }]}>
              <View style={[styles.riskDot, { backgroundColor: getRiskColor(todayUV.risk_category) }]} />
              <Text style={[styles.riskText, { color: getRiskColor(todayUV.risk_category) }]}>
                {todayUV.risk_category.replace('_', ' ').toUpperCase()} RISK
              </Text>
              {todayUV.daily_risk_score && (
                <Text style={styles.riskScore}>Score: {todayUV.daily_risk_score.toFixed(0)}</Text>
              )}
            </View>

            <TouchableOpacity
              style={styles.viewDetailsButton}
              onPress={() => router.push('/uv-dashboard')}
            >
              <Text style={styles.viewDetailsText}>View Full Dashboard</Text>
              <Ionicons name="chevron-forward" size={16} color="#8b5cf6" />
            </TouchableOpacity>
          </>
        ) : (
          <View style={styles.noDataContainer}>
            <Ionicons name="cloud-offline-outline" size={40} color="#9ca3af" />
            <Text style={styles.noDataText}>No UV data recorded today</Text>
            <Text style={styles.noDataSubtext}>Sync your wearable device to start tracking</Text>
          </View>
        )}
      </View>
    );
  };

  const renderDeviceCard = (device: WearableDevice) => (
    <View key={device.id} style={styles.deviceCard}>
      <View style={styles.deviceHeader}>
        <View style={[styles.deviceIconContainer, { backgroundColor: device.is_connected ? '#dcfce7' : '#f3f4f6' }]}>
          <Ionicons
            name={getDeviceIcon(device.device_type) as any}
            size={24}
            color={device.is_connected ? '#10b981' : '#9ca3af'}
          />
        </View>
        <View style={styles.deviceInfo}>
          <Text style={styles.deviceName}>{device.device_name}</Text>
          <Text style={styles.deviceType}>
            {device.device_type.replace('_', ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
          </Text>
        </View>
        <View style={[styles.statusBadge, { backgroundColor: device.is_connected ? '#dcfce7' : '#fef2f2' }]}>
          <View style={[styles.statusDot, { backgroundColor: device.is_connected ? '#10b981' : '#ef4444' }]} />
          <Text style={[styles.statusText, { color: device.is_connected ? '#10b981' : '#ef4444' }]}>
            {device.is_connected ? 'Connected' : 'Disconnected'}
          </Text>
        </View>
      </View>

      <View style={styles.deviceStats}>
        <View style={styles.deviceStatItem}>
          <Ionicons name="analytics-outline" size={16} color="#6b7280" />
          <Text style={styles.deviceStatText}>{device.total_uv_readings} readings</Text>
        </View>
        {device.has_uv_sensor && (
          <View style={styles.deviceStatItem}>
            <Ionicons name="sunny-outline" size={16} color="#f59e0b" />
            <Text style={styles.deviceStatText}>UV Sensor</Text>
          </View>
        )}
        {device.last_sync_at && (
          <View style={styles.deviceStatItem}>
            <Ionicons name="sync-outline" size={16} color="#6b7280" />
            <Text style={styles.deviceStatText}>
              Last sync: {new Date(device.last_sync_at).toLocaleTimeString()}
            </Text>
          </View>
        )}
      </View>

      <View style={styles.deviceSettings}>
        <View style={styles.settingRow}>
          <Text style={styles.settingLabel}>Auto-sync</Text>
          <Switch
            value={device.auto_sync_enabled}
            onValueChange={() => toggleAutoSync(device.id, device.auto_sync_enabled)}
            trackColor={{ false: '#d1d5db', true: '#a78bfa' }}
            thumbColor={device.auto_sync_enabled ? '#8b5cf6' : '#f4f3f4'}
          />
        </View>
        <View style={styles.settingRow}>
          <Text style={styles.settingLabel}>UV Alert Threshold</Text>
          <Text style={styles.settingValue}>UV {device.uv_alert_threshold}</Text>
        </View>
      </View>

      <View style={styles.deviceActions}>
        <TouchableOpacity
          style={[styles.actionButton, styles.syncButton]}
          onPress={() => syncDevice(device.id, device.device_type)}
          disabled={syncing === device.id || !device.is_connected}
        >
          {syncing === device.id ? (
            <ActivityIndicator size="small" color="#8b5cf6" />
          ) : (
            <>
              <Ionicons name="sync-outline" size={18} color="#8b5cf6" />
              <Text style={styles.syncButtonText}>Sync Now</Text>
            </>
          )}
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.actionButton, styles.disconnectButton]}
          onPress={() => disconnectDevice(device.id, device.device_name)}
        >
          <Ionicons name="unlink-outline" size={18} color="#ef4444" />
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderAddDeviceModal = () => (
    <View style={styles.addDeviceModal}>
      <View style={styles.modalHeader}>
        <Text style={styles.modalTitle}>Connect a Wearable</Text>
        <TouchableOpacity onPress={() => setShowAddDevice(false)}>
          <Ionicons name="close" size={24} color="#6b7280" />
        </TouchableOpacity>
      </View>

      <Text style={styles.modalSubtitle}>
        Connect your wearable device to automatically track UV exposure
      </Text>

      {supportedDevices.length === 0 && (
        <View style={{ padding: 20, alignItems: 'center' }}>
          <ActivityIndicator size="large" color="#8b5cf6" />
          <Text style={{ marginTop: 10, color: '#6b7280' }}>Loading devices...</Text>
          <Text style={{ marginTop: 5, color: '#9ca3af', fontSize: 12 }}>
            If this takes too long, check if backend is running
          </Text>
        </View>
      )}

      <ScrollView style={styles.deviceList}>
        {supportedDevices.map((device) => (
          <TouchableOpacity
            key={device.type}
            style={styles.supportedDeviceCard}
            onPress={() => connectDevice(device.type)}
          >
            <View style={styles.supportedDeviceIcon}>
              <Ionicons name={getDeviceIcon(device.type) as any} size={32} color="#8b5cf6" />
            </View>
            <View style={styles.supportedDeviceInfo}>
              <Text style={styles.supportedDeviceName}>{device.name}</Text>
              <Text style={styles.supportedDeviceNotes}>{device.notes}</Text>
              <View style={styles.capabilitiesTags}>
                {device.capabilities.uv_sensor && (
                  <View style={styles.capabilityTag}>
                    <Ionicons name="sunny" size={12} color="#f59e0b" />
                    <Text style={styles.capabilityText}>UV Sensor</Text>
                  </View>
                )}
                {device.capabilities.activity && (
                  <View style={styles.capabilityTag}>
                    <Ionicons name="footsteps" size={12} color="#10b981" />
                    <Text style={styles.capabilityText}>Activity</Text>
                  </View>
                )}
                {device.capabilities.location && (
                  <View style={styles.capabilityTag}>
                    <Ionicons name="location" size={12} color="#3b82f6" />
                    <Text style={styles.capabilityText}>GPS</Text>
                  </View>
                )}
              </View>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#9ca3af" />
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#8b5cf6" />
        <Text style={styles.loadingText}>Loading wearables...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#1f2937" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Wearable Devices</Text>
        <TouchableOpacity onPress={() => setShowAddDevice(true)} style={styles.addButton}>
          <Ionicons name="add" size={24} color="#8b5cf6" />
        </TouchableOpacity>
      </View>

      {showAddDevice ? (
        renderAddDeviceModal()
      ) : (
        <ScrollView
          style={styles.content}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={['#8b5cf6']} />
          }
        >
          {renderTodayUVCard()}

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Connected Devices</Text>
            {devices.length > 0 ? (
              devices.map(renderDeviceCard)
            ) : (
              <View style={styles.emptyState}>
                <Ionicons name="watch-outline" size={48} color="#d1d5db" />
                <Text style={styles.emptyTitle}>No Devices Connected</Text>
                <Text style={styles.emptyText}>
                  Connect your Apple Watch, Fitbit, or Garmin to track UV exposure automatically
                </Text>
                <TouchableOpacity
                  style={styles.connectButton}
                  onPress={() => setShowAddDevice(true)}
                >
                  <Ionicons name="add-circle-outline" size={20} color="#fff" />
                  <Text style={styles.connectButtonText}>Connect Device</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>

          <View style={styles.infoSection}>
            <Ionicons name="information-circle-outline" size={20} color="#6b7280" />
            <Text style={styles.infoText}>
              UV data is used to analyze correlations between sun exposure and changes in your tracked lesions.
              This helps identify potential risk factors and personalize sun protection recommendations.
            </Text>
          </View>
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#6b7280',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  backButton: {
    padding: 4,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  addButton: {
    padding: 4,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  uvCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  uvCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  uvIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#fef3c7',
    justifyContent: 'center',
    alignItems: 'center',
  },
  uvCardTitle: {
    marginLeft: 12,
  },
  uvCardTitleText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  uvCardDate: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  uvStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 16,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  uvStatItem: {
    alignItems: 'center',
  },
  uvStatValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  uvStatLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  uvStatDivider: {
    width: 1,
    backgroundColor: '#e5e7eb',
  },
  riskBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    marginTop: 16,
  },
  riskDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  riskText: {
    fontSize: 14,
    fontWeight: '600',
  },
  riskScore: {
    fontSize: 12,
    color: '#6b7280',
    marginLeft: 12,
  },
  viewDetailsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 16,
    paddingVertical: 12,
    backgroundColor: '#f5f3ff',
    borderRadius: 8,
  },
  viewDetailsText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#8b5cf6',
    marginRight: 4,
  },
  noDataContainer: {
    alignItems: 'center',
    paddingVertical: 24,
  },
  noDataText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#4b5563',
    marginTop: 12,
  },
  noDataSubtext: {
    fontSize: 14,
    color: '#9ca3af',
    marginTop: 4,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  deviceCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 1,
  },
  deviceHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  deviceIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  deviceInfo: {
    flex: 1,
    marginLeft: 12,
  },
  deviceName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  deviceType: {
    fontSize: 13,
    color: '#6b7280',
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 12,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 4,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '500',
  },
  deviceStats: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  deviceStatItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  deviceStatText: {
    fontSize: 13,
    color: '#6b7280',
  },
  deviceSettings: {
    paddingTop: 12,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 8,
  },
  settingLabel: {
    fontSize: 14,
    color: '#4b5563',
  },
  settingValue: {
    fontSize: 14,
    color: '#6b7280',
  },
  deviceActions: {
    flexDirection: 'row',
    marginTop: 12,
    gap: 8,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    gap: 6,
  },
  syncButton: {
    flex: 1,
    backgroundColor: '#f5f3ff',
  },
  syncButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#8b5cf6',
  },
  disconnectButton: {
    backgroundColor: '#fef2f2',
    paddingHorizontal: 12,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 40,
    backgroundColor: '#fff',
    borderRadius: 12,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginTop: 8,
    marginHorizontal: 32,
  },
  connectButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#8b5cf6',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    marginTop: 20,
    gap: 8,
  },
  connectButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  infoSection: {
    flexDirection: 'row',
    backgroundColor: '#f0f9ff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
    gap: 12,
  },
  infoText: {
    flex: 1,
    fontSize: 13,
    color: '#1e40af',
    lineHeight: 20,
  },
  addDeviceModal: {
    flex: 1,
    padding: 16,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  modalSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 20,
  },
  deviceList: {
    flex: 1,
  },
  supportedDeviceCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 1,
  },
  supportedDeviceIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#f5f3ff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  supportedDeviceInfo: {
    flex: 1,
    marginLeft: 12,
  },
  supportedDeviceName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  supportedDeviceNotes: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  capabilitiesTags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 8,
  },
  capabilityTag: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
    paddingVertical: 2,
    paddingHorizontal: 6,
    borderRadius: 4,
    gap: 4,
  },
  capabilityText: {
    fontSize: 10,
    color: '#4b5563',
  },
});
