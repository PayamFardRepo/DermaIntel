import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
  RefreshControl,
  Modal,
  Switch,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../config';

interface Provider {
  provider_id: string;
  provider_name: string;
  location: string;
}

interface DayAvailability {
  day_of_week: number;
  day_name: string;
  start_time: string;
  end_time: string;
  break_start: string | null;
  break_end: string | null;
  location: string;
  is_telemedicine_day: boolean;
}

interface BlockedTime {
  id: number;
  start: string;
  end: string;
  reason: string;
  notes: string | null;
}

const DAYS_OF_WEEK = [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday',
];

const BLOCK_REASONS = [
  { value: 'vacation', label: 'Vacation', icon: 'airplane', color: '#4CAF50' },
  { value: 'conference', label: 'Conference', icon: 'school', color: '#2196F3' },
  { value: 'personal', label: 'Personal', icon: 'person', color: '#9C27B0' },
  { value: 'emergency', label: 'Emergency', icon: 'alert-circle', color: '#F44336' },
  { value: 'holiday', label: 'Holiday', icon: 'calendar', color: '#FF9800' },
  { value: 'training', label: 'Training', icon: 'book', color: '#00BCD4' },
  { value: 'other', label: 'Other', icon: 'ellipsis-horizontal', color: '#607D8B' },
];

export default function ProviderAvailabilityScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'schedule' | 'blocked'>('schedule');
  const [providers, setProviders] = useState<Provider[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<Provider | null>(null);
  const [availability, setAvailability] = useState<DayAvailability[]>([]);
  const [blockedTimes, setBlockedTimes] = useState<BlockedTime[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Provider selector modal
  const [showProviderModal, setShowProviderModal] = useState(false);

  // Edit day modal
  const [showEditDayModal, setShowEditDayModal] = useState(false);
  const [editingDay, setEditingDay] = useState<number | null>(null);
  const [startTime, setStartTime] = useState('09:00');
  const [endTime, setEndTime] = useState('17:00');
  const [breakStart, setBreakStart] = useState('');
  const [breakEnd, setBreakEnd] = useState('');
  const [location, setLocation] = useState('');
  const [isTelemedicineDay, setIsTelemedicineDay] = useState(false);
  const [saving, setSaving] = useState(false);

  // Block time modal
  const [showBlockModal, setShowBlockModal] = useState(false);
  const [blockStartDate, setBlockStartDate] = useState('');
  const [blockStartTime, setBlockStartTime] = useState('09:00');
  const [blockEndDate, setBlockEndDate] = useState('');
  const [blockEndTime, setBlockEndTime] = useState('17:00');
  const [blockReason, setBlockReason] = useState('vacation');
  const [blockNotes, setBlockNotes] = useState('');

  useEffect(() => {
    fetchProviders();
  }, []);

  useEffect(() => {
    if (selectedProvider) {
      fetchProviderAvailability(selectedProvider.provider_id);
    }
  }, [selectedProvider]);

  const fetchProviders = async () => {
    try {
      const token = await AsyncStorage.getItem('userToken');
      const response = await fetch(`${API_URL}/providers`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setProviders(data.providers || []);
        if (data.providers && data.providers.length > 0) {
          setSelectedProvider(data.providers[0]);
        }
      }
    } catch (error) {
      console.error('Error fetching providers:', error);
      Alert.alert('Error', 'Failed to fetch providers');
    } finally {
      setLoading(false);
    }
  };

  const fetchProviderAvailability = async (providerId: string) => {
    setRefreshing(true);
    try {
      const token = await AsyncStorage.getItem('userToken');
      const response = await fetch(`${API_URL}/providers/${providerId}/availability`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setAvailability(data.schedule || []);
        setBlockedTimes(data.blocked_times || []);
      }
    } catch (error) {
      console.error('Error fetching availability:', error);
      Alert.alert('Error', 'Failed to fetch availability');
    } finally {
      setRefreshing(false);
    }
  };

  const onRefresh = useCallback(() => {
    if (selectedProvider) {
      fetchProviderAvailability(selectedProvider.provider_id);
    }
  }, [selectedProvider]);

  const openEditDayModal = (dayIndex: number) => {
    const existing = availability.find(a => a.day_of_week === dayIndex);
    setEditingDay(dayIndex);

    if (existing) {
      setStartTime(existing.start_time || '09:00');
      setEndTime(existing.end_time || '17:00');
      setBreakStart(existing.break_start || '');
      setBreakEnd(existing.break_end || '');
      setLocation(existing.location || '');
      setIsTelemedicineDay(existing.is_telemedicine_day || false);
    } else {
      setStartTime('09:00');
      setEndTime('17:00');
      setBreakStart('');
      setBreakEnd('');
      setLocation('');
      setIsTelemedicineDay(false);
    }

    setShowEditDayModal(true);
  };

  const handleSaveDay = async () => {
    if (!selectedProvider || editingDay === null) return;

    if (!startTime || !endTime) {
      Alert.alert('Required Fields', 'Please enter start and end times');
      return;
    }

    setSaving(true);
    try {
      const token = await AsyncStorage.getItem('userToken');
      const response = await fetch(
        `${API_URL}/providers/${selectedProvider.provider_id}/availability`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            day_of_week: editingDay,
            start_time: startTime,
            end_time: endTime,
            break_start: breakStart || null,
            break_end: breakEnd || null,
            location: location || null,
            is_telemedicine_day: isTelemedicineDay,
            provider_name: selectedProvider.provider_name,
          }),
        }
      );

      if (response.ok) {
        Alert.alert('Success', `Availability set for ${DAYS_OF_WEEK[editingDay]}`);
        setShowEditDayModal(false);
        fetchProviderAvailability(selectedProvider.provider_id);
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to save availability');
      }
    } catch (error) {
      console.error('Error saving availability:', error);
      Alert.alert('Error', 'Failed to save availability');
    } finally {
      setSaving(false);
    }
  };

  const openBlockModal = () => {
    const today = new Date();
    const todayStr = today.toISOString().split('T')[0];
    setBlockStartDate(todayStr);
    setBlockEndDate(todayStr);
    setBlockStartTime('09:00');
    setBlockEndTime('17:00');
    setBlockReason('vacation');
    setBlockNotes('');
    setShowBlockModal(true);
  };

  const handleBlockTime = async () => {
    if (!selectedProvider) return;

    if (!blockStartDate || !blockEndDate) {
      Alert.alert('Required Fields', 'Please select start and end dates');
      return;
    }

    setSaving(true);
    try {
      const token = await AsyncStorage.getItem('userToken');
      const startDatetime = `${blockStartDate}T${blockStartTime}:00`;
      const endDatetime = `${blockEndDate}T${blockEndTime}:00`;

      const response = await fetch(
        `${API_URL}/providers/${selectedProvider.provider_id}/block-time`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            start_datetime: startDatetime,
            end_datetime: endDatetime,
            reason: blockReason,
            notes: blockNotes || null,
          }),
        }
      );

      if (response.ok) {
        Alert.alert('Success', 'Time blocked successfully');
        setShowBlockModal(false);
        fetchProviderAvailability(selectedProvider.provider_id);
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to block time');
      }
    } catch (error) {
      console.error('Error blocking time:', error);
      Alert.alert('Error', 'Failed to block time');
    } finally {
      setSaving(false);
    }
  };

  const handleUnblockTime = async (blockId: number) => {
    if (!selectedProvider) return;

    Alert.alert(
      'Remove Block',
      'Are you sure you want to remove this blocked time?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Remove',
          style: 'destructive',
          onPress: async () => {
            try {
              const token = await AsyncStorage.getItem('userToken');
              const response = await fetch(
                `${API_URL}/providers/${selectedProvider.provider_id}/block-time/${blockId}`,
                {
                  method: 'DELETE',
                  headers: {
                    'Authorization': `Bearer ${token}`,
                  },
                }
              );

              if (response.ok) {
                Alert.alert('Success', 'Block removed successfully');
                fetchProviderAvailability(selectedProvider.provider_id);
              } else {
                const error = await response.json();
                Alert.alert('Error', error.detail || 'Failed to remove block');
              }
            } catch (error) {
              console.error('Error removing block:', error);
              Alert.alert('Error', 'Failed to remove block');
            }
          },
        },
      ]
    );
  };

  const formatTime = (timeStr: string) => {
    if (!timeStr) return '';
    const [hours, minutes] = timeStr.split(':');
    const hour = parseInt(hours);
    const ampm = hour >= 12 ? 'PM' : 'AM';
    const hour12 = hour % 12 || 12;
    return `${hour12}:${minutes} ${ampm}`;
  };

  const formatDateTime = (isoStr: string) => {
    const date = new Date(isoStr);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  };

  const getBlockReasonInfo = (reason: string) => {
    return BLOCK_REASONS.find(r => r.value === reason) || BLOCK_REASONS[6];
  };

  const getDayAvailability = (dayIndex: number) => {
    return availability.find(a => a.day_of_week === dayIndex);
  };

  const renderScheduleTab = () => (
    <View style={styles.tabContent}>
      <Text style={styles.sectionTitle}>Weekly Schedule</Text>
      <Text style={styles.sectionSubtitle}>
        Configure regular working hours for each day
      </Text>

      {DAYS_OF_WEEK.map((day, index) => {
        const dayAvail = getDayAvailability(index);
        const isConfigured = !!dayAvail;

        return (
          <TouchableOpacity
            key={day}
            style={[styles.dayCard, !isConfigured && styles.dayCardInactive]}
            onPress={() => openEditDayModal(index)}
          >
            <View style={styles.dayHeader}>
              <View style={styles.dayNameContainer}>
                <View
                  style={[
                    styles.dayIndicator,
                    { backgroundColor: isConfigured ? '#4CAF50' : '#ccc' },
                  ]}
                />
                <Text style={styles.dayName}>{day}</Text>
              </View>
              <Ionicons
                name="chevron-forward"
                size={20}
                color="#999"
              />
            </View>

            {isConfigured ? (
              <View style={styles.dayDetails}>
                <View style={styles.timeRow}>
                  <Ionicons name="time-outline" size={16} color="#667eea" />
                  <Text style={styles.timeText}>
                    {formatTime(dayAvail.start_time)} - {formatTime(dayAvail.end_time)}
                  </Text>
                </View>

                {dayAvail.break_start && dayAvail.break_end && (
                  <View style={styles.timeRow}>
                    <Ionicons name="cafe-outline" size={16} color="#FF9800" />
                    <Text style={styles.breakText}>
                      Break: {formatTime(dayAvail.break_start)} - {formatTime(dayAvail.break_end)}
                    </Text>
                  </View>
                )}

                <View style={styles.tagsRow}>
                  {dayAvail.location && (
                    <View style={styles.locationTag}>
                      <Ionicons name="location-outline" size={12} color="#667eea" />
                      <Text style={styles.locationTagText}>{dayAvail.location}</Text>
                    </View>
                  )}
                  {dayAvail.is_telemedicine_day && (
                    <View style={styles.teleTag}>
                      <Ionicons name="videocam-outline" size={12} color="#fff" />
                      <Text style={styles.teleTagText}>Telemedicine</Text>
                    </View>
                  )}
                </View>
              </View>
            ) : (
              <Text style={styles.notConfiguredText}>Not configured - tap to set hours</Text>
            )}
          </TouchableOpacity>
        );
      })}
    </View>
  );

  const renderBlockedTab = () => (
    <View style={styles.tabContent}>
      <View style={styles.blockedHeader}>
        <View>
          <Text style={styles.sectionTitle}>Blocked Times</Text>
          <Text style={styles.sectionSubtitle}>
            Vacation, conferences, and other unavailable periods
          </Text>
        </View>
        <TouchableOpacity style={styles.addBlockButton} onPress={openBlockModal}>
          <Ionicons name="add" size={20} color="#fff" />
          <Text style={styles.addBlockButtonText}>Block Time</Text>
        </TouchableOpacity>
      </View>

      {blockedTimes.length === 0 ? (
        <View style={styles.emptyBlocked}>
          <Ionicons name="calendar-outline" size={48} color="#ccc" />
          <Text style={styles.emptyBlockedText}>No blocked times</Text>
          <Text style={styles.emptyBlockedSubtext}>
            Add time off for vacations, conferences, etc.
          </Text>
        </View>
      ) : (
        blockedTimes.map((block) => {
          const reasonInfo = getBlockReasonInfo(block.reason);
          return (
            <View key={block.id} style={styles.blockedCard}>
              <View style={styles.blockedLeft}>
                <View style={[styles.blockedIcon, { backgroundColor: reasonInfo.color }]}>
                  <Ionicons name={reasonInfo.icon as any} size={20} color="#fff" />
                </View>
                <View style={styles.blockedInfo}>
                  <Text style={styles.blockedReason}>{reasonInfo.label}</Text>
                  <Text style={styles.blockedDates}>
                    {formatDateTime(block.start)}
                  </Text>
                  <Text style={styles.blockedDates}>
                    to {formatDateTime(block.end)}
                  </Text>
                  {block.notes && (
                    <Text style={styles.blockedNotes}>{block.notes}</Text>
                  )}
                </View>
              </View>
              <TouchableOpacity
                style={styles.removeBlockButton}
                onPress={() => handleUnblockTime(block.id)}
              >
                <Ionicons name="trash-outline" size={18} color="#F44336" />
              </TouchableOpacity>
            </View>
          );
        })
      )}
    </View>
  );

  const renderProviderModal = () => (
    <Modal visible={showProviderModal} animationType="slide" transparent>
      <View style={styles.modalOverlay}>
        <View style={styles.providerModalContent}>
          <View style={styles.providerModalHeader}>
            <Text style={styles.providerModalTitle}>Select Provider</Text>
            <TouchableOpacity onPress={() => setShowProviderModal(false)}>
              <Ionicons name="close" size={24} color="#333" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.providerList}>
            {providers.map((provider) => (
              <TouchableOpacity
                key={provider.provider_id}
                style={[
                  styles.providerItem,
                  selectedProvider?.provider_id === provider.provider_id &&
                    styles.providerItemSelected,
                ]}
                onPress={() => {
                  setSelectedProvider(provider);
                  setShowProviderModal(false);
                }}
              >
                <View style={styles.providerAvatar}>
                  <Ionicons name="person" size={24} color="#667eea" />
                </View>
                <View style={styles.providerInfo}>
                  <Text style={styles.providerName}>{provider.provider_name}</Text>
                  {provider.location && (
                    <Text style={styles.providerLocation}>{provider.location}</Text>
                  )}
                </View>
                {selectedProvider?.provider_id === provider.provider_id && (
                  <Ionicons name="checkmark-circle" size={24} color="#4CAF50" />
                )}
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );

  const renderEditDayModal = () => (
    <Modal visible={showEditDayModal} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setShowEditDayModal(false)}>
            <Text style={styles.modalCancel}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.modalTitle}>
            {editingDay !== null ? DAYS_OF_WEEK[editingDay] : 'Edit Day'}
          </Text>
          <TouchableOpacity onPress={handleSaveDay} disabled={saving}>
            {saving ? (
              <ActivityIndicator size="small" color="#667eea" />
            ) : (
              <Text style={styles.modalSave}>Save</Text>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.modalScroll}>
          <View style={styles.formSection}>
            <Text style={styles.formSectionTitle}>Working Hours</Text>

            <View style={styles.timeInputRow}>
              <View style={styles.timeInputGroup}>
                <Text style={styles.inputLabel}>Start Time</Text>
                <TextInput
                  style={styles.timeInput}
                  value={startTime}
                  onChangeText={setStartTime}
                  placeholder="09:00"
                  keyboardType="numbers-and-punctuation"
                />
              </View>
              <View style={styles.timeSeparator}>
                <Text style={styles.timeSeparatorText}>to</Text>
              </View>
              <View style={styles.timeInputGroup}>
                <Text style={styles.inputLabel}>End Time</Text>
                <TextInput
                  style={styles.timeInput}
                  value={endTime}
                  onChangeText={setEndTime}
                  placeholder="17:00"
                  keyboardType="numbers-and-punctuation"
                />
              </View>
            </View>
          </View>

          <View style={styles.formSection}>
            <Text style={styles.formSectionTitle}>Break Time (Optional)</Text>

            <View style={styles.timeInputRow}>
              <View style={styles.timeInputGroup}>
                <Text style={styles.inputLabel}>Break Start</Text>
                <TextInput
                  style={styles.timeInput}
                  value={breakStart}
                  onChangeText={setBreakStart}
                  placeholder="12:00"
                  keyboardType="numbers-and-punctuation"
                />
              </View>
              <View style={styles.timeSeparator}>
                <Text style={styles.timeSeparatorText}>to</Text>
              </View>
              <View style={styles.timeInputGroup}>
                <Text style={styles.inputLabel}>Break End</Text>
                <TextInput
                  style={styles.timeInput}
                  value={breakEnd}
                  onChangeText={setBreakEnd}
                  placeholder="13:00"
                  keyboardType="numbers-and-punctuation"
                />
              </View>
            </View>
          </View>

          <View style={styles.formSection}>
            <Text style={styles.formSectionTitle}>Location</Text>
            <TextInput
              style={styles.textInput}
              value={location}
              onChangeText={setLocation}
              placeholder="e.g., Main Office, Room 101"
            />
          </View>

          <View style={styles.formSection}>
            <View style={styles.switchRow}>
              <View>
                <Text style={styles.switchLabel}>Telemedicine Day</Text>
                <Text style={styles.switchDescription}>
                  This day includes video consultations
                </Text>
              </View>
              <Switch
                value={isTelemedicineDay}
                onValueChange={setIsTelemedicineDay}
                trackColor={{ false: '#ddd', true: '#667eea' }}
                thumbColor="#fff"
              />
            </View>
          </View>
        </ScrollView>
      </View>
    </Modal>
  );

  const renderBlockModal = () => (
    <Modal visible={showBlockModal} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setShowBlockModal(false)}>
            <Text style={styles.modalCancel}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Block Time</Text>
          <TouchableOpacity onPress={handleBlockTime} disabled={saving}>
            {saving ? (
              <ActivityIndicator size="small" color="#667eea" />
            ) : (
              <Text style={styles.modalSave}>Save</Text>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.modalScroll}>
          <View style={styles.formSection}>
            <Text style={styles.formSectionTitle}>Reason</Text>
            <View style={styles.reasonGrid}>
              {BLOCK_REASONS.map((reason) => (
                <TouchableOpacity
                  key={reason.value}
                  style={[
                    styles.reasonOption,
                    blockReason === reason.value && {
                      backgroundColor: reason.color,
                      borderColor: reason.color,
                    },
                  ]}
                  onPress={() => setBlockReason(reason.value)}
                >
                  <Ionicons
                    name={reason.icon as any}
                    size={20}
                    color={blockReason === reason.value ? '#fff' : reason.color}
                  />
                  <Text
                    style={[
                      styles.reasonLabel,
                      blockReason === reason.value && styles.reasonLabelSelected,
                    ]}
                  >
                    {reason.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          <View style={styles.formSection}>
            <Text style={styles.formSectionTitle}>Start</Text>
            <View style={styles.dateTimeRow}>
              <View style={styles.dateInputGroup}>
                <Text style={styles.inputLabel}>Date</Text>
                <TextInput
                  style={styles.dateInput}
                  value={blockStartDate}
                  onChangeText={setBlockStartDate}
                  placeholder="YYYY-MM-DD"
                />
              </View>
              <View style={styles.timeInputGroupSmall}>
                <Text style={styles.inputLabel}>Time</Text>
                <TextInput
                  style={styles.timeInput}
                  value={blockStartTime}
                  onChangeText={setBlockStartTime}
                  placeholder="09:00"
                  keyboardType="numbers-and-punctuation"
                />
              </View>
            </View>
          </View>

          <View style={styles.formSection}>
            <Text style={styles.formSectionTitle}>End</Text>
            <View style={styles.dateTimeRow}>
              <View style={styles.dateInputGroup}>
                <Text style={styles.inputLabel}>Date</Text>
                <TextInput
                  style={styles.dateInput}
                  value={blockEndDate}
                  onChangeText={setBlockEndDate}
                  placeholder="YYYY-MM-DD"
                />
              </View>
              <View style={styles.timeInputGroupSmall}>
                <Text style={styles.inputLabel}>Time</Text>
                <TextInput
                  style={styles.timeInput}
                  value={blockEndTime}
                  onChangeText={setBlockEndTime}
                  placeholder="17:00"
                  keyboardType="numbers-and-punctuation"
                />
              </View>
            </View>
          </View>

          <View style={styles.formSection}>
            <Text style={styles.formSectionTitle}>Notes (Optional)</Text>
            <TextInput
              style={[styles.textInput, styles.notesInput]}
              value={blockNotes}
              onChangeText={setBlockNotes}
              placeholder="Add any additional notes..."
              multiline
              numberOfLines={3}
            />
          </View>
        </ScrollView>
      </View>
    </Modal>
  );

  if (loading) {
    return (
      <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#fff" />
          <Text style={styles.loadingText}>Loading providers...</Text>
        </View>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Provider Availability</Text>
        <View style={styles.headerRight} />
      </View>

      {/* Provider Selector */}
      <TouchableOpacity
        style={styles.providerSelector}
        onPress={() => setShowProviderModal(true)}
      >
        <View style={styles.providerSelectorLeft}>
          <View style={styles.providerSelectorAvatar}>
            <Ionicons name="person" size={20} color="#667eea" />
          </View>
          <View>
            <Text style={styles.providerSelectorLabel}>Managing schedule for</Text>
            <Text style={styles.providerSelectorName}>
              {selectedProvider?.provider_name || 'Select Provider'}
            </Text>
          </View>
        </View>
        <Ionicons name="chevron-down" size={20} color="#666" />
      </TouchableOpacity>

      {/* Tabs */}
      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'schedule' && styles.activeTab]}
          onPress={() => setActiveTab('schedule')}
        >
          <Ionicons
            name="calendar"
            size={20}
            color={activeTab === 'schedule' ? '#667eea' : '#666'}
          />
          <Text style={[styles.tabText, activeTab === 'schedule' && styles.activeTabText]}>
            Weekly Schedule
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'blocked' && styles.activeTab]}
          onPress={() => setActiveTab('blocked')}
        >
          <Ionicons
            name="close-circle"
            size={20}
            color={activeTab === 'blocked' ? '#667eea' : '#666'}
          />
          <Text style={[styles.tabText, activeTab === 'blocked' && styles.activeTabText]}>
            Blocked Times
          </Text>
          {blockedTimes.length > 0 && (
            <View style={styles.tabBadge}>
              <Text style={styles.tabBadgeText}>{blockedTimes.length}</Text>
            </View>
          )}
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#fff" />
        }
      >
        {activeTab === 'schedule' ? renderScheduleTab() : renderBlockedTab()}
      </ScrollView>

      {renderProviderModal()}
      {renderEditDayModal()}
      {renderBlockModal()}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#fff',
    marginTop: 10,
    fontSize: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 20,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  headerRight: {
    width: 40,
  },
  providerSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginBottom: 15,
    padding: 12,
    borderRadius: 12,
  },
  providerSelectorLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  providerSelectorAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  providerSelectorLabel: {
    fontSize: 12,
    color: '#999',
  },
  providerSelectorName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  tabContainer: {
    flexDirection: 'row',
    marginHorizontal: 20,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 4,
    marginBottom: 15,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderRadius: 10,
    gap: 6,
  },
  activeTab: {
    backgroundColor: '#f0f0f0',
  },
  tabText: {
    fontSize: 13,
    color: '#666',
    fontWeight: '500',
  },
  activeTabText: {
    color: '#667eea',
    fontWeight: '600',
  },
  tabBadge: {
    backgroundColor: '#F44336',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
    marginLeft: 4,
  },
  tabBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
  },
  tabContent: {
    paddingHorizontal: 20,
    paddingBottom: 30,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 4,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.7)',
    marginBottom: 16,
  },
  dayCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
  },
  dayCardInactive: {
    opacity: 0.8,
  },
  dayHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  dayNameContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  dayIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  dayName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  dayDetails: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  timeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
  },
  timeText: {
    fontSize: 14,
    color: '#333',
  },
  breakText: {
    fontSize: 13,
    color: '#666',
  },
  tagsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 8,
  },
  locationTag: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    gap: 4,
  },
  locationTagText: {
    fontSize: 12,
    color: '#667eea',
  },
  teleTag: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#667eea',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    gap: 4,
  },
  teleTagText: {
    fontSize: 12,
    color: '#fff',
  },
  notConfiguredText: {
    fontSize: 13,
    color: '#999',
    marginTop: 8,
    fontStyle: 'italic',
  },
  blockedHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  addBlockButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    gap: 4,
  },
  addBlockButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
  emptyBlocked: {
    alignItems: 'center',
    paddingVertical: 40,
    backgroundColor: '#fff',
    borderRadius: 12,
  },
  emptyBlockedText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginTop: 12,
  },
  emptyBlockedSubtext: {
    fontSize: 14,
    color: '#999',
    marginTop: 4,
  },
  blockedCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
  },
  blockedLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: 12,
  },
  blockedIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  blockedInfo: {
    flex: 1,
  },
  blockedReason: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  blockedDates: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  blockedNotes: {
    fontSize: 12,
    color: '#999',
    fontStyle: 'italic',
    marginTop: 4,
  },
  removeBlockButton: {
    padding: 8,
  },
  // Modal Styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  providerModalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '60%',
  },
  providerModalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  providerModalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  providerList: {
    padding: 10,
  },
  providerItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    borderRadius: 12,
    marginBottom: 8,
    backgroundColor: '#f5f5f5',
  },
  providerItemSelected: {
    backgroundColor: '#E8EAF6',
    borderWidth: 1,
    borderColor: '#667eea',
  },
  providerAvatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  providerInfo: {
    flex: 1,
  },
  providerName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  providerLocation: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 15,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  modalCancel: {
    fontSize: 16,
    color: '#666',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  modalSave: {
    fontSize: 16,
    color: '#667eea',
    fontWeight: '600',
  },
  modalScroll: {
    flex: 1,
    padding: 20,
  },
  formSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  formSectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  timeInputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  timeInputGroup: {
    flex: 1,
  },
  timeInputGroupSmall: {
    flex: 0.6,
  },
  inputLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 6,
  },
  timeInput: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    color: '#333',
    textAlign: 'center',
  },
  timeSeparator: {
    paddingHorizontal: 12,
    paddingBottom: 12,
  },
  timeSeparatorText: {
    fontSize: 14,
    color: '#999',
  },
  textInput: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    fontSize: 15,
    color: '#333',
  },
  notesInput: {
    minHeight: 80,
    textAlignVertical: 'top',
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  switchLabel: {
    fontSize: 15,
    fontWeight: '500',
    color: '#333',
  },
  switchDescription: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  reasonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  reasonOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ddd',
    gap: 6,
  },
  reasonLabel: {
    fontSize: 13,
    color: '#333',
  },
  reasonLabelSelected: {
    color: '#fff',
  },
  dateTimeRow: {
    flexDirection: 'row',
    gap: 12,
  },
  dateInputGroup: {
    flex: 1,
  },
  dateInput: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    fontSize: 15,
    color: '#333',
  },
});
