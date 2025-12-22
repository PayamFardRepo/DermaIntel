/**
 * Appointment Scheduling Screen
 *
 * Features:
 * - Book in-person and telemedicine appointments
 * - View and manage upcoming/past appointments
 * - Calendar integration (iCal download, Google Calendar)
 * - Waitlist management
 * - Appointment reminders
 * - Check-in functionality
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Modal,
  Alert,
  ActivityIndicator,
  Platform,
  TextInput,
  Linking,
  RefreshControl,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

interface AppointmentType {
  type: string;
  name: string;
  duration_minutes: number;
  description: string;
  preparation: string[];
  allows_telemedicine: boolean;
}

interface TimeSlot {
  start_time: string;
  end_time: string;
  date: string;
  time: string;
  provider_id: string;
  provider_name: string;
  is_telemedicine: boolean;
  location: string;
}

interface Appointment {
  appointment_id: string;
  appointment_date: string;
  start_time: string;
  end_time: string;
  duration_minutes: number;
  appointment_type: string;
  appointment_type_name: string;
  status: string;
  provider_name: string;
  is_telemedicine: boolean;
  telemedicine_link: string | null;
  location: string | null;
  reason_for_visit: string | null;
  patient_notes: string | null;
  created_at: string;
}

interface WaitlistEntry {
  waitlist_id: string;
  appointment_type: string;
  preferred_dates: string[];
  flexibility: string;
  priority: number;
  status: string;
  created_at: string;
}

type TabType = 'upcoming' | 'past' | 'book';

export default function AppointmentsScreen() {
  const { t } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // State
  const [activeTab, setActiveTab] = useState<TabType>('upcoming');
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [appointmentTypes, setAppointmentTypes] = useState<AppointmentType[]>([]);
  const [availableSlots, setAvailableSlots] = useState<TimeSlot[]>([]);
  const [waitlist, setWaitlist] = useState<WaitlistEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingSlots, setLoadingSlots] = useState(false);

  // Booking state
  const [selectedType, setSelectedType] = useState<AppointmentType | null>(null);
  const [selectedSlot, setSelectedSlot] = useState<TimeSlot | null>(null);
  const [showBookingModal, setShowBookingModal] = useState(false);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [isTelemedicine, setIsTelemedicine] = useState(false);
  const [reasonForVisit, setReasonForVisit] = useState('');
  const [patientNotes, setPatientNotes] = useState('');
  const [patientPhone, setPatientPhone] = useState('');
  const [isBooking, setIsBooking] = useState(false);

  // Date selection state
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [dateRange, setDateRange] = useState<Date[]>([]);

  // Appointment detail state
  const [selectedAppointment, setSelectedAppointment] = useState<Appointment | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  // Waitlist modal state
  const [showWaitlistModal, setShowWaitlistModal] = useState(false);
  const [waitlistDates, setWaitlistDates] = useState('');
  const [waitlistFlexibility, setWaitlistFlexibility] = useState<'specific' | 'somewhat_flexible' | 'flexible'>('flexible');

  // Auth headers
  const getAuthHeaders = async () => {
    const token = await AsyncStorage.getItem('accessToken');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  };

  // Initialize
  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
      return;
    }
    loadData();
  }, [isAuthenticated]);

  // Load data based on active tab
  useEffect(() => {
    if (activeTab === 'book') {
      loadAppointmentTypes();
    } else {
      loadAppointments();
    }
  }, [activeTab]);

  // Generate date range for calendar picker
  useEffect(() => {
    const dates: Date[] = [];
    const today = new Date();
    for (let i = 0; i < 30; i++) {
      const date = new Date(today);
      date.setDate(today.getDate() + i);
      dates.push(date);
    }
    setDateRange(dates);
  }, []);

  const loadData = async () => {
    setIsLoading(true);
    await Promise.all([
      loadAppointments(),
      loadAppointmentTypes(),
      loadWaitlist(),
    ]);
    setIsLoading(false);
  };

  const loadAppointments = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/appointments?include_past=true`, {
        headers,
      });

      if (response.ok) {
        const data = await response.json();
        setAppointments(data.appointments || []);
      }
    } catch (error) {
      console.error('Error loading appointments:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const loadAppointmentTypes = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/appointments/types`, { headers });

      if (response.ok) {
        const data = await response.json();
        setAppointmentTypes(data.appointment_types || []);
      }
    } catch (error) {
      console.error('Error loading appointment types:', error);
    }
  };

  const loadWaitlist = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/appointments/waitlist`, { headers });

      if (response.ok) {
        const data = await response.json();
        setWaitlist(data.waitlist || []);
      }
    } catch (error) {
      console.error('Error loading waitlist:', error);
    }
  };

  const loadAvailableSlots = async (appointmentType: string, startDate: Date, endDate: Date) => {
    setLoadingSlots(true);
    try {
      const headers = await getAuthHeaders();
      const startStr = startDate.toISOString().split('T')[0];
      const endStr = endDate.toISOString().split('T')[0];

      const url = `${API_BASE_URL}/appointments/slots?start_date=${startStr}&end_date=${endStr}&appointment_type=${appointmentType}&is_telemedicine=${isTelemedicine}`;
      const response = await fetch(url, { headers });

      if (response.ok) {
        const data = await response.json();
        setAvailableSlots(data.slots || []);
      }
    } catch (error) {
      console.error('Error loading slots:', error);
    } finally {
      setLoadingSlots(false);
    }
  };

  const handleSelectType = (type: AppointmentType) => {
    setSelectedType(type);
    setSelectedSlot(null);

    // Load slots for next 7 days
    const startDate = new Date();
    const endDate = new Date();
    endDate.setDate(endDate.getDate() + 7);
    loadAvailableSlots(type.type, startDate, endDate);
  };

  const handleSelectDate = (date: Date) => {
    setSelectedDate(date);
    if (selectedType) {
      const endDate = new Date(date);
      endDate.setDate(date.getDate() + 1);
      loadAvailableSlots(selectedType.type, date, endDate);
    }
  };

  const handleBookAppointment = async () => {
    if (!selectedType || !selectedSlot) return;

    setIsBooking(true);
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/appointments/book`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          appointment_type: selectedType.type,
          appointment_date: selectedSlot.date,
          start_time: selectedSlot.time.substring(0, 5), // HH:MM format
          is_telemedicine: isTelemedicine,
          reason_for_visit: reasonForVisit,
          patient_notes: patientNotes,
          patient_phone: patientPhone,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert(
          'Appointment Booked!',
          `Your ${selectedType.name} appointment has been scheduled for ${formatDate(selectedSlot.date)} at ${formatTime(selectedSlot.time)}.\n\n${data.is_telemedicine ? 'A video call link will be sent to you before the appointment.' : `Location: ${data.location}`}`,
          [{ text: 'OK', onPress: () => {
            setShowConfirmModal(false);
            setShowBookingModal(false);
            resetBookingState();
            setActiveTab('upcoming');
            loadAppointments();
          }}]
        );
      } else {
        const error = await response.json();
        Alert.alert('Booking Failed', error.detail || 'Unable to book appointment');
      }
    } catch (error) {
      console.error('Error booking appointment:', error);
      Alert.alert('Error', 'Failed to book appointment. Please try again.');
    } finally {
      setIsBooking(false);
    }
  };

  const handleCancelAppointment = async (appointment: Appointment) => {
    Alert.alert(
      'Cancel Appointment',
      `Are you sure you want to cancel your ${appointment.appointment_type_name} appointment on ${formatDate(appointment.appointment_date)}?`,
      [
        { text: 'No', style: 'cancel' },
        {
          text: 'Yes, Cancel',
          style: 'destructive',
          onPress: async () => {
            try {
              const headers = await getAuthHeaders();
              const response = await fetch(
                `${API_BASE_URL}/appointments/${appointment.appointment_id}/cancel`,
                {
                  method: 'POST',
                  headers,
                  body: JSON.stringify({ reason: 'Cancelled by patient' }),
                }
              );

              if (response.ok) {
                Alert.alert('Cancelled', 'Your appointment has been cancelled.');
                loadAppointments();
                setShowDetailModal(false);
              } else {
                Alert.alert('Error', 'Failed to cancel appointment');
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to cancel appointment');
            }
          },
        },
      ]
    );
  };

  const handleConfirmAppointment = async (appointment: Appointment) => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(
        `${API_BASE_URL}/appointments/${appointment.appointment_id}/confirm`,
        {
          method: 'POST',
          headers,
        }
      );

      if (response.ok) {
        Alert.alert('Confirmed', 'Your appointment has been confirmed.');
        loadAppointments();
        setShowDetailModal(false);
      } else {
        Alert.alert('Error', 'Failed to confirm appointment');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to confirm appointment');
    }
  };

  const handleCheckIn = async (appointment: Appointment) => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(
        `${API_BASE_URL}/appointments/${appointment.appointment_id}/check-in`,
        {
          method: 'POST',
          headers,
        }
      );

      if (response.ok) {
        Alert.alert('Checked In', 'You have been checked in for your appointment.');
        loadAppointments();
        setShowDetailModal(false);
      } else {
        Alert.alert('Error', 'Failed to check in');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to check in');
    }
  };

  const handleAddToCalendar = async (appointment: Appointment, format: 'ical' | 'google') => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(
        `${API_BASE_URL}/appointments/${appointment.appointment_id}/calendar?format=${format}`,
        { headers }
      );

      if (response.ok) {
        const data = await response.json();
        if (format === 'google' && data.google_calendar_url) {
          Linking.openURL(data.google_calendar_url);
        } else if (format === 'ical' && data.ical_data) {
          // For iOS/Android, we'd need to handle ICS file download
          Alert.alert('Calendar Export', 'ICS file download is available. Opening Google Calendar instead.');
          // As a fallback, generate Google Calendar URL
          const response2 = await fetch(
            `${API_BASE_URL}/appointments/${appointment.appointment_id}/calendar?format=google`,
            { headers }
          );
          if (response2.ok) {
            const data2 = await response2.json();
            if (data2.google_calendar_url) {
              Linking.openURL(data2.google_calendar_url);
            }
          }
        }
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to add to calendar');
    }
  };

  const handleJoinTelemedicine = (appointment: Appointment) => {
    if (appointment.telemedicine_link) {
      Linking.openURL(appointment.telemedicine_link);
    } else {
      Alert.alert('Not Available', 'The video call link is not yet available. Please try again closer to your appointment time.');
    }
  };

  const handleAddToWaitlist = async () => {
    if (!selectedType) return;

    try {
      const headers = await getAuthHeaders();
      const preferredDates = waitlistDates.split(',').map(d => d.trim()).filter(d => d);

      const response = await fetch(`${API_BASE_URL}/appointments/waitlist`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          appointment_type: selectedType.type,
          preferred_dates: preferredDates,
          flexibility: waitlistFlexibility,
          priority: 5,
        }),
      });

      if (response.ok) {
        Alert.alert(
          'Added to Waitlist',
          `You've been added to the waitlist for ${selectedType.name}. We'll notify you when a slot becomes available.`
        );
        setShowWaitlistModal(false);
        loadWaitlist();
      } else {
        Alert.alert('Error', 'Failed to add to waitlist');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to add to waitlist');
    }
  };

  const handleRemoveFromWaitlist = async (waitlistId: string) => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/appointments/waitlist/${waitlistId}`, {
        method: 'DELETE',
        headers,
      });

      if (response.ok) {
        loadWaitlist();
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to remove from waitlist');
    }
  };

  const resetBookingState = () => {
    setSelectedType(null);
    setSelectedSlot(null);
    setIsTelemedicine(false);
    setReasonForVisit('');
    setPatientNotes('');
    setPatientPhone('');
    setAvailableSlots([]);
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadData();
  }, []);

  // Helper functions
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
  };

  const formatTime = (timeStr: string) => {
    const [hours, minutes] = timeStr.split(':');
    const hour = parseInt(hours);
    const ampm = hour >= 12 ? 'PM' : 'AM';
    const hour12 = hour % 12 || 12;
    return `${hour12}:${minutes} ${ampm}`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'scheduled': return '#3b82f6';
      case 'confirmed': return '#10b981';
      case 'checked_in': return '#8b5cf6';
      case 'in_progress': return '#f59e0b';
      case 'completed': return '#6b7280';
      case 'cancelled': return '#ef4444';
      case 'no_show': return '#dc2626';
      default: return '#6b7280';
    }
  };

  const isToday = (dateStr: string) => {
    const today = new Date().toISOString().split('T')[0];
    return dateStr === today;
  };

  // Filter appointments
  const upcomingAppointments = appointments.filter(
    apt => ['scheduled', 'confirmed', 'checked_in'].includes(apt.status) &&
    new Date(apt.appointment_date) >= new Date(new Date().toISOString().split('T')[0])
  );

  const pastAppointments = appointments.filter(
    apt => ['completed', 'cancelled', 'no_show'].includes(apt.status) ||
    new Date(apt.appointment_date) < new Date(new Date().toISOString().split('T')[0])
  );

  // Filter slots by selected date
  const slotsForSelectedDate = availableSlots.filter(
    slot => slot.date === selectedDate.toISOString().split('T')[0]
  );

  // Render appointment card
  const renderAppointmentCard = (appointment: Appointment) => (
    <TouchableOpacity
      key={appointment.appointment_id}
      style={styles.appointmentCard}
      onPress={() => {
        setSelectedAppointment(appointment);
        setShowDetailModal(true);
      }}
    >
      <View style={styles.cardHeader}>
        <View>
          <Text style={styles.appointmentType}>{appointment.appointment_type_name}</Text>
          <Text style={styles.providerName}>{appointment.provider_name}</Text>
        </View>
        <View style={[styles.statusBadge, { backgroundColor: getStatusColor(appointment.status) }]}>
          <Text style={styles.statusText}>{appointment.status.replace('_', ' ').toUpperCase()}</Text>
        </View>
      </View>

      <View style={styles.cardDetails}>
        <View style={styles.detailRow}>
          <Ionicons name="calendar-outline" size={16} color="#6b7280" />
          <Text style={styles.detailText}>{formatDate(appointment.appointment_date)}</Text>
          {isToday(appointment.appointment_date) && (
            <View style={styles.todayBadge}>
              <Text style={styles.todayText}>TODAY</Text>
            </View>
          )}
        </View>
        <View style={styles.detailRow}>
          <Ionicons name="time-outline" size={16} color="#6b7280" />
          <Text style={styles.detailText}>
            {formatTime(appointment.start_time)} ({appointment.duration_minutes} min)
          </Text>
        </View>
        <View style={styles.detailRow}>
          <Ionicons name={appointment.is_telemedicine ? 'videocam-outline' : 'location-outline'} size={16} color="#6b7280" />
          <Text style={styles.detailText}>
            {appointment.is_telemedicine ? 'Video Consultation' : appointment.location || 'Main Clinic'}
          </Text>
        </View>
      </View>

      {appointment.reason_for_visit && (
        <Text style={styles.reasonText} numberOfLines={2}>
          {appointment.reason_for_visit}
        </Text>
      )}

      <View style={styles.cardActions}>
        {appointment.status === 'scheduled' && (
          <TouchableOpacity
            style={[styles.actionButton, styles.confirmButton]}
            onPress={() => handleConfirmAppointment(appointment)}
          >
            <Ionicons name="checkmark-circle-outline" size={18} color="#fff" />
            <Text style={styles.actionButtonText}>Confirm</Text>
          </TouchableOpacity>
        )}
        {appointment.is_telemedicine && ['confirmed', 'checked_in'].includes(appointment.status) && (
          <TouchableOpacity
            style={[styles.actionButton, styles.joinButton]}
            onPress={() => handleJoinTelemedicine(appointment)}
          >
            <Ionicons name="videocam" size={18} color="#fff" />
            <Text style={styles.actionButtonText}>Join Call</Text>
          </TouchableOpacity>
        )}
        {appointment.status === 'confirmed' && isToday(appointment.appointment_date) && (
          <TouchableOpacity
            style={[styles.actionButton, styles.checkInButton]}
            onPress={() => handleCheckIn(appointment)}
          >
            <Ionicons name="log-in-outline" size={18} color="#fff" />
            <Text style={styles.actionButtonText}>Check In</Text>
          </TouchableOpacity>
        )}
      </View>
    </TouchableOpacity>
  );

  // Render appointment type card
  const renderTypeCard = (type: AppointmentType) => (
    <TouchableOpacity
      key={type.type}
      style={[
        styles.typeCard,
        selectedType?.type === type.type && styles.typeCardSelected,
      ]}
      onPress={() => handleSelectType(type)}
    >
      <View style={styles.typeHeader}>
        <Ionicons
          name={
            type.type.includes('telemedicine') ? 'videocam' :
            type.type.includes('surgery') || type.type.includes('biopsy') ? 'cut' :
            type.type.includes('cosmetic') ? 'sparkles' :
            'medical'
          }
          size={24}
          color={selectedType?.type === type.type ? '#2563eb' : '#6b7280'}
        />
        <View style={styles.typeInfo}>
          <Text style={[styles.typeName, selectedType?.type === type.type && styles.typeNameSelected]}>
            {type.name}
          </Text>
          <Text style={styles.typeDuration}>{type.duration_minutes} minutes</Text>
        </View>
        {type.allows_telemedicine && (
          <View style={styles.telemedicineBadge}>
            <Ionicons name="videocam-outline" size={12} color="#2563eb" />
          </View>
        )}
      </View>
      <Text style={styles.typeDescription}>{type.description}</Text>
    </TouchableOpacity>
  );

  // Render booking modal
  const renderBookingModal = () => (
    <Modal visible={showBookingModal} animationType="slide" presentationStyle="pageSheet">
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => {
            setShowBookingModal(false);
            resetBookingState();
          }}>
            <Ionicons name="close" size={28} color="#1e3a5f" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Book Appointment</Text>
          <View style={{ width: 28 }} />
        </View>

        <ScrollView style={styles.modalContent}>
          {/* Step 1: Select Type */}
          <Text style={styles.stepTitle}>1. Select Appointment Type</Text>
          {appointmentTypes.map(renderTypeCard)}

          {selectedType && (
            <>
              {/* Telemedicine toggle */}
              {selectedType.allows_telemedicine && (
                <TouchableOpacity
                  style={styles.telemedicineToggle}
                  onPress={() => setIsTelemedicine(!isTelemedicine)}
                >
                  <View style={styles.toggleLeft}>
                    <Ionicons
                      name={isTelemedicine ? 'videocam' : 'videocam-outline'}
                      size={24}
                      color={isTelemedicine ? '#2563eb' : '#6b7280'}
                    />
                    <Text style={styles.toggleText}>Video Consultation</Text>
                  </View>
                  <View style={[styles.toggleSwitch, isTelemedicine && styles.toggleSwitchOn]}>
                    <View style={[styles.toggleKnob, isTelemedicine && styles.toggleKnobOn]} />
                  </View>
                </TouchableOpacity>
              )}

              {/* Step 2: Select Date */}
              <Text style={styles.stepTitle}>2. Select Date</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.dateScroller}>
                {dateRange.map((date, index) => {
                  const dateStr = date.toISOString().split('T')[0];
                  const isSelected = selectedDate.toISOString().split('T')[0] === dateStr;
                  const isWeekend = date.getDay() === 0 || date.getDay() === 6;

                  return (
                    <TouchableOpacity
                      key={dateStr}
                      style={[
                        styles.dateCard,
                        isSelected && styles.dateCardSelected,
                        isWeekend && styles.dateCardWeekend,
                      ]}
                      onPress={() => handleSelectDate(date)}
                    >
                      <Text style={[styles.dateDayName, isSelected && styles.dateDayNameSelected]}>
                        {date.toLocaleDateString('en-US', { weekday: 'short' })}
                      </Text>
                      <Text style={[styles.dateDay, isSelected && styles.dateDaySelected]}>
                        {date.getDate()}
                      </Text>
                      <Text style={[styles.dateMonth, isSelected && styles.dateMonthSelected]}>
                        {date.toLocaleDateString('en-US', { month: 'short' })}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </ScrollView>

              {/* Step 3: Select Time */}
              <Text style={styles.stepTitle}>3. Select Time</Text>
              {loadingSlots ? (
                <View style={styles.loadingSlots}>
                  <ActivityIndicator size="small" color="#2563eb" />
                  <Text style={styles.loadingSlotsText}>Loading available times...</Text>
                </View>
              ) : slotsForSelectedDate.length === 0 ? (
                <View style={styles.noSlots}>
                  <Ionicons name="calendar-outline" size={48} color="#9ca3af" />
                  <Text style={styles.noSlotsText}>No available slots for this date</Text>
                  <TouchableOpacity
                    style={styles.waitlistButton}
                    onPress={() => setShowWaitlistModal(true)}
                  >
                    <Text style={styles.waitlistButtonText}>Join Waitlist</Text>
                  </TouchableOpacity>
                </View>
              ) : (
                <View style={styles.slotsGrid}>
                  {slotsForSelectedDate.map((slot, index) => (
                    <TouchableOpacity
                      key={index}
                      style={[
                        styles.slotCard,
                        selectedSlot?.time === slot.time && styles.slotCardSelected,
                      ]}
                      onPress={() => setSelectedSlot(slot)}
                    >
                      <Text style={[
                        styles.slotTime,
                        selectedSlot?.time === slot.time && styles.slotTimeSelected,
                      ]}>
                        {formatTime(slot.time)}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              )}

              {selectedSlot && (
                <>
                  {/* Step 4: Details */}
                  <Text style={styles.stepTitle}>4. Appointment Details</Text>
                  <View style={styles.inputGroup}>
                    <Text style={styles.inputLabel}>Reason for Visit</Text>
                    <TextInput
                      style={styles.textInput}
                      value={reasonForVisit}
                      onChangeText={setReasonForVisit}
                      placeholder="Describe your concern..."
                      multiline
                      numberOfLines={3}
                    />
                  </View>
                  <View style={styles.inputGroup}>
                    <Text style={styles.inputLabel}>Additional Notes (Optional)</Text>
                    <TextInput
                      style={styles.textInput}
                      value={patientNotes}
                      onChangeText={setPatientNotes}
                      placeholder="Any additional information..."
                      multiline
                      numberOfLines={2}
                    />
                  </View>
                  <View style={styles.inputGroup}>
                    <Text style={styles.inputLabel}>Phone Number</Text>
                    <TextInput
                      style={styles.textInputSingle}
                      value={patientPhone}
                      onChangeText={setPatientPhone}
                      placeholder="Your phone number"
                      keyboardType="phone-pad"
                    />
                  </View>

                  {/* Preparation instructions */}
                  <View style={styles.preparationSection}>
                    <Text style={styles.preparationTitle}>
                      <Ionicons name="information-circle" size={16} color="#2563eb" /> Preparation Instructions
                    </Text>
                    {selectedType.preparation.map((prep, idx) => (
                      <View key={idx} style={styles.preparationItem}>
                        <Ionicons name="checkmark-circle" size={16} color="#10b981" />
                        <Text style={styles.preparationText}>{prep}</Text>
                      </View>
                    ))}
                  </View>

                  {/* Book button */}
                  <TouchableOpacity
                    style={styles.bookButton}
                    onPress={() => setShowConfirmModal(true)}
                  >
                    <Text style={styles.bookButtonText}>Review & Book Appointment</Text>
                    <Ionicons name="arrow-forward" size={20} color="#fff" />
                  </TouchableOpacity>
                </>
              )}
            </>
          )}

          <View style={{ height: 40 }} />
        </ScrollView>
      </LinearGradient>
    </Modal>
  );

  // Render confirmation modal
  const renderConfirmModal = () => (
    <Modal visible={showConfirmModal} animationType="fade" transparent>
      <View style={styles.confirmOverlay}>
        <View style={styles.confirmModal}>
          <Text style={styles.confirmTitle}>Confirm Appointment</Text>

          <View style={styles.confirmDetails}>
            <View style={styles.confirmRow}>
              <Ionicons name="medical" size={20} color="#2563eb" />
              <Text style={styles.confirmLabel}>{selectedType?.name}</Text>
            </View>
            <View style={styles.confirmRow}>
              <Ionicons name="calendar" size={20} color="#2563eb" />
              <Text style={styles.confirmLabel}>{selectedSlot && formatDate(selectedSlot.date)}</Text>
            </View>
            <View style={styles.confirmRow}>
              <Ionicons name="time" size={20} color="#2563eb" />
              <Text style={styles.confirmLabel}>
                {selectedSlot && formatTime(selectedSlot.time)} ({selectedType?.duration_minutes} min)
              </Text>
            </View>
            <View style={styles.confirmRow}>
              <Ionicons name={isTelemedicine ? 'videocam' : 'location'} size={20} color="#2563eb" />
              <Text style={styles.confirmLabel}>
                {isTelemedicine ? 'Video Consultation' : selectedSlot?.location || 'Main Clinic'}
              </Text>
            </View>
            <View style={styles.confirmRow}>
              <Ionicons name="person" size={20} color="#2563eb" />
              <Text style={styles.confirmLabel}>{selectedSlot?.provider_name}</Text>
            </View>
          </View>

          {reasonForVisit && (
            <View style={styles.confirmReasonBox}>
              <Text style={styles.confirmReasonLabel}>Reason:</Text>
              <Text style={styles.confirmReasonText}>{reasonForVisit}</Text>
            </View>
          )}

          <View style={styles.confirmActions}>
            <TouchableOpacity
              style={styles.confirmCancelButton}
              onPress={() => setShowConfirmModal(false)}
            >
              <Text style={styles.confirmCancelText}>Go Back</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.confirmBookButton}
              onPress={handleBookAppointment}
              disabled={isBooking}
            >
              {isBooking ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <>
                  <Ionicons name="checkmark-circle" size={20} color="#fff" />
                  <Text style={styles.confirmBookText}>Confirm Booking</Text>
                </>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );

  // Render appointment detail modal
  const renderDetailModal = () => {
    if (!selectedAppointment) return null;

    return (
      <Modal visible={showDetailModal} animationType="slide" presentationStyle="pageSheet">
        <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <TouchableOpacity onPress={() => setShowDetailModal(false)}>
              <Ionicons name="close" size={28} color="#1e3a5f" />
            </TouchableOpacity>
            <Text style={styles.modalTitle}>Appointment Details</Text>
            <View style={{ width: 28 }} />
          </View>

          <ScrollView style={styles.modalContent}>
            <View style={styles.detailCard}>
              <View style={styles.detailHeader}>
                <Text style={styles.detailType}>{selectedAppointment.appointment_type_name}</Text>
                <View style={[styles.statusBadge, { backgroundColor: getStatusColor(selectedAppointment.status) }]}>
                  <Text style={styles.statusText}>{selectedAppointment.status.replace('_', ' ').toUpperCase()}</Text>
                </View>
              </View>

              <View style={styles.detailSection}>
                <View style={styles.detailItem}>
                  <Ionicons name="person" size={20} color="#2563eb" />
                  <View>
                    <Text style={styles.detailItemLabel}>Provider</Text>
                    <Text style={styles.detailItemValue}>{selectedAppointment.provider_name}</Text>
                  </View>
                </View>
                <View style={styles.detailItem}>
                  <Ionicons name="calendar" size={20} color="#2563eb" />
                  <View>
                    <Text style={styles.detailItemLabel}>Date</Text>
                    <Text style={styles.detailItemValue}>{formatDate(selectedAppointment.appointment_date)}</Text>
                  </View>
                </View>
                <View style={styles.detailItem}>
                  <Ionicons name="time" size={20} color="#2563eb" />
                  <View>
                    <Text style={styles.detailItemLabel}>Time</Text>
                    <Text style={styles.detailItemValue}>
                      {formatTime(selectedAppointment.start_time)} - {formatTime(selectedAppointment.end_time)}
                    </Text>
                  </View>
                </View>
                <View style={styles.detailItem}>
                  <Ionicons name={selectedAppointment.is_telemedicine ? 'videocam' : 'location'} size={20} color="#2563eb" />
                  <View>
                    <Text style={styles.detailItemLabel}>
                      {selectedAppointment.is_telemedicine ? 'Video Call' : 'Location'}
                    </Text>
                    <Text style={styles.detailItemValue}>
                      {selectedAppointment.is_telemedicine ? 'Video Consultation' : selectedAppointment.location || 'Main Clinic'}
                    </Text>
                  </View>
                </View>
              </View>

              {selectedAppointment.reason_for_visit && (
                <View style={styles.reasonSection}>
                  <Text style={styles.reasonLabel}>Reason for Visit</Text>
                  <Text style={styles.reasonValue}>{selectedAppointment.reason_for_visit}</Text>
                </View>
              )}

              {/* Calendar Integration */}
              <View style={styles.calendarSection}>
                <Text style={styles.calendarTitle}>Add to Calendar</Text>
                <View style={styles.calendarButtons}>
                  <TouchableOpacity
                    style={styles.calendarButton}
                    onPress={() => handleAddToCalendar(selectedAppointment, 'google')}
                  >
                    <Ionicons name="logo-google" size={20} color="#4285f4" />
                    <Text style={styles.calendarButtonText}>Google</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={styles.calendarButton}
                    onPress={() => handleAddToCalendar(selectedAppointment, 'ical')}
                  >
                    <Ionicons name="calendar" size={20} color="#1e3a5f" />
                    <Text style={styles.calendarButtonText}>iCal</Text>
                  </TouchableOpacity>
                </View>
              </View>

              {/* Action buttons based on status */}
              <View style={styles.detailActions}>
                {selectedAppointment.status === 'scheduled' && (
                  <TouchableOpacity
                    style={[styles.detailActionButton, styles.confirmActionButton]}
                    onPress={() => handleConfirmAppointment(selectedAppointment)}
                  >
                    <Ionicons name="checkmark-circle" size={20} color="#fff" />
                    <Text style={styles.detailActionText}>Confirm Appointment</Text>
                  </TouchableOpacity>
                )}
                {selectedAppointment.is_telemedicine && ['confirmed', 'checked_in'].includes(selectedAppointment.status) && (
                  <TouchableOpacity
                    style={[styles.detailActionButton, styles.joinActionButton]}
                    onPress={() => handleJoinTelemedicine(selectedAppointment)}
                  >
                    <Ionicons name="videocam" size={20} color="#fff" />
                    <Text style={styles.detailActionText}>Join Video Call</Text>
                  </TouchableOpacity>
                )}
                {selectedAppointment.status === 'confirmed' && isToday(selectedAppointment.appointment_date) && (
                  <TouchableOpacity
                    style={[styles.detailActionButton, styles.checkInActionButton]}
                    onPress={() => handleCheckIn(selectedAppointment)}
                  >
                    <Ionicons name="log-in" size={20} color="#fff" />
                    <Text style={styles.detailActionText}>Check In</Text>
                  </TouchableOpacity>
                )}
                {['scheduled', 'confirmed'].includes(selectedAppointment.status) && (
                  <TouchableOpacity
                    style={[styles.detailActionButton, styles.cancelActionButton]}
                    onPress={() => handleCancelAppointment(selectedAppointment)}
                  >
                    <Ionicons name="close-circle" size={20} color="#dc2626" />
                    <Text style={[styles.detailActionText, { color: '#dc2626' }]}>Cancel Appointment</Text>
                  </TouchableOpacity>
                )}
              </View>
            </View>
          </ScrollView>
        </LinearGradient>
      </Modal>
    );
  };

  // Render waitlist modal
  const renderWaitlistModal = () => (
    <Modal visible={showWaitlistModal} animationType="fade" transparent>
      <View style={styles.confirmOverlay}>
        <View style={styles.confirmModal}>
          <Text style={styles.confirmTitle}>Join Waitlist</Text>
          <Text style={styles.waitlistSubtitle}>
            We'll notify you when a slot becomes available for {selectedType?.name}
          </Text>

          <View style={styles.inputGroup}>
            <Text style={styles.inputLabel}>Preferred Dates (comma-separated)</Text>
            <TextInput
              style={styles.textInputSingle}
              value={waitlistDates}
              onChangeText={setWaitlistDates}
              placeholder="e.g., 2024-01-15, 2024-01-16"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.inputLabel}>Flexibility</Text>
            <View style={styles.flexibilityOptions}>
              {(['specific', 'somewhat_flexible', 'flexible'] as const).map(option => (
                <TouchableOpacity
                  key={option}
                  style={[
                    styles.flexibilityOption,
                    waitlistFlexibility === option && styles.flexibilityOptionSelected,
                  ]}
                  onPress={() => setWaitlistFlexibility(option)}
                >
                  <Text style={[
                    styles.flexibilityText,
                    waitlistFlexibility === option && styles.flexibilityTextSelected,
                  ]}>
                    {option.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          <View style={styles.confirmActions}>
            <TouchableOpacity
              style={styles.confirmCancelButton}
              onPress={() => setShowWaitlistModal(false)}
            >
              <Text style={styles.confirmCancelText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.confirmBookButton}
              onPress={handleAddToWaitlist}
            >
              <Text style={styles.confirmBookText}>Join Waitlist</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );

  // Render waitlist section
  const renderWaitlistSection = () => {
    if (waitlist.length === 0) return null;

    return (
      <View style={styles.waitlistSection}>
        <Text style={styles.waitlistTitle}>Your Waitlist</Text>
        {waitlist.map(entry => (
          <View key={entry.waitlist_id} style={styles.waitlistCard}>
            <View style={styles.waitlistInfo}>
              <Text style={styles.waitlistType}>
                {entry.appointment_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </Text>
              <Text style={styles.waitlistFlexibility}>Flexibility: {entry.flexibility}</Text>
            </View>
            <TouchableOpacity
              onPress={() => handleRemoveFromWaitlist(entry.waitlist_id)}
            >
              <Ionicons name="close-circle" size={24} color="#dc2626" />
            </TouchableOpacity>
          </View>
        ))}
      </View>
    );
  };

  if (isLoading) {
    return (
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2563eb" />
          <Text style={styles.loadingText}>Loading appointments...</Text>
        </View>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#2563eb" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Appointments</Text>
        <View style={styles.headerSpacer} />
      </View>

      {/* Tabs */}
      <View style={styles.tabs}>
        {(['upcoming', 'past', 'book'] as TabType[]).map(tab => (
          <TouchableOpacity
            key={tab}
            style={[styles.tab, activeTab === tab && styles.tabActive]}
            onPress={() => setActiveTab(tab)}
          >
            <Text style={[styles.tabText, activeTab === tab && styles.tabTextActive]}>
              {tab === 'upcoming' ? 'Upcoming' : tab === 'past' ? 'Past' : 'Book New'}
            </Text>
            {tab === 'upcoming' && upcomingAppointments.length > 0 && (
              <View style={styles.badge}>
                <Text style={styles.badgeText}>{upcomingAppointments.length}</Text>
              </View>
            )}
          </TouchableOpacity>
        ))}
      </View>

      {/* Content */}
      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {activeTab === 'upcoming' && (
          <>
            {renderWaitlistSection()}
            {upcomingAppointments.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="calendar-outline" size={64} color="#9ca3af" />
                <Text style={styles.emptyTitle}>No Upcoming Appointments</Text>
                <Text style={styles.emptyText}>Book an appointment with your dermatologist</Text>
                <TouchableOpacity
                  style={styles.emptyButton}
                  onPress={() => setActiveTab('book')}
                >
                  <Text style={styles.emptyButtonText}>Book Now</Text>
                </TouchableOpacity>
              </View>
            ) : (
              upcomingAppointments.map(renderAppointmentCard)
            )}
          </>
        )}

        {activeTab === 'past' && (
          <>
            {pastAppointments.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="time-outline" size={64} color="#9ca3af" />
                <Text style={styles.emptyTitle}>No Past Appointments</Text>
                <Text style={styles.emptyText}>Your appointment history will appear here</Text>
              </View>
            ) : (
              pastAppointments.map(renderAppointmentCard)
            )}
          </>
        )}

        {activeTab === 'book' && (
          <View style={styles.bookSection}>
            <Text style={styles.bookTitle}>Schedule an Appointment</Text>
            <Text style={styles.bookSubtitle}>
              Choose from our available appointment types to book with a dermatologist
            </Text>
            <TouchableOpacity
              style={styles.startBookingButton}
              onPress={() => setShowBookingModal(true)}
            >
              <Ionicons name="add-circle" size={24} color="#fff" />
              <Text style={styles.startBookingText}>Start Booking</Text>
            </TouchableOpacity>

            {/* Quick booking options */}
            <Text style={styles.quickBookTitle}>Quick Book</Text>
            <View style={styles.quickBookGrid}>
              {appointmentTypes.slice(0, 4).map(type => (
                <TouchableOpacity
                  key={type.type}
                  style={styles.quickBookCard}
                  onPress={() => {
                    setSelectedType(type);
                    setShowBookingModal(true);
                    const startDate = new Date();
                    const endDate = new Date();
                    endDate.setDate(endDate.getDate() + 7);
                    loadAvailableSlots(type.type, startDate, endDate);
                  }}
                >
                  <Ionicons
                    name={
                      type.type.includes('telemedicine') ? 'videocam' :
                      type.type.includes('follow') ? 'refresh' :
                      type.type.includes('skin_check') ? 'body' :
                      'medical'
                    }
                    size={28}
                    color="#2563eb"
                  />
                  <Text style={styles.quickBookName}>{type.name}</Text>
                  <Text style={styles.quickBookDuration}>{type.duration_minutes} min</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}

        <View style={styles.bottomSpacer} />
      </ScrollView>

      {renderBookingModal()}
      {renderConfirmModal()}
      {renderDetailModal()}
      {renderWaitlistModal()}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 20,
    paddingBottom: 15,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  headerSpacer: {
    width: 40,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    color: '#6b7280',
    fontSize: 16,
  },
  tabs: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
  },
  tabActive: {
    backgroundColor: '#dbeafe',
  },
  tabText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
  },
  tabTextActive: {
    color: '#2563eb',
  },
  badge: {
    backgroundColor: '#2563eb',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
    marginLeft: 6,
  },
  badgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  appointmentCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  appointmentType: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  providerName: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 2,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '700',
  },
  cardDetails: {
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
  },
  detailText: {
    fontSize: 14,
    color: '#6b7280',
  },
  todayBadge: {
    backgroundColor: '#fef3c7',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    marginLeft: 8,
  },
  todayText: {
    color: '#92400e',
    fontSize: 10,
    fontWeight: '700',
  },
  reasonText: {
    fontSize: 14,
    color: '#4b5563',
    marginBottom: 12,
    fontStyle: 'italic',
  },
  cardActions: {
    flexDirection: 'row',
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
    flex: 1,
  },
  confirmButton: {
    backgroundColor: '#3b82f6',
  },
  joinButton: {
    backgroundColor: '#10b981',
  },
  checkInButton: {
    backgroundColor: '#8b5cf6',
  },
  actionButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center',
  },
  emptyButton: {
    backgroundColor: '#2563eb',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 20,
  },
  emptyButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
  },
  bookSection: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  bookTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 8,
  },
  bookSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 24,
    paddingHorizontal: 20,
  },
  startBookingButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2563eb',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
    gap: 8,
  },
  startBookingText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 18,
  },
  quickBookTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e3a5f',
    marginTop: 32,
    marginBottom: 16,
    alignSelf: 'flex-start',
  },
  quickBookGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    width: '100%',
  },
  quickBookCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    width: '47%',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  quickBookName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
    marginTop: 8,
    textAlign: 'center',
  },
  quickBookDuration: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  bottomSpacer: {
    height: 40,
  },
  // Modal styles
  modalContainer: {
    flex: 1,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 20,
    paddingBottom: 15,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  modalContent: {
    flex: 1,
    padding: 16,
  },
  stepTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    marginTop: 20,
    marginBottom: 12,
  },
  typeCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  typeCardSelected: {
    borderColor: '#2563eb',
    backgroundColor: '#eff6ff',
  },
  typeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  typeInfo: {
    flex: 1,
  },
  typeName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  typeNameSelected: {
    color: '#2563eb',
  },
  typeDuration: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  typeDescription: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 8,
  },
  telemedicineBadge: {
    backgroundColor: '#dbeafe',
    padding: 6,
    borderRadius: 6,
  },
  telemedicineToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginTop: 12,
  },
  toggleLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  toggleText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  toggleSwitch: {
    width: 50,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#e5e7eb',
    justifyContent: 'center',
    paddingHorizontal: 2,
  },
  toggleSwitchOn: {
    backgroundColor: '#2563eb',
  },
  toggleKnob: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#fff',
  },
  toggleKnobOn: {
    alignSelf: 'flex-end',
  },
  dateScroller: {
    flexGrow: 0,
  },
  dateCard: {
    alignItems: 'center',
    padding: 12,
    marginRight: 10,
    borderRadius: 12,
    backgroundColor: '#fff',
    minWidth: 60,
  },
  dateCardSelected: {
    backgroundColor: '#2563eb',
  },
  dateCardWeekend: {
    opacity: 0.5,
  },
  dateDayName: {
    fontSize: 12,
    color: '#6b7280',
    fontWeight: '500',
  },
  dateDayNameSelected: {
    color: '#fff',
  },
  dateDay: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
    marginVertical: 4,
  },
  dateDaySelected: {
    color: '#fff',
  },
  dateMonth: {
    fontSize: 12,
    color: '#6b7280',
  },
  dateMonthSelected: {
    color: '#dbeafe',
  },
  loadingSlots: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    gap: 10,
  },
  loadingSlotsText: {
    color: '#6b7280',
  },
  noSlots: {
    alignItems: 'center',
    padding: 30,
  },
  noSlotsText: {
    color: '#6b7280',
    marginTop: 12,
    fontSize: 14,
  },
  waitlistButton: {
    backgroundColor: '#f59e0b',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
    marginTop: 16,
  },
  waitlistButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  slotsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  slotCard: {
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  slotCardSelected: {
    backgroundColor: '#2563eb',
    borderColor: '#2563eb',
  },
  slotTime: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  slotTimeSelected: {
    color: '#fff',
  },
  inputGroup: {
    marginBottom: 16,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
    marginBottom: 8,
  },
  textInput: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    fontSize: 14,
    textAlignVertical: 'top',
    minHeight: 80,
  },
  textInputSingle: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    fontSize: 14,
  },
  preparationSection: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
  },
  preparationTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2563eb',
    marginBottom: 12,
  },
  preparationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    marginBottom: 8,
  },
  preparationText: {
    flex: 1,
    fontSize: 13,
    color: '#4b5563',
  },
  bookButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563eb',
    borderRadius: 12,
    padding: 16,
    marginTop: 24,
    gap: 8,
  },
  bookButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  // Confirm modal
  confirmOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  confirmModal: {
    backgroundColor: '#fff',
    borderRadius: 20,
    padding: 24,
    width: '100%',
    maxWidth: 400,
  },
  confirmTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
    textAlign: 'center',
    marginBottom: 20,
  },
  confirmDetails: {
    backgroundColor: '#f8fafc',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  confirmRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  confirmLabel: {
    fontSize: 14,
    color: '#1e3a5f',
    flex: 1,
  },
  confirmReasonBox: {
    backgroundColor: '#fffbeb',
    borderRadius: 8,
    padding: 12,
    marginBottom: 20,
  },
  confirmReasonLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#92400e',
    marginBottom: 4,
  },
  confirmReasonText: {
    fontSize: 14,
    color: '#78350f',
  },
  confirmActions: {
    flexDirection: 'row',
    gap: 12,
  },
  confirmCancelButton: {
    flex: 1,
    padding: 14,
    borderRadius: 10,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
  },
  confirmCancelText: {
    color: '#6b7280',
    fontWeight: '600',
  },
  confirmBookButton: {
    flex: 1,
    flexDirection: 'row',
    padding: 14,
    borderRadius: 10,
    backgroundColor: '#2563eb',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  confirmBookText: {
    color: '#fff',
    fontWeight: '600',
  },
  // Detail modal
  detailCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
  },
  detailHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  detailType: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  detailSection: {
    marginBottom: 20,
  },
  detailItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
    marginBottom: 16,
  },
  detailItemLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
  detailItemValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  reasonSection: {
    backgroundColor: '#f8fafc',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  reasonLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 8,
  },
  reasonValue: {
    fontSize: 14,
    color: '#1e3a5f',
  },
  calendarSection: {
    marginBottom: 20,
  },
  calendarTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  calendarButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  calendarButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#f3f4f6',
    padding: 12,
    borderRadius: 8,
  },
  calendarButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  detailActions: {
    gap: 10,
  },
  detailActionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 14,
    borderRadius: 10,
  },
  confirmActionButton: {
    backgroundColor: '#3b82f6',
  },
  joinActionButton: {
    backgroundColor: '#10b981',
  },
  checkInActionButton: {
    backgroundColor: '#8b5cf6',
  },
  cancelActionButton: {
    backgroundColor: '#fef2f2',
  },
  detailActionText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 15,
  },
  // Waitlist
  waitlistSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 20,
  },
  flexibilityOptions: {
    flexDirection: 'row',
    gap: 8,
  },
  flexibilityOption: {
    flex: 1,
    padding: 10,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
  },
  flexibilityOptionSelected: {
    backgroundColor: '#2563eb',
  },
  flexibilityText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
  },
  flexibilityTextSelected: {
    color: '#fff',
  },
  waitlistSection: {
    marginBottom: 20,
  },
  waitlistTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  waitlistCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#fffbeb',
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
  },
  waitlistInfo: {
    flex: 1,
  },
  waitlistType: {
    fontSize: 14,
    fontWeight: '600',
    color: '#92400e',
  },
  waitlistFlexibility: {
    fontSize: 12,
    color: '#b45309',
    marginTop: 2,
  },
});
