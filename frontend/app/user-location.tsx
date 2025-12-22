import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
  ActivityIndicator,
  Alert,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

interface LocationData {
  city: string;
  state: string;
  country: string;
  zip_code: string;
  latitude: number | null;
  longitude: number | null;
  preferred_distance_miles: number;
}

const US_STATES = [
  { code: 'AL', name: 'Alabama' },
  { code: 'AK', name: 'Alaska' },
  { code: 'AZ', name: 'Arizona' },
  { code: 'AR', name: 'Arkansas' },
  { code: 'CA', name: 'California' },
  { code: 'CO', name: 'Colorado' },
  { code: 'CT', name: 'Connecticut' },
  { code: 'DE', name: 'Delaware' },
  { code: 'FL', name: 'Florida' },
  { code: 'GA', name: 'Georgia' },
  { code: 'HI', name: 'Hawaii' },
  { code: 'ID', name: 'Idaho' },
  { code: 'IL', name: 'Illinois' },
  { code: 'IN', name: 'Indiana' },
  { code: 'IA', name: 'Iowa' },
  { code: 'KS', name: 'Kansas' },
  { code: 'KY', name: 'Kentucky' },
  { code: 'LA', name: 'Louisiana' },
  { code: 'ME', name: 'Maine' },
  { code: 'MD', name: 'Maryland' },
  { code: 'MA', name: 'Massachusetts' },
  { code: 'MI', name: 'Michigan' },
  { code: 'MN', name: 'Minnesota' },
  { code: 'MS', name: 'Mississippi' },
  { code: 'MO', name: 'Missouri' },
  { code: 'MT', name: 'Montana' },
  { code: 'NE', name: 'Nebraska' },
  { code: 'NV', name: 'Nevada' },
  { code: 'NH', name: 'New Hampshire' },
  { code: 'NJ', name: 'New Jersey' },
  { code: 'NM', name: 'New Mexico' },
  { code: 'NY', name: 'New York' },
  { code: 'NC', name: 'North Carolina' },
  { code: 'ND', name: 'North Dakota' },
  { code: 'OH', name: 'Ohio' },
  { code: 'OK', name: 'Oklahoma' },
  { code: 'OR', name: 'Oregon' },
  { code: 'PA', name: 'Pennsylvania' },
  { code: 'RI', name: 'Rhode Island' },
  { code: 'SC', name: 'South Carolina' },
  { code: 'SD', name: 'South Dakota' },
  { code: 'TN', name: 'Tennessee' },
  { code: 'TX', name: 'Texas' },
  { code: 'UT', name: 'Utah' },
  { code: 'VT', name: 'Vermont' },
  { code: 'VA', name: 'Virginia' },
  { code: 'WA', name: 'Washington' },
  { code: 'WV', name: 'West Virginia' },
  { code: 'WI', name: 'Wisconsin' },
  { code: 'WY', name: 'Wyoming' },
];

const DISTANCE_OPTIONS = [10, 25, 50, 100, 200];

export default function UserLocation() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isGettingLocation, setIsGettingLocation] = useState(false);
  const [showStatePicker, setShowStatePicker] = useState(false);

  const [location, setLocation] = useState<LocationData>({
    city: '',
    state: '',
    country: 'USA',
    zip_code: '',
    latitude: null,
    longitude: null,
    preferred_distance_miles: 50,
  });

  const [nearbyDermatologists, setNearbyDermatologists] = useState<number | null>(null);

  useEffect(() => {
    fetchCurrentLocation();
  }, []);

  const fetchCurrentLocation = async () => {
    try {
      setIsLoading(true);
      const token = await AsyncStorage.getItem('accessToken');
      if (!token) {
        router.replace('/');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/profile`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const profile = await response.json();
        setLocation({
          city: profile.city || '',
          state: profile.state || '',
          country: profile.country || 'USA',
          zip_code: profile.zip_code || '',
          latitude: profile.latitude || null,
          longitude: profile.longitude || null,
          preferred_distance_miles: profile.preferred_distance_miles || 50,
        });
      }
    } catch (error) {
      console.error('Error fetching location:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getCurrentLocation = async () => {
    try {
      setIsGettingLocation(true);

      // Try to use expo-location if available
      let Location;
      try {
        Location = require('expo-location');
      } catch {
        Alert.alert(
          'Location Not Available',
          'Please install expo-location package or enter your address manually.'
        );
        setIsGettingLocation(false);
        return;
      }

      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          'Permission Denied',
          'Please enable location permissions in your device settings to use this feature.'
        );
        setIsGettingLocation(false);
        return;
      }

      const currentLocation = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
      });

      setLocation(prev => ({
        ...prev,
        latitude: currentLocation.coords.latitude,
        longitude: currentLocation.coords.longitude,
      }));

      // Try to reverse geocode
      try {
        const [address] = await Location.reverseGeocodeAsync({
          latitude: currentLocation.coords.latitude,
          longitude: currentLocation.coords.longitude,
        });

        if (address) {
          setLocation(prev => ({
            ...prev,
            city: address.city || prev.city,
            state: address.region || prev.state,
            country: address.country || 'USA',
            zip_code: address.postalCode || prev.zip_code,
          }));
        }
      } catch (geocodeError) {
        console.log('Reverse geocoding not available');
      }

      Alert.alert('Success', 'Location updated from GPS');
    } catch (error) {
      console.error('Error getting location:', error);
      Alert.alert('Error', 'Failed to get current location. Please enter manually.');
    } finally {
      setIsGettingLocation(false);
    }
  };

  const saveLocation = async () => {
    try {
      setIsSaving(true);
      const token = await AsyncStorage.getItem('accessToken');
      if (!token) {
        router.replace('/');
        return;
      }

      const formData = new FormData();
      if (location.city) formData.append('city', location.city);
      if (location.state) formData.append('state', location.state);
      if (location.country) formData.append('country', location.country);
      if (location.zip_code) formData.append('zip_code', location.zip_code);
      if (location.latitude !== null) formData.append('latitude', location.latitude.toString());
      if (location.longitude !== null) formData.append('longitude', location.longitude.toString());
      formData.append('preferred_distance_miles', location.preferred_distance_miles.toString());

      const response = await fetch(`${API_BASE_URL}/users/me/location`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        Alert.alert('Success', 'Location saved successfully');
        // Check for nearby dermatologists
        findNearbyDermatologists();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to save location');
      }
    } catch (error) {
      console.error('Error saving location:', error);
      Alert.alert('Error', 'Failed to save location');
    } finally {
      setIsSaving(false);
    }
  };

  const findNearbyDermatologists = async () => {
    try {
      const token = await AsyncStorage.getItem('accessToken');
      if (!token) return;

      const params = new URLSearchParams();
      if (location.city) params.append('city', location.city);
      if (location.state) params.append('state', location.state);
      if (location.latitude) params.append('latitude', location.latitude.toString());
      if (location.longitude) params.append('longitude', location.longitude.toString());
      params.append('max_distance_miles', location.preferred_distance_miles.toString());

      const response = await fetch(`${API_BASE_URL}/specialists/search?${params.toString()}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setNearbyDermatologists(data.dermatologists?.length || 0);
      }
    } catch (error) {
      console.log('Could not fetch nearby dermatologists');
    }
  };

  const getStateName = (code: string) => {
    const state = US_STATES.find(s => s.code === code || s.name === code);
    return state ? state.name : code;
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <LinearGradient colors={['#667eea', '#764ba2']} style={styles.header}>
          <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
            <Text style={styles.backButtonText}>Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>My Location</Text>
          <View style={styles.headerRight} />
        </LinearGradient>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#667eea" />
          <Text style={styles.loadingText}>Loading location...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient colors={['#667eea', '#764ba2']} style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Text style={styles.backButtonText}>Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>My Location</Text>
        <View style={styles.headerRight} />
      </LinearGradient>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* GPS Location Button */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardIcon}>📍</Text>
            <Text style={styles.cardTitle}>Use Current Location</Text>
          </View>
          <Text style={styles.cardDescription}>
            Automatically detect your location using GPS for more accurate dermatologist matching.
          </Text>
          <TouchableOpacity
            style={[styles.gpsButton, isGettingLocation && styles.gpsButtonDisabled]}
            onPress={getCurrentLocation}
            disabled={isGettingLocation}
          >
            {isGettingLocation ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <>
                <Text style={styles.gpsButtonIcon}>🛰️</Text>
                <Text style={styles.gpsButtonText}>Get Current Location</Text>
              </>
            )}
          </TouchableOpacity>
          {location.latitude && location.longitude && (
            <View style={styles.coordinatesContainer}>
              <Text style={styles.coordinatesLabel}>GPS Coordinates:</Text>
              <Text style={styles.coordinatesValue}>
                {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}
              </Text>
            </View>
          )}
        </View>

        {/* Manual Address Entry */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardIcon}>🏠</Text>
            <Text style={styles.cardTitle}>Address</Text>
          </View>

          <View style={styles.formGroup}>
            <Text style={styles.label}>City</Text>
            <TextInput
              style={styles.input}
              value={location.city}
              onChangeText={(text) => setLocation({ ...location, city: text })}
              placeholder="Enter your city"
              placeholderTextColor="#999"
            />
          </View>

          <View style={styles.formGroup}>
            <Text style={styles.label}>State</Text>
            <TouchableOpacity
              style={styles.pickerButton}
              onPress={() => setShowStatePicker(!showStatePicker)}
            >
              <Text style={location.state ? styles.pickerButtonText : styles.pickerPlaceholder}>
                {location.state ? getStateName(location.state) : 'Select state'}
              </Text>
              <Text style={styles.pickerChevron}>{showStatePicker ? '▲' : '▼'}</Text>
            </TouchableOpacity>
            {showStatePicker && (
              <View style={styles.statePickerContainer}>
                <ScrollView style={styles.statePicker} nestedScrollEnabled>
                  {US_STATES.map((state) => (
                    <TouchableOpacity
                      key={state.code}
                      style={[
                        styles.stateOption,
                        location.state === state.code && styles.stateOptionSelected,
                      ]}
                      onPress={() => {
                        setLocation({ ...location, state: state.code });
                        setShowStatePicker(false);
                      }}
                    >
                      <Text
                        style={[
                          styles.stateOptionText,
                          location.state === state.code && styles.stateOptionTextSelected,
                        ]}
                      >
                        {state.name} ({state.code})
                      </Text>
                    </TouchableOpacity>
                  ))}
                </ScrollView>
              </View>
            )}
          </View>

          <View style={styles.formGroup}>
            <Text style={styles.label}>ZIP Code</Text>
            <TextInput
              style={styles.input}
              value={location.zip_code}
              onChangeText={(text) => setLocation({ ...location, zip_code: text })}
              placeholder="Enter ZIP code"
              placeholderTextColor="#999"
              keyboardType="number-pad"
              maxLength={10}
            />
          </View>

          <View style={styles.formGroup}>
            <Text style={styles.label}>Country</Text>
            <TextInput
              style={[styles.input, styles.inputDisabled]}
              value={location.country}
              editable={false}
            />
            <Text style={styles.helperText}>Currently only USA is supported</Text>
          </View>
        </View>

        {/* Search Distance Preference */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardIcon}>🔍</Text>
            <Text style={styles.cardTitle}>Search Preferences</Text>
          </View>
          <Text style={styles.cardDescription}>
            Set how far you're willing to travel to see a dermatologist.
          </Text>

          <View style={styles.distanceContainer}>
            <Text style={styles.distanceLabel}>Maximum Distance</Text>
            <View style={styles.distanceOptions}>
              {DISTANCE_OPTIONS.map((distance) => (
                <TouchableOpacity
                  key={distance}
                  style={[
                    styles.distanceOption,
                    location.preferred_distance_miles === distance && styles.distanceOptionSelected,
                  ]}
                  onPress={() => setLocation({ ...location, preferred_distance_miles: distance })}
                >
                  <Text
                    style={[
                      styles.distanceOptionText,
                      location.preferred_distance_miles === distance && styles.distanceOptionTextSelected,
                    ]}
                  >
                    {distance} mi
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {nearbyDermatologists !== null && (
            <View style={styles.dermatologistCountContainer}>
              <Text style={styles.dermatologistCountIcon}>👨‍⚕️</Text>
              <Text style={styles.dermatologistCountText}>
                {nearbyDermatologists} dermatologist{nearbyDermatologists !== 1 ? 's' : ''} found within {location.preferred_distance_miles} miles
              </Text>
            </View>
          )}
        </View>

        {/* Location Summary */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardIcon}>📋</Text>
            <Text style={styles.cardTitle}>Location Summary</Text>
          </View>
          <View style={styles.summaryContainer}>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Address:</Text>
              <Text style={styles.summaryValue}>
                {location.city && location.state
                  ? `${location.city}, ${location.state} ${location.zip_code || ''}`
                  : 'Not set'}
              </Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>GPS:</Text>
              <Text style={styles.summaryValue}>
                {location.latitude && location.longitude
                  ? `${location.latitude.toFixed(4)}, ${location.longitude.toFixed(4)}`
                  : 'Not set'}
              </Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Search Radius:</Text>
              <Text style={styles.summaryValue}>{location.preferred_distance_miles} miles</Text>
            </View>
          </View>
        </View>

        {/* Save Button */}
        <TouchableOpacity
          style={[styles.saveButton, isSaving && styles.saveButtonDisabled]}
          onPress={saveLocation}
          disabled={isSaving}
        >
          {isSaving ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.saveButtonText}>Save Location</Text>
          )}
        </TouchableOpacity>

        {/* Quick Actions */}
        <View style={styles.quickActions}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => router.push('/consultations' as any)}
          >
            <Text style={styles.actionButtonIcon}>👨‍⚕️</Text>
            <Text style={styles.actionButtonText}>Find Dermatologists</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => router.push('/profile' as any)}
          >
            <Text style={styles.actionButtonIcon}>👤</Text>
            <Text style={styles.actionButtonText}>Edit Profile</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.bottomSpacer} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f7fa',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  backButton: {
    padding: 5,
  },
  backButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  headerRight: {
    width: 50,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    color: '#666',
    fontSize: 16,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  cardIcon: {
    fontSize: 24,
    marginRight: 10,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  cardDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
    lineHeight: 20,
  },
  gpsButton: {
    backgroundColor: '#667eea',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 14,
    borderRadius: 10,
  },
  gpsButtonDisabled: {
    opacity: 0.7,
  },
  gpsButtonIcon: {
    fontSize: 18,
    marginRight: 8,
  },
  gpsButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  coordinatesContainer: {
    marginTop: 12,
    padding: 12,
    backgroundColor: '#f0f4ff',
    borderRadius: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  coordinatesLabel: {
    fontSize: 14,
    color: '#666',
  },
  coordinatesValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#667eea',
  },
  formGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#f5f7fa',
    borderRadius: 10,
    padding: 14,
    fontSize: 16,
    color: '#333',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  inputDisabled: {
    backgroundColor: '#eee',
    color: '#999',
  },
  helperText: {
    fontSize: 12,
    color: '#999',
    marginTop: 4,
  },
  pickerButton: {
    backgroundColor: '#f5f7fa',
    borderRadius: 10,
    padding: 14,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  pickerButtonText: {
    fontSize: 16,
    color: '#333',
  },
  pickerPlaceholder: {
    fontSize: 16,
    color: '#999',
  },
  pickerChevron: {
    fontSize: 12,
    color: '#666',
  },
  statePickerContainer: {
    marginTop: 8,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    overflow: 'hidden',
  },
  statePicker: {
    maxHeight: 200,
    backgroundColor: '#fff',
  },
  stateOption: {
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  stateOptionSelected: {
    backgroundColor: '#f0f4ff',
  },
  stateOptionText: {
    fontSize: 15,
    color: '#333',
  },
  stateOptionTextSelected: {
    color: '#667eea',
    fontWeight: '600',
  },
  distanceContainer: {
    marginTop: 8,
  },
  distanceLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  distanceOptions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  distanceOption: {
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 20,
    backgroundColor: '#f5f7fa',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  distanceOptionSelected: {
    backgroundColor: '#667eea',
    borderColor: '#667eea',
  },
  distanceOptionText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#666',
  },
  distanceOptionTextSelected: {
    color: '#fff',
  },
  dermatologistCountContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 16,
    padding: 12,
    backgroundColor: '#e8f5e9',
    borderRadius: 8,
  },
  dermatologistCountIcon: {
    fontSize: 20,
    marginRight: 10,
  },
  dermatologistCountText: {
    fontSize: 14,
    color: '#2e7d32',
    flex: 1,
  },
  summaryContainer: {
    backgroundColor: '#f5f7fa',
    borderRadius: 10,
    padding: 16,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  summaryLabel: {
    fontSize: 14,
    color: '#666',
  },
  summaryValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    textAlign: 'right',
    flex: 1,
    marginLeft: 10,
  },
  saveButton: {
    backgroundColor: '#667eea',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 20,
  },
  saveButtonDisabled: {
    opacity: 0.7,
  },
  saveButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  quickActions: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  actionButtonIcon: {
    fontSize: 24,
    marginBottom: 8,
  },
  actionButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
    textAlign: 'center',
  },
  bottomSpacer: {
    height: 40,
  },
});
