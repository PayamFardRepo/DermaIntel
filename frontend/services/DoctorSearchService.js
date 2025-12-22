import * as Location from 'expo-location';
import { Platform, Linking } from 'react-native';

/**
 * Service for finding doctors and medical specialists near the user's location
 * Uses Google Places API to search for healthcare providers
 */
class DoctorSearchService {
  constructor() {
    // You can add your Google Places API key here
    // Get one from: https://developers.google.com/maps/documentation/places/web-service/get-api-key
    this.GOOGLE_PLACES_API_KEY = process.env.EXPO_PUBLIC_GOOGLE_PLACES_API_KEY || '';
  }

  /**
   * Request location permissions from the user
   */
  async requestLocationPermission() {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      return status === 'granted';
    } catch (error) {
      console.error('Error requesting location permission:', error);
      return false;
    }
  }

  /**
   * Get the user's current location
   */
  async getCurrentLocation() {
    try {
      const hasPermission = await this.requestLocationPermission();

      if (!hasPermission) {
        throw new Error('Location permission not granted');
      }

      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
      });

      return {
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
      };
    } catch (error) {
      console.error('Error getting location:', error);
      throw error;
    }
  }

  /**
   * Search for doctors/specialists near a location
   * @param {string} specialistType - Type of specialist (e.g., 'dermatologist', 'oncologist')
   * @param {number} latitude - Latitude coordinate
   * @param {number} longitude - Longitude coordinate
   * @param {number} radius - Search radius in meters (default: 8000m / ~5 miles)
   */
  async searchNearbyDoctors(specialistType, latitude, longitude, radius = 8000) {
    try {
      // If no API key, return mock data
      if (!this.GOOGLE_PLACES_API_KEY) {
        console.warn('No Google Places API key configured. Returning mock data.');
        return this.getMockDoctors(specialistType);
      }

      // Construct search query based on specialist type
      const searchQuery = this.getSearchQuery(specialistType);

      // Google Places API Nearby Search
      const url = `https://maps.googleapis.com/maps/api/place/nearbysearch/json?` +
        `location=${latitude},${longitude}` +
        `&radius=${radius}` +
        `&keyword=${encodeURIComponent(searchQuery)}` +
        `&type=doctor` +
        `&key=${this.GOOGLE_PLACES_API_KEY}`;

      const response = await fetch(url);
      const data = await response.json();

      if (data.status !== 'OK' && data.status !== 'ZERO_RESULTS') {
        throw new Error(`Places API error: ${data.status}`);
      }

      // Parse and format the results
      const doctors = (data.results || []).slice(0, 5).map(place => ({
        name: place.name,
        address: place.vicinity,
        rating: place.rating || 'N/A',
        userRatingsTotal: place.user_ratings_total || 0,
        isOpen: place.opening_hours?.open_now ?? null,
        distance: this.calculateDistance(latitude, longitude, place.geometry.location.lat, place.geometry.location.lng),
        placeId: place.place_id,
        location: {
          lat: place.geometry.location.lat,
          lng: place.geometry.location.lng,
        }
      }));

      return doctors;
    } catch (error) {
      console.error('Error searching for doctors:', error);
      // Return mock data as fallback
      return this.getMockDoctors(specialistType);
    }
  }

  /**
   * Get additional details about a specific doctor/place
   * @param {string} placeId - Google Places place_id
   */
  async getDoctorDetails(placeId) {
    try {
      if (!this.GOOGLE_PLACES_API_KEY) {
        return null;
      }

      const url = `https://maps.googleapis.com/maps/api/place/details/json?` +
        `place_id=${placeId}` +
        `&fields=name,formatted_address,formatted_phone_number,website,opening_hours,rating,reviews` +
        `&key=${this.GOOGLE_PLACES_API_KEY}`;

      const response = await fetch(url);
      const data = await response.json();

      if (data.status !== 'OK') {
        throw new Error(`Places API error: ${data.status}`);
      }

      return {
        name: data.result.name,
        address: data.result.formatted_address,
        phone: data.result.formatted_phone_number,
        website: data.result.website,
        rating: data.result.rating,
        reviews: data.result.reviews || [],
        openingHours: data.result.opening_hours?.weekday_text || [],
      };
    } catch (error) {
      console.error('Error getting doctor details:', error);
      return null;
    }
  }

  /**
   * Get search query based on specialist type
   */
  getSearchQuery(specialistType) {
    const queryMap = {
      'Dermatologist': 'dermatologist skin doctor',
      'Surgical Oncologist': 'surgical oncologist cancer surgeon',
      'Medical Oncologist': 'medical oncologist cancer doctor',
      'Radiation Oncologist': 'radiation oncologist',
      'Mohs Surgeon': 'mohs surgery dermatologist',
      'Plastic Surgeon': 'plastic surgeon reconstructive',
      'Primary Care Physician': 'primary care physician family doctor',
      'Allergist': 'allergist allergy doctor',
      'Rheumatologist': 'rheumatologist',
      'Endocrinologist': 'endocrinologist',
      'Ophthalmologist': 'ophthalmologist eye doctor',
      'Immunologist': 'immunologist',
    };

    return queryMap[specialistType] || `${specialistType} doctor`;
  }

  /**
   * Calculate distance between two coordinates (Haversine formula)
   * Returns distance in miles
   */
  calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 3959; // Earth's radius in miles
    const dLat = this.toRad(lat2 - lat1);
    const dLon = this.toRad(lon2 - lon1);

    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(this.toRad(lat1)) * Math.cos(this.toRad(lat2)) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);

    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    const distance = R * c;

    return Math.round(distance * 10) / 10; // Round to 1 decimal place
  }

  toRad(degrees) {
    return degrees * (Math.PI / 180);
  }

  /**
   * Get mock doctor data for testing or when API is not available
   */
  getMockDoctors(specialistType) {
    const mockData = {
      'Dermatologist': [
        { name: 'Dr. Sarah Johnson, MD - Dermatology Associates', address: '123 Medical Plaza, Suite 200', rating: 4.8, userRatingsTotal: 234, isOpen: true, distance: 2.3, phone: '(555) 123-4567' },
        { name: 'Dr. Michael Chen - Advanced Skin Care Center', address: '456 Healthcare Blvd, Floor 3', rating: 4.9, userRatingsTotal: 189, isOpen: true, distance: 3.1, phone: '(555) 234-5678' },
        { name: 'Dr. Emily Rodriguez - City Dermatology Clinic', address: '789 Main Street', rating: 4.7, userRatingsTotal: 156, isOpen: false, distance: 4.5, phone: '(555) 345-6789' },
        { name: 'Dr. David Williams - Comprehensive Dermatology', address: '321 Oak Avenue', rating: 4.6, userRatingsTotal: 203, isOpen: true, distance: 5.2, phone: '(555) 456-7890' },
        { name: 'Dr. Lisa Thompson - Premier Skin Specialists', address: '654 Pine Road', rating: 4.8, userRatingsTotal: 178, isOpen: true, distance: 6.1, phone: '(555) 567-8901' },
      ],
      'Surgical Oncologist': [
        { name: 'Dr. Robert Anderson - Cancer Surgery Center', address: '100 Hospital Drive', rating: 4.9, userRatingsTotal: 145, isOpen: true, distance: 3.4, phone: '(555) 678-9012' },
        { name: 'Dr. Jennifer Lee - Oncology Surgical Associates', address: '200 Medical Center Way', rating: 4.8, userRatingsTotal: 132, isOpen: true, distance: 4.2, phone: '(555) 789-0123' },
        { name: 'Dr. Thomas Martinez - Advanced Oncology Surgery', address: '300 Health Plaza', rating: 4.7, userRatingsTotal: 98, isOpen: false, distance: 5.8, phone: '(555) 890-1234' },
        { name: 'Dr. Patricia White - Regional Cancer Institute', address: '400 University Blvd', rating: 4.9, userRatingsTotal: 167, isOpen: true, distance: 6.5, phone: '(555) 901-2345' },
        { name: 'Dr. James Taylor - Comprehensive Cancer Care', address: '500 Medical Park', rating: 4.6, userRatingsTotal: 112, isOpen: true, distance: 7.2, phone: '(555) 012-3456' },
      ],
      'Primary Care Physician': [
        { name: 'Dr. Amanda Foster - Family Health Center', address: '111 Community Drive', rating: 4.7, userRatingsTotal: 289, isOpen: true, distance: 1.2, phone: '(555) 111-2222' },
        { name: 'Dr. Christopher Moore - Primary Care Clinic', address: '222 Wellness Way', rating: 4.8, userRatingsTotal: 256, isOpen: true, distance: 1.8, phone: '(555) 222-3333' },
        { name: 'Dr. Maria Garcia - Downtown Medical Group', address: '333 Central Avenue', rating: 4.6, userRatingsTotal: 198, isOpen: true, distance: 2.5, phone: '(555) 333-4444' },
        { name: 'Dr. Kevin Brown - Neighborhood Health', address: '444 Local Street', rating: 4.7, userRatingsTotal: 223, isOpen: false, distance: 3.2, phone: '(555) 444-5555' },
        { name: 'Dr. Rachel Kim - Community Care Physicians', address: '555 City Plaza', rating: 4.9, userRatingsTotal: 267, isOpen: true, distance: 3.9, phone: '(555) 555-6666' },
      ],
    };

    // Return specific mock data or generic dermatologist data
    return mockData[specialistType] || mockData['Dermatologist'];
  }

  /**
   * Open the location in a map app
   */
  openInMaps(latitude, longitude, label) {
    const scheme = Platform.select({
      ios: 'maps:',
      android: 'geo:',
    });
    const url = Platform.select({
      ios: `${scheme}${latitude},${longitude}?q=${encodeURIComponent(label)}`,
      android: `${scheme}${latitude},${longitude}?q=${encodeURIComponent(label)}`,
    });

    Linking.openURL(url);
  }

  /**
   * Call a phone number
   */
  callPhone(phoneNumber) {
    const url = `tel:${phoneNumber.replace(/[^0-9]/g, '')}`;
    Linking.openURL(url);
  }
}

export default new DoctorSearchService();
