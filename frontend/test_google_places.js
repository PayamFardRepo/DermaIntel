/**
 * Test script to verify Google Places API is working
 * Run this after adding your API key to .env
 *
 * Usage: node test_google_places.js
 */

// Load environment variables
require('dotenv').config();

const API_KEY = process.env.EXPO_PUBLIC_GOOGLE_PLACES_API_KEY;

console.log('\n🔍 Testing Google Places API Configuration...\n');

// Check if API key is set
if (!API_KEY || API_KEY === 'YOUR_API_KEY_HERE') {
  console.log('❌ ERROR: API key not configured!');
  console.log('   Please add your Google Places API key to the .env file');
  console.log('   File location: frontend/.env\n');
  process.exit(1);
}

console.log('✅ API Key found:', API_KEY.substring(0, 20) + '...');

// Test API with a sample location (New York City)
const testLocation = {
  latitude: 40.7128,
  longitude: -74.0060,
  name: 'New York City'
};

console.log(`\n🔎 Searching for dermatologists near ${testLocation.name}...\n`);

const url = `https://maps.googleapis.com/maps/api/place/nearbysearch/json?` +
  `location=${testLocation.latitude},${testLocation.longitude}` +
  `&radius=5000` +
  `&keyword=dermatologist` +
  `&type=doctor` +
  `&key=${API_KEY}`;

fetch(url)
  .then(response => response.json())
  .then(data => {
    if (data.status === 'OK') {
      console.log('✅ SUCCESS! Google Places API is working!\n');
      console.log(`📊 Found ${data.results.length} doctors\n`);

      // Show first 3 results
      console.log('First 3 doctors found:\n');
      data.results.slice(0, 3).forEach((place, index) => {
        console.log(`${index + 1}. ${place.name}`);
        console.log(`   Address: ${place.vicinity}`);
        console.log(`   Rating: ${place.rating || 'N/A'} ⭐`);
        console.log(`   Status: ${place.opening_hours?.open_now ? 'Open' : 'Closed'}\n`);
      });

      console.log('✨ Your app will now show REAL doctors!\n');
    } else if (data.status === 'REQUEST_DENIED') {
      console.log('❌ ERROR: Request denied by Google');
      console.log('   Possible issues:');
      console.log('   1. Places API is not enabled in Google Cloud Console');
      console.log('   2. Billing is not set up');
      console.log('   3. API key restrictions are too strict');
      console.log(`   4. Error message: ${data.error_message || 'None'}\n`);
    } else if (data.status === 'ZERO_RESULTS') {
      console.log('⚠️  No results found (this is OK - API is working!)');
      console.log('   The API is configured correctly.\n');
    } else {
      console.log(`❌ ERROR: ${data.status}`);
      console.log(`   Message: ${data.error_message || 'Unknown error'}\n`);
    }
  })
  .catch(error => {
    console.log('❌ ERROR: Failed to connect to Google Places API');
    console.log(`   ${error.message}\n`);
  });
