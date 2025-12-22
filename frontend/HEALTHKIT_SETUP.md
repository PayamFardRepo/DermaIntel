# Apple HealthKit Integration Setup

This guide explains how to set up the Apple HealthKit integration for real Apple Watch data syncing.

## Prerequisites

1. **Apple Developer Account** ($99/year) - Required for HealthKit entitlements
2. **Mac with Xcode** - Required to build the iOS app
3. **Physical iPhone** - HealthKit doesn't work on simulators
4. **Apple Watch** (optional) - For workout and activity data

## Step 1: Install EAS CLI

```bash
npm install -g eas-cli
```

## Step 2: Login to Expo

```bash
eas login
```

## Step 3: Configure EAS Build

Create an `eas.json` file in the frontend folder if it doesn't exist:

```json
{
  "cli": {
    "version": ">= 3.0.0"
  },
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal",
      "ios": {
        "simulator": false
      }
    },
    "preview": {
      "distribution": "internal"
    },
    "production": {}
  },
  "submit": {
    "production": {}
  }
}
```

## Step 4: Create Development Build

Run this command to create a development build for iOS:

```bash
cd frontend
eas build --profile development --platform ios
```

This will:
1. Upload your project to Expo's build servers
2. Build a custom development client with HealthKit enabled
3. Provide a link to download and install the app

## Step 5: Install on Your iPhone

1. When the build completes, you'll get a QR code or download link
2. Scan the QR code with your iPhone camera
3. Install the development build (you may need to trust the developer in Settings > General > Device Management)

## Step 6: Run the App

Start the development server:

```bash
npx expo start --dev-client
```

Then open the installed development app on your iPhone and connect to the server.

## Using HealthKit

Once the app is running:

1. Go to **Wearables** screen
2. Tap the **+** button
3. Select **Apple Health**
4. Grant permissions when prompted
5. Your workout and activity data will sync automatically

## What Data is Synced

| Data Type | Source | Notes |
|-----------|--------|-------|
| Outdoor Workouts | Apple Watch/iPhone | Walking, running, cycling, etc. |
| Activity Data | Apple Watch | Steps, active energy, exercise minutes |
| Heart Rate | Apple Watch | Used for activity detection |
| UV Exposure | Third-party apps | If you use a UV tracking app |

## Troubleshooting

### "HealthKit is not available"
- Make sure you're using the development build, not Expo Go
- Ensure you're on a physical iPhone, not a simulator

### "Authorization Denied"
- Go to Settings > Privacy > Health > SkinLesionDetection
- Enable all the data types you want to share

### Build Fails
- Make sure your Apple Developer account is properly configured
- Check that you have accepted the latest Apple Developer agreements

## Local Development (Alternative)

If you prefer to build locally instead of using EAS:

1. Generate native project:
   ```bash
   npx expo prebuild
   ```

2. Open in Xcode:
   ```bash
   open ios/SkinLesionDetection.xcworkspace
   ```

3. Configure signing with your Apple Developer account

4. Build and run on your device

## Cost Summary

| Item | Cost |
|------|------|
| Apple Developer Account | $99/year |
| HealthKit Library | Free (open source) |
| EAS Build | Free tier available (30 builds/month) |
| **Total** | **$99/year** |
