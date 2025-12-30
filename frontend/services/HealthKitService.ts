/**
 * HealthKit Service
 *
 * Provides direct integration with Apple HealthKit for reading health data
 * from Apple Watch and iPhone using react-native-health.
 *
 * Requirements:
 * - Expo Development Build with HealthKit native code
 * - Apple Developer Account ($99/year)
 * - Physical iPhone with Apple Watch paired
 *
 * IMPORTANT: You need to rebuild the dev client after adding HealthKit:
 * cd frontend && eas build --profile development --platform ios
 */

import { Platform } from 'react-native';

// Types for health data
export interface UVExposureReading {
  startDate: Date;
  endDate: Date;
  value: number; // UV index
  sourceName: string;
}

export interface WorkoutData {
  id: string;
  workoutType: string;
  startDate: Date;
  endDate: Date;
  duration: number; // seconds
  totalEnergyBurned?: number;
  totalDistance?: number;
  sourceName: string;
  isOutdoor: boolean;
  // Location data for UV lookup
  startLocation?: { latitude: number; longitude: number };
  endLocation?: { latitude: number; longitude: number };
}

export interface ActivityData {
  date: Date;
  activeEnergyBurned: number;
  stepCount: number;
  exerciseMinutes: number;
  standHours: number;
}

export interface HealthKitStatus {
  isAvailable: boolean;
  isAuthorized: boolean;
  error?: string;
  needsRebuild?: boolean;
}

export interface SyncResult {
  success: boolean;
  message: string;
  workoutsCount?: number;
  outdoorMinutes?: number;
  uvReadingsCount?: number;
}

// HealthKit module - dynamically loaded
let AppleHealthKit: any = null;
let healthKitLoadError: string | null = null;
let isHealthKitLoaded = false;

// Permissions we request from HealthKit
const HEALTHKIT_PERMISSIONS = {
  permissions: {
    read: [
      'ActiveEnergyBurned',
      'BasalEnergyBurned',
      'StepCount',
      'DistanceWalkingRunning',
      'DistanceCycling',
      'FlightsClimbed',
      'HeartRate',
      'RestingHeartRate',
      'HeartRateVariability',
      'Workout',
      'SleepAnalysis',
      'MindfulSession',
      'AppleExerciseTime',
      'AppleStandTime',
    ],
    write: [],
  },
};

// Outdoor workout types
const OUTDOOR_WORKOUT_TYPES = [
  'Walking',
  'Running',
  'Cycling',
  'Hiking',
  'Golf',
  'Tennis',
  'Soccer',
  'Basketball',
  'Baseball',
  'Softball',
  'Volleyball',
  'AmericanFootball',
  'Lacrosse',
  'Rugby',
  'Sailing',
  'Surfing',
  'Skateboarding',
  'Snowboarding',
  'CrossCountrySkiing',
  'DownhillSkiing',
  'Paddling',
  'Rowing',
  'TraditionalStrengthTraining', // Outdoor gyms
  'FunctionalStrengthTraining',
  'CrossTraining',
  'Elliptical', // Sometimes outdoor
  'StairClimbing', // Stadium stairs
  'Swimming', // Outdoor swimming
  'OpenWaterSwim',
  'PaddleSports',
  'Play', // Kids playing outside
  'Yoga', // Outdoor yoga
  'TrackAndField',
  'Triathlon',
  'WaterFitness',
  'WaterPolo',
  'WaterSports',
  'Other',
];

/**
 * Try to load the HealthKit native module
 */
const loadHealthKit = async (): Promise<boolean> => {
  if (isHealthKitLoaded) {
    return AppleHealthKit !== null;
  }

  if (Platform.OS !== 'ios') {
    healthKitLoadError = 'HealthKit is only available on iOS';
    isHealthKitLoaded = true;
    return false;
  }

  try {
    // Dynamic import to avoid crash if module not available
    const healthModule = require('react-native-health');
    AppleHealthKit = healthModule.default || healthModule;
    isHealthKitLoaded = true;
    console.log('[HealthKit] Native module loaded successfully');
    return true;
  } catch (error: any) {
    healthKitLoadError = error.message || 'Failed to load HealthKit module';
    isHealthKitLoaded = true;
    console.log('[HealthKit] Native module not available:', healthKitLoadError);
    return false;
  }
};

class HealthKitService {
  private isInitialized = false;
  private authorizationStatus: 'unknown' | 'authorized' | 'denied' = 'unknown';

  /**
   * Check if HealthKit is available on this device
   */
  async checkAvailability(): Promise<HealthKitStatus> {
    if (Platform.OS !== 'ios') {
      return {
        isAvailable: false,
        isAuthorized: false,
        error: 'HealthKit is only available on iOS devices',
      };
    }

    const loaded = await loadHealthKit();

    if (!loaded) {
      return {
        isAvailable: false,
        isAuthorized: false,
        needsRebuild: true,
        error: healthKitLoadError || 'HealthKit requires a development build. Run: eas build --profile development --platform ios',
      };
    }

    return new Promise((resolve) => {
      AppleHealthKit.isAvailable((error: any, available: boolean) => {
        if (error) {
          resolve({
            isAvailable: false,
            isAuthorized: false,
            error: error.message || 'Failed to check HealthKit availability',
          });
        } else {
          resolve({
            isAvailable: available,
            isAuthorized: this.authorizationStatus === 'authorized',
            error: available ? undefined : 'HealthKit is not available on this device',
          });
        }
      });
    });
  }

  /**
   * Request authorization to access HealthKit data
   */
  async requestAuthorization(): Promise<boolean> {
    if (Platform.OS !== 'ios') {
      console.warn('[HealthKit] Not available on this platform');
      return false;
    }

    const loaded = await loadHealthKit();
    if (!loaded) {
      console.warn('[HealthKit] Native module not loaded');
      return false;
    }

    return new Promise((resolve) => {
      AppleHealthKit.initHealthKit(HEALTHKIT_PERMISSIONS, (error: any) => {
        if (error) {
          console.error('[HealthKit] Authorization failed:', error);
          this.authorizationStatus = 'denied';
          resolve(false);
        } else {
          console.log('[HealthKit] Authorization granted');
          this.authorizationStatus = 'authorized';
          this.isInitialized = true;
          resolve(true);
        }
      });
    });
  }

  /**
   * Get workout samples from HealthKit
   */
  async getWorkouts(startDate: Date, endDate: Date): Promise<WorkoutData[]> {
    if (!this.isInitialized || !AppleHealthKit) {
      console.warn('[HealthKit] Not initialized');
      return [];
    }

    return new Promise((resolve) => {
      const options = {
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
        type: 'Workout',
      };

      AppleHealthKit.getSamples(options, (error: any, results: any[]) => {
        if (error) {
          console.error('[HealthKit] Failed to get workouts:', error);
          resolve([]);
          return;
        }

        const workouts: WorkoutData[] = (results || []).map((workout: any) => {
          const workoutType = workout.activityName || workout.activityType || 'Unknown';
          const isOutdoor = this.isOutdoorWorkout(workoutType);

          return {
            id: workout.id || `workout_${Date.now()}_${Math.random()}`,
            workoutType,
            startDate: new Date(workout.start || workout.startDate),
            endDate: new Date(workout.end || workout.endDate),
            duration: workout.duration || 0,
            totalEnergyBurned: workout.calories || workout.energyBurned,
            totalDistance: workout.distance,
            sourceName: workout.sourceName || 'Apple Health',
            isOutdoor,
            startLocation: workout.startLocation,
            endLocation: workout.endLocation,
          };
        });

        console.log(`[HealthKit] Retrieved ${workouts.length} workouts`);
        resolve(workouts);
      });
    });
  }

  /**
   * Check if a workout type is typically outdoor
   */
  private isOutdoorWorkout(workoutType: string): boolean {
    const type = workoutType.toLowerCase().replace(/\s+/g, '');
    return OUTDOOR_WORKOUT_TYPES.some(outdoor =>
      type.includes(outdoor.toLowerCase())
    );
  }

  /**
   * Get step count for a date range
   */
  async getStepCount(startDate: Date, endDate: Date): Promise<number> {
    if (!this.isInitialized || !AppleHealthKit) {
      return 0;
    }

    return new Promise((resolve) => {
      const options = {
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      };

      AppleHealthKit.getStepCount(options, (error: any, results: any) => {
        if (error) {
          console.error('[HealthKit] Failed to get step count:', error);
          resolve(0);
        } else {
          resolve(results?.value || 0);
        }
      });
    });
  }

  /**
   * Get active energy burned for a date range
   */
  async getActiveEnergy(startDate: Date, endDate: Date): Promise<number> {
    if (!this.isInitialized || !AppleHealthKit) {
      return 0;
    }

    return new Promise((resolve) => {
      const options = {
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      };

      AppleHealthKit.getActiveEnergyBurned(options, (error: any, results: any[]) => {
        if (error) {
          console.error('[HealthKit] Failed to get active energy:', error);
          resolve(0);
        } else {
          const total = (results || []).reduce((sum, r) => sum + (r.value || 0), 0);
          resolve(total);
        }
      });
    });
  }

  /**
   * Get heart rate samples
   */
  async getHeartRateSamples(startDate: Date, endDate: Date): Promise<{ date: Date; value: number }[]> {
    if (!this.isInitialized || !AppleHealthKit) {
      return [];
    }

    return new Promise((resolve) => {
      const options = {
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
        ascending: false,
        limit: 100,
      };

      AppleHealthKit.getHeartRateSamples(options, (error: any, results: any[]) => {
        if (error) {
          console.error('[HealthKit] Failed to get heart rate:', error);
          resolve([]);
        } else {
          const samples = (results || []).map(r => ({
            date: new Date(r.startDate),
            value: r.value,
          }));
          resolve(samples);
        }
      });
    });
  }

  /**
   * Get exercise time (Apple Exercise Minutes)
   */
  async getExerciseTime(startDate: Date, endDate: Date): Promise<number> {
    if (!this.isInitialized || !AppleHealthKit) {
      return 0;
    }

    return new Promise((resolve) => {
      const options = {
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      };

      AppleHealthKit.getAppleExerciseTime(options, (error: any, results: any[]) => {
        if (error) {
          console.error('[HealthKit] Failed to get exercise time:', error);
          resolve(0);
        } else {
          const total = (results || []).reduce((sum, r) => sum + (r.value || 0), 0);
          resolve(total);
        }
      });
    });
  }

  /**
   * Get daily activity summary
   */
  async getDailyActivity(date: Date): Promise<ActivityData | null> {
    if (!this.isInitialized || !AppleHealthKit) {
      return null;
    }

    const startOfDay = new Date(date);
    startOfDay.setHours(0, 0, 0, 0);

    const endOfDay = new Date(date);
    endOfDay.setHours(23, 59, 59, 999);

    try {
      const [stepCount, activeEnergyBurned, exerciseMinutes] = await Promise.all([
        this.getStepCount(startOfDay, endOfDay),
        this.getActiveEnergy(startOfDay, endOfDay),
        this.getExerciseTime(startOfDay, endOfDay),
      ]);

      return {
        date,
        stepCount,
        activeEnergyBurned,
        exerciseMinutes,
        standHours: 0, // Stand hours require different API
      };
    } catch (error) {
      console.error('[HealthKit] Failed to get daily activity:', error);
      return null;
    }
  }

  /**
   * Estimate outdoor exposure based on workouts
   */
  async estimateOutdoorExposure(startDate: Date, endDate: Date): Promise<{
    totalOutdoorMinutes: number;
    workouts: WorkoutData[];
    estimatedUVExposure: number;
    workoutLocations: Array<{ lat: number; lon: number; time: Date }>;
  }> {
    const workouts = await this.getWorkouts(startDate, endDate);
    const outdoorWorkouts = workouts.filter(w => w.isOutdoor);

    const totalOutdoorMinutes = outdoorWorkouts.reduce(
      (sum, w) => sum + (w.duration / 60), 0
    );

    // Collect locations from workouts for UV lookup
    const workoutLocations: Array<{ lat: number; lon: number; time: Date }> = [];
    for (const workout of outdoorWorkouts) {
      if (workout.startLocation) {
        workoutLocations.push({
          lat: workout.startLocation.latitude,
          lon: workout.startLocation.longitude,
          time: workout.startDate,
        });
      }
    }

    // Rough UV exposure estimate based on outdoor time and time of day
    let estimatedUVExposure = 0;
    for (const workout of outdoorWorkouts) {
      const hour = workout.startDate.getHours();
      let uvMultiplier = 1;

      // UV intensity varies by time of day
      if (hour >= 10 && hour <= 14) {
        uvMultiplier = 8; // Peak UV hours
      } else if (hour >= 8 && hour <= 16) {
        uvMultiplier = 5; // Moderate UV
      } else if (hour >= 6 && hour <= 18) {
        uvMultiplier = 2; // Low UV
      } else {
        uvMultiplier = 0; // Nighttime
      }

      // UV dose = UV index * minutes / 60
      estimatedUVExposure += uvMultiplier * (workout.duration / 60) / 60;
    }

    return {
      totalOutdoorMinutes: Math.round(totalOutdoorMinutes),
      workouts: outdoorWorkouts,
      estimatedUVExposure: Math.round(estimatedUVExposure * 10) / 10,
      workoutLocations,
    };
  }

  /**
   * Sync health data to the backend
   */
  async syncToBackend(
    apiUrl: string,
    token: string,
    days: number = 7
  ): Promise<SyncResult> {
    try {
      // Check if initialized
      if (!this.isInitialized) {
        const authorized = await this.requestAuthorization();
        if (!authorized) {
          return {
            success: false,
            message: 'HealthKit authorization required',
          };
        }
      }

      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - days);

      console.log(`[HealthKit] Syncing data from ${startDate.toISOString()} to ${endDate.toISOString()}`);

      // Get outdoor exposure data
      const outdoorData = await this.estimateOutdoorExposure(startDate, endDate);

      console.log(`[HealthKit] Found ${outdoorData.workouts.length} outdoor workouts, ${outdoorData.totalOutdoorMinutes} minutes total`);

      // Send to backend
      const response = await fetch(`${apiUrl}/wearables/healthkit/sync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          startDate: startDate.toISOString(),
          endDate: endDate.toISOString(),
          outdoorMinutes: outdoorData.totalOutdoorMinutes,
          estimatedUVExposure: outdoorData.estimatedUVExposure,
          workouts: outdoorData.workouts.map(w => ({
            id: w.id,
            type: w.workoutType,
            startDate: w.startDate.toISOString(),
            endDate: w.endDate.toISOString(),
            duration: w.duration,
            isOutdoor: w.isOutdoor,
            calories: w.totalEnergyBurned,
            distance: w.totalDistance,
            startLocation: w.startLocation,
            endLocation: w.endLocation,
          })),
          workoutLocations: outdoorData.workoutLocations.map(loc => ({
            latitude: loc.lat,
            longitude: loc.lon,
            timestamp: loc.time.toISOString(),
          })),
          uvReadings: [], // Will be populated by backend using OpenUV
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to sync data');
      }

      const result = await response.json();

      return {
        success: true,
        message: `Synced ${outdoorData.workouts.length} workouts (${outdoorData.totalOutdoorMinutes} outdoor minutes)`,
        workoutsCount: outdoorData.workouts.length,
        outdoorMinutes: outdoorData.totalOutdoorMinutes,
        uvReadingsCount: result.readings_created || 0,
      };
    } catch (error: any) {
      console.error('[HealthKit] Sync failed:', error);
      return {
        success: false,
        message: error.message || 'Failed to sync health data',
      };
    }
  }

  /**
   * Get recent activity summary for display
   */
  async getRecentActivitySummary(days: number = 7): Promise<{
    totalSteps: number;
    totalCalories: number;
    totalExerciseMinutes: number;
    totalOutdoorMinutes: number;
    workoutCount: number;
  } | null> {
    if (!this.isInitialized) {
      return null;
    }

    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    try {
      const [steps, calories, exercise, outdoor] = await Promise.all([
        this.getStepCount(startDate, endDate),
        this.getActiveEnergy(startDate, endDate),
        this.getExerciseTime(startDate, endDate),
        this.estimateOutdoorExposure(startDate, endDate),
      ]);

      return {
        totalSteps: Math.round(steps),
        totalCalories: Math.round(calories),
        totalExerciseMinutes: Math.round(exercise),
        totalOutdoorMinutes: outdoor.totalOutdoorMinutes,
        workoutCount: outdoor.workouts.length,
      };
    } catch (error) {
      console.error('[HealthKit] Failed to get activity summary:', error);
      return null;
    }
  }
}

export default new HealthKitService();
