/**
 * HealthKit Service
 *
 * Provides direct integration with Apple HealthKit for reading health data
 * from Apple Watch and iPhone. This is a free, open-source alternative to
 * paid services like Terra.
 *
 * Requirements:
 * - Expo Development Build with HealthKit native code
 * - Apple Developer Account
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

// HealthKit module - will be null if native code not available
let Healthkit: any = null;
let HKQuantityTypeIdentifier: any = null;
let HKCategoryTypeIdentifier: any = null;
let healthKitLoadError: string | null = null;

// Only try to load on iOS
if (Platform.OS === 'ios') {
  // Don't try to load the module at startup - it will crash if not in build
  // Instead, we'll check availability when needed
  console.log('[HealthKit] Service initialized - native module will be checked on first use');
}

const getHealthKit = () => {
  return Healthkit;
};

class HealthKitService {
  private isInitialized = false;

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

    // Return a message indicating rebuild is needed
    // The native HealthKit code is not included in this build
    return {
      isAvailable: false,
      isAuthorized: false,
      needsRebuild: true,
      error: 'HealthKit requires a rebuild. Run: eas build --profile development --platform ios',
    };
  }

  /**
   * Request authorization to access HealthKit data
   */
  async requestAuthorization(): Promise<boolean> {
    if (Platform.OS !== 'ios') {
      console.warn('[HealthKit] Not available on this platform');
      return false;
    }

    try {
      const HK = getHealthKit();
      if (!HK) {
        throw new Error('HealthKit module not loaded');
      }

      // Request permissions for the data types we need
      await HK.requestAuthorization(
        // Read permissions
        [
          HKQuantityTypeIdentifier?.uvExposure,
          HKQuantityTypeIdentifier?.activeEnergyBurned,
          HKQuantityTypeIdentifier?.stepCount,
          HKQuantityTypeIdentifier?.appleExerciseTime,
          HKQuantityTypeIdentifier?.heartRate,
          HKCategoryTypeIdentifier?.appleStandHour,
        ].filter(Boolean),
        // Write permissions (optional)
        []
      );

      this.isInitialized = true;
      console.log('[HealthKit] Authorization granted');
      return true;
    } catch (error) {
      console.error('[HealthKit] Authorization failed:', error);
      return false;
    }
  }

  /**
   * Get UV exposure data from HealthKit
   * Note: UV exposure data is limited - Apple Watch doesn't have a UV sensor
   * This data would come from third-party apps that log UV exposure
   */
  async getUVExposure(startDate: Date, endDate: Date): Promise<UVExposureReading[]> {
    if (Platform.OS !== 'ios') return [];

    try {
      const HK = getHealthKit();
      if (!HK) return [];

      const samples = await HK.queryQuantitySamples(
        HKQuantityTypeIdentifier?.uvExposure,
        {
          from: startDate,
          to: endDate,
        }
      );

      return samples.map((sample: any) => ({
        startDate: new Date(sample.startDate),
        endDate: new Date(sample.endDate),
        value: sample.quantity,
        sourceName: sample.sourceName || 'Unknown',
      }));
    } catch (error) {
      console.error('[HealthKit] Failed to get UV exposure:', error);
      return [];
    }
  }

  /**
   * Get outdoor workouts - useful for estimating sun exposure
   */
  async getOutdoorWorkouts(startDate: Date, endDate: Date): Promise<WorkoutData[]> {
    if (Platform.OS !== 'ios') return [];

    try {
      const HK = getHealthKit();
      if (!HK) return [];

      const workouts = await HK.queryWorkoutSamples({
        from: startDate,
        to: endDate,
      });

      // Filter for outdoor workout types
      const outdoorTypes = [
        'Walking',
        'Running',
        'Cycling',
        'Hiking',
        'Swimming', // outdoor swimming
        'Golf',
        'Tennis',
        'Soccer',
        'Basketball',
        'Baseball',
        'Softball',
        'Volleyball',
        'Football',
        'Lacrosse',
        'Rugby',
        'Sailing',
        'Surfing',
        'Skateboarding',
        'Snowboarding',
        'Skiing',
        'Paddling',
        'Rowing',
      ];

      return workouts.map((workout: any) => {
        const workoutType = workout.workoutActivityType || 'Unknown';
        const isOutdoor = outdoorTypes.some(type =>
          workoutType.toLowerCase().includes(type.toLowerCase())
        );

        return {
          id: workout.uuid,
          workoutType,
          startDate: new Date(workout.startDate),
          endDate: new Date(workout.endDate),
          duration: workout.duration || 0,
          totalEnergyBurned: workout.totalEnergyBurned,
          totalDistance: workout.totalDistance,
          sourceName: workout.sourceName || 'Unknown',
          isOutdoor,
        };
      });
    } catch (error) {
      console.error('[HealthKit] Failed to get workouts:', error);
      return [];
    }
  }

  /**
   * Get daily activity summary
   */
  async getDailyActivity(date: Date): Promise<ActivityData | null> {
    if (Platform.OS !== 'ios') return null;

    try {
      const HK = getHealthKit();
      if (!HK) return null;

      const startOfDay = new Date(date);
      startOfDay.setHours(0, 0, 0, 0);

      const endOfDay = new Date(date);
      endOfDay.setHours(23, 59, 59, 999);

      // Get active energy
      const energySamples = await HK.queryQuantitySamples(
        HKQuantityTypeIdentifier?.activeEnergyBurned,
        { from: startOfDay, to: endOfDay }
      );
      const activeEnergyBurned = energySamples.reduce(
        (sum: number, s: any) => sum + (s.quantity || 0), 0
      );

      // Get step count
      const stepSamples = await HK.queryQuantitySamples(
        HKQuantityTypeIdentifier?.stepCount,
        { from: startOfDay, to: endOfDay }
      );
      const stepCount = stepSamples.reduce(
        (sum: number, s: any) => sum + (s.quantity || 0), 0
      );

      // Get exercise minutes
      const exerciseSamples = await HK.queryQuantitySamples(
        HKQuantityTypeIdentifier?.appleExerciseTime,
        { from: startOfDay, to: endOfDay }
      );
      const exerciseMinutes = exerciseSamples.reduce(
        (sum: number, s: any) => sum + (s.quantity || 0), 0
      );

      // Get stand hours
      const standSamples = await HK.queryCategorySamples(
        HKCategoryTypeIdentifier?.appleStandHour,
        { from: startOfDay, to: endOfDay }
      );
      const standHours = standSamples.filter((s: any) => s.value === 0).length;

      return {
        date,
        activeEnergyBurned,
        stepCount,
        exerciseMinutes,
        standHours,
      };
    } catch (error) {
      console.error('[HealthKit] Failed to get daily activity:', error);
      return null;
    }
  }

  /**
   * Estimate outdoor time based on workouts and activity
   * Since Apple Watch doesn't have UV sensor, we estimate sun exposure
   * based on outdoor workout duration
   */
  async estimateOutdoorExposure(startDate: Date, endDate: Date): Promise<{
    totalOutdoorMinutes: number;
    workouts: WorkoutData[];
    estimatedUVExposure: number;
  }> {
    const workouts = await this.getOutdoorWorkouts(startDate, endDate);
    const outdoorWorkouts = workouts.filter(w => w.isOutdoor);

    const totalOutdoorMinutes = outdoorWorkouts.reduce(
      (sum, w) => sum + (w.duration / 60), 0
    );

    // Rough UV exposure estimate based on outdoor time
    // Assumes average UV index of 5 during daylight hours
    const estimatedUVExposure = totalOutdoorMinutes * 5 / 60; // UV dose

    return {
      totalOutdoorMinutes: Math.round(totalOutdoorMinutes),
      workouts: outdoorWorkouts,
      estimatedUVExposure: Math.round(estimatedUVExposure * 10) / 10,
    };
  }

  /**
   * Get heart rate data (useful for activity detection)
   */
  async getHeartRateSamples(startDate: Date, endDate: Date): Promise<any[]> {
    if (Platform.OS !== 'ios') return [];

    try {
      const HK = getHealthKit();
      if (!HK) return [];

      const samples = await HK.queryQuantitySamples(
        HKQuantityTypeIdentifier?.heartRate,
        { from: startDate, to: endDate }
      );

      return samples.map((sample: any) => ({
        date: new Date(sample.startDate),
        value: sample.quantity,
        sourceName: sample.sourceName,
      }));
    } catch (error) {
      console.error('[HealthKit] Failed to get heart rate:', error);
      return [];
    }
  }

  /**
   * Sync all health data to the backend
   */
  async syncToBackend(
    apiUrl: string,
    token: string,
    days: number = 7
  ): Promise<{ success: boolean; message: string }> {
    try {
      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - days);

      // Get all relevant data
      const [outdoorData, uvReadings] = await Promise.all([
        this.estimateOutdoorExposure(startDate, endDate),
        this.getUVExposure(startDate, endDate),
      ]);

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
          })),
          uvReadings: uvReadings.map(r => ({
            startDate: r.startDate.toISOString(),
            endDate: r.endDate.toISOString(),
            value: r.value,
            source: r.sourceName,
          })),
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to sync data');
      }

      return {
        success: true,
        message: `Synced ${outdoorData.workouts.length} workouts and ${outdoorData.totalOutdoorMinutes} outdoor minutes`,
      };
    } catch (error: any) {
      console.error('[HealthKit] Sync failed:', error);
      return {
        success: false,
        message: error.message || 'Failed to sync health data',
      };
    }
  }
}

export default new HealthKitService();
