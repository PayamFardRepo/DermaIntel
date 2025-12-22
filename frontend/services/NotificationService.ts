/**
 * Notification Service
 *
 * Handles:
 * - Push notification registration with Expo
 * - In-app notification management
 * - Notification preferences
 */

import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import { Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

// Configure notification handling
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

export interface Notification {
  id: number;
  type: string;
  title: string;
  message: string;
  data: any;
  is_read: boolean;
  priority: string;
  created_at: string;
  read_at: string | null;
}

export interface NotificationResponse {
  total: number;
  unread_count: number;
  notifications: Notification[];
}

class NotificationService {
  private pushToken: string | null = null;
  private notificationListener: any = null;
  private responseListener: any = null;

  /**
   * Initialize notification service
   * Call this when the app starts
   */
  async initialize(): Promise<void> {
    // Set up notification listeners
    this.setupNotificationListeners();

    // Register for push notifications
    await this.registerForPushNotifications();
  }

  /**
   * Register device for push notifications
   */
  async registerForPushNotifications(): Promise<string | null> {
    if (!Device.isDevice) {
      console.log('Push notifications require a physical device');
      return null;
    }

    try {
      // Check existing permissions
      const { status: existingStatus } = await Notifications.getPermissionsAsync();
      let finalStatus = existingStatus;

      // Request permissions if not granted
      if (existingStatus !== 'granted') {
        const { status } = await Notifications.requestPermissionsAsync();
        finalStatus = status;
      }

      if (finalStatus !== 'granted') {
        console.log('Push notification permission denied');
        return null;
      }

      // Get Expo push token
      const tokenData = await Notifications.getExpoPushTokenAsync({
        projectId: process.env.EXPO_PROJECT_ID || undefined,
      });
      this.pushToken = tokenData.data;

      console.log('Push token:', this.pushToken);

      // Register token with backend
      await this.registerTokenWithBackend(this.pushToken);

      // Configure Android channel
      if (Platform.OS === 'android') {
        await Notifications.setNotificationChannelAsync('appointments', {
          name: 'Appointment Reminders',
          importance: Notifications.AndroidImportance.HIGH,
          vibrationPattern: [0, 250, 250, 250],
          lightColor: '#2563eb',
        });

        await Notifications.setNotificationChannelAsync('alerts', {
          name: 'Health Alerts',
          importance: Notifications.AndroidImportance.MAX,
          vibrationPattern: [0, 500, 250, 500],
          lightColor: '#ef4444',
        });

        await Notifications.setNotificationChannelAsync('general', {
          name: 'General Notifications',
          importance: Notifications.AndroidImportance.DEFAULT,
        });
      }

      return this.pushToken;
    } catch (error) {
      console.error('Error registering for push notifications:', error);
      return null;
    }
  }

  /**
   * Register push token with backend
   */
  private async registerTokenWithBackend(token: string): Promise<void> {
    try {
      const authToken = await AsyncStorage.getItem('accessToken');
      if (!authToken) {
        console.log('No auth token, skipping push registration');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/notifications/register-device`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          token: token,
          platform: 'expo',
        }),
      });

      if (response.ok) {
        console.log('Device registered for push notifications');
        await AsyncStorage.setItem('pushToken', token);
      } else {
        console.error('Failed to register device:', await response.text());
      }
    } catch (error) {
      console.error('Error registering token with backend:', error);
    }
  }

  /**
   * Unregister device from push notifications
   */
  async unregisterDevice(): Promise<void> {
    try {
      const authToken = await AsyncStorage.getItem('accessToken');
      if (!authToken) return;

      await fetch(`${API_BASE_URL}/notifications/unregister-device`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json',
        },
      });

      await AsyncStorage.removeItem('pushToken');
      this.pushToken = null;
      console.log('Device unregistered from push notifications');
    } catch (error) {
      console.error('Error unregistering device:', error);
    }
  }

  /**
   * Set up notification listeners
   */
  private setupNotificationListeners(): void {
    // Listener for incoming notifications while app is foregrounded
    this.notificationListener = Notifications.addNotificationReceivedListener(
      (notification) => {
        console.log('Notification received:', notification);
        // You can emit an event here to update UI
      }
    );

    // Listener for notification responses (when user taps notification)
    this.responseListener = Notifications.addNotificationResponseReceivedListener(
      (response) => {
        console.log('Notification response:', response);
        this.handleNotificationResponse(response);
      }
    );
  }

  /**
   * Handle notification tap
   */
  private handleNotificationResponse(response: Notifications.NotificationResponse): void {
    const data = response.notification.request.content.data;

    if (data?.type === 'appointment_reminder') {
      // Navigate to appointment details
      const appointmentId = data.appointment_id;
      console.log('Navigate to appointment:', appointmentId);
      // You would use router.push here in a component context
    } else if (data?.type === 'analysis_complete') {
      // Navigate to analysis results
      const analysisId = data.analysis_id;
      console.log('Navigate to analysis:', analysisId);
    }
  }

  /**
   * Get notifications from backend
   */
  async getNotifications(
    unreadOnly: boolean = false,
    limit: number = 50
  ): Promise<NotificationResponse | null> {
    try {
      const authToken = await AsyncStorage.getItem('accessToken');
      if (!authToken) return null;

      const response = await fetch(
        `${API_BASE_URL}/notifications?unread_only=${unreadOnly}&limit=${limit}`,
        {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (response.ok) {
        return await response.json();
      }
      return null;
    } catch (error) {
      console.error('Error fetching notifications:', error);
      return null;
    }
  }

  /**
   * Mark notification as read
   */
  async markAsRead(notificationId: number): Promise<boolean> {
    try {
      const authToken = await AsyncStorage.getItem('accessToken');
      if (!authToken) return false;

      const response = await fetch(
        `${API_BASE_URL}/notifications/${notificationId}/read`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        }
      );

      return response.ok;
    } catch (error) {
      console.error('Error marking notification as read:', error);
      return false;
    }
  }

  /**
   * Mark all notifications as read
   */
  async markAllAsRead(): Promise<boolean> {
    try {
      const authToken = await AsyncStorage.getItem('accessToken');
      if (!authToken) return false;

      const response = await fetch(`${API_BASE_URL}/notifications/read-all`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json',
        },
      });

      return response.ok;
    } catch (error) {
      console.error('Error marking all notifications as read:', error);
      return false;
    }
  }

  /**
   * Delete notification
   */
  async deleteNotification(notificationId: number): Promise<boolean> {
    try {
      const authToken = await AsyncStorage.getItem('accessToken');
      if (!authToken) return false;

      const response = await fetch(
        `${API_BASE_URL}/notifications/${notificationId}`,
        {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        }
      );

      return response.ok;
    } catch (error) {
      console.error('Error deleting notification:', error);
      return false;
    }
  }

  /**
   * Schedule a local notification
   */
  async scheduleLocalNotification(
    title: string,
    body: string,
    triggerDate: Date,
    data?: any
  ): Promise<string> {
    const trigger = {
      date: triggerDate,
    };

    const identifier = await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data,
        sound: true,
      },
      trigger,
    });

    return identifier;
  }

  /**
   * Cancel scheduled notification
   */
  async cancelScheduledNotification(identifier: string): Promise<void> {
    await Notifications.cancelScheduledNotificationAsync(identifier);
  }

  /**
   * Cancel all scheduled notifications
   */
  async cancelAllScheduledNotifications(): Promise<void> {
    await Notifications.cancelAllScheduledNotificationsAsync();
  }

  /**
   * Get badge count
   */
  async getBadgeCount(): Promise<number> {
    return await Notifications.getBadgeCountAsync();
  }

  /**
   * Set badge count
   */
  async setBadgeCount(count: number): Promise<void> {
    await Notifications.setBadgeCountAsync(count);
  }

  /**
   * Cleanup listeners
   */
  cleanup(): void {
    if (this.notificationListener) {
      Notifications.removeNotificationSubscription(this.notificationListener);
    }
    if (this.responseListener) {
      Notifications.removeNotificationSubscription(this.responseListener);
    }
  }

  /**
   * Get stored push token
   */
  getPushToken(): string | null {
    return this.pushToken;
  }
}

// Export singleton instance
export const notificationService = new NotificationService();
export default notificationService;
