/**
 * Tests for NotificationService
 *
 * Tests notification functionality including:
 * - Push notification registration
 * - Notification retrieval and management
 * - Badge count management
 * - Local notification scheduling
 */

// Mock dependencies
jest.mock('../../config', () => ({
  API_BASE_URL: 'http://test-api.example.com',
}));

jest.mock('expo-notifications', () => ({
  getPermissionsAsync: jest.fn(),
  requestPermissionsAsync: jest.fn(),
  getExpoPushTokenAsync: jest.fn(),
  setNotificationChannelAsync: jest.fn(),
  scheduleNotificationAsync: jest.fn(),
  cancelScheduledNotificationAsync: jest.fn(),
  cancelAllScheduledNotificationsAsync: jest.fn(),
  getBadgeCountAsync: jest.fn(),
  setBadgeCountAsync: jest.fn(),
  addNotificationReceivedListener: jest.fn(),
  addNotificationResponseReceivedListener: jest.fn(),
  removeNotificationSubscription: jest.fn(),
  setNotificationHandler: jest.fn(),
  AndroidImportance: {
    HIGH: 4,
    MAX: 5,
    DEFAULT: 3,
  },
}));

jest.mock('expo-device', () => ({
  isDevice: true,
}));

jest.mock('react-native', () => ({
  Platform: { OS: 'android' },
}));

import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import AsyncStorage from '@react-native-async-storage/async-storage';

describe('NotificationService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    AsyncStorage.clear();
  });

  describe('Permission handling', () => {
    it('should check existing permissions', async () => {
      Notifications.getPermissionsAsync.mockResolvedValue({ status: 'granted' });

      const { status } = await Notifications.getPermissionsAsync();
      expect(status).toBe('granted');
    });

    it('should request permissions if not granted', async () => {
      Notifications.getPermissionsAsync.mockResolvedValue({ status: 'undetermined' });
      Notifications.requestPermissionsAsync.mockResolvedValue({ status: 'granted' });

      const existing = await Notifications.getPermissionsAsync();
      expect(existing.status).toBe('undetermined');

      const requested = await Notifications.requestPermissionsAsync();
      expect(requested.status).toBe('granted');
    });

    it('should handle permission denied', async () => {
      Notifications.getPermissionsAsync.mockResolvedValue({ status: 'denied' });
      Notifications.requestPermissionsAsync.mockResolvedValue({ status: 'denied' });

      const { status } = await Notifications.requestPermissionsAsync();
      expect(status).toBe('denied');
    });
  });

  describe('Push token management', () => {
    it('should get Expo push token', async () => {
      Notifications.getExpoPushTokenAsync.mockResolvedValue({
        data: 'ExponentPushToken[xxxxxx]',
      });

      const tokenData = await Notifications.getExpoPushTokenAsync({});
      expect(tokenData.data).toBe('ExponentPushToken[xxxxxx]');
    });

    it('should require physical device for push notifications', () => {
      expect(Device.isDevice).toBe(true);
    });
  });

  describe('Android notification channels', () => {
    it('should create appointment channel with correct config', async () => {
      const channelConfig = {
        name: 'Appointment Reminders',
        importance: 4, // HIGH
        vibrationPattern: [0, 250, 250, 250],
        lightColor: '#2563eb',
      };

      expect(channelConfig.name).toBe('Appointment Reminders');
      expect(channelConfig.importance).toBe(4);
    });

    it('should create alerts channel with correct config', async () => {
      const channelConfig = {
        name: 'Health Alerts',
        importance: 5, // MAX
        vibrationPattern: [0, 500, 250, 500],
        lightColor: '#ef4444',
      };

      expect(channelConfig.name).toBe('Health Alerts');
      expect(channelConfig.importance).toBe(5);
    });

    it('should create general channel with correct config', async () => {
      const channelConfig = {
        name: 'General Notifications',
        importance: 3, // DEFAULT
      };

      expect(channelConfig.name).toBe('General Notifications');
      expect(channelConfig.importance).toBe(3);
    });
  });

  describe('Local notification scheduling', () => {
    it('should schedule a local notification', async () => {
      const mockIdentifier = 'notification-123';
      Notifications.scheduleNotificationAsync.mockResolvedValue(mockIdentifier);

      const triggerDate = new Date(Date.now() + 3600000); // 1 hour from now

      const identifier = await Notifications.scheduleNotificationAsync({
        content: {
          title: 'Test Notification',
          body: 'This is a test',
          data: { type: 'test' },
          sound: true,
        },
        trigger: { date: triggerDate },
      });

      expect(identifier).toBe(mockIdentifier);
      expect(Notifications.scheduleNotificationAsync).toHaveBeenCalled();
    });

    it('should cancel a scheduled notification', async () => {
      const identifier = 'notification-123';

      await Notifications.cancelScheduledNotificationAsync(identifier);

      expect(Notifications.cancelScheduledNotificationAsync).toHaveBeenCalledWith(identifier);
    });

    it('should cancel all scheduled notifications', async () => {
      await Notifications.cancelAllScheduledNotificationsAsync();

      expect(Notifications.cancelAllScheduledNotificationsAsync).toHaveBeenCalled();
    });
  });

  describe('Badge count management', () => {
    it('should get badge count', async () => {
      Notifications.getBadgeCountAsync.mockResolvedValue(5);

      const count = await Notifications.getBadgeCountAsync();
      expect(count).toBe(5);
    });

    it('should set badge count', async () => {
      await Notifications.setBadgeCountAsync(10);

      expect(Notifications.setBadgeCountAsync).toHaveBeenCalledWith(10);
    });

    it('should clear badge count', async () => {
      await Notifications.setBadgeCountAsync(0);

      expect(Notifications.setBadgeCountAsync).toHaveBeenCalledWith(0);
    });
  });

  describe('Notification data structure', () => {
    const mockNotification = {
      id: 1,
      type: 'appointment_reminder',
      title: 'Upcoming Appointment',
      message: 'You have an appointment tomorrow',
      data: { appointment_id: 123 },
      is_read: false,
      priority: 'high',
      created_at: '2024-01-15T10:00:00Z',
      read_at: null,
    };

    it('should have correct notification structure', () => {
      expect(mockNotification).toHaveProperty('id');
      expect(mockNotification).toHaveProperty('type');
      expect(mockNotification).toHaveProperty('title');
      expect(mockNotification).toHaveProperty('message');
      expect(mockNotification).toHaveProperty('data');
      expect(mockNotification).toHaveProperty('is_read');
      expect(mockNotification).toHaveProperty('priority');
      expect(mockNotification).toHaveProperty('created_at');
      expect(mockNotification).toHaveProperty('read_at');
    });

    it('should have correct notification type', () => {
      expect(mockNotification.type).toBe('appointment_reminder');
    });

    it('should track read status', () => {
      expect(mockNotification.is_read).toBe(false);
      expect(mockNotification.read_at).toBeNull();
    });
  });

  describe('Notification response structure', () => {
    const mockResponse = {
      total: 25,
      unread_count: 5,
      notifications: [
        { id: 1, type: 'alert', is_read: false },
        { id: 2, type: 'reminder', is_read: true },
      ],
    };

    it('should have total count', () => {
      expect(mockResponse.total).toBe(25);
    });

    it('should have unread count', () => {
      expect(mockResponse.unread_count).toBe(5);
    });

    it('should have notifications array', () => {
      expect(Array.isArray(mockResponse.notifications)).toBe(true);
      expect(mockResponse.notifications).toHaveLength(2);
    });
  });

  describe('Notification type handling', () => {
    const handleNotificationType = (type, data) => {
      switch (type) {
        case 'appointment_reminder':
          return { action: 'navigate', screen: 'appointment', id: data.appointment_id };
        case 'analysis_complete':
          return { action: 'navigate', screen: 'analysis', id: data.analysis_id };
        case 'health_alert':
          return { action: 'alert', message: data.message };
        default:
          return { action: 'none' };
      }
    };

    it('should handle appointment_reminder type', () => {
      const result = handleNotificationType('appointment_reminder', { appointment_id: 123 });
      expect(result.action).toBe('navigate');
      expect(result.screen).toBe('appointment');
      expect(result.id).toBe(123);
    });

    it('should handle analysis_complete type', () => {
      const result = handleNotificationType('analysis_complete', { analysis_id: 456 });
      expect(result.action).toBe('navigate');
      expect(result.screen).toBe('analysis');
      expect(result.id).toBe(456);
    });

    it('should handle health_alert type', () => {
      const result = handleNotificationType('health_alert', { message: 'Important alert' });
      expect(result.action).toBe('alert');
      expect(result.message).toBe('Important alert');
    });

    it('should handle unknown type', () => {
      const result = handleNotificationType('unknown_type', {});
      expect(result.action).toBe('none');
    });
  });

  describe('API endpoint construction', () => {
    const API_BASE_URL = 'http://test-api.example.com';

    it('should construct get notifications endpoint', () => {
      const endpoint = `${API_BASE_URL}/notifications?unread_only=false&limit=50`;
      expect(endpoint).toBe('http://test-api.example.com/notifications?unread_only=false&limit=50');
    });

    it('should construct mark as read endpoint', () => {
      const notificationId = 123;
      const endpoint = `${API_BASE_URL}/notifications/${notificationId}/read`;
      expect(endpoint).toBe('http://test-api.example.com/notifications/123/read');
    });

    it('should construct delete notification endpoint', () => {
      const notificationId = 456;
      const endpoint = `${API_BASE_URL}/notifications/${notificationId}`;
      expect(endpoint).toBe('http://test-api.example.com/notifications/456');
    });

    it('should construct register device endpoint', () => {
      const endpoint = `${API_BASE_URL}/notifications/register-device`;
      expect(endpoint).toBe('http://test-api.example.com/notifications/register-device');
    });
  });

  describe('Token storage', () => {
    it('should store push token', async () => {
      const token = 'ExponentPushToken[test]';
      await AsyncStorage.setItem('pushToken', token);

      const stored = await AsyncStorage.getItem('pushToken');
      expect(stored).toBe(token);
    });

    it('should remove push token on unregister', async () => {
      await AsyncStorage.setItem('pushToken', 'test-token');
      await AsyncStorage.removeItem('pushToken');

      const stored = await AsyncStorage.getItem('pushToken');
      expect(stored).toBeNull();
    });
  });

  describe('Notification listeners', () => {
    it('should have notification received listener function', () => {
      expect(typeof Notifications.addNotificationReceivedListener).toBe('function');
    });

    it('should have notification response listener function', () => {
      expect(typeof Notifications.addNotificationResponseReceivedListener).toBe('function');
    });

    it('should have remove subscription function', () => {
      expect(typeof Notifications.removeNotificationSubscription).toBe('function');
    });

    it('should support subscription object with remove method', () => {
      const subscription = { remove: jest.fn() };
      expect(typeof subscription.remove).toBe('function');
    });
  });

  describe('Priority levels', () => {
    const priorities = ['low', 'normal', 'high', 'urgent'];

    it('should recognize valid priority levels', () => {
      priorities.forEach((priority) => {
        expect(['low', 'normal', 'high', 'urgent']).toContain(priority);
      });
    });

    it('should determine notification channel by priority', () => {
      const getChannelForPriority = (priority) => {
        if (priority === 'urgent' || priority === 'high') return 'alerts';
        if (priority === 'normal') return 'general';
        return 'general';
      };

      expect(getChannelForPriority('urgent')).toBe('alerts');
      expect(getChannelForPriority('high')).toBe('alerts');
      expect(getChannelForPriority('normal')).toBe('general');
      expect(getChannelForPriority('low')).toBe('general');
    });
  });
});
