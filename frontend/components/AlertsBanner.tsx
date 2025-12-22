import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Pressable, Alert } from 'react-native';
import { router } from 'expo-router';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';

interface AlertData {
  id: number;
  alert_type: string;
  severity: string;
  priority: number;
  title: string;
  message: string;
  action_required: string;
  action_url: string;
  is_read: boolean;
  created_at: string;
}

export default function AlertsBanner() {
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchAlerts();
  }, []);

  const fetchAlerts = async () => {
    try {
      setIsLoading(true);
      const token = AuthService.getToken();
      if (!token) {
        setIsLoading(false);
        return;
      }

      // Use AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

      const response = await fetch(`${API_BASE_URL}/alerts?unread_only=true`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        // Only show top 2 highest priority alerts
        const alertsArray = data.alerts || data || [];
        setAlerts(alertsArray.slice(0, 2));
      }
    } catch (error: any) {
      // Silently ignore timeout/abort errors - alerts are non-critical
      if (error.name !== 'AbortError') {
        console.error('Error fetching alerts:', error);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const markAsRead = async (alertId: number) => {
    try {
      const token = AuthService.getToken();
      await fetch(`${API_BASE_URL}/alerts/${alertId}/read`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
    } catch (error) {
      console.error('Error marking alert as read:', error);
    }
  };

  const dismissAlert = async (alertId: number) => {
    try {
      const token = AuthService.getToken();
      await fetch(`${API_BASE_URL}/alerts/${alertId}/dismiss`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      // Remove from local state
      setAlerts(alerts.filter(a => a.id !== alertId));
    } catch (error) {
      console.error('Error dismissing alert:', error);
    }
  };

  const handleAlertPress = (alert: AlertData) => {
    // Mark as read
    markAsRead(alert.id);

    // Navigate to action URL if specified
    if (alert.action_url) {
      router.push(alert.action_url as any);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '#dc2626';
      case 'urgent':
        return '#f59e0b';
      case 'warning':
        return '#eab308';
      default:
        return '#3b82f6';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
      case 'urgent':
        return '⚠️';
      case 'warning':
        return '⚡';
      default:
        return 'ℹ️';
    }
  };

  if (isLoading || alerts.length === 0) {
    return null;
  }

  return (
    <View style={styles.container}>
      {alerts.map((alert) => (
        <View
          key={alert.id}
          style={[styles.alertCard, { borderLeftColor: getSeverityColor(alert.severity) }]}
        >
          <Pressable
            style={styles.alertContent}
            onPress={() => handleAlertPress(alert)}
          >
            <View style={styles.alertHeader}>
              <Text style={styles.alertIcon}>{getSeverityIcon(alert.severity)}</Text>
              <Text style={[styles.alertTitle, { color: getSeverityColor(alert.severity) }]}>
                {alert.title}
              </Text>
            </View>
            <Text style={styles.alertMessage} numberOfLines={2}>
              {alert.message}
            </Text>
            {alert.action_required && (
              <Text style={styles.actionRequired}>
                Action: {alert.action_required}
              </Text>
            )}
          </Pressable>

          <Pressable
            style={styles.dismissButton}
            onPress={() => dismissAlert(alert.id)}
          >
            <Text style={styles.dismissButtonText}>✕</Text>
          </Pressable>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    gap: 10,
  },
  alertCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    borderLeftWidth: 4,
    padding: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  alertContent: {
    flex: 1,
  },
  alertHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
    gap: 8,
  },
  alertIcon: {
    fontSize: 16,
  },
  alertTitle: {
    fontSize: 14,
    fontWeight: '600',
    flex: 1,
  },
  alertMessage: {
    fontSize: 13,
    color: '#4b5563',
    marginBottom: 6,
    lineHeight: 18,
  },
  actionRequired: {
    fontSize: 12,
    color: '#6b7280',
    fontStyle: 'italic',
  },
  dismissButton: {
    padding: 4,
    marginLeft: 8,
  },
  dismissButtonText: {
    fontSize: 18,
    color: '#9ca3af',
    fontWeight: '600',
  },
});
