import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useUserSettings, DisplayMode } from '../contexts/UserSettingsContext';

interface DisplayModeToggleProps {
  compact?: boolean;
  showDescription?: boolean;
}

export const DisplayModeToggle: React.FC<DisplayModeToggleProps> = ({
  compact = false,
  showDescription = true
}) => {
  const { settings, isLoading, setDisplayMode } = useUserSettings();

  const handleToggle = async () => {
    const newMode: DisplayMode = settings.displayMode === 'simple' ? 'professional' : 'simple';
    await setDisplayMode(newMode);
  };

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="small" color="#4A90A4" />
      </View>
    );
  }

  if (compact) {
    return (
      <TouchableOpacity
        style={styles.compactToggle}
        onPress={handleToggle}
        accessibilityLabel={`Switch to ${settings.displayMode === 'simple' ? 'professional' : 'simple'} mode`}
      >
        <Ionicons
          name={settings.displayMode === 'professional' ? 'medkit' : 'person'}
          size={20}
          color={settings.displayMode === 'professional' ? '#2E7D32' : '#4A90A4'}
        />
        <Text style={[
          styles.compactLabel,
          settings.displayMode === 'professional' && styles.professionalText
        ]}>
          {settings.displayMode === 'professional' ? 'PRO' : 'Simple'}
        </Text>
      </TouchableOpacity>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Ionicons
          name={settings.displayMode === 'professional' ? 'medkit' : 'eye'}
          size={24}
          color="#4A90A4"
        />
        <Text style={styles.title}>Display Mode</Text>
      </View>

      {showDescription && (
        <Text style={styles.description}>
          {settings.displayMode === 'simple'
            ? 'Simple mode shows easy-to-understand results designed for patients.'
            : 'Professional mode shows detailed clinical data, ABCDE analysis, and technical metrics.'}
        </Text>
      )}

      <View style={styles.toggleContainer}>
        <TouchableOpacity
          style={[
            styles.option,
            settings.displayMode === 'simple' && styles.selectedOption
          ]}
          onPress={() => setDisplayMode('simple')}
        >
          <Ionicons
            name="person"
            size={20}
            color={settings.displayMode === 'simple' ? '#fff' : '#666'}
          />
          <Text style={[
            styles.optionText,
            settings.displayMode === 'simple' && styles.selectedOptionText
          ]}>
            Simple
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.option,
            settings.displayMode === 'professional' && styles.selectedOptionPro
          ]}
          onPress={() => setDisplayMode('professional')}
        >
          <Ionicons
            name="medkit"
            size={20}
            color={settings.displayMode === 'professional' ? '#fff' : '#666'}
          />
          <Text style={[
            styles.optionText,
            settings.displayMode === 'professional' && styles.selectedOptionText
          ]}>
            Professional
          </Text>
        </TouchableOpacity>
      </View>

      {settings.displayMode === 'professional' && !settings.isVerifiedProfessional && (
        <View style={styles.verificationBanner}>
          <Ionicons name="information-circle" size={16} color="#856404" />
          <Text style={styles.verificationText}>
            Professional features available. Consider verifying your credentials for full access to clinic features.
          </Text>
        </View>
      )}

      {settings.isVerifiedProfessional && (
        <View style={styles.verifiedBadge}>
          <Ionicons name="checkmark-circle" size={16} color="#2E7D32" />
          <Text style={styles.verifiedText}>Verified Healthcare Professional</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginVertical: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  loadingContainer: {
    padding: 16,
    alignItems: 'center',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginLeft: 8,
  },
  description: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
    lineHeight: 20,
  },
  toggleContainer: {
    flexDirection: 'row',
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    padding: 4,
  },
  option: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 6,
  },
  selectedOption: {
    backgroundColor: '#4A90A4',
  },
  selectedOptionPro: {
    backgroundColor: '#2E7D32',
  },
  optionText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#666',
    marginLeft: 6,
  },
  selectedOptionText: {
    color: '#fff',
  },
  compactToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  compactLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#4A90A4',
    marginLeft: 4,
  },
  professionalText: {
    color: '#2E7D32',
  },
  verificationBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff3cd',
    padding: 10,
    borderRadius: 6,
    marginTop: 12,
  },
  verificationText: {
    flex: 1,
    fontSize: 12,
    color: '#856404',
    marginLeft: 8,
  },
  verifiedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#e8f5e9',
    padding: 10,
    borderRadius: 6,
    marginTop: 12,
  },
  verifiedText: {
    fontSize: 12,
    color: '#2E7D32',
    fontWeight: '500',
    marginLeft: 6,
  },
});

export default DisplayModeToggle;
