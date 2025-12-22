import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  Pressable,
  Alert,
  Clipboard
} from 'react-native';

interface TeledermatologyShareProps {
  analysisId: number;
  onShareComplete: (shareData: ShareResult) => void;
  existingShare?: ExistingShareData;
}

interface ShareResult {
  share_token: string;
  share_url: string;
  dermatologist_name: string;
  dermatologist_email: string;
}

interface ExistingShareData {
  dermatologist_name?: string;
  dermatologist_email?: string;
  share_date?: string;
  share_message?: string;
  dermatologist_reviewed?: boolean;
  dermatologist_notes?: string;
  dermatologist_recommendation?: string;
}

export default function TeledermatologyShare({ analysisId, onShareComplete, existingShare }: TeledermatologyShareProps) {
  const [dermatologistName, setDermatologistName] = useState('');
  const [dermatologistEmail, setDermatologistEmail] = useState('');
  const [message, setMessage] = useState('');

  const handleShare = () => {
    if (!dermatologistName.trim()) {
      Alert.alert('Required', 'Please enter the dermatologist\'s name.');
      return;
    }

    if (!dermatologistEmail.trim()) {
      Alert.alert('Required', 'Please enter the dermatologist\'s email.');
      return;
    }

    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(dermatologistEmail)) {
      Alert.alert('Invalid Email', 'Please enter a valid email address.');
      return;
    }

    // Call the parent handler
    const shareData = {
      dermatologist_name: dermatologistName,
      dermatologist_email: dermatologistEmail,
      share_message: message || undefined
    };

    // Pass to parent for API call
    onShareComplete(shareData as any);
  };

  if (existingShare && existingShare.dermatologist_name) {
    return (
      <View style={styles.contentContainer}>
        <Text style={styles.title}>👨‍⚕️ Teledermatology Consultation</Text>

        <View style={styles.sharedSection}>
          <View style={styles.sharedHeader}>
            <Text style={styles.sharedTitle}>✅ Shared with Dermatologist</Text>
            {existingShare.dermatologist_reviewed && (
              <View style={styles.reviewedBadge}>
                <Text style={styles.reviewedBadgeText}>Reviewed</Text>
              </View>
            )}
          </View>

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Dermatologist:</Text>
            <Text style={styles.infoValue}>{existingShare.dermatologist_name}</Text>
          </View>

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Email:</Text>
            <Text style={styles.infoValue}>{existingShare.dermatologist_email}</Text>
          </View>

          {existingShare.share_date && (
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Shared on:</Text>
              <Text style={styles.infoValue}>
                {new Date(existingShare.share_date).toLocaleDateString()}
              </Text>
            </View>
          )}

          {existingShare.share_message && (
            <View style={styles.messageBox}>
              <Text style={styles.messageLabel}>Your Message:</Text>
              <Text style={styles.messageText}>{existingShare.share_message}</Text>
            </View>
          )}

          {existingShare.dermatologist_reviewed && (
            <View style={styles.reviewSection}>
              <Text style={styles.reviewTitle}>📝 Dermatologist's Review</Text>

              {existingShare.dermatologist_notes && (
                <View style={styles.reviewBox}>
                  <Text style={styles.reviewLabel}>Notes:</Text>
                  <Text style={styles.reviewText}>{existingShare.dermatologist_notes}</Text>
                </View>
              )}

              {existingShare.dermatologist_recommendation && (
                <View style={styles.reviewBox}>
                  <Text style={styles.reviewLabel}>Recommendation:</Text>
                  <Text style={styles.reviewText}>{existingShare.dermatologist_recommendation}</Text>
                </View>
              )}
            </View>
          )}

          {!existingShare.dermatologist_reviewed && (
            <Text style={styles.pendingText}>
              ⏳ Awaiting dermatologist review
            </Text>
          )}
        </View>
      </View>
    );
  }

  return (
    <View style={styles.contentContainer}>
      <Text style={styles.title}>👨‍⚕️ Share with Dermatologist</Text>
      <Text style={styles.subtitle}>
        Send this analysis to a dermatologist for professional review and consultation
      </Text>

      <View style={styles.form}>
        <View style={styles.inputGroup}>
          <Text style={styles.label}>Dermatologist's Name *</Text>
          <TextInput
            style={styles.input}
            placeholder="Dr. Jane Smith"
            value={dermatologistName}
            onChangeText={setDermatologistName}
            placeholderTextColor="#9ca3af"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Dermatologist's Email *</Text>
          <TextInput
            style={styles.input}
            placeholder="doctor@example.com"
            value={dermatologistEmail}
            onChangeText={setDermatologistEmail}
            keyboardType="email-address"
            autoCapitalize="none"
            placeholderTextColor="#9ca3af"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Message (Optional)</Text>
          <Text style={styles.hint}>
            Include any specific questions or concerns
          </Text>
          <TextInput
            style={[styles.input, styles.textArea]}
            placeholder="Please review this lesion and provide your professional opinion..."
            multiline
            numberOfLines={4}
            value={message}
            onChangeText={setMessage}
            placeholderTextColor="#9ca3af"
          />
        </View>

        <Pressable style={styles.shareButton} onPress={handleShare}>
          <Text style={styles.shareButtonText}>📤 Share Analysis</Text>
        </Pressable>

        <Text style={styles.infoNote}>
          ℹ️ The dermatologist will receive a secure link to view your analysis, including the image, AI prediction, and all clinical data you've provided.
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  contentContainer: {
    padding: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2d3748',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#64748b',
    marginBottom: 20,
    lineHeight: 20,
  },
  form: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#475569',
    marginBottom: 6,
  },
  hint: {
    fontSize: 12,
    color: '#9ca3af',
    marginBottom: 6,
    fontStyle: 'italic',
  },
  input: {
    borderWidth: 1,
    borderColor: '#cbd5e0',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    color: '#2d3748',
  },
  textArea: {
    minHeight: 100,
    textAlignVertical: 'top',
  },
  shareButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 12,
  },
  shareButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  infoNote: {
    fontSize: 12,
    color: '#64748b',
    fontStyle: 'italic',
    textAlign: 'center',
    lineHeight: 18,
  },
  sharedSection: {
    backgroundColor: '#ecfdf5',
    borderRadius: 12,
    padding: 16,
    borderWidth: 2,
    borderColor: '#10b981',
  },
  sharedHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sharedTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#065f46',
  },
  reviewedBadge: {
    backgroundColor: '#10b981',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  reviewedBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#fff',
  },
  infoRow: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  infoLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#047857',
    width: 110,
  },
  infoValue: {
    fontSize: 14,
    color: '#065f46',
    flex: 1,
  },
  messageBox: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginTop: 12,
  },
  messageLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#047857',
    marginBottom: 6,
  },
  messageText: {
    fontSize: 14,
    color: '#065f46',
    lineHeight: 20,
  },
  reviewSection: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 14,
    marginTop: 16,
  },
  reviewTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#047857',
    marginBottom: 12,
  },
  reviewBox: {
    marginBottom: 12,
  },
  reviewLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#059669',
    marginBottom: 4,
  },
  reviewText: {
    fontSize: 14,
    color: '#065f46',
    lineHeight: 20,
  },
  pendingText: {
    fontSize: 14,
    color: '#f59e0b',
    fontStyle: 'italic',
    textAlign: 'center',
    marginTop: 12,
    fontWeight: '600',
  },
});
