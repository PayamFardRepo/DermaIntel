import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  Pressable,
  Switch
} from 'react-native';

interface MedicalHistoryProps {
  onMedicalHistoryChange: (history: MedicalHistoryData) => void;
  initialHistory?: MedicalHistoryData;
}

export interface MedicalHistoryData {
  family_history_skin_cancer?: boolean;
  family_history_details?: string;
  previous_skin_cancers?: boolean;
  previous_skin_cancers_details?: string;
  immunosuppression?: boolean;
  immunosuppression_details?: string;
  sun_exposure_level?: string;
  sun_exposure_details?: string;
  history_of_sunburns?: boolean;
  sunburn_details?: string;
  tanning_bed_use?: boolean;
  tanning_bed_frequency?: string;
  other_risk_factors?: string;
}

export default function MedicalHistory({ onMedicalHistoryChange, initialHistory }: MedicalHistoryProps) {
  const [familyHistory, setFamilyHistory] = useState<boolean>(initialHistory?.family_history_skin_cancer || false);
  const [familyDetails, setFamilyDetails] = useState<string>(initialHistory?.family_history_details || '');

  const [previousCancers, setPreviousCancers] = useState<boolean>(initialHistory?.previous_skin_cancers || false);
  const [previousCancersDetails, setPreviousCancersDetails] = useState<string>(initialHistory?.previous_skin_cancers_details || '');

  const [immunosuppressed, setImmunosuppressed] = useState<boolean>(initialHistory?.immunosuppression || false);
  const [immunoDetails, setImmunoDetails] = useState<string>(initialHistory?.immunosuppression_details || '');

  const [sunExposure, setSunExposure] = useState<string>(initialHistory?.sun_exposure_level || 'moderate');
  const [sunDetails, setSunDetails] = useState<string>(initialHistory?.sun_exposure_details || '');

  const [hasSunburns, setHasSunburns] = useState<boolean>(initialHistory?.history_of_sunburns || false);
  const [sunburnDetails, setSunburnDetails] = useState<string>(initialHistory?.sunburn_details || '');

  const [tanningBed, setTanningBed] = useState<boolean>(initialHistory?.tanning_bed_use || false);
  const [tanningFrequency, setTanningFrequency] = useState<string>(initialHistory?.tanning_bed_frequency || 'never');

  const [otherRisks, setOtherRisks] = useState<string>(initialHistory?.other_risk_factors || '');

  const updateHistory = () => {
    const historyData: MedicalHistoryData = {
      family_history_skin_cancer: familyHistory,
      family_history_details: familyHistory && familyDetails ? familyDetails : undefined,
      previous_skin_cancers: previousCancers,
      previous_skin_cancers_details: previousCancers && previousCancersDetails ? previousCancersDetails : undefined,
      immunosuppression: immunosuppressed,
      immunosuppression_details: immunosuppressed && immunoDetails ? immunoDetails : undefined,
      sun_exposure_level: sunExposure,
      sun_exposure_details: sunDetails || undefined,
      history_of_sunburns: hasSunburns,
      sunburn_details: hasSunburns && sunburnDetails ? sunburnDetails : undefined,
      tanning_bed_use: tanningBed,
      tanning_bed_frequency: tanningBed ? tanningFrequency : undefined,
      other_risk_factors: otherRisks || undefined
    };

    onMedicalHistoryChange(historyData);
  };

  React.useEffect(() => {
    updateHistory();
  }, [familyHistory, familyDetails, previousCancers, previousCancersDetails, immunosuppressed, immunoDetails,
      sunExposure, sunDetails, hasSunburns, sunburnDetails, tanningBed, tanningFrequency, otherRisks]);

  return (
    <View style={styles.contentContainer}>
      <Text style={styles.title}>🏥 Medical History & Risk Factors</Text>
      <Text style={styles.subtitle}>
        Track important risk factors for skin conditions and cancer
      </Text>

      {/* Family History Section */}
      <View style={styles.section}>
        <View style={styles.toggleHeader}>
          <Text style={styles.sectionTitle}>👨‍👩‍👧‍👦 Family History of Skin Cancer</Text>
          <Switch
            value={familyHistory}
            onValueChange={setFamilyHistory}
            trackColor={{ false: '#e5e7eb', true: '#fca5a5' }}
            thumbColor={familyHistory ? '#dc2626' : '#f3f4f6'}
          />
        </View>

        {familyHistory && (
          <TextInput
            style={styles.detailsInput}
            placeholder="Which relatives? What type of cancer?"
            multiline
            numberOfLines={3}
            value={familyDetails}
            onChangeText={setFamilyDetails}
            placeholderTextColor="#9ca3af"
          />
        )}
      </View>

      {/* Previous Skin Cancers Section */}
      <View style={styles.section}>
        <View style={styles.toggleHeader}>
          <Text style={styles.sectionTitle}>📋 Previous Skin Cancers</Text>
          <Switch
            value={previousCancers}
            onValueChange={setPreviousCancers}
            trackColor={{ false: '#e5e7eb', true: '#fca5a5' }}
            thumbColor={previousCancers ? '#dc2626' : '#f3f4f6'}
          />
        </View>

        {previousCancers && (
          <TextInput
            style={styles.detailsInput}
            placeholder="Type, location, date of diagnosis, treatment..."
            multiline
            numberOfLines={3}
            value={previousCancersDetails}
            onChangeText={setPreviousCancersDetails}
            placeholderTextColor="#9ca3af"
          />
        )}
      </View>

      {/* Immunosuppression Section */}
      <View style={styles.section}>
        <View style={styles.toggleHeader}>
          <Text style={styles.sectionTitle}>🛡️ Immunosuppression</Text>
          <Switch
            value={immunosuppressed}
            onValueChange={setImmunosuppressed}
            trackColor={{ false: '#e5e7eb', true: '#fdba74' }}
            thumbColor={immunosuppressed ? '#f97316' : '#f3f4f6'}
          />
        </View>

        {immunosuppressed && (
          <TextInput
            style={styles.detailsInput}
            placeholder="Organ transplant, HIV, immunosuppressive medications, autoimmune disease..."
            multiline
            numberOfLines={3}
            value={immunoDetails}
            onChangeText={setImmunoDetails}
            placeholderTextColor="#9ca3af"
          />
        )}
      </View>

      {/* Sun Exposure Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>☀️ Sun Exposure Level</Text>
        <Text style={styles.sectionHint}>
          Consider occupation, outdoor activities, geographic location
        </Text>

        <View style={styles.exposureButtons}>
          {[
            { value: 'minimal', label: 'Minimal', emoji: '🌙' },
            { value: 'moderate', label: 'Moderate', emoji: '⛅' },
            { value: 'high', label: 'High', emoji: '☀️' },
            { value: 'very_high', label: 'Very High', emoji: '🔥' }
          ].map((level) => (
            <Pressable
              key={level.value}
              style={[
                styles.exposureButton,
                sunExposure === level.value && styles.exposureButtonActive
              ]}
              onPress={() => setSunExposure(level.value)}
            >
              <Text style={styles.exposureEmoji}>{level.emoji}</Text>
              <Text style={[
                styles.exposureButtonText,
                sunExposure === level.value && styles.exposureButtonTextActive
              ]}>
                {level.label}
              </Text>
            </Pressable>
          ))}
        </View>

        <TextInput
          style={styles.detailsInput}
          placeholder="Additional details about sun exposure..."
          multiline
          numberOfLines={2}
          value={sunDetails}
          onChangeText={setSunDetails}
          placeholderTextColor="#9ca3af"
        />
      </View>

      {/* Sunburns Section */}
      <View style={styles.section}>
        <View style={styles.toggleHeader}>
          <Text style={styles.sectionTitle}>🔥 History of Severe Sunburns</Text>
          <Switch
            value={hasSunburns}
            onValueChange={setHasSunburns}
            trackColor={{ false: '#e5e7eb', true: '#fb923c' }}
            thumbColor={hasSunburns ? '#ea580c' : '#f3f4f6'}
          />
        </View>

        {hasSunburns && (
          <TextInput
            style={styles.detailsInput}
            placeholder="When? How many? Blistering sunburns in childhood/adolescence..."
            multiline
            numberOfLines={2}
            value={sunburnDetails}
            onChangeText={setSunburnDetails}
            placeholderTextColor="#9ca3af"
          />
        )}
      </View>

      {/* Tanning Bed Section */}
      <View style={styles.section}>
        <View style={styles.toggleHeader}>
          <Text style={styles.sectionTitle}>🛏️ Tanning Bed Use</Text>
          <Switch
            value={tanningBed}
            onValueChange={setTanningBed}
            trackColor={{ false: '#e5e7eb', true: '#c084fc' }}
            thumbColor={tanningBed ? '#9333ea' : '#f3f4f6'}
          />
        </View>

        {tanningBed && (
          <View style={styles.frequencyContainer}>
            <Text style={styles.frequencyLabel}>Frequency:</Text>
            <View style={styles.frequencyButtons}>
              {[
                { value: 'rarely', label: 'Rarely' },
                { value: 'occasionally', label: 'Occasionally' },
                { value: 'regularly', label: 'Regularly' },
                { value: 'frequently', label: 'Frequently' }
              ].map((freq) => (
                <Pressable
                  key={freq.value}
                  style={[
                    styles.frequencyButton,
                    tanningFrequency === freq.value && styles.frequencyButtonActive
                  ]}
                  onPress={() => setTanningFrequency(freq.value)}
                >
                  <Text style={[
                    styles.frequencyButtonText,
                    tanningFrequency === freq.value && styles.frequencyButtonTextActive
                  ]}>
                    {freq.label}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>
        )}
      </View>

      {/* Other Risk Factors Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📝 Other Risk Factors</Text>
        <Text style={styles.sectionHint}>
          Radiation therapy, chemical exposure, scars/burns, etc.
        </Text>
        <TextInput
          style={styles.detailsInput}
          placeholder="Any other relevant risk factors..."
          multiline
          numberOfLines={3}
          value={otherRisks}
          onChangeText={setOtherRisks}
          placeholderTextColor="#9ca3af"
        />
      </View>

      <Text style={styles.infoNote}>
        ℹ️ This information helps assess risk level and provides important context for diagnosis and monitoring.
      </Text>
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
  section: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2d3748',
    flex: 1,
  },
  sectionHint: {
    fontSize: 12,
    color: '#9ca3af',
    marginBottom: 12,
    fontStyle: 'italic',
  },
  toggleHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  detailsInput: {
    borderWidth: 1,
    borderColor: '#cbd5e0',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    color: '#2d3748',
    minHeight: 80,
    textAlignVertical: 'top',
    marginTop: 12,
  },
  exposureButtons: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 12,
  },
  exposureButton: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#cbd5e0',
    backgroundColor: '#f9fafb',
    alignItems: 'center',
  },
  exposureButtonActive: {
    backgroundColor: '#fbbf24',
    borderColor: '#f59e0b',
  },
  exposureEmoji: {
    fontSize: 20,
    marginBottom: 4,
  },
  exposureButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#64748b',
    textAlign: 'center',
  },
  exposureButtonTextActive: {
    color: '#fff',
  },
  frequencyContainer: {
    marginTop: 12,
  },
  frequencyLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#475569',
    marginBottom: 8,
  },
  frequencyButtons: {
    flexDirection: 'row',
    gap: 8,
    flexWrap: 'wrap',
  },
  frequencyButton: {
    flex: 1,
    minWidth: '45%',
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#cbd5e0',
    backgroundColor: '#f9fafb',
    alignItems: 'center',
  },
  frequencyButtonActive: {
    backgroundColor: '#9333ea',
    borderColor: '#7e22ce',
  },
  frequencyButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#64748b',
  },
  frequencyButtonTextActive: {
    color: '#fff',
  },
  infoNote: {
    fontSize: 12,
    color: '#64748b',
    fontStyle: 'italic',
    textAlign: 'center',
    lineHeight: 18,
    marginTop: 8,
  },
});
