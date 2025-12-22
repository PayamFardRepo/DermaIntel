import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  Pressable,
  Switch
} from 'react-native';

interface SymptomTrackerProps {
  onSymptomChange: (symptoms: SymptomData) => void;
  initialSymptoms?: SymptomData;
}

export interface SymptomData {
  symptom_duration?: string;
  symptom_duration_value?: number;
  symptom_duration_unit?: string;
  symptom_changes?: string;
  symptom_itching?: boolean;
  symptom_itching_severity?: number;
  symptom_pain?: boolean;
  symptom_pain_severity?: number;
  symptom_bleeding?: boolean;
  symptom_bleeding_frequency?: string;
  symptom_notes?: string;
}

export default function SymptomTracker({ onSymptomChange, initialSymptoms }: SymptomTrackerProps) {
  const [durationValue, setDurationValue] = useState<string>(
    initialSymptoms?.symptom_duration_value?.toString() || ''
  );
  const [durationUnit, setDurationUnit] = useState<string>(
    initialSymptoms?.symptom_duration_unit || 'days'
  );
  const [changes, setChanges] = useState<string>(initialSymptoms?.symptom_changes || '');

  const [hasItching, setHasItching] = useState<boolean>(initialSymptoms?.symptom_itching || false);
  const [itchingSeverity, setItchingSeverity] = useState<number>(
    initialSymptoms?.symptom_itching_severity || 5
  );

  const [hasPain, setHasPain] = useState<boolean>(initialSymptoms?.symptom_pain || false);
  const [painSeverity, setPainSeverity] = useState<number>(
    initialSymptoms?.symptom_pain_severity || 5
  );

  const [hasBleeding, setHasBleeding] = useState<boolean>(initialSymptoms?.symptom_bleeding || false);
  const [bleedingFrequency, setBleedingFrequency] = useState<string>(
    initialSymptoms?.symptom_bleeding_frequency || 'occasional'
  );

  const [notes, setNotes] = useState<string>(initialSymptoms?.symptom_notes || '');

  const updateSymptoms = () => {
    const durationVal = parseInt(durationValue) || undefined;
    const symptomDuration = durationVal && durationUnit
      ? `${durationVal} ${durationUnit}`
      : undefined;

    const symptomData: SymptomData = {
      symptom_duration: symptomDuration,
      symptom_duration_value: durationVal,
      symptom_duration_unit: durationVal ? durationUnit : undefined,  // Only include if duration value exists
      symptom_changes: changes || undefined,
      symptom_itching: hasItching,
      symptom_itching_severity: hasItching ? itchingSeverity : undefined,
      symptom_pain: hasPain,
      symptom_pain_severity: hasPain ? painSeverity : undefined,
      symptom_bleeding: hasBleeding,
      symptom_bleeding_frequency: hasBleeding ? bleedingFrequency : undefined,
      symptom_notes: notes || undefined
    };

    onSymptomChange(symptomData);
  };

  // Call updateSymptoms whenever any field changes
  React.useEffect(() => {
    updateSymptoms();
  }, [durationValue, durationUnit, changes, hasItching, itchingSeverity, hasPain, painSeverity, hasBleeding, bleedingFrequency, notes]);

  return (
    <View style={styles.contentContainer}>
      <Text style={styles.title}>📋 Symptom Tracker</Text>
      <Text style={styles.subtitle}>
        Record details about the lesion to help track changes over time
      </Text>

      {/* Duration Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>⏱️ How long has this been present?</Text>

        <View style={styles.durationInputContainer}>
          <TextInput
            style={styles.durationValueInput}
            placeholder="e.g., 2"
            keyboardType="numeric"
            value={durationValue}
            onChangeText={setDurationValue}
            placeholderTextColor="#9ca3af"
          />

          <View style={styles.unitSelector}>
            {['days', 'weeks', 'months', 'years'].map((unit) => (
              <Pressable
                key={unit}
                style={[
                  styles.unitButton,
                  durationUnit === unit && styles.unitButtonActive
                ]}
                onPress={() => setDurationUnit(unit)}
              >
                <Text style={[
                  styles.unitButtonText,
                  durationUnit === unit && styles.unitButtonTextActive
                ]}>
                  {unit}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>
      </View>

      {/* Changes Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📈 Have you noticed any changes?</Text>
        <Text style={styles.sectionHint}>
          e.g., growing larger, changing color, texture changes
        </Text>
        <TextInput
          style={styles.textArea}
          placeholder="Describe any changes you've observed..."
          multiline
          numberOfLines={3}
          value={changes}
          onChangeText={setChanges}
          placeholderTextColor="#9ca3af"
        />
      </View>

      {/* Itching Section */}
      <View style={styles.section}>
        <View style={styles.symptomHeader}>
          <Text style={styles.sectionTitle}>🤚 Itching</Text>
          <Switch
            value={hasItching}
            onValueChange={setHasItching}
            trackColor={{ false: '#e5e7eb', true: '#93c5fd' }}
            thumbColor={hasItching ? '#3b82f6' : '#f3f4f6'}
          />
        </View>

        {hasItching && (
          <View style={styles.severityContainer}>
            <Text style={styles.severityLabel}>Severity: {itchingSeverity}/10</Text>
            <View style={styles.severityButtons}>
              {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((level) => (
                <Pressable
                  key={level}
                  style={[
                    styles.severityButton,
                    itchingSeverity === level && styles.severityButtonActive,
                    level > 7 && itchingSeverity === level && styles.severityButtonHighActive
                  ]}
                  onPress={() => setItchingSeverity(level)}
                >
                  <Text style={[
                    styles.severityButtonText,
                    itchingSeverity === level && styles.severityButtonTextActive
                  ]}>
                    {level}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>
        )}
      </View>

      {/* Pain Section */}
      <View style={styles.section}>
        <View style={styles.symptomHeader}>
          <Text style={styles.sectionTitle}>😖 Pain or Discomfort</Text>
          <Switch
            value={hasPain}
            onValueChange={setHasPain}
            trackColor={{ false: '#e5e7eb', true: '#fca5a5' }}
            thumbColor={hasPain ? '#ef4444' : '#f3f4f6'}
          />
        </View>

        {hasPain && (
          <View style={styles.severityContainer}>
            <Text style={styles.severityLabel}>Severity: {painSeverity}/10</Text>
            <View style={styles.severityButtons}>
              {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((level) => (
                <Pressable
                  key={level}
                  style={[
                    styles.severityButton,
                    painSeverity === level && styles.severityButtonActive,
                    level > 7 && painSeverity === level && styles.severityButtonHighActive
                  ]}
                  onPress={() => setPainSeverity(level)}
                >
                  <Text style={[
                    styles.severityButtonText,
                    painSeverity === level && styles.severityButtonTextActive
                  ]}>
                    {level}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>
        )}
      </View>

      {/* Bleeding Section */}
      <View style={styles.section}>
        <View style={styles.symptomHeader}>
          <Text style={styles.sectionTitle}>🩸 Bleeding</Text>
          <Switch
            value={hasBleeding}
            onValueChange={setHasBleeding}
            trackColor={{ false: '#e5e7eb', true: '#f87171' }}
            thumbColor={hasBleeding ? '#dc2626' : '#f3f4f6'}
          />
        </View>

        {hasBleeding && (
          <View style={styles.frequencyContainer}>
            <Text style={styles.frequencyLabel}>Frequency:</Text>
            <View style={styles.frequencyButtons}>
              {[
                { value: 'rare', label: 'Rare' },
                { value: 'occasional', label: 'Occasional' },
                { value: 'frequent', label: 'Frequent' }
              ].map((freq) => (
                <Pressable
                  key={freq.value}
                  style={[
                    styles.frequencyButton,
                    bleedingFrequency === freq.value && styles.frequencyButtonActive
                  ]}
                  onPress={() => setBleedingFrequency(freq.value)}
                >
                  <Text style={[
                    styles.frequencyButtonText,
                    bleedingFrequency === freq.value && styles.frequencyButtonTextActive
                  ]}>
                    {freq.label}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>
        )}
      </View>

      {/* Additional Notes */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📝 Additional Notes</Text>
        <Text style={styles.sectionHint}>
          Any other symptoms or observations
        </Text>
        <TextInput
          style={styles.textArea}
          placeholder="e.g., affects sleep, worsens in sunlight, etc."
          multiline
          numberOfLines={3}
          value={notes}
          onChangeText={setNotes}
          placeholderTextColor="#9ca3af"
        />
      </View>

      <Text style={styles.infoNote}>
        ℹ️ Tracking symptoms helps monitor the lesion's progression and provides valuable context for medical professionals.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
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
    marginBottom: 8,
  },
  sectionHint: {
    fontSize: 12,
    color: '#9ca3af',
    marginBottom: 12,
    fontStyle: 'italic',
  },
  durationInputContainer: {
    gap: 12,
  },
  durationValueInput: {
    borderWidth: 1,
    borderColor: '#cbd5e0',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    color: '#2d3748',
  },
  unitSelector: {
    flexDirection: 'row',
    gap: 8,
  },
  unitButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#cbd5e0',
    backgroundColor: '#f9fafb',
    alignItems: 'center',
  },
  unitButtonActive: {
    backgroundColor: '#3b82f6',
    borderColor: '#2563eb',
  },
  unitButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#64748b',
  },
  unitButtonTextActive: {
    color: '#fff',
  },
  textArea: {
    borderWidth: 1,
    borderColor: '#cbd5e0',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    color: '#2d3748',
    minHeight: 80,
    textAlignVertical: 'top',
  },
  symptomHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  severityContainer: {
    marginTop: 12,
  },
  severityLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#475569',
    marginBottom: 8,
  },
  severityButtons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  severityButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: '#cbd5e0',
    backgroundColor: '#f9fafb',
    justifyContent: 'center',
    alignItems: 'center',
  },
  severityButtonActive: {
    backgroundColor: '#3b82f6',
    borderColor: '#2563eb',
  },
  severityButtonHighActive: {
    backgroundColor: '#ef4444',
    borderColor: '#dc2626',
  },
  severityButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#64748b',
  },
  severityButtonTextActive: {
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
  },
  frequencyButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#cbd5e0',
    backgroundColor: '#f9fafb',
    alignItems: 'center',
  },
  frequencyButtonActive: {
    backgroundColor: '#dc2626',
    borderColor: '#b91c1c',
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
    marginTop: 8,
    marginBottom: 16,
    lineHeight: 18,
  },
});
