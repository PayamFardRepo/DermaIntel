import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  Pressable,
  ScrollView
} from 'react-native';

interface MedicationListProps {
  onMedicationChange: (medications: Medication[]) => void;
  initialMedications?: Medication[];
}

export interface Medication {
  id?: string;
  name: string;
  dosage: string;
  start_date: string;
  purpose: string;
  skin_reaction: 'yes' | 'no' | 'unknown';
}

export default function MedicationList({ onMedicationChange, initialMedications }: MedicationListProps) {
  const [medications, setMedications] = useState<Medication[]>(
    initialMedications && initialMedications.length > 0
      ? initialMedications
      : [{ id: Date.now().toString(), name: '', dosage: '', start_date: '', purpose: '', skin_reaction: 'unknown' }]
  );

  const updateMedication = (index: number, field: keyof Medication, value: string) => {
    const updatedMeds = [...medications];
    updatedMeds[index] = { ...updatedMeds[index], [field]: value };
    setMedications(updatedMeds);
    onMedicationChange(updatedMeds);
  };

  const addMedication = () => {
    const newMed: Medication = {
      id: Date.now().toString(),
      name: '',
      dosage: '',
      start_date: '',
      purpose: '',
      skin_reaction: 'unknown'
    };
    const updatedMeds = [...medications, newMed];
    setMedications(updatedMeds);
    onMedicationChange(updatedMeds);
  };

  const removeMedication = (index: number) => {
    if (medications.length === 1) {
      // Keep at least one empty medication form
      const updatedMeds = [{ id: Date.now().toString(), name: '', dosage: '', start_date: '', purpose: '', skin_reaction: 'unknown' as const }];
      setMedications(updatedMeds);
      onMedicationChange(updatedMeds);
    } else {
      const updatedMeds = medications.filter((_, i) => i !== index);
      setMedications(updatedMeds);
      onMedicationChange(updatedMeds);
    }
  };

  return (
    <View style={styles.contentContainer}>
      <Text style={styles.title}>💊 Medication List</Text>
      <Text style={styles.subtitle}>
        Document all medications that might cause or affect skin reactions
      </Text>

      {medications.map((med, index) => (
        <View key={med.id || index} style={styles.medicationCard}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardTitle}>Medication {index + 1}</Text>
            {medications.length > 1 && (
              <Pressable onPress={() => removeMedication(index)} style={styles.removeButton}>
                <Text style={styles.removeButtonText}>Remove</Text>
              </Pressable>
            )}
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Medication Name *</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g., Aspirin, Lisinopril"
              value={med.name}
              onChangeText={(value) => updateMedication(index, 'name', value)}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Dosage</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g., 10mg, 20mg twice daily"
              value={med.dosage}
              onChangeText={(value) => updateMedication(index, 'dosage', value)}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Start Date</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g., 2024-01-15 or Jan 2024"
              value={med.start_date}
              onChangeText={(value) => updateMedication(index, 'start_date', value)}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Purpose/Condition</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g., Blood pressure, Pain relief"
              value={med.purpose}
              onChangeText={(value) => updateMedication(index, 'purpose', value)}
              placeholderTextColor="#9ca3af"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Suspected Skin Reaction?</Text>
            <View style={styles.reactionButtons}>
              {[
                { value: 'yes', label: 'Yes', color: '#ef4444' },
                { value: 'no', label: 'No', color: '#22c55e' },
                { value: 'unknown', label: 'Unknown', color: '#64748b' }
              ].map((option) => (
                <Pressable
                  key={option.value}
                  style={[
                    styles.reactionButton,
                    med.skin_reaction === option.value && {
                      backgroundColor: option.color,
                      borderColor: option.color
                    }
                  ]}
                  onPress={() => updateMedication(index, 'skin_reaction', option.value)}
                >
                  <Text
                    style={[
                      styles.reactionButtonText,
                      med.skin_reaction === option.value && styles.reactionButtonTextActive
                    ]}
                  >
                    {option.label}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>
        </View>
      ))}

      <Pressable style={styles.addButton} onPress={addMedication}>
        <Text style={styles.addButtonText}>+ Add Another Medication</Text>
      </Pressable>

      <Text style={styles.infoNote}>
        ℹ️ Some medications can cause skin reactions or photosensitivity. This information helps identify potential drug-induced skin conditions.
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
  medicationCard: {
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
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2d3748',
  },
  removeButton: {
    backgroundColor: '#fee2e2',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  removeButtonText: {
    color: '#dc2626',
    fontSize: 12,
    fontWeight: '600',
  },
  inputGroup: {
    marginBottom: 12,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#475569',
    marginBottom: 6,
  },
  input: {
    borderWidth: 1,
    borderColor: '#cbd5e0',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    color: '#2d3748',
  },
  reactionButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  reactionButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#cbd5e0',
    backgroundColor: '#f9fafb',
    alignItems: 'center',
  },
  reactionButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#64748b',
  },
  reactionButtonTextActive: {
    color: '#fff',
  },
  addButton: {
    backgroundColor: '#3b82f6',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 16,
  },
  addButtonText: {
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
});
