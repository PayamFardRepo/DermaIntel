import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
  RefreshControl,
  Modal,
  Switch,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../config';

interface ConsultationNote {
  id: number;
  consultation_id?: number;
  referral_id?: number;
  chief_complaint: string;
  history_of_present_illness?: string;
  physical_examination?: string;
  diagnosis?: string;
  differential_diagnoses?: string[];
  icd_codes?: string[];
  treatment_plan?: string;
  prescriptions?: any[];
  procedures_performed?: string[];
  follow_up_recommended: boolean;
  follow_up_timeframe?: string;
  note_status: string;
  signed_by_provider: boolean;
  created_at: string;
  updated_at?: string;
}

interface Prescription {
  medication: string;
  dosage: string;
  frequency: string;
  duration: string;
}

export default function ConsultationNotesScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'notes' | 'drafts'>('notes');
  const [notes, setNotes] = useState<ConsultationNote[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Create/Edit Modal
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingNote, setEditingNote] = useState<ConsultationNote | null>(null);
  const [saving, setSaving] = useState(false);

  // Form fields
  const [chiefComplaint, setChiefComplaint] = useState('');
  const [historyOfPresentIllness, setHistoryOfPresentIllness] = useState('');
  const [physicalExamination, setPhysicalExamination] = useState('');
  const [diagnosis, setDiagnosis] = useState('');
  const [differentialDiagnoses, setDifferentialDiagnoses] = useState('');
  const [icdCodes, setIcdCodes] = useState('');
  const [treatmentPlan, setTreatmentPlan] = useState('');
  const [proceduresPerformed, setProceduresPerformed] = useState('');
  const [followUpRecommended, setFollowUpRecommended] = useState(false);
  const [followUpTimeframe, setFollowUpTimeframe] = useState('');
  const [noteStatus, setNoteStatus] = useState<'draft' | 'final'>('draft');

  // Prescriptions
  const [prescriptions, setPrescriptions] = useState<Prescription[]>([]);
  const [showPrescriptionModal, setShowPrescriptionModal] = useState(false);
  const [currentPrescription, setCurrentPrescription] = useState<Prescription>({
    medication: '',
    dosage: '',
    frequency: '',
    duration: '',
  });

  // Detail Modal
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [selectedNote, setSelectedNote] = useState<ConsultationNote | null>(null);

  // SOAP Section Accordion
  const [expandedSection, setExpandedSection] = useState<string | null>('subjective');

  useEffect(() => {
    fetchNotes();
  }, []);

  const fetchNotes = async () => {
    try {
      const token = await AsyncStorage.getItem('userToken');
      const response = await fetch(`${API_URL}/consultation-notes`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setNotes(data);
      }
    } catch (error) {
      console.error('Error fetching notes:', error);
      Alert.alert('Error', 'Failed to fetch consultation notes');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchNotes();
  }, []);

  const resetForm = () => {
    setChiefComplaint('');
    setHistoryOfPresentIllness('');
    setPhysicalExamination('');
    setDiagnosis('');
    setDifferentialDiagnoses('');
    setIcdCodes('');
    setTreatmentPlan('');
    setProceduresPerformed('');
    setFollowUpRecommended(false);
    setFollowUpTimeframe('');
    setNoteStatus('draft');
    setPrescriptions([]);
    setEditingNote(null);
    setExpandedSection('subjective');
  };

  const openCreateModal = () => {
    resetForm();
    setShowCreateModal(true);
  };

  const openEditModal = (note: ConsultationNote) => {
    if (note.note_status !== 'draft') {
      Alert.alert('Cannot Edit', 'Only draft notes can be edited');
      return;
    }

    setEditingNote(note);
    setChiefComplaint(note.chief_complaint || '');
    setHistoryOfPresentIllness(note.history_of_present_illness || '');
    setPhysicalExamination(note.physical_examination || '');
    setDiagnosis(note.diagnosis || '');
    setDifferentialDiagnoses(note.differential_diagnoses?.join(', ') || '');
    setIcdCodes(note.icd_codes?.join(', ') || '');
    setTreatmentPlan(note.treatment_plan || '');
    setProceduresPerformed(note.procedures_performed?.join(', ') || '');
    setFollowUpRecommended(note.follow_up_recommended || false);
    setFollowUpTimeframe(note.follow_up_timeframe || '');
    setNoteStatus(note.note_status as 'draft' | 'final');
    setPrescriptions(note.prescriptions || []);
    setShowCreateModal(true);
  };

  const handleSaveNote = async () => {
    if (!chiefComplaint.trim()) {
      Alert.alert('Required Field', 'Please enter the chief complaint');
      return;
    }

    setSaving(true);
    try {
      const token = await AsyncStorage.getItem('userToken');
      const noteData = {
        chief_complaint: chiefComplaint.trim(),
        history_of_present_illness: historyOfPresentIllness.trim() || null,
        physical_examination: physicalExamination.trim() || null,
        diagnosis: diagnosis.trim() || null,
        differential_diagnoses: differentialDiagnoses.trim()
          ? differentialDiagnoses.split(',').map(d => d.trim()).filter(d => d)
          : null,
        icd_codes: icdCodes.trim()
          ? icdCodes.split(',').map(c => c.trim()).filter(c => c)
          : null,
        treatment_plan: treatmentPlan.trim() || null,
        prescriptions: prescriptions.length > 0 ? prescriptions : null,
        procedures_performed: proceduresPerformed.trim()
          ? proceduresPerformed.split(',').map(p => p.trim()).filter(p => p)
          : null,
        follow_up_recommended: followUpRecommended,
        follow_up_timeframe: followUpTimeframe.trim() || null,
        note_status: noteStatus,
      };

      const url = editingNote
        ? `${API_URL}/consultation-notes/${editingNote.id}`
        : `${API_URL}/consultation-notes`;

      const method = editingNote ? 'PUT' : 'POST';

      const response = await fetch(url, {
        method,
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(noteData),
      });

      if (response.ok) {
        Alert.alert(
          'Success',
          editingNote ? 'Note updated successfully' : 'Note created successfully'
        );
        setShowCreateModal(false);
        resetForm();
        fetchNotes();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to save note');
      }
    } catch (error) {
      console.error('Error saving note:', error);
      Alert.alert('Error', 'Failed to save note');
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteNote = async (noteId: number) => {
    Alert.alert(
      'Delete Note',
      'Are you sure you want to delete this note? This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              const token = await AsyncStorage.getItem('userToken');
              const response = await fetch(`${API_URL}/consultation-notes/${noteId}`, {
                method: 'DELETE',
                headers: {
                  'Authorization': `Bearer ${token}`,
                },
              });

              if (response.ok) {
                Alert.alert('Success', 'Note deleted successfully');
                fetchNotes();
                if (showDetailModal) {
                  setShowDetailModal(false);
                }
              } else {
                const error = await response.json();
                Alert.alert('Error', error.detail || 'Failed to delete note');
              }
            } catch (error) {
              console.error('Error deleting note:', error);
              Alert.alert('Error', 'Failed to delete note');
            }
          },
        },
      ]
    );
  };

  const handleFinalizeNote = async (noteId: number) => {
    Alert.alert(
      'Finalize Note',
      'Once finalized, this note cannot be edited. Are you sure?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Finalize',
          onPress: async () => {
            try {
              const token = await AsyncStorage.getItem('userToken');
              const response = await fetch(`${API_URL}/consultation-notes/${noteId}`, {
                method: 'PUT',
                headers: {
                  'Authorization': `Bearer ${token}`,
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({ note_status: 'final' }),
              });

              if (response.ok) {
                Alert.alert('Success', 'Note finalized successfully');
                fetchNotes();
                if (showDetailModal && selectedNote?.id === noteId) {
                  setSelectedNote({ ...selectedNote, note_status: 'final' });
                }
              } else {
                const error = await response.json();
                Alert.alert('Error', error.detail || 'Failed to finalize note');
              }
            } catch (error) {
              console.error('Error finalizing note:', error);
              Alert.alert('Error', 'Failed to finalize note');
            }
          },
        },
      ]
    );
  };

  const addPrescription = () => {
    if (!currentPrescription.medication.trim()) {
      Alert.alert('Required', 'Please enter medication name');
      return;
    }
    setPrescriptions([...prescriptions, currentPrescription]);
    setCurrentPrescription({ medication: '', dosage: '', frequency: '', duration: '' });
    setShowPrescriptionModal(false);
  };

  const removePrescription = (index: number) => {
    setPrescriptions(prescriptions.filter((_, i) => i !== index));
  };

  const openDetailModal = (note: ConsultationNote) => {
    setSelectedNote(note);
    setShowDetailModal(true);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'draft': return '#FFA726';
      case 'final': return '#4CAF50';
      case 'amended': return '#2196F3';
      default: return '#9E9E9E';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const filteredNotes = notes.filter(note => {
    if (activeTab === 'drafts') {
      return note.note_status === 'draft';
    }
    return note.note_status !== 'draft';
  });

  const renderNoteCard = (note: ConsultationNote) => (
    <TouchableOpacity
      key={note.id}
      style={styles.noteCard}
      onPress={() => openDetailModal(note)}
    >
      <View style={styles.noteHeader}>
        <View style={styles.noteHeaderLeft}>
          <Ionicons name="document-text" size={20} color="#667eea" />
          <Text style={styles.noteTitle} numberOfLines={1}>
            {note.chief_complaint}
          </Text>
        </View>
        <View style={[styles.statusBadge, { backgroundColor: getStatusColor(note.note_status) }]}>
          <Text style={styles.statusText}>{note.note_status.toUpperCase()}</Text>
        </View>
      </View>

      {note.diagnosis && (
        <View style={styles.noteRow}>
          <Text style={styles.noteLabel}>Diagnosis:</Text>
          <Text style={styles.noteValue} numberOfLines={1}>{note.diagnosis}</Text>
        </View>
      )}

      <View style={styles.noteFooter}>
        <Text style={styles.noteDate}>{formatDate(note.created_at)}</Text>
        <View style={styles.noteActions}>
          {note.note_status === 'draft' && (
            <>
              <TouchableOpacity
                style={styles.actionButton}
                onPress={() => openEditModal(note)}
              >
                <Ionicons name="pencil" size={16} color="#667eea" />
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.actionButton}
                onPress={() => handleFinalizeNote(note.id)}
              >
                <Ionicons name="checkmark-circle" size={16} color="#4CAF50" />
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.actionButton}
                onPress={() => handleDeleteNote(note.id)}
              >
                <Ionicons name="trash" size={16} color="#F44336" />
              </TouchableOpacity>
            </>
          )}
          {note.follow_up_recommended && (
            <View style={styles.followUpIndicator}>
              <Ionicons name="calendar" size={14} color="#FF9800" />
            </View>
          )}
        </View>
      </View>
    </TouchableOpacity>
  );

  const renderSOAPSection = (
    title: string,
    sectionKey: string,
    icon: string,
    color: string,
    content: React.ReactNode
  ) => (
    <View style={styles.soapSection}>
      <TouchableOpacity
        style={[styles.soapHeader, { borderLeftColor: color }]}
        onPress={() => setExpandedSection(expandedSection === sectionKey ? null : sectionKey)}
      >
        <View style={styles.soapHeaderLeft}>
          <Ionicons name={icon as any} size={20} color={color} />
          <Text style={[styles.soapTitle, { color }]}>{title}</Text>
        </View>
        <Ionicons
          name={expandedSection === sectionKey ? 'chevron-up' : 'chevron-down'}
          size={20}
          color="#666"
        />
      </TouchableOpacity>
      {expandedSection === sectionKey && (
        <View style={styles.soapContent}>
          {content}
        </View>
      )}
    </View>
  );

  const renderCreateModal = () => (
    <Modal
      visible={showCreateModal}
      animationType="slide"
      presentationStyle="pageSheet"
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setShowCreateModal(false)}>
            <Text style={styles.modalCancel}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.modalTitle}>
            {editingNote ? 'Edit Note' : 'New Consultation Note'}
          </Text>
          <TouchableOpacity onPress={handleSaveNote} disabled={saving}>
            {saving ? (
              <ActivityIndicator size="small" color="#667eea" />
            ) : (
              <Text style={styles.modalSave}>Save</Text>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.modalScroll}>
          {/* Subjective - S */}
          {renderSOAPSection(
            'Subjective',
            'subjective',
            'person',
            '#4CAF50',
            <>
              <Text style={styles.inputLabel}>Chief Complaint *</Text>
              <TextInput
                style={styles.textInput}
                value={chiefComplaint}
                onChangeText={setChiefComplaint}
                placeholder="Primary reason for visit"
                multiline
              />

              <Text style={styles.inputLabel}>History of Present Illness</Text>
              <TextInput
                style={[styles.textInput, styles.largeInput]}
                value={historyOfPresentIllness}
                onChangeText={setHistoryOfPresentIllness}
                placeholder="Detailed history of current condition"
                multiline
                numberOfLines={4}
              />
            </>
          )}

          {/* Objective - O */}
          {renderSOAPSection(
            'Objective',
            'objective',
            'eye',
            '#2196F3',
            <>
              <Text style={styles.inputLabel}>Physical Examination</Text>
              <TextInput
                style={[styles.textInput, styles.largeInput]}
                value={physicalExamination}
                onChangeText={setPhysicalExamination}
                placeholder="Physical examination findings"
                multiline
                numberOfLines={4}
              />

              <Text style={styles.inputLabel}>Procedures Performed</Text>
              <TextInput
                style={styles.textInput}
                value={proceduresPerformed}
                onChangeText={setProceduresPerformed}
                placeholder="Comma-separated procedures"
              />
            </>
          )}

          {/* Assessment - A */}
          {renderSOAPSection(
            'Assessment',
            'assessment',
            'clipboard',
            '#FF9800',
            <>
              <Text style={styles.inputLabel}>Diagnosis</Text>
              <TextInput
                style={styles.textInput}
                value={diagnosis}
                onChangeText={setDiagnosis}
                placeholder="Primary diagnosis"
              />

              <Text style={styles.inputLabel}>Differential Diagnoses</Text>
              <TextInput
                style={styles.textInput}
                value={differentialDiagnoses}
                onChangeText={setDifferentialDiagnoses}
                placeholder="Comma-separated differential diagnoses"
              />

              <Text style={styles.inputLabel}>ICD Codes</Text>
              <TextInput
                style={styles.textInput}
                value={icdCodes}
                onChangeText={setIcdCodes}
                placeholder="Comma-separated ICD codes (e.g., L20.9, L40.0)"
              />
            </>
          )}

          {/* Plan - P */}
          {renderSOAPSection(
            'Plan',
            'plan',
            'list',
            '#9C27B0',
            <>
              <Text style={styles.inputLabel}>Treatment Plan</Text>
              <TextInput
                style={[styles.textInput, styles.largeInput]}
                value={treatmentPlan}
                onChangeText={setTreatmentPlan}
                placeholder="Treatment recommendations"
                multiline
                numberOfLines={4}
              />

              <View style={styles.prescriptionHeader}>
                <Text style={styles.inputLabel}>Prescriptions</Text>
                <TouchableOpacity
                  style={styles.addButton}
                  onPress={() => setShowPrescriptionModal(true)}
                >
                  <Ionicons name="add" size={16} color="#fff" />
                  <Text style={styles.addButtonText}>Add</Text>
                </TouchableOpacity>
              </View>

              {prescriptions.map((rx, index) => (
                <View key={index} style={styles.prescriptionItem}>
                  <View style={styles.prescriptionInfo}>
                    <Text style={styles.prescriptionMed}>{rx.medication}</Text>
                    <Text style={styles.prescriptionDetails}>
                      {rx.dosage} - {rx.frequency} for {rx.duration}
                    </Text>
                  </View>
                  <TouchableOpacity onPress={() => removePrescription(index)}>
                    <Ionicons name="close-circle" size={20} color="#F44336" />
                  </TouchableOpacity>
                </View>
              ))}

              <View style={styles.followUpSection}>
                <View style={styles.switchRow}>
                  <Text style={styles.inputLabel}>Follow-up Recommended</Text>
                  <Switch
                    value={followUpRecommended}
                    onValueChange={setFollowUpRecommended}
                    trackColor={{ false: '#ddd', true: '#667eea' }}
                    thumbColor="#fff"
                  />
                </View>

                {followUpRecommended && (
                  <>
                    <Text style={styles.inputLabel}>Follow-up Timeframe</Text>
                    <TextInput
                      style={styles.textInput}
                      value={followUpTimeframe}
                      onChangeText={setFollowUpTimeframe}
                      placeholder="e.g., 2 weeks, 1 month"
                    />
                  </>
                )}
              </View>
            </>
          )}

          {/* Note Status */}
          <View style={styles.statusSection}>
            <Text style={styles.sectionTitle}>Note Status</Text>
            <View style={styles.statusOptions}>
              <TouchableOpacity
                style={[
                  styles.statusOption,
                  noteStatus === 'draft' && styles.statusOptionActive,
                ]}
                onPress={() => setNoteStatus('draft')}
              >
                <Ionicons
                  name="create-outline"
                  size={20}
                  color={noteStatus === 'draft' ? '#fff' : '#666'}
                />
                <Text style={[
                  styles.statusOptionText,
                  noteStatus === 'draft' && styles.statusOptionTextActive,
                ]}>
                  Draft
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[
                  styles.statusOption,
                  noteStatus === 'final' && styles.statusOptionActive,
                ]}
                onPress={() => setNoteStatus('final')}
              >
                <Ionicons
                  name="checkmark-circle-outline"
                  size={20}
                  color={noteStatus === 'final' ? '#fff' : '#666'}
                />
                <Text style={[
                  styles.statusOptionText,
                  noteStatus === 'final' && styles.statusOptionTextActive,
                ]}>
                  Final
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </ScrollView>
      </View>
    </Modal>
  );

  const renderPrescriptionModal = () => (
    <Modal
      visible={showPrescriptionModal}
      animationType="slide"
      transparent
    >
      <View style={styles.prescriptionModalOverlay}>
        <View style={styles.prescriptionModalContent}>
          <Text style={styles.prescriptionModalTitle}>Add Prescription</Text>

          <Text style={styles.inputLabel}>Medication *</Text>
          <TextInput
            style={styles.textInput}
            value={currentPrescription.medication}
            onChangeText={(text) => setCurrentPrescription({ ...currentPrescription, medication: text })}
            placeholder="Medication name"
          />

          <Text style={styles.inputLabel}>Dosage</Text>
          <TextInput
            style={styles.textInput}
            value={currentPrescription.dosage}
            onChangeText={(text) => setCurrentPrescription({ ...currentPrescription, dosage: text })}
            placeholder="e.g., 10mg"
          />

          <Text style={styles.inputLabel}>Frequency</Text>
          <TextInput
            style={styles.textInput}
            value={currentPrescription.frequency}
            onChangeText={(text) => setCurrentPrescription({ ...currentPrescription, frequency: text })}
            placeholder="e.g., twice daily"
          />

          <Text style={styles.inputLabel}>Duration</Text>
          <TextInput
            style={styles.textInput}
            value={currentPrescription.duration}
            onChangeText={(text) => setCurrentPrescription({ ...currentPrescription, duration: text })}
            placeholder="e.g., 14 days"
          />

          <View style={styles.prescriptionModalButtons}>
            <TouchableOpacity
              style={styles.cancelButton}
              onPress={() => setShowPrescriptionModal(false)}
            >
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.confirmButton}
              onPress={addPrescription}
            >
              <Text style={styles.confirmButtonText}>Add</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );

  const renderDetailModal = () => (
    <Modal
      visible={showDetailModal}
      animationType="slide"
      presentationStyle="pageSheet"
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setShowDetailModal(false)}>
            <Ionicons name="close" size={24} color="#333" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Note Details</Text>
          <View style={styles.detailActions}>
            {selectedNote?.note_status === 'draft' && (
              <>
                <TouchableOpacity
                  style={styles.headerAction}
                  onPress={() => {
                    setShowDetailModal(false);
                    openEditModal(selectedNote);
                  }}
                >
                  <Ionicons name="pencil" size={20} color="#667eea" />
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.headerAction}
                  onPress={() => handleDeleteNote(selectedNote.id)}
                >
                  <Ionicons name="trash" size={20} color="#F44336" />
                </TouchableOpacity>
              </>
            )}
          </View>
        </View>

        {selectedNote && (
          <ScrollView style={styles.modalScroll}>
            <View style={styles.detailHeader}>
              <View style={[
                styles.detailStatusBadge,
                { backgroundColor: getStatusColor(selectedNote.note_status) }
              ]}>
                <Text style={styles.detailStatusText}>
                  {selectedNote.note_status.toUpperCase()}
                </Text>
              </View>
              <Text style={styles.detailDate}>
                Created: {formatDate(selectedNote.created_at)}
              </Text>
              {selectedNote.updated_at && (
                <Text style={styles.detailDate}>
                  Updated: {formatDate(selectedNote.updated_at)}
                </Text>
              )}
            </View>

            {/* Subjective */}
            <View style={styles.detailSection}>
              <View style={[styles.detailSectionHeader, { backgroundColor: '#E8F5E9' }]}>
                <Ionicons name="person" size={20} color="#4CAF50" />
                <Text style={[styles.detailSectionTitle, { color: '#4CAF50' }]}>
                  Subjective
                </Text>
              </View>
              <View style={styles.detailSectionContent}>
                <Text style={styles.detailLabel}>Chief Complaint</Text>
                <Text style={styles.detailValue}>{selectedNote.chief_complaint}</Text>

                {selectedNote.history_of_present_illness && (
                  <>
                    <Text style={styles.detailLabel}>History of Present Illness</Text>
                    <Text style={styles.detailValue}>{selectedNote.history_of_present_illness}</Text>
                  </>
                )}
              </View>
            </View>

            {/* Objective */}
            {(selectedNote.physical_examination || selectedNote.procedures_performed?.length) && (
              <View style={styles.detailSection}>
                <View style={[styles.detailSectionHeader, { backgroundColor: '#E3F2FD' }]}>
                  <Ionicons name="eye" size={20} color="#2196F3" />
                  <Text style={[styles.detailSectionTitle, { color: '#2196F3' }]}>
                    Objective
                  </Text>
                </View>
                <View style={styles.detailSectionContent}>
                  {selectedNote.physical_examination && (
                    <>
                      <Text style={styles.detailLabel}>Physical Examination</Text>
                      <Text style={styles.detailValue}>{selectedNote.physical_examination}</Text>
                    </>
                  )}

                  {selectedNote.procedures_performed && selectedNote.procedures_performed.length > 0 && (
                    <>
                      <Text style={styles.detailLabel}>Procedures Performed</Text>
                      {selectedNote.procedures_performed.map((proc, index) => (
                        <View key={index} style={styles.listItem}>
                          <Ionicons name="checkmark-circle" size={16} color="#2196F3" />
                          <Text style={styles.listItemText}>{proc}</Text>
                        </View>
                      ))}
                    </>
                  )}
                </View>
              </View>
            )}

            {/* Assessment */}
            {(selectedNote.diagnosis || selectedNote.differential_diagnoses?.length || selectedNote.icd_codes?.length) && (
              <View style={styles.detailSection}>
                <View style={[styles.detailSectionHeader, { backgroundColor: '#FFF3E0' }]}>
                  <Ionicons name="clipboard" size={20} color="#FF9800" />
                  <Text style={[styles.detailSectionTitle, { color: '#FF9800' }]}>
                    Assessment
                  </Text>
                </View>
                <View style={styles.detailSectionContent}>
                  {selectedNote.diagnosis && (
                    <>
                      <Text style={styles.detailLabel}>Diagnosis</Text>
                      <Text style={styles.detailValueHighlight}>{selectedNote.diagnosis}</Text>
                    </>
                  )}

                  {selectedNote.differential_diagnoses && selectedNote.differential_diagnoses.length > 0 && (
                    <>
                      <Text style={styles.detailLabel}>Differential Diagnoses</Text>
                      {selectedNote.differential_diagnoses.map((dx, index) => (
                        <View key={index} style={styles.listItem}>
                          <Text style={styles.listNumber}>{index + 1}.</Text>
                          <Text style={styles.listItemText}>{dx}</Text>
                        </View>
                      ))}
                    </>
                  )}

                  {selectedNote.icd_codes && selectedNote.icd_codes.length > 0 && (
                    <>
                      <Text style={styles.detailLabel}>ICD Codes</Text>
                      <View style={styles.tagContainer}>
                        {selectedNote.icd_codes.map((code, index) => (
                          <View key={index} style={styles.icdTag}>
                            <Text style={styles.icdTagText}>{code}</Text>
                          </View>
                        ))}
                      </View>
                    </>
                  )}
                </View>
              </View>
            )}

            {/* Plan */}
            {(selectedNote.treatment_plan || selectedNote.prescriptions?.length || selectedNote.follow_up_recommended) && (
              <View style={styles.detailSection}>
                <View style={[styles.detailSectionHeader, { backgroundColor: '#F3E5F5' }]}>
                  <Ionicons name="list" size={20} color="#9C27B0" />
                  <Text style={[styles.detailSectionTitle, { color: '#9C27B0' }]}>
                    Plan
                  </Text>
                </View>
                <View style={styles.detailSectionContent}>
                  {selectedNote.treatment_plan && (
                    <>
                      <Text style={styles.detailLabel}>Treatment Plan</Text>
                      <Text style={styles.detailValue}>{selectedNote.treatment_plan}</Text>
                    </>
                  )}

                  {selectedNote.prescriptions && selectedNote.prescriptions.length > 0 && (
                    <>
                      <Text style={styles.detailLabel}>Prescriptions</Text>
                      {selectedNote.prescriptions.map((rx: any, index: number) => (
                        <View key={index} style={styles.prescriptionCard}>
                          <View style={styles.prescriptionIcon}>
                            <Ionicons name="medical" size={16} color="#9C27B0" />
                          </View>
                          <View style={styles.prescriptionCardInfo}>
                            <Text style={styles.prescriptionCardMed}>{rx.medication}</Text>
                            <Text style={styles.prescriptionCardDetails}>
                              {rx.dosage} - {rx.frequency}
                            </Text>
                            {rx.duration && (
                              <Text style={styles.prescriptionCardDuration}>
                                Duration: {rx.duration}
                              </Text>
                            )}
                          </View>
                        </View>
                      ))}
                    </>
                  )}

                  {selectedNote.follow_up_recommended && (
                    <View style={styles.followUpCard}>
                      <Ionicons name="calendar" size={24} color="#FF9800" />
                      <View style={styles.followUpInfo}>
                        <Text style={styles.followUpTitle}>Follow-up Recommended</Text>
                        {selectedNote.follow_up_timeframe && (
                          <Text style={styles.followUpTimeframe}>
                            {selectedNote.follow_up_timeframe}
                          </Text>
                        )}
                      </View>
                    </View>
                  )}
                </View>
              </View>
            )}

            {selectedNote.note_status === 'draft' && (
              <TouchableOpacity
                style={styles.finalizeButton}
                onPress={() => handleFinalizeNote(selectedNote.id)}
              >
                <Ionicons name="checkmark-circle" size={20} color="#fff" />
                <Text style={styles.finalizeButtonText}>Finalize Note</Text>
              </TouchableOpacity>
            )}
          </ScrollView>
        )}
      </View>
    </Modal>
  );

  if (loading) {
    return (
      <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#fff" />
          <Text style={styles.loadingText}>Loading notes...</Text>
        </View>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Consultation Notes</Text>
        <TouchableOpacity onPress={openCreateModal} style={styles.addNoteButton}>
          <Ionicons name="add" size={24} color="#fff" />
        </TouchableOpacity>
      </View>

      {/* Tabs */}
      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'notes' && styles.activeTab]}
          onPress={() => setActiveTab('notes')}
        >
          <Ionicons
            name="document-text"
            size={20}
            color={activeTab === 'notes' ? '#667eea' : '#666'}
          />
          <Text style={[styles.tabText, activeTab === 'notes' && styles.activeTabText]}>
            Final Notes
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'drafts' && styles.activeTab]}
          onPress={() => setActiveTab('drafts')}
        >
          <Ionicons
            name="create"
            size={20}
            color={activeTab === 'drafts' ? '#667eea' : '#666'}
          />
          <Text style={[styles.tabText, activeTab === 'drafts' && styles.activeTabText]}>
            Drafts
          </Text>
          {notes.filter(n => n.note_status === 'draft').length > 0 && (
            <View style={styles.tabBadge}>
              <Text style={styles.tabBadgeText}>
                {notes.filter(n => n.note_status === 'draft').length}
              </Text>
            </View>
          )}
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#fff" />
        }
      >
        {filteredNotes.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons
              name={activeTab === 'drafts' ? 'create-outline' : 'document-text-outline'}
              size={64}
              color="rgba(255,255,255,0.5)"
            />
            <Text style={styles.emptyText}>
              {activeTab === 'drafts' ? 'No draft notes' : 'No consultation notes yet'}
            </Text>
            <TouchableOpacity style={styles.emptyButton} onPress={openCreateModal}>
              <Text style={styles.emptyButtonText}>Create New Note</Text>
            </TouchableOpacity>
          </View>
        ) : (
          filteredNotes.map(renderNoteCard)
        )}
      </ScrollView>

      {renderCreateModal()}
      {renderPrescriptionModal()}
      {renderDetailModal()}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#fff',
    marginTop: 10,
    fontSize: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 20,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  addNoteButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  tabContainer: {
    flexDirection: 'row',
    marginHorizontal: 20,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 4,
    marginBottom: 15,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderRadius: 10,
    gap: 6,
  },
  activeTab: {
    backgroundColor: '#f0f0f0',
  },
  tabText: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  activeTabText: {
    color: '#667eea',
    fontWeight: '600',
  },
  tabBadge: {
    backgroundColor: '#667eea',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
    marginLeft: 4,
  },
  tabBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
  },
  noteCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  noteHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  noteHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: 8,
  },
  noteTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    flex: 1,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  statusText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  noteRow: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  noteLabel: {
    fontSize: 13,
    color: '#666',
    marginRight: 8,
  },
  noteValue: {
    fontSize: 13,
    color: '#333',
    flex: 1,
  },
  noteFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  noteDate: {
    fontSize: 12,
    color: '#999',
  },
  noteActions: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    padding: 4,
  },
  followUpIndicator: {
    padding: 4,
  },
  emptyState: {
    alignItems: 'center',
    paddingTop: 60,
  },
  emptyText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 16,
    marginTop: 16,
    marginBottom: 20,
  },
  emptyButton: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  emptyButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  // Modal Styles
  modalContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 15,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  modalCancel: {
    fontSize: 16,
    color: '#666',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  modalSave: {
    fontSize: 16,
    color: '#667eea',
    fontWeight: '600',
  },
  modalScroll: {
    flex: 1,
    padding: 20,
  },
  // SOAP Sections
  soapSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    marginBottom: 12,
    overflow: 'hidden',
  },
  soapHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderLeftWidth: 4,
  },
  soapHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  soapTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  soapContent: {
    padding: 16,
    paddingTop: 0,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
    marginTop: 12,
    marginBottom: 6,
  },
  textInput: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    fontSize: 15,
    color: '#333',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  largeInput: {
    minHeight: 100,
    textAlignVertical: 'top',
  },
  prescriptionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 12,
  },
  addButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#667eea',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    gap: 4,
  },
  addButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  prescriptionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
  },
  prescriptionInfo: {
    flex: 1,
  },
  prescriptionMed: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  prescriptionDetails: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  followUpSection: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statusSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  statusOptions: {
    flexDirection: 'row',
    gap: 12,
  },
  statusOption: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#f5f5f5',
    gap: 8,
  },
  statusOptionActive: {
    backgroundColor: '#667eea',
  },
  statusOptionText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#666',
  },
  statusOptionTextActive: {
    color: '#fff',
  },
  // Prescription Modal
  prescriptionModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    padding: 20,
  },
  prescriptionModalContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
  },
  prescriptionModalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 16,
  },
  prescriptionModalButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 20,
  },
  cancelButton: {
    flex: 1,
    padding: 14,
    borderRadius: 8,
    backgroundColor: '#f5f5f5',
    alignItems: 'center',
  },
  cancelButtonText: {
    fontSize: 16,
    color: '#666',
    fontWeight: '500',
  },
  confirmButton: {
    flex: 1,
    padding: 14,
    borderRadius: 8,
    backgroundColor: '#667eea',
    alignItems: 'center',
  },
  confirmButtonText: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '600',
  },
  // Detail Modal
  detailActions: {
    flexDirection: 'row',
    gap: 16,
  },
  headerAction: {
    padding: 4,
  },
  detailHeader: {
    alignItems: 'center',
    marginBottom: 20,
  },
  detailStatusBadge: {
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 16,
    marginBottom: 10,
  },
  detailStatusText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  detailDate: {
    fontSize: 13,
    color: '#666',
    marginTop: 4,
  },
  detailSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    marginBottom: 12,
    overflow: 'hidden',
  },
  detailSectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    gap: 10,
  },
  detailSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  detailSectionContent: {
    padding: 16,
    paddingTop: 0,
  },
  detailLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#999',
    textTransform: 'uppercase',
    marginTop: 12,
    marginBottom: 4,
  },
  detailValue: {
    fontSize: 15,
    color: '#333',
    lineHeight: 22,
  },
  detailValueHighlight: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  listItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 6,
    gap: 8,
  },
  listNumber: {
    fontSize: 14,
    color: '#666',
    width: 20,
  },
  listItemText: {
    fontSize: 14,
    color: '#333',
    flex: 1,
  },
  tagContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
    gap: 8,
  },
  icdTag: {
    backgroundColor: '#FFF3E0',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 4,
  },
  icdTagText: {
    fontSize: 13,
    color: '#E65100',
    fontWeight: '500',
  },
  prescriptionCard: {
    flexDirection: 'row',
    backgroundColor: '#F3E5F5',
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
    gap: 12,
  },
  prescriptionIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  prescriptionCardInfo: {
    flex: 1,
  },
  prescriptionCardMed: {
    fontSize: 15,
    fontWeight: '600',
    color: '#333',
  },
  prescriptionCardDetails: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  prescriptionCardDuration: {
    fontSize: 12,
    color: '#9C27B0',
    marginTop: 4,
  },
  followUpCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF3E0',
    borderRadius: 8,
    padding: 16,
    marginTop: 12,
    gap: 12,
  },
  followUpInfo: {
    flex: 1,
  },
  followUpTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#E65100',
  },
  followUpTimeframe: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  finalizeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#4CAF50',
    padding: 16,
    borderRadius: 12,
    marginTop: 10,
    marginBottom: 30,
    gap: 8,
  },
  finalizeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
