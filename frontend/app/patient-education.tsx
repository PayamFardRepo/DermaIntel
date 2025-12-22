import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  Pressable,
  ActivityIndicator,
  Alert,
  Modal,
  Platform,
  Linking,
  Share
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { useTranslation } from 'react-i18next';
import { API_BASE_URL } from '../config';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';

interface ConditionSummary {
  id: string;
  name: string;
  category: string;
  severity: string;
}

interface ConditionContent {
  id: string;
  name: string;
  description: string;
  symptoms: string[];
  causes: string[];
  care_instructions: string[];
  treatment_options: string[];
  warning_signs: string[];
  when_to_seek_help: string[];
  images?: string[];
}

interface DeliveryConfig {
  email_configured: boolean;
  sms_configured: boolean;
}

const LANGUAGE_OPTIONS = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Espanol' },
  { code: 'fr', label: 'Francais' },
  { code: 'de', label: 'Deutsch' },
  { code: 'zh', label: '中文' },
  { code: 'ja', label: '日本語' },
  { code: 'ar', label: 'العربية' },
];

export default function PatientEducationScreen() {
  const { t, i18n } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // State
  const [conditions, setConditions] = useState<ConditionSummary[]>([]);
  const [filteredConditions, setFilteredConditions] = useState<ConditionSummary[]>([]);
  const [selectedCondition, setSelectedCondition] = useState<ConditionContent | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [deliveryConfig, setDeliveryConfig] = useState<DeliveryConfig | null>(null);
  const [showLanguageModal, setShowLanguageModal] = useState(false);
  const [showSendModal, setShowSendModal] = useState(false);
  const [sendMethod, setSendMethod] = useState<'email' | 'sms'>('email');
  const [contactInfo, setContactInfo] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  // Get auth token
  const getAuthHeaders = async () => {
    const token = await AsyncStorage.getItem('accessToken');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  };

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    }
  }, [isAuthenticated]);

  // Load conditions on mount
  useEffect(() => {
    if (isAuthenticated) {
      loadConditions();
      loadDeliveryConfig();
    }
  }, [isAuthenticated, selectedLanguage]);

  // Filter conditions based on search and category
  useEffect(() => {
    let filtered = conditions;

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(c =>
        c.name.toLowerCase().includes(query) ||
        c.category.toLowerCase().includes(query)
      );
    }

    if (selectedCategory) {
      filtered = filtered.filter(c => c.category === selectedCategory);
    }

    setFilteredConditions(filtered);
  }, [conditions, searchQuery, selectedCategory]);

  const loadConditions = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(
        `${API_BASE_URL}/education/conditions?language=${selectedLanguage}`,
        { headers: await getAuthHeaders() }
      );

      if (!response.ok) throw new Error('Failed to load conditions');

      const data = await response.json();
      setConditions(data.conditions || []);
    } catch (error) {
      console.error('Error loading conditions:', error);
      Alert.alert('Error', 'Failed to load educational content');
    } finally {
      setIsLoading(false);
    }
  };

  const loadDeliveryConfig = async () => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/education/delivery-config`,
        { headers: await getAuthHeaders() }
      );

      if (response.ok) {
        const data = await response.json();
        setDeliveryConfig(data);
      }
    } catch (error) {
      console.error('Error loading delivery config:', error);
    }
  };

  const loadConditionContent = async (conditionId: string) => {
    try {
      setIsLoadingContent(true);
      const response = await fetch(
        `${API_BASE_URL}/education/content/${conditionId}?language=${selectedLanguage}`,
        { headers: await getAuthHeaders() }
      );

      if (!response.ok) throw new Error('Failed to load content');

      const data = await response.json();
      setSelectedCondition(data);
    } catch (error) {
      console.error('Error loading content:', error);
      Alert.alert('Error', 'Failed to load condition details');
    } finally {
      setIsLoadingContent(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      loadConditions();
      return;
    }

    try {
      setIsLoading(true);
      const response = await fetch(
        `${API_BASE_URL}/education/search?query=${encodeURIComponent(searchQuery)}&language=${selectedLanguage}`,
        { headers: await getAuthHeaders() }
      );

      if (!response.ok) throw new Error('Search failed');

      const data = await response.json();
      setConditions(data.results || []);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generatePDF = async () => {
    if (!selectedCondition) return;

    try {
      setIsGeneratingPDF(true);

      const formData = new FormData();
      formData.append('condition_id', selectedCondition.id);
      formData.append('patient_name', user?.username || 'Patient');
      formData.append('language', selectedLanguage);

      const token = await AsyncStorage.getItem('accessToken');
      const response = await fetch(`${API_BASE_URL}/education/generate-pdf`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) throw new Error('PDF generation failed');

      // Get the blob
      const blob = await response.blob();

      // Convert to base64 for mobile
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64data = reader.result as string;
        const base64 = base64data.split(',')[1];

        // Save to file system
        const fileName = `${selectedCondition.id}_education_${Date.now()}.pdf`;
        const fileUri = `${FileSystem.documentDirectory}${fileName}`;

        await FileSystem.writeAsStringAsync(fileUri, base64, {
          encoding: FileSystem.EncodingType.Base64,
        });

        // Share the file
        if (await Sharing.isAvailableAsync()) {
          await Sharing.shareAsync(fileUri, {
            mimeType: 'application/pdf',
            dialogTitle: 'Patient Education Handout',
          });
        } else {
          Alert.alert('Success', 'PDF saved to device');
        }
      };
      reader.readAsDataURL(blob);

    } catch (error) {
      console.error('PDF generation error:', error);
      Alert.alert('Error', 'Failed to generate PDF');
    } finally {
      setIsGeneratingPDF(false);
    }
  };

  const sendMaterial = async () => {
    if (!selectedCondition || !contactInfo.trim()) {
      Alert.alert('Error', 'Please enter contact information');
      return;
    }

    try {
      setIsSending(true);

      const formData = new FormData();
      formData.append('condition_id', selectedCondition.id);
      formData.append('patient_name', user?.username || 'Patient');
      formData.append('language', selectedLanguage);
      formData.append('delivery_method', sendMethod);
      formData.append('contact', contactInfo);

      const token = await AsyncStorage.getItem('accessToken');
      const response = await fetch(`${API_BASE_URL}/education/send`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to send');
      }

      Alert.alert('Success', `Educational material sent via ${sendMethod.toUpperCase()}`);
      setShowSendModal(false);
      setContactInfo('');

    } catch (error: any) {
      console.error('Send error:', error);
      Alert.alert('Error', error.message || 'Failed to send material');
    } finally {
      setIsSending(false);
    }
  };

  // Get unique categories
  const categories = [...new Set(conditions.map(c => c.category))].sort();

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'mild': return '#10b981';
      case 'moderate': return '#f59e0b';
      case 'severe': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const renderConditionList = () => (
    <View style={styles.listContainer}>
      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <TextInput
          style={styles.searchInput}
          placeholder="Search conditions..."
          placeholderTextColor="#9ca3af"
          value={searchQuery}
          onChangeText={setSearchQuery}
          onSubmitEditing={handleSearch}
        />
        <Pressable style={styles.searchButton} onPress={handleSearch}>
          <Text style={styles.searchButtonText}>Search</Text>
        </Pressable>
      </View>

      {/* Language Selector */}
      <Pressable
        style={styles.languageSelector}
        onPress={() => setShowLanguageModal(true)}
      >
        <Text style={styles.languageLabel}>Language:</Text>
        <Text style={styles.languageValue}>
          {LANGUAGE_OPTIONS.find(l => l.code === selectedLanguage)?.label || 'English'}
        </Text>
      </Pressable>

      {/* Category Filter */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.categoryScroll}>
        <Pressable
          style={[styles.categoryChip, !selectedCategory && styles.categoryChipActive]}
          onPress={() => setSelectedCategory(null)}
        >
          <Text style={[styles.categoryChipText, !selectedCategory && styles.categoryChipTextActive]}>
            All
          </Text>
        </Pressable>
        {categories.map(category => (
          <Pressable
            key={category}
            style={[styles.categoryChip, selectedCategory === category && styles.categoryChipActive]}
            onPress={() => setSelectedCategory(category === selectedCategory ? null : category)}
          >
            <Text style={[styles.categoryChipText, selectedCategory === category && styles.categoryChipTextActive]}>
              {category}
            </Text>
          </Pressable>
        ))}
      </ScrollView>

      {/* Conditions List */}
      {isLoading ? (
        <ActivityIndicator size="large" color="#2563eb" style={styles.loader} />
      ) : filteredConditions.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateText}>No conditions found</Text>
          <Text style={styles.emptyStateSubtext}>Try adjusting your search or filters</Text>
        </View>
      ) : (
        <ScrollView style={styles.conditionsList}>
          {filteredConditions.map(condition => (
            <Pressable
              key={condition.id}
              style={styles.conditionCard}
              onPress={() => loadConditionContent(condition.id)}
            >
              <View style={styles.conditionHeader}>
                <Text style={styles.conditionName}>{condition.name}</Text>
                <View style={[styles.severityBadge, { backgroundColor: getSeverityColor(condition.severity) }]}>
                  <Text style={styles.severityText}>{condition.severity}</Text>
                </View>
              </View>
              <Text style={styles.conditionCategory}>{condition.category}</Text>
            </Pressable>
          ))}
        </ScrollView>
      )}
    </View>
  );

  const renderConditionDetail = () => {
    if (!selectedCondition) return null;

    return (
      <ScrollView style={styles.detailContainer}>
        {/* Back Button */}
        <Pressable style={styles.backButton} onPress={() => setSelectedCondition(null)}>
          <Text style={styles.backButtonText}>Back to List</Text>
        </Pressable>

        {/* Condition Title */}
        <Text style={styles.detailTitle}>{selectedCondition.name}</Text>
        <Text style={styles.detailDescription}>{selectedCondition.description}</Text>

        {/* Action Buttons */}
        <View style={styles.actionButtons}>
          <Pressable
            style={[styles.actionButton, styles.pdfButton]}
            onPress={generatePDF}
            disabled={isGeneratingPDF}
          >
            {isGeneratingPDF ? (
              <ActivityIndicator color="#fff" size="small" />
            ) : (
              <Text style={styles.actionButtonText}>Download PDF</Text>
            )}
          </Pressable>

          <Pressable
            style={[styles.actionButton, styles.sendButton]}
            onPress={() => setShowSendModal(true)}
          >
            <Text style={styles.actionButtonText}>Send to Patient</Text>
          </Pressable>
        </View>

        {/* Symptoms */}
        {selectedCondition.symptoms?.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Symptoms</Text>
            {selectedCondition.symptoms.map((symptom, index) => (
              <View key={index} style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{symptom}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Causes */}
        {selectedCondition.causes?.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Causes</Text>
            {selectedCondition.causes.map((cause, index) => (
              <View key={index} style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{cause}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Care Instructions */}
        {selectedCondition.care_instructions?.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Care Instructions</Text>
            {selectedCondition.care_instructions.map((instruction, index) => (
              <View key={index} style={styles.numberedItem}>
                <Text style={styles.number}>{index + 1}.</Text>
                <Text style={styles.bulletText}>{instruction}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Treatment Options */}
        {selectedCondition.treatment_options?.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Treatment Options</Text>
            {selectedCondition.treatment_options.map((treatment, index) => (
              <View key={index} style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{treatment}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Warning Signs */}
        {selectedCondition.warning_signs?.length > 0 && (
          <View style={[styles.section, styles.warningSection]}>
            <Text style={[styles.sectionTitle, styles.warningTitle]}>Warning Signs</Text>
            {selectedCondition.warning_signs.map((sign, index) => (
              <View key={index} style={styles.bulletItem}>
                <Text style={[styles.bullet, styles.warningBullet]}>!</Text>
                <Text style={[styles.bulletText, styles.warningText]}>{sign}</Text>
              </View>
            ))}
          </View>
        )}

        {/* When to Seek Help */}
        {selectedCondition.when_to_seek_help?.length > 0 && (
          <View style={[styles.section, styles.helpSection]}>
            <Text style={[styles.sectionTitle, styles.helpTitle]}>When to Seek Medical Help</Text>
            {selectedCondition.when_to_seek_help.map((item, index) => (
              <View key={index} style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{item}</Text>
              </View>
            ))}
          </View>
        )}

        <View style={styles.bottomSpacer} />
      </ScrollView>
    );
  };

  return (
    <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable onPress={() => router.back()} style={styles.headerBack}>
          <Text style={styles.headerBackText}>Back</Text>
        </Pressable>
        <Text style={styles.headerTitle}>Patient Education</Text>
        <View style={styles.headerSpacer} />
      </View>

      {/* Content */}
      {isLoadingContent ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2563eb" />
          <Text style={styles.loadingText}>Loading content...</Text>
        </View>
      ) : selectedCondition ? (
        renderConditionDetail()
      ) : (
        renderConditionList()
      )}

      {/* Language Modal */}
      <Modal
        visible={showLanguageModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowLanguageModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Language</Text>
            {LANGUAGE_OPTIONS.map(lang => (
              <Pressable
                key={lang.code}
                style={[
                  styles.languageOption,
                  selectedLanguage === lang.code && styles.languageOptionActive
                ]}
                onPress={() => {
                  setSelectedLanguage(lang.code);
                  setShowLanguageModal(false);
                }}
              >
                <Text style={[
                  styles.languageOptionText,
                  selectedLanguage === lang.code && styles.languageOptionTextActive
                ]}>
                  {lang.label}
                </Text>
              </Pressable>
            ))}
            <Pressable style={styles.modalClose} onPress={() => setShowLanguageModal(false)}>
              <Text style={styles.modalCloseText}>Cancel</Text>
            </Pressable>
          </View>
        </View>
      </Modal>

      {/* Send Modal */}
      <Modal
        visible={showSendModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowSendModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Send Educational Material</Text>

            {/* Method Selection */}
            <View style={styles.methodSelector}>
              <Pressable
                style={[styles.methodButton, sendMethod === 'email' && styles.methodButtonActive]}
                onPress={() => setSendMethod('email')}
                disabled={!deliveryConfig?.email_configured}
              >
                <Text style={[
                  styles.methodButtonText,
                  sendMethod === 'email' && styles.methodButtonTextActive,
                  !deliveryConfig?.email_configured && styles.methodButtonDisabled
                ]}>
                  Email
                </Text>
              </Pressable>
              <Pressable
                style={[styles.methodButton, sendMethod === 'sms' && styles.methodButtonActive]}
                onPress={() => setSendMethod('sms')}
                disabled={!deliveryConfig?.sms_configured}
              >
                <Text style={[
                  styles.methodButtonText,
                  sendMethod === 'sms' && styles.methodButtonTextActive,
                  !deliveryConfig?.sms_configured && styles.methodButtonDisabled
                ]}>
                  SMS
                </Text>
              </Pressable>
            </View>

            {/* Contact Input */}
            <TextInput
              style={styles.contactInput}
              placeholder={sendMethod === 'email' ? 'Enter email address' : 'Enter phone number'}
              placeholderTextColor="#9ca3af"
              value={contactInfo}
              onChangeText={setContactInfo}
              keyboardType={sendMethod === 'email' ? 'email-address' : 'phone-pad'}
              autoCapitalize="none"
            />

            {/* Send Button */}
            <Pressable
              style={styles.sendModalButton}
              onPress={sendMaterial}
              disabled={isSending}
            >
              {isSending ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <Text style={styles.sendModalButtonText}>Send</Text>
              )}
            </Pressable>

            <Pressable style={styles.modalClose} onPress={() => setShowSendModal(false)}>
              <Text style={styles.modalCloseText}>Cancel</Text>
            </Pressable>
          </View>
        </View>
      </Modal>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 20,
    paddingBottom: 15,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  headerBack: {
    padding: 8,
  },
  headerBackText: {
    color: '#2563eb',
    fontSize: 16,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  headerSpacer: {
    width: 50,
  },
  listContainer: {
    flex: 1,
    padding: 16,
  },
  searchContainer: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  searchInput: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 10,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  searchButton: {
    marginLeft: 8,
    backgroundColor: '#2563eb',
    borderRadius: 10,
    paddingHorizontal: 16,
    justifyContent: 'center',
  },
  searchButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  languageSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  languageLabel: {
    color: '#6b7280',
    marginRight: 8,
  },
  languageValue: {
    color: '#2563eb',
    fontWeight: '600',
  },
  categoryScroll: {
    flexGrow: 0,
    marginBottom: 12,
  },
  categoryChip: {
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  categoryChipActive: {
    backgroundColor: '#2563eb',
    borderColor: '#2563eb',
  },
  categoryChipText: {
    color: '#6b7280',
    fontSize: 14,
  },
  categoryChipTextActive: {
    color: '#fff',
  },
  loader: {
    marginTop: 40,
  },
  emptyState: {
    alignItems: 'center',
    marginTop: 60,
  },
  emptyStateText: {
    fontSize: 18,
    color: '#6b7280',
    fontWeight: '600',
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#9ca3af',
    marginTop: 4,
  },
  conditionsList: {
    flex: 1,
  },
  conditionCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  conditionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  conditionName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e3a5f',
    flex: 1,
  },
  severityBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  severityText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  conditionCategory: {
    color: '#6b7280',
    fontSize: 14,
    marginTop: 4,
  },
  detailContainer: {
    flex: 1,
    padding: 16,
  },
  backButton: {
    marginBottom: 16,
  },
  backButtonText: {
    color: '#2563eb',
    fontSize: 16,
  },
  detailTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 8,
  },
  detailDescription: {
    fontSize: 16,
    color: '#4b5563',
    lineHeight: 24,
    marginBottom: 20,
  },
  actionButtons: {
    flexDirection: 'row',
    marginBottom: 24,
    gap: 12,
  },
  actionButton: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  pdfButton: {
    backgroundColor: '#2563eb',
  },
  sendButton: {
    backgroundColor: '#10b981',
  },
  actionButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
  },
  section: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 12,
  },
  bulletItem: {
    flexDirection: 'row',
    marginBottom: 8,
    paddingRight: 16,
  },
  bullet: {
    color: '#2563eb',
    fontSize: 16,
    marginRight: 8,
    fontWeight: '600',
  },
  bulletText: {
    flex: 1,
    fontSize: 15,
    color: '#4b5563',
    lineHeight: 22,
  },
  numberedItem: {
    flexDirection: 'row',
    marginBottom: 8,
    paddingRight: 16,
  },
  number: {
    color: '#2563eb',
    fontSize: 15,
    marginRight: 8,
    fontWeight: '600',
    minWidth: 20,
  },
  warningSection: {
    backgroundColor: '#fef2f2',
    borderColor: '#fecaca',
  },
  warningTitle: {
    color: '#dc2626',
  },
  warningBullet: {
    color: '#dc2626',
  },
  warningText: {
    color: '#7f1d1d',
  },
  helpSection: {
    backgroundColor: '#f0fdf4',
    borderColor: '#bbf7d0',
  },
  helpTitle: {
    color: '#166534',
  },
  bottomSpacer: {
    height: 40,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    color: '#6b7280',
    fontSize: 16,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
    marginBottom: 20,
    textAlign: 'center',
  },
  languageOption: {
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderRadius: 10,
    marginBottom: 8,
    backgroundColor: '#f3f4f6',
  },
  languageOptionActive: {
    backgroundColor: '#2563eb',
  },
  languageOptionText: {
    fontSize: 16,
    color: '#4b5563',
    textAlign: 'center',
  },
  languageOptionTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  modalClose: {
    marginTop: 12,
    paddingVertical: 12,
  },
  modalCloseText: {
    color: '#6b7280',
    fontSize: 16,
    textAlign: 'center',
  },
  methodSelector: {
    flexDirection: 'row',
    marginBottom: 16,
    gap: 12,
  },
  methodButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
  },
  methodButtonActive: {
    backgroundColor: '#2563eb',
  },
  methodButtonText: {
    fontSize: 16,
    color: '#4b5563',
    fontWeight: '600',
  },
  methodButtonTextActive: {
    color: '#fff',
  },
  methodButtonDisabled: {
    color: '#d1d5db',
  },
  contactInput: {
    backgroundColor: '#f3f4f6',
    borderRadius: 10,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  sendModalButton: {
    backgroundColor: '#10b981',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  sendModalButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
