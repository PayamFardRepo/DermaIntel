import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  ActivityIndicator,
  Alert,
  Image,
  TextInput,
  Switch,
  Platform,
  Dimensions
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as SecureStore from 'expo-secure-store';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { API_BASE_URL } from '../config';

const screenWidth = Dimensions.get('window').width;

interface AugmentationType {
  id: string;
  name: string;
  description: string;
  techniques: string[];
}

interface AugmentedImage {
  index: number;
  image_base64: string;
  parameters: any;
}

export default function DataAugmentationScreen() {
  const { isAuthenticated, user } = useAuth();
  const router = useRouter();

  // State
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [augmentationTypes, setAugmentationTypes] = useState<AugmentationType[]>([]);
  const [selectedTypes, setSelectedTypes] = useState<string[]>(['geometric', 'color', 'dermatology']);
  const [numAugmentations, setNumAugmentations] = useState<number>(5);
  const [augmentedImages, setAugmentedImages] = useState<AugmentedImage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isLoadingTypes, setIsLoadingTypes] = useState<boolean>(true);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [previewType, setPreviewType] = useState<string>('geometric');
  const [showPreviewModal, setShowPreviewModal] = useState<boolean>(false);
  const [rareConditions, setRareConditions] = useState<string[]>([]);
  const [currentConfig, setCurrentConfig] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'augment' | 'statistics' | 'settings'>('augment');

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    }
  }, [isAuthenticated]);

  // Load augmentation types and config
  useEffect(() => {
    loadAugmentationTypes();
    loadRareConditions();
    loadConfig();
  }, []);

  const loadAugmentationTypes = async () => {
    try {
      const token = await SecureStore.getItemAsync('auth_token');
      const response = await fetch(`${API_BASE_URL}/augmentation/types`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setAugmentationTypes(data.augmentation_types);
      }
    } catch (error) {
      console.error('Error loading augmentation types:', error);
    } finally {
      setIsLoadingTypes(false);
    }
  };

  const loadRareConditions = async () => {
    try {
      const token = await SecureStore.getItemAsync('auth_token');
      const response = await fetch(`${API_BASE_URL}/augmentation/rare-conditions`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setRareConditions(data.rare_conditions);
      }
    } catch (error) {
      console.error('Error loading rare conditions:', error);
    }
  };

  const loadConfig = async () => {
    try {
      const token = await SecureStore.getItemAsync('auth_token');
      const response = await fetch(`${API_BASE_URL}/augmentation/config`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setCurrentConfig(data);
      }
    } catch (error) {
      console.error('Error loading config:', error);
    }
  };

  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permissionResult.granted) {
      Alert.alert('Permission Required', 'Please allow access to your photo library.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
      base64: true
    });

    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0].uri);
      setAugmentedImages([]);
    }
  };

  const toggleAugmentationType = (typeId: string) => {
    setSelectedTypes(prev => {
      if (prev.includes(typeId)) {
        return prev.filter(t => t !== typeId);
      } else {
        return [...prev, typeId];
      }
    });
  };

  const generateAugmentations = async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select an image first.');
      return;
    }

    if (selectedTypes.length === 0) {
      Alert.alert('No Types Selected', 'Please select at least one augmentation type.');
      return;
    }

    setIsLoading(true);
    setAugmentedImages([]);

    try {
      const token = await SecureStore.getItemAsync('auth_token');

      // Create form data
      const formData = new FormData();

      // Fetch the image and add to form data
      const imageResponse = await fetch(selectedImage);
      const imageBlob = await imageResponse.blob();
      formData.append('file', {
        uri: selectedImage,
        type: 'image/jpeg',
        name: 'image.jpg'
      } as any);

      formData.append('augmentation_types', selectedTypes.join(','));
      formData.append('num_augmentations', numAugmentations.toString());

      const response = await fetch(`${API_BASE_URL}/augmentation/augment-image`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        setAugmentedImages(data.augmented_images);
      } else {
        const errorData = await response.json();
        Alert.alert('Error', errorData.detail || 'Failed to generate augmentations');
      }
    } catch (error: any) {
      console.error('Augmentation error:', error);
      Alert.alert('Error', error.message || 'Failed to generate augmentations');
    } finally {
      setIsLoading(false);
    }
  };

  const previewAugmentation = async (typeId: string) => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select an image first.');
      return;
    }

    setPreviewType(typeId);
    setIsLoading(true);

    try {
      const token = await SecureStore.getItemAsync('auth_token');

      const formData = new FormData();
      formData.append('file', {
        uri: selectedImage,
        type: 'image/jpeg',
        name: 'image.jpg'
      } as any);
      formData.append('augmentation_type', typeId);

      const response = await fetch(`${API_BASE_URL}/augmentation/preview`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        setPreviewImage(data.augmented_image);
        setShowPreviewModal(true);
      }
    } catch (error) {
      console.error('Preview error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const renderAugmentTab = () => (
    <View style={styles.tabContent}>
      {/* Image Selection */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>1. Select Image</Text>
        <Pressable style={styles.imagePickerButton} onPress={pickImage}>
          {selectedImage ? (
            <Image source={{ uri: selectedImage }} style={styles.selectedImagePreview} />
          ) : (
            <View style={styles.imagePlaceholder}>
              <Ionicons name="image-outline" size={48} color="#9ca3af" />
              <Text style={styles.imagePlaceholderText}>Tap to select an image</Text>
            </View>
          )}
        </Pressable>
      </View>

      {/* Augmentation Types */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>2. Select Augmentation Types</Text>
        {isLoadingTypes ? (
          <ActivityIndicator size="small" color="#667eea" />
        ) : (
          <View style={styles.typesGrid}>
            {augmentationTypes.map((type) => (
              <Pressable
                key={type.id}
                style={[
                  styles.typeCard,
                  selectedTypes.includes(type.id) && styles.typeCardSelected
                ]}
                onPress={() => toggleAugmentationType(type.id)}
                onLongPress={() => previewAugmentation(type.id)}
              >
                <View style={styles.typeCardHeader}>
                  <Ionicons
                    name={
                      type.id === 'geometric' ? 'resize' :
                      type.id === 'color' ? 'color-palette' :
                      type.id === 'noise' ? 'radio' :
                      type.id === 'advanced' ? 'construct' :
                      type.id === 'dermatology' ? 'medkit' :
                      'git-merge'
                    }
                    size={24}
                    color={selectedTypes.includes(type.id) ? '#667eea' : '#6b7280'}
                  />
                  {selectedTypes.includes(type.id) && (
                    <Ionicons name="checkmark-circle" size={20} color="#10b981" />
                  )}
                </View>
                <Text style={[
                  styles.typeCardTitle,
                  selectedTypes.includes(type.id) && styles.typeCardTitleSelected
                ]}>
                  {type.name}
                </Text>
                <Text style={styles.typeCardDescription} numberOfLines={2}>
                  {type.description}
                </Text>
                <Text style={styles.typeCardHint}>Long press to preview</Text>
              </Pressable>
            ))}
          </View>
        )}
      </View>

      {/* Number of Augmentations */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>3. Number of Augmentations</Text>
        <View style={styles.sliderContainer}>
          <Pressable
            style={styles.sliderButton}
            onPress={() => setNumAugmentations(Math.max(1, numAugmentations - 1))}
          >
            <Ionicons name="remove" size={24} color="#667eea" />
          </Pressable>
          <View style={styles.sliderValue}>
            <Text style={styles.sliderValueText}>{numAugmentations}</Text>
          </View>
          <Pressable
            style={styles.sliderButton}
            onPress={() => setNumAugmentations(Math.min(20, numAugmentations + 1))}
          >
            <Ionicons name="add" size={24} color="#667eea" />
          </Pressable>
        </View>
        <Text style={styles.sliderHint}>Generate 1-20 augmented versions</Text>
      </View>

      {/* Generate Button */}
      <Pressable
        style={[styles.generateButton, isLoading && styles.generateButtonDisabled]}
        onPress={generateAugmentations}
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="white" />
        ) : (
          <>
            <Ionicons name="sparkles" size={20} color="white" />
            <Text style={styles.generateButtonText}>Generate Augmentations</Text>
          </>
        )}
      </Pressable>

      {/* Results */}
      {augmentedImages.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Generated Images ({augmentedImages.length})</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <View style={styles.resultsRow}>
              {augmentedImages.map((img, index) => (
                <View key={index} style={styles.resultImageContainer}>
                  <Image
                    source={{ uri: `data:image/jpeg;base64,${img.image_base64}` }}
                    style={styles.resultImage}
                  />
                  <Text style={styles.resultImageLabel}>#{index + 1}</Text>
                </View>
              ))}
            </View>
          </ScrollView>
        </View>
      )}
    </View>
  );

  const renderStatisticsTab = () => (
    <View style={styles.tabContent}>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Rare Conditions Requiring Augmentation</Text>
        <Text style={styles.sectionDescription}>
          These conditions typically have fewer training samples and benefit most from data augmentation.
        </Text>
        <View style={styles.conditionsGrid}>
          {rareConditions.map((condition, index) => (
            <View key={index} style={styles.conditionBadge}>
              <Ionicons name="warning" size={14} color="#f59e0b" />
              <Text style={styles.conditionText}>
                {condition.replace(/_/g, ' ')}
              </Text>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Dataset Balancing</Text>
        <Text style={styles.sectionDescription}>
          Professional users can view dataset statistics and get recommendations for augmentation strategies.
        </Text>
        <View style={styles.proFeatureCard}>
          <Ionicons name="lock-closed" size={24} color="#9ca3af" />
          <Text style={styles.proFeatureText}>
            Full dataset analysis and batch augmentation are available for professional accounts.
          </Text>
        </View>
      </View>
    </View>
  );

  const renderSettingsTab = () => (
    <View style={styles.tabContent}>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Current Configuration</Text>
        {currentConfig ? (
          <View style={styles.configContainer}>
            {Object.entries(currentConfig).map(([category, settings]: [string, any]) => (
              <View key={category} style={styles.configCategory}>
                <Text style={styles.configCategoryTitle}>
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </Text>
                {Object.entries(settings).map(([key, value]: [string, any]) => (
                  <View key={key} style={styles.configRow}>
                    <Text style={styles.configKey}>{key.replace(/_/g, ' ')}</Text>
                    <Text style={styles.configValue}>
                      {Array.isArray(value) ? `[${value.join(', ')}]` : String(value)}
                    </Text>
                  </View>
                ))}
              </View>
            ))}
          </View>
        ) : (
          <ActivityIndicator size="small" color="#667eea" />
        )}
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#667eea', '#764ba2']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <Pressable onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </Pressable>
          <View style={styles.headerTitleContainer}>
            <Text style={styles.headerTitle}>Data Augmentation</Text>
            <Text style={styles.headerSubtitle}>Generate synthetic training data</Text>
          </View>
        </View>
      </LinearGradient>

      {/* Tab Bar */}
      <View style={styles.tabBar}>
        <Pressable
          style={[styles.tab, activeTab === 'augment' && styles.tabActive]}
          onPress={() => setActiveTab('augment')}
        >
          <Ionicons
            name="images"
            size={20}
            color={activeTab === 'augment' ? '#667eea' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'augment' && styles.tabTextActive]}>
            Augment
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tab, activeTab === 'statistics' && styles.tabActive]}
          onPress={() => setActiveTab('statistics')}
        >
          <Ionicons
            name="stats-chart"
            size={20}
            color={activeTab === 'statistics' ? '#667eea' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'statistics' && styles.tabTextActive]}>
            Statistics
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tab, activeTab === 'settings' && styles.tabActive]}
          onPress={() => setActiveTab('settings')}
        >
          <Ionicons
            name="settings"
            size={20}
            color={activeTab === 'settings' ? '#667eea' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === 'settings' && styles.tabTextActive]}>
            Settings
          </Text>
        </Pressable>
      </View>

      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {activeTab === 'augment' && renderAugmentTab()}
        {activeTab === 'statistics' && renderStatisticsTab()}
        {activeTab === 'settings' && renderSettingsTab()}
      </ScrollView>

      {/* Preview Modal */}
      {showPreviewModal && previewImage && (
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Preview: {previewType}</Text>
              <Pressable onPress={() => setShowPreviewModal(false)}>
                <Ionicons name="close" size={24} color="#374151" />
              </Pressable>
            </View>
            <View style={styles.previewComparison}>
              <View style={styles.previewImageContainer}>
                <Text style={styles.previewLabel}>Original</Text>
                {selectedImage && (
                  <Image source={{ uri: selectedImage }} style={styles.previewImage} />
                )}
              </View>
              <Ionicons name="arrow-forward" size={24} color="#9ca3af" />
              <View style={styles.previewImageContainer}>
                <Text style={styles.previewLabel}>Augmented</Text>
                <Image
                  source={{ uri: `data:image/jpeg;base64,${previewImage}` }}
                  style={styles.previewImage}
                />
              </View>
            </View>
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6',
  },
  header: {
    paddingTop: Platform.OS === 'ios' ? 50 : 30,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  backButton: {
    padding: 8,
    marginRight: 12,
  },
  headerTitleContainer: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: 'white',
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: 4,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: 'white',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
  },
  tabActive: {
    borderBottomWidth: 2,
    borderBottomColor: '#667eea',
  },
  tabText: {
    marginLeft: 8,
    fontSize: 14,
    fontWeight: '500',
    color: '#6b7280',
  },
  tabTextActive: {
    color: '#667eea',
  },
  scrollView: {
    flex: 1,
  },
  tabContent: {
    padding: 16,
  },
  section: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  sectionDescription: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 16,
    lineHeight: 20,
  },
  imagePickerButton: {
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: '#e5e7eb',
    borderStyle: 'dashed',
    borderRadius: 12,
    overflow: 'hidden',
    minHeight: 200,
  },
  selectedImagePreview: {
    width: '100%',
    height: 200,
    resizeMode: 'cover',
  },
  imagePlaceholder: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
  },
  imagePlaceholderText: {
    marginTop: 12,
    fontSize: 14,
    color: '#9ca3af',
  },
  typesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -6,
  },
  typeCard: {
    width: (screenWidth - 64) / 2,
    margin: 6,
    padding: 12,
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  typeCardSelected: {
    borderColor: '#667eea',
    backgroundColor: 'rgba(102, 126, 234, 0.05)',
  },
  typeCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  typeCardTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 4,
  },
  typeCardTitleSelected: {
    color: '#667eea',
  },
  typeCardDescription: {
    fontSize: 12,
    color: '#6b7280',
    lineHeight: 16,
  },
  typeCardHint: {
    fontSize: 10,
    color: '#9ca3af',
    marginTop: 8,
    fontStyle: 'italic',
  },
  sliderContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  sliderButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  sliderValue: {
    width: 80,
    alignItems: 'center',
    marginHorizontal: 16,
  },
  sliderValueText: {
    fontSize: 32,
    fontWeight: '700',
    color: '#667eea',
  },
  sliderHint: {
    fontSize: 12,
    color: '#9ca3af',
    textAlign: 'center',
    marginTop: 8,
  },
  generateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#667eea',
    paddingVertical: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  generateButtonDisabled: {
    opacity: 0.6,
  },
  generateButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  resultsRow: {
    flexDirection: 'row',
    paddingVertical: 8,
  },
  resultImageContainer: {
    marginRight: 12,
    alignItems: 'center',
  },
  resultImage: {
    width: 120,
    height: 120,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
  },
  resultImageLabel: {
    marginTop: 4,
    fontSize: 12,
    color: '#6b7280',
  },
  conditionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
  },
  conditionBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(245, 158, 11, 0.1)',
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 20,
    marginRight: 8,
    marginBottom: 8,
  },
  conditionText: {
    marginLeft: 4,
    fontSize: 12,
    color: '#92400e',
    textTransform: 'capitalize',
  },
  proFeatureCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    padding: 16,
    borderRadius: 12,
  },
  proFeatureText: {
    flex: 1,
    marginLeft: 12,
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
  },
  configContainer: {
    marginTop: 8,
  },
  configCategory: {
    marginBottom: 16,
  },
  configCategoryTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#667eea',
    marginBottom: 8,
    textTransform: 'capitalize',
  },
  configRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  configKey: {
    fontSize: 13,
    color: '#6b7280',
    textTransform: 'capitalize',
  },
  configValue: {
    fontSize: 13,
    color: '#1f2937',
    fontWeight: '500',
  },
  modalOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    width: '100%',
    maxWidth: 400,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    textTransform: 'capitalize',
  },
  previewComparison: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  previewImageContainer: {
    flex: 1,
    alignItems: 'center',
  },
  previewLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 8,
  },
  previewImage: {
    width: 140,
    height: 140,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
  },
});
