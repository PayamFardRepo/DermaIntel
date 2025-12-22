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
  Image,
  Share,
  Clipboard,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../config';

interface Analysis {
  id: number;
  predicted_class: string;
  lesion_confidence: number;
  image_url?: string;
  created_at: string;
  shared_with_dermatologist: boolean;
  dermatologist_name?: string;
  dermatologist_email?: string;
  share_date?: string;
  share_message?: string;
  share_token?: string;
  dermatologist_reviewed: boolean;
  dermatologist_notes?: string;
  dermatologist_recommendation?: string;
  dermatologist_review_date?: string;
}

export default function SharedAnalysisScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'shared' | 'available'>('shared');
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Share modal
  const [showShareModal, setShowShareModal] = useState(false);
  const [selectedAnalysis, setSelectedAnalysis] = useState<Analysis | null>(null);
  const [dermatologistName, setDermatologistName] = useState('');
  const [dermatologistEmail, setDermatologistEmail] = useState('');
  const [shareMessage, setShareMessage] = useState('');
  const [sharing, setSharing] = useState(false);

  // Detail modal
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [detailAnalysis, setDetailAnalysis] = useState<Analysis | null>(null);

  // Share link modal
  const [showLinkModal, setShowLinkModal] = useState(false);
  const [shareLink, setShareLink] = useState('');

  useEffect(() => {
    fetchAnalyses();
  }, []);

  const fetchAnalyses = async () => {
    try {
      const token = await AsyncStorage.getItem('userToken');
      const response = await fetch(`${API_URL}/history`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setAnalyses(data.history || []);
      }
    } catch (error) {
      console.error('Error fetching analyses:', error);
      Alert.alert('Error', 'Failed to fetch analyses');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchAnalyses();
  }, []);

  const getFilteredAnalyses = () => {
    if (activeTab === 'shared') {
      return analyses.filter(a => a.shared_with_dermatologist);
    }
    return analyses.filter(a => !a.shared_with_dermatologist);
  };

  const openShareModal = (analysis: Analysis) => {
    setSelectedAnalysis(analysis);
    setDermatologistName('');
    setDermatologistEmail('');
    setShareMessage('');
    setShowShareModal(true);
  };

  const handleShare = async () => {
    if (!selectedAnalysis) return;

    if (!dermatologistName.trim() || !dermatologistEmail.trim()) {
      Alert.alert('Required Fields', 'Please enter dermatologist name and email');
      return;
    }

    if (!dermatologistEmail.includes('@')) {
      Alert.alert('Invalid Email', 'Please enter a valid email address');
      return;
    }

    setSharing(true);
    try {
      const token = await AsyncStorage.getItem('userToken');
      const formData = new FormData();
      formData.append('dermatologist_name', dermatologistName.trim());
      formData.append('dermatologist_email', dermatologistEmail.trim());
      if (shareMessage.trim()) {
        formData.append('share_message', shareMessage.trim());
      }

      const response = await fetch(
        `${API_URL}/analysis/share-with-dermatologist/${selectedAnalysis.id}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
          body: formData,
        }
      );

      if (response.ok) {
        const data = await response.json();
        setShowShareModal(false);

        // Show share link
        const fullLink = `${API_URL}${data.share_url}`;
        setShareLink(fullLink);
        setShowLinkModal(true);

        fetchAnalyses();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to share analysis');
      }
    } catch (error) {
      console.error('Error sharing analysis:', error);
      Alert.alert('Error', 'Failed to share analysis');
    } finally {
      setSharing(false);
    }
  };

  const copyShareLink = async () => {
    try {
      await Clipboard.setString(shareLink);
      Alert.alert('Copied', 'Share link copied to clipboard');
    } catch (error) {
      Alert.alert('Error', 'Failed to copy link');
    }
  };

  const shareViaSystem = async () => {
    try {
      await Share.share({
        message: `View my skin analysis: ${shareLink}`,
        url: shareLink,
      });
    } catch (error) {
      console.error('Error sharing:', error);
    }
  };

  const openDetailModal = (analysis: Analysis) => {
    setDetailAnalysis(analysis);
    setShowDetailModal(true);
  };

  const getStatusColor = (analysis: Analysis) => {
    if (analysis.dermatologist_reviewed) return '#4CAF50';
    if (analysis.shared_with_dermatologist) return '#FF9800';
    return '#9E9E9E';
  };

  const getStatusText = (analysis: Analysis) => {
    if (analysis.dermatologist_reviewed) return 'Reviewed';
    if (analysis.shared_with_dermatologist) return 'Pending Review';
    return 'Not Shared';
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return '';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const formatDateTime = (dateString: string) => {
    if (!dateString) return '';
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#4CAF50';
    if (confidence >= 0.5) return '#FF9800';
    return '#F44336';
  };

  const renderAnalysisCard = (analysis: Analysis) => (
    <TouchableOpacity
      key={analysis.id}
      style={styles.analysisCard}
      onPress={() => analysis.shared_with_dermatologist ? openDetailModal(analysis) : openShareModal(analysis)}
    >
      <View style={styles.cardHeader}>
        {analysis.image_url ? (
          <Image
            source={{ uri: `${API_URL}${analysis.image_url}` }}
            style={styles.thumbnail}
          />
        ) : (
          <View style={styles.thumbnailPlaceholder}>
            <Ionicons name="image-outline" size={24} color="#ccc" />
          </View>
        )}
        <View style={styles.cardInfo}>
          <Text style={styles.diagnosisText}>{analysis.predicted_class || 'Unknown'}</Text>
          <View style={styles.confidenceRow}>
            <View style={styles.confidenceBar}>
              <View
                style={[
                  styles.confidenceFill,
                  {
                    width: `${(analysis.lesion_confidence || 0) * 100}%`,
                    backgroundColor: getConfidenceColor(analysis.lesion_confidence || 0),
                  },
                ]}
              />
            </View>
            <Text style={styles.confidenceText}>
              {((analysis.lesion_confidence || 0) * 100).toFixed(0)}%
            </Text>
          </View>
          <Text style={styles.dateText}>{formatDate(analysis.created_at)}</Text>
        </View>
        <View style={[styles.statusBadge, { backgroundColor: getStatusColor(analysis) }]}>
          <Text style={styles.statusText}>{getStatusText(analysis)}</Text>
        </View>
      </View>

      {analysis.shared_with_dermatologist && (
        <View style={styles.shareInfo}>
          <View style={styles.shareRow}>
            <Ionicons name="person" size={14} color="#667eea" />
            <Text style={styles.shareLabel}>Shared with:</Text>
            <Text style={styles.shareValue}>{analysis.dermatologist_name}</Text>
          </View>
          {analysis.share_date && (
            <View style={styles.shareRow}>
              <Ionicons name="calendar" size={14} color="#999" />
              <Text style={styles.shareLabel}>Shared on:</Text>
              <Text style={styles.shareValue}>{formatDate(analysis.share_date)}</Text>
            </View>
          )}
          {analysis.dermatologist_reviewed && (
            <View style={styles.reviewedBadge}>
              <Ionicons name="checkmark-circle" size={16} color="#4CAF50" />
              <Text style={styles.reviewedText}>
                Reviewed on {formatDate(analysis.dermatologist_review_date || '')}
              </Text>
            </View>
          )}
        </View>
      )}

      {!analysis.shared_with_dermatologist && (
        <View style={styles.sharePrompt}>
          <Ionicons name="share-outline" size={16} color="#667eea" />
          <Text style={styles.sharePromptText}>Tap to share with a dermatologist</Text>
        </View>
      )}
    </TouchableOpacity>
  );

  const renderShareModal = () => (
    <Modal visible={showShareModal} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setShowShareModal(false)}>
            <Text style={styles.modalCancel}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Share Analysis</Text>
          <TouchableOpacity onPress={handleShare} disabled={sharing}>
            {sharing ? (
              <ActivityIndicator size="small" color="#667eea" />
            ) : (
              <Text style={styles.modalSave}>Share</Text>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.modalScroll}>
          {selectedAnalysis && (
            <View style={styles.analysisPreview}>
              {selectedAnalysis.image_url ? (
                <Image
                  source={{ uri: `${API_URL}${selectedAnalysis.image_url}` }}
                  style={styles.previewImage}
                />
              ) : (
                <View style={styles.previewPlaceholder}>
                  <Ionicons name="image-outline" size={48} color="#ccc" />
                </View>
              )}
              <View style={styles.previewInfo}>
                <Text style={styles.previewDiagnosis}>
                  {selectedAnalysis.predicted_class || 'Unknown'}
                </Text>
                <Text style={styles.previewDate}>
                  Analysis from {formatDate(selectedAnalysis.created_at)}
                </Text>
              </View>
            </View>
          )}

          <View style={styles.formSection}>
            <Text style={styles.formSectionTitle}>Dermatologist Information</Text>

            <Text style={styles.inputLabel}>Name *</Text>
            <TextInput
              style={styles.textInput}
              value={dermatologistName}
              onChangeText={setDermatologistName}
              placeholder="Dr. John Smith"
            />

            <Text style={styles.inputLabel}>Email *</Text>
            <TextInput
              style={styles.textInput}
              value={dermatologistEmail}
              onChangeText={setDermatologistEmail}
              placeholder="doctor@clinic.com"
              keyboardType="email-address"
              autoCapitalize="none"
            />
          </View>

          <View style={styles.formSection}>
            <Text style={styles.formSectionTitle}>Message (Optional)</Text>
            <TextInput
              style={[styles.textInput, styles.messageInput]}
              value={shareMessage}
              onChangeText={setShareMessage}
              placeholder="Add a message for the dermatologist..."
              multiline
              numberOfLines={4}
            />
          </View>

          <View style={styles.infoBox}>
            <Ionicons name="information-circle" size={20} color="#667eea" />
            <Text style={styles.infoText}>
              The dermatologist will receive a secure link to view this analysis.
              They can review the AI findings and provide their professional opinion.
            </Text>
          </View>
        </ScrollView>
      </View>
    </Modal>
  );

  const renderLinkModal = () => (
    <Modal visible={showLinkModal} animationType="fade" transparent>
      <View style={styles.linkModalOverlay}>
        <View style={styles.linkModalContent}>
          <View style={styles.linkModalHeader}>
            <Ionicons name="checkmark-circle" size={48} color="#4CAF50" />
            <Text style={styles.linkModalTitle}>Analysis Shared!</Text>
            <Text style={styles.linkModalSubtitle}>
              Share this link with the dermatologist
            </Text>
          </View>

          <View style={styles.linkBox}>
            <Text style={styles.linkText} numberOfLines={2}>{shareLink}</Text>
          </View>

          <View style={styles.linkActions}>
            <TouchableOpacity style={styles.copyButton} onPress={copyShareLink}>
              <Ionicons name="copy-outline" size={20} color="#fff" />
              <Text style={styles.copyButtonText}>Copy Link</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.shareButton} onPress={shareViaSystem}>
              <Ionicons name="share-outline" size={20} color="#667eea" />
              <Text style={styles.shareButtonText}>Share</Text>
            </TouchableOpacity>
          </View>

          <TouchableOpacity
            style={styles.doneButton}
            onPress={() => setShowLinkModal(false)}
          >
            <Text style={styles.doneButtonText}>Done</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  const renderDetailModal = () => (
    <Modal visible={showDetailModal} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setShowDetailModal(false)}>
            <Ionicons name="close" size={24} color="#333" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Shared Analysis</Text>
          <View style={{ width: 24 }} />
        </View>

        {detailAnalysis && (
          <ScrollView style={styles.modalScroll}>
            {/* Analysis Image */}
            <View style={styles.detailImageContainer}>
              {detailAnalysis.image_url ? (
                <Image
                  source={{ uri: `${API_URL}${detailAnalysis.image_url}` }}
                  style={styles.detailImage}
                  resizeMode="contain"
                />
              ) : (
                <View style={styles.detailImagePlaceholder}>
                  <Ionicons name="image-outline" size={64} color="#ccc" />
                </View>
              )}
            </View>

            {/* AI Analysis */}
            <View style={styles.detailSection}>
              <View style={styles.detailSectionHeader}>
                <Ionicons name="analytics" size={20} color="#667eea" />
                <Text style={styles.detailSectionTitle}>AI Analysis</Text>
              </View>
              <View style={styles.detailCard}>
                <Text style={styles.detailDiagnosis}>
                  {detailAnalysis.predicted_class || 'Unknown'}
                </Text>
                <View style={styles.detailConfidence}>
                  <Text style={styles.detailConfidenceLabel}>Confidence:</Text>
                  <View style={styles.detailConfidenceBar}>
                    <View
                      style={[
                        styles.detailConfidenceFill,
                        {
                          width: `${(detailAnalysis.lesion_confidence || 0) * 100}%`,
                          backgroundColor: getConfidenceColor(detailAnalysis.lesion_confidence || 0),
                        },
                      ]}
                    />
                  </View>
                  <Text style={styles.detailConfidenceValue}>
                    {((detailAnalysis.lesion_confidence || 0) * 100).toFixed(0)}%
                  </Text>
                </View>
                <Text style={styles.detailDate}>
                  Analyzed on {formatDateTime(detailAnalysis.created_at)}
                </Text>
              </View>
            </View>

            {/* Share Details */}
            <View style={styles.detailSection}>
              <View style={styles.detailSectionHeader}>
                <Ionicons name="share-social" size={20} color="#FF9800" />
                <Text style={styles.detailSectionTitle}>Share Details</Text>
              </View>
              <View style={styles.detailCard}>
                <View style={styles.detailRow}>
                  <Text style={styles.detailLabel}>Shared with</Text>
                  <Text style={styles.detailValue}>{detailAnalysis.dermatologist_name}</Text>
                </View>
                <View style={styles.detailRow}>
                  <Text style={styles.detailLabel}>Email</Text>
                  <Text style={styles.detailValue}>{detailAnalysis.dermatologist_email}</Text>
                </View>
                <View style={styles.detailRow}>
                  <Text style={styles.detailLabel}>Shared on</Text>
                  <Text style={styles.detailValue}>
                    {formatDateTime(detailAnalysis.share_date || '')}
                  </Text>
                </View>
                {detailAnalysis.share_message && (
                  <View style={styles.messageBox}>
                    <Text style={styles.messageLabel}>Your message:</Text>
                    <Text style={styles.messageContent}>{detailAnalysis.share_message}</Text>
                  </View>
                )}
              </View>
            </View>

            {/* Dermatologist Review */}
            {detailAnalysis.dermatologist_reviewed ? (
              <View style={styles.detailSection}>
                <View style={styles.detailSectionHeader}>
                  <Ionicons name="checkmark-circle" size={20} color="#4CAF50" />
                  <Text style={styles.detailSectionTitle}>Dermatologist Review</Text>
                </View>
                <View style={[styles.detailCard, styles.reviewCard]}>
                  <View style={styles.reviewHeader}>
                    <Ionicons name="medical" size={24} color="#4CAF50" />
                    <Text style={styles.reviewedOn}>
                      Reviewed on {formatDateTime(detailAnalysis.dermatologist_review_date || '')}
                    </Text>
                  </View>

                  {detailAnalysis.dermatologist_notes && (
                    <View style={styles.reviewSection}>
                      <Text style={styles.reviewSectionTitle}>Notes</Text>
                      <Text style={styles.reviewText}>{detailAnalysis.dermatologist_notes}</Text>
                    </View>
                  )}

                  {detailAnalysis.dermatologist_recommendation && (
                    <View style={styles.reviewSection}>
                      <Text style={styles.reviewSectionTitle}>Recommendations</Text>
                      <Text style={styles.reviewText}>
                        {detailAnalysis.dermatologist_recommendation}
                      </Text>
                    </View>
                  )}
                </View>
              </View>
            ) : (
              <View style={styles.pendingReview}>
                <Ionicons name="time-outline" size={32} color="#FF9800" />
                <Text style={styles.pendingTitle}>Awaiting Review</Text>
                <Text style={styles.pendingText}>
                  The dermatologist has not yet reviewed this analysis.
                  You will be notified when they provide their feedback.
                </Text>
              </View>
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
          <Text style={styles.loadingText}>Loading analyses...</Text>
        </View>
      </LinearGradient>
    );
  }

  const filteredAnalyses = getFilteredAnalyses();
  const sharedCount = analyses.filter(a => a.shared_with_dermatologist).length;
  const reviewedCount = analyses.filter(a => a.dermatologist_reviewed).length;

  return (
    <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Shared Analyses</Text>
        <View style={styles.headerRight} />
      </View>

      {/* Stats */}
      <View style={styles.statsContainer}>
        <View style={styles.statBox}>
          <Text style={styles.statValue}>{sharedCount}</Text>
          <Text style={styles.statLabel}>Shared</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statBox}>
          <Text style={styles.statValue}>{reviewedCount}</Text>
          <Text style={styles.statLabel}>Reviewed</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statBox}>
          <Text style={styles.statValue}>{sharedCount - reviewedCount}</Text>
          <Text style={styles.statLabel}>Pending</Text>
        </View>
      </View>

      {/* Tabs */}
      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'shared' && styles.activeTab]}
          onPress={() => setActiveTab('shared')}
        >
          <Ionicons
            name="share-social"
            size={20}
            color={activeTab === 'shared' ? '#667eea' : '#666'}
          />
          <Text style={[styles.tabText, activeTab === 'shared' && styles.activeTabText]}>
            Shared
          </Text>
          {sharedCount > 0 && (
            <View style={styles.tabBadge}>
              <Text style={styles.tabBadgeText}>{sharedCount}</Text>
            </View>
          )}
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'available' && styles.activeTab]}
          onPress={() => setActiveTab('available')}
        >
          <Ionicons
            name="images"
            size={20}
            color={activeTab === 'available' ? '#667eea' : '#666'}
          />
          <Text style={[styles.tabText, activeTab === 'available' && styles.activeTabText]}>
            Available to Share
          </Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#fff" />
        }
      >
        {filteredAnalyses.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons
              name={activeTab === 'shared' ? 'share-social-outline' : 'images-outline'}
              size={64}
              color="rgba(255,255,255,0.5)"
            />
            <Text style={styles.emptyText}>
              {activeTab === 'shared'
                ? 'No shared analyses yet'
                : 'No analyses available to share'}
            </Text>
            <Text style={styles.emptySubtext}>
              {activeTab === 'shared'
                ? 'Share your analyses with dermatologists for professional review'
                : 'Complete a skin analysis first to share it'}
            </Text>
          </View>
        ) : (
          filteredAnalyses.map(renderAnalysisCard)
        )}
      </ScrollView>

      {renderShareModal()}
      {renderLinkModal()}
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
  headerRight: {
    width: 40,
  },
  statsContainer: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.15)',
    marginHorizontal: 20,
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
  },
  statBox: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  statLabel: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 4,
  },
  statDivider: {
    width: 1,
    backgroundColor: 'rgba(255,255,255,0.3)',
    marginHorizontal: 10,
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
    fontSize: 13,
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
  analysisCard: {
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
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  thumbnail: {
    width: 60,
    height: 60,
    borderRadius: 8,
  },
  thumbnailPlaceholder: {
    width: 60,
    height: 60,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardInfo: {
    flex: 1,
    marginLeft: 12,
  },
  diagnosisText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  confidenceBar: {
    flex: 1,
    height: 6,
    backgroundColor: '#e0e0e0',
    borderRadius: 3,
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 3,
  },
  confidenceText: {
    fontSize: 12,
    color: '#666',
    width: 35,
  },
  dateText: {
    fontSize: 12,
    color: '#999',
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
  shareInfo: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  shareRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
    gap: 6,
  },
  shareLabel: {
    fontSize: 13,
    color: '#666',
  },
  shareValue: {
    fontSize: 13,
    color: '#333',
    fontWeight: '500',
  },
  reviewedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#E8F5E9',
    padding: 8,
    borderRadius: 6,
    marginTop: 8,
    gap: 6,
  },
  reviewedText: {
    fontSize: 13,
    color: '#4CAF50',
    fontWeight: '500',
  },
  sharePrompt: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#eee',
    gap: 6,
  },
  sharePromptText: {
    fontSize: 13,
    color: '#667eea',
  },
  emptyState: {
    alignItems: 'center',
    paddingTop: 60,
  },
  emptyText: {
    color: 'rgba(255,255,255,0.9)',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
  },
  emptySubtext: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
    paddingHorizontal: 40,
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
  analysisPreview: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    alignItems: 'center',
  },
  previewImage: {
    width: 80,
    height: 80,
    borderRadius: 8,
  },
  previewPlaceholder: {
    width: 80,
    height: 80,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  previewInfo: {
    flex: 1,
    marginLeft: 16,
  },
  previewDiagnosis: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  previewDate: {
    fontSize: 13,
    color: '#666',
    marginTop: 4,
  },
  formSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  formSectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  inputLabel: {
    fontSize: 13,
    color: '#666',
    marginBottom: 6,
    marginTop: 8,
  },
  textInput: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    fontSize: 15,
    color: '#333',
  },
  messageInput: {
    minHeight: 100,
    textAlignVertical: 'top',
  },
  infoBox: {
    flexDirection: 'row',
    backgroundColor: '#E8EAF6',
    borderRadius: 8,
    padding: 12,
    gap: 10,
    marginBottom: 30,
  },
  infoText: {
    flex: 1,
    fontSize: 13,
    color: '#667eea',
    lineHeight: 20,
  },
  // Link Modal
  linkModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    padding: 20,
  },
  linkModalContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
  },
  linkModalHeader: {
    alignItems: 'center',
    marginBottom: 20,
  },
  linkModalTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    marginTop: 12,
  },
  linkModalSubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  linkBox: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    width: '100%',
    marginBottom: 20,
  },
  linkText: {
    fontSize: 13,
    color: '#333',
    textAlign: 'center',
  },
  linkActions: {
    flexDirection: 'row',
    gap: 12,
    width: '100%',
    marginBottom: 16,
  },
  copyButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#667eea',
    padding: 14,
    borderRadius: 8,
    gap: 8,
  },
  copyButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  shareButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f0f0f0',
    padding: 14,
    borderRadius: 8,
    gap: 8,
  },
  shareButtonText: {
    color: '#667eea',
    fontWeight: '600',
  },
  doneButton: {
    width: '100%',
    padding: 14,
    alignItems: 'center',
  },
  doneButtonText: {
    color: '#666',
    fontSize: 16,
  },
  // Detail Modal
  detailImageContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    alignItems: 'center',
  },
  detailImage: {
    width: '100%',
    height: 200,
    borderRadius: 8,
  },
  detailImagePlaceholder: {
    width: '100%',
    height: 200,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  detailSection: {
    marginBottom: 16,
  },
  detailSectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  detailSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  detailCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
  },
  detailDiagnosis: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  detailConfidence: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  detailConfidenceLabel: {
    fontSize: 14,
    color: '#666',
    marginRight: 8,
  },
  detailConfidenceBar: {
    flex: 1,
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    marginRight: 8,
  },
  detailConfidenceFill: {
    height: '100%',
    borderRadius: 4,
  },
  detailConfidenceValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  detailDate: {
    fontSize: 13,
    color: '#999',
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  detailLabel: {
    fontSize: 14,
    color: '#666',
  },
  detailValue: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  messageBox: {
    marginTop: 12,
    padding: 12,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
  },
  messageLabel: {
    fontSize: 12,
    color: '#999',
    marginBottom: 4,
  },
  messageContent: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  reviewCard: {
    borderLeftWidth: 4,
    borderLeftColor: '#4CAF50',
  },
  reviewHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  reviewedOn: {
    fontSize: 13,
    color: '#666',
  },
  reviewSection: {
    marginBottom: 12,
  },
  reviewSectionTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#999',
    textTransform: 'uppercase',
    marginBottom: 4,
  },
  reviewText: {
    fontSize: 15,
    color: '#333',
    lineHeight: 22,
  },
  pendingReview: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 24,
    alignItems: 'center',
    marginBottom: 30,
  },
  pendingTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FF9800',
    marginTop: 12,
  },
  pendingText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
    lineHeight: 20,
  },
});
