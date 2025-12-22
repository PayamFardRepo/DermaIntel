import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Platform
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import AuthService from '../services/AuthService';
import { API_BASE_URL } from '../config';
import { useTranslation } from 'react-i18next';

interface FamilyMember {
  id: number;
  relationship_type: string;
  relationship_side?: string;
  name?: string;
  gender?: string;
  year_of_birth?: number;
  is_alive: boolean;
  has_skin_cancer: boolean;
  has_melanoma: boolean;
  melanoma_count: number;
  skin_cancer_count: number;
}

export default function FamilyHistoryScreen() {
  const { isAuthenticated, logout } = useAuth();
  const router = useRouter();
  const { t } = useTranslation();
  const [familyMembers, setFamilyMembers] = useState<FamilyMember[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/');
    } else {
      loadFamilyHistory();
    }
  }, [isAuthenticated]);

  const loadFamilyHistory = async () => {
    try {
      const token = AuthService.getToken();
      if (!token) {
        Alert.alert(t('familyHistory.common.authError'), t('familyHistory.common.loginAgain'));
        logout();
        return;
      }

      const response = await fetch(`${API_BASE_URL}/family-history`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setFamilyMembers(data.family_members);
      } else if (response.status === 401) {
        Alert.alert(t('familyHistory.common.sessionExpired'), t('familyHistory.common.loginAgain'));
        logout();
      } else {
        Alert.alert(t('familyHistory.common.error'), t('familyHistory.common.loadFailed'));
      }
    } catch (error) {
      console.error('Error loading family history:', error);
      Alert.alert(t('familyHistory.common.error'), t('familyHistory.common.networkError'));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const deleteFamilyMember = async (memberId: number) => {
    Alert.alert(
      t('familyHistory.card.deleteTitle'),
      t('familyHistory.card.deleteConfirm'),
      [
        { text: t('familyHistory.common.cancel'), style: 'cancel' },
        {
          text: t('familyHistory.card.delete'),
          style: 'destructive',
          onPress: async () => {
            try {
              const token = AuthService.getToken();
              const response = await fetch(`${API_BASE_URL}/family-history/${memberId}`, {
                method: 'DELETE',
                headers: {
                  'Authorization': `Bearer ${token}`
                }
              });

              if (response.ok) {
                Alert.alert(t('familyHistory.common.success'), t('familyHistory.card.deleteSuccess'));
                loadFamilyHistory();
              } else {
                Alert.alert(t('familyHistory.common.error'), t('familyHistory.card.deleteFailed'));
              }
            } catch (error) {
              Alert.alert(t('familyHistory.common.error'), t('familyHistory.common.networkError'));
            }
          }
        }
      ]
    );
  };

  const getRelationshipLabel = (member: FamilyMember) => {
    let label = member.relationship_type.replace('_', ' ');
    if (member.relationship_side) {
      label += ` (${member.relationship_side})`;
    }
    return label.charAt(0).toUpperCase() + label.slice(1);
  };

  const getRiskBadgeColor = (member: FamilyMember) => {
    if (member.has_melanoma) return '#dc2626';
    if (member.has_skin_cancer) return '#f59e0b';
    return '#10b981';
  };

  const renderFamilyMemberCard = (member: FamilyMember) => {
    return (
      <View key={member.id} style={styles.memberCard}>
        <View style={styles.cardHeader}>
          <View style={styles.headerLeft}>
            <Ionicons
              name="person"
              size={24}
              color="#4b5563"
              style={styles.personIcon}
            />
            <View>
              <Text style={styles.memberName}>
                {member.name || getRelationshipLabel(member)}
              </Text>
              {member.name && (
                <Text style={styles.relationshipText}>
                  {getRelationshipLabel(member)}
                </Text>
              )}
            </View>
          </View>
          <View style={[styles.riskBadge, { backgroundColor: getRiskBadgeColor(member) }]}>
            <Text style={styles.riskText}>
              {member.has_melanoma ? t('familyHistory.card.melanoma') : member.has_skin_cancer ? t('familyHistory.card.skinCancer') : t('familyHistory.card.healthy')}
            </Text>
          </View>
        </View>

        {member.year_of_birth && (
          <Text style={styles.detailText}>
            {t('familyHistory.card.born', { year: member.year_of_birth })}
            {!member.is_alive && ` (${t('familyHistory.card.deceased')})`}
          </Text>
        )}

        {member.has_skin_cancer && (
          <View style={styles.cancerInfo}>
            <Ionicons name="warning" size={16} color="#f59e0b" />
            <Text style={styles.cancerText}>
              {member.skin_cancer_count} skin cancer{member.skin_cancer_count !== 1 ? 's' : ''}
              {member.has_melanoma && ` • ${member.melanoma_count} melanoma${member.melanoma_count !== 1 ? 's' : ''}`}
            </Text>
          </View>
        )}

        <View style={styles.cardActions}>
          <TouchableOpacity
            style={styles.editButton}
            onPress={() => router.push(`/edit-family-member?id=${member.id}` as any)}
          >
            <Ionicons name="create-outline" size={20} color="#3b82f6" />
            <Text style={styles.editButtonText}>{t('familyHistory.card.edit')}</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.deleteButton}
            onPress={() => deleteFamilyMember(member.id)}
          >
            <Ionicons name="trash-outline" size={20} color="#dc2626" />
            <Text style={styles.deleteButtonText}>{t('familyHistory.card.delete')}</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#3b82f6" />
        <Text style={styles.loadingText}>{t('familyHistory.loading')}</Text>
      </View>
    );
  }

  return (
    <LinearGradient
      colors={['#1e3a8a', '#3b82f6', '#60a5fa']}
      style={styles.container}
    >
      <ScrollView style={styles.scrollView}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => router.back()}
            style={styles.backButton}
          >
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.title}>{t('familyHistory.title')}</Text>
          <TouchableOpacity
            onPress={() => router.push('/genetic-risk' as any)}
            style={styles.riskButton}
          >
            <Ionicons name="analytics" size={24} color="white" />
          </TouchableOpacity>
        </View>

        {/* Info Card */}
        <View style={styles.infoCard}>
          <Ionicons name="information-circle" size={24} color="#3b82f6" />
          <Text style={styles.infoText}>
            {t('familyHistory.infoText')}
          </Text>
        </View>

        {/* Stats Summary */}
        {familyMembers.length > 0 && (
          <View style={styles.statsCard}>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{familyMembers.length}</Text>
              <Text style={styles.statLabel}>{t('familyHistory.stats.familyMembers')}</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>
                {familyMembers.filter(m => m.has_skin_cancer).length}
              </Text>
              <Text style={styles.statLabel}>{t('familyHistory.stats.withSkinCancer')}</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>
                {familyMembers.filter(m => m.has_melanoma).length}
              </Text>
              <Text style={styles.statLabel}>{t('familyHistory.stats.withMelanoma')}</Text>
            </View>
          </View>
        )}

        {/* Family Members List */}
        <View style={styles.membersSection}>
          {familyMembers.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="people-outline" size={64} color="rgba(255,255,255,0.5)" />
              <Text style={styles.emptyTitle}>{t('familyHistory.emptyState.title')}</Text>
              <Text style={styles.emptyText}>
                {t('familyHistory.emptyState.subtitle')}
              </Text>
            </View>
          ) : (
            familyMembers.map(renderFamilyMemberCard)
          )}
        </View>

        {/* Add Button */}
        <TouchableOpacity
          style={styles.addButton}
          onPress={() => router.push('/add-family-member' as any)}
        >
          <Ionicons name="add-circle" size={24} color="white" />
          <Text style={styles.addButtonText}>{t('familyHistory.buttons.addFamilyMember')}</Text>
        </TouchableOpacity>

        {/* View Risk Profile Button */}
        {familyMembers.length > 0 && (
          <TouchableOpacity
            style={styles.riskProfileButton}
            onPress={() => router.push('/genetic-risk' as any)}
          >
            <Ionicons name="analytics" size={24} color="white" />
            <Text style={styles.riskProfileButtonText}>{t('familyHistory.buttons.viewGeneticRisk')}</Text>
          </TouchableOpacity>
        )}
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1e3a8a',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: 'white',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 16,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.2)',
  },
  backButton: {
    padding: 8,
  },
  riskButton: {
    padding: 8,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    flex: 1,
    textAlign: 'center',
  },
  infoCard: {
    margin: 20,
    padding: 16,
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'flex-start',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  infoText: {
    flex: 1,
    marginLeft: 12,
    fontSize: 14,
    color: '#1e40af',
    lineHeight: 20,
  },
  statsCard: {
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 20,
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 12,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
  },
  statNumber: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#1e3a8a',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
    textAlign: 'center',
  },
  statDivider: {
    width: 1,
    height: 40,
    backgroundColor: '#e5e7eb',
  },
  membersSection: {
    paddingHorizontal: 20,
  },
  memberCard: {
    backgroundColor: 'rgba(255,255,255,0.95)',
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
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  personIcon: {
    marginRight: 12,
  },
  memberName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  relationshipText: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 2,
  },
  riskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  riskText: {
    color: 'white',
    fontSize: 11,
    fontWeight: 'bold',
  },
  detailText: {
    fontSize: 14,
    color: '#4b5563',
    marginBottom: 8,
  },
  cancerInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    backgroundColor: '#fef3c7',
    borderRadius: 6,
    marginBottom: 12,
  },
  cancerText: {
    marginLeft: 8,
    fontSize: 13,
    color: '#92400e',
    fontWeight: '500',
  },
  cardActions: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  editButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 10,
    backgroundColor: '#eff6ff',
    borderRadius: 8,
    gap: 6,
  },
  editButtonText: {
    color: '#3b82f6',
    fontSize: 14,
    fontWeight: '600',
  },
  deleteButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 10,
    backgroundColor: '#fee2e2',
    borderRadius: 8,
    gap: 6,
  },
  deleteButtonText: {
    color: '#dc2626',
    fontSize: 14,
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
    paddingHorizontal: 32,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyText: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.8)',
    textAlign: 'center',
    lineHeight: 20,
  },
  addButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    margin: 20,
    padding: 16,
    backgroundColor: '#10b981',
    borderRadius: 12,
    gap: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 5,
  },
  addButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  riskProfileButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 16,
    backgroundColor: '#8b5cf6',
    borderRadius: 12,
    gap: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 5,
  },
  riskProfileButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
