import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  Linking,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useRouter } from 'expo-router';
import { useTranslation } from 'react-i18next';

interface Community {
  name: string;
  platform: 'inspire' | 'healthunlocked' | 'other';
  url: string;
  members?: string;
  description: string;
}

interface CommunityCategory {
  condition: string;
  icon: string;
  description: string;
  communities: Community[];
}

const COMMUNITY_CATEGORIES: CommunityCategory[] = [
  {
    condition: 'Melanoma & Skin Cancer',
    icon: '🎗️',
    description: 'Connect with survivors, patients, and caregivers',
    communities: [
      {
        name: 'Melanoma Exchange',
        platform: 'inspire',
        url: 'https://www.inspire.com/groups/melanoma-exchange/',
        members: '15,000+',
        description: 'Support for melanoma patients and families',
      },
      {
        name: 'Melanoma Research Foundation Forum',
        platform: 'other',
        url: 'https://forum.melanoma.org/',
        description: 'Official MRF patient forum',
      },
      {
        name: 'Skin Cancer Community',
        platform: 'healthunlocked',
        url: 'https://healthunlocked.com/melanoma-skin',
        description: 'Melanoma & skin cancer discussions',
      },
    ],
  },
  {
    condition: 'Eczema & Atopic Dermatitis',
    icon: '🧴',
    description: 'Parents, adults, and caregivers managing eczema',
    communities: [
      {
        name: 'National Eczema Association',
        platform: 'inspire',
        url: 'https://www.inspire.com/groups/national-eczema-association/',
        members: '25,000+',
        description: 'Largest eczema community online',
      },
      {
        name: 'MY SKIN Community',
        platform: 'healthunlocked',
        url: 'https://healthunlocked.com/my-skin',
        description: 'General skin conditions including eczema',
      },
    ],
  },
  {
    condition: 'Psoriasis',
    icon: '🔬',
    description: 'Living with and managing psoriasis',
    communities: [
      {
        name: 'Psoriasis Community',
        platform: 'inspire',
        url: 'https://www.inspire.com/groups/psoriasis-community/',
        members: '40,000+',
        description: 'One of the largest psoriasis communities',
      },
      {
        name: 'MY SKIN Community',
        platform: 'healthunlocked',
        url: 'https://healthunlocked.com/my-skin',
        description: 'Psoriasis discussions and support',
      },
    ],
  },
  {
    condition: 'Lupus & Autoimmune',
    icon: '🦋',
    description: 'Autoimmune conditions affecting the skin',
    communities: [
      {
        name: 'Lupus Connect',
        platform: 'inspire',
        url: 'https://www.inspire.com/groups/lupusconnect/',
        members: '50,000+',
        description: 'Support for lupus patients and families',
      },
      {
        name: 'LUPUS UK',
        platform: 'healthunlocked',
        url: 'https://healthunlocked.com/lupusuk',
        members: '20,000+',
        description: 'UK lupus community with expert input',
      },
    ],
  },
  {
    condition: 'Acne & Rosacea',
    icon: '✨',
    description: 'Managing acne and rosacea at any age',
    communities: [
      {
        name: 'Autoimmune Diseases (Rosacea)',
        platform: 'inspire',
        url: 'https://www.inspire.com/groups/autoimmune-disease/',
        members: '12,000+',
        description: 'Rosacea discussions in autoimmune community',
      },
      {
        name: 'MY SKIN Community',
        platform: 'healthunlocked',
        url: 'https://healthunlocked.com/my-skin',
        description: 'Acne and rosacea support',
      },
    ],
  },
  {
    condition: 'Vitiligo',
    icon: '🤍',
    description: 'Embracing and treating vitiligo',
    communities: [
      {
        name: 'Vitiligo Support International',
        platform: 'other',
        url: 'https://vitiligosupport.org/',
        description: 'Global vitiligo support network',
      },
      {
        name: 'The Vitiligo Society',
        platform: 'other',
        url: 'https://vitiligosociety.org/',
        description: 'UK-based vitiligo charity since 1985',
      },
    ],
  },
  {
    condition: 'Burns & Wound Care',
    icon: '🩹',
    description: 'Burn survivors and wound care support',
    communities: [
      {
        name: 'Phoenix Society for Burn Survivors',
        platform: 'other',
        url: 'https://www.phoenix-society.org/',
        members: '8,000+',
        description: 'Peer support for burn survivors',
      },
      {
        name: 'Wound Healing Community',
        platform: 'inspire',
        url: 'https://www.inspire.com/groups/wound-healing/',
        description: 'Wound care discussions and support',
      },
    ],
  },
  {
    condition: 'General Skin Health',
    icon: '🧬',
    description: 'General dermatology discussions',
    communities: [
      {
        name: 'MY SKIN Community',
        platform: 'healthunlocked',
        url: 'https://healthunlocked.com/my-skin',
        members: '10,000+',
        description: 'All skin conditions welcome',
      },
      {
        name: 'Autoimmune Diseases',
        platform: 'inspire',
        url: 'https://www.inspire.com/groups/autoimmune-disease/',
        description: 'Skin-related autoimmune conditions',
      },
    ],
  },
];

export default function PatientCommunitiesScreen() {
  const { t } = useTranslation();
  const router = useRouter();
  const [expandedCategory, setExpandedCategory] = useState<string | null>('Melanoma & Skin Cancer');

  const openCommunity = async (url: string, name: string) => {
    try {
      const supported = await Linking.canOpenURL(url);
      if (supported) {
        await Linking.openURL(url);
      } else {
        Alert.alert('Error', `Cannot open ${name}. Please visit ${url} in your browser.`);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to open community link');
    }
  };

  const getPlatformInfo = (platform: string) => {
    switch (platform) {
      case 'inspire':
        return { color: '#4A90D9', label: 'Inspire' };
      case 'healthunlocked':
        return { color: '#00A99D', label: 'HealthUnlocked' };
      default:
        return { color: '#6B7280', label: 'External' };
    }
  };

  const toggleCategory = (condition: string) => {
    setExpandedCategory(expandedCategory === condition ? null : condition);
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#667eea', '#764ba2']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <Pressable onPress={() => router.back()} style={styles.backButton}>
            <Text style={styles.backButtonText}>{'<'} Back</Text>
          </Pressable>
          <Text style={styles.headerTitle}>Patient Communities</Text>
          <Text style={styles.headerSubtitle}>
            Connect with others who understand your journey
          </Text>
        </View>
      </LinearGradient>

      <ScrollView style={styles.content}>
        {/* Platform Info Cards */}
        <View style={styles.platformsSection}>
          <Text style={styles.sectionTitle}>Trusted Platforms</Text>
          <View style={styles.platformCards}>
            <Pressable
              style={[styles.platformCard, { borderLeftColor: '#4A90D9' }]}
              onPress={() => openCommunity('https://www.inspire.com/', 'Inspire')}
            >
              <Text style={styles.platformName}>Inspire</Text>
              <Text style={styles.platformDescription}>
                2M+ members across 250+ health communities. US-based, patient-to-patient support.
              </Text>
              <Text style={[styles.platformLink, { color: '#4A90D9' }]}>Visit inspire.com →</Text>
            </Pressable>

            <Pressable
              style={[styles.platformCard, { borderLeftColor: '#00A99D' }]}
              onPress={() => openCommunity('https://healthunlocked.com/', 'HealthUnlocked')}
            >
              <Text style={styles.platformName}>HealthUnlocked</Text>
              <Text style={styles.platformDescription}>
                NHS-partnered platform with 1M+ members. Moderated communities with health org backing.
              </Text>
              <Text style={[styles.platformLink, { color: '#00A99D' }]}>Visit healthunlocked.com →</Text>
            </Pressable>
          </View>
        </View>

        {/* Disclaimer */}
        <View style={styles.disclaimerCard}>
          <Text style={styles.disclaimerIcon}>ℹ️</Text>
          <Text style={styles.disclaimerText}>
            These are independent community platforms. Information shared is peer support, not medical advice.
            Always consult your healthcare provider for medical decisions.
          </Text>
        </View>

        {/* Condition Categories */}
        <Text style={styles.sectionTitle}>Find Your Community</Text>

        {COMMUNITY_CATEGORIES.map((category) => (
          <View key={category.condition} style={styles.categoryContainer}>
            <Pressable
              style={styles.categoryHeader}
              onPress={() => toggleCategory(category.condition)}
            >
              <Text style={styles.categoryIcon}>{category.icon}</Text>
              <View style={styles.categoryInfo}>
                <Text style={styles.categoryTitle}>{category.condition}</Text>
                <Text style={styles.categoryDescription}>{category.description}</Text>
              </View>
              <Text style={styles.expandIcon}>
                {expandedCategory === category.condition ? '▼' : '▶'}
              </Text>
            </Pressable>

            {expandedCategory === category.condition && (
              <View style={styles.communitiesList}>
                {category.communities.map((community, index) => {
                  const platformInfo = getPlatformInfo(community.platform);
                  return (
                    <Pressable
                      key={index}
                      style={styles.communityCard}
                      onPress={() => openCommunity(community.url, community.name)}
                    >
                      <View style={styles.communityHeader}>
                        <Text style={styles.communityName}>{community.name}</Text>
                        <View style={[styles.platformBadge, { backgroundColor: platformInfo.color }]}>
                          <Text style={styles.platformBadgeText}>{platformInfo.label}</Text>
                        </View>
                      </View>
                      <Text style={styles.communityDescription}>{community.description}</Text>
                      {community.members && (
                        <Text style={styles.communityMembers}>
                          👥 {community.members} members
                        </Text>
                      )}
                      <Text style={styles.joinLink}>Join Community →</Text>
                    </Pressable>
                  );
                })}
              </View>
            )}
          </View>
        ))}

        {/* Tips Section */}
        <View style={styles.tipsSection}>
          <Text style={styles.tipsTitle}>💡 Getting the Most from Communities</Text>
          <View style={styles.tipsList}>
            <Text style={styles.tipItem}>• Introduce yourself and share your journey</Text>
            <Text style={styles.tipItem}>• Ask questions - no question is too small</Text>
            <Text style={styles.tipItem}>• Support others when you can</Text>
            <Text style={styles.tipItem}>• Respect privacy - don't share others' info</Text>
            <Text style={styles.tipItem}>• Report any concerning content to moderators</Text>
          </View>
        </View>

        <View style={styles.bottomPadding} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    paddingTop: 50,
    paddingBottom: 25,
    paddingHorizontal: 20,
  },
  headerContent: {
    alignItems: 'center',
  },
  backButton: {
    position: 'absolute',
    left: 0,
    top: 0,
  },
  backButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 10,
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.9)',
    marginTop: 5,
    textAlign: 'center',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  platformsSection: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 12,
    marginTop: 8,
  },
  platformCards: {
    gap: 12,
  },
  platformCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  platformName: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 6,
  },
  platformDescription: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
    marginBottom: 8,
  },
  platformLink: {
    fontSize: 14,
    fontWeight: '600',
  },
  disclaimerCard: {
    backgroundColor: '#fef3c7',
    borderRadius: 12,
    padding: 14,
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#fcd34d',
  },
  disclaimerIcon: {
    fontSize: 18,
    marginRight: 10,
  },
  disclaimerText: {
    flex: 1,
    fontSize: 13,
    color: '#92400e',
    lineHeight: 18,
  },
  categoryContainer: {
    marginBottom: 12,
    backgroundColor: '#fff',
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  categoryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
  },
  categoryIcon: {
    fontSize: 28,
    marginRight: 14,
  },
  categoryInfo: {
    flex: 1,
  },
  categoryTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  categoryDescription: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  expandIcon: {
    fontSize: 12,
    color: '#9ca3af',
  },
  communitiesList: {
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
    padding: 12,
    backgroundColor: '#f9fafb',
    gap: 10,
  },
  communityCard: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 14,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  communityHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 6,
  },
  communityName: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937',
    flex: 1,
    marginRight: 8,
  },
  platformBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  platformBadgeText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
  },
  communityDescription: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 6,
  },
  communityMembers: {
    fontSize: 12,
    color: '#9ca3af',
    marginBottom: 8,
  },
  joinLink: {
    fontSize: 14,
    fontWeight: '600',
    color: '#667eea',
  },
  tipsSection: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    marginTop: 8,
    borderWidth: 1,
    borderColor: '#bfdbfe',
  },
  tipsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e40af',
    marginBottom: 10,
  },
  tipsList: {
    gap: 6,
  },
  tipItem: {
    fontSize: 14,
    color: '#1e3a8a',
    lineHeight: 20,
  },
  bottomPadding: {
    height: 40,
  },
});
