import React from 'react';
import { View, StyleSheet, Pressable, Text, ScrollView } from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';

/**
 * Dermoscopy Mode Information Screen
 * Explains dermoscopy features and how to use them
 */
export default function DermoscopyScreen() {
  const router = useRouter();
  const { t } = useTranslation();

  return (
    <View style={styles.container}>
      {/* Background */}
      <LinearGradient
        colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.background}
      />

      {/* Header with back button */}
      <View style={styles.header}>
        <Pressable
          style={styles.backButton}
          onPress={() => router.back()}
        >
          <Ionicons name="arrow-back" size={24} color="#2c5282" />
          <Text style={styles.backButtonText}>{t('common.back')}</Text>
        </Pressable>

        <Text style={styles.headerTitle}>{t('dermoscopy.title')}</Text>
        <Text style={styles.headerSubtitle}>{t('dermoscopy.subtitle')}</Text>
      </View>

      {/* Content */}
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {/* What is Dermoscopy */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="medical" size={28} color="#3b82f6" />
            <Text style={styles.cardTitle}>{t('dermoscopy.whatIsDermoscopy')}</Text>
          </View>

          <Text style={styles.cardText}>
            {t('dermoscopy.dermoscopyDescription')}
          </Text>

          <View style={styles.highlightBox}>
            <Ionicons name="trending-up" size={24} color="#10b981" />
            <Text style={styles.highlightText}>
              {t('dermoscopy.accuracyImprovement')}
            </Text>
          </View>
        </View>

        {/* How It Works */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="bulb" size={28} color="#f59e0b" />
            <Text style={styles.cardTitle}>{t('dermoscopy.howItWorksTitle')}</Text>
          </View>

          <View style={styles.stepsList}>
            <View style={styles.stepItem}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>1</Text>
              </View>
              <View style={styles.stepContent}>
                <Text style={styles.stepTitle}>{t('dermoscopy.magnificationTitle')}</Text>
                <Text style={styles.stepText}>
                  {t('dermoscopy.magnificationDescription')}
                </Text>
              </View>
            </View>

            <View style={styles.stepItem}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>2</Text>
              </View>
              <View style={styles.stepContent}>
                <Text style={styles.stepTitle}>{t('dermoscopy.illuminationTitle')}</Text>
                <Text style={styles.stepText}>
                  {t('dermoscopy.illuminationDescription')}
                </Text>
              </View>
            </View>

            <View style={styles.stepItem}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>3</Text>
              </View>
              <View style={styles.stepContent}>
                <Text style={styles.stepTitle}>{t('dermoscopy.patternAnalysisTitle')}</Text>
                <Text style={styles.stepText}>
                  {t('dermoscopy.patternAnalysisDescription')}
                </Text>
              </View>
            </View>
          </View>
        </View>

        {/* Dermoscopic Structures */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="scan" size={28} color="#8b5cf6" />
            <Text style={styles.cardTitle}>{t('dermoscopy.dermoscopicStructuresTitle')}</Text>
          </View>
          <Text style={styles.cardSubtitle}>
            {t('dermoscopy.structuresSubtitle')}
          </Text>

          <View style={styles.structuresList}>
            <View style={styles.structureItem}>
              <Ionicons name="grid-outline" size={24} color="#3b82f6" />
              <View style={styles.structureContent}>
                <Text style={styles.structureTitle}>{t('dermoscopy.pigmentNetworkTitle')}</Text>
                <Text style={styles.structureText}>
                  {t('dermoscopy.pigmentNetworkDescription')}
                </Text>
              </View>
            </View>

            <View style={styles.structureItem}>
              <Ionicons name="ellipse-outline" size={24} color="#3b82f6" />
              <View style={styles.structureContent}>
                <Text style={styles.structureTitle}>{t('dermoscopy.dotsGlobulesTitle')}</Text>
                <Text style={styles.structureText}>
                  {t('dermoscopy.dotsGlobulesDescription')}
                </Text>
              </View>
            </View>

            <View style={styles.structureItem}>
              <Ionicons name="water-outline" size={24} color="#3b82f6" />
              <View style={styles.structureContent}>
                <Text style={styles.structureTitle}>{t('dermoscopy.blueWhiteVeilTitle')}</Text>
                <Text style={styles.structureText}>
                  {t('dermoscopy.blueWhiteVeilDescription')}
                </Text>
              </View>
            </View>

            <View style={styles.structureItem}>
              <Ionicons name="trending-up-outline" size={24} color="#3b82f6" />
              <View style={styles.structureContent}>
                <Text style={styles.structureTitle}>{t('dermoscopy.radialStreaksTitle')}</Text>
                <Text style={styles.structureText}>
                  {t('dermoscopy.radialStreaksDescription')}
                </Text>
              </View>
            </View>

            <View style={styles.structureItem}>
              <Ionicons name="git-network-outline" size={24} color="#3b82f6" />
              <View style={styles.structureContent}>
                <Text style={styles.structureTitle}>{t('dermoscopy.vascularPatternsTitle')}</Text>
                <Text style={styles.structureText}>
                  {t('dermoscopy.vascularPatternsDescription')}
                </Text>
              </View>
            </View>

            <View style={styles.structureItem}>
              <Ionicons name="analytics-outline" size={24} color="#3b82f6" />
              <View style={styles.structureContent}>
                <Text style={styles.structureTitle}>{t('dermoscopy.regressionStructuresTitle')}</Text>
                <Text style={styles.structureText}>
                  {t('dermoscopy.regressionStructuresDescription')}
                </Text>
              </View>
            </View>
          </View>
        </View>

        {/* How to Capture */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="camera" size={28} color="#10b981" />
            <Text style={styles.cardTitle}>{t('dermoscopy.howToCaptureTitle')}</Text>
          </View>

          <View style={styles.tipsList}>
            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={20} color="#10b981" />
              <Text style={styles.tipText}>
                <Text style={styles.tipBold}>{t('dermoscopy.tip1Title')}</Text> {t('dermoscopy.tip1Description')}
              </Text>
            </View>

            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={20} color="#10b981" />
              <Text style={styles.tipText}>
                <Text style={styles.tipBold}>{t('dermoscopy.tip2Title')}</Text> {t('dermoscopy.tip2Description')}
              </Text>
            </View>

            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={20} color="#10b981" />
              <Text style={styles.tipText}>
                <Text style={styles.tipBold}>{t('dermoscopy.tip3Title')}</Text> {t('dermoscopy.tip3Description')}
              </Text>
            </View>

            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={20} color="#10b981" />
              <Text style={styles.tipText}>
                <Text style={styles.tipBold}>{t('dermoscopy.tip4Title')}</Text> {t('dermoscopy.tip4Description')}
              </Text>
            </View>

            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={20} color="#10b981" />
              <Text style={styles.tipText}>
                <Text style={styles.tipBold}>{t('dermoscopy.tip5Title')}</Text> {t('dermoscopy.tip5Description')}
              </Text>
            </View>
          </View>
        </View>

        {/* AI Analysis */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="cog" size={28} color="#ec4899" />
            <Text style={styles.cardTitle}>{t('dermoscopy.aiAnalysisTitle')}</Text>
          </View>

          <Text style={styles.cardText}>
            {t('dermoscopy.aiAnalysisDescription')}
          </Text>

          <View style={styles.featuresList}>
            <View style={styles.featureItem}>
              <View style={styles.featureIcon}>
                <Ionicons name="checkmark" size={16} color="#fff" />
              </View>
              <Text style={styles.featureText}>
                {t('dermoscopy.feature1')}
              </Text>
            </View>

            <View style={styles.featureItem}>
              <View style={styles.featureIcon}>
                <Ionicons name="checkmark" size={16} color="#fff" />
              </View>
              <Text style={styles.featureText}>
                {t('dermoscopy.feature2')}
              </Text>
            </View>

            <View style={styles.featureItem}>
              <View style={styles.featureIcon}>
                <Ionicons name="checkmark" size={16} color="#fff" />
              </View>
              <Text style={styles.featureText}>
                {t('dermoscopy.feature3')}
              </Text>
            </View>

            <View style={styles.featureItem}>
              <View style={styles.featureIcon}>
                <Ionicons name="checkmark" size={16} color="#fff" />
              </View>
              <Text style={styles.featureText}>
                {t('dermoscopy.feature4')}
              </Text>
            </View>

            <View style={styles.featureItem}>
              <View style={styles.featureIcon}>
                <Ionicons name="checkmark" size={16} color="#fff" />
              </View>
              <Text style={styles.featureText}>
                {t('dermoscopy.feature5')}
              </Text>
            </View>
          </View>
        </View>

        {/* Diagnostic Algorithms */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="apps" size={28} color="#6366f1" />
            <Text style={styles.cardTitle}>{t('dermoscopy.diagnosticAlgorithmsTitle')}</Text>
          </View>

          <View style={styles.algorithmsList}>
            <View style={styles.algorithmItem}>
              <Text style={styles.algorithmTitle}>{t('dermoscopy.sevenPointTitle')}</Text>
              <Text style={styles.algorithmText}>
                {t('dermoscopy.sevenPointDescription')}
              </Text>
            </View>

            <View style={styles.algorithmItem}>
              <Text style={styles.algorithmTitle}>{t('dermoscopy.abcdRuleTitle')}</Text>
              <Text style={styles.algorithmText}>
                {t('dermoscopy.abcdRuleDescription')}
              </Text>
            </View>

            <View style={styles.algorithmItem}>
              <Text style={styles.algorithmTitle}>{t('dermoscopy.menziesMethodTitle')}</Text>
              <Text style={styles.algorithmText}>
                {t('dermoscopy.menziesMethodDescription')}
              </Text>
            </View>
          </View>
        </View>

        {/* Limitations */}
        <View style={styles.warningCard}>
          <Ionicons name="warning" size={32} color="#f59e0b" />
          <Text style={styles.warningTitle}>{t('dermoscopy.limitationsTitle')}</Text>
          <View style={styles.warningList}>
            <Text style={styles.warningText}>
              {t('dermoscopy.limitation1')}
            </Text>
            <Text style={styles.warningText}>
              {t('dermoscopy.limitation2')}
            </Text>
            <Text style={styles.warningText}>
              {t('dermoscopy.limitation3')}
            </Text>
            <Text style={styles.warningText}>
              {t('dermoscopy.limitation4')}
            </Text>
            <Text style={styles.warningText}>
              {t('dermoscopy.limitation5')}
            </Text>
          </View>
        </View>

        {/* Get Started */}
        <View style={styles.getStartedCard}>
          <LinearGradient
            colors={['#3b82f6', '#2563eb']}
            style={styles.getStartedGradient}
          >
            <Ionicons name="rocket" size={48} color="#fff" />
            <Text style={styles.getStartedTitle}>{t('dermoscopy.getStartedTitle')}</Text>
            <Text style={styles.getStartedText}>
              {t('dermoscopy.getStartedDescription')}
            </Text>
            <Pressable
              style={styles.getStartedButton}
              onPress={() => router.push('/')}
            >
              <Text style={styles.getStartedButtonText}>{t('dermoscopy.goToAnalysis')}</Text>
              <Ionicons name="arrow-forward" size={20} color="#3b82f6" />
            </Pressable>
          </LinearGradient>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  background: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  header: {
    paddingTop: 50,
    paddingHorizontal: 16,
    paddingBottom: 16,
    backgroundColor: 'transparent',
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    alignSelf: 'flex-start',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    marginBottom: 16,
  },
  backButtonText: {
    marginLeft: 8,
    fontSize: 16,
    fontWeight: '600',
    color: '#2c5282',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: '#1f2937',
    marginLeft: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginLeft: 4,
    marginTop: 4,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 32,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 16,
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1f2937',
  },
  cardSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 16,
  },
  cardText: {
    fontSize: 15,
    color: '#4b5563',
    lineHeight: 22,
    marginBottom: 16,
  },
  highlightBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
    backgroundColor: '#ecfdf5',
    padding: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#10b981',
  },
  highlightText: {
    flex: 1,
    fontSize: 14,
    color: '#047857',
    lineHeight: 20,
    fontWeight: '500',
  },
  stepsList: {
    gap: 16,
  },
  stepItem: {
    flexDirection: 'row',
    gap: 16,
  },
  stepNumber: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#3b82f6',
    justifyContent: 'center',
    alignItems: 'center',
  },
  stepNumberText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#fff',
  },
  stepContent: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  stepText: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
  },
  structuresList: {
    gap: 16,
  },
  structureItem: {
    flexDirection: 'row',
    gap: 12,
  },
  structureContent: {
    flex: 1,
  },
  structureTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  structureText: {
    fontSize: 13,
    color: '#6b7280',
    lineHeight: 18,
  },
  tipsList: {
    gap: 12,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
  },
  tipText: {
    flex: 1,
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
  },
  tipBold: {
    fontWeight: '600',
    color: '#1f2937',
  },
  featuresList: {
    gap: 12,
    marginTop: 16,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
  },
  featureIcon: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#10b981',
    justifyContent: 'center',
    alignItems: 'center',
  },
  featureText: {
    flex: 1,
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
  },
  algorithmsList: {
    gap: 16,
  },
  algorithmItem: {
    backgroundColor: '#f9fafb',
    padding: 16,
    borderRadius: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#6366f1',
  },
  algorithmTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  algorithmText: {
    fontSize: 13,
    color: '#6b7280',
    lineHeight: 18,
  },
  warningCard: {
    backgroundColor: '#fef3c7',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#f59e0b',
  },
  warningTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#92400e',
    marginTop: 12,
    marginBottom: 16,
  },
  warningList: {
    alignSelf: 'stretch',
    gap: 8,
  },
  warningText: {
    fontSize: 14,
    color: '#92400e',
    lineHeight: 20,
  },
  getStartedCard: {
    borderRadius: 16,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 5,
  },
  getStartedGradient: {
    padding: 32,
    alignItems: 'center',
  },
  getStartedTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#fff',
    marginTop: 16,
    marginBottom: 12,
  },
  getStartedText: {
    fontSize: 15,
    color: '#fff',
    textAlign: 'center',
    lineHeight: 22,
    opacity: 0.95,
    marginBottom: 24,
  },
  getStartedButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#fff',
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 12,
  },
  getStartedButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#3b82f6',
  },
});
