import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, Pressable } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import { HelpTooltip } from './HelpTooltip';

/**
 * Regulatory Compliance Component
 * Displays FDA and CE Mark approval pathway information
 */
export default function RegulatoryCompliance() {
  const { t } = useTranslation();
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={true}>
      {/* Header */}
      <View style={styles.header}>
        <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 8 }}>
          <Ionicons name="shield-checkmark" size={32} color="#2c5282" />
          <Text style={styles.title}>{t('regulatory.title')}</Text>
          <HelpTooltip
            title={t('regulatory.title')}
            content={t('regulatory.tooltipContent')}
            size={20}
            color="#2c5282"
          />
        </View>
        <Text style={styles.subtitle}>{t('regulatory.subtitle')}</Text>
      </View>

      {/* Important Disclaimer */}
      <View style={styles.disclaimerCard}>
        <Ionicons name="warning" size={24} color="#d97706" style={styles.disclaimerIcon} />
        <View style={{ flex: 1 }}>
          <Text style={styles.disclaimerTitle}>{t('regulatory.currentStatus')}</Text>
          <Text style={styles.disclaimerText}>
            {t('regulatory.disclaimer')}
          </Text>
        </View>
      </View>

      {/* FDA Section */}
      <Pressable
        style={styles.sectionCard}
        onPress={() => toggleSection('fda')}
      >
        <View style={styles.sectionHeader}>
          <View style={{ flexDirection: 'row', alignItems: 'center', flex: 1 }}>
            <Text style={styles.sectionTitle}>{t('regulatory.fdaTitle')}</Text>
            <HelpTooltip
              title={t('regulatory.fdaTitle')}
              content={t('regulatory.fdaTooltip')}
              size={18}
              color="#2c5282"
            />
          </View>
          <Ionicons
            name={expandedSection === 'fda' ? 'chevron-up' : 'chevron-down'}
            size={24}
            color="#4a5568"
          />
        </View>

        {expandedSection === 'fda' && (
          <View style={styles.sectionContent}>
            {/* Classification */}
            <View style={styles.subsection}>
              <Text style={styles.subsectionTitle}>{t('regulatory.deviceClassification')}</Text>
              <View style={styles.infoBox}>
                <Text style={styles.infoLabel}>{t('regulatory.likelyClassification')}</Text>
                <Text style={styles.infoValue}>{t('regulatory.classII')}</Text>
                <Text style={styles.infoDescription}>
                  {t('regulatory.classIIDescription')}
                </Text>
              </View>
            </View>

            {/* Regulatory Pathway */}
            <View style={styles.subsection}>
              <Text style={styles.subsectionTitle}>{t('regulatory.pathway510k')}</Text>
              <View style={styles.stepsList}>
                <View style={styles.stepItem}>
                  <View style={styles.stepNumber}><Text style={styles.stepNumberText}>1</Text></View>
                  <View style={styles.stepContent}>
                    <Text style={styles.stepTitle}>{t('regulatory.step1Title')}</Text>
                    <Text style={styles.stepDescription}>{t('regulatory.step1Description')}</Text>
                  </View>
                </View>

                <View style={styles.stepItem}>
                  <View style={styles.stepNumber}><Text style={styles.stepNumberText}>2</Text></View>
                  <View style={styles.stepContent}>
                    <Text style={styles.stepTitle}>{t('regulatory.step2Title')}</Text>
                    <Text style={styles.stepDescription}>{t('regulatory.step2Description')}</Text>
                  </View>
                </View>

                <View style={styles.stepItem}>
                  <View style={styles.stepNumber}><Text style={styles.stepNumberText}>3</Text></View>
                  <View style={styles.stepContent}>
                    <Text style={styles.stepTitle}>{t('regulatory.step3Title')}</Text>
                    <Text style={styles.stepDescription}>{t('regulatory.step3Description')}</Text>
                  </View>
                </View>

                <View style={styles.stepItem}>
                  <View style={styles.stepNumber}><Text style={styles.stepNumberText}>4</Text></View>
                  <View style={styles.stepContent}>
                    <Text style={styles.stepTitle}>{t('regulatory.step4Title')}</Text>
                    <Text style={styles.stepDescription}>{t('regulatory.step4Description')}</Text>
                  </View>
                </View>

                <View style={styles.stepItem}>
                  <View style={styles.stepNumber}><Text style={styles.stepNumberText}>5</Text></View>
                  <View style={styles.stepContent}>
                    <Text style={styles.stepTitle}>{t('regulatory.step5Title')}</Text>
                    <Text style={styles.stepDescription}>{t('regulatory.step5Description')}</Text>
                  </View>
                </View>
              </View>
            </View>

            {/* Key Requirements */}
            <View style={styles.subsection}>
              <Text style={styles.subsectionTitle}>{t('regulatory.keyFDARequirements')}</Text>
              <View style={styles.requirementsList}>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.requirement1')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.requirement2')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.requirement3')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.requirement4')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.requirement5')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.requirement6')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.requirement7')}</Text>
                </View>
              </View>
            </View>

            {/* Estimated Timeline & Costs */}
            <View style={styles.subsection}>
              <Text style={styles.subsectionTitle}>{t('regulatory.estimatedTimeline')}</Text>
              <View style={styles.estimatesGrid}>
                <View style={styles.estimateCard}>
                  <Ionicons name="time-outline" size={24} color="#4299e1" />
                  <Text style={styles.estimateLabel}>{t('regulatory.developmentTime')}</Text>
                  <Text style={styles.estimateValue}>{t('regulatory.devTime')}</Text>
                </View>
                <View style={styles.estimateCard}>
                  <Ionicons name="cash-outline" size={24} color="#22c55e" />
                  <Text style={styles.estimateLabel}>{t('regulatory.totalCost')}</Text>
                  <Text style={styles.estimateValue}>{t('regulatory.cost')}</Text>
                </View>
                <View style={styles.estimateCard}>
                  <Ionicons name="people-outline" size={24} color="#f59e0b" />
                  <Text style={styles.estimateLabel}>{t('regulatory.clinicalStudy')}</Text>
                  <Text style={styles.estimateValue}>{t('regulatory.patients')}</Text>
                </View>
                <View style={styles.estimateCard}>
                  <Ionicons name="calendar-outline" size={24} color="#ef4444" />
                  <Text style={styles.estimateLabel}>{t('regulatory.fdaReview')}</Text>
                  <Text style={styles.estimateValue}>{t('regulatory.reviewTime')}</Text>
                </View>
              </View>
            </View>
          </View>
        )}
      </Pressable>

      {/* CE Mark Section */}
      <Pressable
        style={styles.sectionCard}
        onPress={() => toggleSection('ce')}
      >
        <View style={styles.sectionHeader}>
          <View style={{ flexDirection: 'row', alignItems: 'center', flex: 1 }}>
            <Text style={styles.sectionTitle}>{t('regulatory.ceMarkTitle')}</Text>
            <HelpTooltip
              title={t('regulatory.ceMarkTitle')}
              content={t('regulatory.ceMarkTooltip')}
              size={18}
              color="#2c5282"
            />
          </View>
          <Ionicons
            name={expandedSection === 'ce' ? 'chevron-up' : 'chevron-down'}
            size={24}
            color="#4a5568"
          />
        </View>

        {expandedSection === 'ce' && (
          <View style={styles.sectionContent}>
            {/* Classification */}
            <View style={styles.subsection}>
              <Text style={styles.subsectionTitle}>{t('regulatory.classificationMDR')}</Text>
              <View style={styles.infoBox}>
                <Text style={styles.infoLabel}>{t('regulatory.likelyClassification')}</Text>
                <Text style={styles.infoValue}>{t('regulatory.classIIaIIb')}</Text>
                <Text style={styles.infoDescription}>
                  {t('regulatory.mdrDescription')}
                </Text>
              </View>
            </View>

            {/* Regulatory Pathway */}
            <View style={styles.subsection}>
              <Text style={styles.subsectionTitle}>{t('regulatory.ceMarkConformity')}</Text>
              <View style={styles.stepsList}>
                <View style={styles.stepItem}>
                  <View style={styles.stepNumber}><Text style={styles.stepNumberText}>1</Text></View>
                  <View style={styles.stepContent}>
                    <Text style={styles.stepTitle}>{t('regulatory.ceStep1Title')}</Text>
                    <Text style={styles.stepDescription}>{t('regulatory.ceStep1Description')}</Text>
                  </View>
                </View>

                <View style={styles.stepItem}>
                  <View style={styles.stepNumber}><Text style={styles.stepNumberText}>2</Text></View>
                  <View style={styles.stepContent}>
                    <Text style={styles.stepTitle}>{t('regulatory.ceStep2Title')}</Text>
                    <Text style={styles.stepDescription}>{t('regulatory.ceStep2Description')}</Text>
                  </View>
                </View>

                <View style={styles.stepItem}>
                  <View style={styles.stepNumber}><Text style={styles.stepNumberText}>3</Text></View>
                  <View style={styles.stepContent}>
                    <Text style={styles.stepTitle}>{t('regulatory.ceStep3Title')}</Text>
                    <Text style={styles.stepDescription}>{t('regulatory.ceStep3Description')}</Text>
                  </View>
                </View>

                <View style={styles.stepItem}>
                  <View style={styles.stepNumber}><Text style={styles.stepNumberText}>4</Text></View>
                  <View style={styles.stepContent}>
                    <Text style={styles.stepTitle}>{t('regulatory.ceStep4Title')}</Text>
                    <Text style={styles.stepDescription}>{t('regulatory.ceStep4Description')}</Text>
                  </View>
                </View>

                <View style={styles.stepItem}>
                  <View style={styles.stepNumber}><Text style={styles.stepNumberText}>5</Text></View>
                  <View style={styles.stepContent}>
                    <Text style={styles.stepTitle}>{t('regulatory.ceStep5Title')}</Text>
                    <Text style={styles.stepDescription}>{t('regulatory.ceStep5Description')}</Text>
                  </View>
                </View>

                <View style={styles.stepItem}>
                  <View style={styles.stepNumber}><Text style={styles.stepNumberText}>6</Text></View>
                  <View style={styles.stepContent}>
                    <Text style={styles.stepTitle}>{t('regulatory.ceStep6Title')}</Text>
                    <Text style={styles.stepDescription}>{t('regulatory.ceStep6Description')}</Text>
                  </View>
                </View>
              </View>
            </View>

            {/* Key Requirements */}
            <View style={styles.subsection}>
              <Text style={styles.subsectionTitle}>{t('regulatory.keyMDRRequirements')}</Text>
              <View style={styles.requirementsList}>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.mdrReq1')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.mdrReq2')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.mdrReq3')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.mdrReq4')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.mdrReq5')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.mdrReq6')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#22c55e" />
                  <Text style={styles.requirementText}>{t('regulatory.mdrReq7')}</Text>
                </View>
              </View>
            </View>

            {/* Estimated Timeline & Costs */}
            <View style={styles.subsection}>
              <Text style={styles.subsectionTitle}>{t('regulatory.estimatedTimeline')}</Text>
              <View style={styles.estimatesGrid}>
                <View style={styles.estimateCard}>
                  <Ionicons name="time-outline" size={24} color="#4299e1" />
                  <Text style={styles.estimateLabel}>{t('regulatory.developmentTime')}</Text>
                  <Text style={styles.estimateValue}>{t('regulatory.ceDevTime')}</Text>
                </View>
                <View style={styles.estimateCard}>
                  <Ionicons name="cash-outline" size={24} color="#22c55e" />
                  <Text style={styles.estimateLabel}>{t('regulatory.totalCost')}</Text>
                  <Text style={styles.estimateValue}>{t('regulatory.ceCost')}</Text>
                </View>
                <View style={styles.estimateCard}>
                  <Ionicons name="people-outline" size={24} color="#f59e0b" />
                  <Text style={styles.estimateLabel}>{t('regulatory.notifiedBodyFees')}</Text>
                  <Text style={styles.estimateValue}>{t('regulatory.nbFees')}</Text>
                </View>
                <View style={styles.estimateCard}>
                  <Ionicons name="calendar-outline" size={24} color="#ef4444" />
                  <Text style={styles.estimateLabel}>{t('regulatory.nbReview')}</Text>
                  <Text style={styles.estimateValue}>{t('regulatory.reviewTime')}</Text>
                </View>
              </View>
            </View>
          </View>
        )}
      </Pressable>

      {/* AI/ML Specific Considerations */}
      <Pressable
        style={styles.sectionCard}
        onPress={() => toggleSection('aiml')}
      >
        <View style={styles.sectionHeader}>
          <View style={{ flexDirection: 'row', alignItems: 'center', flex: 1 }}>
            <Text style={styles.sectionTitle}>{t('regulatory.aimlTitle')}</Text>
            <HelpTooltip
              title={t('regulatory.aimlTitle')}
              content={t('regulatory.aimlTooltip')}
              size={18}
              color="#2c5282"
            />
          </View>
          <Ionicons
            name={expandedSection === 'aiml' ? 'chevron-up' : 'chevron-down'}
            size={24}
            color="#4a5568"
          />
        </View>

        {expandedSection === 'aiml' && (
          <View style={styles.sectionContent}>
            {/* Algorithm Transparency */}
            <View style={styles.subsection}>
              <Text style={styles.subsectionTitle}>{t('regulatory.algorithmTransparency')}</Text>
              <View style={styles.requirementsList}>
                <View style={styles.requirementItem}>
                  <Ionicons name="document-text" size={20} color="#4299e1" />
                  <Text style={styles.requirementText}>{t('regulatory.transparency1')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="document-text" size={20} color="#4299e1" />
                  <Text style={styles.requirementText}>{t('regulatory.transparency2')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="document-text" size={20} color="#4299e1" />
                  <Text style={styles.requirementText}>{t('regulatory.transparency3')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="document-text" size={20} color="#4299e1" />
                  <Text style={styles.requirementText}>{t('regulatory.transparency4')}</Text>
                </View>
              </View>
            </View>

            {/* Continuous Learning */}
            <View style={styles.subsection}>
              <Text style={styles.subsectionTitle}>{t('regulatory.continuousLearning')}</Text>
              <View style={styles.infoBox}>
                <Text style={styles.infoDescription}>
                  {t('regulatory.learningDescription')}
                </Text>
              </View>
              <View style={styles.requirementsList}>
                <View style={styles.requirementItem}>
                  <Ionicons name="git-branch" size={20} color="#6366f1" />
                  <Text style={styles.requirementText}>{t('regulatory.learning1')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="git-branch" size={20} color="#6366f1" />
                  <Text style={styles.requirementText}>{t('regulatory.learning2')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="git-branch" size={20} color="#6366f1" />
                  <Text style={styles.requirementText}>{t('regulatory.learning3')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="git-branch" size={20} color="#6366f1" />
                  <Text style={styles.requirementText}>{t('regulatory.learning4')}</Text>
                </View>
              </View>
            </View>

            {/* Bias & Fairness */}
            <View style={styles.subsection}>
              <Text style={styles.subsectionTitle}>{t('regulatory.biasMitigation')}</Text>
              <View style={styles.requirementsList}>
                <View style={styles.requirementItem}>
                  <Ionicons name="people" size={20} color="#ef4444" />
                  <Text style={styles.requirementText}>{t('regulatory.bias1')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="people" size={20} color="#ef4444" />
                  <Text style={styles.requirementText}>{t('regulatory.bias2')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="people" size={20} color="#ef4444" />
                  <Text style={styles.requirementText}>{t('regulatory.bias3')}</Text>
                </View>
                <View style={styles.requirementItem}>
                  <Ionicons name="people" size={20} color="#ef4444" />
                  <Text style={styles.requirementText}>{t('regulatory.bias4')}</Text>
                </View>
              </View>
            </View>
          </View>
        )}
      </Pressable>

      {/* Additional Resources */}
      <View style={styles.resourcesCard}>
        <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 16 }}>
          <Ionicons name="book" size={24} color="#2c5282" style={{ marginRight: 8 }} />
          <Text style={styles.resourcesTitle}>{t('regulatory.additionalResources')}</Text>
        </View>

        <View style={styles.resourcesList}>
          <View style={styles.resourceItem}>
            <Text style={styles.resourceLabel}>{t('regulatory.fdaGuidanceLabel')}</Text>
            <Text style={styles.resourceLink}>{t('regulatory.fdaGuidanceLink')}</Text>
          </View>
          <View style={styles.resourceItem}>
            <Text style={styles.resourceLabel}>{t('regulatory.euMDRLabel')}</Text>
            <Text style={styles.resourceLink}>{t('regulatory.euMDRLink')}</Text>
          </View>
          <View style={styles.resourceItem}>
            <Text style={styles.resourceLabel}>{t('regulatory.standardsLabel')}</Text>
            <Text style={styles.resourceLink}>{t('regulatory.standardsLink')}</Text>
          </View>
          <View style={styles.resourceItem}>
            <Text style={styles.resourceLabel}>{t('regulatory.guidanceLabel')}</Text>
            <Text style={styles.resourceLink}>{t('regulatory.guidanceLink')}</Text>
          </View>
        </View>
      </View>

      {/* Bottom Spacing */}
      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafb',
  },
  header: {
    padding: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c5282',
    marginLeft: 12,
  },
  subtitle: {
    fontSize: 14,
    color: '#64748b',
    marginTop: 4,
  },
  disclaimerCard: {
    flexDirection: 'row',
    backgroundColor: '#fef3c7',
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
    padding: 16,
    margin: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  disclaimerIcon: {
    marginRight: 12,
  },
  disclaimerTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#92400e',
    marginBottom: 8,
  },
  disclaimerText: {
    fontSize: 14,
    color: '#78350f',
    lineHeight: 20,
  },
  sectionCard: {
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginBottom: 12,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2d3748',
    marginRight: 8,
  },
  sectionContent: {
    padding: 16,
  },
  subsection: {
    marginBottom: 24,
  },
  subsectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
  },
  infoBox: {
    backgroundColor: '#f0f9ff',
    borderLeftWidth: 4,
    borderLeftColor: '#0ea5e9',
    padding: 12,
    borderRadius: 8,
  },
  infoLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#0c4a6e',
    marginBottom: 4,
  },
  infoValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#0369a1',
    marginBottom: 8,
  },
  infoDescription: {
    fontSize: 14,
    color: '#0c4a6e',
    lineHeight: 20,
  },
  stepsList: {
    gap: 12,
  },
  stepItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  stepNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#4299e1',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  stepNumberText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  stepContent: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#2d3748',
    marginBottom: 4,
  },
  stepDescription: {
    fontSize: 14,
    color: '#64748b',
    lineHeight: 20,
  },
  requirementsList: {
    gap: 12,
  },
  requirementItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  requirementText: {
    flex: 1,
    fontSize: 14,
    color: '#4a5568',
    marginLeft: 12,
    lineHeight: 20,
  },
  estimatesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  estimateCard: {
    flex: 1,
    minWidth: 140,
    backgroundColor: '#f8fafc',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  estimateLabel: {
    fontSize: 12,
    color: '#64748b',
    marginTop: 8,
    textAlign: 'center',
  },
  estimateValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1e293b',
    marginTop: 4,
    textAlign: 'center',
  },
  resourcesCard: {
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginTop: 12,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  resourcesTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  resourcesList: {
    gap: 12,
  },
  resourceItem: {
    borderLeftWidth: 3,
    borderLeftColor: '#4299e1',
    paddingLeft: 12,
  },
  resourceLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#2d3748',
    marginBottom: 4,
  },
  resourceLink: {
    fontSize: 13,
    color: '#0369a1',
    fontStyle: 'italic',
  },
});
