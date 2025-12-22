import React from 'react';
import { View, Text, StyleSheet, ScrollView, Pressable, Image } from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useTranslation } from 'react-i18next';

export default function HelpScreen() {
  const router = useRouter();
  const { t } = useTranslation();

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#f8fafb', '#e8f4f8', '#f0f8ff']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.background}
      />

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Image source={require('../assets/icon.png')} style={styles.appIcon} />
          <Text style={styles.title}>📚 {t('help.title')}</Text>
          <Text style={styles.subtitle}>{t('help.subtitle')}</Text>
        </View>

        {/* Getting Started */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🚀 {t('help.gettingStarted.title')}</Text>
          <Text style={styles.text}>
            {t('help.gettingStarted.content')}
          </Text>
        </View>

        {/* Taking Photos */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📸 {t('help.takingPhotos.title')}</Text>
          <Text style={styles.text}>{t('help.takingPhotos.intro')}</Text>

          <View style={styles.bulletList}>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                <Text style={styles.bold}>{t('help.takingPhotos.goodLighting')}</Text> {t('help.takingPhotos.goodLightingText')}
              </Text>
            </View>

            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                <Text style={styles.bold}>{t('help.takingPhotos.focus')}</Text> {t('help.takingPhotos.focusText')}
              </Text>
            </View>

            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                <Text style={styles.bold}>{t('help.takingPhotos.distance')}</Text> {t('help.takingPhotos.distanceText')}
              </Text>
            </View>

            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                <Text style={styles.bold}>{t('help.takingPhotos.referenceObject')}</Text> {t('help.takingPhotos.referenceObjectText')}
              </Text>
            </View>

            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                <Text style={styles.bold}>{t('help.takingPhotos.avoidFlash')}</Text> {t('help.takingPhotos.avoidFlashText')}
              </Text>
            </View>
          </View>
        </View>

        {/* Analysis Process */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🔬 {t('help.understandingAnalysis.title')}</Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.understandingAnalysis.step1Title')}</Text>
            <Text style={styles.text}>
              {t('help.understandingAnalysis.step1Text')}
            </Text>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.understandingAnalysis.step2Title')}</Text>
            <Text style={styles.text}>
              {t('help.understandingAnalysis.step2Text')}
            </Text>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.understandingAnalysis.step3Title')}</Text>
            <Text style={styles.text}>
              {t('help.understandingAnalysis.step3Text')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.understandingAnalysis.step3Item1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.understandingAnalysis.step3Item2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.understandingAnalysis.step3Item3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.understandingAnalysis.step3Item4')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.understandingAnalysis.step3Item5')}</Text>
              </View>
            </View>
          </View>
        </View>

        {/* Infectious Disease Classification */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🦠 {t('help.infectiousDisease.title')}</Text>
          <Text style={styles.text}>
            {t('help.infectiousDisease.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.infectiousDisease.whatItDetects')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>🦠</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.infectiousDisease.bacterial')}</Text> {t('help.infectiousDisease.bacterialList')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>🍄</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.infectiousDisease.fungal')}</Text> {t('help.infectiousDisease.fungalList')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>🦠</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.infectiousDisease.viral')}</Text> {t('help.infectiousDisease.viralList')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>🪲</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.infectiousDisease.parasitic')}</Text> {t('help.infectiousDisease.parasiticList')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.infectiousDisease.howToUse')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>1.</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.step1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>2.</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.step2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>3.</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.step3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>4.</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.step4')}</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.infectiousDisease.whatYouGet')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.infectionType')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.severity')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.contagiousness')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.symptoms')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.treatment')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.urgency')}</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>⚠️ {t('help.infectiousDisease.importantNotes')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.note1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.note2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.infectiousDisease.note3')}</Text>
              </View>
            </View>
          </View>
        </View>

        {/* Understanding Results */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📊 {t('help.resultsInterpretation.title')}</Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.resultsInterpretation.confidenceLevels')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.confidenceBullet}>🟢</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.resultsInterpretation.highConfidence')}</Text> {t('help.resultsInterpretation.highConfidenceText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.confidenceBullet}>🟡</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.resultsInterpretation.moderateConfidence')}</Text> {t('help.resultsInterpretation.moderateConfidenceText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.confidenceBullet}>🔴</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.resultsInterpretation.lowConfidence')}</Text> {t('help.resultsInterpretation.lowConfidenceText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.resultsInterpretation.riskLevels')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🟢</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.resultsInterpretation.lowRisk')}</Text> {t('help.resultsInterpretation.lowRiskText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🟡</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.resultsInterpretation.moderateRisk')}</Text> {t('help.resultsInterpretation.moderateRiskText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.resultsInterpretation.highRisk')}</Text> {t('help.resultsInterpretation.highRiskText')}
                </Text>
              </View>
            </View>
          </View>
        </View>

        {/* AI Explainability Heatmap */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🔥 {t('help.aiExplainability.title')}</Text>
          <Text style={styles.text}>
            {t('help.aiExplainability.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.aiExplainability.whatIsIt')}</Text>
            <Text style={styles.text}>
              {t('help.aiExplainability.whatIsItText')}
            </Text>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.aiExplainability.howToRead')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.aiExplainability.redOrange')}</Text> {t('help.aiExplainability.redOrangeText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.aiExplainability.yellow')}</Text> {t('help.aiExplainability.yellowText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.aiExplainability.blueGreen')}</Text> {t('help.aiExplainability.blueGreenText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.aiExplainability.whyItMatters')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.aiExplainability.matter1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.aiExplainability.matter2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.aiExplainability.matter3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.aiExplainability.matter4')}</Text>
              </View>
            </View>
          </View>
        </View>

        {/* Plain English Explanation */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>💬 {t('help.plainEnglish.title')}</Text>
          <Text style={styles.text}>
            {t('help.plainEnglish.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.plainEnglish.whatYouGet')}</Text>
            <Text style={styles.text}>
              {t('help.plainEnglish.whatYouGetText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.plainEnglish.item1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.plainEnglish.item2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.plainEnglish.item3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.plainEnglish.item4')}</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>🔊 {t('help.plainEnglish.textToSpeech')}</Text>
            <Text style={styles.text}>
              {t('help.plainEnglish.textToSpeechText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.plainEnglish.tts1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.plainEnglish.tts2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.plainEnglish.tts3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.plainEnglish.tts4')}</Text>
              </View>
            </View>
          </View>
        </View>

        {/* AI Chat Assistant */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🤖 AI Chat Assistant</Text>
          <Text style={styles.text}>
            Get instant, AI-powered explanations about your skin conditions using advanced GPT-4 technology.
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>Learn More About Conditions</Text>
            <Text style={styles.text}>
              After receiving an analysis, tap the "Learn More About This Condition" button to get a detailed,
              patient-friendly explanation of the diagnosed condition.
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>What it is:</Text> Clear description of the condition
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Common causes:</Text> Why this condition might occur
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Symptoms:</Text> What to look out for
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Treatment options:</Text> General guidance on management
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>Differential Diagnosis Reasoning</Text>
            <Text style={styles.text}>
              Tap "Show Diagnostic Reasoning" to see how the AI reached its diagnosis with a detailed
              chain-of-thought explanation.
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>1.</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Initial Assessment:</Text> Key features observed in your image
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>2.</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Primary Diagnosis Reasoning:</Text> Why the top diagnosis was selected
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>3.</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Differential Considerations:</Text> Why other conditions were considered
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>4.</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Key Distinguishing Features:</Text> What separates diagnoses
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>5.</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Recommended Next Steps:</Text> Confirmation tests and follow-up
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>Where to Find It</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>📱</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Home Screen:</Text> After any analysis, look for the blue "Learn More" button
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>📋</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Analysis History:</Text> Each past analysis has a "Learn More" option
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>🔍</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Analysis Details:</Text> Full explanation and reasoning available
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.warningBox}>
            <Text style={styles.warningText}>
              ⚠️ AI explanations are for educational purposes only. Always consult a healthcare provider for medical advice and treatment decisions.
            </Text>
          </View>
        </View>

        {/* Data Augmentation */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🔄 Data Augmentation</Text>
          <Text style={styles.text}>
            Generate synthetic training data for rare skin conditions to improve AI model accuracy. This feature is primarily for researchers and medical professionals.
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>Augmentation Types</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>📐</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Geometric:</Text> Rotation, scaling, flipping, shear transformations
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>🎨</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Color:</Text> Brightness, contrast, saturation, hue adjustments
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>📡</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Noise:</Text> Gaussian noise, blur, sharpening effects
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>⚡</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Advanced:</Text> Elastic deformation, grid distortion, random erasing
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>🏥</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Dermatology:</Text> Flash simulation, skin tone variation, lesion enhancement
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>🔀</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>Mixup/CutMix:</Text> Image blending for hybrid sample generation
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>How to Use</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>1.</Text>
                <Text style={styles.bulletText}>Navigate to Data Augmentation from the main menu</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>2.</Text>
                <Text style={styles.bulletText}>Select an image to augment</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>3.</Text>
                <Text style={styles.bulletText}>Choose augmentation types (long-press to preview)</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>4.</Text>
                <Text style={styles.bulletText}>Set the number of augmented versions (1-20)</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>5.</Text>
                <Text style={styles.bulletText}>Generate and review results</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>Rare Conditions</Text>
            <Text style={styles.text}>
              The system identifies 15 rare skin conditions that benefit most from data augmentation, including melanoma subtypes, Merkel cell carcinoma, and other uncommon diagnoses.
            </Text>
          </View>
        </View>

        {/* Clinical Decision Support */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>⚕️ {t('help.clinicalDecision.title')}</Text>
          <Text style={styles.text}>
            {t('help.clinicalDecision.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.clinicalDecision.whatsIncluded')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.clinicalDecision.urgencyLevel')}</Text> {t('help.clinicalDecision.urgencyLevelText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.clinicalDecision.treatmentProtocol')}</Text> {t('help.clinicalDecision.treatmentProtocolText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.clinicalDecision.medicationInfo')}</Text> {t('help.clinicalDecision.medicationInfoText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.clinicalDecision.drugInteraction')}</Text> {t('help.clinicalDecision.drugInteractionText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.clinicalDecision.insurance')}</Text> {t('help.clinicalDecision.insuranceText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.clinicalDecision.followUp')}</Text> {t('help.clinicalDecision.followUpText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.clinicalDecision.patientEducation')}</Text> {t('help.clinicalDecision.patientEducationText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.clinicalDecision.evidenceBased')}</Text>
            <Text style={styles.text}>
              {t('help.clinicalDecision.evidenceBasedText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.clinicalDecision.aad')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.clinicalDecision.nccn')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.clinicalDecision.fda')}</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>⚠️ {t('help.clinicalDecision.importantNote')}</Text>
            <Text style={styles.text}>
              {t('help.clinicalDecision.importantNoteText')}
            </Text>
          </View>
        </View>

        {/* Find Nearby Specialists */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🏥 {t('help.findSpecialists.title')}</Text>
          <Text style={styles.text}>
            {t('help.findSpecialists.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.findSpecialists.howItWorks')}</Text>
            <View style={styles.numberList}>
              <View style={styles.numberItem}>
                <Text style={styles.numberBullet}>1.</Text>
                <Text style={styles.bulletText}>
                  {t('help.findSpecialists.step1')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.numberBullet}>2.</Text>
                <Text style={styles.bulletText}>
                  {t('help.findSpecialists.step2')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.numberBullet}>3.</Text>
                <Text style={styles.bulletText}>
                  {t('help.findSpecialists.step3')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.numberBullet}>4.</Text>
                <Text style={styles.bulletText}>
                  {t('help.findSpecialists.step4')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.numberBullet}>5.</Text>
                <Text style={styles.bulletText}>
                  {t('help.findSpecialists.step5')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.findSpecialists.specialistTypes')}</Text>
            <Text style={styles.text}>
              {t('help.findSpecialists.specialistTypesText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.findSpecialists.dermatologist')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.findSpecialists.surgicalOncologist')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.findSpecialists.plasticSurgeon')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.findSpecialists.allergist')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.findSpecialists.infectiousDisease')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.findSpecialists.andMore')}</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>📍 {t('help.findSpecialists.locationPermissions')}</Text>
            <Text style={styles.text}>
              {t('help.findSpecialists.locationPermissionsText')}
            </Text>
          </View>
        </View>

        {/* Progress Tracking */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📊 {t('help.realTimeProgress.title')}</Text>
          <Text style={styles.text}>
            {t('help.realTimeProgress.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.realTimeProgress.whatYouSee')}</Text>
            <Text style={styles.text}>
              {t('help.realTimeProgress.whatYouSeeText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.realTimeProgress.stage1')}</Text> {t('help.realTimeProgress.stage1Text')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.realTimeProgress.stage2')}</Text> {t('help.realTimeProgress.stage2Text')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.realTimeProgress.stage3')}</Text> {t('help.realTimeProgress.stage3Text')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.realTimeProgress.stage4')}</Text> {t('help.realTimeProgress.stage4Text')}
                </Text>
              </View>
            </View>
            <Text style={styles.text} style={{marginTop: 10}}>
              {t('help.realTimeProgress.conclusion')}
            </Text>
          </View>
        </View>

        {/* Insurance Pre-Authorization AI */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📋 {t('help.insurancePreAuth.title')}</Text>
          <Text style={styles.text}>
            {t('help.insurancePreAuth.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>📄 {t('help.insurancePreAuth.whatYouGet')}</Text>
            <View style={styles.numberList}>
              <View style={styles.numberItem}>
                <Text style={styles.number}>1.</Text>
                <Text style={styles.numberText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.medicalNecessity')}</Text> - {t('help.insurancePreAuth.medicalNecessityText')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>2.</Text>
                <Text style={styles.numberText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.preAuthForm')}</Text> - {t('help.insurancePreAuth.preAuthFormText')}
                  {'\n'}  • {t('help.insurancePreAuth.icd10')}
                  {'\n'}  • {t('help.insurancePreAuth.cpt')}
                  {'\n'}  • {t('help.insurancePreAuth.clinicalRationale')}
                  {'\n'}  • {t('help.insurancePreAuth.urgencyClass')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>3.</Text>
                <Text style={styles.numberText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.clinicalSummary')}</Text> - {t('help.insurancePreAuth.clinicalSummaryText')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>4.</Text>
                <Text style={styles.numberText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.supportingEvidence')}</Text> - {t('help.insurancePreAuth.supportingEvidenceText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>🏥 {t('help.insurancePreAuth.recommendedProcedures')}</Text>
            <Text style={styles.text}>
              {t('help.insurancePreAuth.recommendedProceduresText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.dermoscopy')}</Text> - {t('help.insurancePreAuth.dermoscopyText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.biopsy')}</Text> - {t('help.insurancePreAuth.biopsyText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.excision')}</Text> - {t('help.insurancePreAuth.excisionText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.phototherapy')}</Text> - {t('help.insurancePreAuth.phototherapyText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.destruction')}</Text> - {t('help.insurancePreAuth.destructionText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>📊 {t('help.insurancePreAuth.authorizationSummary')}</Text>
            <Text style={styles.text}>
              {t('help.insurancePreAuth.authorizationSummaryText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.insurancePreAuth.primaryDiagnosis')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.insurancePreAuth.urgencyClassification')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.insurancePreAuth.diagnosticConfidence')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.insurancePreAuth.timeline')}</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>📤 {t('help.insurancePreAuth.exportShare')}</Text>
            <Text style={styles.text}>
              {t('help.insurancePreAuth.exportShareText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.exportPDF')}</Text> - {t('help.insurancePreAuth.exportPDFText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.shareDoc')}</Text> - {t('help.insurancePreAuth.shareDocText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.viewFullDocs')}</Text> - {t('help.insurancePreAuth.viewFullDocsText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>💡 {t('help.insurancePreAuth.howToUse')}</Text>
            <View style={styles.numberList}>
              <View style={styles.numberItem}>
                <Text style={styles.number}>1.</Text>
                <Text style={styles.numberText}>{t('help.insurancePreAuth.howToUseStep1')}</Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>2.</Text>
                <Text style={styles.numberText}>{t('help.insurancePreAuth.howToUseStep2')}</Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>3.</Text>
                <Text style={styles.numberText}>{t('help.insurancePreAuth.howToUseStep3')}</Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>4.</Text>
                <Text style={styles.numberText}>{t('help.insurancePreAuth.howToUseStep4')}</Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>5.</Text>
                <Text style={styles.numberText}>{t('help.insurancePreAuth.howToUseStep5')}</Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>6.</Text>
                <Text style={styles.numberText}>{t('help.insurancePreAuth.howToUseStep6')}</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>⚕️ {t('help.insurancePreAuth.clinicalGuidelines')}</Text>
            <Text style={styles.text}>
              {t('help.insurancePreAuth.clinicalGuidelinesText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.insurancePreAuth.guidelineAAD')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.insurancePreAuth.guidelineNCCN')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.insurancePreAuth.guidelineSCF')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.insurancePreAuth.guidelineAADA')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.insurancePreAuth.guidelineNPF')}</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>🎯 {t('help.insurancePreAuth.whyItMatters')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.fasterApprovals')}</Text> - {t('help.insurancePreAuth.fasterApprovalsText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.betterCoverage')}</Text> - {t('help.insurancePreAuth.betterCoverageText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.savesTime')}</Text> - {t('help.insurancePreAuth.savesTimeText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.reducesDenials')}</Text> - {t('help.insurancePreAuth.reducesDenialsText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.professionalQuality')}</Text> - {t('help.insurancePreAuth.professionalQualityText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>🎯 {t('help.insurancePreAuth.approvalLikelihood')}</Text>
            <Text style={styles.text}>
              {t('help.insurancePreAuth.approvalLikelihoodText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.diagnosticConfidenceFactor')}</Text> - {t('help.insurancePreAuth.diagnosticConfidenceFactorText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.clinicalUrgency')}</Text> - {t('help.insurancePreAuth.clinicalUrgencyText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.conditionType')}</Text> - {t('help.insurancePreAuth.conditionTypeText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.supportingEvidenceFactor')}</Text> - {t('help.insurancePreAuth.supportingEvidenceFactorText')}
                </Text>
              </View>
            </View>
            <Text style={styles.text}>
              {t('help.insurancePreAuth.approvalScore')}
            </Text>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>📊 {t('help.insurancePreAuth.statusTracking')}</Text>
            <Text style={styles.text}>
              {t('help.insurancePreAuth.statusTrackingText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.statusDraft')}</Text> - {t('help.insurancePreAuth.statusDraftText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.statusSubmitted')}</Text> - {t('help.insurancePreAuth.statusSubmittedText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.statusUnderReview')}</Text> - {t('help.insurancePreAuth.statusUnderReviewText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.statusApproved')}</Text> - {t('help.insurancePreAuth.statusApprovedText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.statusDenied')}</Text> - {t('help.insurancePreAuth.statusDeniedText')}
                </Text>
              </View>
            </View>
            <Text style={styles.text}>
              {t('help.insurancePreAuth.updateStatus')}
            </Text>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>📝 {t('help.insurancePreAuth.autoFillForms')}</Text>
            <Text style={styles.text}>
              {t('help.insurancePreAuth.autoFillFormsText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.cms1500')}</Text> - {t('help.insurancePreAuth.cms1500Text')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.ub04')}</Text> - {t('help.insurancePreAuth.ub04Text')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={{fontWeight: 'bold'}}>{t('help.insurancePreAuth.genericPreAuth')}</Text> - {t('help.insurancePreAuth.genericPreAuthText')}
                </Text>
              </View>
            </View>
            <Text style={styles.text}>
              {t('help.insurancePreAuth.autoFillNote')}
            </Text>
          </View>

          <View style={styles.importantNote}>
            <Text style={styles.importantTitle}>⚠️ {t('help.insurancePreAuth.importantNotes')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  {t('help.insurancePreAuth.importantNote1')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  {t('help.insurancePreAuth.importantNote2')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  {t('help.insurancePreAuth.importantNote3')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  {t('help.insurancePreAuth.importantNote4')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  {t('help.insurancePreAuth.importantNote5')}
                </Text>
              </View>
            </View>
          </View>
        </View>

        {/* Analysis History */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📋 {t('help.historyManagement.title')}</Text>
          <Text style={styles.text}>
            {t('help.historyManagement.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.historyManagement.whatYouCanDo')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.historyManagement.viewPastAnalyses')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.historyManagement.trackChanges')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.historyManagement.markLocations')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.historyManagement.recordSymptoms')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.historyManagement.documentMedications')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.historyManagement.addBiopsyResults')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.historyManagement.exportReports')}</Text>
              </View>
            </View>
          </View>
        </View>

        {/* Teledermatology */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>👨‍⚕️ {t('help.sharingDermatologist.title')}</Text>
          <Text style={styles.text}>
            {t('help.sharingDermatologist.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sharingDermatologist.howToShare')}</Text>
            <View style={styles.numberList}>
              <View style={styles.numberItem}>
                <Text style={styles.number}>1.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sharingDermatologist.step1')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>2.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sharingDermatologist.step2')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>3.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sharingDermatologist.step3')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>4.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sharingDermatologist.step4')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>5.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sharingDermatologist.step5')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sharingDermatologist.whatDermatologistSees')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.sharingDermatologist.professionalWebPage')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.sharingDermatologist.downloadablePDF')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.sharingDermatologist.skinLesionImage')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.sharingDermatologist.clinicalInfo')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.sharingDermatologist.yourMessage')}</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sharingDermatologist.dermatologistReview')}</Text>
            <Text style={styles.text}>
              {t('help.sharingDermatologist.dermatologistReviewText')}
            </Text>
          </View>
        </View>

        {/* Body Map */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🗺️ {t('help.bodyMap.title')}</Text>
          <Text style={styles.text}>
            {t('help.bodyMap.intro')}
          </Text>
          <View style={styles.bulletList}>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                {t('help.bodyMap.selectRegion')}
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                {t('help.bodyMap.chooseSide')}
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                {t('help.bodyMap.trackMultiple')}
              </Text>
            </View>
          </View>
        </View>

        {/* Lesion Tracking */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🔍 {t('help.lesionTracking.title')}</Text>
          <Text style={styles.text}>
            {t('help.lesionTracking.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.lesionTracking.creatingTrackedLesion')}</Text>
            <View style={styles.numberList}>
              <View style={styles.numberItem}>
                <Text style={styles.number}>1.</Text>
                <Text style={styles.bulletText}>
                  {t('help.lesionTracking.step1')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>2.</Text>
                <Text style={styles.bulletText}>
                  {t('help.lesionTracking.step2')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>3.</Text>
                <Text style={styles.bulletText}>
                  {t('help.lesionTracking.step3')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>4.</Text>
                <Text style={styles.bulletText}>
                  {t('help.lesionTracking.step4')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.lesionTracking.aiChangeDetection')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.lesionTracking.sizeChanges')}</Text> {t('help.lesionTracking.sizeChangesText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.lesionTracking.colorChanges')}</Text> {t('help.lesionTracking.colorChangesText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.lesionTracking.shapeChanges')}</Text> {t('help.lesionTracking.shapeChangesText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.lesionTracking.symptomChanges')}</Text> {t('help.lesionTracking.symptomChangesText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.lesionTracking.abcdeCriteria')}</Text> {t('help.lesionTracking.abcdeCriteriaText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.lesionTracking.riskEscalation')}</Text> {t('help.lesionTracking.riskEscalationText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.lesionTracking.viewingTrackedLesions')}</Text>
            <Text style={styles.text}>
              {t('help.lesionTracking.viewingTrackedLesionsText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.lesionTracking.completeTimeline')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.lesionTracking.sideBySide')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.lesionTracking.growthCharts')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.lesionTracking.alertBadges')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.lesionTracking.clinicalRecommendations')}</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.lesionTracking.whenToSeek')}</Text>
            <Text style={styles.text}>
              {t('help.lesionTracking.whenToSeekText')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.lesionTracking.rapidGrowth')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.lesionTracking.colorDarkening')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.lesionTracking.borderIrregularity')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.lesionTracking.newSymptoms')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.lesionTracking.riskLevelEscalation')}</Text>
              </View>
            </View>
          </View>
        </View>

        {/* Family History & Genetic Risk */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🧬 {t('help.familyHistory.title')}</Text>
          <Text style={styles.text}>
            {t('help.familyHistory.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.familyHistory.addingFamilyMembers')}</Text>
            <View style={styles.numberList}>
              <View style={styles.numberItem}>
                <Text style={styles.number}>1.</Text>
                <Text style={styles.bulletText}>
                  {t('help.familyHistory.step1')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>2.</Text>
                <Text style={styles.bulletText}>
                  {t('help.familyHistory.step2')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>3.</Text>
                <Text style={styles.bulletText}>
                  {t('help.familyHistory.step3')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>4.</Text>
                <Text style={styles.bulletText}>
                  {t('help.familyHistory.step4')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.familyHistory.geneticRiskFactors')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.familyHistory.familyHistoryPatterns')}</Text> {t('help.familyHistory.familyHistoryPatternsText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.familyHistory.multiplePrimaryMelanomas')}</Text> {t('help.familyHistory.multiplePrimaryMelanomasText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.familyHistory.earlyOnsetMelanoma')}</Text> {t('help.familyHistory.earlyOnsetMelanomaText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.familyHistory.pancreaticCancerLink')}</Text> {t('help.familyHistory.pancreaticCancerLinkText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.familyHistory.personalRiskFactors')}</Text> {t('help.familyHistory.personalRiskFactorsText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.familyHistory.riskAssessmentResults')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🟢</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.familyHistory.lowRisk')}</Text> {t('help.familyHistory.lowRiskText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🟡</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.familyHistory.moderateRisk')}</Text> {t('help.familyHistory.moderateRiskText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.familyHistory.highRisk')}</Text> {t('help.familyHistory.highRiskText')}
                </Text>
              </View>
            </View>
          </View>
        </View>

        {/* Genetic Testing Integration */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🧪 {t('help.geneticTesting.title')}</Text>
          <Text style={styles.text}>
            {t('help.geneticTesting.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.geneticTesting.supportedTests')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.geneticTesting.cdkn2a')}</Text> {t('help.geneticTesting.cdkn2aText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.geneticTesting.cdk4')}</Text> {t('help.geneticTesting.cdk4Text')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.geneticTesting.bap1')}</Text> {t('help.geneticTesting.bap1Text')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.geneticTesting.mc1r')}</Text> {t('help.geneticTesting.mc1rText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.geneticTesting.mitf')}</Text> {t('help.geneticTesting.mitfText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.geneticTesting.howToAdd')}</Text>
            <View style={styles.numberList}>
              <View style={styles.numberItem}>
                <Text style={styles.number}>1.</Text>
                <Text style={styles.bulletText}>
                  {t('help.geneticTesting.addStep1')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>2.</Text>
                <Text style={styles.bulletText}>
                  {t('help.geneticTesting.addStep2')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>3.</Text>
                <Text style={styles.bulletText}>
                  {t('help.geneticTesting.addStep3')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>4.</Text>
                <Text style={styles.bulletText}>
                  {t('help.geneticTesting.addStep4')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.geneticTesting.counselingRecommended')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.geneticTesting.threePlusFamily')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.geneticTesting.multiplePrimary')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.geneticTesting.beforeAge40')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.geneticTesting.familyPancreatic')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.geneticTesting.positiveResult')}</Text>
              </View>
            </View>
          </View>
        </View>

        {/* Alert System */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🔔 {t('help.smartAlerts.title')}</Text>
          <Text style={styles.text}>
            {t('help.smartAlerts.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.smartAlerts.typesTitle')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.smartAlerts.followUpReminders')}</Text> {t('help.smartAlerts.followUpRemindersText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.smartAlerts.changeDetection')}</Text> {t('help.smartAlerts.changeDetectionText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.smartAlerts.highRiskFindings')}</Text> {t('help.smartAlerts.highRiskFindingsText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.smartAlerts.screeningReminders')}</Text> {t('help.smartAlerts.screeningRemindersText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.smartAlerts.geneticCounseling')}</Text> {t('help.smartAlerts.geneticCounselingText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.smartAlerts.priorityLevels')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.smartAlerts.critical')}</Text> {t('help.smartAlerts.criticalText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🟡</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.smartAlerts.important')}</Text> {t('help.smartAlerts.importantText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.confidenceBullet}>🟢</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.smartAlerts.routine')}</Text> {t('help.smartAlerts.routineText')}
                </Text>
              </View>
            </View>
          </View>
        </View>

        {/* Predictive Analytics Dashboard */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📊 {t('help.predictiveAnalytics.title')}</Text>
          <Text style={styles.text}>
            {t('help.predictiveAnalytics.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.predictiveAnalytics.growthForecasting')}</Text>
            <Text style={styles.text}>
              {t('help.predictiveAnalytics.growthIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.sizePredictions')}</Text> {t('help.predictiveAnalytics.sizePredictionsText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.growthTrend')}</Text> {t('help.predictiveAnalytics.growthTrendText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.riskEscalation')}</Text> {t('help.predictiveAnalytics.riskEscalationText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.confidenceScoring')}</Text> {t('help.predictiveAnalytics.confidenceScoringText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.changeVelocity')}</Text> {t('help.predictiveAnalytics.changeVelocityText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.predictiveAnalytics.personalizedSchedule')}</Text>
            <Text style={styles.text}>
              {t('help.predictiveAnalytics.scheduleIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.selfExam')}</Text> {t('help.predictiveAnalytics.selfExamText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.dermaVisit')}</Text> {t('help.predictiveAnalytics.dermaVisitText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.individualLesion')}</Text> {t('help.predictiveAnalytics.individualLesionText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.geneticAppt')}</Text> {t('help.predictiveAnalytics.geneticApptText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.oneTapComplete')}</Text> {t('help.predictiveAnalytics.oneTapCompleteText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.predictiveAnalytics.riskTrend')}</Text>
            <Text style={styles.text}>
              {t('help.predictiveAnalytics.riskTrendIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.overallScore')}</Text> {t('help.predictiveAnalytics.overallScoreText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.melanomaSpecific')}</Text> {t('help.predictiveAnalytics.melanomaSpecificText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.historicalSnapshots')}</Text> {t('help.predictiveAnalytics.historicalSnapshotsText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.contributingFactors')}</Text> {t('help.predictiveAnalytics.contributingFactorsText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.predictiveAnalytics.accessingAnalytics')}</Text>
            <View style={styles.numberList}>
              <View style={styles.numberItem}>
                <Text style={styles.number}>1.</Text>
                <Text style={styles.bulletText}>
                  {t('help.predictiveAnalytics.accessStep1')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>2.</Text>
                <Text style={styles.bulletText}>
                  {t('help.predictiveAnalytics.accessStep2')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>3.</Text>
                <Text style={styles.bulletText}>
                  {t('help.predictiveAnalytics.accessStep3')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>4.</Text>
                <Text style={styles.bulletText}>
                  {t('help.predictiveAnalytics.accessStep4')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>5.</Text>
                <Text style={styles.bulletText}>
                  {t('help.predictiveAnalytics.accessStep5')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.predictiveAnalytics.forecastAccuracy')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.confidenceBullet}>🟢</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.highConfidence')}</Text> {t('help.predictiveAnalytics.highConfidenceText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🟡</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.mediumConfidence')}</Text> {t('help.predictiveAnalytics.mediumConfidenceText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.predictiveAnalytics.lowConfidence')}</Text> {t('help.predictiveAnalytics.lowConfidenceText')}
                </Text>
              </View>
            </View>
            <Text style={styles.text} style={{ marginTop: 12 }}>
              {'\n'}
              <Text style={styles.bold}>{t('help.predictiveAnalytics.importantNote')}</Text> {t('help.predictiveAnalytics.importantNoteText')}
            </Text>
          </View>
        </View>

        {/* Sun Exposure & UV Tracking */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>☀️ {t('help.sunExposure.title')}</Text>
          <Text style={styles.text}>
            {t('help.sunExposure.intro')}
          </Text>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sunExposure.whyTrack')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.understandRisk')}</Text> {t('help.sunExposure.understandRiskText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.correlateLesions')}</Text> {t('help.sunExposure.correlateLesionsText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.improveHabits')}</Text> {t('help.sunExposure.improveHabitsText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.preventDamage')}</Text> {t('help.sunExposure.preventDamageText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sunExposure.loggingExposure')}</Text>
            <View style={styles.numberList}>
              <View style={styles.numberItem}>
                <Text style={styles.number}>1.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sunExposure.logStep1')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>2.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sunExposure.logStep2')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>3.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sunExposure.logStep3')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>4.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sunExposure.logStep4')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>5.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sunExposure.logStep5')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>6.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sunExposure.logStep6')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>7.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sunExposure.logStep7')}
                </Text>
              </View>
              <View style={styles.numberItem}>
                <Text style={styles.number}>8.</Text>
                <Text style={styles.bulletText}>
                  {t('help.sunExposure.logStep8')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sunExposure.uvIndex')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.confidenceBullet}>🟢</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.low')}</Text> {t('help.sunExposure.lowText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.confidenceBullet}>🟡</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.moderate')}</Text> {t('help.sunExposure.moderateText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🟠</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.high')}</Text> {t('help.sunExposure.highText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.veryHigh')}</Text> {t('help.sunExposure.veryHighText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.extreme')}</Text> {t('help.sunExposure.extremeText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sunExposure.viewingHistory')}</Text>
            <Text style={styles.text}>
              {t('help.sunExposure.historyIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.sunExposure.dateAndDuration')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.sunExposure.locationActivity')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.sunExposure.protectionUsed')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.sunExposure.skinReactions')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.sunExposure.uvDose')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.sunExposure.colorCoded')}</Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sunExposure.statistics')}</Text>
            <Text style={styles.text}>
              {t('help.sunExposure.statsIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.totalExposures')}</Text> {t('help.sunExposure.totalExposuresText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.avgMax')}</Text> {t('help.sunExposure.avgMaxText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.sunburnEvents')}</Text> {t('help.sunExposure.sunburnEventsText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.protectionRate')}</Text> {t('help.sunExposure.protectionRateText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.avgSPF')}</Text> {t('help.sunExposure.avgSPFText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.highRiskExposures')}</Text> {t('help.sunExposure.highRiskExposuresText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.personalizedRecs')}</Text> {t('help.sunExposure.personalizedRecsText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sunExposure.lesionCorrelation')}</Text>
            <Text style={styles.text}>
              {t('help.sunExposure.correlationIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.correlationScore')}</Text> {t('help.sunExposure.correlationScoreText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.riskFactorsId')}</Text> {t('help.sunExposure.riskFactorsIdText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.protectiveFactors')}</Text> {t('help.sunExposure.protectiveFactorsText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.preventionRecs')}</Text> {t('help.sunExposure.preventionRecsText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.screeningUrgency')}</Text> {t('help.sunExposure.screeningUrgencyText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sunExposure.correlationTypes')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.strongPositive')}</Text> {t('help.sunExposure.strongPositiveText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🟡</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.moderatePositive')}</Text> {t('help.sunExposure.moderatePositiveText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.confidenceBullet}>🟢</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.weakNone')}</Text> {t('help.sunExposure.weakNoneText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sunExposure.bestPractices')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.useSunscreen')}</Text> {t('help.sunExposure.useSunscreenText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.reapply')}</Text> {t('help.sunExposure.reapplyText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.seekShade')}</Text> {t('help.sunExposure.seekShadeText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.wearClothing')}</Text> {t('help.sunExposure.wearClothingText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.avoidTanning')}</Text> {t('help.sunExposure.avoidTanningText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.checkIndex')}</Text> {t('help.sunExposure.checkIndexText')}
                </Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>
                  <Text style={styles.bold}>{t('help.sunExposure.beCareful')}</Text> {t('help.sunExposure.beCarefulText')}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.sunExposure.concernsTitle')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.sunExposure.multipleSunburns')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.sunExposure.frequentHighUV')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.sunExposure.lowCompliance')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.sunExposure.strongCorrelation')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}>{t('help.sunExposure.newChanging')}</Text>
              </View>
            </View>
            <Text style={styles.text} style={{ marginTop: 12 }}>
              {'\n'}
              {t('help.sunExposure.concernsFooter')}
            </Text>
          </View>
        </View>

        {/* Treatment Monitoring */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>💊 {t('help.treatmentMonitoring.title')}</Text>

          {/* Why Track Treatment Effectiveness? */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.treatmentMonitoring.whyTrack')}</Text>
            <Text style={styles.text}>
              {t('help.treatmentMonitoring.whyTrackIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.trackObjective')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.monitorAdherence')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.documentSideEffects')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.compareBeforeAfter')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.provideRecords')}</Text>
              </View>
            </View>
          </View>

          {/* Creating a Treatment Record */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.treatmentMonitoring.creatingRecord')}</Text>
            <Text style={styles.text}>
              {t('help.treatmentMonitoring.createIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>1.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.createStep1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>2.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.createStep2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>3.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.createStep3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>4.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.createStep4')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>5.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.createStep5')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>6.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.createStep6')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>7.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.createStep7')}</Text>
              </View>
            </View>
          </View>

          {/* Treatment Types */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.treatmentMonitoring.treatmentTypes')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>🧴</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.topical')}</Text> {t('help.treatmentMonitoring.topicalText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>💊</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.oral')}</Text> {t('help.treatmentMonitoring.oralText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>💉</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.injection')}</Text> {t('help.treatmentMonitoring.injectionText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>🏥</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.procedure')}</Text> {t('help.treatmentMonitoring.procedureText')}</Text>
              </View>
            </View>
          </View>

          {/* Logging Doses */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.treatmentMonitoring.loggingDoses')}</Text>
            <Text style={styles.text}>
              {t('help.treatmentMonitoring.loggingIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>1.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.logStep1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>2.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.logStep2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>3.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.logStep3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>4.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.logStep4')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>5.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.logStep5')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>6.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.logStep6')}</Text>
              </View>
            </View>
            <Text style={styles.text} style={{ marginTop: 12 }}>
              {'\n'}
              <Text style={styles.bold}>{t('help.treatmentMonitoring.loggingTip')}</Text> {t('help.treatmentMonitoring.loggingTipText')}
            </Text>
          </View>

          {/* Effectiveness Assessments */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.treatmentMonitoring.effectiveness')}</Text>
            <Text style={styles.text}>
              {t('help.treatmentMonitoring.effectivenessIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>1.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.effectStep1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>2.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.effectStep2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>3.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.effectStep3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>4.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.effectStep4')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>5.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.effectStep5')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>6.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.effectStep6')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>7.</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.effectStep7')}</Text>
              </View>
            </View>
          </View>

          {/* Understanding Effectiveness Scores */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.treatmentMonitoring.understandingScores')}</Text>
            <Text style={styles.text}>
              {t('help.treatmentMonitoring.scoresIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.successIcon}>🟢</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.score75100')}</Text> {t('help.treatmentMonitoring.score75100Text')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>🔵</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.score5074')}</Text> {t('help.treatmentMonitoring.score5074Text')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.warningIcon}>🟡</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.score2549')}</Text> {t('help.treatmentMonitoring.score2549Text')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.score024')}</Text> {t('help.treatmentMonitoring.score024Text')}</Text>
              </View>
            </View>
            <Text style={styles.text} style={{ marginTop: 12 }}>
              {'\n'}
              {t('help.treatmentMonitoring.scoreConsiders')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.sizeChange')}</Text> {t('help.treatmentMonitoring.sizeChangeText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.colorChange')}</Text> {t('help.treatmentMonitoring.colorChangeText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.inflammation')}</Text> {t('help.treatmentMonitoring.inflammationText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.symptoms')}</Text> {t('help.treatmentMonitoring.symptomsText')}</Text>
              </View>
            </View>
          </View>

          {/* Adherence Tracking */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.treatmentMonitoring.adherence')}</Text>
            <Text style={styles.text}>
              {t('help.treatmentMonitoring.adherenceIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.successIcon}>✓</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.adher90100')}</Text> {t('help.treatmentMonitoring.adher90100Text')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>~</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.adher7089')}</Text> {t('help.treatmentMonitoring.adher7089Text')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.warningIcon}>!</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.adher5069')}</Text> {t('help.treatmentMonitoring.adher5069Text')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>✗</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.adherBelow50')}</Text> {t('help.treatmentMonitoring.adherBelow50Text')}</Text>
              </View>
            </View>
            <Text style={styles.text} style={{ marginTop: 12 }}>
              {'\n'}
              <Text style={styles.bold}>{t('help.treatmentMonitoring.adherenceTip')}</Text> {t('help.treatmentMonitoring.adherenceTipText')}
            </Text>
          </View>

          {/* When to Adjust Treatment */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.treatmentMonitoring.whenAdjust')}</Text>
            <Text style={styles.text}>
              {t('help.treatmentMonitoring.adjustIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>⚠️</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.scoreBelow25')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>⚠️</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.increasingSize')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>⚠️</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.severeSide')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>⚠️</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.highAdherNoImprove')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>⚠️</Text>
                <Text style={styles.bulletText}>{t('help.treatmentMonitoring.newSymptoms')}</Text>
              </View>
            </View>
            <Text style={styles.text} style={{ marginTop: 12 }}>
              {'\n'}
              <Text style={styles.bold}>{t('help.treatmentMonitoring.adjustFooter')}</Text>
            </Text>
          </View>

          {/* Best Practices */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.treatmentMonitoring.bestPractices')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>1.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.docBaseline')}</Text> {t('help.treatmentMonitoring.docBaselineText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>2.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.logConsistent')}</Text> {t('help.treatmentMonitoring.logConsistentText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>3.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.assessRegular')}</Text> {t('help.treatmentMonitoring.assessRegularText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>4.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.sameConditions')}</Text> {t('help.treatmentMonitoring.sameConditionsText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>5.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.noteReactions')}</Text> {t('help.treatmentMonitoring.noteReactionsText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>6.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.bePatient')}</Text> {t('help.treatmentMonitoring.bePatientText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>7.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.treatmentMonitoring.shareProvider')}</Text> {t('help.treatmentMonitoring.shareProviderText')}</Text>
              </View>
            </View>
          </View>
        </View>

        {/* Dermatologist Integration */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>👨‍⚕️ {t('help.dermatologistIntegration.title')}</Text>

          {/* Overview */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.dermatologistIntegration.professionalCare')}</Text>
            <Text style={styles.text}>
              {t('help.dermatologistIntegration.careIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.videoConsults')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.formalReferrals')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.secondOpinions')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.accessDirectory')}</Text>
              </View>
            </View>
          </View>

          {/* Finding a Dermatologist */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.dermatologistIntegration.findingDerma')}</Text>
            <Text style={styles.text}>
              {t('help.dermatologistIntegration.findIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>1.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.findStep1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>2.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.findStep2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>3.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.findStep3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>4.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.findStep4')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>5.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.findStep5')}</Text>
              </View>
            </View>
            <Text style={styles.text} style={{ marginTop: 12 }}>
              {'\n'}
              {t('help.dermatologistIntegration.verifiedBadge')}
            </Text>
          </View>

          {/* Video Consultations */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>📹 {t('help.dermatologistIntegration.videoTitle')}</Text>
            <Text style={styles.text}>
              {t('help.dermatologistIntegration.videoIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>1.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.videoStep1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>2.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.videoStep2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>3.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.videoStep3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>4.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.videoStep4')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>5.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.videoStep5')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>6.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.videoStep6')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>7.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.videoStep7')}</Text>
              </View>
            </View>
            <Text style={styles.text} style={{ marginTop: 12 }}>
              {'\n'}
              <Text style={styles.bold}>{t('help.dermatologistIntegration.duringConsult')}</Text>
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.viewAppointment')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.joinCall')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.videoLink')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.dermaAdds')}</Text>
              </View>
            </View>
          </View>

          {/* Referrals */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>📋 {t('help.dermatologistIntegration.referralsTitle')}</Text>
            <Text style={styles.text}>
              {t('help.dermatologistIntegration.referralsIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>1.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.refStep1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>2.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.refStep2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>3.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.refStep3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>4.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.refStep4')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>5.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.refStep5')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>6.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.refStep6')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>7.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.refStep7')}</Text>
              </View>
            </View>
            <Text style={styles.text} style={{ marginTop: 12 }}>
              {'\n'}
              <Text style={styles.bold}>{t('help.dermatologistIntegration.referralStatuses')}</Text>
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>⏳</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.pending')}</Text> {t('help.dermatologistIntegration.pendingText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.successIcon}>✓</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.accepted')}</Text> {t('help.dermatologistIntegration.acceptedText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>📅</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.scheduled')}</Text> {t('help.dermatologistIntegration.scheduledText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.successIcon}>✓</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.seen')}</Text> {t('help.dermatologistIntegration.seenText')}</Text>
              </View>
            </View>
          </View>

          {/* Second Opinions */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>💭 {t('help.dermatologistIntegration.secondOpinionTitle')}</Text>
            <Text style={styles.text}>
              {t('help.dermatologistIntegration.secondOpinionIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>1.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.opinionStep1')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>2.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.opinionStep2')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>3.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.opinionStep3')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>4.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.opinionStep4')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>5.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.opinionStep5')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>6.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.opinionStep6')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>7.</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.opinionStep7')}</Text>
              </View>
            </View>
            <Text style={styles.text} style={{ marginTop: 12 }}>
              {'\n'}
              <Text style={styles.bold}>{t('help.dermatologistIntegration.whenSeekOpinion')}</Text>
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.warningIcon}>⚠️</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.cancerDiagnosis')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.warningIcon}>⚠️</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.aggressiveTreatment')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.warningIcon}>⚠️</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.notWorking')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.warningIcon}>⚠️</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.uncertainDiagnosis')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>ℹ️</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.peaceOfMind')}</Text>
              </View>
            </View>
          </View>

          {/* Urgency Levels */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.dermatologistIntegration.urgencyLevels')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.successIcon}>🟢</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.routineUrgency')}</Text> {t('help.dermatologistIntegration.routineUrgencyText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.warningIcon}>🟡</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.semiUrgent')}</Text> {t('help.dermatologistIntegration.semiUrgentText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.riskBullet}>🔴</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.urgent')}</Text> {t('help.dermatologistIntegration.urgentText')}</Text>
              </View>
            </View>
            <Text style={styles.text} style={{ marginTop: 12 }}>
              {'\n'}
              <Text style={styles.bold}>{t('help.dermatologistIntegration.emergencyNote')}</Text> {t('help.dermatologistIntegration.emergencyNoteText')}
            </Text>
          </View>

          {/* Specializations Explained */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.dermatologistIntegration.specializations')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>🔬</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.mohsSurgery')}</Text> {t('help.dermatologistIntegration.mohsSurgeryText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>💉</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.cosmetic')}</Text> {t('help.dermatologistIntegration.cosmeticText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>👶</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.pediatric')}</Text> {t('help.dermatologistIntegration.pediatricText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>🎗️</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.skinCancer')}</Text> {t('help.dermatologistIntegration.skinCancerText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>🔍</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.dermatopathology')}</Text> {t('help.dermatologistIntegration.dermatopathologyText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.infoIcon}>💊</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.medicalDerma')}</Text> {t('help.dermatologistIntegration.medicalDermaText')}</Text>
              </View>
            </View>
          </View>

          {/* Tips for Video Consultations */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.dermatologistIntegration.videoBestPractices')}</Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>1.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.prepareBeforehand')}</Text> {t('help.dermatologistIntegration.prepareBeforehandText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>2.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.goodLighting')}</Text> {t('help.dermatologistIntegration.goodLightingText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>3.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.quietSpace')}</Text> {t('help.dermatologistIntegration.quietSpaceText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>4.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.stableInternet')}</Text> {t('help.dermatologistIntegration.stableInternetText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>5.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.cameraReady')}</Text> {t('help.dermatologistIntegration.cameraReadyText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>6.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.documentsReady')}</Text> {t('help.dermatologistIntegration.documentsReadyText')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.stepNumber}>7.</Text>
                <Text style={styles.bulletText}><Text style={styles.bold}>{t('help.dermatologistIntegration.joinEarly')}</Text> {t('help.dermatologistIntegration.joinEarlyText')}</Text>
              </View>
            </View>
          </View>

          {/* Privacy & Security */}
          <View style={styles.subsection}>
            <Text style={styles.subsectionTitle}>{t('help.dermatologistIntegration.privacySecurity')}</Text>
            <Text style={styles.text}>
              {t('help.dermatologistIntegration.privacyIntro')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.successIcon}>🔒</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.hipaa')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.successIcon}>🔒</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.encrypted')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.successIcon}>🔒</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.protectedRecords')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.successIcon}>🔒</Text>
                <Text style={styles.bulletText}>{t('help.dermatologistIntegration.secureAuth')}</Text>
              </View>
            </View>
          </View>
        </View>

        {/* Important Notes */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>⚠️ {t('help.disclaimer.title')}</Text>
          <View style={styles.disclaimerBox}>
            <Text style={styles.disclaimerText}>
              <Text style={styles.bold}>{t('help.disclaimer.notSubstitute')}</Text>
              {'\n\n'}
              {t('help.disclaimer.educationalOnly')}
            </Text>
            <View style={styles.bulletList}>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.disclaimerText}>{t('help.disclaimer.definitiveDiagnosis')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.disclaimerText}>{t('help.disclaimer.changingLesions')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.disclaimerText}>{t('help.disclaimer.highRiskResults')}</Text>
              </View>
              <View style={styles.bulletItem}>
                <Text style={styles.bullet}>•</Text>
                <Text style={styles.disclaimerText}>{t('help.disclaimer.regularScreenings')}</Text>
              </View>
            </View>
            <Text style={styles.disclaimerText}>
              {'\n'}
              {t('help.disclaimer.aiMistakes')}
            </Text>
          </View>
        </View>

        {/* Tips */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>💡 {t('help.proTips.title')}</Text>
          <View style={styles.bulletList}>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                <Text style={styles.bold}>{t('help.proTips.trackChanges')}</Text> {t('help.proTips.trackChangesText')}
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                <Text style={styles.bold}>{t('help.proTips.completeHistory')}</Text> {t('help.proTips.completeHistoryText')}
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                <Text style={styles.bold}>{t('help.proTips.savePDFs')}</Text> {t('help.proTips.savePDFsText')}
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                <Text style={styles.bold}>{t('help.proTips.useQuality')}</Text> {t('help.proTips.useQualityText')}
              </Text>
            </View>
          </View>
        </View>

        {/* Conditions */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🔬 {t('help.conditions.title')}</Text>
          <Text style={styles.text}>{t('help.conditions.intro')}</Text>

          <View style={styles.conditionsGrid}>
            <Text style={styles.conditionItem}>• {t('help.conditions.melanoma')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.basal')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.squamous')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.actinic')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.nevus')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.benign')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.dermatofibroma')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.vascular')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.eczema')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.psoriasis')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.acne')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.rosacea')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.seborrheic')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.contact')}</Text>
            <Text style={styles.conditionItem}>• {t('help.conditions.more')}</Text>
          </View>
        </View>

        {/* Regulatory Compliance */}
        <Pressable
          style={styles.regulatoryButton}
          onPress={() => router.push('/regulatory')}
        >
          <View style={styles.regulatoryContent}>
            <View style={styles.regulatoryIcon}>
              <Text style={styles.regulatoryIconText}>🏛️</Text>
            </View>
            <View style={styles.regulatoryTextContent}>
              <Text style={styles.regulatoryTitle}>{t('help.links.regulatoryTitle')}</Text>
              <Text style={styles.regulatoryDescription}>
                {t('help.links.regulatoryDescription')}
              </Text>
            </View>
            <Text style={styles.regulatoryArrow}>→</Text>
          </View>
        </Pressable>

        {/* Audit Trail */}
        <Pressable
          style={styles.regulatoryButton}
          onPress={() => router.push('/audit')}
        >
          <View style={styles.regulatoryContent}>
            <View style={styles.regulatoryIcon}>
              <Text style={styles.regulatoryIconText}>📋</Text>
            </View>
            <View style={styles.regulatoryTextContent}>
              <Text style={styles.regulatoryTitle}>{t('help.links.auditTitle')}</Text>
              <Text style={styles.regulatoryDescription}>
                {t('help.links.auditDescription')}
              </Text>
            </View>
            <Text style={styles.regulatoryArrow}>→</Text>
          </View>
        </Pressable>

        {/* Dermoscopy Mode */}
        <Pressable
          style={styles.regulatoryButton}
          onPress={() => router.push('/dermoscopy')}
        >
          <View style={styles.regulatoryContent}>
            <View style={styles.regulatoryIcon}>
              <Text style={styles.regulatoryIconText}>🔬</Text>
            </View>
            <View style={styles.regulatoryTextContent}>
              <Text style={styles.regulatoryTitle}>{t('help.links.dermoscopyTitle')}</Text>
              <Text style={styles.regulatoryDescription}>
                {t('help.links.dermoscopyDescription')}
              </Text>
            </View>
            <Text style={styles.regulatoryArrow}>→</Text>
          </View>
        </Pressable>

        {/* Support */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📞 {t('help.needHelp.title')}</Text>
          <Text style={styles.text}>
            {t('help.needHelp.intro')}
          </Text>
          <View style={styles.bulletList}>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                {t('help.needHelp.checkPermissions')}
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                {t('help.needHelp.stableConnection')}
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                {t('help.needHelp.restart')}
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                {t('help.needHelp.consultProvider')}
              </Text>
            </View>
          </View>
        </View>

        {/* Privacy */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🔒 {t('help.privacySecurity.title')}</Text>
          <View style={styles.bulletList}>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                {t('help.privacySecurity.encrypted')}
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                {t('help.privacySecurity.onlyYou')}
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                {t('help.privacySecurity.secureLinks')}
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.bulletText}>
                {t('help.privacySecurity.canDelete')}
              </Text>
            </View>
          </View>
        </View>

        <View style={styles.footer}>
          <Text style={styles.footerText}>
            {t('help.footer.thankYou')}{'\n'}
            {t('help.footer.stayVigilant')}
          </Text>
        </View>
      </ScrollView>

      <View style={styles.buttonContainer}>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Text style={styles.backButtonText}>← {t('common.backToApp')}</Text>
        </Pressable>
      </View>
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
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingTop: 50,
    paddingBottom: 100,
  },
  header: {
    marginBottom: 30,
    alignItems: 'center',
  },
  appIcon: {
    width: 80,
    height: 80,
    borderRadius: 16,
    marginBottom: 16,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#4a5568',
    textAlign: 'center',
  },
  section: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
  },
  subsection: {
    marginTop: 16,
  },
  subsectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#4a5568',
    marginBottom: 8,
  },
  text: {
    fontSize: 15,
    color: '#2d3748',
    lineHeight: 22,
  },
  bold: {
    fontWeight: '600',
    color: '#1a202c',
  },
  bulletList: {
    marginTop: 12,
  },
  bulletItem: {
    flexDirection: 'row',
    marginBottom: 10,
    paddingLeft: 8,
  },
  bullet: {
    fontSize: 15,
    color: '#4299e1',
    marginRight: 8,
    fontWeight: 'bold',
  },
  confidenceBullet: {
    fontSize: 18,
    marginRight: 8,
  },
  riskBullet: {
    fontSize: 18,
    marginRight: 8,
  },
  bulletText: {
    fontSize: 15,
    color: '#2d3748',
    lineHeight: 22,
    flex: 1,
  },
  numberList: {
    marginTop: 12,
  },
  numberItem: {
    flexDirection: 'row',
    marginBottom: 12,
    paddingLeft: 8,
  },
  numberBullet: {
    fontSize: 15,
    color: '#4299e1',
    marginRight: 8,
    fontWeight: 'bold',
    minWidth: 20,
  },
  numberText: {
    flex: 1,
    fontSize: 14,
    color: '#2d3748',
    lineHeight: 20,
  },
  stepNumber: {
    fontSize: 15,
    color: '#4299e1',
    marginRight: 8,
    fontWeight: 'bold',
    minWidth: 20,
  },
  infoIcon: {
    fontSize: 16,
    marginRight: 8,
  },
  successIcon: {
    fontSize: 16,
    marginRight: 8,
  },
  warningIcon: {
    fontSize: 16,
    marginRight: 8,
  },
  importantNote: {
    backgroundColor: '#fef3c7',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  importantTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#92400e',
    marginBottom: 12,
  },
  number: {
    fontSize: 15,
    color: '#4299e1',
    marginRight: 8,
    fontWeight: 'bold',
    minWidth: 20,
  },
  disclaimerBox: {
    backgroundColor: '#fef5e7',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#d69e2e',
    marginTop: 12,
  },
  disclaimerText: {
    fontSize: 14,
    color: '#744210',
    lineHeight: 20,
  },
  conditionsGrid: {
    marginTop: 12,
  },
  conditionItem: {
    fontSize: 14,
    color: '#2d3748',
    lineHeight: 24,
    paddingLeft: 8,
  },
  footer: {
    marginTop: 20,
    padding: 20,
    backgroundColor: 'rgba(66, 153, 225, 0.1)',
    borderRadius: 12,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 15,
    color: '#2c5282',
    textAlign: 'center',
    lineHeight: 22,
  },
  buttonContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#fff',
    padding: 20,
    paddingBottom: 30,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 5,
  },
  backButton: {
    backgroundColor: '#4299e1',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  backButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  regulatoryButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 16,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    borderLeftWidth: 4,
    borderLeftColor: '#4299e1',
  },
  regulatoryContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
  },
  regulatoryIcon: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#ebf8ff',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  regulatoryIconText: {
    fontSize: 28,
  },
  regulatoryTextContent: {
    flex: 1,
  },
  regulatoryTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 4,
  },
  regulatoryDescription: {
    fontSize: 13,
    color: '#64748b',
    lineHeight: 18,
  },
  regulatoryArrow: {
    fontSize: 24,
    color: '#4299e1',
    marginLeft: 8,
  },
});
