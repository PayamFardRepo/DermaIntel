import React, { createContext, useContext, useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { I18nManager } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const LANGUAGE_KEY = '@language';
const RTL_KEY = '@rtl_enabled';

export type SupportedLanguage = 'en' | 'es' | 'fr' | 'de' | 'zh' | 'ja' | 'pt' | 'ar';

// RTL languages
const RTL_LANGUAGES: SupportedLanguage[] = ['ar'];

interface LanguageContextType {
  currentLanguage: SupportedLanguage;
  changeLanguage: (language: SupportedLanguage) => Promise<void>;
  isRTL: boolean;
  isRTLLanguage: (lang: SupportedLanguage) => boolean;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { i18n } = useTranslation();
  const [currentLanguage, setCurrentLanguage] = useState<SupportedLanguage>('en');
  const [isRTL, setIsRTL] = useState(false);

  // Check if a language requires RTL
  const isRTLLanguage = (lang: SupportedLanguage): boolean => {
    return RTL_LANGUAGES.includes(lang);
  };

  useEffect(() => {
    // Initialize language from storage
    const initLanguage = async () => {
      try {
        const savedLanguage = await AsyncStorage.getItem(LANGUAGE_KEY);
        const savedRTL = await AsyncStorage.getItem(RTL_KEY);

        if (savedLanguage) {
          const lang = savedLanguage as SupportedLanguage;
          const shouldBeRTL = isRTLLanguage(lang);

          setCurrentLanguage(lang);
          await i18n.changeLanguage(lang);
          setIsRTL(shouldBeRTL);

          // Update RTL setting for React Native
          if (I18nManager.isRTL !== shouldBeRTL) {
            I18nManager.allowRTL(shouldBeRTL);
            I18nManager.forceRTL(shouldBeRTL);
          }
        }
      } catch (error) {
        console.error('Error loading language:', error);
      }
    };
    initLanguage();
  }, []);

  const changeLanguage = async (language: SupportedLanguage) => {
    try {
      const shouldBeRTL = isRTLLanguage(language);
      const needsReload = I18nManager.isRTL !== shouldBeRTL;

      await i18n.changeLanguage(language);
      await AsyncStorage.setItem(LANGUAGE_KEY, language);
      await AsyncStorage.setItem(RTL_KEY, shouldBeRTL.toString());

      setCurrentLanguage(language);
      setIsRTL(shouldBeRTL);

      // Update RTL setting for React Native
      if (needsReload) {
        I18nManager.allowRTL(shouldBeRTL);
        I18nManager.forceRTL(shouldBeRTL);

        // Note: In a real app, you might need to reload the app for RTL changes to take full effect
        // This can be done with: RNRestart.Restart() or similar
        console.log('RTL setting changed. Some components may require app reload for full effect.');
      }
    } catch (error) {
      console.error('Error changing language:', error);
    }
  };

  return (
    <LanguageContext.Provider value={{ currentLanguage, changeLanguage, isRTL, isRTLLanguage }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};
