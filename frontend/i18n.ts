import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Import English translations synchronously to ensure they're available immediately
import enTranslations from './locales/en.json';

const LANGUAGE_KEY = '@language';

// Language detection plugin
const languageDetector = {
  type: 'languageDetector' as const,
  async: true,
  detect: async (callback: (lng: string) => void) => {
    try {
      const savedLanguage = await AsyncStorage.getItem(LANGUAGE_KEY);
      if (savedLanguage) {
        callback(savedLanguage);
      } else {
        callback('en');
      }
    } catch (error) {
      console.error('Error detecting language:', error);
      callback('en');
    }
  },
  init: () => {},
  cacheUserLanguage: async (language: string) => {
    try {
      await AsyncStorage.setItem(LANGUAGE_KEY, language);
    } catch (error) {
      console.error('Error caching language:', error);
    }
  },
};

// Translation loader mapping
const translationLoaders: Record<string, () => Promise<any>> = {
  en: () => import('./locales/en.json'),
  es: () => import('./locales/es.json'),
  fr: () => import('./locales/fr.json'),
  de: () => import('./locales/de.json'),
  zh: () => import('./locales/zh.json'),
  ja: () => import('./locales/ja.json'),
  pt: () => import('./locales/pt.json'),
  ar: () => import('./locales/ar.json'),
};

// Lazy load translations
const loadResources = async (language: string) => {
  try {
    const loader = translationLoaders[language];
    if (!loader) {
      console.error(`No loader found for language: ${language}`);
      return null;
    }
    const translations = await loader();
    return translations.default || translations;
  } catch (error) {
    console.error(`Failed to load ${language} translations:`, error);
    return null;
  }
};

i18n
  .use(languageDetector)
  .use(initReactI18next)
  .init({
    compatibilityJSON: 'v3',
    resources: {
      en: {
        translation: enTranslations,
      },
    },
    fallbackLng: 'en',
    supportedLngs: ['en', 'es', 'fr', 'de', 'zh', 'ja', 'pt', 'ar'],
    interpolation: {
      escapeValue: false,
    },
    react: {
      useSuspense: false,
    },
  });

// Add lazy loading for language changes
i18n.on('languageChanged', async (lng) => {
  if (!i18n.hasResourceBundle(lng, 'translation')) {
    const translations = await loadResources(lng);
    if (translations) {
      i18n.addResourceBundle(lng, 'translation', translations, true, true);
    }
  }
});

export default i18n;
