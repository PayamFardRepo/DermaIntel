import i18n from '../i18n';

/**
 * Format a date according to the current language setting
 * @param date - Date object or ISO string
 * @param format - 'short' | 'long' | 'relative' | 'time' | 'dateTime'
 * @returns Formatted date string
 */
export const formatDate = (
  date: Date | string,
  format: 'short' | 'long' | 'relative' | 'time' | 'dateTime' = 'short'
): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const currentLang = i18n.language || 'en';

  if (isNaN(dateObj.getTime())) {
    return 'Invalid date';
  }

  switch (format) {
    case 'relative':
      return formatRelativeTime(dateObj);

    case 'short':
      return new Intl.DateTimeFormat(currentLang, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      }).format(dateObj);

    case 'long':
      return new Intl.DateTimeFormat(currentLang, {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        weekday: 'long',
      }).format(dateObj);

    case 'time':
      return new Intl.DateTimeFormat(currentLang, {
        hour: '2-digit',
        minute: '2-digit',
      }).format(dateObj);

    case 'dateTime':
      return new Intl.DateTimeFormat(currentLang, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }).format(dateObj);

    default:
      return dateObj.toLocaleDateString(currentLang);
  }
};

/**
 * Format a date as relative time (e.g., "2 hours ago", "Yesterday")
 * @param date - Date object or ISO string
 * @returns Relative time string
 */
export const formatRelativeTime = (date: Date | string): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diffInMs = now.getTime() - dateObj.getTime();
  const diffInMinutes = Math.floor(diffInMs / (1000 * 60));
  const diffInHours = Math.floor(diffInMs / (1000 * 60 * 60));
  const diffInDays = Math.floor(diffInMs / (1000 * 60 * 60 * 24));
  const diffInWeeks = Math.floor(diffInDays / 7);
  const diffInMonths = Math.floor(diffInDays / 30);
  const diffInYears = Math.floor(diffInDays / 365);

  // Just now (less than 1 minute)
  if (diffInMinutes < 1) {
    return i18n.t('dateTime.justNow');
  }

  // Minutes ago
  if (diffInMinutes < 60) {
    return i18n.t('dateTime.minutesAgo', { count: diffInMinutes });
  }

  // Hours ago
  if (diffInHours < 24) {
    return i18n.t('dateTime.hoursAgo', { count: diffInHours });
  }

  // Today/Yesterday
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const dateObjMidnight = new Date(dateObj);
  dateObjMidnight.setHours(0, 0, 0, 0);

  if (dateObjMidnight.getTime() === today.getTime()) {
    return i18n.t('dateTime.today');
  }

  if (dateObjMidnight.getTime() === yesterday.getTime()) {
    return i18n.t('dateTime.yesterday');
  }

  // Days ago (within a week)
  if (diffInDays < 7) {
    return i18n.t('dateTime.daysAgo', { count: diffInDays });
  }

  // Weeks ago
  if (diffInWeeks < 4) {
    return i18n.t('dateTime.weeksAgo', { count: diffInWeeks });
  }

  // Months ago
  if (diffInMonths < 12) {
    return i18n.t('dateTime.monthsAgo', { count: diffInMonths });
  }

  // Years ago
  return i18n.t('dateTime.yearsAgo', { count: diffInYears });
};

/**
 * Format a time duration in seconds to a human-readable string
 * @param seconds - Duration in seconds
 * @returns Formatted duration string
 */
export const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
};

/**
 * Get localized month names
 * @param format - 'long' | 'short'
 * @returns Array of month names
 */
export const getMonthNames = (format: 'long' | 'short' = 'long'): string[] => {
  const currentLang = i18n.language || 'en';
  const months: string[] = [];

  for (let i = 0; i < 12; i++) {
    const date = new Date(2024, i, 1);
    months.push(
      new Intl.DateTimeFormat(currentLang, { month: format }).format(date)
    );
  }

  return months;
};

/**
 * Get localized day names
 * @param format - 'long' | 'short' | 'narrow'
 * @returns Array of day names
 */
export const getDayNames = (
  format: 'long' | 'short' | 'narrow' = 'long'
): string[] => {
  const currentLang = i18n.language || 'en';
  const days: string[] = [];

  // Start with Sunday (day 0)
  for (let i = 0; i < 7; i++) {
    const date = new Date(2024, 0, i); // January 2024 starts on Monday
    days.push(
      new Intl.DateTimeFormat(currentLang, { weekday: format }).format(date)
    );
  }

  return days;
};

/**
 * Format a number according to the current language
 * @param number - Number to format
 * @param options - Intl.NumberFormatOptions
 * @returns Formatted number string
 */
export const formatNumber = (
  number: number,
  options?: Intl.NumberFormatOptions
): string => {
  const currentLang = i18n.language || 'en';
  return new Intl.NumberFormat(currentLang, options).format(number);
};

/**
 * Format a percentage according to the current language
 * @param value - Value between 0 and 1
 * @param decimals - Number of decimal places (default: 1)
 * @returns Formatted percentage string
 */
export const formatPercentage = (value: number, decimals: number = 1): string => {
  const currentLang = i18n.language || 'en';
  return new Intl.NumberFormat(currentLang, {
    style: 'percent',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

export default {
  formatDate,
  formatRelativeTime,
  formatDuration,
  getMonthNames,
  getDayNames,
  formatNumber,
  formatPercentage,
};
