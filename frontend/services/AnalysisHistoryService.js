import { API_ENDPOINTS, REQUEST_TIMEOUT } from '../config';
import AuthService from './AuthService';

class AnalysisHistoryService {
  /**
   * Get user's analysis history with pagination
   * @param {number} skip - Number of records to skip (for pagination)
   * @param {number} limit - Number of records to return
   * @returns {Promise<Array>} Array of analysis records
   */
  async getAnalysisHistory(skip = 0, limit = 20) {
    try {
      const token = AuthService.getToken();
      if (!token) {
        throw new Error('Authentication required');
      }

      const url = new URL(API_ENDPOINTS.ANALYSIS_HISTORY);
      url.searchParams.append('skip', skip.toString());
      url.searchParams.append('limit', limit.toString());

      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        timeout: REQUEST_TIMEOUT,
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Failed to fetch analysis history: ${errorData}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Analysis history fetch error:', error);
      throw error;
    }
  }

  /**
   * Get specific analysis by ID
   * @param {number} analysisId - The analysis ID to fetch
   * @returns {Promise<Object>} Analysis record
   */
  async getAnalysisById(analysisId) {
    try {
      const token = AuthService.getToken();
      if (!token) {
        throw new Error('Authentication required');
      }

      const response = await fetch(`${API_ENDPOINTS.ANALYSIS_HISTORY}/${analysisId}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        timeout: REQUEST_TIMEOUT,
      });

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Analysis not found');
        }
        const errorData = await response.text();
        throw new Error(`Failed to fetch analysis: ${errorData}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Analysis fetch error:', error);
      throw error;
    }
  }

  /**
   * Get user's analysis statistics
   * @returns {Promise<Object>} Analysis statistics
   */
  async getAnalysisStatistics() {
    try {
      const token = AuthService.getToken();
      if (!token) {
        throw new Error('Authentication required');
      }

      const response = await fetch(API_ENDPOINTS.ANALYSIS_STATS, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        timeout: REQUEST_TIMEOUT,
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Failed to fetch analysis statistics: ${errorData}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Analysis statistics fetch error:', error);
      throw error;
    }
  }

  /**
   * Get extended user information including profile and stats
   * @returns {Promise<Object>} Extended user information
   */
  async getExtendedUserInfo() {
    try {
      const token = AuthService.getToken();
      if (!token) {
        throw new Error('Authentication required');
      }

      const response = await fetch(API_ENDPOINTS.USER_EXTENDED, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        timeout: REQUEST_TIMEOUT,
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Failed to fetch extended user info: ${errorData}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Extended user info fetch error:', error);
      throw error;
    }
  }

  /**
   * Format analysis result for display
   * @param {Object} analysis - Raw analysis data from API
   * @returns {Object} Formatted analysis data
   */
  formatAnalysisForDisplay(analysis) {
    const formatDate = (dateString) => {
      // The backend sends UTC timestamps without timezone info
      // We need to explicitly treat them as UTC if they don't already include timezone
      let date;
      if (dateString.includes('T') && !dateString.includes('Z') && !dateString.includes('+')) {
        // ISO format without timezone, treat as UTC
        date = new Date(dateString + 'Z');
      } else {
        // Already has timezone info or is in a different format
        date = new Date(dateString);
      }

      return {
        date: date.toLocaleDateString(),
        time: date.toLocaleTimeString(),
        relative: this.getRelativeTime(date)
      };
    };

    const getRiskColor = (riskLevel) => {
      switch (riskLevel?.toLowerCase()) {
        case 'high': return '#dc3545';
        case 'medium': return '#ffc107';
        case 'low': return '#28a745';
        default: return '#6c757d';
      }
    };

    const getConfidenceLevel = (confidence) => {
      if (confidence >= 0.9) return { level: 'Very High', color: '#28a745' };
      if (confidence >= 0.8) return { level: 'High', color: '#20c997' };
      if (confidence >= 0.7) return { level: 'Medium', color: '#ffc107' };
      if (confidence >= 0.6) return { level: 'Low', color: '#fd7e14' };
      return { level: 'Very Low', color: '#dc3545' };
    };

    // ALWAYS calculate confidence based on analysis type - don't use pre-calculated confidence field
    // For lesion analyses: use lesion_confidence (cancer type classification, e.g., 52% BCC)
    // For non-lesion analyses: use binary_confidence (lesion detection, e.g., 81% is lesion)
    let displayConfidence;
    if (analysis.is_lesion && analysis.lesion_confidence != null) {
      displayConfidence = analysis.lesion_confidence;
    } else if (analysis.binary_confidence != null) {
      displayConfidence = analysis.binary_confidence;
    } else {
      displayConfidence = analysis.lesion_confidence || 0;
    }

    return {
      id: analysis.id,
      type: analysis.analysis_type,
      isLesion: analysis.is_lesion,
      predictedClass: analysis.predicted_class,
      confidence: displayConfidence,
      confidenceLevel: getConfidenceLevel(displayConfidence || 0),
      riskLevel: analysis.risk_level,
      riskColor: getRiskColor(analysis.risk_level),
      recommendation: analysis.risk_recommendation,
      processingTime: analysis.processing_time_seconds,
      modelVersion: analysis.model_version,
      createdAt: formatDate(analysis.created_at),
      imageFilename: analysis.image_filename,
      probabilities: analysis.binary_probabilities || analysis.lesion_probabilities || {},
    };
  }

  /**
   * Get relative time string (e.g., "2 hours ago")
   * @param {Date} date - The date to compare
   * @returns {string} Relative time string
   */
  getRelativeTime(date) {
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);

    if (diffInSeconds < 60) return 'Just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
    if (diffInSeconds < 2592000) return `${Math.floor(diffInSeconds / 86400)} days ago`;
    if (diffInSeconds < 31536000) return `${Math.floor(diffInSeconds / 2592000)} months ago`;
    return `${Math.floor(diffInSeconds / 31536000)} years ago`;
  }

  /**
   * Get analysis summary statistics
   * @param {Array} analyses - Array of analysis records
   * @returns {Object} Summary statistics
   */
  getAnalysisSummary(analyses) {
    if (!analyses || analyses.length === 0) {
      return {
        totalAnalyses: 0,
        lesionDetections: 0,
        nonLesionDetections: 0,
        averageConfidence: 0,
        riskDistribution: {},
        recentActivity: []
      };
    }

    const lesionDetections = analyses.filter(a => a.is_lesion).length;
    const nonLesionDetections = analyses.length - lesionDetections;

    const confidences = analyses
      .map(a => a.binary_confidence || a.lesion_confidence)
      .filter(c => c !== null && c !== undefined);

    const averageConfidence = confidences.length > 0
      ? confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length
      : 0;

    const riskDistribution = analyses.reduce((acc, analysis) => {
      const risk = analysis.risk_level || 'unknown';
      acc[risk] = (acc[risk] || 0) + 1;
      return acc;
    }, {});

    return {
      totalAnalyses: analyses.length,
      lesionDetections,
      nonLesionDetections,
      averageConfidence: Math.round(averageConfidence * 100) / 100,
      riskDistribution,
      recentActivity: analyses.slice(0, 5).map(a => this.formatAnalysisForDisplay(a))
    };
  }

  /**
   * Get authentication token for API requests
   * @returns {string|null} Authentication token
   */
  async getAuthToken() {
    return AuthService.getToken();
  }
}

export default new AnalysisHistoryService();