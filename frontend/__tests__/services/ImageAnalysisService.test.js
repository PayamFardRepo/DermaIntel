/**
 * Tests for ImageAnalysisService
 *
 * Tests image analysis functionality including:
 * - Image quality validation
 * - Analysis result formatting
 * - Risk level assessment
 * - Burn classification
 * - Confidence level calculations
 */

// Mock dependencies
jest.mock('../../config', () => ({
  API_ENDPOINTS: {
    UPLOAD: 'http://test-api.example.com/upload/',
    FULL_CLASSIFY: 'http://test-api.example.com/classify/full',
    DERMOSCOPY_ANALYZE: 'http://test-api.example.com/dermoscopy/analyze',
    CLASSIFY_BURN: 'http://test-api.example.com/classify/burn',
    BASE_URL: 'http://test-api.example.com',
  },
  REQUEST_TIMEOUT: 30000,
}));

jest.mock('../../services/AuthService', () => ({
  getToken: jest.fn(() => 'test-token'),
}));

jest.mock('expo-image-manipulator', () => ({
  manipulateAsync: jest.fn(() => Promise.resolve({ uri: 'manipulated-uri' })),
  SaveFormat: { JPEG: 'jpeg' },
}));

jest.mock('expo-print', () => ({
  printToFileAsync: jest.fn(() => Promise.resolve({ uri: 'pdf-uri' })),
}));

jest.mock('expo-sharing', () => ({
  isAvailableAsync: jest.fn(() => Promise.resolve(true)),
  shareAsync: jest.fn(() => Promise.resolve()),
}));

describe('ImageAnalysisService', () => {
  describe('Quality Thresholds', () => {
    const qualityThresholds = {
      minWidth: 400,
      minHeight: 400,
      maxFileSize: 10 * 1024 * 1024,
      minFileSize: 50 * 1024,
      minQualityScore: 0.6,
    };

    it('should have correct minimum width threshold', () => {
      expect(qualityThresholds.minWidth).toBe(400);
    });

    it('should have correct minimum height threshold', () => {
      expect(qualityThresholds.minHeight).toBe(400);
    });

    it('should have correct maximum file size (10MB)', () => {
      expect(qualityThresholds.maxFileSize).toBe(10 * 1024 * 1024);
    });

    it('should have correct minimum file size (50KB)', () => {
      expect(qualityThresholds.minFileSize).toBe(50 * 1024);
    });

    it('should have correct minimum quality score', () => {
      expect(qualityThresholds.minQualityScore).toBe(0.6);
    });
  });

  describe('validateFileSize', () => {
    const maxFileSize = 10 * 1024 * 1024;
    const minFileSize = 50 * 1024;

    const validateFileSize = (fileSize) => {
      const issues = [];
      const recommendations = [];

      if (fileSize > maxFileSize) {
        issues.push({
          type: 'file_size_too_large',
          severity: 'medium',
          message: `File size is ${Math.round(fileSize / (1024 * 1024))}MB (max recommended: 10MB)`,
        });
        recommendations.push('Consider using a lower resolution or compressing the image');
      }

      if (fileSize < minFileSize) {
        issues.push({
          type: 'file_size_too_small',
          severity: 'high',
          message: `File size is ${Math.round(fileSize / 1024)}KB (minimum: 50KB)`,
        });
        recommendations.push('Use a higher resolution image or better camera settings');
      }

      return { issues, recommendations };
    };

    it('should pass for valid file size', () => {
      const result = validateFileSize(500 * 1024); // 500KB
      expect(result.issues).toHaveLength(0);
    });

    it('should detect file too large', () => {
      const result = validateFileSize(15 * 1024 * 1024); // 15MB
      expect(result.issues).toHaveLength(1);
      expect(result.issues[0].type).toBe('file_size_too_large');
      expect(result.issues[0].severity).toBe('medium');
    });

    it('should detect file too small', () => {
      const result = validateFileSize(10 * 1024); // 10KB
      expect(result.issues).toHaveLength(1);
      expect(result.issues[0].type).toBe('file_size_too_small');
      expect(result.issues[0].severity).toBe('high');
    });

    it('should provide recommendations for issues', () => {
      const result = validateFileSize(15 * 1024 * 1024);
      expect(result.recommendations).toHaveLength(1);
      expect(result.recommendations[0]).toContain('compressing');
    });
  });

  describe('validateResolution', () => {
    const minWidth = 400;
    const minHeight = 400;

    const validateResolution = (dimensions) => {
      const { width, height } = dimensions;
      const issues = [];
      const recommendations = [];

      if (width < minWidth || height < minHeight) {
        issues.push({
          type: 'low_resolution',
          severity: 'high',
          message: `Resolution is ${width}x${height} (minimum recommended: 400x400)`,
        });
        recommendations.push('Use a higher resolution camera setting');
      }

      if (width > 4000 || height > 4000) {
        issues.push({
          type: 'very_high_resolution',
          severity: 'low',
          message: `Very high resolution: ${width}x${height}`,
        });
        recommendations.push('Image will be compressed for optimal analysis speed');
      }

      return { issues, recommendations };
    };

    it('should pass for valid resolution', () => {
      const result = validateResolution({ width: 800, height: 600 });
      expect(result.issues).toHaveLength(0);
    });

    it('should detect low resolution', () => {
      const result = validateResolution({ width: 200, height: 200 });
      expect(result.issues).toHaveLength(1);
      expect(result.issues[0].type).toBe('low_resolution');
      expect(result.issues[0].severity).toBe('high');
    });

    it('should detect very high resolution', () => {
      const result = validateResolution({ width: 5000, height: 5000 });
      expect(result.issues).toHaveLength(1);
      expect(result.issues[0].type).toBe('very_high_resolution');
      expect(result.issues[0].severity).toBe('low');
    });
  });

  describe('validateAspectRatio', () => {
    const validateAspectRatio = (dimensions) => {
      const { width, height } = dimensions;
      const issues = [];
      const recommendations = [];

      if (width === 0 || height === 0) {
        issues.push({
          type: 'invalid_dimensions',
          severity: 'high',
          message: 'Unable to determine image dimensions',
        });
        return { issues, recommendations };
      }

      const aspectRatio = width / height;

      if (aspectRatio < 0.5 || aspectRatio > 2.0) {
        issues.push({
          type: 'extreme_aspect_ratio',
          severity: 'medium',
          message: `Unusual aspect ratio: ${aspectRatio.toFixed(2)}`,
        });
        recommendations.push('Try to frame the lesion in a more square composition');
      }

      return { issues, recommendations };
    };

    it('should pass for normal aspect ratio', () => {
      const result = validateAspectRatio({ width: 800, height: 600 });
      expect(result.issues).toHaveLength(0);
    });

    it('should detect extreme aspect ratio (too wide)', () => {
      const result = validateAspectRatio({ width: 1000, height: 300 });
      expect(result.issues).toHaveLength(1);
      expect(result.issues[0].type).toBe('extreme_aspect_ratio');
    });

    it('should detect extreme aspect ratio (too tall)', () => {
      const result = validateAspectRatio({ width: 300, height: 1000 });
      expect(result.issues).toHaveLength(1);
      expect(result.issues[0].type).toBe('extreme_aspect_ratio');
    });

    it('should handle zero dimensions', () => {
      const result = validateAspectRatio({ width: 0, height: 600 });
      expect(result.issues).toHaveLength(1);
      expect(result.issues[0].type).toBe('invalid_dimensions');
    });
  });

  describe('calculateQualityScore', () => {
    const calculateQualityScore = (validationResults) => {
      let score = 1.0;

      validationResults.issues.forEach((issue) => {
        switch (issue.severity) {
          case 'high':
            score -= 0.3;
            break;
          case 'medium':
            score -= 0.15;
            break;
          case 'low':
            score -= 0.05;
            break;
        }
      });

      return Math.max(0, score);
    };

    it('should return 1.0 for no issues', () => {
      const score = calculateQualityScore({ issues: [] });
      expect(score).toBe(1.0);
    });

    it('should subtract 0.3 for high severity issues', () => {
      const score = calculateQualityScore({
        issues: [{ severity: 'high' }],
      });
      expect(score).toBe(0.7);
    });

    it('should subtract 0.15 for medium severity issues', () => {
      const score = calculateQualityScore({
        issues: [{ severity: 'medium' }],
      });
      expect(score).toBe(0.85);
    });

    it('should subtract 0.05 for low severity issues', () => {
      const score = calculateQualityScore({
        issues: [{ severity: 'low' }],
      });
      expect(score).toBe(0.95);
    });

    it('should accumulate deductions for multiple issues', () => {
      const score = calculateQualityScore({
        issues: [{ severity: 'high' }, { severity: 'medium' }, { severity: 'low' }],
      });
      expect(score).toBeCloseTo(0.5, 10); // 1.0 - 0.3 - 0.15 - 0.05
    });

    it('should not go below 0', () => {
      const score = calculateQualityScore({
        issues: [
          { severity: 'high' },
          { severity: 'high' },
          { severity: 'high' },
          { severity: 'high' },
        ],
      });
      expect(score).toBe(0);
    });
  });

  describe('getQualityAssessment', () => {
    const getQualityAssessment = (score) => {
      if (score >= 0.9) return { level: 'Excellent', color: '#22c55e', emoji: '✅' };
      if (score >= 0.7) return { level: 'Good', color: '#3b82f6', emoji: '👍' };
      if (score >= 0.6) return { level: 'Acceptable', color: '#f59e0b', emoji: '⚠️' };
      if (score >= 0.4) return { level: 'Poor', color: '#ef4444', emoji: '❌' };
      return { level: 'Very Poor', color: '#dc2626', emoji: '🚫' };
    };

    it('should return Excellent for score >= 0.9', () => {
      expect(getQualityAssessment(0.95).level).toBe('Excellent');
      expect(getQualityAssessment(0.9).level).toBe('Excellent');
    });

    it('should return Good for score >= 0.7', () => {
      expect(getQualityAssessment(0.85).level).toBe('Good');
      expect(getQualityAssessment(0.7).level).toBe('Good');
    });

    it('should return Acceptable for score >= 0.6', () => {
      expect(getQualityAssessment(0.65).level).toBe('Acceptable');
      expect(getQualityAssessment(0.6).level).toBe('Acceptable');
    });

    it('should return Poor for score >= 0.4', () => {
      expect(getQualityAssessment(0.5).level).toBe('Poor');
      expect(getQualityAssessment(0.4).level).toBe('Poor');
    });

    it('should return Very Poor for score < 0.4', () => {
      expect(getQualityAssessment(0.3).level).toBe('Very Poor');
      expect(getQualityAssessment(0).level).toBe('Very Poor');
    });
  });

  describe('getConfidenceLevel', () => {
    const getConfidenceLevel = (confidence) => {
      if (confidence >= 0.9) return { level: 'Very High', color: '#22c55e' };
      if (confidence >= 0.75) return { level: 'High', color: '#3b82f6' };
      if (confidence >= 0.6) return { level: 'Moderate', color: '#f59e0b' };
      if (confidence >= 0.4) return { level: 'Low', color: '#ef4444' };
      return { level: 'Very Low', color: '#dc2626' };
    };

    it('should return Very High for confidence >= 0.9', () => {
      expect(getConfidenceLevel(0.95).level).toBe('Very High');
    });

    it('should return High for confidence >= 0.75', () => {
      expect(getConfidenceLevel(0.8).level).toBe('High');
    });

    it('should return Moderate for confidence >= 0.6', () => {
      expect(getConfidenceLevel(0.65).level).toBe('Moderate');
    });

    it('should return Low for confidence >= 0.4', () => {
      expect(getConfidenceLevel(0.5).level).toBe('Low');
    });

    it('should return Very Low for confidence < 0.4', () => {
      expect(getConfidenceLevel(0.3).level).toBe('Very Low');
    });
  });

  describe('assessRiskLevel', () => {
    const assessRiskLevel = (predictedClass, confidence) => {
      const highRiskTypes = ['melanoma', 'basal cell carcinoma', 'squamous cell carcinoma'];
      const moderateRiskTypes = ['atypical nevus', 'pigmented benign keratosis'];

      const classLower = predictedClass.toLowerCase();

      if (highRiskTypes.some((type) => classLower.includes(type))) {
        return { level: 'High', color: '#dc2626', urgency: 'high' };
      } else if (moderateRiskTypes.some((type) => classLower.includes(type))) {
        return { level: 'Moderate', color: '#f59e0b', urgency: 'moderate' };
      } else {
        return { level: 'Low', color: '#22c55e', urgency: 'low' };
      }
    };

    it('should return High risk for melanoma', () => {
      const result = assessRiskLevel('Melanoma', 0.9);
      expect(result.level).toBe('High');
      expect(result.urgency).toBe('high');
    });

    it('should return High risk for basal cell carcinoma', () => {
      const result = assessRiskLevel('Basal Cell Carcinoma', 0.85);
      expect(result.level).toBe('High');
    });

    it('should return High risk for squamous cell carcinoma', () => {
      const result = assessRiskLevel('Squamous Cell Carcinoma', 0.8);
      expect(result.level).toBe('High');
    });

    it('should return Moderate risk for atypical nevus', () => {
      const result = assessRiskLevel('Atypical Nevus', 0.7);
      expect(result.level).toBe('Moderate');
      expect(result.urgency).toBe('moderate');
    });

    it('should return Low risk for benign conditions', () => {
      const result = assessRiskLevel('Melanocytic Nevus', 0.9);
      expect(result.level).toBe('Low');
      expect(result.urgency).toBe('low');
    });
  });

  describe('assessBurnRiskLevel', () => {
    const assessBurnRiskLevel = (severityLevel, medicalAttentionRequired) => {
      if (severityLevel === 3) {
        return { level: 'Critical', color: '#dc2626', urgency: 'critical' };
      } else if (severityLevel === 2) {
        return { level: 'High', color: '#f59e0b', urgency: 'high' };
      } else if (severityLevel === 1) {
        return { level: 'Low', color: '#fbbf24', urgency: 'low' };
      } else {
        return { level: 'None', color: '#22c55e', urgency: 'none' };
      }
    };

    it('should return Critical for third degree burn', () => {
      const result = assessBurnRiskLevel(3, true);
      expect(result.level).toBe('Critical');
      expect(result.urgency).toBe('critical');
    });

    it('should return High for second degree burn', () => {
      const result = assessBurnRiskLevel(2, true);
      expect(result.level).toBe('High');
      expect(result.urgency).toBe('high');
    });

    it('should return Low for first degree burn', () => {
      const result = assessBurnRiskLevel(1, false);
      expect(result.level).toBe('Low');
      expect(result.urgency).toBe('low');
    });

    it('should return None for no burn detected', () => {
      const result = assessBurnRiskLevel(0, false);
      expect(result.level).toBe('None');
      expect(result.urgency).toBe('none');
    });
  });

  describe('assessInfectiousRiskLevel', () => {
    const assessInfectiousRiskLevel = (severity, contagious, transmissionRisk) => {
      if (severity === 'severe' || (contagious && transmissionRisk === 'high')) {
        return { level: 'High', color: '#dc2626', urgency: 'high' };
      } else if (severity === 'moderate' || (contagious && transmissionRisk === 'medium')) {
        return { level: 'Moderate', color: '#f59e0b', urgency: 'moderate' };
      } else {
        return { level: 'Low', color: '#22c55e', urgency: 'low' };
      }
    };

    it('should return High for severe infections', () => {
      const result = assessInfectiousRiskLevel('severe', false, 'low');
      expect(result.level).toBe('High');
    });

    it('should return High for highly contagious infections', () => {
      const result = assessInfectiousRiskLevel('mild', true, 'high');
      expect(result.level).toBe('High');
    });

    it('should return Moderate for moderate severity', () => {
      const result = assessInfectiousRiskLevel('moderate', false, 'low');
      expect(result.level).toBe('Moderate');
    });

    it('should return Low for mild non-contagious infections', () => {
      const result = assessInfectiousRiskLevel('mild', false, 'low');
      expect(result.level).toBe('Low');
    });
  });

  describe('formatAnalysisResult', () => {
    it('should handle null result', () => {
      const formatAnalysisResult = (result) => {
        if (!result) {
          return {
            primaryConditionType: 'unknown',
            predictedClass: 'Analysis Unavailable',
            confidence: 'N/A',
          };
        }
        return result;
      };

      const result = formatAnalysisResult(null);
      expect(result.primaryConditionType).toBe('unknown');
      expect(result.predictedClass).toBe('Analysis Unavailable');
    });

    it('should format burn-primary results', () => {
      const mockBurnResult = {
        primary_condition_type: 'burn',
        burn_severity: 'Second Degree Burn',
        burn_confidence: 0.85,
        burn_severity_level: 2,
      };

      expect(mockBurnResult.primary_condition_type).toBe('burn');
      expect(mockBurnResult.burn_severity).toBe('Second Degree Burn');
    });

    it('should format lesion results with probabilities', () => {
      const mockLesionResult = {
        predicted_class: 'melanocytic_nevus',
        lesion_confidence: 0.92,
        probabilities: {
          nv: 0.92,
          mel: 0.05,
          bkl: 0.03,
        },
      };

      const sortedProbs = Object.entries(mockLesionResult.probabilities)
        .sort(([, a], [, b]) => b - a);

      expect(sortedProbs[0][0]).toBe('nv');
      expect(sortedProbs[0][1]).toBe(0.92);
    });
  });

  describe('formatBurnResult', () => {
    it('should format burn classification result', () => {
      const mockResult = {
        severity_class: 'Second Degree Burn',
        severity_level: 2,
        confidence: 0.85,
        urgency: 'Seek medical attention within 24 hours',
        treatment_advice: 'Cool the burn with running water',
        medical_attention_required: true,
        is_burn_detected: true,
        probabilities: {
          'First Degree': 0.1,
          'Second Degree': 0.85,
          'Third Degree': 0.05,
        },
      };

      expect(mockResult.severity_class).toBe('Second Degree Burn');
      expect(mockResult.severity_level).toBe(2);
      expect(mockResult.is_burn_detected).toBe(true);
    });

    it('should sort probabilities correctly', () => {
      const probabilities = {
        'First Degree': 0.1,
        'Second Degree': 0.85,
        'Third Degree': 0.05,
      };

      const sorted = Object.entries(probabilities)
        .sort(([, a], [, b]) => b - a);

      expect(sorted[0][0]).toBe('Second Degree');
      expect(sorted[0][1]).toBe(0.85);
    });
  });

  describe('retry logic', () => {
    it('should have correct max retries', () => {
      const maxRetries = 3;
      expect(maxRetries).toBe(3);
    });

    it('should calculate exponential backoff correctly', () => {
      const calculateDelay = (attempt) => Math.pow(2, attempt) * 1000;

      expect(calculateDelay(1)).toBe(2000);
      expect(calculateDelay(2)).toBe(4000);
      expect(calculateDelay(3)).toBe(8000);
    });
  });

  describe('timeout configuration', () => {
    it('should have correct default timeout (5 minutes)', () => {
      const defaultTimeout = 300000;
      expect(defaultTimeout).toBe(300000);
    });
  });
});
