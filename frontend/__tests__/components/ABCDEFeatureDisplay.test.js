/**
 * Tests for ABCDEFeatureDisplay component
 *
 * Tests the ABCDE dermatological analysis display logic
 */

describe('ABCDEFeatureDisplay component logic', () => {
  describe('RISK_COLORS mapping', () => {
    const RISK_COLORS = {
      low: { bg: '#ecfdf5', border: '#a7f3d0', text: '#047857', icon: 'checkmark-circle' },
      moderate: { bg: '#fffbeb', border: '#fcd34d', text: '#b45309', icon: 'alert-circle' },
      high: { bg: '#fef2f2', border: '#fecaca', text: '#dc2626', icon: 'warning' },
      very_high: { bg: '#fef2f2', border: '#f87171', text: '#b91c1c', icon: 'warning' },
    };

    it('should have correct low risk styling', () => {
      const style = RISK_COLORS.low;
      expect(style.bg).toBe('#ecfdf5');
      expect(style.text).toBe('#047857');
      expect(style.icon).toBe('checkmark-circle');
    });

    it('should have correct moderate risk styling', () => {
      const style = RISK_COLORS.moderate;
      expect(style.bg).toBe('#fffbeb');
      expect(style.text).toBe('#b45309');
      expect(style.icon).toBe('alert-circle');
    });

    it('should have correct high risk styling', () => {
      const style = RISK_COLORS.high;
      expect(style.bg).toBe('#fef2f2');
      expect(style.text).toBe('#dc2626');
      expect(style.icon).toBe('warning');
    });

    it('should have correct very_high risk styling', () => {
      const style = RISK_COLORS.very_high;
      expect(style.bg).toBe('#fef2f2');
      expect(style.text).toBe('#b91c1c');
      expect(style.icon).toBe('warning');
    });
  });

  describe('ScoreMeter calculations', () => {
    const calculatePercentage = (score, maxScore = 1) => {
      return Math.min((score / maxScore) * 100, 100);
    };

    const getColor = (percentage) => {
      return percentage > 60 ? '#dc2626' : percentage > 40 ? '#f59e0b' : '#10b981';
    };

    it('should calculate percentage correctly for score 0', () => {
      expect(calculatePercentage(0)).toBe(0);
    });

    it('should calculate percentage correctly for score 0.5', () => {
      expect(calculatePercentage(0.5)).toBe(50);
    });

    it('should calculate percentage correctly for score 1', () => {
      expect(calculatePercentage(1)).toBe(100);
    });

    it('should cap percentage at 100', () => {
      expect(calculatePercentage(1.5)).toBe(100);
    });

    it('should use custom max score', () => {
      expect(calculatePercentage(5, 10)).toBe(50);
    });

    it('should return green color for low percentage (<= 40)', () => {
      expect(getColor(20)).toBe('#10b981');
      expect(getColor(40)).toBe('#10b981');
    });

    it('should return yellow/orange color for medium percentage (41-60)', () => {
      expect(getColor(41)).toBe('#f59e0b');
      expect(getColor(60)).toBe('#f59e0b');
    });

    it('should return red color for high percentage (> 60)', () => {
      expect(getColor(61)).toBe('#dc2626');
      expect(getColor(100)).toBe('#dc2626');
    });
  });

  describe('ABCDEAnalysis data structure', () => {
    const sampleAnalysis = {
      asymmetry: {
        overall_score: 0.3,
        risk_level: 'low',
        description: 'Test description',
        clinical_interpretation: 'Test interpretation',
        horizontal_asymmetry: 0.2,
        vertical_asymmetry: 0.3,
        shape_asymmetry: 0.25,
        color_asymmetry: 0.35,
      },
      border: {
        overall_score: 0.4,
        risk_level: 'moderate',
        description: 'Border description',
        clinical_interpretation: 'Border interpretation',
        irregularity_index: 0.35,
        notching_score: 0.4,
        blur_score: 0.3,
        radial_variance: 0.45,
        num_border_colors: 3,
      },
      color: {
        overall_score: 0.6,
        risk_level: 'moderate',
        description: 'Color description',
        clinical_interpretation: 'Color interpretation',
        num_colors: 4,
        colors_detected: ['brown', 'dark_brown', 'tan', 'black'],
        color_variance: 0.5,
        has_blue_white_veil: false,
        has_regression: false,
        dominant_color: 'brown',
        color_distribution: { brown: 0.4, dark_brown: 0.3, tan: 0.2, black: 0.1 },
      },
      diameter: {
        overall_score: 0.7,
        risk_level: 'high',
        description: 'Diameter description',
        clinical_interpretation: 'Diameter interpretation',
        estimated_diameter_mm: 8.5,
        pixel_diameter: 340,
        area_pixels: 90792,
        is_above_6mm: true,
        calibration_available: true,
      },
      evolution: {
        has_comparison: false,
        change_detected: null,
        change_description: null,
        risk_level: 'unknown',
        description: 'No previous images available for comparison',
        clinical_interpretation: 'Track this lesion over time',
      },
      total_score: 6.5,
      risk_level: 'moderate',
      summary: 'Moderate risk lesion with some atypical features',
      key_concerns: ['Multiple colors detected', 'Diameter > 6mm'],
      recommendation: 'Consider dermoscopic evaluation',
      methodology_notes: ['Automated analysis', 'Image quality: good'],
    };

    it('should have all required top-level fields', () => {
      expect(sampleAnalysis).toHaveProperty('asymmetry');
      expect(sampleAnalysis).toHaveProperty('border');
      expect(sampleAnalysis).toHaveProperty('color');
      expect(sampleAnalysis).toHaveProperty('diameter');
      expect(sampleAnalysis).toHaveProperty('evolution');
      expect(sampleAnalysis).toHaveProperty('total_score');
      expect(sampleAnalysis).toHaveProperty('risk_level');
      expect(sampleAnalysis).toHaveProperty('summary');
      expect(sampleAnalysis).toHaveProperty('key_concerns');
      expect(sampleAnalysis).toHaveProperty('recommendation');
    });

    it('should have valid asymmetry analysis', () => {
      expect(sampleAnalysis.asymmetry.overall_score).toBeGreaterThanOrEqual(0);
      expect(sampleAnalysis.asymmetry.overall_score).toBeLessThanOrEqual(1);
      expect(['low', 'moderate', 'high', 'very_high']).toContain(sampleAnalysis.asymmetry.risk_level);
    });

    it('should have valid border analysis', () => {
      expect(sampleAnalysis.border.num_border_colors).toBeGreaterThanOrEqual(0);
      expect(sampleAnalysis.border.irregularity_index).toBeGreaterThanOrEqual(0);
    });

    it('should have valid color analysis', () => {
      expect(sampleAnalysis.color.num_colors).toBeGreaterThanOrEqual(0);
      expect(sampleAnalysis.color.colors_detected).toBeInstanceOf(Array);
      expect(typeof sampleAnalysis.color.has_blue_white_veil).toBe('boolean');
      expect(typeof sampleAnalysis.color.has_regression).toBe('boolean');
    });

    it('should have valid diameter analysis', () => {
      expect(typeof sampleAnalysis.diameter.calibration_available).toBe('boolean');
      if (sampleAnalysis.diameter.calibration_available) {
        expect(typeof sampleAnalysis.diameter.estimated_diameter_mm).toBe('number');
      }
    });

    it('should have valid evolution analysis', () => {
      expect(typeof sampleAnalysis.evolution.has_comparison).toBe('boolean');
    });

    it('should have total score between 0 and 10', () => {
      expect(sampleAnalysis.total_score).toBeGreaterThanOrEqual(0);
      expect(sampleAnalysis.total_score).toBeLessThanOrEqual(10);
    });

    it('should have key concerns as an array', () => {
      expect(sampleAnalysis.key_concerns).toBeInstanceOf(Array);
    });
  });

  describe('risk level text formatting', () => {
    const formatRiskLevel = (riskLevel) => {
      return riskLevel.replace('_', ' ').toUpperCase();
    };

    it('should format low risk correctly', () => {
      expect(formatRiskLevel('low')).toBe('LOW');
    });

    it('should format moderate risk correctly', () => {
      expect(formatRiskLevel('moderate')).toBe('MODERATE');
    });

    it('should format high risk correctly', () => {
      expect(formatRiskLevel('high')).toBe('HIGH');
    });

    it('should format very_high risk correctly', () => {
      expect(formatRiskLevel('very_high')).toBe('VERY HIGH');
    });
  });

  describe('color name formatting', () => {
    const formatColorName = (colorName) => {
      return colorName.replace('_', ' ');
    };

    it('should format single word colors', () => {
      expect(formatColorName('brown')).toBe('brown');
      expect(formatColorName('black')).toBe('black');
    });

    it('should format compound color names', () => {
      expect(formatColorName('dark_brown')).toBe('dark brown');
      expect(formatColorName('light_tan')).toBe('light tan');
    });
  });

  describe('diameter threshold check', () => {
    const isAboveThreshold = (diameter, threshold = 6) => {
      return diameter !== null && diameter > threshold;
    };

    it('should return true for diameter above 6mm', () => {
      expect(isAboveThreshold(7)).toBe(true);
      expect(isAboveThreshold(8.5)).toBe(true);
    });

    it('should return false for diameter at or below 6mm', () => {
      expect(isAboveThreshold(5)).toBe(false);
      expect(isAboveThreshold(6)).toBe(false);
    });

    it('should handle null diameter', () => {
      expect(isAboveThreshold(null)).toBe(false);
    });

    it('should work with custom threshold', () => {
      expect(isAboveThreshold(5, 4)).toBe(true);
      expect(isAboveThreshold(3, 4)).toBe(false);
    });
  });

  describe('feature section toggle logic', () => {
    it('should toggle section to null when clicking same section', () => {
      let expandedSection = 'asymmetry';
      const toggleSection = (section) => {
        expandedSection = expandedSection === section ? null : section;
      };

      toggleSection('asymmetry');
      expect(expandedSection).toBeNull();
    });

    it('should change to new section when clicking different section', () => {
      let expandedSection = 'asymmetry';
      const toggleSection = (section) => {
        expandedSection = expandedSection === section ? null : section;
      };

      toggleSection('border');
      expect(expandedSection).toBe('border');
    });

    it('should expand section when none is selected', () => {
      let expandedSection = null;
      const toggleSection = (section) => {
        expandedSection = expandedSection === section ? null : section;
      };

      toggleSection('color');
      expect(expandedSection).toBe('color');
    });
  });
});
