/**
 * Tests for AnalysisHistoryService
 *
 * Tests analysis history functionality including:
 * - Fetching analysis history
 * - Getting individual analyses
 * - Statistics and formatting
 */

// Mock config before importing services
jest.mock('../../config', () => ({
  API_BASE_URL: 'http://test-api.example.com',
  API_ENDPOINTS: {
    ANALYSIS_HISTORY: 'http://test-api.example.com/analysis/history',
    ANALYSIS_STATS: 'http://test-api.example.com/analysis/stats',
    USER_EXTENDED: 'http://test-api.example.com/me/extended',
  },
  REQUEST_TIMEOUT: 30000,
}));

// Mock AuthService
jest.mock('../../services/AuthService', () => ({
  getToken: jest.fn(),
}));

import AnalysisHistoryService from '../../services/AnalysisHistoryService';
import AuthService from '../../services/AuthService';

describe('AnalysisHistoryService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    AuthService.getToken.mockReturnValue('test-token');
  });

  describe('getAnalysisHistory', () => {
    it('should fetch analysis history successfully', async () => {
      const mockHistory = [
        {
          id: 1,
          analysis_type: 'full',
          is_lesion: true,
          predicted_class: 'melanocytic_nevus',
          binary_confidence: 0.95,
          risk_level: 'low',
          created_at: '2024-01-15T10:30:00',
        },
        {
          id: 2,
          analysis_type: 'full',
          is_lesion: false,
          binary_confidence: 0.88,
          risk_level: 'low',
          created_at: '2024-01-14T15:45:00',
        },
      ];

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockHistory),
      });

      const result = await AnalysisHistoryService.getAnalysisHistory();

      expect(result).toHaveLength(2);
      expect(result[0].id).toBe(1);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/analysis/history'),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test-token',
          }),
        })
      );
    });

    it('should pass pagination parameters', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve([]),
      });

      await AnalysisHistoryService.getAnalysisHistory(10, 5);

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('skip=10'),
        expect.any(Object)
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('limit=5'),
        expect.any(Object)
      );
    });

    it('should throw error when not authenticated', async () => {
      AuthService.getToken.mockReturnValue(null);

      await expect(
        AnalysisHistoryService.getAnalysisHistory()
      ).rejects.toThrow('Authentication required');
    });

    it('should handle API errors', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: () => Promise.resolve('Internal server error'),
      });

      await expect(
        AnalysisHistoryService.getAnalysisHistory()
      ).rejects.toThrow(/Failed to fetch analysis history/);
    });
  });

  describe('getAnalysisById', () => {
    it('should fetch specific analysis successfully', async () => {
      const mockAnalysis = {
        id: 1,
        analysis_type: 'full',
        is_lesion: true,
        predicted_class: 'melanocytic_nevus',
        binary_confidence: 0.95,
        risk_level: 'low',
        created_at: '2024-01-15T10:30:00',
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockAnalysis),
      });

      const result = await AnalysisHistoryService.getAnalysisById(1);

      expect(result.id).toBe(1);
      expect(result.predicted_class).toBe('melanocytic_nevus');
    });

    it('should throw error for non-existent analysis', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      await expect(
        AnalysisHistoryService.getAnalysisById(999)
      ).rejects.toThrow('Analysis not found');
    });
  });

  describe('getAnalysisStatistics', () => {
    it('should fetch statistics successfully', async () => {
      const mockStats = {
        total_analyses: 25,
        lesion_detections: 18,
        average_confidence: 0.87,
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockStats),
      });

      const result = await AnalysisHistoryService.getAnalysisStatistics();

      expect(result.total_analyses).toBe(25);
      expect(result.lesion_detections).toBe(18);
    });
  });

  describe('getExtendedUserInfo', () => {
    it('should fetch extended user info successfully', async () => {
      const mockUserInfo = {
        id: 1,
        username: 'testuser',
        total_analyses: 25,
        last_analysis_date: '2024-01-15T10:30:00',
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockUserInfo),
      });

      const result = await AnalysisHistoryService.getExtendedUserInfo();

      expect(result.username).toBe('testuser');
      expect(result.total_analyses).toBe(25);
    });
  });

  describe('formatAnalysisForDisplay', () => {
    it('should format analysis data correctly', () => {
      const rawAnalysis = {
        id: 1,
        analysis_type: 'full',
        is_lesion: true,
        predicted_class: 'melanocytic_nevus',
        binary_confidence: 0.95,
        risk_level: 'low',
        risk_recommendation: 'Monitor for changes',
        processing_time_seconds: 1.5,
        model_version: 'v1.0',
        created_at: '2024-01-15T10:30:00Z',
        image_filename: 'test.jpg',
      };

      const formatted = AnalysisHistoryService.formatAnalysisForDisplay(rawAnalysis);

      expect(formatted.id).toBe(1);
      expect(formatted.isLesion).toBe(true);
      expect(formatted.predictedClass).toBe('melanocytic_nevus');
      expect(formatted.confidence).toBe(0.95);
      expect(formatted.riskLevel).toBe('low');
      expect(formatted.riskColor).toBe('#28a745'); // Green for low risk
      expect(formatted.confidenceLevel.level).toBe('Very High');
    });

    it('should return correct risk colors', () => {
      const highRisk = { risk_level: 'high', created_at: '2024-01-15T10:30:00Z' };
      const mediumRisk = { risk_level: 'medium', created_at: '2024-01-15T10:30:00Z' };
      const lowRisk = { risk_level: 'low', created_at: '2024-01-15T10:30:00Z' };

      expect(AnalysisHistoryService.formatAnalysisForDisplay(highRisk).riskColor).toBe('#dc3545');
      expect(AnalysisHistoryService.formatAnalysisForDisplay(mediumRisk).riskColor).toBe('#ffc107');
      expect(AnalysisHistoryService.formatAnalysisForDisplay(lowRisk).riskColor).toBe('#28a745');
    });

    it('should return correct confidence levels', () => {
      const veryHigh = { binary_confidence: 0.95, created_at: '2024-01-15T10:30:00Z' };
      const high = { binary_confidence: 0.85, created_at: '2024-01-15T10:30:00Z' };
      const medium = { binary_confidence: 0.75, created_at: '2024-01-15T10:30:00Z' };
      const low = { binary_confidence: 0.65, created_at: '2024-01-15T10:30:00Z' };
      const veryLow = { binary_confidence: 0.55, created_at: '2024-01-15T10:30:00Z' };

      expect(AnalysisHistoryService.formatAnalysisForDisplay(veryHigh).confidenceLevel.level).toBe('Very High');
      expect(AnalysisHistoryService.formatAnalysisForDisplay(high).confidenceLevel.level).toBe('High');
      expect(AnalysisHistoryService.formatAnalysisForDisplay(medium).confidenceLevel.level).toBe('Medium');
      expect(AnalysisHistoryService.formatAnalysisForDisplay(low).confidenceLevel.level).toBe('Low');
      expect(AnalysisHistoryService.formatAnalysisForDisplay(veryLow).confidenceLevel.level).toBe('Very Low');
    });
  });

  describe('getRelativeTime', () => {
    it('should return "Just now" for recent times', () => {
      const now = new Date();
      expect(AnalysisHistoryService.getRelativeTime(now)).toBe('Just now');
    });

    it('should return minutes ago', () => {
      const thirtyMinutesAgo = new Date(Date.now() - 30 * 60 * 1000);
      expect(AnalysisHistoryService.getRelativeTime(thirtyMinutesAgo)).toBe('30 minutes ago');
    });

    it('should return hours ago', () => {
      const threeHoursAgo = new Date(Date.now() - 3 * 60 * 60 * 1000);
      expect(AnalysisHistoryService.getRelativeTime(threeHoursAgo)).toBe('3 hours ago');
    });

    it('should return days ago', () => {
      const twoDaysAgo = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000);
      expect(AnalysisHistoryService.getRelativeTime(twoDaysAgo)).toBe('2 days ago');
    });
  });

  describe('getAnalysisSummary', () => {
    it('should return correct summary for analyses', () => {
      const analyses = [
        { is_lesion: true, binary_confidence: 0.9, risk_level: 'low', created_at: '2024-01-15T10:30:00Z' },
        { is_lesion: true, binary_confidence: 0.8, risk_level: 'medium', created_at: '2024-01-14T10:30:00Z' },
        { is_lesion: false, binary_confidence: 0.95, risk_level: 'low', created_at: '2024-01-13T10:30:00Z' },
        { is_lesion: true, binary_confidence: 0.7, risk_level: 'high', created_at: '2024-01-12T10:30:00Z' },
      ];

      const summary = AnalysisHistoryService.getAnalysisSummary(analyses);

      expect(summary.totalAnalyses).toBe(4);
      expect(summary.lesionDetections).toBe(3);
      expect(summary.nonLesionDetections).toBe(1);
      expect(summary.averageConfidence).toBeCloseTo(0.84, 1);
      expect(summary.riskDistribution.low).toBe(2);
      expect(summary.riskDistribution.medium).toBe(1);
      expect(summary.riskDistribution.high).toBe(1);
    });

    it('should handle empty analyses array', () => {
      const summary = AnalysisHistoryService.getAnalysisSummary([]);

      expect(summary.totalAnalyses).toBe(0);
      expect(summary.lesionDetections).toBe(0);
      expect(summary.nonLesionDetections).toBe(0);
      expect(summary.averageConfidence).toBe(0);
    });

    it('should handle null/undefined analyses', () => {
      const summaryNull = AnalysisHistoryService.getAnalysisSummary(null);
      const summaryUndefined = AnalysisHistoryService.getAnalysisSummary(undefined);

      expect(summaryNull.totalAnalyses).toBe(0);
      expect(summaryUndefined.totalAnalyses).toBe(0);
    });
  });
});
