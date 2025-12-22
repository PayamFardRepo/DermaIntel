import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { VictoryChart, VictoryLine, VictoryScatter, VictoryAxis, VictoryArea } from 'victory-native';

interface ConfidenceDataPoint {
  date: Date;
  confidence: number; // 0-1
  diagnosis?: string;
}

interface ConfidenceTimelineChartProps {
  data: ConfidenceDataPoint[];
  title?: string;
}

const ConfidenceTimelineChart: React.FC<ConfidenceTimelineChartProps> = ({
  data,
  title = 'AI Confidence Over Time'
}) => {
  const screenWidth = Dimensions.get('window').width;
  const chartWidth = screenWidth - 40;

  if (!data || data.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Text style={styles.emptyText}>No confidence data available</Text>
      </View>
    );
  }

  // Transform data for Victory charts
  const chartData = data.map((point) => ({
    x: point.date,
    y: point.confidence * 100, // Convert to percentage
    diagnosis: point.diagnosis
  }));

  // Calculate statistics
  const confidences = data.map(d => d.confidence);
  const avgConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;
  const minConfidence = Math.min(...confidences);
  const maxConfidence = Math.max(...confidences);
  const currentConfidence = confidences[confidences.length - 1];

  // Count confidence levels
  const highConfidenceCount = confidences.filter(c => c >= 0.8).length;
  const mediumConfidenceCount = confidences.filter(c => c >= 0.5 && c < 0.8).length;
  const lowConfidenceCount = confidences.filter(c => c < 0.5).length;

  // Determine overall trend
  const getTrendColor = (conf: number) => {
    if (conf >= 0.8) return '#10b981'; // High confidence - green
    if (conf >= 0.5) return '#f59e0b'; // Medium confidence - amber
    return '#dc2626'; // Low confidence - red
  };

  const getConfidenceLevel = (conf: number) => {
    if (conf >= 0.8) return 'High';
    if (conf >= 0.5) return 'Medium';
    return 'Low';
  };

  // Check if confidence is improving or declining
  let trend = 'Stable';
  if (data.length >= 2) {
    const firstHalfAvg = confidences.slice(0, Math.floor(confidences.length / 2))
      .reduce((a, b) => a + b, 0) / Math.floor(confidences.length / 2);
    const secondHalfAvg = confidences.slice(Math.floor(confidences.length / 2))
      .reduce((a, b) => a + b, 0) / (confidences.length - Math.floor(confidences.length / 2));

    if (secondHalfAvg > firstHalfAvg + 0.1) trend = 'Improving';
    else if (secondHalfAvg < firstHalfAvg - 0.1) trend = 'Declining';
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>{title}</Text>
        <View style={[styles.trendBadge, { backgroundColor: getTrendColor(currentConfidence) }]}>
          <Text style={styles.trendBadgeText}>{getConfidenceLevel(currentConfidence)}</Text>
        </View>
      </View>

      {/* Stats Cards */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>Current</Text>
          <Text style={[styles.statValue, { color: getTrendColor(currentConfidence) }]}>
            {(currentConfidence * 100).toFixed(1)}%
          </Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>Average</Text>
          <Text style={styles.statValue}>
            {(avgConfidence * 100).toFixed(1)}%
          </Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>Trend</Text>
          <Text style={[
            styles.statValue,
            {
              color: trend === 'Improving' ? '#10b981' :
                     trend === 'Declining' ? '#dc2626' : '#718096'
            }
          ]}>
            {trend === 'Improving' ? '↗' : trend === 'Declining' ? '↘' : '→'}
          </Text>
        </View>
      </View>

      {/* Chart */}
      <View style={styles.chartContainer}>
        <VictoryChart
          width={chartWidth}
          height={250}
          padding={{ left: 60, right: 30, top: 20, bottom: 50 }}
          domain={{ y: [0, 100] }}
        >
          {/* X Axis - Dates */}
          <VictoryAxis
            tickFormat={(date) => {
              const d = new Date(date);
              return `${d.getMonth() + 1}/${d.getDate()}`;
            }}
            style={{
              tickLabels: { fontSize: 10, padding: 5, angle: -45 },
              axis: { stroke: '#cbd5e0' },
              grid: { stroke: '#e2e8f0', strokeDasharray: '4,4' }
            }}
          />

          {/* Y Axis - Confidence % */}
          <VictoryAxis
            dependentAxis
            label="Confidence (%)"
            tickFormat={(t) => `${t}%`}
            style={{
              axisLabel: { fontSize: 12, padding: 40, fill: '#4a5568' },
              tickLabels: { fontSize: 10, padding: 5, fill: '#4a5568' },
              axis: { stroke: '#cbd5e0' },
              grid: { stroke: '#e2e8f0', strokeDasharray: '4,4' }
            }}
          />

          {/* Confidence threshold line at 70% */}
          <VictoryLine
            data={[
              { x: chartData[0].x, y: 70 },
              { x: chartData[chartData.length - 1].x, y: 70 }
            ]}
            style={{
              data: {
                stroke: '#f59e0b',
                strokeWidth: 2,
                strokeDasharray: '5,5'
              }
            }}
          />

          {/* High confidence threshold line at 80% */}
          <VictoryLine
            data={[
              { x: chartData[0].x, y: 80 },
              { x: chartData[chartData.length - 1].x, y: 80 }
            ]}
            style={{
              data: {
                stroke: '#10b981',
                strokeWidth: 2,
                strokeDasharray: '5,5'
              }
            }}
          />

          {/* Area under the line */}
          <VictoryArea
            data={chartData}
            style={{
              data: {
                fill: '#4299e1',
                fillOpacity: 0.1,
                stroke: 'none'
              }
            }}
          />

          {/* Line */}
          <VictoryLine
            data={chartData}
            style={{
              data: {
                stroke: '#4299e1',
                strokeWidth: 3
              }
            }}
            interpolation="monotoneX"
          />

          {/* Data Points */}
          <VictoryScatter
            data={chartData}
            size={({ datum }) => {
              // Larger points for low confidence (needs attention)
              return datum.y < 50 ? 8 : 6;
            }}
            style={{
              data: {
                fill: ({ datum }) => getTrendColor(datum.y / 100),
                stroke: '#ffffff',
                strokeWidth: 2
              }
            }}
          />
        </VictoryChart>
      </View>

      {/* Confidence Distribution */}
      <View style={styles.distributionContainer}>
        <Text style={styles.distributionTitle}>Confidence Distribution</Text>
        <View style={styles.distributionRow}>
          <View style={styles.distributionItem}>
            <View style={[styles.distributionBar, { backgroundColor: '#10b981', width: `${(highConfidenceCount / data.length) * 100}%` }]} />
            <Text style={styles.distributionLabel}>
              High (&gt;80%): {highConfidenceCount} / {data.length}
            </Text>
          </View>
          <View style={styles.distributionItem}>
            <View style={[styles.distributionBar, { backgroundColor: '#f59e0b', width: `${(mediumConfidenceCount / data.length) * 100}%` }]} />
            <Text style={styles.distributionLabel}>
              Medium (50-80%): {mediumConfidenceCount} / {data.length}
            </Text>
          </View>
          <View style={styles.distributionItem}>
            <View style={[styles.distributionBar, { backgroundColor: '#dc2626', width: `${(lowConfidenceCount / data.length) * 100}%` }]} />
            <Text style={styles.distributionLabel}>
              Low (&lt;50%): {lowConfidenceCount} / {data.length}
            </Text>
          </View>
        </View>
      </View>

      {/* Legend */}
      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendLine, { backgroundColor: '#10b981', borderStyle: 'dashed' }]} />
          <Text style={styles.legendText}>High confidence threshold (80%)</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendLine, { backgroundColor: '#f59e0b', borderStyle: 'dashed' }]} />
          <Text style={styles.legendText}>Recommended minimum (70%)</Text>
        </View>
      </View>

      {/* Low Confidence Warning */}
      {currentConfidence < 0.7 && (
        <View style={styles.warningBox}>
          <Text style={styles.warningText}>
            ⚠️ Current confidence is below recommended threshold. Consider retaking the photo with better lighting and focus, or consult a dermatologist for professional evaluation.
          </Text>
        </View>
      )}

      {/* Declining Trend Alert */}
      {trend === 'Declining' && (
        <View style={styles.alertBox}>
          <Text style={styles.alertText}>
            📉 AI confidence is declining over time. This may indicate image quality issues or lesion changes that make classification more difficult. Consider professional evaluation.
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 16,
    marginVertical: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c5282',
  },
  trendBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  trendBadgeText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#f7fafc',
    borderRadius: 8,
    padding: 12,
    marginHorizontal: 4,
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 11,
    color: '#718096',
    marginBottom: 4,
  },
  statValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2d3748',
  },
  chartContainer: {
    alignItems: 'center',
    marginBottom: 12,
  },
  distributionContainer: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
  },
  distributionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2c5282',
    marginBottom: 12,
  },
  distributionRow: {
    gap: 12,
  },
  distributionItem: {
    marginBottom: 8,
  },
  distributionBar: {
    height: 8,
    borderRadius: 4,
    marginBottom: 4,
  },
  distributionLabel: {
    fontSize: 12,
    color: '#4a5568',
  },
  legend: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  legendLine: {
    width: 30,
    height: 2,
    marginRight: 8,
  },
  legendText: {
    fontSize: 11,
    color: '#4a5568',
  },
  emptyContainer: {
    backgroundColor: '#f7fafc',
    borderRadius: 12,
    padding: 40,
    alignItems: 'center',
    marginVertical: 12,
  },
  emptyText: {
    fontSize: 14,
    color: '#718096',
  },
  warningBox: {
    backgroundColor: '#fef2f2',
    borderRadius: 8,
    padding: 12,
    marginTop: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#dc2626',
  },
  warningText: {
    fontSize: 13,
    color: '#991b1b',
    lineHeight: 18,
  },
  alertBox: {
    backgroundColor: '#fef3c7',
    borderRadius: 8,
    padding: 12,
    marginTop: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  alertText: {
    fontSize: 13,
    color: '#92400e',
    lineHeight: 18,
  },
});

export default ConfidenceTimelineChart;
