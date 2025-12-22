import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { VictoryChart, VictoryLine, VictoryScatter, VictoryAxis, VictoryArea } from 'victory-native';

interface SizeDataPoint {
  date: Date;
  size: number; // in mm²
  label?: string;
}

interface SizeTimelineChartProps {
  data: SizeDataPoint[];
  title?: string;
}

const SizeTimelineChart: React.FC<SizeTimelineChartProps> = ({ data, title = 'Size Over Time' }) => {
  const screenWidth = Dimensions.get('window').width;
  const chartWidth = screenWidth - 40;

  if (!data || data.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Text style={styles.emptyText}>No size data available</Text>
      </View>
    );
  }

  // Transform data for Victory charts
  const chartData = data.map((point) => ({
    x: point.date,
    y: point.size,
    label: point.label
  }));

  // Calculate growth zones
  const minSize = Math.min(...data.map(d => d.size));
  const maxSize = Math.max(...data.map(d => d.size));
  const sizeRange = maxSize - minSize;
  const growthThreshold = minSize + (sizeRange * 0.2); // 20% growth is concerning

  // Determine overall trend
  const firstSize = data[0].size;
  const lastSize = data[data.length - 1].size;
  const growthPercent = ((lastSize - firstSize) / firstSize) * 100;

  // Calculate growth rate if we have date range
  let growthRate = null;
  if (data.length >= 2) {
    const firstDate = data[0].date.getTime();
    const lastDate = data[data.length - 1].date.getTime();
    const monthsDiff = (lastDate - firstDate) / (1000 * 60 * 60 * 24 * 30);
    if (monthsDiff > 0) {
      const sizeDiff = lastSize - firstSize;
      growthRate = sizeDiff / monthsDiff;
    }
  }

  const getTrendColor = () => {
    if (growthPercent > 20) return '#dc2626'; // Red - significant growth
    if (growthPercent > 10) return '#f59e0b'; // Amber - moderate growth
    return '#10b981'; // Green - stable
  };

  const getTrendText = () => {
    if (growthPercent > 20) return 'Significant Growth';
    if (growthPercent > 10) return 'Moderate Growth';
    if (growthPercent > 0) return 'Slight Growth';
    if (growthPercent < -10) return 'Shrinking';
    return 'Stable';
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>{title}</Text>
        <View style={[styles.trendBadge, { backgroundColor: getTrendColor() }]}>
          <Text style={styles.trendBadgeText}>{getTrendText()}</Text>
        </View>
      </View>

      {/* Stats Cards */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>Current</Text>
          <Text style={styles.statValue}>{lastSize.toFixed(1)} mm²</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>Change</Text>
          <Text style={[styles.statValue, { color: getTrendColor() }]}>
            {growthPercent > 0 ? '+' : ''}{growthPercent.toFixed(1)}%
          </Text>
        </View>
        {growthRate !== null && (
          <View style={styles.statCard}>
            <Text style={styles.statLabel}>Growth Rate</Text>
            <Text style={styles.statValue}>
              {growthRate > 0 ? '+' : ''}{growthRate.toFixed(2)} mm²/mo
            </Text>
          </View>
        )}
      </View>

      {/* Chart */}
      <View style={styles.chartContainer}>
        <VictoryChart
          width={chartWidth}
          height={250}
          padding={{ left: 60, right: 30, top: 20, bottom: 50 }}
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

          {/* Y Axis - Size */}
          <VictoryAxis
            dependentAxis
            label="Size (mm²)"
            style={{
              axisLabel: { fontSize: 12, padding: 40, fill: '#4a5568' },
              tickLabels: { fontSize: 10, padding: 5, fill: '#4a5568' },
              axis: { stroke: '#cbd5e0' },
              grid: { stroke: '#e2e8f0', strokeDasharray: '4,4' }
            }}
          />

          {/* Area under the line (gradient effect) */}
          <VictoryArea
            data={chartData}
            style={{
              data: {
                fill: getTrendColor(),
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
                stroke: getTrendColor(),
                strokeWidth: 3
              }
            }}
            interpolation="monotoneX"
          />

          {/* Data Points */}
          <VictoryScatter
            data={chartData}
            size={6}
            style={{
              data: {
                fill: getTrendColor(),
                stroke: '#ffffff',
                strokeWidth: 2
              }
            }}
          />
        </VictoryChart>
      </View>

      {/* Legend */}
      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: '#10b981' }]} />
          <Text style={styles.legendText}>Stable (&lt;10% growth)</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: '#f59e0b' }]} />
          <Text style={styles.legendText}>Moderate (10-20%)</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: '#dc2626' }]} />
          <Text style={styles.legendText}>Concerning (&gt;20%)</Text>
        </View>
      </View>

      {/* Clinical Note */}
      {growthPercent > 20 && (
        <View style={styles.warningBox}>
          <Text style={styles.warningText}>
            ⚠️ Rapid growth detected. Consider consulting a dermatologist for evaluation.
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
  legend: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 6,
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
});

export default SizeTimelineChart;
