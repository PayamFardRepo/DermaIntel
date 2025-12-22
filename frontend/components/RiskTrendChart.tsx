import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { VictoryChart, VictoryBar, VictoryAxis, VictoryLabel } from 'victory-native';

interface RiskDataPoint {
  date: Date;
  riskLevel: 'low' | 'medium' | 'high';
  diagnosis?: string;
}

interface RiskTrendChartProps {
  data: RiskDataPoint[];
  title?: string;
}

const RiskTrendChart: React.FC<RiskTrendChartProps> = ({ data, title = 'Risk Level Trend' }) => {
  const screenWidth = Dimensions.get('window').width;
  const chartWidth = screenWidth - 40;

  if (!data || data.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Text style={styles.emptyText}>No risk data available</Text>
      </View>
    );
  }

  // Map risk levels to numeric values
  const riskToValue = (risk: string): number => {
    switch (risk) {
      case 'low': return 1;
      case 'medium': return 2;
      case 'high': return 3;
      default: return 1;
    }
  };

  // Map risk levels to colors
  const riskToColor = (risk: string): string => {
    switch (risk) {
      case 'low': return '#10b981';
      case 'medium': return '#f59e0b';
      case 'high': return '#dc2626';
      default: return '#6b7280';
    }
  };

  // Transform data for Victory charts
  const chartData = data.map((point, index) => ({
    x: index + 1,
    y: riskToValue(point?.riskLevel || 'low'),
    label: (point?.riskLevel || 'low').toUpperCase(),
    color: riskToColor(point?.riskLevel || 'low'),
    date: point?.date || new Date(),
    diagnosis: point?.diagnosis || ''
  }));

  // Check for risk escalation
  const hasEscalation = data.some((point, index) => {
    if (index === 0) return false;
    return riskToValue(point.riskLevel) > riskToValue(data[index - 1].riskLevel);
  });

  // Get current risk level
  const currentRisk = data[data.length - 1].riskLevel;
  const currentRiskColor = riskToColor(currentRisk);

  // Count each risk level
  const riskCounts = {
    low: data.filter(d => d.riskLevel === 'low').length,
    medium: data.filter(d => d.riskLevel === 'medium').length,
    high: data.filter(d => d.riskLevel === 'high').length
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>{title}</Text>
        <View style={[styles.currentRiskBadge, { backgroundColor: currentRiskColor }]}>
          <Text style={styles.currentRiskText}>Current: {currentRisk.toUpperCase()}</Text>
        </View>
      </View>

      {/* Stats Cards */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statValue}>{data.length}</Text>
          <Text style={styles.statLabel}>Total Checks</Text>
        </View>
        <View style={styles.statCard}>
          <View style={styles.riskCountRow}>
            <View style={[styles.riskDot, { backgroundColor: '#10b981' }]} />
            <Text style={styles.statValue}>{riskCounts.low}</Text>
          </View>
          <Text style={styles.statLabel}>Low Risk</Text>
        </View>
        <View style={styles.statCard}>
          <View style={styles.riskCountRow}>
            <View style={[styles.riskDot, { backgroundColor: '#f59e0b' }]} />
            <Text style={styles.statValue}>{riskCounts.medium}</Text>
          </View>
          <Text style={styles.statLabel}>Medium Risk</Text>
        </View>
        <View style={styles.statCard}>
          <View style={styles.riskCountRow}>
            <View style={[styles.riskDot, { backgroundColor: '#dc2626' }]} />
            <Text style={styles.statValue}>{riskCounts.high}</Text>
          </View>
          <Text style={styles.statLabel}>High Risk</Text>
        </View>
      </View>

      {/* Chart */}
      <View style={styles.chartContainer}>
        <VictoryChart
          width={chartWidth}
          height={250}
          padding={{ left: 60, right: 30, top: 30, bottom: 50 }}
          domain={{ y: [0, 4] }}
        >
          {/* X Axis - Analysis Number */}
          <VictoryAxis
            label="Analysis #"
            style={{
              axisLabel: { fontSize: 12, padding: 30, fill: '#4a5568' },
              tickLabels: { fontSize: 10, padding: 5, fill: '#4a5568' },
              axis: { stroke: '#cbd5e0' },
              grid: { stroke: '#e2e8f0', strokeDasharray: '4,4' }
            }}
          />

          {/* Y Axis - Risk Level */}
          <VictoryAxis
            dependentAxis
            label="Risk Level"
            tickValues={[1, 2, 3]}
            tickFormat={(t) => {
              if (t === 1) return 'LOW';
              if (t === 2) return 'MED';
              if (t === 3) return 'HIGH';
              return '';
            }}
            style={{
              axisLabel: { fontSize: 12, padding: 40, fill: '#4a5568' },
              tickLabels: { fontSize: 10, padding: 5, fill: '#4a5568' },
              axis: { stroke: '#cbd5e0' },
              grid: { stroke: '#e2e8f0', strokeDasharray: '4,4' }
            }}
          />

          {/* Bars */}
          <VictoryBar
            data={chartData}
            style={{
              data: {
                fill: ({ datum }) => datum.color,
                width: 30
              }
            }}
            labels={({ datum }) => datum.label}
            labelComponent={
              <VictoryLabel
                style={{ fontSize: 10, fill: '#ffffff', fontWeight: 'bold' }}
                dy={-10}
              />
            }
          />
        </VictoryChart>
      </View>

      {/* Timeline Details */}
      <View style={styles.timelineContainer}>
        <Text style={styles.timelineTitle}>Risk History</Text>
        {data.slice().reverse().map((point, index) => (
          <View key={index} style={styles.timelineItem}>
            <View style={[styles.timelineDot, { backgroundColor: riskToColor(point.riskLevel) }]} />
            <View style={styles.timelineContent}>
              <View style={styles.timelineHeader}>
                <Text style={styles.timelineDate}>
                  {point.date.toLocaleDateString()}
                </Text>
                <View style={[styles.timelineRiskBadge, { backgroundColor: riskToColor(point.riskLevel) }]}>
                  <Text style={styles.timelineRiskText}>{point.riskLevel.toUpperCase()}</Text>
                </View>
              </View>
              {point.diagnosis && (
                <Text style={styles.timelineDiagnosis}>{point.diagnosis}</Text>
              )}
            </View>
          </View>
        ))}
      </View>

      {/* Risk Escalation Alert */}
      {hasEscalation && (
        <View style={styles.alertBox}>
          <Text style={styles.alertText}>
            ⚠️ Risk escalation detected in your tracking history. Consider scheduling a dermatologist consultation.
          </Text>
        </View>
      )}

      {/* Current High Risk Alert */}
      {currentRisk === 'high' && (
        <View style={styles.warningBox}>
          <Text style={styles.warningText}>
            🚨 Current risk level is HIGH. Immediate dermatologist evaluation is strongly recommended.
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
  currentRiskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  currentRiskText: {
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
    marginHorizontal: 2,
    alignItems: 'center',
  },
  riskCountRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  riskDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  statLabel: {
    fontSize: 10,
    color: '#718096',
    marginTop: 4,
    textAlign: 'center',
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
  timelineContainer: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
  },
  timelineTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c5282',
    marginBottom: 12,
  },
  timelineItem: {
    flexDirection: 'row',
    marginBottom: 12,
    alignItems: 'flex-start',
  },
  timelineDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginTop: 4,
    marginRight: 12,
  },
  timelineContent: {
    flex: 1,
  },
  timelineHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  timelineDate: {
    fontSize: 13,
    color: '#4a5568',
    fontWeight: '600',
  },
  timelineRiskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  timelineRiskText: {
    color: '#ffffff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  timelineDiagnosis: {
    fontSize: 12,
    color: '#718096',
    marginTop: 2,
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
    fontWeight: '600',
  },
});

export default RiskTrendChart;
