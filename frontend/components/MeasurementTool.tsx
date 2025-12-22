import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Image,
  Dimensions,
  Alert,
  ScrollView,
  TextInput,
  Modal
} from 'react-native';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface MeasurementLine {
  id: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  distanceMm?: number;
  label: string;
}

interface CalibrationData {
  calibration_found: boolean;
  calibration_type: string | null;
  pixels_per_mm: number | null;
  confidence: number;
  detected_objects: any[];
}

interface MeasurementToolProps {
  imageUri: string;
  calibration: CalibrationData;
  onMeasurementsComplete?: (measurements: MeasurementLine[]) => void;
}

export default function MeasurementTool({
  imageUri,
  calibration,
  onMeasurementsComplete
}: MeasurementToolProps) {
  const [measurements, setMeasurements] = useState<MeasurementLine[]>([]);
  const [currentLine, setCurrentLine] = useState<{ x: number; y: number } | null>(null);
  const [imageLayout, setImageLayout] = useState<{ width: number; height: number; x: number; y: number } | null>(null);
  const [manualCalibration, setManualCalibration] = useState(false);
  const [calibrationDistance, setCalibrationDistance] = useState('');
  const [pixelsPerMm, setPixelsPerMm] = useState(calibration.pixels_per_mm);

  const measurementIdCounter = useRef(0);

  const handleImageLayout = (event: any) => {
    const { width, height, x, y } = event.nativeEvent.layout;
    setImageLayout({ width, height, x, y });
  };

  const handleTouchStart = (event: any) => {
    if (!imageLayout) return;

    const { locationX, locationY } = event.nativeEvent;
    setCurrentLine({ x: locationX, y: locationY });
  };

  const handleTouchMove = (event: any) => {
    // Visual feedback during drawing could be added here
  };

  const handleTouchEnd = (event: any) => {
    if (!currentLine || !imageLayout) return;

    const { locationX, locationY } = event.nativeEvent;

    // Calculate distance in pixels
    const dx = locationX - currentLine.x;
    const dy = locationY - currentLine.y;
    const distancePixels = Math.sqrt(dx * dx + dy * dy);

    // Don't create measurement if line is too short (likely accidental tap)
    if (distancePixels < 10) {
      setCurrentLine(null);
      return;
    }

    // Calculate distance in mm if calibrated
    let distanceMm = undefined;
    if (pixelsPerMm) {
      distanceMm = distancePixels / pixelsPerMm;
    }

    const newMeasurement: MeasurementLine = {
      id: measurementIdCounter.current++,
      x1: currentLine.x,
      y1: currentLine.y,
      x2: locationX,
      y2: locationY,
      distanceMm,
      label: `Measurement ${measurements.length + 1}`
    };

    setMeasurements([...measurements, newMeasurement]);
    setCurrentLine(null);
  };

  const deleteMeasurement = (id: number) => {
    setMeasurements(measurements.filter(m => m.id !== id));
  };

  const clearAllMeasurements = () => {
    Alert.alert(
      'Clear All Measurements',
      'Are you sure you want to delete all measurements?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: () => setMeasurements([])
        }
      ]
    );
  };

  const handleManualCalibration = () => {
    if (!calibrationDistance || isNaN(parseFloat(calibrationDistance))) {
      Alert.alert('Invalid Input', 'Please enter a valid distance in millimeters');
      return;
    }

    if (measurements.length === 0) {
      Alert.alert('No Reference Line', 'Please draw a line on a known distance (like a coin or ruler) first');
      return;
    }

    // Use the last measurement as calibration reference
    const refLine = measurements[measurements.length - 1];
    const dx = refLine.x2 - refLine.x1;
    const dy = refLine.y2 - refLine.y1;
    const distancePixels = Math.sqrt(dx * dx + dy * dy);
    const knownDistanceMm = parseFloat(calibrationDistance);

    const newPixelsPerMm = distancePixels / knownDistanceMm;
    setPixelsPerMm(newPixelsPerMm);

    // Recalculate all measurements with new calibration
    const updatedMeasurements = measurements.map(m => {
      const dx = m.x2 - m.x1;
      const dy = m.y2 - m.y1;
      const distancePixels = Math.sqrt(dx * dx + dy * dy);
      return {
        ...m,
        distanceMm: distancePixels / newPixelsPerMm
      };
    });

    setMeasurements(updatedMeasurements);
    setManualCalibration(false);
    Alert.alert('Calibration Updated', `New calibration: ${newPixelsPerMm.toFixed(2)} pixels/mm`);
  };

  const saveMeasurements = () => {
    if (onMeasurementsComplete) {
      onMeasurementsComplete(measurements);
    }
  };

  return (
    <View style={styles.container}>
      <ScrollView>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>📏 Measurement Tool</Text>
          {pixelsPerMm && (
            <Text style={styles.calibrationInfo}>
              ✓ Calibrated: {pixelsPerMm.toFixed(2)} px/mm
            </Text>
          )}
          {!pixelsPerMm && (
            <Text style={styles.calibrationWarning}>
              ⚠ No calibration - measurements in pixels only
            </Text>
          )}
        </View>

        {/* Calibration Detection Info */}
        {calibration.calibration_found && (
          <View style={styles.calibrationDetected}>
            <Text style={styles.calibrationDetectedText}>
              🎯 Auto-detected: {calibration.calibration_type}
            </Text>
            <Text style={styles.calibrationConfidence}>
              Confidence: {Math.round(calibration.confidence * 100)}%
            </Text>
          </View>
        )}

        {/* Instructions */}
        <View style={styles.instructions}>
          <Text style={styles.instructionsTitle}>How to measure:</Text>
          <Text style={styles.instructionsText}>1. Tap and drag on the image to draw a measurement line</Text>
          <Text style={styles.instructionsText}>2. Multiple measurements can be drawn</Text>
          <Text style={styles.instructionsText}>3. Tap on a measurement below to delete it</Text>
          {!pixelsPerMm && (
            <Text style={styles.instructionsText}>4. Use "Manual Calibration" if no automatic calibration detected</Text>
          )}
        </View>

        {/* Image with measurement overlay */}
        <View
          style={styles.imageContainer}
          onLayout={handleImageLayout}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
        >
          <Image
            source={{ uri: imageUri }}
            style={styles.image}
            resizeMode="contain"
          />

          {/* Canvas overlay for measurements using View components */}
          {imageLayout && measurements.map((measurement) => {
            const dx = measurement.x2 - measurement.x1;
            const dy = measurement.y2 - measurement.y1;
            const length = Math.sqrt(dx * dx + dy * dy);
            const angle = Math.atan2(dy, dx) * (180 / Math.PI);
            const midX = (measurement.x1 + measurement.x2) / 2;
            const midY = (measurement.y1 + measurement.y2) / 2;

            return (
              <React.Fragment key={measurement.id}>
                {/* Measurement line */}
                <View
                  style={[
                    styles.measurementLine,
                    {
                      left: measurement.x1,
                      top: measurement.y1,
                      width: length,
                      transform: [{ rotate: `${angle}deg` }],
                    }
                  ]}
                />
                {/* Start point */}
                <View
                  style={[
                    styles.measurementPoint,
                    { left: measurement.x1 - 6, top: measurement.y1 - 6 }
                  ]}
                />
                {/* End point */}
                <View
                  style={[
                    styles.measurementPoint,
                    { left: measurement.x2 - 6, top: measurement.y2 - 6 }
                  ]}
                />
                {/* Distance label */}
                {measurement.distanceMm && (
                  <View
                    style={[
                      styles.measurementLabel,
                      { left: midX - 40, top: midY - 25 }
                    ]}
                  >
                    <Text style={styles.measurementLabelText}>
                      {measurement.distanceMm.toFixed(1)}mm
                    </Text>
                  </View>
                )}
              </React.Fragment>
            );
          })}

          {/* Current line being drawn */}
          {currentLine && (
            <View
              style={[
                styles.currentPoint,
                { left: currentLine.x - 6, top: currentLine.y - 6 }
              ]}
            />
          )}
        </View>

        {/* Measurements list */}
        {measurements.length > 0 && (
          <View style={styles.measurementsList}>
            <Text style={styles.measurementsTitle}>Measurements:</Text>
            {measurements.map((measurement) => (
              <Pressable
                key={measurement.id}
                style={styles.measurementItem}
                onPress={() => deleteMeasurement(measurement.id)}
              >
                <Text style={styles.measurementItemLabel}>{measurement.label}</Text>
                <Text style={styles.measurementValue}>
                  {measurement.distanceMm
                    ? `${measurement.distanceMm.toFixed(2)} mm`
                    : 'Not calibrated'}
                </Text>
                <Text style={styles.deleteHint}>Tap to delete</Text>
              </Pressable>
            ))}
          </View>
        )}

        {/* Action buttons */}
        <View style={styles.actions}>
          {!pixelsPerMm && (
            <Pressable
              style={styles.calibrateButton}
              onPress={() => setManualCalibration(true)}
            >
              <Text style={styles.calibrateButtonText}>Manual Calibration</Text>
            </Pressable>
          )}

          {measurements.length > 0 && (
            <>
              <Pressable style={styles.clearButton} onPress={clearAllMeasurements}>
                <Text style={styles.clearButtonText}>Clear All</Text>
              </Pressable>

              <Pressable style={styles.saveButton} onPress={saveMeasurements}>
                <Text style={styles.saveButtonText}>Save Measurements</Text>
              </Pressable>
            </>
          )}
        </View>
      </ScrollView>

      {/* Manual calibration modal */}
      <Modal
        visible={manualCalibration}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setManualCalibration(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Manual Calibration</Text>
            <Text style={styles.modalInstructions}>
              Draw a line on a known distance (e.g., a coin or ruler mark), then enter the actual distance in millimeters:
            </Text>

            <TextInput
              style={styles.modalInput}
              placeholder="Distance in mm (e.g., 24.26 for US Quarter)"
              keyboardType="decimal-pad"
              value={calibrationDistance}
              onChangeText={setCalibrationDistance}
            />

            <View style={styles.modalButtons}>
              <Pressable
                style={styles.modalCancelButton}
                onPress={() => setManualCalibration(false)}
              >
                <Text style={styles.modalCancelButtonText}>Cancel</Text>
              </Pressable>

              <Pressable
                style={styles.modalSaveButton}
                onPress={handleManualCalibration}
              >
                <Text style={styles.modalSaveButtonText}>Set Calibration</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  calibrationInfo: {
    fontSize: 14,
    color: '#28a745',
    fontWeight: '600',
  },
  calibrationWarning: {
    fontSize: 14,
    color: '#ffc107',
    fontWeight: '600',
  },
  calibrationDetected: {
    backgroundColor: '#e8f5e9',
    padding: 12,
    margin: 16,
    borderRadius: 8,
  },
  calibrationDetectedText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2e7d32',
    marginBottom: 4,
  },
  calibrationConfidence: {
    fontSize: 12,
    color: '#558b2f',
  },
  instructions: {
    backgroundColor: '#fff',
    padding: 16,
    margin: 16,
    marginTop: 0,
    borderRadius: 8,
  },
  instructionsTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
  },
  instructionsText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  imageContainer: {
    width: SCREEN_WIDTH - 32,
    height: SCREEN_WIDTH - 32,
    backgroundColor: '#000',
    margin: 16,
    marginTop: 0,
    borderRadius: 8,
    overflow: 'hidden',
    position: 'relative',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  measurementLine: {
    position: 'absolute',
    height: 3,
    backgroundColor: '#00ff00',
    transformOrigin: 'left',
  },
  measurementPoint: {
    position: 'absolute',
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#00ff00',
    borderWidth: 2,
    borderColor: '#fff',
  },
  currentPoint: {
    position: 'absolute',
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#ffff00',
    borderWidth: 2,
    borderColor: '#fff',
  },
  measurementLabel: {
    position: 'absolute',
    backgroundColor: 'rgba(0, 255, 0, 0.9)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    minWidth: 80,
  },
  measurementLabelText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  measurementsList: {
    backgroundColor: '#fff',
    padding: 16,
    margin: 16,
    marginTop: 0,
    borderRadius: 8,
  },
  measurementsTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  measurementItem: {
    padding: 12,
    backgroundColor: '#f8f9fa',
    borderRadius: 6,
    marginBottom: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#00ff00',
  },
  measurementItemLabel: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
  },
  measurementValue: {
    fontSize: 16,
    color: '#2c5282',
    marginBottom: 4,
  },
  deleteHint: {
    fontSize: 12,
    color: '#999',
    fontStyle: 'italic',
  },
  actions: {
    padding: 16,
    gap: 12,
  },
  calibrateButton: {
    backgroundColor: '#2196F3',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  calibrateButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  clearButton: {
    backgroundColor: '#dc3545',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  clearButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  saveButton: {
    backgroundColor: '#28a745',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  saveButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalContent: {
    backgroundColor: '#fff',
    padding: 24,
    borderRadius: 12,
    width: SCREEN_WIDTH - 64,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  modalInstructions: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
  },
  modalInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 6,
    padding: 12,
    fontSize: 16,
    marginBottom: 16,
  },
  modalButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  modalCancelButton: {
    flex: 1,
    padding: 12,
    backgroundColor: '#6c757d',
    borderRadius: 6,
    alignItems: 'center',
  },
  modalCancelButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  modalSaveButton: {
    flex: 1,
    padding: 12,
    backgroundColor: '#28a745',
    borderRadius: 6,
    alignItems: 'center',
  },
  modalSaveButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
