import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  ScrollView,
  Dimensions
} from 'react-native';
import { Svg, Circle, Path, G, Text as SvgText } from 'react-native-svg';
import { HelpTooltip, InlineHelp } from './HelpTooltip';

const { width } = Dimensions.get('window');
const BODY_MAP_WIDTH = Math.min(width - 40, 400);
const BODY_MAP_HEIGHT = BODY_MAP_WIDTH * 1.6; // 5:8 aspect ratio for body diagram

interface BodyMapSelectorProps {
  onLocationSelect: (location: {
    body_location: string;
    body_sublocation?: string;
    body_side?: string;
    body_map_x: number;
    body_map_y: number;
  }) => void;
  selectedLocation?: {
    body_location?: string;
    body_sublocation?: string;
    body_side?: string;
    body_map_coordinates?: { x: number; y: number };
  };
}

// Body regions mapping
const BODY_REGIONS = {
  // Head regions (y: 5-15%)
  head: { label: 'Head', centerX: 50, centerY: 8, radius: 8 },
  face: { label: 'Face', centerX: 50, centerY: 10, radius: 5 },

  // Neck (y: 15-18%)
  neck: { label: 'Neck', centerX: 50, centerY: 16, radius: 4 },

  // Upper body (y: 18-45%)
  chest: { label: 'Chest', centerX: 50, centerY: 28, radius: 10 },
  abdomen: { label: 'Abdomen', centerX: 50, centerY: 40, radius: 9 },
  back_upper: { label: 'Upper Back', centerX: 50, centerY: 28, radius: 10 },
  back_lower: { label: 'Lower Back', centerX: 50, centerY: 40, radius: 9 },

  // Arms
  shoulder_left: { label: 'Left Shoulder', centerX: 25, centerY: 22, radius: 6 },
  shoulder_right: { label: 'Right Shoulder', centerX: 75, centerY: 22, radius: 6 },
  arm_left_upper: { label: 'Left Upper Arm', centerX: 20, centerY: 32, radius: 5 },
  arm_right_upper: { label: 'Right Upper Arm', centerX: 80, centerY: 32, radius: 5 },
  arm_left_lower: { label: 'Left Forearm', centerX: 15, centerY: 42, radius: 4 },
  arm_right_lower: { label: 'Right Forearm', centerX: 85, centerY: 42, radius: 4 },
  hand_left: { label: 'Left Hand', centerX: 12, centerY: 50, radius: 4 },
  hand_right: { label: 'Right Hand', centerX: 88, centerY: 50, radius: 4 },

  // Legs (y: 45-95%)
  hip_left: { label: 'Left Hip', centerX: 42, centerY: 48, radius: 5 },
  hip_right: { label: 'Right Hip', centerX: 58, centerY: 48, radius: 5 },
  thigh_left: { label: 'Left Thigh', centerX: 42, centerY: 60, radius: 6 },
  thigh_right: { label: 'Right Thigh', centerX: 58, centerY: 60, radius: 6 },
  knee_left: { label: 'Left Knee', centerX: 42, centerY: 70, radius: 4 },
  knee_right: { label: 'Right Knee', centerX: 58, centerY: 70, radius: 4 },
  leg_left_lower: { label: 'Left Lower Leg', centerX: 42, centerY: 80, radius: 4 },
  leg_right_lower: { label: 'Right Lower Leg', centerX: 58, centerY: 80, radius: 4 },
  foot_left: { label: 'Left Foot', centerX: 42, centerY: 90, radius: 4 },
  foot_right: { label: 'Right Foot', centerX: 58, centerY: 90, radius: 4 },
};

export default function BodyMapSelector({ onLocationSelect, selectedLocation }: BodyMapSelectorProps) {
  const [view, setView] = useState<'front' | 'back'>('front');
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);

  const handleMapPress = (regionKey: string, region: typeof BODY_REGIONS[keyof typeof BODY_REGIONS]) => {
    const x = region.centerX;
    const y = region.centerY;

    // Determine body side
    let bodySide = 'center';
    if (regionKey.includes('_left')) {
      bodySide = 'left';
    } else if (regionKey.includes('_right')) {
      bodySide = 'right';
    }

    // Determine body location and sublocation
    const bodyLocation = regionKey.replace(/_left|_right|_upper|_lower/g, '');
    let bodySublocation = '';
    if (regionKey.includes('_upper')) {
      bodySublocation = 'upper';
    } else if (regionKey.includes('_lower')) {
      bodySublocation = 'lower';
    }

    setSelectedRegion(regionKey);

    onLocationSelect({
      body_location: bodyLocation,
      body_sublocation: bodySublocation || undefined,
      body_side: bodySide,
      body_map_x: x,
      body_map_y: y
    });
  };

  const renderBodyDiagram = () => {
    const visibleRegions = Object.entries(BODY_REGIONS).filter(([key]) => {
      if (view === 'front') {
        return !key.includes('back_');
      } else {
        // For back view, show back-specific regions and exclude front-specific ones
        return key.includes('back_') || !key.includes('chest') && !key.includes('abdomen') && !key.includes('face');
      }
    });

    return (
      <Svg width={BODY_MAP_WIDTH} height={BODY_MAP_HEIGHT} style={styles.bodySvg}>
        {/* Body outline */}
        <Path
          d={`
            M ${BODY_MAP_WIDTH * 0.5} ${BODY_MAP_HEIGHT * 0.05}
            Q ${BODY_MAP_WIDTH * 0.5} ${BODY_MAP_HEIGHT * 0.15} ${BODY_MAP_WIDTH * 0.5} ${BODY_MAP_HEIGHT * 0.18}
            L ${BODY_MAP_WIDTH * 0.45} ${BODY_MAP_HEIGHT * 0.45}
            L ${BODY_MAP_WIDTH * 0.42} ${BODY_MAP_HEIGHT * 0.95}
            M ${BODY_MAP_WIDTH * 0.5} ${BODY_MAP_HEIGHT * 0.18}
            L ${BODY_MAP_WIDTH * 0.55} ${BODY_MAP_HEIGHT * 0.45}
            L ${BODY_MAP_WIDTH * 0.58} ${BODY_MAP_HEIGHT * 0.95}
          `}
          stroke="#cbd5e0"
          strokeWidth="2"
          fill="none"
        />

        {/* Clickable body regions */}
        {visibleRegions.map(([key, region]) => {
          const cx = (region.centerX / 100) * BODY_MAP_WIDTH;
          const cy = (region.centerY / 100) * BODY_MAP_HEIGHT;
          const r = (region.radius / 100) * BODY_MAP_WIDTH;

          const isSelected = selectedRegion === key ||
            (selectedLocation?.body_map_coordinates?.x === region.centerX &&
             selectedLocation?.body_map_coordinates?.y === region.centerY);

          return (
            <G key={key}>
              <Circle
                cx={cx}
                cy={cy}
                r={r}
                fill={isSelected ? '#4299e1' : '#e2e8f0'}
                fillOpacity={0.6}
                stroke={isSelected ? '#2b6cb0' : '#94a3b8'}
                strokeWidth={isSelected ? 3 : 1}
                onPress={() => handleMapPress(key, region)}
              />
              {isSelected && (
                <SvgText
                  x={cx}
                  y={cy + r + 12}
                  fontSize="10"
                  fill="#2d3748"
                  textAnchor="middle"
                  fontWeight="bold"
                >
                  {region.label}
                </SvgText>
              )}
            </G>
          );
        })}
      </Svg>
    );
  };

  return (
    <View style={styles.container}>
      <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 8 }}>
        <Text style={styles.title}>Select Lesion Location</Text>
        <HelpTooltip
          title="Body Map Location"
          content="Selecting the lesion location helps track lesions over time, identify patterns, and correlate findings with body regions. This information is stored with your analysis for future reference and can help your dermatologist assess changes. You can tap on the body diagram or select from the list below."
          size={20}
          color="#2d3748"
        />
      </View>
      <InlineHelp text="Tap on the body diagram below to mark where the lesion is located, or select from the list" color="#718096" />

      {/* View toggle */}
      <View style={styles.viewToggle}>
        <Pressable
          style={[styles.viewButton, view === 'front' && styles.viewButtonActive]}
          onPress={() => setView('front')}
        >
          <Text style={[styles.viewButtonText, view === 'front' && styles.viewButtonTextActive]}>
            Front View
          </Text>
        </Pressable>
        <Pressable
          style={[styles.viewButton, view === 'back' && styles.viewButtonActive]}
          onPress={() => setView('back')}
        >
          <Text style={[styles.viewButtonText, view === 'back' && styles.viewButtonTextActive]}>
            Back View
          </Text>
        </Pressable>
        <HelpTooltip
          title="Front/Back View"
          content="Toggle between front and back views of the body to select the correct location. The front view shows the face, chest, abdomen, and front of limbs. The back view shows the upper and lower back, and back of limbs."
          size={18}
          color="#4299e1"
        />
      </View>

      {/* Body diagram */}
      <View style={styles.diagramContainer}>
        {renderBodyDiagram()}
      </View>

      {/* Selected location info */}
      {selectedRegion && (
        <View style={styles.selectedInfo}>
          <Text style={styles.selectedLabel}>Selected Location:</Text>
          <Text style={styles.selectedValue}>{BODY_REGIONS[selectedRegion]?.label}</Text>
        </View>
      )}

      {/* Quick select list */}
      <Text style={styles.quickSelectTitle}>Or select from list:</Text>
      <ScrollView style={styles.quickSelectScroll} contentContainerStyle={styles.quickSelectContent}>
        {Object.entries(BODY_REGIONS).map(([key, region]) => (
          <Pressable
            key={key}
            style={[
              styles.quickSelectItem,
              selectedRegion === key && styles.quickSelectItemActive
            ]}
            onPress={() => handleMapPress(key, region)}
          >
            <Text style={[
              styles.quickSelectItemText,
              selectedRegion === key && styles.quickSelectItemTextActive
            ]}>
              {region.label}
            </Text>
          </Pressable>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2d3748',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#718096',
    marginBottom: 16,
  },
  viewToggle: {
    flexDirection: 'row',
    marginBottom: 16,
    gap: 8,
  },
  viewButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#cbd5e0',
    backgroundColor: '#fff',
    alignItems: 'center',
  },
  viewButtonActive: {
    backgroundColor: '#4299e1',
    borderColor: '#2b6cb0',
  },
  viewButtonText: {
    fontSize: 14,
    color: '#4a5568',
    fontWeight: '600',
  },
  viewButtonTextActive: {
    color: '#fff',
  },
  diagramContainer: {
    alignItems: 'center',
    marginBottom: 20,
    backgroundColor: '#f7fafc',
    borderRadius: 12,
    padding: 16,
  },
  bodySvg: {
    backgroundColor: 'transparent',
  },
  selectedInfo: {
    backgroundColor: '#ebf8ff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  selectedLabel: {
    fontSize: 12,
    color: '#2c5282',
    fontWeight: '600',
  },
  selectedValue: {
    fontSize: 16,
    color: '#2b6cb0',
    fontWeight: 'bold',
    marginTop: 4,
  },
  quickSelectTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2d3748',
    marginBottom: 8,
  },
  quickSelectScroll: {
    maxHeight: 200,
  },
  quickSelectContent: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  quickSelectItem: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
    backgroundColor: '#edf2f7',
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  quickSelectItemActive: {
    backgroundColor: '#4299e1',
    borderColor: '#2b6cb0',
  },
  quickSelectItemText: {
    fontSize: 13,
    color: '#4a5568',
    fontWeight: '500',
  },
  quickSelectItemTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
});
