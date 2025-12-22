import React from 'react';
import { View, Text, StyleSheet, ScrollView, Dimensions } from 'react-native';
import Svg, { Circle, Line, Text as SvgText } from 'react-native-svg';

interface FamilyMember {
  id: number;
  name?: string;
  relationship_type: string;
  relationship_side?: string;
  has_skin_cancer: boolean;
  has_melanoma: boolean;
  gender?: string;
}

interface Props {
  familyMembers: FamilyMember[];
}

export default function FamilyTreeVisualization({ familyMembers }: Props) {
  const screenWidth = Dimensions.get('window').width - 40;

  // Group family members by generation
  const generations = {
    grandparents: familyMembers.filter(m => m.relationship_type === 'grandparent'),
    parents: familyMembers.filter(m => m.relationship_type === 'parent'),
    auntsUncles: familyMembers.filter(m => m.relationship_type === 'aunt_uncle'),
    siblings: familyMembers.filter(m => m.relationship_type === 'sibling'),
    user: [{ id: 0, name: 'You', relationship_type: 'self', has_skin_cancer: false, has_melanoma: false }],
    children: familyMembers.filter(m => m.relationship_type === 'child'),
    cousins: familyMembers.filter(m => m.relationship_type === 'cousin')
  };

  const getMemberColor = (member: FamilyMember) => {
    if (member.has_melanoma) return '#dc2626'; // Red for melanoma
    if (member.has_skin_cancer) return '#f59e0b'; // Orange for other skin cancer
    return '#10b981'; // Green for healthy
  };

  const getMemberShape = (member: FamilyMember) => {
    if (member.gender === 'male') return 'square';
    if (member.gender === 'female') return 'circle';
    return 'diamond';
  };

  const renderMember = (member: FamilyMember, x: number, y: number, size: number) => {
    const color = getMemberColor(member);
    const shape = getMemberShape(member);
    const label = member.name || member.relationship_type;

    return (
      <View key={member.id}>
        {shape === 'circle' && (
          <Circle
            cx={x}
            cy={y}
            r={size / 2}
            fill={color}
            stroke="#1f2937"
            strokeWidth={2}
          />
        )}
        {shape === 'square' && (
          <View
            style={[
              styles.squareNode,
              {
                left: x - size / 2,
                top: y - size / 2,
                width: size,
                height: size,
                backgroundColor: color,
                borderColor: '#1f2937',
                borderWidth: 2,
                position: 'absolute'
              }
            ]}
          />
        )}
        <SvgText
          x={x}
          y={y + size + 15}
          fontSize="10"
          fill="#1f2937"
          textAnchor="middle"
        >
          {label.length > 10 ? label.substring(0, 10) + '...' : label}
        </SvgText>
      </View>
    );
  };

  const nodeSize = 30;
  const levelHeight = 100;
  const nodeSpacing = 80;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Family Tree</Text>

      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendSquare, { backgroundColor: '#dc2626' }]} />
          <Text style={styles.legendText}>Melanoma</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendSquare, { backgroundColor: '#f59e0b' }]} />
          <Text style={styles.legendText}>Skin Cancer</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendSquare, { backgroundColor: '#10b981' }]} />
          <Text style={styles.legendText}>Healthy</Text>
        </View>
      </View>

      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendCircle, { backgroundColor: '#6b7280' }]} />
          <Text style={styles.legendText}>Female</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendSquare, { backgroundColor: '#6b7280', width: 16, height: 16 }]} />
          <Text style={styles.legendText}>Male</Text>
        </View>
      </View>

      <ScrollView horizontal showsHorizontalScrollIndicator={true} style={styles.scrollView}>
        <View style={styles.treeContainer}>
          <Svg width={Math.max(screenWidth, nodeSpacing * 6)} height={levelHeight * 5}>
            {/* Grandparents Level */}
            {generations.grandparents.map((member, index) => {
              const x = (index + 1) * nodeSpacing;
              const y = 30;
              return renderMember(member, x, y, nodeSize);
            })}

            {/* Parents Level */}
            {generations.parents.map((member, index) => {
              const x = (index + 2) * nodeSpacing;
              const y = levelHeight + 30;

              // Draw connection to grandparents
              const side = member.relationship_side;
              const grandparentIndex = side === 'maternal' ? 0 : 1;
              if (grandparentIndex < generations.grandparents.length) {
                const parentX = (grandparentIndex + 1) * nodeSpacing;
                const parentY = 30 + nodeSize / 2;

                <Line
                  x1={parentX}
                  y1={parentY}
                  x2={x}
                  y2={y - nodeSize / 2}
                  stroke="#9ca3af"
                  strokeWidth={2}
                />
              }

              return renderMember(member, x, y, nodeSize);
            })}

            {/* Aunts/Uncles Level */}
            {generations.auntsUncles.map((member, index) => {
              const x = (index + 1) * nodeSpacing;
              const y = levelHeight + 30;
              return renderMember(member, x, y, nodeSize);
            })}

            {/* User Level (center) */}
            {generations.user.map((member, index) => {
              const x = (screenWidth > nodeSpacing * 6 ? screenWidth : nodeSpacing * 6) / 2;
              const y = levelHeight * 2 + 30;
              return renderMember(member as FamilyMember, x, y, nodeSize + 10); // Larger node for user
            })}

            {/* Siblings Level */}
            {generations.siblings.map((member, index) => {
              const userX = (screenWidth > nodeSpacing * 6 ? screenWidth : nodeSpacing * 6) / 2;
              const x = userX + (index + 1 - generations.siblings.length / 2) * nodeSpacing;
              const y = levelHeight * 2 + 30;
              return renderMember(member, x, y, nodeSize);
            })}

            {/* Children Level */}
            {generations.children.map((member, index) => {
              const userX = (screenWidth > nodeSpacing * 6 ? screenWidth : nodeSpacing * 6) / 2;
              const x = userX + (index - generations.children.length / 2) * nodeSpacing;
              const y = levelHeight * 3 + 30;
              return renderMember(member, x, y, nodeSize);
            })}

            {/* Cousins Level */}
            {generations.cousins.map((member, index) => {
              const x = (index + 1) * nodeSpacing;
              const y = levelHeight * 2 + 30;
              return renderMember(member, x, y, nodeSize);
            })}
          </Svg>
        </View>
      </ScrollView>

      <View style={styles.summaryCard}>
        <Text style={styles.summaryTitle}>Family Summary</Text>
        <Text style={styles.summaryText}>
          {generations.grandparents.filter(m => m.has_skin_cancer).length} / {generations.grandparents.length} Grandparents affected
        </Text>
        <Text style={styles.summaryText}>
          {generations.parents.filter(m => m.has_skin_cancer).length} / {generations.parents.length} Parents affected
        </Text>
        <Text style={styles.summaryText}>
          {generations.siblings.filter(m => m.has_skin_cancer).length} / {generations.siblings.length} Siblings affected
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 16,
    marginVertical: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 12,
  },
  legend: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 12,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendSquare: {
    width: 20,
    height: 20,
    borderRadius: 2,
  },
  legendCircle: {
    width: 20,
    height: 20,
    borderRadius: 10,
  },
  legendText: {
    fontSize: 12,
    color: '#6b7280',
  },
  scrollView: {
    maxHeight: 400,
  },
  treeContainer: {
    paddingVertical: 20,
  },
  squareNode: {
    borderRadius: 4,
  },
  summaryCard: {
    backgroundColor: '#ede9fe',
    borderRadius: 8,
    padding: 12,
    marginTop: 16,
  },
  summaryTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#7c3aed',
    marginBottom: 8,
  },
  summaryText: {
    fontSize: 13,
    color: '#5b21b6',
    marginBottom: 4,
  },
});
