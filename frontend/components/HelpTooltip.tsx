import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Modal, StyleSheet, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';

interface HelpTooltipProps {
  title: string;
  content: string;
  size?: number;
  color?: string;
  iconStyle?: 'circle' | 'info' | 'question';
  position?: 'top' | 'bottom' | 'left' | 'right';
}

/**
 * Reusable HelpTooltip component for displaying contextual help
 * Usage: <HelpTooltip title="Feature Name" content="Detailed explanation..." />
 */
export const HelpTooltip: React.FC<HelpTooltipProps> = ({
  title,
  content,
  size = 20,
  color = '#3b82f6',
  iconStyle = 'info',
  position = 'bottom'
}) => {
  const { t } = useTranslation();
  const [modalVisible, setModalVisible] = useState(false);

  const getIconName = () => {
    switch (iconStyle) {
      case 'circle':
        return 'information-circle';
      case 'question':
        return 'help-circle';
      case 'info':
      default:
        return 'information-circle-outline';
    }
  };

  return (
    <>
      <TouchableOpacity
        onPress={() => setModalVisible(true)}
        style={styles.iconContainer}
        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
      >
        <Ionicons name={getIconName()} size={size} color={color} />
      </TouchableOpacity>

      <Modal
        animationType="fade"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setModalVisible(false)}
        >
          <View style={styles.modalContainer}>
            <TouchableOpacity
              activeOpacity={1}
              onPress={(e) => e.stopPropagation()}
              style={styles.modalContent}
            >
              {/* Header */}
              <View style={styles.modalHeader}>
                <Ionicons name={getIconName()} size={24} color={color} />
                <Text style={styles.modalTitle}>{title}</Text>
                <TouchableOpacity
                  onPress={() => setModalVisible(false)}
                  style={styles.closeButton}
                  hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                >
                  <Ionicons name="close" size={24} color="#666" />
                </TouchableOpacity>
              </View>

              {/* Content */}
              <ScrollView style={styles.modalBody} showsVerticalScrollIndicator={true}>
                <Text style={styles.modalText}>{content}</Text>
              </ScrollView>

              {/* Footer */}
              <View style={styles.modalFooter}>
                <TouchableOpacity
                  style={styles.gotItButton}
                  onPress={() => setModalVisible(false)}
                >
                  <Text style={styles.gotItButtonText}>{t('common.gotIt')}</Text>
                </TouchableOpacity>
              </View>
            </TouchableOpacity>
          </View>
        </TouchableOpacity>
      </Modal>
    </>
  );
};

/**
 * Inline help text component (simpler alternative to modal tooltip)
 */
interface InlineHelpProps {
  text: string;
  color?: string;
}

export const InlineHelp: React.FC<InlineHelpProps> = ({ text, color = '#666' }) => {
  return (
    <View style={styles.inlineHelpContainer}>
      <Ionicons name="information-circle-outline" size={16} color={color} style={styles.inlineIcon} />
      <Text style={[styles.inlineHelpText, { color }]}>{text}</Text>
    </View>
  );
};

/**
 * Help badge for highlighting important information
 */
interface HelpBadgeProps {
  text: string;
  type?: 'info' | 'warning' | 'success' | 'error';
}

export const HelpBadge: React.FC<HelpBadgeProps> = ({ text, type = 'info' }) => {
  const badgeStyles = {
    info: { bg: '#dbeafe', text: '#1e40af', icon: 'information-circle' },
    warning: { bg: '#fef3c7', text: '#92400e', icon: 'warning' },
    success: { bg: '#d1fae5', text: '#065f46', icon: 'checkmark-circle' },
    error: { bg: '#fee2e2', text: '#991b1b', icon: 'alert-circle' },
  };

  const style = badgeStyles[type];

  return (
    <View style={[styles.helpBadge, { backgroundColor: style.bg }]}>
      <Ionicons name={style.icon as any} size={16} color={style.text} style={styles.badgeIcon} />
      <Text style={[styles.helpBadgeText, { color: style.text }]}>{text}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  iconContainer: {
    padding: 4,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContainer: {
    width: '90%',
    maxWidth: 500,
    maxHeight: '80%',
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  modalTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginLeft: 12,
  },
  closeButton: {
    padding: 4,
  },
  modalBody: {
    padding: 20,
    maxHeight: 400,
  },
  modalText: {
    fontSize: 15,
    lineHeight: 24,
    color: '#374151',
  },
  modalFooter: {
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  gotItButton: {
    backgroundColor: '#3b82f6',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    alignItems: 'center',
  },
  gotItButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  inlineHelpContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
    marginBottom: 8,
  },
  inlineIcon: {
    marginRight: 6,
  },
  inlineHelpText: {
    fontSize: 13,
    lineHeight: 18,
    flex: 1,
  },
  helpBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    marginVertical: 8,
  },
  badgeIcon: {
    marginRight: 8,
  },
  helpBadgeText: {
    fontSize: 13,
    lineHeight: 18,
    flex: 1,
  },
});

export default HelpTooltip;
