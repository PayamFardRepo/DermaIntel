import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
  RefreshControl,
  Modal,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../config';

interface PaymentConfig {
  stripe_configured: boolean;
  stripe_publishable_key: string | null;
  supported_currencies: string[];
  payment_methods: string[];
  demo_mode: boolean;
}

interface Consultation {
  id: number;
  dermatologist_name?: string;
  consultation_type: string;
  scheduled_datetime?: string;
  consultation_fee?: number;
  payment_status: string;
  status: string;
  created_at: string;
}

interface PaymentIntent {
  client_secret: string;
  payment_intent_id: string;
  amount: number;
  currency: string;
  demo_mode?: boolean;
}

interface CreditPackage {
  id: string;
  name: string;
  credits: number;
  price: number;
  savings?: string;
}

const CREDIT_PACKAGES: CreditPackage[] = [
  { id: 'basic', name: 'Basic', credits: 1, price: 75 },
  { id: 'standard', name: 'Standard', credits: 3, price: 200, savings: 'Save $25' },
  { id: 'premium', name: 'Premium', credits: 5, price: 300, savings: 'Save $75' },
];

export default function PaymentsScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'pending' | 'history' | 'credits'>('pending');
  const [paymentConfig, setPaymentConfig] = useState<PaymentConfig | null>(null);
  const [consultations, setConsultations] = useState<Consultation[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Payment modal
  const [showPaymentModal, setShowPaymentModal] = useState(false);
  const [selectedConsultation, setSelectedConsultation] = useState<Consultation | null>(null);
  const [paymentIntent, setPaymentIntent] = useState<PaymentIntent | null>(null);
  const [processing, setProcessing] = useState(false);

  // Card form (for demo purposes)
  const [cardNumber, setCardNumber] = useState('');
  const [expiryDate, setExpiryDate] = useState('');
  const [cvv, setCvv] = useState('');
  const [cardholderName, setCardholderName] = useState('');

  // Credits purchase modal
  const [showCreditsModal, setShowCreditsModal] = useState(false);
  const [selectedPackage, setSelectedPackage] = useState<CreditPackage | null>(null);

  // Payment history (stored locally for demo)
  const [paymentHistory, setPaymentHistory] = useState<any[]>([]);

  useEffect(() => {
    fetchPaymentConfig();
    fetchConsultations();
    loadPaymentHistory();
  }, []);

  const fetchPaymentConfig = async () => {
    try {
      const token = await AsyncStorage.getItem('userToken');
      const response = await fetch(`${API_URL}/payments/config`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setPaymentConfig(data);
      }
    } catch (error) {
      console.error('Error fetching payment config:', error);
    }
  };

  const fetchConsultations = async () => {
    try {
      const token = await AsyncStorage.getItem('userToken');
      const response = await fetch(`${API_URL}/consultations`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setConsultations(data.consultations || []);
      }
    } catch (error) {
      console.error('Error fetching consultations:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const loadPaymentHistory = async () => {
    try {
      const history = await AsyncStorage.getItem('paymentHistory');
      if (history) {
        setPaymentHistory(JSON.parse(history));
      }
    } catch (error) {
      console.error('Error loading payment history:', error);
    }
  };

  const savePaymentHistory = async (payment: any) => {
    try {
      const newHistory = [payment, ...paymentHistory];
      setPaymentHistory(newHistory);
      await AsyncStorage.setItem('paymentHistory', JSON.stringify(newHistory));
    } catch (error) {
      console.error('Error saving payment history:', error);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchConsultations();
  }, []);

  const createPaymentIntent = async (consultation: Consultation) => {
    try {
      const token = await AsyncStorage.getItem('userToken');
      const formData = new FormData();
      formData.append('consultation_id', consultation.id.toString());
      formData.append('amount', (consultation.consultation_fee || 75).toString());
      formData.append('currency', 'usd');

      const response = await fetch(`${API_URL}/payments/create-intent`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        return data as PaymentIntent;
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to create payment');
        return null;
      }
    } catch (error) {
      console.error('Error creating payment intent:', error);
      Alert.alert('Error', 'Failed to create payment');
      return null;
    }
  };

  const openPaymentModal = async (consultation: Consultation) => {
    setSelectedConsultation(consultation);
    setShowPaymentModal(true);
    setProcessing(true);

    const intent = await createPaymentIntent(consultation);
    setPaymentIntent(intent);
    setProcessing(false);

    // Reset form
    setCardNumber('');
    setExpiryDate('');
    setCvv('');
    setCardholderName('');
  };

  const handlePayment = async () => {
    if (!selectedConsultation || !paymentIntent) return;

    // Validate card (basic validation for demo)
    if (!cardNumber || cardNumber.length < 16) {
      Alert.alert('Invalid Card', 'Please enter a valid card number');
      return;
    }
    if (!expiryDate || expiryDate.length < 5) {
      Alert.alert('Invalid Expiry', 'Please enter a valid expiry date (MM/YY)');
      return;
    }
    if (!cvv || cvv.length < 3) {
      Alert.alert('Invalid CVV', 'Please enter a valid CVV');
      return;
    }

    setProcessing(true);
    try {
      const token = await AsyncStorage.getItem('userToken');
      const formData = new FormData();
      formData.append('consultation_id', selectedConsultation.id.toString());
      formData.append('payment_intent_id', paymentIntent.payment_intent_id);

      const response = await fetch(`${API_URL}/payments/confirm`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        // Save to payment history
        savePaymentHistory({
          id: paymentIntent.payment_intent_id,
          consultation_id: selectedConsultation.id,
          amount: paymentIntent.amount,
          currency: paymentIntent.currency,
          status: 'paid',
          description: `Consultation with ${selectedConsultation.dermatologist_name || 'Dermatologist'}`,
          date: new Date().toISOString(),
        });

        Alert.alert('Success', 'Payment completed successfully!');
        setShowPaymentModal(false);
        fetchConsultations();
      } else {
        const error = await response.json();
        Alert.alert('Payment Failed', error.detail || 'Payment could not be processed');
      }
    } catch (error) {
      console.error('Error processing payment:', error);
      Alert.alert('Error', 'Payment processing failed');
    } finally {
      setProcessing(false);
    }
  };

  const handlePurchaseCredits = async () => {
    if (!selectedPackage) return;

    // For demo, just simulate purchase
    setProcessing(true);
    setTimeout(() => {
      savePaymentHistory({
        id: `cr_${Date.now()}`,
        type: 'credits',
        package: selectedPackage.name,
        credits: selectedPackage.credits,
        amount: selectedPackage.price,
        currency: 'usd',
        status: 'paid',
        date: new Date().toISOString(),
      });

      Alert.alert('Success', `Successfully purchased ${selectedPackage.credits} credits!`);
      setShowCreditsModal(false);
      setSelectedPackage(null);
      setProcessing(false);
    }, 1500);
  };

  const formatCurrency = (amount: number, currency: string = 'usd') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency.toUpperCase(),
    }).format(amount);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatCardNumber = (value: string) => {
    const cleaned = value.replace(/\D/g, '');
    const formatted = cleaned.match(/.{1,4}/g)?.join(' ') || cleaned;
    return formatted.substring(0, 19);
  };

  const formatExpiryDate = (value: string) => {
    const cleaned = value.replace(/\D/g, '');
    if (cleaned.length >= 2) {
      return cleaned.substring(0, 2) + '/' + cleaned.substring(2, 4);
    }
    return cleaned;
  };

  const getPaymentStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'paid': return '#4CAF50';
      case 'pending': return '#FF9800';
      case 'failed': return '#F44336';
      case 'refunded': return '#9C27B0';
      default: return '#9E9E9E';
    }
  };

  const getPendingConsultations = () => {
    return consultations.filter(c =>
      c.payment_status === 'pending' || !c.payment_status
    );
  };

  const renderPendingTab = () => {
    const pending = getPendingConsultations();

    return (
      <View style={styles.tabContent}>
        {pending.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons name="checkmark-circle-outline" size={64} color="rgba(255,255,255,0.5)" />
            <Text style={styles.emptyText}>No pending payments</Text>
            <Text style={styles.emptySubtext}>
              All your consultations are paid up!
            </Text>
          </View>
        ) : (
          pending.map((consultation) => (
            <View key={consultation.id} style={styles.pendingCard}>
              <View style={styles.pendingHeader}>
                <View style={styles.pendingIcon}>
                  <Ionicons name="videocam" size={24} color="#667eea" />
                </View>
                <View style={styles.pendingInfo}>
                  <Text style={styles.pendingTitle}>
                    {consultation.dermatologist_name || 'Video Consultation'}
                  </Text>
                  <Text style={styles.pendingType}>{consultation.consultation_type}</Text>
                  {consultation.scheduled_datetime && (
                    <Text style={styles.pendingDate}>
                      Scheduled: {formatDateTime(consultation.scheduled_datetime)}
                    </Text>
                  )}
                </View>
              </View>

              <View style={styles.pendingAmount}>
                <Text style={styles.amountLabel}>Amount Due</Text>
                <Text style={styles.amountValue}>
                  {formatCurrency(consultation.consultation_fee || 75)}
                </Text>
              </View>

              <TouchableOpacity
                style={styles.payButton}
                onPress={() => openPaymentModal(consultation)}
              >
                <Ionicons name="card" size={20} color="#fff" />
                <Text style={styles.payButtonText}>Pay Now</Text>
              </TouchableOpacity>
            </View>
          ))
        )}
      </View>
    );
  };

  const renderHistoryTab = () => (
    <View style={styles.tabContent}>
      {paymentHistory.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="receipt-outline" size={64} color="rgba(255,255,255,0.5)" />
          <Text style={styles.emptyText}>No payment history</Text>
          <Text style={styles.emptySubtext}>
            Your completed payments will appear here
          </Text>
        </View>
      ) : (
        paymentHistory.map((payment, index) => (
          <View key={payment.id || index} style={styles.historyCard}>
            <View style={styles.historyHeader}>
              <View style={styles.historyIcon}>
                <Ionicons
                  name={payment.type === 'credits' ? 'star' : 'checkmark-circle'}
                  size={20}
                  color="#4CAF50"
                />
              </View>
              <View style={styles.historyInfo}>
                <Text style={styles.historyTitle}>
                  {payment.type === 'credits'
                    ? `${payment.credits} Second Opinion Credits`
                    : payment.description || 'Consultation Payment'}
                </Text>
                <Text style={styles.historyDate}>{formatDate(payment.date)}</Text>
              </View>
              <View style={styles.historyAmount}>
                <Text style={styles.historyAmountValue}>
                  {formatCurrency(payment.amount, payment.currency)}
                </Text>
                <View style={[
                  styles.historyStatus,
                  { backgroundColor: getPaymentStatusColor(payment.status) }
                ]}>
                  <Text style={styles.historyStatusText}>{payment.status}</Text>
                </View>
              </View>
            </View>
          </View>
        ))
      )}
    </View>
  );

  const renderCreditsTab = () => (
    <View style={styles.tabContent}>
      <Text style={styles.creditsTitle}>Second Opinion Credits</Text>
      <Text style={styles.creditsSubtitle}>
        Purchase credits for discounted second opinion consultations
      </Text>

      {CREDIT_PACKAGES.map((pkg) => (
        <TouchableOpacity
          key={pkg.id}
          style={[
            styles.creditPackage,
            selectedPackage?.id === pkg.id && styles.creditPackageSelected,
          ]}
          onPress={() => setSelectedPackage(pkg)}
        >
          <View style={styles.packageHeader}>
            <View style={styles.packageInfo}>
              <Text style={styles.packageName}>{pkg.name}</Text>
              <Text style={styles.packageCredits}>{pkg.credits} credit{pkg.credits > 1 ? 's' : ''}</Text>
            </View>
            {pkg.savings && (
              <View style={styles.savingsBadge}>
                <Text style={styles.savingsText}>{pkg.savings}</Text>
              </View>
            )}
          </View>
          <View style={styles.packagePrice}>
            <Text style={styles.priceValue}>{formatCurrency(pkg.price)}</Text>
            <Text style={styles.pricePerCredit}>
              ({formatCurrency(pkg.price / pkg.credits)}/credit)
            </Text>
          </View>
          {selectedPackage?.id === pkg.id && (
            <View style={styles.selectedIndicator}>
              <Ionicons name="checkmark-circle" size={24} color="#4CAF50" />
            </View>
          )}
        </TouchableOpacity>
      ))}

      <TouchableOpacity
        style={[styles.purchaseButton, !selectedPackage && styles.purchaseButtonDisabled]}
        onPress={() => selectedPackage && setShowCreditsModal(true)}
        disabled={!selectedPackage}
      >
        <Ionicons name="cart" size={20} color="#fff" />
        <Text style={styles.purchaseButtonText}>
          {selectedPackage
            ? `Purchase ${selectedPackage.credits} Credits for ${formatCurrency(selectedPackage.price)}`
            : 'Select a Package'}
        </Text>
      </TouchableOpacity>

      <View style={styles.creditsInfo}>
        <Ionicons name="information-circle" size={20} color="rgba(255,255,255,0.7)" />
        <Text style={styles.creditsInfoText}>
          Credits never expire and can be used for any second opinion consultation
        </Text>
      </View>
    </View>
  );

  const renderPaymentModal = () => (
    <Modal visible={showPaymentModal} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity onPress={() => setShowPaymentModal(false)}>
            <Ionicons name="close" size={24} color="#333" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Payment</Text>
          <View style={{ width: 24 }} />
        </View>

        <ScrollView style={styles.modalScroll}>
          {paymentConfig?.demo_mode && (
            <View style={styles.demoWarning}>
              <Ionicons name="information-circle" size={20} color="#FF9800" />
              <Text style={styles.demoWarningText}>
                Demo Mode: Use test card 4242 4242 4242 4242
              </Text>
            </View>
          )}

          {/* Order Summary */}
          <View style={styles.orderSummary}>
            <Text style={styles.orderTitle}>Order Summary</Text>
            {selectedConsultation && (
              <View style={styles.orderItem}>
                <Text style={styles.orderItemName}>
                  Consultation with {selectedConsultation.dermatologist_name || 'Dermatologist'}
                </Text>
                <Text style={styles.orderItemPrice}>
                  {formatCurrency(selectedConsultation.consultation_fee || 75)}
                </Text>
              </View>
            )}
            <View style={styles.orderTotal}>
              <Text style={styles.orderTotalLabel}>Total</Text>
              <Text style={styles.orderTotalValue}>
                {formatCurrency(paymentIntent?.amount || selectedConsultation?.consultation_fee || 75)}
              </Text>
            </View>
          </View>

          {/* Card Form */}
          <View style={styles.cardForm}>
            <Text style={styles.cardFormTitle}>Card Details</Text>

            <Text style={styles.inputLabel}>Card Number</Text>
            <View style={styles.cardInputContainer}>
              <Ionicons name="card-outline" size={20} color="#999" style={styles.cardIcon} />
              <TextInput
                style={styles.cardInput}
                value={cardNumber}
                onChangeText={(text) => setCardNumber(formatCardNumber(text))}
                placeholder="1234 5678 9012 3456"
                keyboardType="numeric"
                maxLength={19}
              />
            </View>

            <View style={styles.cardRow}>
              <View style={styles.cardHalf}>
                <Text style={styles.inputLabel}>Expiry Date</Text>
                <TextInput
                  style={styles.textInput}
                  value={expiryDate}
                  onChangeText={(text) => setExpiryDate(formatExpiryDate(text))}
                  placeholder="MM/YY"
                  keyboardType="numeric"
                  maxLength={5}
                />
              </View>
              <View style={styles.cardHalf}>
                <Text style={styles.inputLabel}>CVV</Text>
                <TextInput
                  style={styles.textInput}
                  value={cvv}
                  onChangeText={setCvv}
                  placeholder="123"
                  keyboardType="numeric"
                  maxLength={4}
                  secureTextEntry
                />
              </View>
            </View>

            <Text style={styles.inputLabel}>Cardholder Name</Text>
            <TextInput
              style={styles.textInput}
              value={cardholderName}
              onChangeText={setCardholderName}
              placeholder="John Doe"
              autoCapitalize="words"
            />
          </View>

          {/* Security Info */}
          <View style={styles.securityInfo}>
            <Ionicons name="lock-closed" size={16} color="#4CAF50" />
            <Text style={styles.securityText}>
              Your payment is secured with 256-bit encryption
            </Text>
          </View>

          {/* Pay Button */}
          <TouchableOpacity
            style={[styles.confirmPayButton, processing && styles.confirmPayButtonDisabled]}
            onPress={handlePayment}
            disabled={processing}
          >
            {processing ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <>
                <Ionicons name="lock-closed" size={18} color="#fff" />
                <Text style={styles.confirmPayButtonText}>
                  Pay {formatCurrency(paymentIntent?.amount || selectedConsultation?.consultation_fee || 75)}
                </Text>
              </>
            )}
          </TouchableOpacity>

          {/* Accepted Cards */}
          <View style={styles.acceptedCards}>
            <Text style={styles.acceptedCardsText}>We accept</Text>
            <View style={styles.cardLogos}>
              <View style={styles.cardLogo}>
                <Text style={styles.cardLogoText}>VISA</Text>
              </View>
              <View style={styles.cardLogo}>
                <Text style={styles.cardLogoText}>MC</Text>
              </View>
              <View style={styles.cardLogo}>
                <Text style={styles.cardLogoText}>AMEX</Text>
              </View>
            </View>
          </View>
        </ScrollView>
      </View>
    </Modal>
  );

  const renderCreditsModal = () => (
    <Modal visible={showCreditsModal} animationType="fade" transparent>
      <View style={styles.creditsModalOverlay}>
        <View style={styles.creditsModalContent}>
          <Text style={styles.creditsModalTitle}>Confirm Purchase</Text>

          {selectedPackage && (
            <View style={styles.creditsModalSummary}>
              <Text style={styles.creditsModalPackage}>{selectedPackage.name} Package</Text>
              <Text style={styles.creditsModalCredits}>
                {selectedPackage.credits} credit{selectedPackage.credits > 1 ? 's' : ''}
              </Text>
              <Text style={styles.creditsModalPrice}>
                {formatCurrency(selectedPackage.price)}
              </Text>
            </View>
          )}

          <View style={styles.creditsModalButtons}>
            <TouchableOpacity
              style={styles.cancelButton}
              onPress={() => setShowCreditsModal(false)}
            >
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.confirmButton, processing && styles.confirmButtonDisabled]}
              onPress={handlePurchaseCredits}
              disabled={processing}
            >
              {processing ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <Text style={styles.confirmButtonText}>Confirm</Text>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );

  if (loading) {
    return (
      <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#fff" />
          <Text style={styles.loadingText}>Loading...</Text>
        </View>
      </LinearGradient>
    );
  }

  const pendingCount = getPendingConsultations().length;
  const totalPaid = paymentHistory.reduce((sum, p) => sum + (p.amount || 0), 0);

  return (
    <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Payments</Text>
        <View style={styles.headerRight} />
      </View>

      {/* Stats */}
      <View style={styles.statsContainer}>
        <View style={styles.statBox}>
          <Ionicons name="time-outline" size={24} color="#FF9800" />
          <Text style={styles.statValue}>{pendingCount}</Text>
          <Text style={styles.statLabel}>Pending</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statBox}>
          <Ionicons name="checkmark-circle-outline" size={24} color="#4CAF50" />
          <Text style={styles.statValue}>{paymentHistory.length}</Text>
          <Text style={styles.statLabel}>Completed</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statBox}>
          <Ionicons name="wallet-outline" size={24} color="#2196F3" />
          <Text style={styles.statValue}>{formatCurrency(totalPaid)}</Text>
          <Text style={styles.statLabel}>Total Paid</Text>
        </View>
      </View>

      {/* Tabs */}
      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'pending' && styles.activeTab]}
          onPress={() => setActiveTab('pending')}
        >
          <Ionicons
            name="time"
            size={18}
            color={activeTab === 'pending' ? '#667eea' : '#666'}
          />
          <Text style={[styles.tabText, activeTab === 'pending' && styles.activeTabText]}>
            Pending
          </Text>
          {pendingCount > 0 && (
            <View style={styles.tabBadge}>
              <Text style={styles.tabBadgeText}>{pendingCount}</Text>
            </View>
          )}
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'history' && styles.activeTab]}
          onPress={() => setActiveTab('history')}
        >
          <Ionicons
            name="receipt"
            size={18}
            color={activeTab === 'history' ? '#667eea' : '#666'}
          />
          <Text style={[styles.tabText, activeTab === 'history' && styles.activeTabText]}>
            History
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'credits' && styles.activeTab]}
          onPress={() => setActiveTab('credits')}
        >
          <Ionicons
            name="star"
            size={18}
            color={activeTab === 'credits' ? '#667eea' : '#666'}
          />
          <Text style={[styles.tabText, activeTab === 'credits' && styles.activeTabText]}>
            Credits
          </Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#fff" />
        }
      >
        {activeTab === 'pending' && renderPendingTab()}
        {activeTab === 'history' && renderHistoryTab()}
        {activeTab === 'credits' && renderCreditsTab()}
      </ScrollView>

      {renderPaymentModal()}
      {renderCreditsModal()}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#fff',
    marginTop: 10,
    fontSize: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 20,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  headerRight: {
    width: 40,
  },
  statsContainer: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.15)',
    marginHorizontal: 20,
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
  },
  statBox: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 4,
  },
  statLabel: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 2,
  },
  statDivider: {
    width: 1,
    backgroundColor: 'rgba(255,255,255,0.3)',
    marginHorizontal: 10,
  },
  tabContainer: {
    flexDirection: 'row',
    marginHorizontal: 20,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 4,
    marginBottom: 15,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderRadius: 10,
    gap: 4,
  },
  activeTab: {
    backgroundColor: '#f0f0f0',
  },
  tabText: {
    fontSize: 13,
    color: '#666',
    fontWeight: '500',
  },
  activeTabText: {
    color: '#667eea',
    fontWeight: '600',
  },
  tabBadge: {
    backgroundColor: '#F44336',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
    marginLeft: 2,
  },
  tabBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
  },
  tabContent: {
    paddingHorizontal: 20,
    paddingBottom: 30,
  },
  emptyState: {
    alignItems: 'center',
    paddingTop: 60,
  },
  emptyText: {
    color: 'rgba(255,255,255,0.9)',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
  },
  emptySubtext: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
  // Pending Card
  pendingCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  pendingHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  pendingIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#E8EAF6',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  pendingInfo: {
    flex: 1,
  },
  pendingTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  pendingType: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  pendingDate: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
  },
  pendingAmount: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  amountLabel: {
    fontSize: 14,
    color: '#666',
  },
  amountValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  payButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#667eea',
    padding: 14,
    borderRadius: 8,
    gap: 8,
  },
  payButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  // History Card
  historyCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
  },
  historyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  historyIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#E8F5E9',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  historyInfo: {
    flex: 1,
  },
  historyTitle: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
  },
  historyDate: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
  },
  historyAmount: {
    alignItems: 'flex-end',
  },
  historyAmountValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  historyStatus: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
    marginTop: 4,
  },
  historyStatusText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
    textTransform: 'uppercase',
  },
  // Credits Tab
  creditsTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 4,
  },
  creditsSubtitle: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.7)',
    marginBottom: 20,
  },
  creditPackage: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  creditPackageSelected: {
    borderColor: '#4CAF50',
  },
  packageHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  packageInfo: {},
  packageName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  packageCredits: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  savingsBadge: {
    backgroundColor: '#E8F5E9',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  savingsText: {
    fontSize: 12,
    color: '#4CAF50',
    fontWeight: '600',
  },
  packagePrice: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 8,
  },
  priceValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  pricePerCredit: {
    fontSize: 13,
    color: '#999',
  },
  selectedIndicator: {
    position: 'absolute',
    top: 12,
    right: 12,
  },
  purchaseButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#4CAF50',
    padding: 16,
    borderRadius: 12,
    marginTop: 8,
    gap: 8,
  },
  purchaseButtonDisabled: {
    backgroundColor: 'rgba(255,255,255,0.3)',
  },
  purchaseButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  creditsInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 16,
    gap: 8,
  },
  creditsInfoText: {
    flex: 1,
    fontSize: 13,
    color: 'rgba(255,255,255,0.7)',
  },
  // Payment Modal
  modalContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 15,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  modalScroll: {
    flex: 1,
    padding: 20,
  },
  demoWarning: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF3E0',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
    gap: 8,
  },
  demoWarningText: {
    flex: 1,
    fontSize: 13,
    color: '#E65100',
  },
  orderSummary: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  orderTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  orderItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  orderItemName: {
    fontSize: 14,
    color: '#333',
    flex: 1,
  },
  orderItemPrice: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  orderTotal: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingTop: 12,
    marginTop: 4,
  },
  orderTotalLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  orderTotalValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#667eea',
  },
  cardForm: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  cardFormTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  inputLabel: {
    fontSize: 13,
    color: '#666',
    marginBottom: 6,
    marginTop: 8,
  },
  cardInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  cardIcon: {
    paddingLeft: 12,
  },
  cardInput: {
    flex: 1,
    padding: 12,
    fontSize: 16,
    color: '#333',
  },
  cardRow: {
    flexDirection: 'row',
    gap: 12,
  },
  cardHalf: {
    flex: 1,
  },
  textInput: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    color: '#333',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  securityInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    marginBottom: 16,
  },
  securityText: {
    fontSize: 12,
    color: '#4CAF50',
  },
  confirmPayButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#667eea',
    padding: 16,
    borderRadius: 12,
    gap: 8,
  },
  confirmPayButtonDisabled: {
    opacity: 0.7,
  },
  confirmPayButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  acceptedCards: {
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 30,
  },
  acceptedCardsText: {
    fontSize: 12,
    color: '#999',
    marginBottom: 8,
  },
  cardLogos: {
    flexDirection: 'row',
    gap: 12,
  },
  cardLogo: {
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 4,
  },
  cardLogoText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#666',
  },
  // Credits Modal
  creditsModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    padding: 20,
  },
  creditsModalContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
  },
  creditsModalTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    textAlign: 'center',
    marginBottom: 20,
  },
  creditsModalSummary: {
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginBottom: 20,
  },
  creditsModalPackage: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  creditsModalCredits: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  creditsModalPrice: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#667eea',
    marginTop: 8,
  },
  creditsModalButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  cancelButton: {
    flex: 1,
    padding: 14,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
    alignItems: 'center',
  },
  cancelButtonText: {
    fontSize: 16,
    color: '#666',
    fontWeight: '500',
  },
  confirmButton: {
    flex: 1,
    padding: 14,
    borderRadius: 8,
    backgroundColor: '#4CAF50',
    alignItems: 'center',
  },
  confirmButtonDisabled: {
    opacity: 0.7,
  },
  confirmButtonText: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '600',
  },
});
