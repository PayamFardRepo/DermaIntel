/**
 * Billing & Insurance Pre-Authorization Screen
 *
 * Features:
 * - View billing records from analyses
 * - Search CPT and ICD-10 codes
 * - View and manage insurance pre-authorizations
 * - Track pre-auth status (Draft, Submitted, Approved, Denied)
 * - Generate and download CMS-1500 forms
 * - Download pre-authorization PDFs
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Modal,
  Alert,
  ActivityIndicator,
  Platform,
  TextInput,
  RefreshControl,
  Linking,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import * as SecureStore from 'expo-secure-store';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import { API_BASE_URL } from '../config';

interface BillingCode {
  code_type: string;
  code: string;
  description: string;
  category: string;
  typical_reimbursement?: number;
}

interface BillingRecord {
  id: number;
  analysis_id: number;
  diagnosis: string;
  procedure_date: string;
  cpt_codes: BillingCode[];
  icd10_codes: BillingCode[];
  total_charges: number;
  estimated_reimbursement: number;
  status: string;
  created_at: string;
  preauth_status?: string;
  insurance_preauthorization?: any;
}

interface PreAuthData {
  form_data?: {
    diagnosis?: {
      primary_diagnosis: string;
      icd10_code: string;
      confidence_level: string;
      diagnostic_method: string;
    };
    procedures_requested?: Array<{
      code: string;
      description: string;
      rationale: string;
    }>;
    urgency?: string;
    clinical_rationale?: string;
    estimated_timeline?: string;
  };
  medical_necessity_letter?: string;
  clinical_summary?: string;
  submission_status?: {
    current_status: string;
    status_description: string;
    submitted_date?: string;
    decision_date?: string;
    notes?: string;
  };
}

interface Appeal {
  id: number;
  appeal_id: string;
  claim_number: string;
  insurance_company: string;
  diagnosis: string;
  denial_reason: string;
  denial_reason_text?: string;
  appeal_level: string;
  appeal_status: string;
  letter_content?: string;
  success_likelihood?: number;
  deadline?: string;
  created_at: string;
  submitted_date?: string;
  outcome?: string;
}

interface DenialReason {
  value: string;
  label: string;
  description: string;
}

type TabType = 'billing' | 'preauth' | 'appeals' | 'codes' | 'hsa';

interface HSAExpense {
  id: string;
  date: string;
  description: string;
  category: string;
  category_name: string;
  provider: string;
  amount: number;
  eligible: boolean;
  procedure_code: string;
  diagnosis: string;
  notes: string;
}

interface HSASummary {
  total_eligible: number;
  total_ineligible: number;
  total_expenses: number;
  expense_count: number;
  eligible_count: number;
}

const PREAUTH_STATUSES = [
  { key: 'DRAFT', label: 'Draft', color: '#6b7280', icon: 'document-outline' },
  { key: 'SUBMITTED', label: 'Submitted', color: '#3b82f6', icon: 'paper-plane-outline' },
  { key: 'UNDER_REVIEW', label: 'Under Review', color: '#f59e0b', icon: 'time-outline' },
  { key: 'APPROVED', label: 'Approved', color: '#10b981', icon: 'checkmark-circle-outline' },
  { key: 'DENIED', label: 'Denied', color: '#ef4444', icon: 'close-circle-outline' },
  { key: 'ADDITIONAL_INFO_REQUIRED', label: 'Info Required', color: '#8b5cf6', icon: 'alert-circle-outline' },
];

const APPEAL_STATUSES = [
  { key: 'draft', label: 'Draft', color: '#6b7280', icon: 'document-outline' },
  { key: 'submitted', label: 'Submitted', color: '#3b82f6', icon: 'paper-plane-outline' },
  { key: 'under_review', label: 'Under Review', color: '#f59e0b', icon: 'time-outline' },
  { key: 'additional_info_requested', label: 'Info Requested', color: '#8b5cf6', icon: 'alert-circle-outline' },
  { key: 'approved', label: 'Approved', color: '#10b981', icon: 'checkmark-circle-outline' },
  { key: 'denied', label: 'Denied', color: '#ef4444', icon: 'close-circle-outline' },
  { key: 'escalated', label: 'Escalated', color: '#6366f1', icon: 'arrow-up-circle-outline' },
];

const APPEAL_LEVELS = [
  { key: 'first_level', label: 'First Level Appeal', description: 'Initial appeal to insurance company' },
  { key: 'second_level', label: 'Second Level Appeal', description: 'Escalation after first denial' },
  { key: 'external_review', label: 'External Review', description: 'Independent third-party review' },
  { key: 'state_insurance', label: 'State Insurance Commissioner', description: 'File complaint with state regulator' },
];

export default function BillingInsuranceScreen() {
  const { t } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // State
  const [activeTab, setActiveTab] = useState<TabType>('billing');
  const [billingRecords, setBillingRecords] = useState<BillingRecord[]>([]);
  const [searchResults, setSearchResults] = useState<BillingCode[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);

  // Detail modal state
  const [selectedRecord, setSelectedRecord] = useState<BillingRecord | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  // Pre-auth modal state
  const [showPreAuthModal, setShowPreAuthModal] = useState(false);
  const [selectedPreAuth, setSelectedPreAuth] = useState<any>(null);
  const [isUpdatingStatus, setIsUpdatingStatus] = useState(false);

  // Appeals state
  const [appeals, setAppeals] = useState<Appeal[]>([]);
  const [denialReasons, setDenialReasons] = useState<DenialReason[]>([]);
  const [showAppealModal, setShowAppealModal] = useState(false);
  const [showNewAppealModal, setShowNewAppealModal] = useState(false);
  const [selectedAppeal, setSelectedAppeal] = useState<Appeal | null>(null);
  const [isGeneratingAppeal, setIsGeneratingAppeal] = useState(false);
  const [newAppealData, setNewAppealData] = useState({
    record: null as BillingRecord | null,
    denialReason: 'medical_necessity',
    insuranceCompany: '',
    claimNumber: '',
    denialReasonText: '',
  });

  // HSA/FSA state
  const [hsaExpenses, setHsaExpenses] = useState<HSAExpense[]>([]);
  const [hsaSummary, setHsaSummary] = useState<HSASummary | null>(null);
  const [hsaYear, setHsaYear] = useState(new Date().getFullYear());
  const [isLoadingHsa, setIsLoadingHsa] = useState(false);
  const [showReceiptModal, setShowReceiptModal] = useState(false);
  const [selectedExpense, setSelectedExpense] = useState<HSAExpense | null>(null);
  const [receiptHtml, setReceiptHtml] = useState<string>('');

  // Auth headers
  const getAuthHeaders = async () => {
    const token = await SecureStore.getItemAsync('auth_token');
    console.log('[Auth] Token exists:', !!token, token ? `(${token.substring(0, 20)}...)` : '(null)');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  };

  // Initialize
  useEffect(() => {
    console.log('[Init] isAuthenticated:', isAuthenticated);
    if (!isAuthenticated) {
      console.log('[Init] Not authenticated, redirecting...');
      router.replace('/');
      return;
    }
    console.log('[Init] Loading data...');
    loadData();
  }, [isAuthenticated]);

  // Load HSA expenses when tab becomes active
  useEffect(() => {
    if (activeTab === 'hsa' && hsaExpenses.length === 0) {
      loadHsaExpenses();
    }
  }, [activeTab]);

  const loadData = async () => {
    console.log('[LoadData] Starting...');
    setIsLoading(true);
    await Promise.all([
      loadBillingRecords(),
      loadAppeals(),
      loadDenialReasons(),
    ]);
    setIsLoading(false);
  };

  const loadBillingRecords = async () => {
    try {
      const headers = await getAuthHeaders();
      console.log('[Billing] Fetching from:', `${API_BASE_URL}/billing/records`);
      console.log('[Billing] Headers:', JSON.stringify(headers));
      const response = await fetch(`${API_BASE_URL}/billing/records`, { headers });

      console.log('[Billing] Response status:', response.status);
      if (response.ok) {
        const data = await response.json();
        console.log('[Billing] Records received:', data.records?.length || 0);
        setBillingRecords(data.records || []);
      } else {
        const errorText = await response.text();
        console.error('[Billing] Error response:', errorText);
      }
    } catch (error) {
      console.error('Error loading billing records:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const loadDenialReasons = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/appeals/denial-reasons`, { headers });

      if (response.ok) {
        const data = await response.json();
        setDenialReasons(data.reasons || []);
      }
    } catch (error) {
      console.error('Error loading denial reasons:', error);
    }
  };

  const loadAppeals = async () => {
    try {
      const headers = await getAuthHeaders();
      console.log('[Appeals] Fetching from:', `${API_BASE_URL}/appeals`);
      const response = await fetch(`${API_BASE_URL}/appeals`, { headers });

      console.log('[Appeals] Response status:', response.status);
      if (response.ok) {
        const data = await response.json();
        console.log('[Appeals] Appeals received:', data.appeals?.length || 0);
        setAppeals(data.appeals || []);
      } else {
        const errorText = await response.text();
        console.error('[Appeals] Error response:', errorText);
      }
    } catch (error) {
      console.error('Error loading appeals:', error);
    }
  };

  // HSA/FSA Functions
  const loadHsaExpenses = async (year?: number) => {
    try {
      setIsLoadingHsa(true);
      const headers = await getAuthHeaders();
      const targetYear = year || hsaYear;
      console.log('[HSA] Fetching expenses for year:', targetYear);

      const response = await fetch(`${API_BASE_URL}/costs/hsa-fsa/expenses?year=${targetYear}`, { headers });

      if (response.ok) {
        const data = await response.json();
        console.log('[HSA] Expenses received:', data.expenses?.length || 0);
        setHsaExpenses(data.expenses || []);
        setHsaSummary(data.summary || null);
      } else {
        const errorText = await response.text();
        console.error('[HSA] Error response:', errorText);
      }
    } catch (error) {
      console.error('Error loading HSA expenses:', error);
    } finally {
      setIsLoadingHsa(false);
    }
  };

  const generateHsaReceipt = async (expenseId: string) => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/costs/hsa-fsa/receipt/${expenseId}`, { headers });

      if (response.ok) {
        const data = await response.json();
        setReceiptHtml(data.receipt_html || '');
        setShowReceiptModal(true);
      } else {
        Alert.alert('Error', 'Failed to generate receipt');
      }
    } catch (error) {
      console.error('Error generating receipt:', error);
      Alert.alert('Error', 'Failed to generate receipt');
    }
  };

  const downloadHsaReceipt = async () => {
    if (!receiptHtml) return;

    try {
      const { uri } = await Print.printToFileAsync({ html: receiptHtml });
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(uri, {
          mimeType: 'application/pdf',
          dialogTitle: 'HSA/FSA Receipt',
          UTI: 'com.adobe.pdf'
        });
      } else {
        Alert.alert('Success', `Receipt saved to: ${uri}`);
      }
    } catch (error) {
      console.error('Error downloading receipt:', error);
      Alert.alert('Error', 'Failed to download receipt');
    }
  };

  const downloadYearSummary = async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/costs/hsa-fsa/year-summary/${hsaYear}`, { headers });

      if (response.ok) {
        const data = await response.json();

        // Generate summary HTML
        const summaryHtml = `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    .header { text-align: center; border-bottom: 2px solid #2563eb; padding-bottom: 20px; margin-bottom: 20px; }
    .header h1 { color: #2563eb; margin: 0; }
    .section { margin-bottom: 20px; }
    .section-title { font-weight: bold; font-size: 18px; color: #1f2937; margin-bottom: 10px; }
    .summary-box { background: #f0fdf4; border: 1px solid #10b981; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .summary-row { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .summary-label { color: #6b7280; }
    .summary-value { font-weight: bold; font-size: 20px; color: #059669; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { border: 1px solid #e5e7eb; padding: 10px; text-align: left; }
    th { background: #f3f4f6; }
    .footer { text-align: center; color: #9ca3af; font-size: 12px; margin-top: 40px; }
  </style>
</head>
<body>
  <div class="header">
    <h1>HSA/FSA Annual Summary</h1>
    <p>Tax Year: ${data.year}</p>
    <p>Generated: ${new Date().toLocaleDateString()}</p>
  </div>

  <div class="summary-box">
    <div class="summary-row">
      <span class="summary-label">Total Eligible Expenses:</span>
      <span class="summary-value">$${data.summary.total_eligible_expenses.toFixed(2)}</span>
    </div>
    <div class="summary-row">
      <span class="summary-label">Total Ineligible:</span>
      <span style="font-weight: bold; color: #ef4444;">$${data.summary.total_ineligible_expenses.toFixed(2)}</span>
    </div>
    <div class="summary-row">
      <span class="summary-label">Total Expenses:</span>
      <span style="font-weight: bold;">$${data.summary.total_all_expenses.toFixed(2)}</span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Monthly Breakdown</div>
    <table>
      <tr><th>Month</th><th>Eligible</th><th>Ineligible</th><th>Total</th><th>Count</th></tr>
      ${data.monthly_breakdown.map((m: any) => `
        <tr>
          <td>${m.month_name}</td>
          <td>$${m.eligible_amount.toFixed(2)}</td>
          <td>$${m.ineligible_amount.toFixed(2)}</td>
          <td>$${m.total_amount.toFixed(2)}</td>
          <td>${m.expense_count}</td>
        </tr>
      `).join('')}
    </table>
  </div>

  <div class="section">
    <div class="section-title">Tax Notes</div>
    <ul>
      ${data.tax_notes.map((note: string) => `<li>${note}</li>`).join('')}
    </ul>
  </div>

  <div class="footer">
    <p>This summary is for informational purposes. Consult a tax professional for advice.</p>
  </div>
</body>
</html>
        `;

        const { uri } = await Print.printToFileAsync({ html: summaryHtml });
        if (await Sharing.isAvailableAsync()) {
          await Sharing.shareAsync(uri, {
            mimeType: 'application/pdf',
            dialogTitle: 'HSA/FSA Year Summary',
            UTI: 'com.adobe.pdf'
          });
        }
      } else {
        Alert.alert('Error', 'Failed to generate year summary');
      }
    } catch (error) {
      console.error('Error downloading year summary:', error);
      Alert.alert('Error', 'Failed to download year summary');
    }
  };

  const generateAppealLetter = async (analysisId: number, denialReason: string, insuranceCompany: string, claimNumber: string, denialReasonText?: string) => {
    setIsGeneratingAppeal(true);
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/appeals/generate-from-analysis/${analysisId}`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          denial_reason: denialReason,
          insurance_company: insuranceCompany,
          claim_number: claimNumber,
          denial_reason_text: denialReasonText || undefined,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        Alert.alert('Success', 'Appeal letter generated successfully');
        setShowNewAppealModal(false);
        loadAppeals();
        // Show the generated appeal
        setSelectedAppeal(data);
        setShowAppealModal(true);
        return data;
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to generate appeal letter');
      }
    } catch (error) {
      console.error('Error generating appeal letter:', error);
      Alert.alert('Error', 'Failed to generate appeal letter');
    } finally {
      setIsGeneratingAppeal(false);
    }
  };

  const searchCodes = async (query: string) => {
    if (!query || query.length < 2) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(
        `${API_BASE_URL}/billing/codes/search?query=${encodeURIComponent(query)}`,
        { headers }
      );

      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.codes || []);
      }
    } catch (error) {
      console.error('Error searching codes:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const generateBilling = async (analysisId: number) => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/billing/generate/${analysisId}`, {
        method: 'POST',
        headers,
      });

      if (response.ok) {
        Alert.alert('Success', 'Billing record generated successfully');
        loadBillingRecords();
      } else {
        Alert.alert('Error', 'Failed to generate billing record');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to generate billing record');
    }
  };

  const exportCMS1500 = async (recordId: number) => {
    try {
      // Find the billing record
      const record = billingRecords.find(r => r.id === recordId);
      if (!record) {
        Alert.alert('Error', 'Billing record not found');
        return;
      }

      const cptCodes = record.cpt_codes || [];
      const icdCodes = record.icd10_codes || [];
      const serviceDate = record.procedure_date ? new Date(record.procedure_date).toLocaleDateString() : new Date().toLocaleDateString();

      // Generate CMS-1500 style HTML
      const html = `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>CMS-1500 Health Insurance Claim Form</title>
          <style>
            body { font-family: 'Courier New', monospace; padding: 20px; font-size: 11px; }
            .form-container { border: 2px solid #000; padding: 10px; max-width: 800px; margin: 0 auto; }
            .form-header { text-align: center; border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 15px; }
            .form-header h1 { margin: 0; font-size: 16px; }
            .form-header p { margin: 5px 0 0 0; font-size: 10px; }
            .section { border: 1px solid #000; margin-bottom: 10px; }
            .section-header { background: #f0f0f0; padding: 5px; font-weight: bold; border-bottom: 1px solid #000; font-size: 10px; }
            .section-content { padding: 8px; }
            .row { display: flex; border-bottom: 1px solid #ddd; }
            .row:last-child { border-bottom: none; }
            .cell { flex: 1; padding: 5px; border-right: 1px solid #ddd; }
            .cell:last-child { border-right: none; }
            .cell-label { font-size: 8px; color: #666; text-transform: uppercase; }
            .cell-value { font-size: 11px; margin-top: 2px; font-weight: bold; }
            .field-box { border: 1px solid #000; padding: 3px 5px; margin: 2px; display: inline-block; min-width: 80px; }
            .checkbox { display: inline-block; width: 12px; height: 12px; border: 1px solid #000; margin-right: 5px; text-align: center; line-height: 12px; }
            .checked { background: #000; color: #fff; }
            .service-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            .service-table th, .service-table td { border: 1px solid #000; padding: 5px; text-align: left; font-size: 10px; }
            .service-table th { background: #f0f0f0; font-size: 9px; }
            .totals { margin-top: 15px; text-align: right; }
            .totals-row { margin: 5px 0; }
            .totals-label { display: inline-block; width: 150px; }
            .totals-value { display: inline-block; width: 100px; text-align: right; font-weight: bold; }
            .signature-section { margin-top: 20px; border-top: 2px solid #000; padding-top: 15px; }
            .signature-line { border-bottom: 1px solid #000; height: 30px; margin: 10px 0; }
            .footer { margin-top: 20px; font-size: 9px; color: #666; text-align: center; border-top: 1px solid #ddd; padding-top: 10px; }
          </style>
        </head>
        <body>
          <div class="form-container">
            <div class="form-header">
              <h1>HEALTH INSURANCE CLAIM FORM</h1>
              <p>CMS-1500 (02/12) - APPROVED OMB-0938-1197</p>
            </div>

            <!-- Patient/Insured Info -->
            <div class="section">
              <div class="section-header">1. PATIENT AND INSURED INFORMATION</div>
              <div class="section-content">
                <div class="row">
                  <div class="cell">
                    <div class="cell-label">1a. Insured's ID Number</div>
                    <div class="cell-value">[INSURED ID]</div>
                  </div>
                  <div class="cell">
                    <div class="cell-label">2. Patient's Name</div>
                    <div class="cell-value">${user?.full_name || '[PATIENT NAME]'}</div>
                  </div>
                  <div class="cell">
                    <div class="cell-label">3. Patient's Birth Date</div>
                    <div class="cell-value">[DOB]</div>
                  </div>
                </div>
                <div class="row">
                  <div class="cell">
                    <div class="cell-label">4. Insured's Name</div>
                    <div class="cell-value">[INSURED NAME]</div>
                  </div>
                  <div class="cell">
                    <div class="cell-label">5. Patient's Address</div>
                    <div class="cell-value">[ADDRESS]</div>
                  </div>
                  <div class="cell">
                    <div class="cell-label">6. Patient Relationship to Insured</div>
                    <div class="cell-value"><span class="checkbox checked">X</span> Self</div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Diagnosis Codes -->
            <div class="section">
              <div class="section-header">21. DIAGNOSIS OR NATURE OF ILLNESS OR INJURY (ICD-10-CM)</div>
              <div class="section-content">
                <div class="row">
                  ${icdCodes.map((icd: any, i: number) => `
                    <div class="cell">
                      <div class="cell-label">${String.fromCharCode(65 + i)}.</div>
                      <div class="cell-value">${icd.code}</div>
                      <div style="font-size: 9px; color: #666;">${icd.description}</div>
                    </div>
                  `).join('')}
                  ${icdCodes.length < 4 ? '<div class="cell"></div>'.repeat(4 - icdCodes.length) : ''}
                </div>
              </div>
            </div>

            <!-- Service Lines -->
            <div class="section">
              <div class="section-header">24. SERVICES</div>
              <div class="section-content">
                <table class="service-table">
                  <thead>
                    <tr>
                      <th>A. Date of Service</th>
                      <th>B. Place of Service</th>
                      <th>D. Procedures/Services (CPT/HCPCS)</th>
                      <th>E. Diagnosis Pointer</th>
                      <th>F. Charges</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${cptCodes.map((cpt: any, i: number) => `
                      <tr>
                        <td>${serviceDate}</td>
                        <td>11 (Office)</td>
                        <td><strong>${cpt.code}</strong> - ${cpt.description}</td>
                        <td>A</td>
                        <td>$${(record.total_charges / cptCodes.length).toFixed(2)}</td>
                      </tr>
                    `).join('')}
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Totals -->
            <div class="section">
              <div class="section-header">28. TOTAL CHARGES & PAYMENT</div>
              <div class="section-content">
                <div class="totals">
                  <div class="totals-row">
                    <span class="totals-label">28. Total Charges:</span>
                    <span class="totals-value">$${record.total_charges?.toFixed(2) || '0.00'}</span>
                  </div>
                  <div class="totals-row">
                    <span class="totals-label">29. Amount Paid:</span>
                    <span class="totals-value">$0.00</span>
                  </div>
                  <div class="totals-row">
                    <span class="totals-label">30. Balance Due:</span>
                    <span class="totals-value">$${record.total_charges?.toFixed(2) || '0.00'}</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- Provider Info -->
            <div class="section">
              <div class="section-header">31-33. PROVIDER INFORMATION</div>
              <div class="section-content">
                <div class="row">
                  <div class="cell">
                    <div class="cell-label">31. Signature of Physician</div>
                    <div class="signature-line"></div>
                    <div style="font-size: 9px;">Date: ${new Date().toLocaleDateString()}</div>
                  </div>
                  <div class="cell">
                    <div class="cell-label">32. Service Facility</div>
                    <div class="cell-value">[FACILITY NAME]</div>
                    <div style="font-size: 9px;">[FACILITY ADDRESS]</div>
                  </div>
                  <div class="cell">
                    <div class="cell-label">33. Billing Provider Info</div>
                    <div class="cell-value">[PROVIDER NAME]</div>
                    <div style="font-size: 9px;">NPI: [NPI NUMBER]</div>
                  </div>
                </div>
              </div>
            </div>

            <div class="footer">
              <p>Generated by SkinLesionDetection AI-Assisted Dermatology System</p>
              <p>This is a simplified CMS-1500 form. Complete all required fields before submission.</p>
              <p>Form generated on: ${new Date().toLocaleString()}</p>
            </div>
          </div>
        </body>
        </html>
      `;

      // Generate PDF
      const { uri } = await Print.printToFileAsync({ html });
      console.log('CMS-1500 PDF generated at:', uri);

      // Share/Save the PDF
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(uri, {
          mimeType: 'application/pdf',
          dialogTitle: 'CMS-1500 Claim Form',
          UTI: 'com.adobe.pdf'
        });
      } else {
        Alert.alert('Success', 'CMS-1500 form generated successfully');
      }
    } catch (error) {
      console.error('Error generating CMS-1500:', error);
      Alert.alert('Error', 'Failed to export CMS-1500');
    }
  };

  const downloadPreAuthPDF = async (analysisId: number) => {
    try {
      // Find the record with preauth data
      const record = billingRecords.find(r => r.analysis_id === analysisId);
      if (!record || !record.insurance_preauthorization) {
        Alert.alert('Error', 'No pre-authorization data found');
        return;
      }

      const preauth = record.insurance_preauthorization;
      const formData = preauth.form_data || {};
      const diagnosis = formData.diagnosis || {};
      const procedures = formData.procedures_requested || [];

      // Generate HTML for PDF
      const html = `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Pre-Authorization Request</title>
          <style>
            body { font-family: Arial, sans-serif; padding: 40px; color: #333; }
            .header { text-align: center; border-bottom: 2px solid #2563eb; padding-bottom: 20px; margin-bottom: 30px; }
            .header h1 { color: #2563eb; margin: 0; font-size: 24px; }
            .header p { color: #666; margin: 5px 0 0 0; }
            .section { margin-bottom: 25px; }
            .section-title { font-size: 16px; font-weight: bold; color: #2563eb; border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-bottom: 15px; }
            .field { margin-bottom: 12px; }
            .field-label { font-weight: bold; color: #555; font-size: 12px; text-transform: uppercase; }
            .field-value { margin-top: 4px; font-size: 14px; }
            .code-box { background: #f3f4f6; padding: 10px; border-radius: 5px; margin: 5px 0; }
            .code { font-family: monospace; font-weight: bold; color: #2563eb; }
            .status { display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 12px; }
            .status-approved { background: #d1fae5; color: #065f46; }
            .status-denied { background: #fee2e2; color: #991b1b; }
            .status-pending { background: #fef3c7; color: #92400e; }
            .letter-box { background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 8px; margin-top: 15px; }
            .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 11px; color: #666; text-align: center; }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>Insurance Pre-Authorization Request</h1>
            <p>Generated on ${new Date().toLocaleDateString()}</p>
          </div>

          <div class="section">
            <div class="section-title">Patient & Diagnosis Information</div>
            <div class="field">
              <div class="field-label">Primary Diagnosis</div>
              <div class="field-value">${diagnosis.primary_diagnosis || record.diagnosis || 'N/A'}</div>
            </div>
            <div class="field">
              <div class="field-label">ICD-10 Code</div>
              <div class="field-value"><span class="code">${diagnosis.icd10_code || record.icd10_codes?.[0]?.code || 'N/A'}</span></div>
            </div>
            <div class="field">
              <div class="field-label">Confidence Level</div>
              <div class="field-value">${diagnosis.confidence_level || 'N/A'}</div>
            </div>
            <div class="field">
              <div class="field-label">Diagnostic Method</div>
              <div class="field-value">${diagnosis.diagnostic_method || 'AI-assisted dermoscopy'}</div>
            </div>
          </div>

          <div class="section">
            <div class="section-title">Procedures Requested</div>
            ${procedures.map((proc: any) => `
              <div class="code-box">
                <div><span class="code">${proc.code}</span> - ${proc.description}</div>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">Rationale: ${proc.rationale}</div>
              </div>
            `).join('')}
          </div>

          <div class="section">
            <div class="section-title">Clinical Information</div>
            <div class="field">
              <div class="field-label">Urgency</div>
              <div class="field-value">${formData.urgency || 'Routine'}</div>
            </div>
            <div class="field">
              <div class="field-label">Clinical Rationale</div>
              <div class="field-value">${formData.clinical_rationale || 'N/A'}</div>
            </div>
            <div class="field">
              <div class="field-label">Estimated Timeline</div>
              <div class="field-value">${formData.estimated_timeline || 'N/A'}</div>
            </div>
          </div>

          <div class="section">
            <div class="section-title">Authorization Status</div>
            <div class="field">
              <span class="status ${
                record.preauth_status === 'APPROVED' ? 'status-approved' :
                record.preauth_status === 'DENIED' ? 'status-denied' : 'status-pending'
              }">${record.preauth_status || 'PENDING'}</span>
            </div>
          </div>

          ${preauth.medical_necessity_letter ? `
          <div class="section">
            <div class="section-title">Medical Necessity Statement</div>
            <div class="letter-box">
              ${preauth.medical_necessity_letter}
            </div>
          </div>
          ` : ''}

          <div class="footer">
            <p>This document was generated by SkinLesionDetection AI-Assisted Dermatology System</p>
            <p>For questions, please contact your healthcare provider</p>
          </div>
        </body>
        </html>
      `;

      // Generate PDF
      const { uri } = await Print.printToFileAsync({ html });
      console.log('PDF generated at:', uri);

      // Share/Save the PDF
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(uri, {
          mimeType: 'application/pdf',
          dialogTitle: 'Pre-Authorization Documentation',
          UTI: 'com.adobe.pdf'
        });
      } else {
        Alert.alert('Success', 'PDF generated successfully');
      }
    } catch (error) {
      console.error('Error generating PDF:', error);
      Alert.alert('Error', 'Failed to generate pre-authorization PDF');
    }
  };

  const updatePreAuthStatus = async (analysisId: number, newStatus: string, notes?: string) => {
    setIsUpdatingStatus(true);
    try {
      const headers = await getAuthHeaders();
      const url = `${API_BASE_URL}/analysis/preauth-status/${analysisId}?status=${newStatus}${notes ? `&notes=${encodeURIComponent(notes)}` : ''}`;

      const response = await fetch(url, {
        method: 'PATCH',
        headers,
      });

      if (response.ok) {
        Alert.alert('Success', `Status updated to ${newStatus}`);
        loadBillingRecords();
        setShowPreAuthModal(false);
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to update status');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to update status');
    } finally {
      setIsUpdatingStatus(false);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadBillingRecords();
  }, []);

  // Helper functions
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const formatCurrency = (amount: number) => {
    return `$${amount.toFixed(2)}`;
  };

  const getStatusInfo = (status: string) => {
    return PREAUTH_STATUSES.find(s => s.key === status) || PREAUTH_STATUSES[0];
  };

  const getAppealStatusInfo = (status: string) => {
    return APPEAL_STATUSES.find(s => s.key === status) || APPEAL_STATUSES[0];
  };

  const getAppealLevelInfo = (level: string) => {
    return APPEAL_LEVELS.find(l => l.key === level) || APPEAL_LEVELS[0];
  };

  // Filter records with pre-auth data
  const recordsWithPreAuth = billingRecords.filter(r => r.insurance_preauthorization);

  // Filter records with denied status (for appeal generation)
  const deniedRecords = billingRecords.filter(r => r.preauth_status === 'DENIED');

  // Render billing record card
  const renderBillingCard = (record: BillingRecord) => {
    const hasPreAuth = !!record.insurance_preauthorization;
    const preAuthStatus = record.preauth_status || 'DRAFT';
    const statusInfo = getStatusInfo(preAuthStatus);

    return (
      <TouchableOpacity
        key={record.id}
        style={styles.billingCard}
        onPress={() => {
          setSelectedRecord(record);
          setShowDetailModal(true);
        }}
      >
        <View style={styles.cardHeader}>
          <View style={styles.cardTitleRow}>
            <Ionicons name="document-text" size={20} color="#2563eb" />
            <Text style={styles.cardDiagnosis}>{record.diagnosis}</Text>
          </View>
          <Text style={styles.cardDate}>{formatDate(record.procedure_date)}</Text>
        </View>

        <View style={styles.codesContainer}>
          <View style={styles.codeSection}>
            <Text style={styles.codeSectionTitle}>CPT Codes</Text>
            {record.cpt_codes.map((code, idx) => (
              <View key={idx} style={styles.codeTag}>
                <Text style={styles.codeText}>{code.code}</Text>
              </View>
            ))}
          </View>
          <View style={styles.codeSection}>
            <Text style={styles.codeSectionTitle}>ICD-10</Text>
            {record.icd10_codes.map((code, idx) => (
              <View key={idx} style={[styles.codeTag, styles.icdTag]}>
                <Text style={[styles.codeText, styles.icdText]}>{code.code}</Text>
              </View>
            ))}
          </View>
        </View>

        <View style={styles.chargesRow}>
          <View style={styles.chargeItem}>
            <Text style={styles.chargeLabel}>Total Charges</Text>
            <Text style={styles.chargeValue}>{formatCurrency(record.total_charges)}</Text>
          </View>
          <View style={styles.chargeItem}>
            <Text style={styles.chargeLabel}>Est. Reimbursement</Text>
            <Text style={[styles.chargeValue, styles.reimbursementValue]}>
              {formatCurrency(record.estimated_reimbursement)}
            </Text>
          </View>
        </View>

        {hasPreAuth && (
          <View style={styles.preAuthStatusRow}>
            <Ionicons name={statusInfo.icon as any} size={16} color={statusInfo.color} />
            <Text style={[styles.preAuthStatusText, { color: statusInfo.color }]}>
              Pre-Auth: {statusInfo.label}
            </Text>
          </View>
        )}

        <View style={styles.cardActions}>
          <TouchableOpacity
            style={styles.cardActionButton}
            onPress={() => exportCMS1500(record.id)}
          >
            <Ionicons name="download-outline" size={16} color="#2563eb" />
            <Text style={styles.cardActionText}>CMS-1500</Text>
          </TouchableOpacity>
          {hasPreAuth && (
            <TouchableOpacity
              style={styles.cardActionButton}
              onPress={() => {
                setSelectedPreAuth({ ...record, analysisId: record.analysis_id });
                setShowPreAuthModal(true);
              }}
            >
              <Ionicons name="shield-checkmark-outline" size={16} color="#10b981" />
              <Text style={[styles.cardActionText, { color: '#10b981' }]}>Pre-Auth</Text>
            </TouchableOpacity>
          )}
        </View>
      </TouchableOpacity>
    );
  };

  // Render pre-auth card
  const renderPreAuthCard = (record: BillingRecord) => {
    const preAuth = record.insurance_preauthorization as PreAuthData;
    const status = record.preauth_status || preAuth?.submission_status?.current_status || 'DRAFT';
    const statusInfo = getStatusInfo(status);

    return (
      <TouchableOpacity
        key={`preauth-${record.id}`}
        style={styles.preAuthCard}
        onPress={() => {
          setSelectedPreAuth({ ...record, analysisId: record.analysis_id });
          setShowPreAuthModal(true);
        }}
      >
        <View style={styles.preAuthHeader}>
          <View style={[styles.statusIndicator, { backgroundColor: statusInfo.color }]} />
          <View style={styles.preAuthInfo}>
            <Text style={styles.preAuthDiagnosis}>
              {preAuth?.form_data?.diagnosis?.primary_diagnosis || record.diagnosis}
            </Text>
            <Text style={styles.preAuthIcd}>
              {preAuth?.form_data?.diagnosis?.icd10_code || 'N/A'}
            </Text>
          </View>
          <View style={[styles.statusBadge, { backgroundColor: `${statusInfo.color}20` }]}>
            <Ionicons name={statusInfo.icon as any} size={14} color={statusInfo.color} />
            <Text style={[styles.statusBadgeText, { color: statusInfo.color }]}>
              {statusInfo.label}
            </Text>
          </View>
        </View>

        <View style={styles.preAuthDetails}>
          <View style={styles.preAuthDetailRow}>
            <Text style={styles.preAuthDetailLabel}>Urgency:</Text>
            <Text style={styles.preAuthDetailValue}>
              {preAuth?.form_data?.urgency || 'Routine'}
            </Text>
          </View>
          <View style={styles.preAuthDetailRow}>
            <Text style={styles.preAuthDetailLabel}>Procedures:</Text>
            <Text style={styles.preAuthDetailValue}>
              {preAuth?.form_data?.procedures_requested?.length || 0} requested
            </Text>
          </View>
          {preAuth?.submission_status?.submitted_date && (
            <View style={styles.preAuthDetailRow}>
              <Text style={styles.preAuthDetailLabel}>Submitted:</Text>
              <Text style={styles.preAuthDetailValue}>
                {formatDate(preAuth.submission_status.submitted_date)}
              </Text>
            </View>
          )}
        </View>

        <View style={styles.preAuthActions}>
          <TouchableOpacity
            style={styles.preAuthActionBtn}
            onPress={() => downloadPreAuthPDF(record.analysis_id)}
          >
            <Ionicons name="document-attach-outline" size={16} color="#2563eb" />
            <Text style={styles.preAuthActionText}>Download PDF</Text>
          </TouchableOpacity>
        </View>
      </TouchableOpacity>
    );
  };

  // Render code search result
  const renderCodeResult = (code: BillingCode, index: number) => (
    <View key={index} style={styles.codeResultCard}>
      <View style={styles.codeResultHeader}>
        <View style={[
          styles.codeTypeBadge,
          { backgroundColor: code.code_type === 'CPT' ? '#dbeafe' : '#fef3c7' }
        ]}>
          <Text style={[
            styles.codeTypeBadgeText,
            { color: code.code_type === 'CPT' ? '#2563eb' : '#92400e' }
          ]}>
            {code.code_type}
          </Text>
        </View>
        <Text style={styles.codeResultCode}>{code.code}</Text>
      </View>
      <Text style={styles.codeResultDescription}>{code.description}</Text>
      <View style={styles.codeResultFooter}>
        <Text style={styles.codeResultCategory}>{code.category}</Text>
        {code.typical_reimbursement && (
          <Text style={styles.codeResultReimbursement}>
            Est. {formatCurrency(code.typical_reimbursement)}
          </Text>
        )}
      </View>
    </View>
  );

  // Render detail modal
  const renderDetailModal = () => {
    if (!selectedRecord) return null;

    return (
      <Modal visible={showDetailModal} animationType="slide" presentationStyle="pageSheet">
        <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <TouchableOpacity onPress={() => setShowDetailModal(false)}>
              <Ionicons name="close" size={28} color="#1e3a5f" />
            </TouchableOpacity>
            <Text style={styles.modalTitle}>Billing Details</Text>
            <View style={{ width: 28 }} />
          </View>

          <ScrollView style={styles.modalContent}>
            <View style={styles.detailSection}>
              <Text style={styles.detailSectionTitle}>Diagnosis</Text>
              <Text style={styles.detailDiagnosis}>{selectedRecord.diagnosis}</Text>
              <Text style={styles.detailDate}>
                Procedure Date: {formatDate(selectedRecord.procedure_date)}
              </Text>
            </View>

            <View style={styles.detailSection}>
              <Text style={styles.detailSectionTitle}>CPT Codes</Text>
              {selectedRecord.cpt_codes.map((code, idx) => (
                <View key={idx} style={styles.detailCodeRow}>
                  <View style={styles.detailCodeBadge}>
                    <Text style={styles.detailCodeBadgeText}>{code.code}</Text>
                  </View>
                  <View style={styles.detailCodeInfo}>
                    <Text style={styles.detailCodeDescription}>{code.description}</Text>
                    <Text style={styles.detailCodeCategory}>{code.category}</Text>
                  </View>
                </View>
              ))}
            </View>

            <View style={styles.detailSection}>
              <Text style={styles.detailSectionTitle}>ICD-10 Codes</Text>
              {selectedRecord.icd10_codes.map((code, idx) => (
                <View key={idx} style={styles.detailCodeRow}>
                  <View style={[styles.detailCodeBadge, styles.icdBadge]}>
                    <Text style={[styles.detailCodeBadgeText, styles.icdBadgeText]}>{code.code}</Text>
                  </View>
                  <View style={styles.detailCodeInfo}>
                    <Text style={styles.detailCodeDescription}>{code.description}</Text>
                  </View>
                </View>
              ))}
            </View>

            <View style={styles.detailSection}>
              <Text style={styles.detailSectionTitle}>Financial Summary</Text>
              <View style={styles.financialRow}>
                <Text style={styles.financialLabel}>Total Charges</Text>
                <Text style={styles.financialValue}>{formatCurrency(selectedRecord.total_charges)}</Text>
              </View>
              <View style={styles.financialRow}>
                <Text style={styles.financialLabel}>Estimated Reimbursement (80%)</Text>
                <Text style={[styles.financialValue, styles.reimbursementText]}>
                  {formatCurrency(selectedRecord.estimated_reimbursement)}
                </Text>
              </View>
            </View>

            <View style={styles.detailActions}>
              <TouchableOpacity
                style={styles.detailActionBtn}
                onPress={() => exportCMS1500(selectedRecord.id)}
              >
                <Ionicons name="download-outline" size={20} color="#fff" />
                <Text style={styles.detailActionBtnText}>Export CMS-1500</Text>
              </TouchableOpacity>

              {selectedRecord.insurance_preauthorization && (
                <TouchableOpacity
                  style={[styles.detailActionBtn, styles.preAuthBtn]}
                  onPress={() => {
                    setShowDetailModal(false);
                    setSelectedPreAuth({ ...selectedRecord, analysisId: selectedRecord.analysis_id });
                    setShowPreAuthModal(true);
                  }}
                >
                  <Ionicons name="shield-checkmark-outline" size={20} color="#fff" />
                  <Text style={styles.detailActionBtnText}>View Pre-Auth</Text>
                </TouchableOpacity>
              )}
            </View>

            <View style={{ height: 40 }} />
          </ScrollView>
        </LinearGradient>
      </Modal>
    );
  };

  // Render pre-auth modal
  const renderPreAuthModal = () => {
    if (!selectedPreAuth) return null;

    const preAuth = selectedPreAuth.insurance_preauthorization as PreAuthData;
    const currentStatus = selectedPreAuth.preauth_status ||
      preAuth?.submission_status?.current_status || 'DRAFT';
    const statusInfo = getStatusInfo(currentStatus);

    return (
      <Modal visible={showPreAuthModal} animationType="slide" presentationStyle="pageSheet">
        <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <TouchableOpacity onPress={() => setShowPreAuthModal(false)}>
              <Ionicons name="close" size={28} color="#1e3a5f" />
            </TouchableOpacity>
            <Text style={styles.modalTitle}>Pre-Authorization</Text>
            <View style={{ width: 28 }} />
          </View>

          <ScrollView style={styles.modalContent}>
            {/* Status Banner */}
            <View style={[styles.statusBanner, { backgroundColor: `${statusInfo.color}15` }]}>
              <Ionicons name={statusInfo.icon as any} size={32} color={statusInfo.color} />
              <View style={styles.statusBannerInfo}>
                <Text style={[styles.statusBannerTitle, { color: statusInfo.color }]}>
                  {statusInfo.label}
                </Text>
                <Text style={styles.statusBannerSubtitle}>Current Status</Text>
              </View>
            </View>

            {/* Diagnosis Info */}
            <View style={styles.preAuthSection}>
              <Text style={styles.preAuthSectionTitle}>Diagnosis Information</Text>
              <View style={styles.preAuthInfoCard}>
                <View style={styles.preAuthInfoRow}>
                  <Text style={styles.preAuthInfoLabel}>Primary Diagnosis</Text>
                  <Text style={styles.preAuthInfoValue}>
                    {preAuth?.form_data?.diagnosis?.primary_diagnosis || 'N/A'}
                  </Text>
                </View>
                <View style={styles.preAuthInfoRow}>
                  <Text style={styles.preAuthInfoLabel}>ICD-10 Code</Text>
                  <Text style={styles.preAuthInfoValue}>
                    {preAuth?.form_data?.diagnosis?.icd10_code || 'N/A'}
                  </Text>
                </View>
                <View style={styles.preAuthInfoRow}>
                  <Text style={styles.preAuthInfoLabel}>Confidence Level</Text>
                  <Text style={styles.preAuthInfoValue}>
                    {preAuth?.form_data?.diagnosis?.confidence_level || 'N/A'}
                  </Text>
                </View>
                <View style={styles.preAuthInfoRow}>
                  <Text style={styles.preAuthInfoLabel}>Urgency</Text>
                  <Text style={[styles.preAuthInfoValue, styles.urgencyText]}>
                    {preAuth?.form_data?.urgency || 'Routine'}
                  </Text>
                </View>
              </View>
            </View>

            {/* Requested Procedures */}
            {preAuth?.form_data?.procedures_requested && (
              <View style={styles.preAuthSection}>
                <Text style={styles.preAuthSectionTitle}>Requested Procedures</Text>
                {preAuth.form_data.procedures_requested.map((proc, idx) => (
                  <View key={idx} style={styles.procedureCard}>
                    <View style={styles.procedureHeader}>
                      <View style={styles.procedureCodeBadge}>
                        <Text style={styles.procedureCodeText}>{proc.code}</Text>
                      </View>
                      <Text style={styles.procedureDescription}>{proc.description}</Text>
                    </View>
                    <Text style={styles.procedureRationale}>{proc.rationale}</Text>
                  </View>
                ))}
              </View>
            )}

            {/* Clinical Rationale */}
            {preAuth?.form_data?.clinical_rationale && (
              <View style={styles.preAuthSection}>
                <Text style={styles.preAuthSectionTitle}>Clinical Rationale</Text>
                <View style={styles.rationaleCard}>
                  <Text style={styles.rationaleText}>
                    {preAuth.form_data.clinical_rationale}
                  </Text>
                </View>
              </View>
            )}

            {/* Timeline */}
            {preAuth?.form_data?.estimated_timeline && (
              <View style={styles.preAuthSection}>
                <Text style={styles.preAuthSectionTitle}>Estimated Timeline</Text>
                <View style={styles.timelineCard}>
                  <Ionicons name="time-outline" size={20} color="#2563eb" />
                  <Text style={styles.timelineText}>
                    {preAuth.form_data.estimated_timeline}
                  </Text>
                </View>
              </View>
            )}

            {/* Status Update */}
            <View style={styles.preAuthSection}>
              <Text style={styles.preAuthSectionTitle}>Update Status</Text>
              <View style={styles.statusUpdateGrid}>
                {PREAUTH_STATUSES.map(status => (
                  <TouchableOpacity
                    key={status.key}
                    style={[
                      styles.statusUpdateBtn,
                      currentStatus === status.key && styles.statusUpdateBtnActive,
                      { borderColor: status.color }
                    ]}
                    onPress={() => {
                      if (status.key !== currentStatus) {
                        Alert.alert(
                          'Update Status',
                          `Change status to "${status.label}"?`,
                          [
                            { text: 'Cancel', style: 'cancel' },
                            {
                              text: 'Update',
                              onPress: () => updatePreAuthStatus(selectedPreAuth.analysisId, status.key)
                            }
                          ]
                        );
                      }
                    }}
                    disabled={isUpdatingStatus}
                  >
                    <Ionicons
                      name={status.icon as any}
                      size={18}
                      color={currentStatus === status.key ? '#fff' : status.color}
                    />
                    <Text style={[
                      styles.statusUpdateBtnText,
                      currentStatus === status.key && styles.statusUpdateBtnTextActive,
                      { color: currentStatus === status.key ? '#fff' : status.color }
                    ]}>
                      {status.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Actions */}
            <View style={styles.preAuthModalActions}>
              <TouchableOpacity
                style={styles.preAuthModalActionBtn}
                onPress={() => downloadPreAuthPDF(selectedPreAuth.analysisId)}
              >
                <Ionicons name="document-attach-outline" size={20} color="#fff" />
                <Text style={styles.preAuthModalActionText}>Download Full Documentation</Text>
              </TouchableOpacity>
            </View>

            <View style={{ height: 40 }} />
          </ScrollView>
        </LinearGradient>
      </Modal>
    );
  };

  // Render appeal detail modal
  const renderAppealModal = () => {
    if (!selectedAppeal) return null;

    const statusInfo = getAppealStatusInfo(selectedAppeal.appeal_status);
    const levelInfo = getAppealLevelInfo(selectedAppeal.appeal_level);

    return (
      <Modal visible={showAppealModal} animationType="slide" presentationStyle="pageSheet">
        <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <TouchableOpacity onPress={() => setShowAppealModal(false)}>
              <Ionicons name="close" size={28} color="#1e3a5f" />
            </TouchableOpacity>
            <Text style={styles.modalTitle}>Appeal Details</Text>
            <View style={{ width: 28 }} />
          </View>

          <ScrollView style={styles.modalContent}>
            {/* Status Banner */}
            <View style={[styles.statusBanner, { backgroundColor: `${statusInfo.color}15` }]}>
              <Ionicons name={statusInfo.icon as any} size={32} color={statusInfo.color} />
              <View style={styles.statusBannerInfo}>
                <Text style={[styles.statusBannerTitle, { color: statusInfo.color }]}>
                  {statusInfo.label}
                </Text>
                <Text style={styles.statusBannerSubtitle}>{levelInfo.label}</Text>
              </View>
              {selectedAppeal.success_likelihood !== undefined && (
                <View style={styles.successLikelihoodBadge}>
                  <Text style={styles.successLikelihoodLabel}>Success</Text>
                  <Text style={[styles.successLikelihoodValue, {
                    color: selectedAppeal.success_likelihood >= 70 ? '#10b981' :
                           selectedAppeal.success_likelihood >= 50 ? '#f59e0b' : '#ef4444'
                  }]}>
                    {selectedAppeal.success_likelihood}%
                  </Text>
                </View>
              )}
            </View>

            {/* Claim Details */}
            <View style={styles.preAuthSection}>
              <Text style={styles.preAuthSectionTitle}>Claim Information</Text>
              <View style={styles.preAuthInfoCard}>
                <View style={styles.preAuthInfoRow}>
                  <Text style={styles.preAuthInfoLabel}>Insurance</Text>
                  <Text style={styles.preAuthInfoValue}>{selectedAppeal.insurance_company}</Text>
                </View>
                <View style={styles.preAuthInfoRow}>
                  <Text style={styles.preAuthInfoLabel}>Claim Number</Text>
                  <Text style={styles.preAuthInfoValue}>{selectedAppeal.claim_number}</Text>
                </View>
                <View style={styles.preAuthInfoRow}>
                  <Text style={styles.preAuthInfoLabel}>Diagnosis</Text>
                  <Text style={styles.preAuthInfoValue}>{selectedAppeal.diagnosis}</Text>
                </View>
                <View style={styles.preAuthInfoRow}>
                  <Text style={styles.preAuthInfoLabel}>Denial Reason</Text>
                  <Text style={styles.preAuthInfoValue}>
                    {selectedAppeal.denial_reason.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </Text>
                </View>
                {selectedAppeal.deadline && (
                  <View style={styles.preAuthInfoRow}>
                    <Text style={styles.preAuthInfoLabel}>Appeal Deadline</Text>
                    <Text style={[styles.preAuthInfoValue, { color: '#ef4444' }]}>
                      {formatDate(selectedAppeal.deadline)}
                    </Text>
                  </View>
                )}
              </View>
            </View>

            {/* Appeal Letter Content */}
            {selectedAppeal.letter_content && (
              <View style={styles.preAuthSection}>
                <Text style={styles.preAuthSectionTitle}>Appeal Letter</Text>
                <View style={styles.letterContentCard}>
                  <ScrollView style={styles.letterScrollView} nestedScrollEnabled>
                    <Text style={styles.letterContentText}>{selectedAppeal.letter_content}</Text>
                  </ScrollView>
                </View>
              </View>
            )}

            {/* Actions */}
            <View style={styles.preAuthModalActions}>
              <TouchableOpacity
                style={styles.preAuthModalActionBtn}
                onPress={() => {
                  Alert.alert('Copy to Clipboard', 'Appeal letter copied!');
                }}
              >
                <Ionicons name="copy-outline" size={20} color="#fff" />
                <Text style={styles.preAuthModalActionText}>Copy Letter</Text>
              </TouchableOpacity>
            </View>

            <View style={{ height: 40 }} />
          </ScrollView>
        </LinearGradient>
      </Modal>
    );
  };

  // Render new appeal modal
  const renderNewAppealModal = () => {
    return (
      <Modal visible={showNewAppealModal} animationType="slide" presentationStyle="pageSheet">
        <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <TouchableOpacity onPress={() => setShowNewAppealModal(false)}>
              <Ionicons name="close" size={28} color="#1e3a5f" />
            </TouchableOpacity>
            <Text style={styles.modalTitle}>Generate Appeal</Text>
            <View style={{ width: 28 }} />
          </View>

          <ScrollView style={styles.modalContent}>
            {/* Select Billing Record */}
            <View style={styles.preAuthSection}>
              <Text style={styles.preAuthSectionTitle}>Select Billing Record</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.recordSelector}>
                {billingRecords.map(record => (
                  <TouchableOpacity
                    key={record.id}
                    style={[
                      styles.recordSelectorItem,
                      newAppealData.record?.id === record.id && styles.recordSelectorItemActive
                    ]}
                    onPress={() => setNewAppealData({ ...newAppealData, record })}
                  >
                    <Text style={[
                      styles.recordSelectorDiagnosis,
                      newAppealData.record?.id === record.id && styles.recordSelectorTextActive
                    ]}>
                      {record.diagnosis}
                    </Text>
                    <Text style={[
                      styles.recordSelectorDate,
                      newAppealData.record?.id === record.id && styles.recordSelectorTextActive
                    ]}>
                      {formatDate(record.procedure_date)}
                    </Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            </View>

            {/* Insurance Company */}
            <View style={styles.preAuthSection}>
              <Text style={styles.preAuthSectionTitle}>Insurance Company</Text>
              <TextInput
                style={styles.appealInput}
                placeholder="Enter insurance company name"
                value={newAppealData.insuranceCompany}
                onChangeText={(text) => setNewAppealData({ ...newAppealData, insuranceCompany: text })}
                placeholderTextColor="#9ca3af"
              />
            </View>

            {/* Claim Number */}
            <View style={styles.preAuthSection}>
              <Text style={styles.preAuthSectionTitle}>Claim Number</Text>
              <TextInput
                style={styles.appealInput}
                placeholder="Enter claim number from denial"
                value={newAppealData.claimNumber}
                onChangeText={(text) => setNewAppealData({ ...newAppealData, claimNumber: text })}
                placeholderTextColor="#9ca3af"
              />
            </View>

            {/* Denial Reason */}
            <View style={styles.preAuthSection}>
              <Text style={styles.preAuthSectionTitle}>Denial Reason</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.denialReasonSelector}>
                {[
                  { value: 'medical_necessity', label: 'Medical Necessity' },
                  { value: 'not_covered', label: 'Not Covered' },
                  { value: 'experimental', label: 'Experimental' },
                  { value: 'prior_auth_required', label: 'Prior Auth Required' },
                  { value: 'cosmetic', label: 'Cosmetic' },
                  { value: 'coding_error', label: 'Coding Error' },
                  { value: 'other', label: 'Other' },
                ].map(reason => (
                  <TouchableOpacity
                    key={reason.value}
                    style={[
                      styles.denialReasonItem,
                      newAppealData.denialReason === reason.value && styles.denialReasonItemActive
                    ]}
                    onPress={() => setNewAppealData({ ...newAppealData, denialReason: reason.value })}
                  >
                    <Text style={[
                      styles.denialReasonText,
                      newAppealData.denialReason === reason.value && styles.denialReasonTextActive
                    ]}>
                      {reason.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            </View>

            {/* Additional Details */}
            <View style={styles.preAuthSection}>
              <Text style={styles.preAuthSectionTitle}>Denial Details (Optional)</Text>
              <TextInput
                style={[styles.appealInput, styles.appealTextArea]}
                placeholder="Enter the exact wording of the denial reason from your EOB..."
                value={newAppealData.denialReasonText}
                onChangeText={(text) => setNewAppealData({ ...newAppealData, denialReasonText: text })}
                placeholderTextColor="#9ca3af"
                multiline
                numberOfLines={4}
              />
            </View>

            {/* Generate Button */}
            <TouchableOpacity
              style={[styles.generateAppealButton, isGeneratingAppeal && styles.buttonDisabled]}
              onPress={() => {
                if (!newAppealData.record) {
                  Alert.alert('Error', 'Please select a billing record');
                  return;
                }
                if (!newAppealData.insuranceCompany) {
                  Alert.alert('Error', 'Please enter the insurance company name');
                  return;
                }
                if (!newAppealData.claimNumber) {
                  Alert.alert('Error', 'Please enter the claim number');
                  return;
                }
                generateAppealLetter(
                  newAppealData.record.analysis_id,
                  newAppealData.denialReason,
                  newAppealData.insuranceCompany,
                  newAppealData.claimNumber,
                  newAppealData.denialReasonText
                );
              }}
              disabled={isGeneratingAppeal}
            >
              {isGeneratingAppeal ? (
                <>
                  <ActivityIndicator size="small" color="#fff" />
                  <Text style={styles.generateAppealButtonText}>Generating...</Text>
                </>
              ) : (
                <>
                  <Ionicons name="document-text-outline" size={20} color="#fff" />
                  <Text style={styles.generateAppealButtonText}>Generate Appeal Letter</Text>
                </>
              )}
            </TouchableOpacity>

            <View style={{ height: 40 }} />
          </ScrollView>
        </LinearGradient>
      </Modal>
    );
  };

  if (isLoading) {
    return (
      <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2563eb" />
          <Text style={styles.loadingText}>Loading billing records...</Text>
        </View>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient colors={['#f0f9ff', '#e0f2fe', '#bae6fd']} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#2563eb" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Insurance & Appeals</Text>
        <View style={styles.headerSpacer} />
      </View>

      {/* Tabs */}
      <View style={styles.tabs}>
        {(['billing', 'preauth', 'appeals', 'codes', 'hsa'] as TabType[]).map(tab => {
          const tabConfig = {
            billing: { icon: 'receipt-outline' as const, label: 'Bills' },
            preauth: { icon: 'shield-checkmark-outline' as const, label: 'Auth' },
            appeals: { icon: 'document-text-outline' as const, label: 'Appeals' },
            codes: { icon: 'barcode-outline' as const, label: 'Codes' },
            hsa: { icon: 'wallet-outline' as const, label: 'HSA' },
          };
          const { icon, label } = tabConfig[tab];
          const isActive = activeTab === tab;
          const badgeCount = tab === 'preauth' ? recordsWithPreAuth.length :
                            tab === 'appeals' ? appeals.length : 0;

          return (
            <TouchableOpacity
              key={tab}
              style={[styles.tab, isActive && styles.tabActive]}
              onPress={() => setActiveTab(tab)}
            >
              <View style={styles.tabIconWrapper}>
                <Ionicons name={icon} size={20} color={isActive ? '#2563eb' : '#6b7280'} />
                {badgeCount > 0 && (
                  <View style={styles.tabBadge}>
                    <Text style={styles.tabBadgeText}>{badgeCount}</Text>
                  </View>
                )}
              </View>
              <Text style={[styles.tabText, isActive && styles.tabTextActive]}>
                {label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {/* Content */}
      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {activeTab === 'billing' && (
          <>
            {billingRecords.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="receipt-outline" size={64} color="#9ca3af" />
                <Text style={styles.emptyTitle}>No Billing Records</Text>
                <Text style={styles.emptyText}>
                  Billing records will appear here after you complete skin analyses
                </Text>
              </View>
            ) : (
              billingRecords.map(renderBillingCard)
            )}
          </>
        )}

        {activeTab === 'preauth' && (
          <>
            {recordsWithPreAuth.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="shield-outline" size={64} color="#9ca3af" />
                <Text style={styles.emptyTitle}>No Pre-Authorizations</Text>
                <Text style={styles.emptyText}>
                  Insurance pre-authorization data will appear here for analyses that require it
                </Text>
              </View>
            ) : (
              recordsWithPreAuth.map(renderPreAuthCard)
            )}
          </>
        )}

        {activeTab === 'appeals' && (
          <>
            {/* New Appeal Button */}
            <TouchableOpacity
              style={styles.newAppealButton}
              onPress={() => {
                if (billingRecords.length === 0) {
                  Alert.alert('No Records', 'You need billing records to generate an appeal letter.');
                  return;
                }
                setNewAppealData({
                  record: billingRecords[0],
                  denialReason: 'medical_necessity',
                  insuranceCompany: '',
                  claimNumber: '',
                  denialReasonText: '',
                });
                setShowNewAppealModal(true);
              }}
            >
              <Ionicons name="add-circle-outline" size={20} color="#fff" />
              <Text style={styles.newAppealButtonText}>Generate New Appeal</Text>
            </TouchableOpacity>

            {appeals.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="document-text-outline" size={64} color="#9ca3af" />
                <Text style={styles.emptyTitle}>No Appeals Yet</Text>
                <Text style={styles.emptyText}>
                  Generate an appeal letter when a claim is denied by insurance
                </Text>
              </View>
            ) : (
              appeals.map((appeal) => {
                const statusInfo = getAppealStatusInfo(appeal.appeal_status);
                const levelInfo = getAppealLevelInfo(appeal.appeal_level);
                return (
                  <TouchableOpacity
                    key={appeal.id}
                    style={styles.appealCard}
                    onPress={() => {
                      setSelectedAppeal(appeal);
                      setShowAppealModal(true);
                    }}
                  >
                    <View style={styles.appealHeader}>
                      <View style={[styles.statusIndicator, { backgroundColor: statusInfo.color }]} />
                      <View style={styles.appealInfo}>
                        <Text style={styles.appealDiagnosis}>{appeal.diagnosis}</Text>
                        <Text style={styles.appealInsurance}>{appeal.insurance_company}</Text>
                      </View>
                      <View style={[styles.statusBadge, { backgroundColor: `${statusInfo.color}20` }]}>
                        <Ionicons name={statusInfo.icon as any} size={14} color={statusInfo.color} />
                        <Text style={[styles.statusBadgeText, { color: statusInfo.color }]}>
                          {statusInfo.label}
                        </Text>
                      </View>
                    </View>

                    <View style={styles.appealDetails}>
                      <View style={styles.appealDetailRow}>
                        <Text style={styles.appealDetailLabel}>Appeal Level:</Text>
                        <Text style={styles.appealDetailValue}>{levelInfo.label}</Text>
                      </View>
                      <View style={styles.appealDetailRow}>
                        <Text style={styles.appealDetailLabel}>Denial Reason:</Text>
                        <Text style={styles.appealDetailValue}>
                          {appeal.denial_reason.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                        </Text>
                      </View>
                      {appeal.success_likelihood !== undefined && (
                        <View style={styles.appealDetailRow}>
                          <Text style={styles.appealDetailLabel}>Success Likelihood:</Text>
                          <Text style={[styles.appealDetailValue, {
                            color: appeal.success_likelihood >= 70 ? '#10b981' :
                                   appeal.success_likelihood >= 50 ? '#f59e0b' : '#ef4444'
                          }]}>
                            {appeal.success_likelihood}%
                          </Text>
                        </View>
                      )}
                      {appeal.deadline && (
                        <View style={styles.appealDetailRow}>
                          <Text style={styles.appealDetailLabel}>Deadline:</Text>
                          <Text style={styles.appealDetailValue}>{formatDate(appeal.deadline)}</Text>
                        </View>
                      )}
                    </View>

                    <View style={styles.appealActions}>
                      <TouchableOpacity
                        style={styles.appealActionBtn}
                        onPress={() => {
                          setSelectedAppeal(appeal);
                          setShowAppealModal(true);
                        }}
                      >
                        <Ionicons name="eye-outline" size={16} color="#2563eb" />
                        <Text style={styles.appealActionText}>View Letter</Text>
                      </TouchableOpacity>
                    </View>
                  </TouchableOpacity>
                );
              })
            )}
          </>
        )}

        {activeTab === 'codes' && (
          <>
            <View style={styles.searchContainer}>
              <View style={styles.searchInputWrapper}>
                <Ionicons name="search" size={20} color="#6b7280" />
                <TextInput
                  style={styles.searchInput}
                  placeholder="Search CPT or ICD-10 codes..."
                  value={searchQuery}
                  onChangeText={text => {
                    setSearchQuery(text);
                    searchCodes(text);
                  }}
                  placeholderTextColor="#9ca3af"
                />
                {searchQuery.length > 0 && (
                  <TouchableOpacity onPress={() => {
                    setSearchQuery('');
                    setSearchResults([]);
                  }}>
                    <Ionicons name="close-circle" size={20} color="#9ca3af" />
                  </TouchableOpacity>
                )}
              </View>
            </View>

            {isSearching ? (
              <View style={styles.searchingContainer}>
                <ActivityIndicator size="small" color="#2563eb" />
                <Text style={styles.searchingText}>Searching...</Text>
              </View>
            ) : searchResults.length > 0 ? (
              searchResults.map(renderCodeResult)
            ) : searchQuery.length >= 2 ? (
              <View style={styles.noResultsContainer}>
                <Ionicons name="search-outline" size={48} color="#9ca3af" />
                <Text style={styles.noResultsText}>No codes found for "{searchQuery}"</Text>
              </View>
            ) : (
              <View style={styles.searchHint}>
                <Ionicons name="information-circle-outline" size={24} color="#6b7280" />
                <Text style={styles.searchHintText}>
                  Enter at least 2 characters to search for CPT or ICD-10 codes
                </Text>
              </View>
            )}
          </>
        )}

        {activeTab === 'hsa' && (
          <>
            {/* HSA Year Selector */}
            <View style={styles.hsaYearSelector}>
              <TouchableOpacity
                style={styles.hsaYearBtn}
                onPress={() => {
                  const newYear = hsaYear - 1;
                  setHsaYear(newYear);
                  loadHsaExpenses(newYear);
                }}
              >
                <Ionicons name="chevron-back" size={24} color="#2563eb" />
              </TouchableOpacity>
              <Text style={styles.hsaYearText}>Tax Year {hsaYear}</Text>
              <TouchableOpacity
                style={styles.hsaYearBtn}
                onPress={() => {
                  const newYear = hsaYear + 1;
                  if (newYear <= new Date().getFullYear()) {
                    setHsaYear(newYear);
                    loadHsaExpenses(newYear);
                  }
                }}
                disabled={hsaYear >= new Date().getFullYear()}
              >
                <Ionicons name="chevron-forward" size={24} color={hsaYear >= new Date().getFullYear() ? '#d1d5db' : '#2563eb'} />
              </TouchableOpacity>
            </View>

            {/* HSA Summary Cards */}
            {hsaSummary && (
              <View style={styles.hsaSummaryContainer}>
                <View style={[styles.hsaSummaryCard, styles.hsaEligibleCard]}>
                  <Ionicons name="checkmark-circle" size={28} color="#10b981" />
                  <Text style={styles.hsaSummaryLabel}>Eligible Expenses</Text>
                  <Text style={styles.hsaSummaryValue}>${hsaSummary.total_eligible.toFixed(2)}</Text>
                  <Text style={styles.hsaSummaryCount}>{hsaSummary.eligible_count} items</Text>
                </View>
                <View style={[styles.hsaSummaryCard, styles.hsaIneligibleCard]}>
                  <Ionicons name="close-circle" size={28} color="#ef4444" />
                  <Text style={styles.hsaSummaryLabel}>Ineligible</Text>
                  <Text style={[styles.hsaSummaryValue, { color: '#ef4444' }]}>${hsaSummary.total_ineligible.toFixed(2)}</Text>
                  <Text style={styles.hsaSummaryCount}>{hsaSummary.expense_count - hsaSummary.eligible_count} items</Text>
                </View>
              </View>
            )}

            {/* Download Year Summary Button */}
            <TouchableOpacity style={styles.downloadSummaryBtn} onPress={downloadYearSummary}>
              <Ionicons name="download-outline" size={20} color="#fff" />
              <Text style={styles.downloadSummaryBtnText}>Download {hsaYear} Tax Summary (PDF)</Text>
            </TouchableOpacity>

            {/* Info Banner */}
            <View style={styles.hsaInfoBanner}>
              <Ionicons name="information-circle" size={20} color="#2563eb" />
              <Text style={styles.hsaInfoText}>
                HSA/FSA eligible expenses are determined based on IRS Publication 502. Medical procedures for diagnosis or treatment are generally eligible.
              </Text>
            </View>

            {/* Expenses List */}
            {isLoadingHsa ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#2563eb" />
                <Text style={styles.loadingText}>Loading expenses...</Text>
              </View>
            ) : hsaExpenses.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="wallet-outline" size={64} color="#9ca3af" />
                <Text style={styles.emptyTitle}>No Expenses Found</Text>
                <Text style={styles.emptyText}>
                  Medical expenses from your skin analyses will appear here for HSA/FSA tracking
                </Text>
                <TouchableOpacity
                  style={styles.hsaRefreshBtn}
                  onPress={() => loadHsaExpenses(hsaYear)}
                >
                  <Ionicons name="refresh" size={20} color="#2563eb" />
                  <Text style={styles.hsaRefreshBtnText}>Refresh Expenses</Text>
                </TouchableOpacity>
              </View>
            ) : (
              hsaExpenses.map((expense) => (
                <View key={expense.id} style={[styles.hsaExpenseCard, !expense.eligible && styles.hsaExpenseIneligible]}>
                  <View style={styles.hsaExpenseHeader}>
                    <View style={styles.hsaExpenseLeft}>
                      <View style={[styles.hsaEligibilityBadge, expense.eligible ? styles.hsaEligibleBadge : styles.hsaIneligibleBadge]}>
                        <Ionicons
                          name={expense.eligible ? 'checkmark-circle' : 'close-circle'}
                          size={16}
                          color={expense.eligible ? '#10b981' : '#ef4444'}
                        />
                        <Text style={[styles.hsaEligibilityText, { color: expense.eligible ? '#10b981' : '#ef4444' }]}>
                          {expense.eligible ? 'HSA/FSA Eligible' : 'Not Eligible'}
                        </Text>
                      </View>
                      <Text style={styles.hsaExpenseDate}>{new Date(expense.date).toLocaleDateString()}</Text>
                    </View>
                    <Text style={styles.hsaExpenseAmount}>${expense.amount.toFixed(2)}</Text>
                  </View>
                  <Text style={styles.hsaExpenseDescription}>{expense.description}</Text>
                  {expense.provider && (
                    <Text style={styles.hsaExpenseProvider}>Provider: {expense.provider}</Text>
                  )}
                  <View style={styles.hsaExpenseMeta}>
                    <View style={styles.hsaCategoryBadge}>
                      <Text style={styles.hsaCategoryText}>{expense.category_name}</Text>
                    </View>
                    {expense.procedure_code && (
                      <Text style={styles.hsaCodeText}>CPT: {expense.procedure_code}</Text>
                    )}
                  </View>
                  {expense.eligible && (
                    <TouchableOpacity
                      style={styles.hsaReceiptBtn}
                      onPress={() => {
                        setSelectedExpense(expense);
                        generateHsaReceipt(expense.id);
                      }}
                    >
                      <Ionicons name="document-text-outline" size={18} color="#2563eb" />
                      <Text style={styles.hsaReceiptBtnText}>Generate Receipt</Text>
                    </TouchableOpacity>
                  )}
                </View>
              ))
            )}
          </>
        )}

        <View style={styles.bottomSpacer} />
      </ScrollView>

      {renderDetailModal()}
      {renderPreAuthModal()}
      {renderAppealModal()}
      {renderNewAppealModal()}

      {/* HSA Receipt Modal */}
      <Modal
        visible={showReceiptModal}
        animationType="slide"
        presentationStyle="pageSheet"
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <TouchableOpacity onPress={() => setShowReceiptModal(false)}>
              <Ionicons name="close" size={24} color="#1f2937" />
            </TouchableOpacity>
            <Text style={styles.modalTitle}>HSA/FSA Receipt</Text>
            <TouchableOpacity onPress={downloadHsaReceipt}>
              <Ionicons name="download-outline" size={24} color="#2563eb" />
            </TouchableOpacity>
          </View>
          <ScrollView style={styles.receiptPreview}>
            {selectedExpense && (
              <View style={styles.receiptContent}>
                <View style={styles.receiptHeader}>
                  <Ionicons name="medical" size={40} color="#2563eb" />
                  <Text style={styles.receiptTitle}>Medical Expense Receipt</Text>
                  <Text style={styles.receiptSubtitle}>For HSA/FSA Reimbursement</Text>
                </View>

                <View style={styles.receiptSection}>
                  <Text style={styles.receiptSectionTitle}>Expense Details</Text>
                  <View style={styles.receiptRow}>
                    <Text style={styles.receiptLabel}>Date:</Text>
                    <Text style={styles.receiptValue}>{new Date(selectedExpense.date).toLocaleDateString()}</Text>
                  </View>
                  <View style={styles.receiptRow}>
                    <Text style={styles.receiptLabel}>Description:</Text>
                    <Text style={styles.receiptValue}>{selectedExpense.description}</Text>
                  </View>
                  <View style={styles.receiptRow}>
                    <Text style={styles.receiptLabel}>Amount:</Text>
                    <Text style={[styles.receiptValue, styles.receiptAmount]}>${selectedExpense.amount.toFixed(2)}</Text>
                  </View>
                  {selectedExpense.provider && (
                    <View style={styles.receiptRow}>
                      <Text style={styles.receiptLabel}>Provider:</Text>
                      <Text style={styles.receiptValue}>{selectedExpense.provider}</Text>
                    </View>
                  )}
                  {selectedExpense.procedure_code && (
                    <View style={styles.receiptRow}>
                      <Text style={styles.receiptLabel}>CPT Code:</Text>
                      <Text style={styles.receiptValue}>{selectedExpense.procedure_code}</Text>
                    </View>
                  )}
                  {selectedExpense.diagnosis && (
                    <View style={styles.receiptRow}>
                      <Text style={styles.receiptLabel}>Diagnosis:</Text>
                      <Text style={styles.receiptValue}>{selectedExpense.diagnosis}</Text>
                    </View>
                  )}
                </View>

                <View style={styles.receiptSection}>
                  <Text style={styles.receiptSectionTitle}>Eligibility</Text>
                  <View style={[styles.eligibilityBox, styles.eligibleBox]}>
                    <Ionicons name="checkmark-circle" size={24} color="#10b981" />
                    <Text style={styles.eligibilityText}>
                      This expense qualifies for HSA/FSA reimbursement under IRS Publication 502 as a medical expense for {selectedExpense.category_name.toLowerCase()}.
                    </Text>
                  </View>
                </View>

                <TouchableOpacity style={styles.downloadReceiptBtn} onPress={downloadHsaReceipt}>
                  <Ionicons name="download-outline" size={20} color="#fff" />
                  <Text style={styles.downloadReceiptBtnText}>Download PDF Receipt</Text>
                </TouchableOpacity>
              </View>
            )}
          </ScrollView>
        </View>
      </Modal>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 20,
    paddingBottom: 15,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  headerSpacer: {
    width: 40,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    color: '#6b7280',
    fontSize: 16,
  },
  tabs: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
    gap: 4,
  },
  tab: {
    flex: 1,
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    paddingHorizontal: 4,
    borderRadius: 10,
    gap: 4,
  },
  tabActive: {
    backgroundColor: '#dbeafe',
  },
  tabIconWrapper: {
    position: 'relative',
  },
  tabBadge: {
    position: 'absolute',
    top: -6,
    right: -10,
    backgroundColor: '#ef4444',
    borderRadius: 8,
    minWidth: 16,
    height: 16,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 4,
  },
  tabBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  tabText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#6b7280',
    textAlign: 'center',
  },
  tabTextActive: {
    color: '#2563eb',
  },
  badge: {
    backgroundColor: '#2563eb',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
    marginLeft: 4,
  },
  badgeText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  billingCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  cardTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flex: 1,
  },
  cardDiagnosis: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
    flex: 1,
  },
  cardDate: {
    fontSize: 13,
    color: '#6b7280',
  },
  codesContainer: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 12,
  },
  codeSection: {
    flex: 1,
  },
  codeSectionTitle: {
    fontSize: 11,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 6,
    textTransform: 'uppercase',
  },
  codeTag: {
    backgroundColor: '#dbeafe',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    alignSelf: 'flex-start',
    marginBottom: 4,
  },
  codeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#2563eb',
  },
  icdTag: {
    backgroundColor: '#fef3c7',
  },
  icdText: {
    color: '#92400e',
  },
  chargesRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
    marginBottom: 12,
  },
  chargeItem: {
    alignItems: 'center',
  },
  chargeLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginBottom: 4,
  },
  chargeValue: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  reimbursementValue: {
    color: '#10b981',
  },
  preAuthStatusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingTop: 8,
    paddingBottom: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  preAuthStatusText: {
    fontSize: 13,
    fontWeight: '600',
  },
  cardActions: {
    flexDirection: 'row',
    gap: 12,
  },
  cardActionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
    backgroundColor: '#f3f4f6',
  },
  cardActionText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#2563eb',
  },
  // Pre-auth card styles
  preAuthCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  preAuthHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  statusIndicator: {
    width: 4,
    height: 40,
    borderRadius: 2,
    marginRight: 12,
  },
  preAuthInfo: {
    flex: 1,
  },
  preAuthDiagnosis: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  preAuthIcd: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
  },
  statusBadgeText: {
    fontSize: 12,
    fontWeight: '600',
  },
  preAuthDetails: {
    backgroundColor: '#f8fafc',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  preAuthDetailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  preAuthDetailLabel: {
    fontSize: 13,
    color: '#6b7280',
  },
  preAuthDetailValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  preAuthActions: {
    flexDirection: 'row',
  },
  preAuthActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
    backgroundColor: '#eff6ff',
  },
  preAuthActionText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#2563eb',
  },
  // Search styles
  searchContainer: {
    marginBottom: 16,
  },
  searchInputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 10,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
    color: '#1e3a5f',
  },
  searchingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 40,
  },
  searchingText: {
    color: '#6b7280',
  },
  noResultsContainer: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  noResultsText: {
    color: '#6b7280',
    marginTop: 12,
    fontSize: 14,
  },
  searchHint: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
  },
  searchHintText: {
    flex: 1,
    color: '#6b7280',
    fontSize: 14,
  },
  codeResultCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
  },
  codeResultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 8,
  },
  codeTypeBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  codeTypeBadgeText: {
    fontSize: 11,
    fontWeight: '700',
  },
  codeResultCode: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  codeResultDescription: {
    fontSize: 14,
    color: '#4b5563',
    marginBottom: 8,
  },
  codeResultFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  codeResultCategory: {
    fontSize: 12,
    color: '#6b7280',
  },
  codeResultReimbursement: {
    fontSize: 13,
    fontWeight: '600',
    color: '#10b981',
  },
  // Empty state
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center',
    paddingHorizontal: 40,
  },
  bottomSpacer: {
    height: 40,
  },
  // Modal styles
  modalContainer: {
    flex: 1,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 20,
    paddingBottom: 15,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  modalContent: {
    flex: 1,
    padding: 16,
  },
  detailSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  detailSectionTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#6b7280',
    marginBottom: 12,
    textTransform: 'uppercase',
  },
  detailDiagnosis: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  detailDate: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 4,
  },
  detailCodeRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
    marginBottom: 12,
  },
  detailCodeBadge: {
    backgroundColor: '#dbeafe',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
  },
  detailCodeBadgeText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#2563eb',
  },
  icdBadge: {
    backgroundColor: '#fef3c7',
  },
  icdBadgeText: {
    color: '#92400e',
  },
  detailCodeInfo: {
    flex: 1,
  },
  detailCodeDescription: {
    fontSize: 14,
    color: '#1e3a5f',
    fontWeight: '500',
  },
  detailCodeCategory: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  financialRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  financialLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  financialValue: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  reimbursementText: {
    color: '#10b981',
  },
  detailActions: {
    gap: 12,
    marginTop: 8,
  },
  detailActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#2563eb',
    paddingVertical: 14,
    borderRadius: 10,
  },
  preAuthBtn: {
    backgroundColor: '#10b981',
  },
  detailActionBtnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  // Pre-auth modal styles
  statusBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
  },
  statusBannerInfo: {
    flex: 1,
  },
  statusBannerTitle: {
    fontSize: 24,
    fontWeight: '700',
  },
  statusBannerSubtitle: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  preAuthSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  preAuthSectionTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#6b7280',
    marginBottom: 12,
    textTransform: 'uppercase',
  },
  preAuthInfoCard: {
    backgroundColor: '#f8fafc',
    borderRadius: 8,
    padding: 12,
  },
  preAuthInfoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  preAuthInfoLabel: {
    fontSize: 13,
    color: '#6b7280',
  },
  preAuthInfoValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1e3a5f',
    flex: 1,
    textAlign: 'right',
  },
  urgencyText: {
    color: '#f59e0b',
  },
  procedureCard: {
    backgroundColor: '#f8fafc',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  procedureHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 8,
  },
  procedureCodeBadge: {
    backgroundColor: '#dbeafe',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  procedureCodeText: {
    fontSize: 12,
    fontWeight: '700',
    color: '#2563eb',
  },
  procedureDescription: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
    flex: 1,
  },
  procedureRationale: {
    fontSize: 13,
    color: '#6b7280',
    fontStyle: 'italic',
  },
  rationaleCard: {
    backgroundColor: '#fffbeb',
    borderRadius: 8,
    padding: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#f59e0b',
  },
  rationaleText: {
    fontSize: 14,
    color: '#78350f',
    lineHeight: 20,
  },
  timelineCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: '#eff6ff',
    borderRadius: 8,
    padding: 12,
  },
  timelineText: {
    fontSize: 14,
    color: '#1e40af',
    flex: 1,
  },
  statusUpdateGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  statusUpdateBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 8,
    borderWidth: 2,
    backgroundColor: '#fff',
  },
  statusUpdateBtnActive: {
    backgroundColor: '#2563eb',
    borderColor: '#2563eb',
  },
  statusUpdateBtnText: {
    fontSize: 13,
    fontWeight: '600',
  },
  statusUpdateBtnTextActive: {
    color: '#fff',
  },
  preAuthModalActions: {
    marginTop: 8,
  },
  preAuthModalActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#2563eb',
    paddingVertical: 14,
    borderRadius: 10,
  },
  preAuthModalActionText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  // Appeal styles
  newAppealButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#2563eb',
    paddingVertical: 14,
    borderRadius: 12,
    marginBottom: 16,
  },
  newAppealButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  appealCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  appealHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  appealInfo: {
    flex: 1,
  },
  appealDiagnosis: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  appealInsurance: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  appealDetails: {
    backgroundColor: '#f8fafc',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  appealDetailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  appealDetailLabel: {
    fontSize: 13,
    color: '#6b7280',
  },
  appealDetailValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1e3a5f',
  },
  appealActions: {
    flexDirection: 'row',
  },
  appealActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
    backgroundColor: '#eff6ff',
  },
  appealActionText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#2563eb',
  },
  successLikelihoodBadge: {
    alignItems: 'center',
    backgroundColor: '#f8fafc',
    borderRadius: 8,
    padding: 8,
  },
  successLikelihoodLabel: {
    fontSize: 10,
    color: '#6b7280',
    marginBottom: 2,
  },
  successLikelihoodValue: {
    fontSize: 20,
    fontWeight: '700',
  },
  letterContentCard: {
    backgroundColor: '#f8fafc',
    borderRadius: 8,
    padding: 12,
    maxHeight: 300,
  },
  letterScrollView: {
    maxHeight: 280,
  },
  letterContentText: {
    fontSize: 13,
    color: '#1e3a5f',
    lineHeight: 20,
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
  },
  recordSelector: {
    marginBottom: 8,
  },
  recordSelectorItem: {
    backgroundColor: '#f1f5f9',
    borderRadius: 8,
    padding: 12,
    marginRight: 10,
    minWidth: 150,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  recordSelectorItemActive: {
    backgroundColor: '#dbeafe',
    borderColor: '#2563eb',
  },
  recordSelectorDiagnosis: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e3a5f',
    marginBottom: 4,
  },
  recordSelectorDate: {
    fontSize: 12,
    color: '#6b7280',
  },
  recordSelectorTextActive: {
    color: '#2563eb',
  },
  appealInput: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 14,
    fontSize: 16,
    color: '#1e3a5f',
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  appealTextArea: {
    minHeight: 100,
    textAlignVertical: 'top',
  },
  denialReasonSelector: {
    marginBottom: 8,
  },
  denialReasonItem: {
    backgroundColor: '#f1f5f9',
    borderRadius: 20,
    paddingVertical: 8,
    paddingHorizontal: 16,
    marginRight: 8,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  denialReasonItemActive: {
    backgroundColor: '#dbeafe',
    borderColor: '#2563eb',
  },
  denialReasonText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
  },
  denialReasonTextActive: {
    color: '#2563eb',
  },
  generateAppealButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#10b981',
    paddingVertical: 16,
    borderRadius: 12,
    marginTop: 16,
  },
  generateAppealButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  // HSA/FSA Styles
  hsaYearSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  hsaYearBtn: {
    padding: 8,
  },
  hsaYearText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e3a5f',
    marginHorizontal: 20,
  },
  hsaSummaryContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  hsaSummaryCard: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  hsaEligibleCard: {
    borderWidth: 2,
    borderColor: '#10b981',
  },
  hsaIneligibleCard: {
    borderWidth: 2,
    borderColor: '#ef4444',
  },
  hsaSummaryLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 8,
  },
  hsaSummaryValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#10b981',
    marginTop: 4,
  },
  hsaSummaryCount: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 4,
  },
  downloadSummaryBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#2563eb',
    paddingVertical: 14,
    borderRadius: 12,
    marginBottom: 16,
  },
  downloadSummaryBtnText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
  hsaInfoBanner: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
    gap: 10,
  },
  hsaInfoText: {
    flex: 1,
    fontSize: 13,
    color: '#1e40af',
    lineHeight: 18,
  },
  hsaRefreshBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#eff6ff',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 10,
    marginTop: 16,
  },
  hsaRefreshBtnText: {
    color: '#2563eb',
    fontSize: 14,
    fontWeight: '600',
  },
  hsaExpenseCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    borderLeftWidth: 4,
    borderLeftColor: '#10b981',
  },
  hsaExpenseIneligible: {
    borderLeftColor: '#ef4444',
    opacity: 0.8,
  },
  hsaExpenseHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  hsaExpenseLeft: {
    flex: 1,
  },
  hsaEligibilityBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginBottom: 4,
  },
  hsaEligibleBadge: {},
  hsaIneligibleBadge: {},
  hsaEligibilityText: {
    fontSize: 12,
    fontWeight: '600',
  },
  hsaExpenseDate: {
    fontSize: 12,
    color: '#9ca3af',
  },
  hsaExpenseAmount: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e3a5f',
  },
  hsaExpenseDescription: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 6,
  },
  hsaExpenseProvider: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 10,
  },
  hsaExpenseMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  hsaCategoryBadge: {
    backgroundColor: '#f3f4f6',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 6,
  },
  hsaCategoryText: {
    fontSize: 12,
    color: '#6b7280',
  },
  hsaCodeText: {
    fontSize: 12,
    color: '#6b7280',
  },
  hsaReceiptBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#eff6ff',
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2563eb',
  },
  hsaReceiptBtnText: {
    color: '#2563eb',
    fontSize: 14,
    fontWeight: '600',
  },
  receiptPreview: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  receiptContent: {
    padding: 20,
  },
  receiptHeader: {
    alignItems: 'center',
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
    marginBottom: 20,
  },
  receiptTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#1e3a5f',
    marginTop: 12,
  },
  receiptSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 4,
  },
  receiptSection: {
    marginBottom: 20,
  },
  receiptSectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 12,
  },
  receiptRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  receiptLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  receiptValue: {
    fontSize: 14,
    color: '#1f2937',
    fontWeight: '500',
    flex: 1,
    textAlign: 'right',
  },
  receiptAmount: {
    fontSize: 18,
    fontWeight: '700',
    color: '#10b981',
  },
  eligibilityBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
    padding: 16,
    borderRadius: 12,
  },
  eligibleBox: {
    backgroundColor: '#f0fdf4',
    borderWidth: 1,
    borderColor: '#10b981',
  },
  eligibilityText: {
    flex: 1,
    fontSize: 14,
    color: '#166534',
    lineHeight: 20,
  },
  downloadReceiptBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#10b981',
    paddingVertical: 16,
    borderRadius: 12,
    marginTop: 20,
  },
  downloadReceiptBtnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
