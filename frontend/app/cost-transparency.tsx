import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Alert,
  RefreshControl,
  Modal,
  Platform,
  Linking,
} from 'react-native';
import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import { Ionicons } from '@expo/vector-icons';
import { API_URL } from '../config';

// Types
interface ProcedureEstimate {
  procedure_key: string;
  procedure_code: string;
  procedure_name: string;
  description: string;
  average_cost: number;
  cost_range_low: number;
  cost_range_high: number;
  medicare_rate: number;
  typical_insurance_coverage: number;
  estimated_out_of_pocket: number;
  factors_affecting_cost: string[];
}

interface Provider {
  provider_id: string;
  provider_name: string;
  specialty: string;
  location: string;
  city: string;
  distance_miles: number;
  consultation_fee: number;
  average_procedure_cost: number;
  accepts_insurance: boolean;
  insurance_networks: string[];
  rating: number;
  review_count: number;
  wait_time_days: number;
  telemedicine_available: boolean;
  telemedicine_fee: number;
  specializations: string[];
}

interface MedicationPrice {
  pharmacy_name: string;
  pharmacy_address: string;
  price: number;
  original_price: number;
  savings: number;
  savings_percent: number;
  coupon_available: boolean;
  coupon_code?: string;
  requires_membership: boolean;
  estimated?: boolean;
}

interface MedicationInfo {
  brand_name: string;
  generic_name: string;
  drug_class: string;
  manufacturer?: string;
  product_type?: string;
  route?: string;
  available_dosages: string[];
  available_forms: string[];
  selected_dosage: string;
  quantity: number;
  indications?: string;
  ndc_codes?: string[];
}

interface PriceEstimate {
  estimated_low: number;
  estimated_mid: number;
  estimated_high: number;
  unit: string;
  drug_class: string;
}

interface ExternalLinks {
  goodrx: string;
  rxsaver: string;
  costco: string;
  blink_health: string;
  amazon_pharmacy: string;
  singlecare: string;
  manufacturer_coupon: string;
}

type TabType = 'procedures' | 'providers' | 'medications' | 'calculator';

export default function CostTransparencyScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<TabType>('procedures');
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // Procedure estimates state
  const [procedureEstimates, setProcedureEstimates] = useState<ProcedureEstimate[]>([]);
  const [selectedProcedure, setSelectedProcedure] = useState<ProcedureEstimate | null>(null);
  const [showProcedureModal, setShowProcedureModal] = useState(false);

  // Provider comparison state
  const [providers, setProviders] = useState<Provider[]>([]);
  const [insuranceFilter, setInsuranceFilter] = useState('');
  const [maxDistance, setMaxDistance] = useState(25);
  const [sortBy, setSortBy] = useState<'price' | 'distance' | 'rating' | 'wait_time'>('price');
  const [selectedProvider, setSelectedProvider] = useState<Provider | null>(null);
  const [showProviderModal, setShowProviderModal] = useState(false);

  // Medication prices state
  const [medicationSearch, setMedicationSearch] = useState('');
  const [medicationInfo, setMedicationInfo] = useState<MedicationInfo | null>(null);
  const [medicationPrices, setMedicationPrices] = useState<MedicationPrice[]>([]);
  const [medicationQuantity, setMedicationQuantity] = useState(30);
  const [medicationSuggestions, setMedicationSuggestions] = useState<string[]>([]);
  const [externalLinks, setExternalLinks] = useState<ExternalLinks | null>(null);
  const [priceEstimate, setPriceEstimate] = useState<PriceEstimate | null>(null);
  const [dataSource, setDataSource] = useState<'openfda' | 'demo' | null>(null);
  const [disclaimer, setDisclaimer] = useState<string>('');

  // Calculator state
  const [selectedProcedures, setSelectedProcedures] = useState<string[]>([]);
  const [selectedMedications, setSelectedMedications] = useState<string[]>([]);
  const [insuranceType, setInsuranceType] = useState('private');
  const [deductibleRemaining, setDeductibleRemaining] = useState(0);
  const [calculatorResult, setCalculatorResult] = useState<any>(null);

  const getAuthHeaders = async () => {
    const token = await SecureStore.getItemAsync('auth_token');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  };

  // Load procedure estimates
  const loadProcedureEstimates = useCallback(async () => {
    try {
      setLoading(true);
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_URL}/costs/procedures`, { headers });

      if (response.ok) {
        const data = await response.json();
        setProcedureEstimates(data.estimates || []);
      }
    } catch (error) {
      console.error('Error loading procedures:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  // Load providers
  const loadProviders = useCallback(async () => {
    try {
      setLoading(true);
      const headers = await getAuthHeaders();
      let url = `${API_URL}/costs/providers?max_distance=${maxDistance}&sort_by=${sortBy}`;
      if (insuranceFilter) {
        url += `&insurance_network=${encodeURIComponent(insuranceFilter)}`;
      }

      const response = await fetch(url, { headers });

      if (response.ok) {
        const data = await response.json();
        setProviders(data.providers || []);
      }
    } catch (error) {
      console.error('Error loading providers:', error);
    } finally {
      setLoading(false);
    }
  }, [maxDistance, sortBy, insuranceFilter]);

  // Search medications
  const searchMedications = async () => {
    if (!medicationSearch.trim()) return;

    try {
      setLoading(true);
      const headers = await getAuthHeaders();
      const url = `${API_URL}/costs/medications?medication_name=${encodeURIComponent(medicationSearch)}&quantity=${medicationQuantity}`;

      const response = await fetch(url, { headers });

      if (response.ok) {
        const data = await response.json();
        if (data.medication_found) {
          setMedicationInfo(data.medication);
          setMedicationPrices(data.prices || []);
          setMedicationSuggestions([]);
          setExternalLinks(data.external_links || null);
          setPriceEstimate(data.price_estimate || null);
          setDataSource(data.data_source || null);
          setDisclaimer(data.disclaimer || '');
        } else {
          setMedicationInfo(null);
          setMedicationPrices([]);
          setMedicationSuggestions(data.suggestions || data.common_dermatology_drugs || []);
          setExternalLinks(data.external_links || null);
          setPriceEstimate(null);
          setDataSource(null);
          setDisclaimer('');
        }
      }
    } catch (error) {
      console.error('Error searching medications:', error);
    } finally {
      setLoading(false);
    }
  };

  // Calculate total cost
  const calculateTotalCost = async () => {
    try {
      setLoading(true);
      const headers = await getAuthHeaders();

      const response = await fetch(`${API_URL}/costs/calculate`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          procedures: selectedProcedures,
          medications: selectedMedications,
          insurance_type: insuranceType,
          deductible_remaining: deductibleRemaining,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setCalculatorResult(data);
      }
    } catch (error) {
      console.error('Error calculating costs:', error);
    } finally {
      setLoading(false);
    }
  };

  // Load initial data based on active tab
  useEffect(() => {
    if (activeTab === 'procedures') {
      loadProcedureEstimates();
    } else if (activeTab === 'providers') {
      loadProviders();
    }
  }, [activeTab, loadProcedureEstimates, loadProviders]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    if (activeTab === 'procedures') {
      await loadProcedureEstimates();
    } else if (activeTab === 'providers') {
      await loadProviders();
    }
    setRefreshing(false);
  }, [activeTab, loadProcedureEstimates, loadProviders]);

  // Render tabs
  const renderTabs = () => (
    <View style={styles.tabContainer}>
      {[
        { key: 'procedures', label: 'Procedures', icon: 'medical' },
        { key: 'providers', label: 'Providers', icon: 'people' },
        { key: 'medications', label: 'Rx Prices', icon: 'medkit' },
        { key: 'calculator', label: 'Calculator', icon: 'calculator' },
      ].map((tab) => (
        <TouchableOpacity
          key={tab.key}
          style={[styles.tab, activeTab === tab.key && styles.activeTab]}
          onPress={() => setActiveTab(tab.key as TabType)}
        >
          <Ionicons
            name={tab.icon as any}
            size={20}
            color={activeTab === tab.key ? '#2563eb' : '#6b7280'}
          />
          <Text style={[styles.tabText, activeTab === tab.key && styles.activeTabText]}>
            {tab.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  // Render procedure estimates
  const renderProceduresTab = () => (
    <View style={styles.tabContent}>
      <Text style={styles.sectionTitle}>Dermatology Procedure Costs</Text>
      <Text style={styles.sectionSubtitle}>Estimated costs based on national averages</Text>

      {loading ? (
        <ActivityIndicator size="large" color="#2563eb" style={styles.loader} />
      ) : (
        procedureEstimates.map((proc) => (
          <TouchableOpacity
            key={proc.procedure_key}
            style={styles.procedureCard}
            onPress={() => {
              setSelectedProcedure(proc);
              setShowProcedureModal(true);
            }}
          >
            <View style={styles.procedureHeader}>
              <View style={styles.procedureInfo}>
                <Text style={styles.procedureName}>{proc.procedure_name}</Text>
                <Text style={styles.procedureCode}>CPT: {proc.procedure_code}</Text>
              </View>
              <View style={styles.priceContainer}>
                <Text style={styles.priceLabel}>Avg. Cost</Text>
                <Text style={styles.priceValue}>${proc.average_cost.toFixed(0)}</Text>
              </View>
            </View>

            <Text style={styles.procedureDescription} numberOfLines={2}>
              {proc.description}
            </Text>

            <View style={styles.procedureFooter}>
              <View style={styles.costRange}>
                <Ionicons name="trending-up" size={14} color="#6b7280" />
                <Text style={styles.rangeText}>
                  Range: ${proc.cost_range_low} - ${proc.cost_range_high}
                </Text>
              </View>
              <View style={styles.outOfPocket}>
                <Text style={styles.oopLabel}>Est. Out-of-Pocket:</Text>
                <Text style={styles.oopValue}>${proc.estimated_out_of_pocket.toFixed(0)}</Text>
              </View>
            </View>

            <View style={styles.coverageBar}>
              <View
                style={[
                  styles.coverageFill,
                  { width: `${proc.typical_insurance_coverage * 100}%` }
                ]}
              />
              <Text style={styles.coverageText}>
                {(proc.typical_insurance_coverage * 100).toFixed(0)}% insurance coverage
              </Text>
            </View>
          </TouchableOpacity>
        ))
      )}
    </View>
  );

  // Render provider comparison
  const renderProvidersTab = () => (
    <View style={styles.tabContent}>
      <Text style={styles.sectionTitle}>Compare Dermatologist Prices</Text>

      {/* Filters */}
      <View style={styles.filterContainer}>
        <View style={styles.filterRow}>
          <Text style={styles.filterLabel}>Insurance:</Text>
          <TextInput
            style={styles.filterInput}
            placeholder="e.g., Blue Cross"
            value={insuranceFilter}
            onChangeText={setInsuranceFilter}
            onEndEditing={loadProviders}
          />
        </View>

        <View style={styles.filterRow}>
          <Text style={styles.filterLabel}>Max Distance:</Text>
          <View style={styles.distanceButtons}>
            {[10, 25, 50].map((dist) => (
              <TouchableOpacity
                key={dist}
                style={[styles.distanceBtn, maxDistance === dist && styles.distanceBtnActive]}
                onPress={() => setMaxDistance(dist)}
              >
                <Text style={[
                  styles.distanceBtnText,
                  maxDistance === dist && styles.distanceBtnTextActive
                ]}>
                  {dist} mi
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={styles.filterRow}>
          <Text style={styles.filterLabel}>Sort By:</Text>
          <View style={styles.sortButtons}>
            {[
              { key: 'price', label: 'Price' },
              { key: 'distance', label: 'Distance' },
              { key: 'rating', label: 'Rating' },
            ].map((sort) => (
              <TouchableOpacity
                key={sort.key}
                style={[styles.sortBtn, sortBy === sort.key && styles.sortBtnActive]}
                onPress={() => setSortBy(sort.key as any)}
              >
                <Text style={[
                  styles.sortBtnText,
                  sortBy === sort.key && styles.sortBtnTextActive
                ]}>
                  {sort.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </View>

      {loading ? (
        <ActivityIndicator size="large" color="#2563eb" style={styles.loader} />
      ) : providers.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="search" size={48} color="#d1d5db" />
          <Text style={styles.emptyText}>No providers found matching your criteria</Text>
        </View>
      ) : (
        providers.map((provider) => (
          <TouchableOpacity
            key={provider.provider_id}
            style={styles.providerCard}
            onPress={() => {
              setSelectedProvider(provider);
              setShowProviderModal(true);
            }}
          >
            <View style={styles.providerHeader}>
              <View style={styles.providerInfo}>
                <Text style={styles.providerName}>{provider.provider_name}</Text>
                <Text style={styles.providerSpecialty}>{provider.specialty}</Text>
              </View>
              <View style={styles.ratingContainer}>
                <Ionicons name="star" size={16} color="#f59e0b" />
                <Text style={styles.ratingText}>{provider.rating}</Text>
                <Text style={styles.reviewCount}>({provider.review_count})</Text>
              </View>
            </View>

            <View style={styles.providerDetails}>
              <View style={styles.detailItem}>
                <Ionicons name="location" size={14} color="#6b7280" />
                <Text style={styles.detailText}>
                  {provider.city} ({provider.distance_miles.toFixed(1)} mi)
                </Text>
              </View>
              <View style={styles.detailItem}>
                <Ionicons name="time" size={14} color="#6b7280" />
                <Text style={styles.detailText}>
                  {provider.wait_time_days} day wait
                </Text>
              </View>
            </View>

            <View style={styles.providerPricing}>
              <View style={styles.priceItem}>
                <Text style={styles.priceItemLabel}>Consultation</Text>
                <Text style={styles.priceItemValue}>${provider.consultation_fee}</Text>
              </View>
              {provider.telemedicine_available && (
                <View style={styles.priceItem}>
                  <Text style={styles.priceItemLabel}>Telemedicine</Text>
                  <Text style={styles.priceItemValue}>${provider.telemedicine_fee}</Text>
                </View>
              )}
            </View>

            <View style={styles.providerTags}>
              {provider.accepts_insurance && (
                <View style={styles.tag}>
                  <Ionicons name="checkmark-circle" size={12} color="#10b981" />
                  <Text style={styles.tagText}>Insurance Accepted</Text>
                </View>
              )}
              {provider.telemedicine_available && (
                <View style={[styles.tag, styles.tagBlue]}>
                  <Ionicons name="videocam" size={12} color="#2563eb" />
                  <Text style={[styles.tagText, styles.tagTextBlue]}>Telemedicine</Text>
                </View>
              )}
            </View>
          </TouchableOpacity>
        ))
      )}
    </View>
  );

  // Open external link
  const openExternalLink = (url: string) => {
    Linking.openURL(url).catch((err) => {
      console.error('Failed to open URL:', err);
      Alert.alert('Error', 'Unable to open link');
    });
  };

  // Render medication prices (GoodRx-style with OpenFDA integration)
  const renderMedicationsTab = () => (
    <View style={styles.tabContent}>
      <Text style={styles.sectionTitle}>Medication Price Comparison</Text>
      <Text style={styles.sectionSubtitle}>Real drug data from FDA with estimated prices</Text>

      {/* Search */}
      <View style={styles.searchContainer}>
        <View style={styles.searchInputContainer}>
          <Ionicons name="search" size={20} color="#6b7280" style={styles.searchIcon} />
          <TextInput
            style={styles.searchInput}
            placeholder="Search any FDA-approved medication"
            value={medicationSearch}
            onChangeText={setMedicationSearch}
            onSubmitEditing={searchMedications}
          />
        </View>

        <View style={styles.quantityContainer}>
          <Text style={styles.quantityLabel}>Quantity:</Text>
          <View style={styles.quantityButtons}>
            {[30, 60, 90].map((qty) => (
              <TouchableOpacity
                key={qty}
                style={[styles.qtyBtn, medicationQuantity === qty && styles.qtyBtnActive]}
                onPress={() => setMedicationQuantity(qty)}
              >
                <Text style={[
                  styles.qtyBtnText,
                  medicationQuantity === qty && styles.qtyBtnTextActive
                ]}>
                  {qty}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <TouchableOpacity style={styles.searchButton} onPress={searchMedications}>
          <Text style={styles.searchButtonText}>Search Prices</Text>
        </TouchableOpacity>
      </View>

      {/* Suggestions */}
      {medicationSuggestions.length > 0 && (
        <View style={styles.suggestionsContainer}>
          <Text style={styles.suggestionsTitle}>Try searching for:</Text>
          <View style={styles.suggestionsList}>
            {medicationSuggestions.slice(0, 8).map((suggestion) => (
              <TouchableOpacity
                key={suggestion}
                style={styles.suggestionChip}
                onPress={() => {
                  setMedicationSearch(suggestion);
                  setTimeout(searchMedications, 100);
                }}
              >
                <Text style={styles.suggestionText}>{suggestion}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      )}

      {loading ? (
        <ActivityIndicator size="large" color="#2563eb" style={styles.loader} />
      ) : medicationInfo ? (
        <>
          {/* Data Source Badge */}
          <View style={styles.dataSourceBadge}>
            <Ionicons
              name={dataSource === 'openfda' ? 'checkmark-circle' : 'information-circle'}
              size={14}
              color={dataSource === 'openfda' ? '#059669' : '#2563eb'}
            />
            <Text style={styles.dataSourceText}>
              {dataSource === 'openfda' ? 'FDA Verified Data' : 'Curated Data'}
            </Text>
          </View>

          {/* Medication Info */}
          <View style={styles.medicationInfoCard}>
            <Text style={styles.medicationBrand}>{medicationInfo.brand_name}</Text>
            <Text style={styles.medicationGeneric}>{medicationInfo.generic_name}</Text>

            <View style={styles.drugClassBadge}>
              <Text style={styles.drugClassText}>{medicationInfo.drug_class}</Text>
            </View>

            {medicationInfo.manufacturer && (
              <Text style={styles.manufacturerText}>
                Manufacturer: {medicationInfo.manufacturer}
              </Text>
            )}

            <View style={styles.medicationDetails}>
              <Text style={styles.medicationDetail}>
                Dosages: {medicationInfo.available_dosages.join(', ')}
              </Text>
              <Text style={styles.medicationDetail}>
                Forms: {medicationInfo.available_forms.join(', ')}
              </Text>
              <Text style={styles.medicationDetail}>
                Route: {medicationInfo.route || 'Topical'}
              </Text>
              <Text style={styles.medicationDetail}>
                Quantity: {medicationInfo.quantity} count
              </Text>
            </View>

            {medicationInfo.indications && (
              <View style={styles.indicationsContainer}>
                <Text style={styles.indicationsTitle}>Indications:</Text>
                <Text style={styles.indicationsText} numberOfLines={3}>
                  {medicationInfo.indications}
                </Text>
              </View>
            )}
          </View>

          {/* Price Estimate Range */}
          {priceEstimate && (
            <View style={styles.priceEstimateCard}>
              <Text style={styles.priceEstimateTitle}>Estimated Price Range</Text>
              <View style={styles.priceRangeRow}>
                <View style={styles.priceRangeItem}>
                  <Text style={styles.priceRangeLabel}>Low</Text>
                  <Text style={styles.priceRangeLow}>${priceEstimate.estimated_low}</Text>
                </View>
                <View style={styles.priceRangeItem}>
                  <Text style={styles.priceRangeLabel}>Typical</Text>
                  <Text style={styles.priceRangeMid}>${priceEstimate.estimated_mid}</Text>
                </View>
                <View style={styles.priceRangeItem}>
                  <Text style={styles.priceRangeLabel}>High</Text>
                  <Text style={styles.priceRangeHigh}>${priceEstimate.estimated_high}</Text>
                </View>
              </View>
              <Text style={styles.priceRangeUnit}>Per {priceEstimate.unit}</Text>
            </View>
          )}

          {/* External Links - Get Exact Prices */}
          {externalLinks && (
            <View style={styles.externalLinksCard}>
              <Text style={styles.externalLinksTitle}>Get Exact Local Prices</Text>
              <Text style={styles.externalLinksSubtitle}>
                Click below for real-time prices at pharmacies near you
              </Text>

              <View style={styles.externalLinksList}>
                <TouchableOpacity
                  style={[styles.externalLinkBtn, styles.goodrxBtn]}
                  onPress={() => openExternalLink(externalLinks.goodrx)}
                >
                  <Text style={styles.externalLinkBtnText}>GoodRx</Text>
                  <Ionicons name="open-outline" size={14} color="#fff" />
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.externalLinkBtn, styles.rxsaverBtn]}
                  onPress={() => openExternalLink(externalLinks.rxsaver)}
                >
                  <Text style={styles.externalLinkBtnText}>RxSaver</Text>
                  <Ionicons name="open-outline" size={14} color="#fff" />
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.externalLinkBtn, styles.singlecareBtn]}
                  onPress={() => openExternalLink(externalLinks.singlecare)}
                >
                  <Text style={styles.externalLinkBtnText}>SingleCare</Text>
                  <Ionicons name="open-outline" size={14} color="#fff" />
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.externalLinkBtn, styles.amazonBtn]}
                  onPress={() => openExternalLink(externalLinks.amazon_pharmacy)}
                >
                  <Text style={styles.externalLinkBtnText}>Amazon Pharmacy</Text>
                  <Ionicons name="open-outline" size={14} color="#fff" />
                </TouchableOpacity>
              </View>
            </View>
          )}

          {/* Pharmacy Prices */}
          <Text style={styles.pricesTitle}>
            Estimated Pharmacy Prices
            {medicationPrices[0]?.estimated && (
              <Text style={styles.estimatedLabel}> (Estimates)</Text>
            )}
          </Text>

          {medicationPrices.map((price, index) => (
            <View
              key={`${price.pharmacy_name}-${index}`}
              style={[styles.pharmacyCard, index === 0 && styles.lowestPriceCard]}
            >
              {index === 0 && (
                <View style={styles.lowestPriceBadge}>
                  <Text style={styles.lowestPriceBadgeText}>LOWEST PRICE</Text>
                </View>
              )}

              <View style={styles.pharmacyHeader}>
                <View>
                  <Text style={styles.pharmacyName}>{price.pharmacy_name}</Text>
                  <Text style={styles.pharmacyAddress}>{price.pharmacy_address}</Text>
                </View>
                <View style={styles.pharmacyPricing}>
                  <Text style={styles.pharmacyPrice}>${price.price.toFixed(2)}</Text>
                  <Text style={styles.pharmacyOriginal}>
                    <Text style={styles.strikethrough}>${price.original_price.toFixed(2)}</Text>
                  </Text>
                </View>
              </View>

              <View style={styles.savingsRow}>
                <View style={styles.savingsBadge}>
                  <Text style={styles.savingsText}>
                    Save ${price.savings.toFixed(2)} ({price.savings_percent.toFixed(0)}%)
                  </Text>
                </View>
                {price.coupon_code && (
                  <View style={styles.couponBadge}>
                    <Ionicons name="pricetag" size={12} color="#7c3aed" />
                    <Text style={styles.couponText}>{price.coupon_code}</Text>
                  </View>
                )}
                {price.requires_membership && (
                  <View style={styles.membershipBadge}>
                    <Text style={styles.membershipText}>Membership Required</Text>
                  </View>
                )}
                {price.estimated && (
                  <View style={styles.estimatedBadge}>
                    <Text style={styles.estimatedBadgeText}>Est.</Text>
                  </View>
                )}
              </View>
            </View>
          ))}

          {/* Disclaimer */}
          {disclaimer && (
            <View style={styles.disclaimerCard}>
              <Ionicons name="information-circle" size={16} color="#6b7280" />
              <Text style={styles.disclaimerText}>{disclaimer}</Text>
            </View>
          )}
        </>
      ) : (
        <View style={styles.emptyState}>
          <Ionicons name="medkit" size={48} color="#d1d5db" />
          <Text style={styles.emptyText}>
            Search for any FDA-approved medication
          </Text>
          <Text style={styles.emptySubtext}>
            Examples: tretinoin, hydrocortisone, doxycycline, clobetasol, acyclovir
          </Text>
          <Text style={styles.emptySubtext}>
            Data from FDA with links to GoodRx, RxSaver & more
          </Text>
        </View>
      )}
    </View>
  );

  // Render cost calculator
  const renderCalculatorTab = () => (
    <View style={styles.tabContent}>
      <Text style={styles.sectionTitle}>Treatment Cost Calculator</Text>
      <Text style={styles.sectionSubtitle}>
        Estimate your total out-of-pocket costs
      </Text>

      {/* Procedure Selection */}
      <View style={styles.calcSection}>
        <Text style={styles.calcSectionTitle}>Select Procedures</Text>
        <View style={styles.checkboxList}>
          {[
            { key: 'consultation', label: 'Office Visit' },
            { key: 'skin_biopsy', label: 'Skin Biopsy' },
            { key: 'dermoscopy', label: 'Dermoscopy' },
            { key: 'excision_benign', label: 'Benign Excision' },
            { key: 'cryotherapy', label: 'Cryotherapy' },
          ].map((proc) => (
            <TouchableOpacity
              key={proc.key}
              style={styles.checkboxItem}
              onPress={() => {
                if (selectedProcedures.includes(proc.key)) {
                  setSelectedProcedures(prev => prev.filter(p => p !== proc.key));
                } else {
                  setSelectedProcedures(prev => [...prev, proc.key]);
                }
              }}
            >
              <View style={[
                styles.checkbox,
                selectedProcedures.includes(proc.key) && styles.checkboxChecked
              ]}>
                {selectedProcedures.includes(proc.key) && (
                  <Ionicons name="checkmark" size={14} color="#fff" />
                )}
              </View>
              <Text style={styles.checkboxLabel}>{proc.label}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Medication Selection */}
      <View style={styles.calcSection}>
        <Text style={styles.calcSectionTitle}>Select Medications</Text>
        <View style={styles.checkboxList}>
          {[
            { key: 'tretinoin', label: 'Tretinoin (Retin-A)' },
            { key: 'hydrocortisone', label: 'Hydrocortisone' },
            { key: 'doxycycline', label: 'Doxycycline' },
          ].map((med) => (
            <TouchableOpacity
              key={med.key}
              style={styles.checkboxItem}
              onPress={() => {
                if (selectedMedications.includes(med.key)) {
                  setSelectedMedications(prev => prev.filter(m => m !== med.key));
                } else {
                  setSelectedMedications(prev => [...prev, med.key]);
                }
              }}
            >
              <View style={[
                styles.checkbox,
                selectedMedications.includes(med.key) && styles.checkboxChecked
              ]}>
                {selectedMedications.includes(med.key) && (
                  <Ionicons name="checkmark" size={14} color="#fff" />
                )}
              </View>
              <Text style={styles.checkboxLabel}>{med.label}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Insurance Type */}
      <View style={styles.calcSection}>
        <Text style={styles.calcSectionTitle}>Insurance Type</Text>
        <View style={styles.insuranceButtons}>
          {[
            { key: 'private', label: 'Private' },
            { key: 'medicare', label: 'Medicare' },
            { key: 'medicaid', label: 'Medicaid' },
            { key: 'none', label: 'Uninsured' },
          ].map((ins) => (
            <TouchableOpacity
              key={ins.key}
              style={[styles.insBtn, insuranceType === ins.key && styles.insBtnActive]}
              onPress={() => setInsuranceType(ins.key)}
            >
              <Text style={[
                styles.insBtnText,
                insuranceType === ins.key && styles.insBtnTextActive
              ]}>
                {ins.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Deductible */}
      <View style={styles.calcSection}>
        <Text style={styles.calcSectionTitle}>Deductible Remaining</Text>
        <TextInput
          style={styles.deductibleInput}
          placeholder="$0"
          keyboardType="numeric"
          value={deductibleRemaining > 0 ? `$${deductibleRemaining}` : ''}
          onChangeText={(text) => {
            const num = parseInt(text.replace(/[^0-9]/g, '')) || 0;
            setDeductibleRemaining(num);
          }}
        />
      </View>

      {/* Calculate Button */}
      <TouchableOpacity
        style={styles.calculateButton}
        onPress={calculateTotalCost}
      >
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="calculator" size={20} color="#fff" />
            <Text style={styles.calculateButtonText}>Calculate Total Cost</Text>
          </>
        )}
      </TouchableOpacity>

      {/* Results */}
      {calculatorResult && (
        <View style={styles.resultsCard}>
          <Text style={styles.resultsTitle}>Cost Estimate</Text>

          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Procedure Costs:</Text>
            <Text style={styles.resultValue}>
              ${calculatorResult.summary.total_procedure_cost.toFixed(2)}
            </Text>
          </View>

          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Medication Costs:</Text>
            <Text style={styles.resultValue}>
              ${calculatorResult.summary.total_medication_cost.toFixed(2)}
            </Text>
          </View>

          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Insurance Pays:</Text>
            <Text style={[styles.resultValue, styles.greenText]}>
              -${calculatorResult.summary.insurance_pays.toFixed(2)}
            </Text>
          </View>

          <View style={styles.divider} />

          <View style={styles.resultRow}>
            <Text style={styles.totalLabel}>Your Estimated Total:</Text>
            <Text style={styles.totalValue}>
              ${calculatorResult.summary.your_estimated_total.toFixed(2)}
            </Text>
          </View>

          <View style={styles.paymentOptions}>
            <Text style={styles.paymentTitle}>Payment Options:</Text>
            {calculatorResult.payment_options.map((option: string, index: number) => (
              <View key={index} style={styles.paymentOption}>
                <Ionicons name="checkmark-circle" size={14} color="#10b981" />
                <Text style={styles.paymentText}>{option}</Text>
              </View>
            ))}
          </View>

          <Text style={styles.disclaimer}>{calculatorResult.disclaimer}</Text>
        </View>
      )}
    </View>
  );

  // Procedure Detail Modal
  const renderProcedureModal = () => (
    <Modal
      visible={showProcedureModal}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={() => setShowProcedureModal(false)}
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <Text style={styles.modalTitle}>Procedure Details</Text>
          <TouchableOpacity onPress={() => setShowProcedureModal(false)}>
            <Ionicons name="close" size={24} color="#1f2937" />
          </TouchableOpacity>
        </View>

        {selectedProcedure && (
          <ScrollView style={styles.modalContent}>
            <Text style={styles.modalProcedureName}>{selectedProcedure.procedure_name}</Text>
            <Text style={styles.modalProcedureCode}>CPT Code: {selectedProcedure.procedure_code}</Text>

            <Text style={styles.modalDescription}>{selectedProcedure.description}</Text>

            <View style={styles.costBreakdown}>
              <Text style={styles.breakdownTitle}>Cost Breakdown</Text>

              <View style={styles.breakdownRow}>
                <Text style={styles.breakdownLabel}>Average Cost:</Text>
                <Text style={styles.breakdownValue}>${selectedProcedure.average_cost.toFixed(2)}</Text>
              </View>

              <View style={styles.breakdownRow}>
                <Text style={styles.breakdownLabel}>Cost Range:</Text>
                <Text style={styles.breakdownValue}>
                  ${selectedProcedure.cost_range_low} - ${selectedProcedure.cost_range_high}
                </Text>
              </View>

              <View style={styles.breakdownRow}>
                <Text style={styles.breakdownLabel}>Medicare Rate:</Text>
                <Text style={styles.breakdownValue}>${selectedProcedure.medicare_rate.toFixed(2)}</Text>
              </View>

              <View style={styles.breakdownRow}>
                <Text style={styles.breakdownLabel}>Insurance Coverage:</Text>
                <Text style={styles.breakdownValue}>
                  {(selectedProcedure.typical_insurance_coverage * 100).toFixed(0)}%
                </Text>
              </View>

              <View style={[styles.breakdownRow, styles.highlightRow]}>
                <Text style={styles.breakdownLabelBold}>Est. Out-of-Pocket:</Text>
                <Text style={styles.breakdownValueBold}>
                  ${selectedProcedure.estimated_out_of_pocket.toFixed(2)}
                </Text>
              </View>
            </View>

            <View style={styles.factorsSection}>
              <Text style={styles.factorsTitle}>Factors Affecting Cost</Text>
              {selectedProcedure.factors_affecting_cost.map((factor, index) => (
                <View key={index} style={styles.factorItem}>
                  <Ionicons name="information-circle" size={16} color="#6b7280" />
                  <Text style={styles.factorText}>{factor}</Text>
                </View>
              ))}
            </View>
          </ScrollView>
        )}
      </View>
    </Modal>
  );

  // Provider Detail Modal
  const renderProviderModal = () => (
    <Modal
      visible={showProviderModal}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={() => setShowProviderModal(false)}
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <Text style={styles.modalTitle}>Provider Details</Text>
          <TouchableOpacity onPress={() => setShowProviderModal(false)}>
            <Ionicons name="close" size={24} color="#1f2937" />
          </TouchableOpacity>
        </View>

        {selectedProvider && (
          <ScrollView style={styles.modalContent}>
            <View style={styles.providerModalHeader}>
              <Text style={styles.providerModalName}>{selectedProvider.provider_name}</Text>
              <Text style={styles.providerModalSpecialty}>{selectedProvider.specialty}</Text>

              <View style={styles.providerModalRating}>
                <Ionicons name="star" size={20} color="#f59e0b" />
                <Text style={styles.providerModalRatingText}>{selectedProvider.rating}</Text>
                <Text style={styles.providerModalReviews}>
                  ({selectedProvider.review_count} reviews)
                </Text>
              </View>
            </View>

            <View style={styles.providerModalInfo}>
              <View style={styles.infoRow}>
                <Ionicons name="location" size={18} color="#6b7280" />
                <View>
                  <Text style={styles.infoText}>{selectedProvider.location}</Text>
                  <Text style={styles.infoSubtext}>
                    {selectedProvider.city} - {selectedProvider.distance_miles.toFixed(1)} miles away
                  </Text>
                </View>
              </View>

              <View style={styles.infoRow}>
                <Ionicons name="time" size={18} color="#6b7280" />
                <Text style={styles.infoText}>
                  Average wait time: {selectedProvider.wait_time_days} days
                </Text>
              </View>
            </View>

            <View style={styles.pricingSection}>
              <Text style={styles.pricingSectionTitle}>Pricing</Text>

              <View style={styles.pricingRow}>
                <Text style={styles.pricingLabel}>Office Consultation:</Text>
                <Text style={styles.pricingValue}>${selectedProvider.consultation_fee}</Text>
              </View>

              {selectedProvider.telemedicine_available && (
                <View style={styles.pricingRow}>
                  <Text style={styles.pricingLabel}>Telemedicine Visit:</Text>
                  <Text style={styles.pricingValue}>${selectedProvider.telemedicine_fee}</Text>
                </View>
              )}

              <View style={styles.pricingRow}>
                <Text style={styles.pricingLabel}>Avg. Procedure Cost:</Text>
                <Text style={styles.pricingValue}>${selectedProvider.average_procedure_cost}</Text>
              </View>
            </View>

            <View style={styles.insuranceSection}>
              <Text style={styles.insuranceSectionTitle}>Accepted Insurance</Text>
              <View style={styles.insuranceList}>
                {selectedProvider.insurance_networks.map((network, index) => (
                  <View key={index} style={styles.insuranceChip}>
                    <Text style={styles.insuranceChipText}>{network}</Text>
                  </View>
                ))}
              </View>
            </View>

            <View style={styles.specialSection}>
              <Text style={styles.specialSectionTitle}>Specializations</Text>
              <View style={styles.specialList}>
                {selectedProvider.specializations.map((spec, index) => (
                  <View key={index} style={styles.specialChip}>
                    <Ionicons name="medical" size={12} color="#2563eb" />
                    <Text style={styles.specialChipText}>{spec}</Text>
                  </View>
                ))}
              </View>
            </View>

            <TouchableOpacity
              style={styles.bookButton}
              onPress={() => {
                setShowProviderModal(false);
                router.push('/appointments');
              }}
            >
              <Ionicons name="calendar" size={20} color="#fff" />
              <Text style={styles.bookButtonText}>Book Appointment</Text>
            </TouchableOpacity>
          </ScrollView>
        )}
      </View>
    </Modal>
  );

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#1f2937" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Cost Transparency</Text>
        <View style={{ width: 40 }} />
      </View>

      {renderTabs()}

      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {activeTab === 'procedures' && renderProceduresTab()}
        {activeTab === 'providers' && renderProvidersTab()}
        {activeTab === 'medications' && renderMedicationsTab()}
        {activeTab === 'calculator' && renderCalculatorTab()}
      </ScrollView>

      {renderProcedureModal()}
      {renderProviderModal()}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 60 : 20,
    paddingBottom: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
  },
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingVertical: 8,
    paddingHorizontal: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 8,
    borderRadius: 8,
  },
  activeTab: {
    backgroundColor: '#eff6ff',
  },
  tabText: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 4,
  },
  activeTabText: {
    color: '#2563eb',
    fontWeight: '600',
  },
  scrollView: {
    flex: 1,
  },
  tabContent: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 4,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 16,
  },
  loader: {
    marginTop: 40,
  },

  // Procedure Cards
  procedureCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  procedureHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  procedureInfo: {
    flex: 1,
  },
  procedureName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  procedureCode: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  priceContainer: {
    alignItems: 'flex-end',
  },
  priceLabel: {
    fontSize: 11,
    color: '#6b7280',
  },
  priceValue: {
    fontSize: 20,
    fontWeight: '700',
    color: '#2563eb',
  },
  procedureDescription: {
    fontSize: 13,
    color: '#4b5563',
    lineHeight: 18,
    marginBottom: 12,
  },
  procedureFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  costRange: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  rangeText: {
    fontSize: 12,
    color: '#6b7280',
    marginLeft: 4,
  },
  outOfPocket: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  oopLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginRight: 4,
  },
  oopValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#059669',
  },
  coverageBar: {
    height: 20,
    backgroundColor: '#e5e7eb',
    borderRadius: 10,
    overflow: 'hidden',
    position: 'relative',
  },
  coverageFill: {
    position: 'absolute',
    top: 0,
    left: 0,
    bottom: 0,
    backgroundColor: '#10b981',
    borderRadius: 10,
  },
  coverageText: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    textAlign: 'center',
    lineHeight: 20,
    fontSize: 11,
    fontWeight: '500',
    color: '#1f2937',
  },

  // Provider Cards
  filterContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  filterRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  filterLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
    width: 100,
  },
  filterInput: {
    flex: 1,
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 14,
  },
  distanceButtons: {
    flexDirection: 'row',
    flex: 1,
  },
  distanceBtn: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
    marginRight: 8,
  },
  distanceBtnActive: {
    backgroundColor: '#2563eb',
  },
  distanceBtnText: {
    fontSize: 14,
    color: '#4b5563',
  },
  distanceBtnTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  sortButtons: {
    flexDirection: 'row',
    flex: 1,
  },
  sortBtn: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
    marginRight: 8,
  },
  sortBtnActive: {
    backgroundColor: '#2563eb',
  },
  sortBtnText: {
    fontSize: 13,
    color: '#4b5563',
  },
  sortBtnTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  providerCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  providerHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  providerInfo: {
    flex: 1,
  },
  providerName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  providerSpecialty: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  ratingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  ratingText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginLeft: 4,
  },
  reviewCount: {
    fontSize: 12,
    color: '#6b7280',
    marginLeft: 2,
  },
  providerDetails: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  detailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 16,
  },
  detailText: {
    fontSize: 13,
    color: '#4b5563',
    marginLeft: 4,
  },
  providerPricing: {
    flexDirection: 'row',
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  priceItem: {
    flex: 1,
    alignItems: 'center',
  },
  priceItemLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
  priceItemValue: {
    fontSize: 18,
    fontWeight: '700',
    color: '#2563eb',
    marginTop: 2,
  },
  providerTags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  tag: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ecfdf5',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    marginRight: 8,
  },
  tagBlue: {
    backgroundColor: '#eff6ff',
  },
  tagText: {
    fontSize: 12,
    color: '#059669',
    marginLeft: 4,
  },
  tagTextBlue: {
    color: '#2563eb',
  },

  // Medications
  searchContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  searchInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    paddingHorizontal: 12,
  },
  searchIcon: {
    marginRight: 8,
  },
  searchInput: {
    flex: 1,
    paddingVertical: 12,
    fontSize: 14,
  },
  quantityContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
  },
  quantityLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
    marginRight: 12,
  },
  quantityButtons: {
    flexDirection: 'row',
  },
  qtyBtn: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
    marginRight: 8,
  },
  qtyBtnActive: {
    backgroundColor: '#2563eb',
  },
  qtyBtnText: {
    fontSize: 14,
    color: '#4b5563',
  },
  qtyBtnTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  searchButton: {
    backgroundColor: '#2563eb',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 12,
  },
  searchButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  suggestionsContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  suggestionsTitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 8,
  },
  suggestionsList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  suggestionChip: {
    backgroundColor: '#eff6ff',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
    marginBottom: 8,
  },
  suggestionText: {
    fontSize: 14,
    color: '#2563eb',
  },
  medicationInfoCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  medicationBrand: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1f2937',
  },
  medicationGeneric: {
    fontSize: 16,
    color: '#4b5563',
    marginTop: 2,
  },
  medicationClass: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 4,
  },
  medicationDetails: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  medicationDetail: {
    fontSize: 13,
    color: '#4b5563',
    marginBottom: 4,
  },
  pricesTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  pharmacyCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  lowestPriceCard: {
    borderWidth: 2,
    borderColor: '#10b981',
  },
  lowestPriceBadge: {
    position: 'absolute',
    top: -10,
    left: 12,
    backgroundColor: '#10b981',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  lowestPriceBadgeText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#fff',
  },
  pharmacyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  pharmacyName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  pharmacyAddress: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  pharmacyPricing: {
    alignItems: 'flex-end',
  },
  pharmacyPrice: {
    fontSize: 24,
    fontWeight: '700',
    color: '#10b981',
  },
  pharmacyOriginal: {
    fontSize: 14,
    color: '#9ca3af',
  },
  strikethrough: {
    textDecorationLine: 'line-through',
  },
  savingsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
  },
  savingsBadge: {
    backgroundColor: '#ecfdf5',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    marginRight: 8,
  },
  savingsText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#059669',
  },
  couponBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f5f3ff',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    marginRight: 8,
  },
  couponText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#7c3aed',
    marginLeft: 4,
  },
  membershipBadge: {
    backgroundColor: '#fef3c7',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  membershipText: {
    fontSize: 11,
    color: '#92400e',
  },

  // New OpenFDA/External Links Styles
  dataSourceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0fdf4',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    alignSelf: 'flex-start',
    marginBottom: 12,
  },
  dataSourceText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#059669',
    marginLeft: 6,
  },
  drugClassBadge: {
    backgroundColor: '#eff6ff',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    alignSelf: 'flex-start',
    marginTop: 8,
  },
  drugClassText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#2563eb',
  },
  manufacturerText: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 8,
  },
  indicationsContainer: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  indicationsTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 4,
  },
  indicationsText: {
    fontSize: 13,
    color: '#4b5563',
    lineHeight: 18,
  },
  priceEstimateCard: {
    backgroundColor: '#fef3c7',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  priceEstimateTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#92400e',
    marginBottom: 12,
    textAlign: 'center',
  },
  priceRangeRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  priceRangeItem: {
    alignItems: 'center',
  },
  priceRangeLabel: {
    fontSize: 11,
    color: '#78716c',
    marginBottom: 4,
  },
  priceRangeLow: {
    fontSize: 18,
    fontWeight: '700',
    color: '#059669',
  },
  priceRangeMid: {
    fontSize: 18,
    fontWeight: '700',
    color: '#d97706',
  },
  priceRangeHigh: {
    fontSize: 18,
    fontWeight: '700',
    color: '#dc2626',
  },
  priceRangeUnit: {
    fontSize: 11,
    color: '#78716c',
    textAlign: 'center',
    marginTop: 8,
  },
  externalLinksCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 2,
    borderColor: '#2563eb',
  },
  externalLinksTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1f2937',
    textAlign: 'center',
  },
  externalLinksSubtitle: {
    fontSize: 13,
    color: '#6b7280',
    textAlign: 'center',
    marginTop: 4,
    marginBottom: 16,
  },
  externalLinksList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 8,
  },
  externalLinkBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    marginHorizontal: 4,
    marginVertical: 4,
  },
  externalLinkBtnText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
    marginRight: 6,
  },
  goodrxBtn: {
    backgroundColor: '#f59e0b',
  },
  rxsaverBtn: {
    backgroundColor: '#10b981',
  },
  singlecareBtn: {
    backgroundColor: '#8b5cf6',
  },
  amazonBtn: {
    backgroundColor: '#1f2937',
  },
  estimatedLabel: {
    fontSize: 12,
    fontWeight: '400',
    color: '#9ca3af',
  },
  estimatedBadge: {
    backgroundColor: '#f3f4f6',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  estimatedBadgeText: {
    fontSize: 10,
    color: '#6b7280',
    fontWeight: '500',
  },
  disclaimerCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 12,
    marginTop: 16,
    marginBottom: 24,
  },
  disclaimerText: {
    fontSize: 12,
    color: '#6b7280',
    marginLeft: 8,
    flex: 1,
    lineHeight: 16,
  },

  // Calculator
  calcSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  calcSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  checkboxList: {

  },
  checkboxItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 4,
    borderWidth: 2,
    borderColor: '#d1d5db',
    marginRight: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxChecked: {
    backgroundColor: '#2563eb',
    borderColor: '#2563eb',
  },
  checkboxLabel: {
    fontSize: 14,
    color: '#374151',
  },
  insuranceButtons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  insBtn: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
    marginRight: 8,
    marginBottom: 8,
  },
  insBtnActive: {
    backgroundColor: '#2563eb',
  },
  insBtnText: {
    fontSize: 14,
    color: '#4b5563',
  },
  insBtnTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  deductibleInput: {
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
  },
  calculateButton: {
    backgroundColor: '#2563eb',
    borderRadius: 12,
    paddingVertical: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  calculateButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  resultsCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 16,
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  resultLabel: {
    fontSize: 14,
    color: '#4b5563',
  },
  resultValue: {
    fontSize: 14,
    fontWeight: '500',
    color: '#1f2937',
  },
  greenText: {
    color: '#059669',
  },
  divider: {
    height: 1,
    backgroundColor: '#e5e7eb',
    marginVertical: 12,
  },
  totalLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  totalValue: {
    fontSize: 20,
    fontWeight: '700',
    color: '#2563eb',
  },
  paymentOptions: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  paymentTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  paymentOption: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  paymentText: {
    fontSize: 13,
    color: '#4b5563',
    marginLeft: 6,
  },
  disclaimer: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 16,
    fontStyle: 'italic',
  },

  // Empty State
  emptyState: {
    alignItems: 'center',
    paddingVertical: 48,
  },
  emptyText: {
    fontSize: 16,
    color: '#6b7280',
    marginTop: 16,
    textAlign: 'center',
  },
  emptySubtext: {
    fontSize: 14,
    color: '#9ca3af',
    marginTop: 8,
    textAlign: 'center',
  },

  // Modal
  modalContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    paddingTop: Platform.OS === 'ios' ? 60 : 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
  },
  modalContent: {
    flex: 1,
    padding: 16,
  },
  modalProcedureName: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1f2937',
  },
  modalProcedureCode: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 4,
    marginBottom: 16,
  },
  modalDescription: {
    fontSize: 15,
    color: '#4b5563',
    lineHeight: 22,
    marginBottom: 24,
  },
  costBreakdown: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
  },
  breakdownTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  breakdownRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  breakdownLabel: {
    fontSize: 14,
    color: '#4b5563',
  },
  breakdownValue: {
    fontSize: 14,
    fontWeight: '500',
    color: '#1f2937',
  },
  highlightRow: {
    backgroundColor: '#ecfdf5',
    marginHorizontal: -16,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginTop: 8,
    marginBottom: 0,
    borderRadius: 8,
  },
  breakdownLabelBold: {
    fontSize: 14,
    fontWeight: '600',
    color: '#059669',
  },
  breakdownValueBold: {
    fontSize: 16,
    fontWeight: '700',
    color: '#059669',
  },
  factorsSection: {
    marginBottom: 24,
  },
  factorsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  factorItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  factorText: {
    fontSize: 14,
    color: '#4b5563',
    marginLeft: 8,
  },

  // Provider Modal
  providerModalHeader: {
    marginBottom: 24,
  },
  providerModalName: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1f2937',
  },
  providerModalSpecialty: {
    fontSize: 16,
    color: '#4b5563',
    marginTop: 4,
  },
  providerModalRating: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
  },
  providerModalRatingText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginLeft: 6,
  },
  providerModalReviews: {
    fontSize: 14,
    color: '#6b7280',
    marginLeft: 4,
  },
  providerModalInfo: {
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  infoText: {
    fontSize: 14,
    color: '#1f2937',
    marginLeft: 12,
  },
  infoSubtext: {
    fontSize: 13,
    color: '#6b7280',
    marginLeft: 12,
  },
  pricingSection: {
    marginBottom: 24,
  },
  pricingSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  pricingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  pricingLabel: {
    fontSize: 14,
    color: '#4b5563',
  },
  pricingValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2563eb',
  },
  insuranceSection: {
    marginBottom: 24,
  },
  insuranceSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  insuranceList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  insuranceChip: {
    backgroundColor: '#ecfdf5',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
    marginBottom: 8,
  },
  insuranceChipText: {
    fontSize: 13,
    color: '#059669',
  },
  specialSection: {
    marginBottom: 24,
  },
  specialSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  specialList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  specialChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#eff6ff',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
    marginBottom: 8,
  },
  specialChipText: {
    fontSize: 13,
    color: '#2563eb',
    marginLeft: 4,
  },
  bookButton: {
    backgroundColor: '#2563eb',
    borderRadius: 12,
    paddingVertical: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 8,
    marginBottom: 32,
  },
  bookButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
});
