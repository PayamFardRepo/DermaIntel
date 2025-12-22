export interface BinaryClassificationResponse {
  probabilities: {
    non_lesion: number;
    lesion: number;
  };
  predicted_class: string;
  binary_pred: number;
  'confidence:': number;
  confidence_boolean: boolean;
}

export interface FullClassificationResponse {
  key_map: Record<string, string>;
  filename: string;
  probabilities: Record<string, number>;
  predicted_class: string;
  lesion_confidence: number;
}

export interface ImagePickerAsset {
  uri: string;
  width?: number;
  height?: number;
  type?: string;
}

export interface BurnClassificationResponse {
  severity_class: string;
  severity_level: number;
  confidence: number;
  probabilities: Record<string, number>;
  urgency: string;
  treatment_advice: string;
  medical_attention_required: boolean;
  is_burn_detected: boolean;
}

export interface DermoscopyResponse {
  pigment_network: {
    detected: boolean;
    type: string;
    regularity_score: number;
    risk_level: string;
    description: string;
    contour_count: number;
    coordinates: any[];
  };
  globules: {
    detected: boolean;
    count: number;
    type: string;
    size_variability: number;
    risk_level: string;
    description: string;
    coordinates: any[];
  };
  streaks: {
    detected: boolean;
    count: number;
    type: string;
    risk_level: string;
    description: string;
    coordinates: any[];
  };
  blue_white_veil: {
    detected: boolean;
    coverage_percentage: number;
    intensity: string;
    risk_level: string;
    description: string;
    coordinates: any[];
  };
  vascular_patterns: {
    detected: boolean;
    type: string;
    risk_level: string;
    description: string;
    vessel_count: number;
    coordinates: any[];
  };
  regression: {
    detected: boolean;
    coverage_percentage: number;
    severity: string;
    risk_level: string;
    description: string;
    coordinates: any[];
  };
  color_analysis: {
    distinct_colors: number;
    variety: string;
    risk_level: string;
    dominant_colors: number[][];
    color_percentages: number[];
  };
  symmetry_analysis: {
    asymmetry_score: number;
    classification: string;
    risk_level: string;
  };
  seven_point_checklist: {
    score: number;
    max_score: number;
    criteria_met: string[];
    interpretation: string;
    urgency: string;
  };
  abcd_score: {
    asymmetry_score: number;
    border_score: number;
    color_score: number;
    structures_score: number;
    total_score: number;
    classification: string;
    recommendation: string;
  };
  risk_assessment: {
    risk_level: string;
    risk_score: number;
    risk_factors: string[];
    recommendation: string;
  };
  overlays: {
    pigment_network?: string;
    globules?: string;
    streaks?: string;
    combined: string;
  };
  timestamp: string;
}