/**
 * Voice-to-Text Clinical Documentation Service
 *
 * Provides hands-free clinical documentation capabilities:
 * - Speech-to-text for symptom capture
 * - Voice commands for app navigation
 * - Auto-generation of SOAP notes from dictation
 * - NLP extraction of clinical data (duration, severity, location)
 * - Clinical terminology recognition
 */

import { API_BASE_URL } from '../config';
import AuthService from './AuthService';

// Types
export interface VoiceCommand {
  type: string;
  action: string;
  parameters: Record<string, any>;
  confidence: number;
}

export interface ExtractedClinicalData {
  type: string;
  value: string;
  normalized: string;
  unit: string | null;
  confidence: number;
  source: string;
}

export interface MedicalTerm {
  term: string;
  category: string;
  type: string;
  position: [number, number];
}

export interface ProcessedVoiceResult {
  original_text: string;
  processed_text: string;
  mode: string;
  timestamp: string;
  command?: VoiceCommand;
  extracted_data?: ExtractedClinicalData[];
  medical_terms?: MedicalTerm[];
}

export interface SOAPNote {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
  structured_data: {
    chief_complaint: string | null;
    duration: string | null;
    severity: string | null;
    location: string | null;
    associated_symptoms: string[];
  };
  confidence_score: number;
  generated_at: string;
  saved_to_analysis?: number;
}

export interface DictationSession {
  session_id: string;
  message: string;
  analysis_id: number | null;
}

export interface DictationSegmentResult {
  session_id: string;
  segment_count: number;
  current_length: number;
  latest_extraction: ExtractedClinicalData[];
}

export interface DictationEndResult {
  session_id: string;
  full_text: string;
  segment_count: number;
  started_at: string;
  ended_at: string;
  soap_note?: SOAPNote;
}

export interface SymptomCaptureResult {
  original_text: string;
  symptoms: {
    duration: string | null;
    severity: string | null;
    location: string | null;
    itching: boolean;
    pain: boolean;
    bleeding: boolean;
    burning: boolean;
    swelling: boolean;
    changes: string[];
    other_symptoms: string[];
  };
  raw_extractions: ExtractedClinicalData[];
  saved_to_analysis?: number;
}

export interface VoiceCommandsReference {
  commands: {
    [category: string]: {
      action: string;
      phrases: string[];
      example: string;
    }[];
  };
  usage: {
    description: string;
    tips: string[];
  };
}

export interface TerminologyReference {
  abbreviations: {
    description: string;
    examples: Record<string, string>;
    total: number;
  };
  conditions: {
    description: string;
    categories: string[];
    sample: Record<string, string[]>;
  };
  body_locations: {
    description: string;
    regions: string[];
    sample: Record<string, string[]>;
  };
  usage_tips: string[];
}

// Voice command action handlers type
export type VoiceCommandHandler = (command: VoiceCommand) => void | Promise<void>;

class VoiceDocumentationService {
  private baseUrl: string;
  private commandHandlers: Map<string, VoiceCommandHandler> = new Map();
  private activeDictationSession: string | null = null;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  /**
   * Get authorization headers for API requests
   */
  private async getAuthHeaders(): Promise<Record<string, string>> {
    const token = await AuthService.getToken();
    return {
      'Authorization': `Bearer ${token}`,
    };
  }

  /**
   * Process voice transcription (from speech-to-text)
   *
   * @param text - Transcribed text from speech
   * @param mode - Processing mode: 'auto', 'command', or 'dictation'
   * @param language - Language code (default: 'en-US')
   */
  async processVoiceInput(
    text: string,
    mode: 'auto' | 'command' | 'dictation' = 'auto',
    language: string = 'en-US'
  ): Promise<ProcessedVoiceResult> {
    const headers = await this.getAuthHeaders();

    const formData = new FormData();
    formData.append('text', text);
    formData.append('mode', mode);
    formData.append('language', language);

    const response = await fetch(`${this.baseUrl}/voice/process`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Voice processing failed: ${response.status}`);
    }

    const result = await response.json();

    // If a command was detected and we have a handler, execute it
    if (result.command && result.mode === 'command') {
      const handler = this.commandHandlers.get(result.command.action);
      if (handler) {
        await handler(result.command);
      }
    }

    return result;
  }

  /**
   * Generate SOAP note from voice dictation
   *
   * @param dictation - Full dictation text
   * @param analysisId - Optional analysis ID to save the note to
   */
  async generateSOAPNote(dictation: string, analysisId?: number): Promise<SOAPNote> {
    const headers = await this.getAuthHeaders();

    const formData = new FormData();
    formData.append('dictation', dictation);
    if (analysisId) {
      formData.append('analysis_id', analysisId.toString());
    }

    const response = await fetch(`${this.baseUrl}/voice/generate-soap`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`SOAP generation failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Start a new dictation session
   *
   * @param analysisId - Optional analysis ID to link dictation to
   */
  async startDictationSession(analysisId?: number): Promise<DictationSession> {
    const headers = await this.getAuthHeaders();

    const formData = new FormData();
    if (analysisId) {
      formData.append('analysis_id', analysisId.toString());
    }

    const response = await fetch(`${this.baseUrl}/voice/dictation/start`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to start dictation: ${response.status}`);
    }

    const result = await response.json();
    this.activeDictationSession = result.session_id;
    return result;
  }

  /**
   * Add transcribed segment to active dictation session
   *
   * @param sessionId - Dictation session ID
   * @param text - Transcribed text segment
   */
  async addDictationSegment(sessionId: string, text: string): Promise<DictationSegmentResult> {
    const headers = await this.getAuthHeaders();

    const formData = new FormData();
    formData.append('text', text);

    const response = await fetch(`${this.baseUrl}/voice/dictation/${sessionId}/add`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to add dictation segment: ${response.status}`);
    }

    return response.json();
  }

  /**
   * End dictation session and optionally generate SOAP note
   *
   * @param sessionId - Dictation session ID
   * @param generateSoap - Whether to generate SOAP note (default: true)
   */
  async endDictationSession(
    sessionId: string,
    generateSoap: boolean = true
  ): Promise<DictationEndResult> {
    const headers = await this.getAuthHeaders();

    const formData = new FormData();
    formData.append('generate_soap', generateSoap.toString());

    const response = await fetch(`${this.baseUrl}/voice/dictation/${sessionId}/end`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to end dictation: ${response.status}`);
    }

    if (this.activeDictationSession === sessionId) {
      this.activeDictationSession = null;
    }

    return response.json();
  }

  /**
   * Capture symptoms from voice input
   *
   * @param text - Voice transcription describing symptoms
   * @param analysisId - Optional analysis ID to save symptoms to
   */
  async captureSymptoms(text: string, analysisId?: number): Promise<SymptomCaptureResult> {
    const headers = await this.getAuthHeaders();

    const formData = new FormData();
    formData.append('text', text);
    if (analysisId) {
      formData.append('analysis_id', analysisId.toString());
    }

    const response = await fetch(`${this.baseUrl}/voice/symptom-capture`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Symptom capture failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Extract clinical data from text using NLP
   *
   * @param text - Text to analyze
   */
  async extractClinicalData(text: string): Promise<{
    original_text: string;
    processed_text: string;
    extracted_data: ExtractedClinicalData[];
    medical_terms: MedicalTerm[];
  }> {
    const headers = await this.getAuthHeaders();

    const formData = new FormData();
    formData.append('text', text);

    const response = await fetch(`${this.baseUrl}/voice/extract-clinical-data`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Clinical data extraction failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Get available voice commands
   */
  async getVoiceCommands(): Promise<VoiceCommandsReference> {
    const response = await fetch(`${this.baseUrl}/voice/commands`);

    if (!response.ok) {
      throw new Error(`Failed to get voice commands: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Get medical terminology reference
   */
  async getTerminologyReference(): Promise<TerminologyReference> {
    const response = await fetch(`${this.baseUrl}/voice/terminology`);

    if (!response.ok) {
      throw new Error(`Failed to get terminology: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Normalize medical text (expand abbreviations, correct terms)
   *
   * @param text - Text to normalize
   * @param expandAbbreviations - Whether to expand abbreviations
   * @param correctTerms - Whether to correct phonetic errors
   */
  async normalizeText(
    text: string,
    expandAbbreviations: boolean = true,
    correctTerms: boolean = true
  ): Promise<{
    original: string;
    phonetically_corrected?: string;
    abbreviations_expanded?: string;
    normalized: string;
    medical_terms: MedicalTerm[];
  }> {
    const headers = await this.getAuthHeaders();

    const formData = new FormData();
    formData.append('text', text);
    formData.append('expand_abbreviations', expandAbbreviations.toString());
    formData.append('correct_terms', correctTerms.toString());

    const response = await fetch(`${this.baseUrl}/voice/normalize-text`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Text normalization failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Register a handler for a voice command action
   *
   * @param action - Command action name (e.g., 'take_photo', 'go_home')
   * @param handler - Function to execute when command is recognized
   */
  registerCommandHandler(action: string, handler: VoiceCommandHandler): void {
    this.commandHandlers.set(action, handler);
  }

  /**
   * Unregister a command handler
   */
  unregisterCommandHandler(action: string): void {
    this.commandHandlers.delete(action);
  }

  /**
   * Check if there's an active dictation session
   */
  hasActiveDictation(): boolean {
    return this.activeDictationSession !== null;
  }

  /**
   * Get active dictation session ID
   */
  getActiveDictationSession(): string | null {
    return this.activeDictationSession;
  }
}

// Export singleton instance
export default new VoiceDocumentationService();
