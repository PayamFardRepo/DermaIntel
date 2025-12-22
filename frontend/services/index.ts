export { default as AuthService } from './AuthService';
export type { User, LoginCredentials, RegisterData } from './AuthService';

export { default as VoiceDocumentationService } from './VoiceDocumentationService';
export type {
  VoiceCommand,
  ExtractedClinicalData,
  MedicalTerm,
  ProcessedVoiceResult,
  SOAPNote,
  DictationSession,
  DictationSegmentResult,
  DictationEndResult,
  SymptomCaptureResult,
  VoiceCommandsReference,
  TerminologyReference,
} from './VoiceDocumentationService';