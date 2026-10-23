export interface User {
  uid: string;
  email: string;
  name: string;
  role: 'doctor' | 'admin';
  createdAt: string;
}

export interface ScanProbabilities {
  Normal: number;
  Abnormal: number;
}

export interface V3InferenceMetadata {
  schemaVersion: 3;
  modelVersion: string;
  contractSha256: string;
  inputMode: 'single_frame';
  researchUseOnly: boolean;

  // Cascade outputs
  binaryResult: 'Non-NAFLD' | 'NAFLD';
  binaryProbNafld: number;
  binaryProbNonNafld: number;
  binaryThreshold: number;
  binaryFoldProbs: number[];
  binaryFoldStd: number;

  gradingPerformed: boolean;
  gradingResult?: 'Grade1_Mild' | 'Grade2_Moderate_Severe';
  gradingProbModerateSevere?: number;
  gradingProbMild?: number;
  gradingThreshold?: number;
  gradingFoldProbs?: number[];
  gradingFoldStd?: number;

  finalLabel: 'Non-NAFLD' | 'NAFLD-Grade1_Mild' | 'NAFLD-Grade2_Moderate_Severe';

  // Execution & Image metadata
  imageWidth?: number;
  imageHeight?: number;
  inferenceLatencyMs?: number;
}

export interface ResearchV1InferenceMetadata {
  schemaVersion: 4;
  provider: 'pytorch_research_v1';
  contractId: 'nafld-pytorch-research-v1-single-frame-v1';
  modelVersion: string;
  datasetVersion: string;
  inputMode: 'single_frame';
  researchUseOnly: true;
  clinicallyValidated: false;
  probabilityCalibrationVerified: false;
  finalStage: 0 | 1 | 2 | 3;
  finalStageLabel: string;
  binaryScoreNafld: number;
  binaryScoreNonNafld: number;
  binaryThreshold: number;
  gradingPerformed: boolean;
  conditionalStageScores?: { '1': number; '2': number; '3': number };
  jointStageScores?: { '0': number; '1': number; '2': number; '3': number };
  binaryArchitecture: 'densenet121';
  stagingArchitecture: 'efficientnet_b0';
  binaryManifestSha256: string;
  stagingManifestSha256: string;
  explanationAvailable: boolean;
  imageStored: boolean;
  serviceInferenceMs?: number;
  totalRequestMs?: number;
  imageWidth?: number;
  imageHeight?: number;
}

export interface Scan extends Partial<Omit<V3InferenceMetadata, 'schemaVersion' | 'modelVersion' | 'inputMode' | 'researchUseOnly' | 'gradingPerformed' | 'imageWidth' | 'imageHeight'>>,
  Partial<Omit<ResearchV1InferenceMetadata, 'schemaVersion' | 'modelVersion' | 'inputMode' | 'researchUseOnly' | 'gradingPerformed' | 'imageWidth' | 'imageHeight'>> {
  id: string;
  patientName: string;
  patientAge: number;
  patientGender: 'Male' | 'Female' | 'Other';
  doctorId: string;
  doctorName: string;
  imageUrl: string;

  // Legacy compatibility fields
  prediction: 'Normal' | 'Abnormal';
  confidence: number;
  probabilities: ScanProbabilities;
  timestamp: string; // ISO format or Firestore Timestamp representation
  schemaVersion?: 3 | 4;
  modelVersion?: string;
  inputMode?: 'single_frame';
  researchUseOnly?: boolean;
  gradingPerformed?: boolean;
  imageWidth?: number;
  imageHeight?: number;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  loading: boolean;
  error: string | null;
}

export interface AdminStats {
  totalScans: number;
  normalCount: number;
  abnormalCount: number;
  scansByDay: { date: string; count: number }[];
  recentScans: Scan[];
}
