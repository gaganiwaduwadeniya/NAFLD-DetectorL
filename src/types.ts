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

export interface Scan extends Partial<V3InferenceMetadata> {
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
