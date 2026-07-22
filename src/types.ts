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

export interface Scan {
  id: string;
  patientName: string;
  patientAge: number;
  patientGender: 'Male' | 'Female' | 'Other';
  doctorId: string;
  doctorName: string;
  imageUrl: string;
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
