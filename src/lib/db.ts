import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { db, isFirebaseConfigured } from './firebase.js';
import { 
  collection, 
  doc, 
  setDoc, 
  getDoc, 
  getDocs, 
  query, 
  where, 
  orderBy,
  limit
} from 'firebase/firestore';
import { User, Scan, AdminStats } from '../types.js';

// Paths for local storage fallback
const DATA_DIR = path.join(process.cwd(), 'data');
const DB_FILE = path.join(DATA_DIR, 'db.json');

// Ensure data directory and file exist
const initLocalDb = () => {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  if (!fs.existsSync(DB_FILE)) {
    const initialData = {
      users: [
        {
          uid: 'admin-1',
          email: 'admin@system.com',
          name: 'System Admin',
          role: 'admin',
          passwordHash: crypto.createHash('sha256').update('admin123').digest('hex'),
          createdAt: new Date().toISOString()
        },
        {
          uid: 'doctor-1',
          email: 'doctor@system.com',
          name: 'Dr. Jane Smith',
          role: 'doctor',
          passwordHash: crypto.createHash('sha256').update('doctor123').digest('hex'),
          createdAt: new Date().toISOString()
        }
      ],
      scans: [
        {
          id: 'scan-1',
          patientName: 'John Doe',
          patientAge: 45,
          patientGender: 'Male',
          doctorId: 'doctor-1',
          doctorName: 'Dr. Jane Smith',
          imageUrl: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800&auto=format&fit=crop&q=60',
          prediction: 'Abnormal',
          confidence: 87.5,
          probabilities: { Normal: 12.5, Abnormal: 87.5 },
          timestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString() // 3 days ago
        },
        {
          id: 'scan-2',
          patientName: 'Alice Johnson',
          patientAge: 32,
          patientGender: 'Female',
          doctorId: 'doctor-1',
          doctorName: 'Dr. Jane Smith',
          imageUrl: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800&auto=format&fit=crop&q=60',
          prediction: 'Normal',
          confidence: 94.2,
          probabilities: { Normal: 94.2, Abnormal: 5.8 },
          timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString() // 1 day ago
        },
        {
          id: 'scan-3',
          patientName: 'Robert Lee',
          patientAge: 61,
          patientGender: 'Male',
          doctorId: 'doctor-1',
          doctorName: 'Dr. Jane Smith',
          imageUrl: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800&auto=format&fit=crop&q=60',
          prediction: 'Abnormal',
          confidence: 76.1,
          probabilities: { Normal: 23.9, Abnormal: 76.1 },
          timestamp: new Date().toISOString()
        }
      ]
    };
    fs.writeFileSync(DB_FILE, JSON.stringify(initialData, null, 2), 'utf-8');
  }
};

// Helper to read local DB
const readLocalDb = () => {
  initLocalDb();
  try {
    const data = fs.readFileSync(DB_FILE, 'utf-8');
    return JSON.parse(data);
  } catch (err) {
    console.error('Error reading local db:', err);
    return { users: [], scans: [] };
  }
};

// Helper to write local DB
const writeLocalDb = (data: any) => {
  initLocalDb();
  try {
    fs.writeFileSync(DB_FILE, JSON.stringify(data, null, 2), 'utf-8');
  } catch (err) {
    console.error('Error writing local db:', err);
  }
};

// --- AUTH & USER DATABASE OPERATIONS ---

export const dbCreateUser = async (user: User & { passwordHash?: string; password?: string }): Promise<User> => {
  const pHash = user.passwordHash || (user.password ? crypto.createHash('sha256').update(user.password).digest('hex') : '');
  
  if (isFirebaseConfigured()) {
    try {
      const userRef = doc(db, 'users', user.uid);
      const userToSave = {
        uid: user.uid,
        email: user.email,
        name: user.name,
        role: user.role,
        createdAt: user.createdAt
      };
      await setDoc(userRef, userToSave);
      return userToSave;
    } catch (error) {
      console.error('Firestore createUser error, trying local fallback:', error);
    }
  }

  // Local fallback
  const localDb = readLocalDb();
  const index = localDb.users.findIndex((u: any) => u.uid === user.uid || u.email === user.email);
  const newUser = {
    uid: user.uid,
    email: user.email,
    name: user.name,
    role: user.role,
    passwordHash: pHash,
    createdAt: user.createdAt
  };

  if (index !== -1) {
    localDb.users[index] = { ...localDb.users[index], ...newUser };
  } else {
    localDb.users.push(newUser);
  }
  writeLocalDb(localDb);
  return {
    uid: newUser.uid,
    email: newUser.email,
    name: newUser.name,
    role: newUser.role,
    createdAt: newUser.createdAt
  };
};

export const dbGetUserByUid = async (uid: string): Promise<User | null> => {
  if (isFirebaseConfigured()) {
    try {
      const userRef = doc(db, 'users', uid);
      const userSnap = await getDoc(userRef);
      if (userSnap.exists()) {
        return userSnap.data() as User;
      }
    } catch (error) {
      console.error('Firestore getUserByUid error, trying local fallback:', error);
    }
  }

  // Local fallback
  const localDb = readLocalDb();
  const user = localDb.users.find((u: any) => u.uid === uid);
  if (user) {
    return {
      uid: user.uid,
      email: user.email,
      name: user.name,
      role: user.role,
      createdAt: user.createdAt
    };
  }
  return null;
};

export const dbGetUserByEmailWithPassword = async (email: string): Promise<(User & { passwordHash: string }) | null> => {
  const localDb = readLocalDb();
  const user = localDb.users.find((u: any) => u.email.toLowerCase() === email.toLowerCase());
  return user || null;
};

export const dbGetAllUsers = async (): Promise<User[]> => {
  if (isFirebaseConfigured()) {
    try {
      const querySnap = await getDocs(collection(db, 'users'));
      const list: User[] = [];
      querySnap.forEach((doc) => {
        list.push(doc.data() as User);
      });
      return list;
    } catch (error) {
      console.error('Firestore getAllUsers error:', error);
    }
  }

  const localDb = readLocalDb();
  return localDb.users.map((u: any) => ({
    uid: u.uid,
    email: u.email,
    name: u.name,
    role: u.role,
    createdAt: u.createdAt
  }));
};

// --- SCANS DATABASE OPERATIONS ---

export const dbCreateScan = async (scan: Scan): Promise<Scan> => {
  if (isFirebaseConfigured()) {
    try {
      const scanRef = doc(db, 'scans', scan.id);
      await setDoc(scanRef, scan);
      return scan;
    } catch (error) {
      console.error('Firestore createScan error, trying local fallback:', error);
    }
  }

  // Local fallback
  const localDb = readLocalDb();
  localDb.scans.unshift(scan); // Add to the beginning for standard descending order
  writeLocalDb(localDb);
  return scan;
};

export const dbGetScans = async (doctorId?: string, role?: 'doctor' | 'admin'): Promise<Scan[]> => {
  if (isFirebaseConfigured()) {
    try {
      let scansRef = collection(db, 'scans');
      let q;
      if (role === 'doctor' && doctorId) {
        q = query(scansRef, where('doctorId', '==', doctorId), orderBy('timestamp', 'desc'));
      } else {
        q = query(scansRef, orderBy('timestamp', 'desc'));
      }
      const querySnap = await getDocs(q);
      const list: Scan[] = [];
      querySnap.forEach((doc) => {
        list.push(doc.data() as Scan);
      });
      return list;
    } catch (error) {
      console.error('Firestore getScans error, trying local fallback:', error);
    }
  }

  // Local fallback
  const localDb = readLocalDb();
  let filtered = localDb.scans;
  if (role === 'doctor' && doctorId) {
    filtered = localDb.scans.filter((s: any) => s.doctorId === doctorId);
  }
  // Sort descending by timestamp
  return filtered.sort((a: any, b: any) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
};

export const dbGetScanById = async (id: string): Promise<Scan | null> => {
  if (isFirebaseConfigured()) {
    try {
      const docRef = doc(db, 'scans', id);
      const docSnap = await getDoc(docRef);
      if (docSnap.exists()) {
        return docSnap.data() as Scan;
      }
    } catch (error) {
      console.error('Firestore getScanById error, trying local fallback:', error);
    }
  }

  const localDb = readLocalDb();
  const scan = localDb.scans.find((s: any) => s.id === id);
  return scan || null;
};

export const dbGetAdminStats = async (): Promise<AdminStats> => {
  const scans = await dbGetScans();
  
  const totalScans = scans.length;
  const normalCount = scans.filter(s => s.prediction === 'Normal').length;
  const abnormalCount = scans.filter(s => s.prediction === 'Abnormal').length;

  // Group scans by date for the last 30 days
  const scansByDayMap: Record<string, number> = {};
  
  // Pre-populate last 10 days with 0 to ensure standard display even if no scans exist
  for (let i = 9; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const dateStr = d.toISOString().split('T')[0];
    scansByDayMap[dateStr] = 0;
  }

  scans.forEach(scan => {
    try {
      const dateStr = scan.timestamp.split('T')[0];
      if (dateStr in scansByDayMap) {
        scansByDayMap[dateStr]++;
      } else {
        // Only include if it is within 30 days
        const scanDate = new Date(dateStr);
        const diffTime = Math.abs(new Date().getTime() - scanDate.getTime());
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        if (diffDays <= 30) {
          scansByDayMap[dateStr] = 1;
        }
      }
    } catch (e) {
      // Ignore parsing errors
    }
  });

  const scansByDay = Object.keys(scansByDayMap)
    .sort()
    .map(date => ({
      date,
      count: scansByDayMap[date]
    }));

  return {
    totalScans,
    normalCount,
    abnormalCount,
    scansByDay,
    recentScans: scans.slice(0, 10)
  };
};
