import { initializeApp, getApps, getApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getAuth } from 'firebase/auth';
import { getStorage } from 'firebase/storage';
import firebaseConfigJson from '../../firebase-applet-config.json';

// Check if Firebase config has valid details or if env variables exist
export const isFirebaseConfigured = (): boolean => {
  return !!(
    firebaseConfigJson?.apiKey ||
    process.env.NEXT_PUBLIC_FIREBASE_API_KEY ||
    process.env.VITE_FIREBASE_API_KEY ||
    process.env.FIREBASE_API_KEY
  );
};

const getFirebaseConfig = () => {
  return {
    apiKey: firebaseConfigJson?.apiKey || process.env.NEXT_PUBLIC_FIREBASE_API_KEY || process.env.VITE_FIREBASE_API_KEY || process.env.FIREBASE_API_KEY,
    authDomain: firebaseConfigJson?.authDomain || process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN || process.env.VITE_FIREBASE_AUTH_DOMAIN || process.env.FIREBASE_AUTH_DOMAIN,
    projectId: firebaseConfigJson?.projectId || process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || process.env.VITE_FIREBASE_PROJECT_ID || process.env.FIREBASE_PROJECT_ID,
    storageBucket: firebaseConfigJson?.storageBucket || process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET || process.env.VITE_FIREBASE_STORAGE_BUCKET || process.env.FIREBASE_STORAGE_BUCKET,
    messagingSenderId: firebaseConfigJson?.messagingSenderId || process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID || process.env.VITE_FIREBASE_MESSAGING_SENDER_ID || process.env.FIREBASE_MESSAGING_SENDER_ID,
    appId: firebaseConfigJson?.appId || process.env.NEXT_PUBLIC_FIREBASE_APP_ID || process.env.VITE_FIREBASE_APP_ID || process.env.FIREBASE_APP_ID,
  };
};

let app;
let db: any = null;
let auth: any = null;
let storage: any = null;

if (isFirebaseConfigured()) {
  try {
    const config = getFirebaseConfig();
    app = getApps().length === 0 ? initializeApp(config) : getApp();
    
    // Determine databaseId if specified in firebase-applet-config.json
    const dbId = (firebaseConfigJson as any)?.firestoreDatabaseId || (firebaseConfigJson as any)?.databaseId;
    db = dbId ? getFirestore(app, dbId) : getFirestore(app);
    
    auth = getAuth(app);
    storage = getStorage(app);
    console.log('Firebase successfully initialized on the backend. Project:', config.projectId, 'Database ID:', dbId || '(default)');
  } catch (error) {
    console.error('Error initializing Firebase:', error);
  }
} else {
  console.log('Firebase env variables not found or incomplete. Falling back to local/in-memory database.');
}

export { app, db, auth, storage };
