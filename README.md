# NAFLD Diagnostic Support System

A complete clinical-grade full-stack web application designed for Non-Alcoholic Fatty Liver Disease (NAFLD) classification and tracking from liver ultrasound scans.

## 🛠️ Tech Stack & Architecture

- **Frontend**: Single Page Application built on React 19, Vite, Tailwind CSS 4, and Recharts.
- **Backend**: Express Server handles authentication, secure database indexing, and relays files to the Flask ML diagnostic service.
- **Database**:
  - **Local-First Fallback (Default)**: Automatically operates out-of-the-box using an extremely robust file-backed JSON database (`data/db.json`) inside the container. Fully supports sessions, password hashing (SHA256), scan records, and administrative stats.
  - **Firebase Integration (Optional)**: Automatically activates and connects to Firestore, Firebase Auth, and Storage when credentials are provided in environment variables.

---

## 🚀 Pre-loaded Demo Accounts

For rapid verification inside the Google AI Studio preview, we have provided two pre-loaded clinic accounts:

### 1. Medical Doctor (Clinician)
- **Email**: `doctor@system.com`
- **Password**: `doctor123`
- *Capabilities*: Upload ultrasound scans (JPG, PNG, BMP), enter patient demographics, execute AI diagnoses, view real-time probability bar charts, and view patient history logs.

### 2. System Admin (Surveillance)
- **Email**: `admin@system.com`
- **Password**: `admin123`
- *Capabilities*: Check global diagnostic counts, view normal vs abnormal charts, review submission timelines, and monitor physician directories.

---

## 📂 Configuration (Environment Variables)

Create a `.env.local` file at the root or define variables in the platform dashboard:

```env
# URL where the separate Flask prediction API is running
ML_SERVICE_URL="http://localhost:5001"

# Secure key used to sign JWT session tokens
JWT_SECRET="nafld-diagnostic-system-secret-key-2026"

# Firebase Credentials (Optional - Falls back to secure local file storage if empty)
NEXT_PUBLIC_FIREBASE_API_KEY=""
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=""
NEXT_PUBLIC_FIREBASE_PROJECT_ID=""
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=""
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=""
NEXT_PUBLIC_FIREBASE_APP_ID=""
```

---

## 🩺 System Features

1. **Secure Session Auth**: Full token-based login and clinician registration.
2. **Ultrasound Workspace**: Clean drag-and-drop or click-to-browse DICOM/Ultrasound frame uploader with pre-submission previews.
3. **High-Fidelity AI Report**:
   - **Colored Badges**: Immediate classification outcome ("Normal" or "Abnormal" steatosis findings).
   - **Confidence Indicator**: Custom progressive confidence meter.
   - **Probability Distribution**: Professional Horizontal Recharts distribution showing detailed probability splits.
   - **HIPAA Disclaimer**: Displays strict clinical utility and liability disclaimers.
4. **History Dossier**: Search patient names/IDs, sort by dates/ages, and filter results instantly.
5. **Surveillance & Tracking**: Admin interface showing real-time timeline analytics of scans.
