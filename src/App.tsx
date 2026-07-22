import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { Navbar } from './components/Navbar';
import { Login } from './pages/Login';
import { Register } from './pages/Register';
import { ForgotPassword } from './pages/ForgotPassword';
import { DoctorDashboard } from './pages/DoctorDashboard';
import { PatientHistory } from './pages/PatientHistory';
import { ScanDetail } from './pages/ScanDetail';
import { AdminDashboard } from './pages/AdminDashboard';
import { ShieldAlert } from 'lucide-react';

// Route Guard for authenticated users
const ProtectedRoute: React.FC<{ children: React.ReactNode; allowedRole?: 'doctor' | 'admin' }> = ({ 
  children, 
  allowedRole 
}) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="flex flex-col items-center space-y-4">
          <div className="h-10 w-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm font-semibold text-slate-500">Decrypting clinical session...</span>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (allowedRole && user.role !== allowedRole) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 px-4">
        <div className="max-w-md w-full text-center space-y-4 bg-white p-8 rounded-2xl border border-slate-100 shadow-sm">
          <div className="inline-flex p-3 bg-red-100 text-red-600 rounded-full">
            <ShieldAlert className="h-6 w-6" />
          </div>
          <h2 className="text-lg font-bold text-slate-800">Permission Denied</h2>
          <p className="text-xs text-slate-500">
            Your clinical account does not hold the permissions required to enter this sector.
          </p>
          <Navigate to="/" replace />
        </div>
      </div>
    );
  }

  return <>{children}</>;
};

// Main Routing and Layout controller
const AppContent: React.FC = () => {
  const { user } = useAuth();

  return (
    <Router>
      <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans selection:bg-blue-100 selection:text-blue-800">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            {/* Public Authentication sector */}
            <Route path="/login" element={!user ? <Login /> : <Navigate to="/" />} />
            <Route path="/register" element={!user ? <Register /> : <Navigate to="/" />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />

            {/* Core Protected workspaces */}
            <Route 
              path="/" 
              element={
                <ProtectedRoute>
                  {user?.role === 'admin' ? <AdminDashboard /> : <DoctorDashboard />}
                </ProtectedRoute>
              } 
            />

            {/* Doctor-Specific Routes */}
            <Route 
              path="/history" 
              element={
                <ProtectedRoute allowedRole="doctor">
                  <PatientHistory />
                </ProtectedRoute>
              } 
            />

            {/* Scan Detail Route */}
            <Route 
              path="/scan/:id" 
              element={
                <ProtectedRoute>
                  <ScanDetail />
                </ProtectedRoute>
              } 
            />

            {/* Fallback Catch-All */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        {/* Global Hospital Grade Footer */}
        <footer className="bg-white border-t border-slate-200 py-6 text-center text-slate-400 text-xs">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
            <div>
              &copy; {new Date().getFullYear()} NAFLD Diagnostic Support System • Level-3 AI Medical Grade System
            </div>
            <div className="flex items-center space-x-4 font-mono text-[10px] tracking-wider uppercase font-medium">
              <span className="flex items-center text-blue-600">
                <span className="h-1.5 w-1.5 rounded-full bg-blue-600 mr-1.5 animate-ping" />
                SYSTEM LIVE (V1.0.4)
              </span>
              <span>HIPAA COMPLIANT DB</span>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  );
};

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
