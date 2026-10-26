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
      <div className="min-h-screen flex items-center justify-center bg-canvas">
        <div className="flex flex-col items-center space-y-4">
          <div className="h-9 w-9 rounded-full border-4 border-brand-200 border-t-brand-600 animate-spin" />
          <span className="text-sm font-medium text-slate-600">Opening research workspace…</span>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (allowedRole && user.role !== allowedRole) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-canvas px-4">
        <div className="surface-panel max-w-md w-full text-center space-y-4 p-8 rounded-xl">
          <div className="inline-flex p-3 bg-red-100 text-red-600 rounded-full">
            <ShieldAlert className="h-6 w-6" />
          </div>
          <h2 className="text-lg font-bold text-slate-800">Permission Denied</h2>
          <p className="text-xs text-slate-500">
            Your account does not hold the permissions required to open this research workspace.
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
      <div className="min-h-screen bg-transparent text-slate-800 flex flex-col font-sans">
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

        {/* Global research-use footer */}
        <footer className="mt-auto border-t border-slate-200 bg-white/80 py-5 text-xs text-slate-500">
          <div className="mx-auto flex max-w-[1440px] flex-col gap-3 px-4 sm:px-6 md:flex-row md:items-center md:justify-between lg:px-8">
            <div>&copy; {new Date().getFullYear()} NAFLD Research Console</div>
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
              <span className="font-medium text-brand-700">Local research environment</span>
              <span>Not clinically validated</span>
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
