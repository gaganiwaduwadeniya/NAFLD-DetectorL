import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, ShieldAlert, Mail, Lock, LogIn, ArrowRight } from 'lucide-react';

export const Login: React.FC = () => {
  const { login, error, clearError } = useAuth();
  const navigate = useNavigate();
  
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);
    clearError();

    if (!email || !password) {
      setLocalError('Please enter both email and password.');
      return;
    }

    setLoading(true);
    try {
      await login(email, password);
      navigate('/');
    } catch (err: any) {
      setLocalError(err.message || 'Login failed. Please check credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handlePreload = (role: 'doctor' | 'admin') => {
    if (role === 'doctor') {
      setEmail('doctor@system.com');
      setPassword('doctor123');
    } else {
      setEmail('admin@system.com');
      setPassword('admin123');
    }
    setLocalError(null);
    clearError();
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 bg-slate-50" id="login-page">
      <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-2xl border border-slate-100 shadow-sm">
        {/* Header */}
        <div className="text-center space-y-3">
          <div className="inline-flex p-3.5 bg-blue-50 text-blue-600 rounded-2xl border border-blue-100/30">
            <Activity className="h-8 w-8 animate-pulse" />
          </div>
          <div>
            <h2 className="font-sans font-extrabold text-slate-900 tracking-tight text-2xl">
              NAFLD Medical Portal
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              Sign in to access ultrasound diagnostics and records
            </p>
          </div>
        </div>

        {/* Form Error Display */}
        {(localError || error) && (
          <div className="bg-red-50 border border-red-100/50 p-4 rounded-xl flex items-start space-x-3 text-red-800 text-xs">
            <ShieldAlert className="h-4 w-4 text-red-600 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold block">Access Denied</span>
              <span>{localError || error}</span>
            </div>
          </div>
        )}

        <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
          {/* Email */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Work Email Address
            </label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 h-4.5 w-4.5" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/10 focus:border-blue-500 bg-white"
                placeholder="physician@hospital.com"
                id="login-email"
              />
            </div>
          </div>

          {/* Password */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center">
              <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
                Security Password
              </label>
              <Link to="/forgot-password" className="text-xs text-blue-600 font-semibold hover:underline">
                Forgot password?
              </Link>
            </div>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 h-4.5 w-4.5" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/10 focus:border-blue-500 bg-white"
                placeholder="••••••••"
                id="login-password"
              />
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold shadow-sm hover:shadow transition-all flex items-center justify-center space-x-2 disabled:bg-blue-400 disabled:cursor-not-allowed cursor-pointer mt-2"
            id="btn-login-submit"
          >
            {loading ? (
              <span>Authenticating...</span>
            ) : (
              <>
                <span>Sign In to System</span>
                <LogIn className="h-4 w-4" />
              </>
            )}
          </button>
        </form>

        {/* Demo Accounts Panel */}
        <div className="border-t border-slate-100 pt-6">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-3 text-center">
            System Pre-Loaded Demo Credentials
          </span>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => handlePreload('doctor')}
              className="px-3 py-2 border border-slate-150 rounded-xl text-xs font-semibold text-slate-700 hover:bg-blue-50/50 hover:text-blue-700 hover:border-blue-200 transition-all text-left flex flex-col justify-between"
              type="button"
              id="preload-doctor-btn"
            >
              <span className="block font-bold">Doctor Role</span>
              <span className="text-[10px] font-mono text-slate-400 block mt-0.5">doctor@system.com</span>
            </button>
            <button
              onClick={() => handlePreload('admin')}
              className="px-3 py-2 border border-slate-150 rounded-xl text-xs font-semibold text-slate-700 hover:bg-indigo-50/50 hover:text-indigo-700 hover:border-indigo-200 transition-all text-left flex flex-col justify-between"
              type="button"
              id="preload-admin-btn"
            >
              <span className="block font-bold">Admin Role</span>
              <span className="text-[10px] font-mono text-slate-400 block mt-0.5">admin@system.com</span>
            </button>
          </div>
        </div>

        {/* Footer Link */}
        <p className="text-center text-xs text-slate-500 pt-2">
          New clinical staff?{' '}
          <Link to="/register" className="text-blue-600 font-semibold hover:underline">
            Register Account
          </Link>
        </p>
      </div>
    </div>
  );
};
