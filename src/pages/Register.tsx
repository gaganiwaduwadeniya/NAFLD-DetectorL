import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, ShieldAlert, Mail, Lock, User as UserIcon, UserPlus, Check } from 'lucide-react';

export const Register: React.FC = () => {
  const { register, error, clearError } = useAuth();
  const navigate = useNavigate();

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<'doctor' | 'admin'>('doctor');
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);
    clearError();

    if (!name || !email || !password || !role) {
      setLocalError('All fields are required.');
      return;
    }

    if (password.length < 6) {
      setLocalError('Password must be at least 6 characters.');
      return;
    }

    setLoading(true);
    try {
      await register(email, password, name, role);
      navigate('/');
    } catch (err: any) {
      setLocalError(err.message || 'Registration failed. Try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 bg-slate-50" id="register-page">
      <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-2xl border border-slate-100 shadow-sm">
        {/* Header */}
        <div className="text-center space-y-3">
          <div className="inline-flex p-3.5 bg-blue-50 text-blue-600 rounded-2xl border border-blue-100/30">
            <UserPlus className="h-8 w-8 text-blue-600" />
          </div>
          <div>
            <h2 className="font-sans font-extrabold text-slate-900 tracking-tight text-2xl">
              Register Clinical Account
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              Create an account to manage diagnostics and patient reports
            </p>
          </div>
        </div>

        {/* Error Notification */}
        {(localError || error) && (
          <div className="bg-red-50 border border-red-100/50 p-4 rounded-xl flex items-start space-x-3 text-red-800 text-xs">
            <ShieldAlert className="h-4 w-4 text-red-600 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold block">Registration Error</span>
              <span>{localError || error}</span>
            </div>
          </div>
        )}

        <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
          {/* Full Name */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Professional Full Name
            </label>
            <div className="relative">
              <UserIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 h-4.5 w-4.5" />
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/10 focus:border-blue-500 bg-white"
                placeholder="Dr. Alexander Wright, MD"
                id="register-name"
              />
            </div>
          </div>

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
                placeholder="alex.wright@hospital.com"
                id="register-email"
              />
            </div>
          </div>

          {/* Password */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Security Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 h-4.5 w-4.5" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/10 focus:border-blue-500 bg-white"
                placeholder="At least 6 characters"
                id="register-password"
              />
            </div>
          </div>

          {/* Role selection */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Professional System Role
            </label>
            <div className="grid grid-cols-2 gap-4">
              {/* Doctor Selection */}
              <button
                type="button"
                onClick={() => setRole('doctor')}
                className={`py-3 px-4 border rounded-xl text-xs font-semibold text-left transition-all relative flex flex-col justify-between ${
                  role === 'doctor'
                    ? 'border-blue-500 bg-blue-50/40 text-blue-800'
                    : 'border-slate-200 hover:bg-slate-50 text-slate-600'
                }`}
                id="role-doctor-select"
              >
                <span className="font-bold">Medical Doctor</span>
                <span className="text-[10px] text-slate-400 block mt-0.5">Diagnose scans & patients</span>
                {role === 'doctor' && (
                  <span className="absolute top-2 right-2 p-0.5 bg-blue-600 text-white rounded-full">
                    <Check className="h-3 w-3" />
                  </span>
                )}
              </button>

              {/* Admin Selection */}
              <button
                type="button"
                onClick={() => setRole('admin')}
                className={`py-3 px-4 border rounded-xl text-xs font-semibold text-left transition-all relative flex flex-col justify-between ${
                  role === 'admin'
                    ? 'border-indigo-500 bg-indigo-50/40 text-indigo-800'
                    : 'border-slate-200 hover:bg-slate-50 text-slate-600'
                }`}
                id="role-admin-select"
              >
                <span className="font-bold">System Admin</span>
                <span className="text-[10px] text-slate-400 block mt-0.5">Monitor clinic metrics</span>
                {role === 'admin' && (
                  <span className="absolute top-2 right-2 p-0.5 bg-indigo-600 text-white rounded-full">
                    <Check className="h-3 w-3" />
                  </span>
                )}
              </button>
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold shadow-sm hover:shadow transition-all flex items-center justify-center space-x-2 disabled:bg-blue-400 disabled:cursor-not-allowed cursor-pointer mt-2"
            id="btn-register-submit"
          >
            {loading ? (
              <span>Registering Clinician...</span>
            ) : (
              <span>Create Account</span>
            )}
          </button>
        </form>

        {/* Navigation Link */}
        <p className="text-center text-xs text-slate-500 pt-2">
          Already registered?{' '}
          <Link to="/login" className="text-blue-600 font-semibold hover:underline">
            Sign In Instead
          </Link>
        </p>
      </div>
    </div>
  );
};
