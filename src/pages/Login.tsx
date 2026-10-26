import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowRight, LockKeyhole, Mail, ShieldAlert, Stethoscope, UserCog } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { AuthShell } from '../components/AuthShell';

export const Login: React.FC = () => {
  const { login, error, clearError } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLocalError(null);
    clearError();
    if (!email || !password) {
      setLocalError('Enter both your email and password.');
      return;
    }
    setLoading(true);
    try {
      await login(email, password);
      navigate('/');
    } catch (err: any) {
      setLocalError(err.message || 'Sign-in failed. Check the credentials and try again.');
    } finally {
      setLoading(false);
    }
  };

  const useDemoAccount = (role: 'doctor' | 'admin') => {
    setEmail(role === 'doctor' ? 'doctor@system.com' : 'admin@system.com');
    setPassword(role === 'doctor' ? 'doctor123' : 'admin123');
    setLocalError(null);
    clearError();
  };

  return (
    <div id="login-page">
      <AuthShell eyebrow="Secure local access" title="Sign in to the research workspace" description="Use a local research account to run the selected model, review versioned outputs, and inspect saved records.">
        {(localError || error) && (
          <div className="mb-5 flex gap-3 rounded-lg border border-rose-200 bg-rose-50 p-3.5 text-sm text-rose-900" role="alert">
            <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <div><strong className="block font-semibold">Unable to sign in</strong><span>{localError || error}</span></div>
          </div>
        )}

        <form className="space-y-5" onSubmit={handleSubmit}>
          <div>
            <label className="field-label" htmlFor="login-email">Email address</label>
            <div className="relative">
              <Mail className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" aria-hidden="true" />
              <input id="login-email" type="email" autoComplete="username" required value={email} onChange={(event) => setEmail(event.target.value)} className="field-control pl-10" placeholder="name@example.com" />
            </div>
          </div>

          <div>
            <div className="mb-2 flex items-center justify-between">
              <label className="text-sm font-semibold text-slate-700" htmlFor="login-password">Password</label>
              <Link to="/forgot-password" className="text-sm font-semibold text-brand-700 hover:text-brand-600">Need help?</Link>
            </div>
            <div className="relative">
              <LockKeyhole className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" aria-hidden="true" />
              <input id="login-password" type="password" autoComplete="current-password" required value={password} onChange={(event) => setPassword(event.target.value)} className="field-control pl-10" placeholder="Enter your password" />
            </div>
          </div>

          <button id="btn-login-submit" type="submit" disabled={loading} className="flex w-full items-center justify-center gap-2 rounded-lg bg-ink-950 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700 disabled:cursor-wait disabled:opacity-60">
            <span>{loading ? 'Verifying account…' : 'Continue to workspace'}</span>
            {!loading && <ArrowRight className="h-4 w-4" aria-hidden="true" />}
          </button>
        </form>

        <div className="my-7 flex items-center gap-3 text-xs font-semibold uppercase tracking-wider text-slate-400"><span className="h-px flex-1 bg-slate-200" />Local demo accounts<span className="h-px flex-1 bg-slate-200" /></div>

        <div className="grid gap-3 sm:grid-cols-2">
          <button id="preload-doctor-btn" type="button" onClick={() => useDemoAccount('doctor')} className="flex items-center gap-3 rounded-lg border border-slate-200 p-3 text-left transition hover:border-brand-200 hover:bg-brand-50">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-100 text-brand-700"><Stethoscope className="h-4 w-4" aria-hidden="true" /></span>
            <span><strong className="block text-sm text-slate-800">Doctor demo</strong><span className="text-xs text-slate-500">Analysis and records</span></span>
          </button>
          <button id="preload-admin-btn" type="button" onClick={() => useDemoAccount('admin')} className="flex items-center gap-3 rounded-lg border border-slate-200 p-3 text-left transition hover:border-brand-200 hover:bg-brand-50">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-slate-700"><UserCog className="h-4 w-4" aria-hidden="true" /></span>
            <span><strong className="block text-sm text-slate-800">Admin demo</strong><span className="text-xs text-slate-500">Local overview</span></span>
          </button>
        </div>

        <p className="mt-7 text-center text-sm text-slate-600">Need a local account? <Link to="/register" className="font-semibold text-brand-700 hover:text-brand-600">Create one</Link></p>
      </AuthShell>
    </div>
  );
};
