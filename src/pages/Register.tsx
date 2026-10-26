import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, Check, LockKeyhole, Mail, ShieldAlert, UserRound } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { AuthShell } from '../components/AuthShell';

export const Register: React.FC = () => {
  const { register, error, clearError } = useAuth();
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<'doctor' | 'admin'>('doctor');
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLocalError(null);
    clearError();
    if (!name || !email || !password) {
      setLocalError('Complete all fields before continuing.');
      return;
    }
    if (password.length < 6) {
      setLocalError('Use at least 6 characters for the local password.');
      return;
    }
    setLoading(true);
    try {
      await register(email, password, name, role);
      navigate('/');
    } catch (err: any) {
      setLocalError(err.message || 'Account creation failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div id="register-page">
      <AuthShell eyebrow="Local account setup" title="Create a research account" description="Accounts created here belong to this local prototype. Choose the minimum role needed for your work.">
        {(localError || error) && (
          <div className="mb-5 flex gap-3 rounded-lg border border-rose-200 bg-rose-50 p-3.5 text-sm text-rose-900" role="alert">
            <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <div><strong className="block font-semibold">Account not created</strong><span>{localError || error}</span></div>
          </div>
        )}

        <form className="space-y-5" onSubmit={handleSubmit}>
          <div><label className="field-label" htmlFor="register-name">Display name</label><div className="relative"><UserRound className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" /><input id="register-name" type="text" autoComplete="name" required value={name} onChange={(e) => setName(e.target.value)} className="field-control pl-10" placeholder="Research reviewer" /></div></div>
          <div><label className="field-label" htmlFor="register-email">Email address</label><div className="relative"><Mail className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" /><input id="register-email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} className="field-control pl-10" placeholder="name@example.com" /></div></div>
          <div><label className="field-label" htmlFor="register-password">Password</label><div className="relative"><LockKeyhole className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" /><input id="register-password" type="password" autoComplete="new-password" required value={password} onChange={(e) => setPassword(e.target.value)} className="field-control pl-10" placeholder="At least 6 characters" /></div></div>

          <fieldset>
            <legend className="field-label">Workspace role</legend>
            <div className="grid gap-3 sm:grid-cols-2">
              {(['doctor', 'admin'] as const).map((option) => (
                <button key={option} type="button" id={`role-${option}-select`} aria-pressed={role === option} onClick={() => setRole(option)} className={`relative rounded-lg border p-3 text-left transition ${role === option ? 'border-brand-500 bg-brand-50 ring-2 ring-brand-500/10' : 'border-slate-200 hover:border-slate-300'}`}>
                  <strong className="block text-sm capitalize text-slate-900">{option}</strong>
                  <span className="mt-1 block text-xs leading-5 text-slate-500">{option === 'doctor' ? 'Run analyses and review own records' : 'Review users and aggregate local records'}</span>
                  {role === option && <Check className="absolute right-3 top-3 h-4 w-4 text-brand-700" aria-hidden="true" />}
                </button>
              ))}
            </div>
          </fieldset>

          <button id="btn-register-submit" type="submit" disabled={loading} className="w-full rounded-lg bg-ink-950 px-4 py-3 text-sm font-semibold text-white transition hover:bg-brand-700 disabled:cursor-wait disabled:opacity-60">{loading ? 'Creating account…' : 'Create local account'}</button>
        </form>

        <Link to="/login" className="mt-7 flex items-center justify-center gap-2 text-sm font-semibold text-brand-700 hover:text-brand-600"><ArrowLeft className="h-4 w-4" />Back to sign in</Link>
      </AuthShell>
    </div>
  );
};
