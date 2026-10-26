import React, { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, ClipboardList, FlaskConical, LayoutDashboard, LogOut, UploadCloud } from 'lucide-react';

type ModelState = 'checking' | 'ready' | 'offline';

export const Navbar: React.FC = () => {
  const { user, logout, getAuthHeaders } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [modelState, setModelState] = useState<ModelState>('checking');

  useEffect(() => {
    if (!user) return;
    let active = true;

    const checkHealth = async () => {
      try {
        const response = await fetch('/api/model/health', { headers: getAuthHeaders() });
        const data = await response.json();
        if (active) setModelState(response.ok && data.ready === true ? 'ready' : 'offline');
      } catch {
        if (active) setModelState('offline');
      }
    };

    checkHealth();
    const timer = window.setInterval(checkHealth, 30_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [user]);

  if (!user) return null;

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const isActive = (path: string) => path === '/'
    ? location.pathname === '/'
    : location.pathname.startsWith(path);

  const navItems = user.role === 'doctor'
    ? [
        { to: '/', label: 'New analysis', icon: UploadCloud },
        { to: '/history', label: 'Records', icon: ClipboardList },
      ]
    : [{ to: '/', label: 'Overview', icon: LayoutDashboard }];

  const healthLabel = modelState === 'ready' ? 'Model ready' : modelState === 'offline' ? 'Model unavailable' : 'Checking model';

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200 bg-white/95 backdrop-blur" id="main-header">
      <div className="mx-auto flex h-16 max-w-[1440px] items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <Link to="/" className="flex min-w-0 items-center gap-3 rounded-md" aria-label="NAFLD research console home">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ink-950 text-cyan-200 shadow-sm">
            <Activity className="h-4.5 w-4.5" aria-hidden="true" />
          </span>
          <span className="min-w-0">
            <span className="block truncate text-sm font-bold tracking-tight text-ink-950 sm:text-base">NAFLD Research</span>
            <span className="hidden text-[10px] font-semibold uppercase tracking-[0.16em] text-brand-700 sm:block">Ultrasound model console</span>
          </span>
        </Link>

        <nav className="hidden items-center gap-1 rounded-lg bg-slate-100 p-1 md:flex" aria-label="Primary navigation" id="nav-menu">
          {navItems.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm font-semibold transition ${
                isActive(to)
                  ? 'bg-white text-ink-950 shadow-sm'
                  : 'text-slate-600 hover:bg-white/70 hover:text-slate-900'
              }`}
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
              {label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2 sm:gap-3">
          <div
            className={`hidden items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold lg:flex ${
              modelState === 'ready'
                ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                : modelState === 'offline'
                  ? 'border-rose-200 bg-rose-50 text-rose-800'
                  : 'border-slate-200 bg-slate-50 text-slate-600'
            }`}
            role="status"
            aria-live="polite"
          >
            <span className={`h-2 w-2 rounded-full ${modelState === 'ready' ? 'bg-emerald-500' : modelState === 'offline' ? 'bg-rose-500' : 'bg-slate-400'}`} />
            {healthLabel}
          </div>

          <div className="flex items-center gap-2 border-l border-slate-200 pl-3">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-brand-100 text-sm font-bold text-brand-700" aria-hidden="true">
              {user.name.charAt(0).toUpperCase()}
            </span>
            <span className="hidden max-w-36 sm:block">
              <span className="block truncate text-xs font-semibold text-slate-800">{user.name}</span>
              <span className="block text-[11px] capitalize text-slate-500">{user.role}</span>
            </span>
          </div>

          <button
            onClick={handleLogout}
            className="inline-flex h-9 w-9 items-center justify-center rounded-lg text-slate-500 transition hover:bg-slate-100 hover:text-slate-900"
            title="Sign out"
            aria-label="Sign out"
            id="btn-logout"
          >
            <LogOut className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="border-t border-slate-100 px-4 py-2 md:hidden">
        <nav className="mx-auto flex max-w-md gap-2" aria-label="Mobile navigation">
          {navItems.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className={`flex min-h-10 flex-1 items-center justify-center gap-2 rounded-lg text-sm font-semibold ${
                isActive(to) ? 'bg-brand-50 text-brand-700' : 'text-slate-600'
              }`}
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
              {label}
            </Link>
          ))}
        </nav>
      </div>

      {modelState === 'offline' && (
        <div className="border-t border-rose-200 bg-rose-50 px-4 py-2 text-center text-xs font-medium text-rose-800" role="alert">
          The selected inference service is unavailable. New analyses are temporarily disabled.
        </div>
      )}
    </header>
  );
};
