import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, LogOut, User as UserIcon, Calendar, ClipboardList, Shield, UploadCloud } from 'lucide-react';

export const Navbar: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  if (!user) return null;

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm" id="main-header">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <Link to="/" className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center text-white shadow-sm">
                <Activity className="h-4 w-4 stroke-[2.5]" />
              </div>
              <div className="flex flex-col">
                <span className="font-sans font-bold text-slate-900 tracking-tight text-lg block leading-tight">
                  LIVER-AI
                </span>
                <span className="font-mono text-[9px] text-blue-600 tracking-wider uppercase font-semibold block">
                  NAFLD DIAGNOSTICS
                </span>
              </div>
            </Link>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex space-x-2" id="nav-menu">
            {user.role === 'doctor' && (
              <>
                <Link
                  to="/"
                  className={`px-3 py-2 rounded-md text-xs font-semibold transition-colors flex items-center space-x-1.5 ${
                    isActive('/') || isActive('/upload')
                      ? 'bg-blue-600/10 text-blue-600 border border-blue-500/20'
                      : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50'
                  }`}
                  id="nav-link-diagnostic"
                >
                  <UploadCloud className="h-4 w-4" />
                  <span>Diagnostic Desk</span>
                </Link>
                <Link
                  to="/history"
                  className={`px-3 py-2 rounded-md text-xs font-semibold transition-colors flex items-center space-x-1.5 ${
                    isActive('/history')
                      ? 'bg-blue-600/10 text-blue-600 border border-blue-500/20'
                      : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50'
                  }`}
                  id="nav-link-scans"
                >
                  <ClipboardList className="h-4 w-4" />
                  <span>Patient Scans</span>
                </Link>
              </>
            )}

            {user.role === 'admin' && (
              <Link
                to="/"
                className={`px-3 py-2 rounded-md text-xs font-semibold transition-colors flex items-center space-x-1.5 ${
                  isActive('/') || isActive('/admin')
                    ? 'bg-blue-600/10 text-blue-600 border border-blue-500/20'
                    : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50'
                }`}
                id="nav-link-admin"
              >
                <Shield className="h-4 w-4" />
                <span>Admin Dashboard</span>
              </Link>
            )}
          </nav>

          {/* User Info & Status Badge */}
          <div className="flex items-center space-x-4">
            <div className="hidden lg:flex items-center gap-2 px-3 py-1 bg-green-50 text-green-700 border border-green-100 rounded-full text-[11px] font-semibold">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span> 
              ML Service Online
            </div>

            <div className="w-px h-6 bg-slate-200 hidden lg:block"></div>

            <div className="flex items-center space-x-3 bg-slate-50 px-3 py-1.5 rounded-full border border-slate-100">
              <div className="h-7 w-7 rounded-full bg-slate-700 text-white flex items-center justify-center font-bold text-xs">
                {user.name.charAt(0).toUpperCase()}
              </div>
              <div className="text-left hidden sm:block">
                <span className="block text-xs font-semibold text-slate-800 leading-none">
                  {user.name}
                </span>
                <span className="block text-[9px] text-slate-500 font-medium leading-none mt-0.5">
                  {user.role === 'admin' ? 'System Administrator' : 'Senior Hepatologist'}
                </span>
              </div>
            </div>

            <button
              onClick={handleLogout}
              className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors cursor-pointer"
              title="Logout System"
              id="btn-logout"
            >
              <LogOut className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
