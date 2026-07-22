import React, { createContext, useContext, useState, useEffect } from 'react';
import { User, AuthState } from '../types';

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string, role: 'doctor' | 'admin') => Promise<void>;
  logout: () => void;
  clearError: () => void;
  getAuthHeaders: () => Record<string, string>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<AuthState>({
    user: null,
    token: null,
    loading: true,
    error: null,
  });

  // Verify and fetch profile on app mount
  useEffect(() => {
    const initAuth = async () => {
      const storedToken = localStorage.getItem('nafld_token');
      const storedUser = localStorage.getItem('nafld_user');

      if (storedToken && storedUser) {
        try {
          const parsedUser = JSON.parse(storedUser);
          
          // Verify with backend
          const res = await fetch('/api/auth/me', {
            headers: {
              'Authorization': `Bearer ${storedToken}`
            }
          });

          if (res.ok) {
            const data = await res.json();
            setState({
              user: data.user,
              token: storedToken,
              loading: false,
              error: null
            });
            localStorage.setItem('nafld_user', JSON.stringify(data.user));
            return;
          }
        } catch (e) {
          console.error('Error verifying auth token:', e);
        }
      }

      // If we reach here, token was missing or invalid
      localStorage.removeItem('nafld_token');
      localStorage.removeItem('nafld_user');
      setState({
        user: null,
        token: null,
        loading: false,
        error: null
      });
    };

    initAuth();
  }, []);

  const login = async (email: string, password: string) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Authentication failed');
      }

      localStorage.setItem('nafld_token', data.token);
      localStorage.setItem('nafld_user', JSON.stringify(data.user));

      setState({
        user: data.user,
        token: data.token,
        loading: false,
        error: null
      });
    } catch (err: any) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err.message || 'Login failed'
      }));
      throw err;
    }
  };

  const register = async (email: string, password: string, name: string, role: 'doctor' | 'admin') => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, name, role })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Registration failed');
      }

      localStorage.setItem('nafld_token', data.token);
      localStorage.setItem('nafld_user', JSON.stringify(data.user));

      setState({
        user: data.user,
        token: data.token,
        loading: false,
        error: null
      });
    } catch (err: any) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err.message || 'Registration failed'
      }));
      throw err;
    }
  };

  const logout = () => {
    localStorage.removeItem('nafld_token');
    localStorage.removeItem('nafld_user');
    setState({
      user: null,
      token: null,
      loading: false,
      error: null
    });
  };

  const clearError = () => {
    setState(prev => ({ ...prev, error: null }));
  };

  const getAuthHeaders = () => {
    return state.token ? { 'Authorization': `Bearer ${state.token}` } : {};
  };

  return (
    <AuthContext.Provider value={{ ...state, login, register, logout, clearError, getAuthHeaders }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
