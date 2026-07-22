import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { ScanTable } from '../components/ScanTable';
import { Scan } from '../types';
import { ClipboardList, AlertCircle, Loader2 } from 'lucide-react';

export const PatientHistory: React.FC = () => {
  const { getAuthHeaders } = useAuth();
  const [scans, setScans] = useState<Scan[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchScans = async () => {
      try {
        setLoading(true);
        const headers = getAuthHeaders();
        const res = await fetch('/api/scans', {
          headers
        });

        if (!res.ok) {
          throw new Error('Failed to retrieve scan archives from server.');
        }

        const data = await res.json();
        setScans(data);
      } catch (err: any) {
        setError(err.message || 'Error fetching records.');
      } finally {
        setLoading(false);
      }
    };

    fetchScans();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" id="patient-history-page">
      <div className="space-y-2 mb-8">
        <h1 className="font-sans font-extrabold text-slate-900 tracking-tight text-3xl flex items-center space-x-3">
          <ClipboardList className="h-8 w-8 text-blue-600" />
          <span>Patient Diagnosis Archive</span>
        </h1>
        <p className="text-sm text-slate-500">
          Query and review historic liver ultrasound diagnostic records and predictions submitted under your clinical account.
        </p>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-24 space-y-3" id="loading-spinner">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
          <span className="text-sm text-slate-500 font-medium">Fetching secure records database...</span>
        </div>
      ) : error ? (
        <div className="bg-red-50 border border-red-100/50 p-5 rounded-2xl flex items-start space-x-3 text-red-800 text-sm max-w-xl mx-auto" id="error-display">
          <AlertCircle className="h-5 w-5 text-red-600 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold block text-red-900 mb-1">Database Sync Error</span>
            <p className="leading-relaxed">{error}</p>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <ScanTable scans={scans} />
        </div>
      )}
    </div>
  );
};
