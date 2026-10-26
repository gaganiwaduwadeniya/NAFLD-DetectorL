import React, { useEffect, useState } from 'react';
import { AlertCircle, ClipboardList, Loader2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { ScanTable } from '../components/ScanTable';
import { Scan } from '../types';

export const PatientHistory: React.FC = () => {
  const { getAuthHeaders } = useAuth();
  const [scans, setScans] = useState<Scan[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchScans = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/scans', { headers: getAuthHeaders() });
        if (!response.ok) throw new Error('Records could not be retrieved from the local store.');
        setScans(await response.json());
      } catch (err: any) {
        setError(err.message || 'Records could not be loaded.');
      } finally {
        setLoading(false);
      }
    };
    fetchScans();
  }, []);

  return (
    <div className="mx-auto w-full max-w-[1440px] px-4 py-7 sm:px-6 sm:py-9 lg:px-8" id="patient-history-page">
      <header className="mb-7 flex items-start gap-4 border-b border-slate-200 pb-6">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-brand-100 text-brand-700"><ClipboardList className="h-5 w-5" /></span>
        <div><p className="eyebrow">Local records</p><h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">Research analysis history</h1><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">Review outputs created under your account. These are research records, not patient diagnoses.</p></div>
      </header>

      {loading ? <div id="loading-spinner" className="flex items-center justify-center gap-3 py-24 text-sm text-slate-600" role="status"><Loader2 className="h-5 w-5 animate-spin text-brand-600" />Loading local records…</div>
        : error ? <div id="error-display" className="mx-auto flex max-w-xl gap-3 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900" role="alert"><AlertCircle className="mt-0.5 h-4 w-4 shrink-0" /><div><strong className="block">Unable to load records</strong><span>{error}</span></div></div>
          : <ScanTable scans={scans} />}
    </div>
  );
};
