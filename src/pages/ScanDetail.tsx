import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AlertCircle, ArrowLeft, Clock3, FileLock2, ImageOff, Loader2, Ruler, UserRound } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { ResultCard } from '../components/ResultCard';
import { Scan } from '../types';

const formatDate = (value: string) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: 'long', timeStyle: 'medium' }).format(date);
};

export const ScanDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { getAuthHeaders, user } = useAuth();
  const [scan, setScan] = useState<Scan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      if (!id) return;
      try {
        const response = await fetch(`/api/scans/${id}`, { headers: getAuthHeaders() });
        if (!response.ok) throw new Error(response.status === 404 ? 'This record does not exist or is not available to your account.' : 'The record could not be retrieved.');
        setScan(await response.json());
      } catch (err: any) {
        setError(err.message || 'The record could not be retrieved.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [id]);

  const goBack = () => navigate(user?.role === 'admin' ? '/' : '/history');

  return (
    <div className="mx-auto w-full max-w-[1440px] px-4 py-7 sm:px-6 sm:py-9 lg:px-8" id="scan-detail-page">
      <header className="mb-7 border-b border-slate-200 pb-6">
        <button id="btn-scan-detail-back" type="button" onClick={goBack} className="inline-flex items-center gap-2 text-sm font-semibold text-brand-700 hover:text-brand-600"><ArrowLeft className="h-4 w-4" />{user?.role === 'admin' ? 'Back to overview' : 'Back to records'}</button>
        {scan && <div className="mt-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between"><div><p className="eyebrow">Saved research record</p><h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">{scan.patientName}</h1></div><p className="text-sm text-slate-500">Created {formatDate(scan.timestamp)}</p></div>}
      </header>

      {loading ? (
        <div id="loading-spinner" className="flex items-center justify-center gap-3 py-24 text-sm text-slate-600" role="status"><Loader2 className="h-5 w-5 animate-spin text-brand-600" />Loading research record…</div>
      ) : error ? (
        <div id="error-display" className="mx-auto flex max-w-xl gap-3 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900" role="alert"><AlertCircle className="mt-0.5 h-4 w-4 shrink-0" /><div><strong className="block">Record unavailable</strong><span>{error}</span></div></div>
      ) : !scan ? (
        <p id="not-found-display" className="py-16 text-center text-sm text-slate-500">Record not found.</p>
      ) : (
        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(430px,0.9fr)]">
          <section className="surface-panel overflow-hidden rounded-xl" id="ultrasound-viewer">
            <div className="border-b border-slate-200 px-5 py-4"><h2 className="text-base font-semibold text-slate-900">Input and record context</h2><p className="mt-1 text-sm text-slate-500">The current privacy setting stores geometry and model output, not image bytes.</p></div>
            {scan.imageUrl ? (
              <div className="flex min-h-80 items-center justify-center bg-slate-950"><img src={scan.imageUrl} alt="Stored liver ultrasound research input" className="max-h-[560px] w-full object-contain" referrerPolicy="no-referrer" /></div>
            ) : (
              <div id="history-image-not-retained" className="flex min-h-80 flex-col items-center justify-center bg-slate-950 px-8 text-center text-slate-300">
                <span className="flex h-12 w-12 items-center justify-center rounded-xl border border-white/10 bg-white/5"><ImageOff className="h-5 w-5" /></span>
                <h3 className="mt-4 text-sm font-semibold text-white">Source image not retained</h3>
                <p className="mt-2 max-w-sm text-sm leading-6 text-slate-400">The model processed the upload in memory. Reconstructing or displaying that image from this record is not possible.</p>
              </div>
            )}
            <dl className="grid gap-4 border-t border-slate-200 bg-slate-50 p-5 sm:grid-cols-2 lg:grid-cols-4">
              <div><dt className="flex items-center gap-1.5 text-xs text-slate-500"><UserRound className="h-3.5 w-3.5" />Reviewer</dt><dd className="mt-1 text-sm font-semibold text-slate-900">{scan.doctorName}</dd></div>
              <div><dt className="flex items-center gap-1.5 text-xs text-slate-500"><Ruler className="h-3.5 w-3.5" />Input geometry</dt><dd className="mt-1 font-mono text-xs text-slate-800">{scan.imageWidth && scan.imageHeight ? `${scan.imageWidth} × ${scan.imageHeight} px` : 'Unavailable'}</dd></div>
              <div><dt className="flex items-center gap-1.5 text-xs text-slate-500"><Clock3 className="h-3.5 w-3.5" />Request time</dt><dd className="mt-1 font-mono text-xs text-slate-800">{scan.totalRequestMs?.toFixed(1) ?? scan.inferenceLatencyMs ?? '—'} ms</dd></div>
              <div><dt className="flex items-center gap-1.5 text-xs text-slate-500"><FileLock2 className="h-3.5 w-3.5" />Record ID</dt><dd className="mt-1 truncate font-mono text-xs text-slate-800" title={scan.id}>{scan.id}</dd></div>
            </dl>
          </section>

          <ResultCard scan={scan} />
        </div>
      )}
    </div>
  );
};
