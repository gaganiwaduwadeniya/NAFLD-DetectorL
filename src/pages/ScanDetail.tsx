import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { ResultCard } from '../components/ResultCard';
import { Scan } from '../types';
import { ArrowLeft, Loader2, AlertCircle, Eye, Clock, Cpu, Ruler } from 'lucide-react';

export const ScanDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { getAuthHeaders, user } = useAuth();

  const [scan, setScan] = useState<Scan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchScanDetails = async () => {
      if (!id) return;
      try {
        setLoading(true);
        const headers = getAuthHeaders();
        const res = await fetch(`/api/scans/${id}`, { headers });

        if (!res.ok) {
          throw new Error('Requested patient scan profile could not be found or retrieved.');
        }

        const data = await res.json();
        setScan(data);
      } catch (err: any) {
        setError(err.message || 'Error loading scan record.');
      } finally {
        setLoading(false);
      }
    };

    fetchScanDetails();
  }, [id]);

  const handleBack = () => {
    if (user?.role === 'admin') {
      navigate('/');
    } else {
      navigate('/history');
    }
  };

  const formatDate = (isoStr: string) => {
    try {
      return new Date(isoStr).toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch (e) {
      return isoStr;
    }
  };

  // Derive resolution string from scan metadata
  const resolutionLabel = (s: Scan): string => {
    if (s.imageWidth && s.imageHeight) {
      return `${s.imageWidth} × ${s.imageHeight} px`;
    }
    return 'Unknown';
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" id="scan-detail-page">
      {/* Back Button & Title block */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
        <button
          onClick={handleBack}
          className="inline-flex items-center space-x-2 px-3 py-1.5 border border-slate-200 hover:border-blue-500 hover:bg-blue-50/20 text-slate-600 hover:text-blue-600 rounded-xl text-xs font-semibold transition-all cursor-pointer w-fit"
          id="btn-scan-detail-back"
        >
          <ArrowLeft className="h-4 w-4" />
          <span>Return to Dashboard</span>
        </button>

        {scan && (
          <div className="text-right text-xs text-slate-500 font-mono">
            Scan Timestamp: {formatDate(scan.timestamp)}
          </div>
        )}
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-24 space-y-3" id="loading-spinner">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
          <span className="text-sm text-slate-500 font-medium">Loading scan record…</span>
        </div>
      ) : error ? (
        <div className="bg-red-50 border border-red-100/50 p-5 rounded-2xl flex items-start space-x-3 text-red-800 text-sm max-w-xl mx-auto" id="error-display">
          <AlertCircle className="h-5 w-5 text-red-600 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold block text-red-900 mb-1">Scan Not Found</span>
            <p className="leading-relaxed">{error}</p>
          </div>
        </div>
      ) : !scan ? (
        <div className="text-center text-slate-500 py-12" id="not-found-display">
          Scan record not found.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left: Ultrasound Image Viewer */}
          <div className="lg:col-span-7 bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-sm" id="ultrasound-viewer">
            <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between bg-slate-50/50">
              <div className="flex items-center space-x-2">
                <Eye className="h-4 w-4 text-blue-600" />
                <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">Ultrasound Viewer</span>
              </div>
            </div>

            {/* Main Image Frame */}
            <div className="bg-slate-950 flex items-center justify-center border-b border-slate-200 min-h-[350px]">
              <img
                src={scan.imageUrl}
                alt="Patient Liver Ultrasound Scan"
                className="w-full max-h-[500px] object-contain mx-auto"
                referrerPolicy="no-referrer"
              />
            </div>

            {/* Technical Metadata panel */}
            <div className="p-5 grid grid-cols-2 md:grid-cols-4 gap-4 bg-slate-50/40 text-xs font-mono">
              <div>
                <span className="block text-[10px] uppercase font-bold text-slate-400 mb-0.5">Assigned Clinician</span>
                <span className="text-slate-700 font-sans font-medium">{scan.doctorName}</span>
              </div>
              <div>
                <span className="block text-[10px] uppercase font-bold text-slate-400 mb-0.5">Clinician ID</span>
                <span className="text-slate-500 font-mono text-[10px] truncate block" title={scan.doctorId}>{scan.doctorId}</span>
              </div>
              <div>
                <span className="block text-[10px] uppercase font-bold text-slate-400 mb-0.5 flex items-center gap-1">
                  <Ruler className="h-3 w-3 inline" /> Resolution
                </span>
                <span className="text-slate-500 font-mono text-[10px]">{resolutionLabel(scan)}</span>
              </div>
              <div>
                <span className="block text-[10px] uppercase font-bold text-slate-400 mb-0.5">Record ID</span>
                <span className="text-slate-500 font-mono text-[10px] truncate block" title={scan.id}>{scan.id}</span>
              </div>
            </div>

            {/* V3 Inference metadata footer */}
            {scan.schemaVersion === 3 && (
              <div className="px-5 py-3 border-t border-slate-100 bg-slate-50/60 flex flex-wrap gap-4 text-[10px] font-mono text-slate-500">
                {scan.inferenceLatencyMs !== undefined && (
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    Latency: {scan.inferenceLatencyMs} ms
                  </span>
                )}
                {scan.modelVersion && (
                  <span className="flex items-center gap-1">
                    <Cpu className="h-3 w-3" />
                    Model: {scan.modelVersion}
                  </span>
                )}
                {scan.inputMode && (
                  <span>Mode: {scan.inputMode}</span>
                )}
              </div>
            )}
          </div>

          {/* Right: Diagnosis Result */}
          <div className="lg:col-span-5">
            <ResultCard scan={scan} />
          </div>
        </div>
      )}
    </div>
  );
};
