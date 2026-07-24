import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { ImageUploader } from '../components/ImageUploader';
import { ResultCard } from '../components/ResultCard';
import { Scan } from '../types';
import {
  Upload, User, Clipboard, Calendar, FileText,
  Activity, ShieldAlert, CheckCircle2, RefreshCw,
  WifiOff
} from 'lucide-react';

export const DoctorDashboard: React.FC = () => {
  const { getAuthHeaders } = useAuth();

  // Patient demographics state
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState<'Male' | 'Female' | 'Other'>('Male');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Status & output state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isModelUnavailable, setIsModelUnavailable] = useState(false);
  const [scanResult, setScanResult] = useState<Scan | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsModelUnavailable(false);

    if (!selectedFile) {
      setError('Please upload an ultrasound image before continuing.');
      return;
    }

    if (!patientName.trim()) {
      setError('Please provide the patient name.');
      return;
    }

    if (!patientAge || isNaN(Number(patientAge)) || Number(patientAge) <= 0) {
      setError('Please enter a valid age.');
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('patientName', patientName.trim());
      formData.append('patientAge', patientAge);
      formData.append('patientGender', patientGender);

      const headers = getAuthHeaders();
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { ...headers },
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        // 503 means the ML service is down — surface a specific banner
        if (res.status === 503) {
          setIsModelUnavailable(true);
          return;
        }
        throw new Error(data.error || 'Prediction request failed.');
      }

      setScanResult(data);
    } catch (err: any) {
      setError(err.message || 'An unexpected server error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPatientName('');
    setPatientAge('');
    setPatientGender('Male');
    setSelectedFile(null);
    setScanResult(null);
    setError(null);
    setIsModelUnavailable(false);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" id="doctor-dashboard">
      <div className="space-y-2 mb-8">
        <h1 className="font-sans font-extrabold text-slate-900 tracking-tight text-3xl">
          Liver Ultrasound Analysis
        </h1>
        <p className="text-sm text-slate-500">
          Upload a liver ultrasound image and enter patient demographics to run the NAFLD detection model.
          For research use only — results are not for clinical diagnosis.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Form and Upload Inputs */}
        <div className="lg:col-span-7 space-y-6">
          <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-6" id="diagnostic-form">
            <h2 className="text-base font-bold text-slate-800 flex items-center space-x-2">
              <Clipboard className="h-5 w-5 text-blue-600" />
              <span>Patient Details</span>
            </h2>

            {/* Model unavailable banner */}
            {isModelUnavailable && (
              <div className="bg-amber-50 border border-amber-200 p-4 rounded-xl flex items-start space-x-3 text-amber-800 text-xs" id="model-unavailable-banner">
                <WifiOff className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold block mb-0.5">Inference Service Unavailable</span>
                  <span>
                    The ML model service is not reachable. Ensure the Flask service is running and
                    <code className="mx-1 px-1 bg-amber-100 rounded font-mono">ML_SERVICE_URL</code>
                    is set correctly, then try again.
                  </span>
                </div>
              </div>
            )}

            {/* Validation error alert */}
            {error && (
              <div className="bg-red-50 border border-red-100/50 p-4 rounded-xl flex items-start space-x-3 text-red-800 text-xs" id="form-error-banner">
                <ShieldAlert className="h-4 w-4 text-red-600 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold block">Error</span>
                  <span>{error}</span>
                </div>
              </div>
            )}

            {/* Demographics row */}
            <div className="grid grid-cols-1 sm:grid-cols-12 gap-4">
              {/* Patient Name */}
              <div className="sm:col-span-6 space-y-1.5">
                <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
                  Patient Full Name
                </label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 h-4 w-4" />
                  <input
                    type="text"
                    required
                    value={patientName}
                    disabled={loading || !!scanResult}
                    onChange={(e) => setPatientName(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/10 focus:border-blue-500 bg-white disabled:bg-slate-50 disabled:text-slate-500"
                    placeholder="E.g. Margaret Carter"
                    id="patient-name"
                  />
                </div>
              </div>

              {/* Patient Age */}
              <div className="sm:col-span-3 space-y-1.5">
                <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
                  Age
                </label>
                <div className="relative">
                  <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 h-4 w-4" />
                  <input
                    type="number"
                    required
                    min="1"
                    max="120"
                    value={patientAge}
                    disabled={loading || !!scanResult}
                    onChange={(e) => setPatientAge(e.target.value)}
                    className="w-full pl-10 pr-3 py-2 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/10 focus:border-blue-500 bg-white disabled:bg-slate-50 disabled:text-slate-500"
                    placeholder="Yrs"
                    id="patient-age"
                  />
                </div>
              </div>

              {/* Patient Gender */}
              <div className="sm:col-span-3 space-y-1.5">
                <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
                  Sex
                </label>
                <select
                  value={patientGender}
                  disabled={loading || !!scanResult}
                  onChange={(e) => setPatientGender(e.target.value as any)}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/10 focus:border-blue-500 bg-white disabled:bg-slate-50 disabled:text-slate-500"
                  id="patient-gender"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>

            {/* Ultrasound Upload */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
                Ultrasound Image
              </label>
              {scanResult ? (
                <div className="border border-slate-200 rounded-xl overflow-hidden bg-slate-950 max-h-[350px]">
                  <img
                    src={scanResult.imageUrl}
                    alt="Analysed ultrasound scan"
                    className="w-full max-h-[350px] object-contain mx-auto"
                    referrerPolicy="no-referrer"
                  />
                </div>
              ) : (
                <ImageUploader selectedFile={selectedFile} onImageSelected={setSelectedFile} />
              )}
            </div>

            {/* Action buttons */}
            <div className="flex space-x-3 pt-2">
              {scanResult ? (
                <button
                  type="button"
                  onClick={handleReset}
                  className="w-full py-2.5 px-4 bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold text-sm rounded-xl transition-all flex items-center justify-center space-x-2 cursor-pointer"
                  id="btn-reset-diagnostic"
                >
                  <RefreshCw className="h-4 w-4" />
                  <span>New Analysis</span>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={loading || !selectedFile || !patientName.trim()}
                  className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-bold text-sm rounded-xl transition-all shadow-sm flex items-center justify-center space-x-2 disabled:bg-blue-300 disabled:cursor-not-allowed cursor-pointer"
                  id="btn-submit-diagnostic"
                >
                  {loading ? (
                    <>
                      <Activity className="h-4 w-4 animate-spin text-white" />
                      <span>Running model…</span>
                    </>
                  ) : (
                    <>
                      <Activity className="h-4 w-4" />
                      <span>Run NAFLD Detection</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </form>
        </div>

        {/* Right Column: Result Output */}
        <div className="lg:col-span-5">
          {scanResult ? (
            <div className="space-y-4 animate-fade-in" id="diagnosis-result-display">
              <ResultCard scan={scanResult} />

              {/* Success badge */}
              <div className="flex items-center space-x-2.5 bg-emerald-50 border border-emerald-100/50 text-emerald-800 p-3.5 rounded-xl text-xs">
                <CheckCircle2 className="h-5 w-5 text-emerald-600 shrink-0" />
                <div>
                  <span className="font-bold block">Scan Record Saved</span>
                  <span>Result has been saved to the database.</span>
                </div>
              </div>
            </div>
          ) : (
            <div
              className="bg-slate-50 border-2 border-dashed border-slate-200 rounded-2xl p-10 text-center h-full min-h-[400px] flex flex-col justify-center items-center space-y-4"
              id="empty-result-placeholder"
            >
              <div className="p-4 bg-white border border-slate-200 text-slate-400 rounded-2xl shadow-sm">
                <FileText className="h-8 w-8" />
              </div>
              <div className="max-w-xs">
                <h3 className="font-sans font-bold text-slate-700 text-sm">Output Panel</h3>
                <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                  Upload a liver ultrasound image and fill in patient details to run the detection model.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
