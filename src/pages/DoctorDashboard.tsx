import React, { useEffect, useState } from 'react';
import { Activity, ArrowRight, CheckCircle2, FileSearch, FlaskConical, ImageOff, Info, RefreshCw, ShieldAlert, Sparkles, WifiOff } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { ImageUploader } from '../components/ImageUploader';
import { ResultCard } from '../components/ResultCard';
import { Scan } from '../types';

const ModelFlowPrimer = () => (
  <section className="surface-panel rounded-xl p-5 sm:p-6" aria-labelledby="model-flow-title">
    <div className="flex items-start justify-between gap-4">
      <div>
        <p className="eyebrow">Output structure</p>
        <h2 id="model-flow-title" className="mt-1 text-xl font-semibold text-ink-950">One gate, then conditional grading</h2>
      </div>
      <FlaskConical className="h-5 w-5 shrink-0 text-brand-600" aria-hidden="true" />
    </div>

    <div className="mt-6 space-y-3">
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="flex gap-3">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-ink-950 text-xs font-bold text-white">1</span>
          <div><h3 className="text-sm font-semibold text-slate-900">DenseNet121 gate</h3><p className="mt-1 text-sm leading-5 text-slate-600">Five fold models are averaged to choose Non-NAFLD or NAFLD at the locked 0.5 threshold.</p></div>
        </div>
      </div>
      <div className="ml-3 h-4 border-l-2 border-dashed border-slate-300" aria-hidden="true" />
      <div className="rounded-lg border border-brand-200 bg-brand-50 p-4">
        <div className="flex gap-3">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-brand-700 text-xs font-bold text-white">2</span>
          <div><h3 className="text-sm font-semibold text-slate-900">EfficientNet-B0 grade</h3><p className="mt-1 text-sm leading-5 text-slate-600">Only a positive NAFLD gate continues to conditional Mild, Moderate, or Severe grading.</p></div>
        </div>
      </div>
    </div>

    <div className="mt-5 grid grid-cols-3 gap-2" aria-label="Possible NAFLD grade outputs">
      <div className="rounded-md bg-amber-50 px-2 py-2 text-center text-xs font-semibold text-amber-800">Mild</div>
      <div className="rounded-md bg-orange-50 px-2 py-2 text-center text-xs font-semibold text-orange-800">Moderate</div>
      <div className="rounded-md bg-rose-50 px-2 py-2 text-center text-xs font-semibold text-rose-800">Severe</div>
    </div>

    <div className="mt-5 flex gap-3 border-t border-slate-200 pt-5 text-sm leading-5 text-slate-600">
      <Info className="mt-0.5 h-4 w-4 shrink-0 text-brand-600" aria-hidden="true" />
      <p>Scores are model outputs, not calibrated clinical probabilities. A single screenshot does not represent a complete ultrasound examination.</p>
    </div>
  </section>
);

export const DoctorDashboard: React.FC = () => {
  const { getAuthHeaders } = useAuth();
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState<'Male' | 'Female' | 'Other'>('Male');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isModelUnavailable, setIsModelUnavailable] = useState(false);
  const [scanResult, setScanResult] = useState<Scan | null>(null);
  const [explanationUrl, setExplanationUrl] = useState<string | null>(null);
  const [explanationMeta, setExplanationMeta] = useState<string | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);

  useEffect(() => () => {
    if (explanationUrl) URL.revokeObjectURL(explanationUrl);
  }, [explanationUrl]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setIsModelUnavailable(false);
    if (!selectedFile) return setError('Add one ultrasound image before running the model.');
    if (!patientName.trim()) return setError('Enter a de-identified research case name.');
    if (!patientAge || Number.isNaN(Number(patientAge)) || Number(patientAge) <= 0 || Number(patientAge) > 120) return setError('Enter an age between 1 and 120.');

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('patientName', patientName.trim());
      formData.append('patientAge', patientAge);
      formData.append('patientGender', patientGender);
      const response = await fetch('/api/predict', { method: 'POST', headers: getAuthHeaders(), body: formData });
      const data = await response.json();
      if (!response.ok) {
        if (response.status === 503) {
          setIsModelUnavailable(true);
          return;
        }
        throw new Error(data.error || 'The analysis request could not be completed.');
      }
      setScanResult(data);
    } catch (err: any) {
      setError(err.message || 'The analysis request could not be completed.');
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
    setExplanationMeta(null);
    setExplanationError(null);
    if (explanationUrl) URL.revokeObjectURL(explanationUrl);
    setExplanationUrl(null);
  };

  const handleExplain = async () => {
    if (!selectedFile || scanResult?.schemaVersion !== 4) return;
    setExplanationLoading(true);
    setExplanationError(null);
    try {
      const form = new FormData();
      form.append('image', selectedFile);
      const response = await fetch('/api/explain', { method: 'POST', headers: getAuthHeaders(), body: form });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.error || 'Grad-CAM generation failed.');
      }
      const blob = await response.blob();
      if (explanationUrl) URL.revokeObjectURL(explanationUrl);
      setExplanationUrl(URL.createObjectURL(blob));
      setExplanationMeta(response.headers.get('x-nafld-explanation'));
    } catch (err: any) {
      setExplanationError(err.message || 'Grad-CAM generation failed.');
    } finally {
      setExplanationLoading(false);
    }
  };

  return (
    <div className="mx-auto w-full max-w-[1440px] px-4 py-7 sm:px-6 sm:py-9 lg:px-8" id="doctor-dashboard">
      <header className="mb-7 flex flex-col gap-4 border-b border-slate-200 pb-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="eyebrow">New analysis</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">Classify a liver ultrasound frame</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">Enter de-identified case information, add one permitted image, and review the two-stage research output.</p>
        </div>
        <span className="inline-flex w-fit items-center gap-2 rounded-full border border-brand-200 bg-brand-50 px-3 py-1.5 text-xs font-semibold text-brand-700"><FlaskConical className="h-3.5 w-3.5" />Research use only</span>
      </header>

      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1.02fr)_minmax(420px,0.98fr)]">
        <form onSubmit={handleSubmit} className="surface-panel overflow-hidden rounded-xl" id="diagnostic-form" aria-busy={loading}>
          <div className="border-b border-slate-200 px-5 py-4 sm:px-6">
            <h2 className="text-base font-semibold text-ink-950">Analysis input</h2>
            <p className="mt-1 text-sm text-slate-500">Required fields are marked below. Do not enter direct identifiers.</p>
          </div>

          <div className="space-y-7 p-5 sm:p-6">
            {isModelUnavailable && <div id="model-unavailable-banner" className="flex gap-3 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900" role="alert"><WifiOff className="mt-0.5 h-4 w-4 shrink-0" /><div><strong className="block">Model service unavailable</strong><span>The request was not saved. Check the local inference service and try again.</span></div></div>}
            {error && <div id="form-error-banner" className="flex gap-3 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900" role="alert"><ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" /><div><strong className="block">Review the input</strong><span>{error}</span></div></div>}

            <section aria-labelledby="case-details-title">
              <div className="mb-4 flex items-center gap-3"><span className="flex h-7 w-7 items-center justify-center rounded-md bg-ink-950 text-xs font-bold text-white">1</span><div><h3 id="case-details-title" className="text-sm font-semibold text-slate-900">Case details</h3><p className="text-xs text-slate-500">Use a study alias rather than a patient name.</p></div></div>
              <div className="grid gap-4 sm:grid-cols-12">
                <div className="sm:col-span-6"><label className="field-label" htmlFor="patient-name">Research case alias <span className="text-rose-600">*</span></label><input id="patient-name" type="text" required autoComplete="off" value={patientName} disabled={loading || !!scanResult} onChange={(e) => setPatientName(e.target.value)} className="field-control" placeholder="e.g. Review case 001" /></div>
                <div className="sm:col-span-3"><label className="field-label" htmlFor="patient-age">Age <span className="text-rose-600">*</span></label><input id="patient-age" type="number" required min="1" max="120" value={patientAge} disabled={loading || !!scanResult} onChange={(e) => setPatientAge(e.target.value)} className="field-control" placeholder="Years" /></div>
                <div className="sm:col-span-3"><label className="field-label" htmlFor="patient-gender">Sex</label><select id="patient-gender" value={patientGender} disabled={loading || !!scanResult} onChange={(e) => setPatientGender(e.target.value as any)} className="field-control"><option value="Male">Male</option><option value="Female">Female</option><option value="Other">Other</option></select></div>
              </div>
            </section>

            <section aria-labelledby="image-input-title">
              <div className="mb-4 flex items-center gap-3"><span className="flex h-7 w-7 items-center justify-center rounded-md bg-ink-950 text-xs font-bold text-white">2</span><div><h3 id="image-input-title" className="text-sm font-semibold text-slate-900">Ultrasound input</h3><p className="text-xs text-slate-500">JPEG or PNG; processed in memory and not retained.</p></div></div>
              {scanResult ? <div id="image-not-retained-notice" className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600"><ImageOff className="mt-0.5 h-4 w-4 shrink-0" /><span>The source image was processed in memory and is not stored in this record.</span></div> : <ImageUploader selectedFile={selectedFile} onImageSelected={setSelectedFile} />}
            </section>

            {scanResult ? (
              <button type="button" onClick={handleReset} id="btn-reset-diagnostic" className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-800 transition hover:border-brand-300 hover:text-brand-700"><RefreshCw className="h-4 w-4" />Start another analysis</button>
            ) : (
              <button type="submit" id="btn-submit-diagnostic" disabled={loading || !selectedFile || !patientName.trim()} className="flex w-full items-center justify-center gap-2 rounded-lg bg-ink-950 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300">
                {loading ? <><Activity className="h-4 w-4 animate-spin" /><span>Running 10-member cascade…</span></> : <><span>Run research analysis</span><ArrowRight className="h-4 w-4" /></>}
              </button>
            )}
            {loading && <p className="text-center text-xs text-slate-500" role="status" aria-live="polite">CPU inference can take several seconds. Keep this page open.</p>}
          </div>
        </form>

        <div className="space-y-4">
          {scanResult ? (
            <div id="diagnosis-result-display" className="space-y-4">
              <ResultCard scan={scanResult} />
              {scanResult.schemaVersion === 4 && scanResult.explanationAvailable && (
                <section id="gradcam-panel" className="surface-panel rounded-xl p-5">
                  <div className="flex items-start justify-between gap-4"><div><h2 className="text-sm font-semibold text-slate-900">Qualitative attention map</h2><p className="mt-1 text-xs leading-5 text-slate-500">Generated from one selected ensemble member. This is not segmentation or proof of model reasoning.</p></div><Sparkles className="h-4 w-4 shrink-0 text-brand-600" /></div>
                  <button id="btn-generate-gradcam" type="button" onClick={handleExplain} disabled={explanationLoading} className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-brand-200 bg-brand-50 px-4 py-2.5 text-sm font-semibold text-brand-700 transition hover:bg-brand-100 disabled:opacity-60">{explanationLoading ? 'Generating map…' : 'Generate Grad-CAM'}</button>
                  {explanationError && <p className="mt-3 text-sm text-rose-700" role="alert">{explanationError}</p>}
                  {explanationUrl && <div className="mt-4 space-y-3"><img id="gradcam-image" src={explanationUrl} alt="Qualitative Grad-CAM overlay" className="w-full rounded-lg border border-slate-200" />{explanationMeta && <details className="text-xs text-slate-500"><summary className="cursor-pointer font-semibold text-slate-700">Explanation provenance</summary><code className="mt-2 block break-all rounded-lg bg-slate-50 p-3">{explanationMeta}</code></details>}</div>}
                </section>
              )}
              <div className="flex gap-3 rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900"><CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" /><div><strong className="block">Research record saved locally</strong><span>Scores and provenance were stored; source image bytes were not retained.</span></div></div>
            </div>
          ) : <ModelFlowPrimer />}

          {!scanResult && <div className="flex gap-3 rounded-lg border border-slate-200 bg-white/70 p-4 text-sm text-slate-600"><FileSearch className="mt-0.5 h-4 w-4 shrink-0 text-brand-600" /><p>The result view will preserve the distinction between gate scores, conditional grade scores, and supplementary derived joint scores.</p></div>}
        </div>
      </div>
    </div>
  );
};
