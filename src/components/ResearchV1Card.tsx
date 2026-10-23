import React from 'react';
import { Beaker, Cpu, Hash, Layers, ShieldAlert } from 'lucide-react';
import { Scan } from '../types';

const pct = (value: number | undefined) => value === undefined ? 'Unavailable' : `${(value * 100).toFixed(1)}%`;

const scoreRows = (
  scores: Record<string, number> | undefined,
  labels: Record<string, string>,
  color: string,
) => scores ? Object.entries(scores).map(([key, value]) => (
  <div key={key} className="space-y-1">
    <div className="flex justify-between text-xs">
      <span className="font-medium text-slate-600">{labels[key]}</span>
      <span className="font-mono font-bold text-slate-700">{pct(value)}</span>
    </div>
    <div className="h-2 overflow-hidden rounded-full bg-slate-100">
      <div className={`h-full rounded-full ${color}`} style={{ width: `${value * 100}%` }} />
    </div>
  </div>
)) : null;

export const ResearchV1Card: React.FC<{ scan: Scan }> = ({ scan }) => {
  const positive = (scan.finalStage ?? 0) > 0;
  const stageLabels = { '1': 'Stage 1 Mild', '2': 'Stage 2 Moderate', '3': 'Stage 3 Severe' };
  const jointLabels = { '0': 'Stage 0 (derived)', '1': 'Stage 1 (derived)', '2': 'Stage 2 (derived)', '3': 'Stage 3 (derived)' };

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm" id={`result-card-${scan.id}`}>
      <div className={`border-b px-6 py-5 ${positive ? 'border-amber-100 bg-amber-50/70' : 'border-emerald-100 bg-emerald-50/70'}`}>
        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">PyTorch research cascade output</span>
        <h2 className="mt-1 text-lg font-extrabold text-slate-900" id="final-stage-label">{scan.finalStageLabel}</h2>
        <p className="mt-1 text-xs text-slate-600">Model output only; not a diagnosis or clinical recommendation.</p>
      </div>

      <div className="space-y-5 p-6">
        <section className="rounded-xl border border-slate-200 bg-slate-50 p-4" aria-label="case summary">
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div><span className="block text-[10px] font-semibold uppercase text-slate-400">Case</span><span className="font-semibold text-slate-800">{scan.patientName}</span></div>
            <div><span className="block text-[10px] font-semibold uppercase text-slate-400">Age / sex</span><span className="font-semibold text-slate-800">{scan.patientAge} / {scan.patientGender}</span></div>
            <div><span className="block text-[10px] font-semibold uppercase text-slate-400">Input</span><span className="font-mono text-slate-600">single frame</span></div>
          </div>
        </section>

        <section className="space-y-3" aria-label="binary model scores">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600">DenseNet121 binary ensemble</h3>
            <span className="font-mono text-[10px] text-slate-500">NAFLD gate = {scan.binaryThreshold?.toFixed(1)}</span>
          </div>
          {scoreRows({ normal: scan.binaryScoreNonNafld ?? 0, nafld: scan.binaryScoreNafld ?? 0 }, { normal: 'Non-NAFLD', nafld: 'NAFLD' }, 'bg-blue-600')}
          <p className="text-[11px] leading-relaxed text-slate-500">
            EfficientNet-B0 grading was {scan.gradingPerformed ? 'performed because the NAFLD score met the locked 0.5 gate.' : 'skipped because the NAFLD score was below the locked 0.5 gate.'}
          </p>
        </section>

        {scan.gradingPerformed && scan.conditionalStageScores && (
          <section className="space-y-3 rounded-xl border border-violet-200 bg-violet-50/40 p-4" aria-label="conditional staging scores">
            <div className="flex items-center gap-2"><Layers className="h-4 w-4 text-violet-600" /><h3 className="text-xs font-bold uppercase tracking-wider text-violet-900">EfficientNet-B0 conditional stage scores</h3></div>
            {scoreRows(scan.conditionalStageScores, stageLabels, 'bg-violet-600')}
            <p className="text-[11px] text-violet-800">These scores are conditional on the binary gate selecting NAFLD.</p>
          </section>
        )}

        <details className="rounded-xl border border-slate-200 bg-white p-4" id="provenance-details">
          <summary className="cursor-pointer text-xs font-bold uppercase tracking-wider text-slate-700">Provenance and derived values</summary>
          <div className="mt-4 space-y-4 text-xs text-slate-600">
            {scan.jointStageScores && (
              <div className="space-y-2">
                <p className="font-semibold text-slate-700">Derived joint scores (supplementary; not independently calibrated)</p>
                {scoreRows(scan.jointStageScores, jointLabels, 'bg-slate-500')}
              </div>
            )}
            <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <div><dt className="text-[10px] uppercase text-slate-400">Model version</dt><dd className="font-mono">{scan.modelVersion}</dd></div>
              <div><dt className="text-[10px] uppercase text-slate-400">Contract</dt><dd className="font-mono break-all">{scan.contractId}</dd></div>
              <div><dt className="text-[10px] uppercase text-slate-400">Binary architecture</dt><dd className="font-mono">{scan.binaryArchitecture} (5 members)</dd></div>
              <div><dt className="text-[10px] uppercase text-slate-400">Staging architecture</dt><dd className="font-mono">{scan.stagingArchitecture} (5 members)</dd></div>
              <div><dt className="text-[10px] uppercase text-slate-400">Service latency</dt><dd className="font-mono">{scan.serviceInferenceMs?.toFixed(1)} ms</dd></div>
              <div><dt className="text-[10px] uppercase text-slate-400">Image retained</dt><dd className="font-mono">{scan.imageStored ? 'yes' : 'no'}</dd></div>
            </dl>
            <div className="space-y-1 border-t border-slate-100 pt-3 font-mono text-[10px] break-all">
              <p className="flex gap-1"><Hash className="h-3 w-3 shrink-0" /> Binary manifest: {scan.binaryManifestSha256}</p>
              <p className="flex gap-1"><Hash className="h-3 w-3 shrink-0" /> Staging manifest: {scan.stagingManifestSha256}</p>
            </div>
          </div>
        </details>

        <div className="flex items-start gap-3 rounded-xl border border-violet-200 bg-violet-50 p-4 text-xs text-violet-900" id="research-only-notice">
          <Beaker className="mt-0.5 h-4 w-4 shrink-0" />
          <p><strong>Research use only.</strong> This internally evaluated model is not clinically validated. Scores are not calibrated clinical probabilities and require qualified human interpretation.</p>
        </div>
        <div className="flex items-start gap-3 rounded-xl border border-blue-100 bg-blue-50 p-4 text-xs text-blue-900">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
          <p>One uploaded frame may not represent a complete ultrasound examination. Do not use this output for autonomous diagnosis or treatment.</p>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-slate-400"><Cpu className="h-3 w-3" /> Exact five-member DenseNet121 and five-member EfficientNet-B0 ensembles</div>
      </div>
    </div>
  );
};
