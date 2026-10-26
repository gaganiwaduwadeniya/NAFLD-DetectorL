import React from 'react';
import { ArrowDown, Check, ChevronDown, Cpu, GitBranch, Hash, Info, ShieldAlert } from 'lucide-react';
import { Scan } from '../types';

const pct = (value: number | undefined) => value === undefined ? 'Unavailable' : `${(value * 100).toFixed(1)}%`;

const stageTheme: Record<number, { badge: string; bar: string; panel: string; label: string }> = {
  0: { badge: 'bg-emerald-100 text-emerald-800', bar: 'bg-emerald-500', panel: 'border-emerald-200 bg-emerald-50', label: 'Non-NAFLD' },
  1: { badge: 'bg-amber-100 text-amber-900', bar: 'bg-amber-500', panel: 'border-amber-200 bg-amber-50', label: 'Mild' },
  2: { badge: 'bg-orange-100 text-orange-900', bar: 'bg-orange-500', panel: 'border-orange-200 bg-orange-50', label: 'Moderate' },
  3: { badge: 'bg-rose-100 text-rose-900', bar: 'bg-rose-600', panel: 'border-rose-200 bg-rose-50', label: 'Severe' },
};

const ScoreBar: React.FC<{ label: string; value: number | undefined; color: string; emphasized?: boolean }> = ({ label, value, color, emphasized }) => (
  <div>
    <div className="mb-1.5 flex items-baseline justify-between gap-4 text-sm">
      <span className={emphasized ? 'font-semibold text-slate-900' : 'font-medium text-slate-600'}>{label}</span>
      <span className="font-mono text-sm font-semibold tabular-nums text-slate-800">{pct(value)}</span>
    </div>
    <div className="h-2.5 overflow-hidden rounded-full bg-slate-100" role="img" aria-label={`${label}: ${pct(value)}`}>
      <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.max(0, Math.min(1, value ?? 0)) * 100}%` }} />
    </div>
  </div>
);

export const ResearchV1Card: React.FC<{ scan: Scan }> = ({ scan }) => {
  const finalStage = scan.finalStage ?? 0;
  const theme = stageTheme[finalStage];
  const gatePositive = finalStage > 0;
  const conditional = scan.conditionalStageScores;
  const stageLabels: Record<string, string> = { '1': 'Mild', '2': 'Moderate', '3': 'Severe' };
  const stageColors: Record<string, string> = { '1': 'bg-amber-500', '2': 'bg-orange-500', '3': 'bg-rose-600' };

  return (
    <article className="surface-panel overflow-hidden rounded-xl" id={`result-card-${scan.id}`}>
      <header className={`border-b p-5 sm:p-6 ${theme.panel}`}>
        <div className="flex items-start justify-between gap-5">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-600">Research classification</p>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950" id="final-stage-label">{scan.finalStageLabel}</h2>
            <p className="mt-1 text-sm text-slate-600">Final cascade output for this single frame</p>
          </div>
          <span className={`shrink-0 rounded-full px-3 py-1.5 text-xs font-bold ${theme.badge}`}>Stage {finalStage}</span>
        </div>
      </header>

      <div className="space-y-6 p-5 sm:p-6">
        <dl className="grid grid-cols-3 gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm">
          <div className="min-w-0"><dt className="text-xs text-slate-500">Case alias</dt><dd className="mt-1 truncate font-semibold text-slate-900">{scan.patientName}</dd></div>
          <div><dt className="text-xs text-slate-500">Age / sex</dt><dd className="mt-1 font-semibold text-slate-900">{scan.patientAge} / {scan.patientGender}</dd></div>
          <div><dt className="text-xs text-slate-500">Input</dt><dd className="mt-1 font-semibold text-slate-900">Single frame</dd></div>
        </dl>

        <section aria-labelledby="gate-title">
          <div className="flex items-start justify-between gap-4">
            <div className="flex gap-3">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-ink-950 text-xs font-bold text-white">1</span>
              <div><h3 id="gate-title" className="text-sm font-semibold text-slate-900">NAFLD gate</h3><p className="mt-0.5 text-xs text-slate-500">DenseNet121 · mean of 5 selected fold members</p></div>
            </div>
            <span className="rounded-md bg-slate-100 px-2 py-1 font-mono text-xs text-slate-600">threshold {(scan.binaryThreshold ?? 0.5).toFixed(1)}</span>
          </div>
          <div className="mt-4 space-y-4">
            <ScoreBar label="Non-NAFLD score" value={scan.binaryScoreNonNafld} color="bg-emerald-500" emphasized={!gatePositive} />
            <ScoreBar label="NAFLD score" value={scan.binaryScoreNafld} color="bg-brand-600" emphasized={gatePositive} />
          </div>
          <div className={`mt-4 flex gap-3 rounded-lg border p-3.5 text-sm ${gatePositive ? 'border-brand-200 bg-brand-50 text-brand-900' : 'border-emerald-200 bg-emerald-50 text-emerald-900'}`}>
            <GitBranch className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <p><strong>Gate decision:</strong> {gatePositive ? `NAFLD score ${pct(scan.binaryScoreNafld)} met the locked threshold, so grading was run.` : `NAFLD score ${pct(scan.binaryScoreNafld)} was below the locked threshold, so grading was not run.`}</p>
          </div>
        </section>

        <div className="flex justify-center text-slate-300" aria-hidden="true"><ArrowDown className="h-5 w-5" /></div>

        <section aria-labelledby="grade-title" className={`rounded-xl border p-4 sm:p-5 ${gatePositive ? 'border-slate-200 bg-white' : 'border-slate-200 bg-slate-50'}`}>
          <div className="flex gap-3">
            <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-xs font-bold ${gatePositive ? 'bg-brand-700 text-white' : 'bg-slate-200 text-slate-500'}`}>2</span>
            <div><h3 id="grade-title" className="text-sm font-semibold text-slate-900">Conditional steatosis grade</h3><p className="mt-0.5 text-xs text-slate-500">EfficientNet-B0 · mean of 5 selected fold members</p></div>
          </div>

          {scan.gradingPerformed && conditional ? (
            <div className="mt-5 space-y-4">
              {Object.entries(conditional).map(([key, value]) => <ScoreBar key={key} label={stageLabels[key]} value={value} color={stageColors[key]} emphasized={Number(key) === finalStage} />)}
              <p className="flex gap-2 text-xs leading-5 text-slate-500"><Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />These three scores are conditional on the gate selecting NAFLD and sum to approximately 100% within that branch.</p>
            </div>
          ) : (
            <div className="mt-4 flex items-center gap-3 rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-600"><Check className="h-4 w-4 shrink-0 text-emerald-600" /><span>Correctly skipped because the NAFLD gate was negative.</span></div>
          )}
        </section>

        <details className="group rounded-lg border border-slate-200" id="provenance-details">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-4 py-3 text-sm font-semibold text-slate-800">
            Technical provenance and supplementary values
            <ChevronDown className="h-4 w-4 transition group-open:rotate-180" aria-hidden="true" />
          </summary>
          <div className="space-y-5 border-t border-slate-200 px-4 py-4 text-sm text-slate-600">
            {scan.jointStageScores && (
              <section>
                <h4 className="font-semibold text-slate-800">Derived joint scores</h4>
                <p className="mt-1 text-xs leading-5 text-slate-500">Supplementary cascade products; not independently calibrated clinical probabilities.</p>
                <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {Object.entries(scan.jointStageScores).map(([key, value]) => <div key={key} className="rounded-lg bg-slate-50 p-3"><span className="block text-xs text-slate-500">Stage {key}</span><strong className="mt-1 block font-mono text-sm text-slate-800">{pct(value)}</strong></div>)}
                </div>
              </section>
            )}
            <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
              <div><dt className="text-xs text-slate-500">Model version</dt><dd className="mt-0.5 break-all font-mono text-xs text-slate-800">{scan.modelVersion}</dd></div>
              <div><dt className="text-xs text-slate-500">Contract</dt><dd className="mt-0.5 break-all font-mono text-xs text-slate-800">{scan.contractId}</dd></div>
              <div><dt className="text-xs text-slate-500">Binary architecture</dt><dd className="mt-0.5 text-slate-800">DenseNet121 × 5</dd></div>
              <div><dt className="text-xs text-slate-500">Staging architecture</dt><dd className="mt-0.5 text-slate-800">EfficientNet-B0 × 5</dd></div>
              <div><dt className="text-xs text-slate-500">Service inference</dt><dd className="mt-0.5 font-mono text-xs text-slate-800">{scan.serviceInferenceMs?.toFixed(1) ?? '—'} ms</dd></div>
              <div><dt className="text-xs text-slate-500">Source image retained</dt><dd className="mt-0.5 text-slate-800">{scan.imageStored ? 'Yes' : 'No'}</dd></div>
            </dl>
            <div className="space-y-2 border-t border-slate-200 pt-4 font-mono text-[11px] text-slate-500">
              <p className="flex gap-2 break-all"><Hash className="mt-0.5 h-3 w-3 shrink-0" />{scan.binaryManifestSha256}</p>
              <p className="flex gap-2 break-all"><Hash className="mt-0.5 h-3 w-3 shrink-0" />{scan.stagingManifestSha256}</p>
            </div>
          </div>
        </details>

        <div id="research-only-notice" className="flex gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm leading-5 text-amber-950">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <p><strong>Research use only.</strong> This internally evaluated model is not clinically validated. Do not use this output for autonomous diagnosis or treatment.</p>
        </div>
        <p className="flex items-center gap-2 text-xs text-slate-500"><Cpu className="h-3.5 w-3.5" />Exact selected PyTorch research-v1 ensemble</p>
      </div>
    </article>
  );
};
