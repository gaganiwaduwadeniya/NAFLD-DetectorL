import React from 'react';
import { Scan } from '../types';
import { ProbabilityChart } from './ProbabilityChart';
import {
  ShieldAlert, CheckCircle, AlertTriangle, FlaskConical,
  Info, Layers, Cpu, Hash, Beaker
} from 'lucide-react';

interface ResultCardProps {
  scan: Scan;
}

// ---- helpers ----------------------------------------------------------------

function pct(v: number | undefined, decimals = 1): string {
  if (v === undefined || v === null) return '—';
  return (v * 100).toFixed(decimals) + '%';
}

function fmt(v: number | undefined, decimals = 4): string {
  if (v === undefined || v === null) return '—';
  return v.toFixed(decimals);
}

// ---- sub-components ---------------------------------------------------------

const LegacyHeader: React.FC<{ isNormal: boolean }> = ({ isNormal }) => (
  <div className={`px-6 py-5 flex items-center justify-between border-b ${
    isNormal ? 'bg-emerald-50/50 border-emerald-100/40' : 'bg-red-50/50 border-red-100/40'
  }`}>
    <div>
      <span className="text-[10px] uppercase tracking-wider font-bold text-slate-400">ML Model Output</span>
      <h3 className="text-xl font-bold text-slate-800">Diagnostic Response</h3>
    </div>
    <div>
      {isNormal ? (
        <span className="inline-flex items-center px-4 py-1.5 rounded-full text-sm font-bold bg-emerald-100 text-emerald-800 shadow-sm">
          <CheckCircle className="h-4 w-4 mr-1.5" />Normal
        </span>
      ) : (
        <span className="inline-flex items-center px-4 py-1.5 rounded-full text-sm font-bold bg-red-100 text-red-800 shadow-sm animate-pulse">
          <AlertTriangle className="h-4 w-4 mr-1.5" />Abnormal
        </span>
      )}
    </div>
  </div>
);

const V3Header: React.FC<{ finalLabel: string; isNafld: boolean }> = ({ finalLabel, isNafld }) => {
  const color = isNafld
    ? 'bg-red-50/50 border-red-100/40'
    : 'bg-emerald-50/50 border-emerald-100/40';
  const badge = isNafld
    ? 'bg-red-100 text-red-800'
    : 'bg-emerald-100 text-emerald-800';
  return (
    <div className={`px-6 py-5 flex items-center justify-between border-b ${color}`}>
      <div>
        <span className="text-[10px] uppercase tracking-wider font-bold text-slate-400">V3 Cascade Output</span>
        <h3 className="text-xl font-bold text-slate-800">Inference Result</h3>
      </div>
      <span className={`inline-flex items-center px-4 py-1.5 rounded-full text-sm font-bold shadow-sm ${badge} ${isNafld ? 'animate-pulse' : ''}`}>
        {isNafld ? <AlertTriangle className="h-4 w-4 mr-1.5" /> : <CheckCircle className="h-4 w-4 mr-1.5" />}
        {finalLabel}
      </span>
    </div>
  );
};

// ---- main component ---------------------------------------------------------

export const ResultCard: React.FC<ResultCardProps> = ({ scan }) => {
  const isV3 = scan.schemaVersion === 3;

  // Legacy path values
  const isNormal = scan.prediction === 'Normal';

  // V3 path values
  const isNafld = isV3 ? scan.binaryResult === 'NAFLD' : !isNormal;
  const finalLabel = isV3 ? (scan.finalLabel ?? scan.binaryResult ?? 'Unknown') : scan.prediction;

  return (
    <div
      className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden"
      id={`result-card-${scan.id}`}
    >
      {/* ── Header ── */}
      {isV3
        ? <V3Header finalLabel={finalLabel} isNafld={isNafld} />
        : <LegacyHeader isNormal={isNormal} />
      }

      <div className="p-6 space-y-5">
        {/* ── Demographics recap ── */}
        <div className="grid grid-cols-3 gap-4 bg-slate-50 p-4 rounded-xl border border-slate-200">
          <div>
            <span className="block text-[10px] uppercase font-semibold text-slate-400">Patient Name</span>
            <span className="font-sans font-semibold text-slate-800 text-sm truncate block">{scan.patientName}</span>
          </div>
          <div>
            <span className="block text-[10px] uppercase font-semibold text-slate-400">Age / Gender</span>
            <span className="font-sans font-semibold text-slate-800 text-sm block">
              {scan.patientAge} yrs • {scan.patientGender}
            </span>
          </div>
          <div>
            <span className="block text-[10px] uppercase font-semibold text-slate-400">Scan ID</span>
            <span className="font-mono text-slate-500 text-xs block truncate" title={scan.id}>{scan.id}</span>
          </div>
        </div>

        {/* ── V3 Binary Probabilities ── */}
        {isV3 ? (
          <div className="space-y-3">
            {/* Probability rows */}
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Ensemble Probability
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                threshold&nbsp;=&nbsp;{fmt(scan.binaryThreshold, 4)}
              </span>
            </div>

            {/* Non-NAFLD bar */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-medium text-slate-600">Non-NAFLD</span>
                <span className="font-mono font-bold text-emerald-700">{pct(scan.binaryProbNonNafld)}</span>
              </div>
              <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden relative">
                <div
                  className="h-full bg-emerald-500 rounded-full transition-all duration-700"
                  style={{ width: `${(scan.binaryProbNonNafld ?? 0) * 100}%` }}
                />
                {/* Threshold marker */}
                {scan.binaryThreshold !== undefined && (
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-slate-400/60"
                    style={{ left: `${scan.binaryThreshold * 100}%` }}
                    title={`Decision threshold: ${fmt(scan.binaryThreshold, 4)}`}
                  />
                )}
              </div>
            </div>

            {/* NAFLD bar */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-medium text-slate-600">NAFLD</span>
                <span className="font-mono font-bold text-red-700">{pct(scan.binaryProbNafld)}</span>
              </div>
              <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden relative">
                <div
                  className="h-full bg-red-500 rounded-full transition-all duration-700"
                  style={{ width: `${(scan.binaryProbNafld ?? 0) * 100}%` }}
                />
                {scan.binaryThreshold !== undefined && (
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-slate-400/60"
                    style={{ left: `${scan.binaryThreshold * 100}%` }}
                  />
                )}
              </div>
            </div>

            {/* Fold std-dev note */}
            {scan.binaryFoldStd !== undefined && (
              <p className="text-[10px] text-slate-400 font-mono mt-1">
                Fold score spread: ±{scan.binaryFoldStd.toFixed(4)}&ensp;
                <span className="text-slate-300">(inter-fold disagreement, not a calibrated confidence interval)</span>
              </p>
            )}

            {/* ProbabilityChart — V3 mode */}
            <div className="pt-1">
              <ProbabilityChart
                probabilities={scan.probabilities}
                binaryProbNafld={scan.binaryProbNafld}
                binaryProbNonNafld={scan.binaryProbNonNafld}
                binaryThreshold={scan.binaryThreshold}
              />
            </div>
          </div>
        ) : (
          /* ── Legacy confidence meter ── */
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Classification Confidence
              </span>
              <span className={`font-mono text-sm font-bold ${isNormal ? 'text-emerald-700' : 'text-red-700'}`}>
                {scan.confidence.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-1000 ${isNormal ? 'bg-emerald-500' : 'bg-red-500'}`}
                style={{ width: `${scan.confidence}%` }}
              />
            </div>
            <div className="pt-2">
              <ProbabilityChart probabilities={scan.probabilities} />
            </div>
          </div>
        )}

        {/* ── V3 Grading Section (only when performed) ── */}
        {isV3 && scan.gradingPerformed && (
          <div className="border border-amber-200 bg-amber-50/60 rounded-xl p-4 space-y-3">
            <div className="flex items-center space-x-2">
              <Layers className="h-4 w-4 text-amber-600" />
              <span className="text-xs font-bold text-amber-800 uppercase tracking-wider">
                Grading — Steatosis Severity
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-slate-800">{scan.gradingResult ?? '—'}</span>
              <span className="text-[10px] font-mono text-slate-400">
                threshold&nbsp;=&nbsp;{fmt(scan.gradingThreshold, 4)}
              </span>
            </div>

            {/* Grade 1 Mild bar */}
            {scan.gradingProbMild !== undefined && (
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-600 font-medium">Grade 1 Mild</span>
                  <span className="font-mono font-bold text-amber-700">{pct(scan.gradingProbMild)}</span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden relative">
                  <div
                    className="h-full bg-amber-400 rounded-full transition-all duration-700"
                    style={{ width: `${(scan.gradingProbMild ?? 0) * 100}%` }}
                  />
                  {scan.gradingThreshold !== undefined && (
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-slate-400/60"
                      style={{ left: `${scan.gradingThreshold * 100}%` }}
                    />
                  )}
                </div>
              </div>
            )}

            {/* Grade 2 Moderate/Severe bar */}
            {scan.gradingProbModerateSevere !== undefined && (
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-600 font-medium">Grade 2 Moderate/Severe</span>
                  <span className="font-mono font-bold text-orange-700">{pct(scan.gradingProbModerateSevere)}</span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden relative">
                  <div
                    className="h-full bg-orange-500 rounded-full transition-all duration-700"
                    style={{ width: `${(scan.gradingProbModerateSevere ?? 0) * 100}%` }}
                  />
                  {scan.gradingThreshold !== undefined && (
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-slate-400/60"
                      style={{ left: `${scan.gradingThreshold * 100}%` }}
                    />
                  )}
                </div>
              </div>
            )}

            {scan.gradingFoldStd !== undefined && (
              <p className="text-[10px] text-slate-400 font-mono">
                Grading fold spread: ±{scan.gradingFoldStd.toFixed(4)}
              </p>
            )}
          </div>
        )}

        {/* ── V3 Model Metadata ── */}
        {isV3 && (
          <div className="border border-slate-100 rounded-xl p-4 space-y-2 bg-slate-50/40">
            <div className="flex items-center space-x-2 mb-2">
              <Cpu className="h-4 w-4 text-slate-400" />
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Model Metadata</span>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
              <div>
                <span className="block text-[10px] text-slate-400 uppercase font-semibold">Model Version</span>
                <span className="font-mono text-slate-700">{scan.modelVersion ?? '—'}</span>
              </div>
              <div>
                <span className="block text-[10px] text-slate-400 uppercase font-semibold">Input Mode</span>
                <span className="font-mono text-slate-700">{scan.inputMode ?? '—'}</span>
              </div>
              <div>
                <span className="block text-[10px] text-slate-400 uppercase font-semibold">Schema Version</span>
                <span className="font-mono text-slate-700">{scan.schemaVersion}</span>
              </div>
              <div>
                <span className="block text-[10px] text-slate-400 uppercase font-semibold">Latency</span>
                <span className="font-mono text-slate-700">
                  {scan.inferenceLatencyMs !== undefined ? `${scan.inferenceLatencyMs} ms` : '—'}
                </span>
              </div>
              {scan.contractSha256 && (
                <div className="col-span-2">
                  <span className="block text-[10px] text-slate-400 uppercase font-semibold flex items-center space-x-1">
                    <Hash className="h-3 w-3 inline" /> Contract SHA-256
                  </span>
                  <span
                    className="font-mono text-slate-500 text-[10px] break-all"
                    title={scan.contractSha256}
                  >
                    {scan.contractSha256.slice(0, 20)}…{scan.contractSha256.slice(-8)}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Research-only notice (V3) ── */}
        {isV3 && scan.researchUseOnly && (
          <div className="flex items-start space-x-3 bg-violet-50 border border-violet-100/60 p-3.5 rounded-xl text-violet-800" id="research-only-notice">
            <Beaker className="h-4 w-4 text-violet-500 shrink-0 mt-0.5" />
            <div className="text-xs leading-relaxed">
              <span className="font-bold block text-violet-900 mb-0.5">Research Use Only</span>
              <p>
                This output is produced by an experimental model for research purposes and has not been validated
                for clinical diagnosis. Do not use as a substitute for professional medical assessment.
              </p>
            </div>
          </div>
        )}

        {/* ── Disclaimer (legacy and V3) ── */}
        <div
          className="flex items-start space-x-3 bg-blue-50 border border-blue-100/50 p-3.5 rounded-xl text-blue-800"
          id="result-disclaimer"
        >
          <ShieldAlert className="h-5 w-5 text-blue-600 shrink-0 mt-0.5" />
          <div className="text-xs leading-relaxed">
            <span className="font-bold block text-blue-900 mb-0.5">Decision Support Tool</span>
            <p>
              For investigational assistance only. Not a substitute for professional medical diagnosis.
              Please correlate findings with patient clinical presentation, history, and other diagnostic modalities.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
