import React from 'react';
import { Scan } from '../types';
import { ProbabilityChart } from './ProbabilityChart';
import { ShieldAlert, CheckCircle, AlertTriangle, Calendar, User, Info } from 'lucide-react';

interface ResultCardProps {
  scan: Scan;
}

export const ResultCard: React.FC<ResultCardProps> = ({ scan }) => {
  const isNormal = scan.prediction === 'Normal';

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden" id={`result-card-${scan.id}`}>
      {/* Dynamic Header */}
      <div className={`px-6 py-5 flex items-center justify-between border-b ${
        isNormal ? 'bg-emerald-50/50 border-emerald-100/40' : 'bg-red-50/50 border-red-100/40'
      }`}>
        <div>
          <span className="text-[10px] uppercase tracking-wider font-bold text-slate-400">Diagnostic Response</span>
          <h3 className="text-xl font-bold text-slate-800">ML Model Output</h3>
        </div>
        
        <div className="flex items-center">
          {isNormal ? (
            <span className="inline-flex items-center px-4 py-1.5 rounded-full text-sm font-bold tracking-tight bg-emerald-100 text-emerald-800 shadow-sm">
              <CheckCircle className="h-4 w-4 mr-1.5" />
              Normal Liver
            </span>
          ) : (
            <span className="inline-flex items-center px-4 py-1.5 rounded-full text-sm font-bold tracking-tight bg-red-100 text-red-800 shadow-sm animate-pulse">
              <AlertTriangle className="h-4 w-4 mr-1.5" />
              Abnormal (Steatosis)
            </span>
          )}
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Demographics Recap */}
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

        {/* Confidence Meter */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Classification Confidence</span>
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
        </div>

        {/* Probability Breakdown Bar Chart */}
        <div className="pt-2">
          <ProbabilityChart probabilities={scan.probabilities} />
        </div>

        {/* Disclaimer Warning */}
        <div className="flex items-start space-x-3 bg-blue-50 border border-blue-100/50 p-3.5 rounded-xl text-blue-800" id="result-disclaimer">
          <ShieldAlert className="h-5 w-5 text-blue-600 shrink-0 mt-0.5" />
          <div className="text-xs leading-relaxed">
            <span className="font-bold block text-blue-900 mb-0.5">Clinical Decision Support System</span>
            <p>For clinical assistance only. Not a substitute for professional medical diagnosis. Please correlate findings with patient clinical presentation, history, and alternative laboratory diagnostic modalities.</p>
          </div>
        </div>
      </div>
    </div>
  );
};
