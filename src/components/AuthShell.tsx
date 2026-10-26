import React from 'react';
import { Activity, Database, Microscope, ShieldCheck } from 'lucide-react';

interface AuthShellProps {
  eyebrow: string;
  title: string;
  description: string;
  children: React.ReactNode;
}

export const AuthShell: React.FC<AuthShellProps> = ({ eyebrow, title, description, children }) => (
  <div className="min-h-[calc(100vh-7.5rem)] px-4 py-8 sm:px-6 lg:flex lg:items-center lg:py-12">
    <div className="mx-auto grid w-full max-w-6xl overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-[0_24px_70px_rgba(15,35,45,0.10)] lg:grid-cols-[0.9fr_1.1fr]">
      <aside className="relative hidden overflow-hidden bg-ink-950 p-10 text-white lg:flex lg:flex-col lg:justify-between" aria-label="Research system overview">
        <div className="absolute -right-24 -top-24 h-72 w-72 rounded-full border border-white/10 bg-brand-500/20" />
        <div className="relative">
          <div className="inline-flex h-11 w-11 items-center justify-center rounded-xl border border-white/15 bg-white/10">
            <Activity className="h-5 w-5 text-cyan-200" aria-hidden="true" />
          </div>
          <p className="mt-8 text-xs font-bold uppercase tracking-[0.18em] text-cyan-200">NAFLD research console</p>
          <h2 className="mt-3 max-w-sm text-3xl font-semibold leading-tight">Traceable ultrasound model evaluation, kept separate from clinical care.</h2>
          <p className="mt-4 max-w-md text-sm leading-6 text-slate-300">A local research workspace for the selected two-stage PyTorch cascade. Outputs remain human-reviewable and explicitly non-clinical.</p>
        </div>

        <div className="relative space-y-4 text-sm text-slate-200">
          <div className="flex gap-3"><Microscope className="mt-0.5 h-4 w-4 shrink-0 text-cyan-200" /><span>Single-frame research inference with locked preprocessing</span></div>
          <div className="flex gap-3"><Database className="mt-0.5 h-4 w-4 shrink-0 text-cyan-200" /><span>Versioned model and dataset provenance in every result</span></div>
          <div className="flex gap-3"><ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-cyan-200" /><span>Uploaded image retention disabled in the current local mode</span></div>
        </div>
      </aside>

      <main className="px-6 py-8 sm:px-10 sm:py-10 lg:px-14 lg:py-12">
        <div className="mb-8">
          <p className="eyebrow">{eyebrow}</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">{title}</h1>
          <p className="mt-3 max-w-lg text-sm leading-6 text-slate-600">{description}</p>
        </div>
        {children}
      </main>
    </div>
  </div>
);
