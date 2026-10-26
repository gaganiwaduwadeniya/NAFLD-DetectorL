import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Info } from 'lucide-react';
import { AuthShell } from '../components/AuthShell';

export const ForgotPassword: React.FC = () => (
  <div id="forgot-password-page">
    <AuthShell eyebrow="Account recovery" title="Password recovery is local" description="This research prototype does not send email or connect to an identity provider.">
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-5 text-sm leading-6 text-amber-950" id="recovery-information">
        <div className="flex gap-3"><Info className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" /><div><h2 className="font-semibold">No reset message will be sent</h2><p className="mt-1">For this local demonstration, return to the sign-in page and use a demo account, or ask the local system owner to recreate your research account.</p></div></div>
      </div>
      <Link to="/login" className="mt-7 flex w-full items-center justify-center gap-2 rounded-lg bg-ink-950 px-4 py-3 text-sm font-semibold text-white transition hover:bg-brand-700"><ArrowLeft className="h-4 w-4" aria-hidden="true" />Return to sign in</Link>
    </AuthShell>
  </div>
);
