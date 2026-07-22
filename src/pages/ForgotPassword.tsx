import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Mail, ArrowLeft, CheckCircle } from 'lucide-react';

export const ForgotPassword: React.FC = () => {
  const [email, setEmail] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Simulate email dispatch
    setSubmitted(true);
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 bg-slate-50" id="forgot-password-page">
      <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-2xl border border-slate-100 shadow-sm">
        <div className="text-center space-y-3">
          <div className="inline-flex p-3.5 bg-blue-50 text-blue-600 rounded-2xl border border-blue-100/30">
            <Mail className="h-8 w-8 text-blue-600" />
          </div>
          <div>
            <h2 className="font-sans font-extrabold text-slate-900 tracking-tight text-2xl">
              Reset Security Credentials
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              Provide your work email address to receive password retrieval steps
            </p>
          </div>
        </div>

        {submitted ? (
          <div className="space-y-6 text-center py-4" id="success-display">
            <div className="bg-emerald-50 border border-emerald-100 p-4 rounded-xl flex items-start space-x-3 text-emerald-800 text-left text-xs">
              <CheckCircle className="h-5 w-5 text-emerald-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold block text-emerald-900 mb-0.5">Reset Instructions Dispatched</span>
                <p>If the account exists in the medical registry, instructions to update your security password have been delivered to <strong className="font-mono">{email}</strong>.</p>
              </div>
            </div>

            <Link
              to="/login"
              className="inline-flex items-center space-x-2 text-sm text-blue-600 font-semibold hover:underline"
            >
              <ArrowLeft className="h-4 w-4" />
              <span>Return to Login Screen</span>
            </Link>
          </div>
        ) : (
          <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
                Work Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 h-4.5 w-4.5" />
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/10 focus:border-blue-500 bg-white"
                  placeholder="physician@hospital.com"
                  id="reset-email"
                />
              </div>
            </div>

            <button
              type="submit"
              className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold shadow-sm hover:shadow transition-all flex items-center justify-center cursor-pointer mt-2"
              id="btn-reset-submit"
            >
              <span>Transmit Reset Link</span>
            </button>

            <div className="text-center">
              <Link
                to="/login"
                className="inline-flex items-center space-x-2 text-xs text-slate-500 hover:text-slate-800 font-medium"
              >
                <ArrowLeft className="h-3.5 w-3.5" />
                <span>Back to authentication portal</span>
              </Link>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};
