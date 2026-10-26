import React, { useEffect, useState } from 'react';
import { AlertCircle, BarChart3, CheckCircle2, ClipboardList, Loader2, ShieldCheck, UsersRound } from 'lucide-react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useAuth } from '../context/AuthContext';
import { StatsCard } from '../components/StatsCard';
import { ScanTable } from '../components/ScanTable';
import { AdminStats, User } from '../types';

const formatDate = (value: string) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(date);
};

export const AdminDashboard: React.FC = () => {
  const { getAuthHeaders } = useAuth();
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const headers = getAuthHeaders();
        const [statsResponse, usersResponse] = await Promise.all([
          fetch('/api/dashboard/stats', { headers }),
          fetch('/api/admin/users', { headers }),
        ]);
        if (!statsResponse.ok || !usersResponse.ok) throw new Error('The local overview could not be loaded.');
        setStats(await statsResponse.json());
        setUsers(await usersResponse.json());
      } catch (err: any) {
        setError(err.message || 'The local overview could not be loaded.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  return (
    <div className="mx-auto w-full max-w-[1440px] px-4 py-7 sm:px-6 sm:py-9 lg:px-8" id="admin-dashboard-page">
      <header className="mb-7 flex flex-col gap-4 border-b border-slate-200 pb-6 sm:flex-row sm:items-end sm:justify-between">
        <div><p className="eyebrow">Local administration</p><h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">Research workspace overview</h1><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">Monitor local research records and accounts. These figures describe prototype usage, not clinical performance.</p></div>
        <span className="inline-flex w-fit items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600"><ShieldCheck className="h-3.5 w-3.5 text-brand-600" />Administrator view</span>
      </header>

      {loading ? (
        <div id="loading-spinner" className="flex items-center justify-center gap-3 py-24 text-sm text-slate-600" role="status"><Loader2 className="h-5 w-5 animate-spin text-brand-600" />Loading local overview…</div>
      ) : error ? (
        <div id="error-display" className="mx-auto flex max-w-xl gap-3 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900" role="alert"><AlertCircle className="mt-0.5 h-4 w-4 shrink-0" /><div><strong className="block">Unable to load overview</strong><span>{error}</span></div></div>
      ) : !stats ? (
        <p className="py-16 text-center text-sm text-slate-500">No overview data is available.</p>
      ) : (
        <div className="space-y-7">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatsCard title="Research records" value={stats.totalScans} icon={<ClipboardList className="h-5 w-5" />} colorClass="bg-brand-50 text-brand-700" description="Saved local outputs" />
            <StatsCard title="Non-NAFLD outputs" value={stats.normalCount} icon={<CheckCircle2 className="h-5 w-5" />} colorClass="bg-emerald-50 text-emerald-700" description={`${stats.totalScans ? ((stats.normalCount / stats.totalScans) * 100).toFixed(1) : '0.0'}% of records`} />
            <StatsCard title="NAFLD branch outputs" value={stats.abnormalCount} icon={<BarChart3 className="h-5 w-5" />} colorClass="bg-amber-50 text-amber-800" description={`${stats.totalScans ? ((stats.abnormalCount / stats.totalScans) * 100).toFixed(1) : '0.0'}% of records`} />
            <StatsCard title="Local accounts" value={users.length} icon={<UsersRound className="h-5 w-5" />} colorClass="bg-slate-100 text-slate-700" description="Doctor and admin roles" />
          </div>

          <div className="grid gap-6 xl:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.6fr)]">
            <section className="surface-panel rounded-xl p-5" id="analytics-timeline-panel">
              <h2 className="text-base font-semibold text-slate-900">Record activity</h2>
              <p className="mt-1 text-sm text-slate-500">Number of saved research records by day</p>
              {stats.scansByDay.length === 0 ? (
                <div className="flex h-64 items-center justify-center rounded-lg bg-slate-50 text-center text-sm text-slate-500">Activity will appear after the first analysis is saved.</div>
              ) : (
                <div className="mt-4 h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={stats.scansByDay} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                      <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
                      <YAxis allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ background: '#10232d', border: 0, borderRadius: 8, color: '#fff', fontSize: 12 }} />
                      <Line type="monotone" dataKey="count" name="Records" stroke="#17718b" strokeWidth={2.5} dot={{ r: 3, fill: '#17718b', strokeWidth: 0 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </section>

            <section className="surface-panel overflow-hidden rounded-xl" id="clinician-directory-panel">
              <div className="border-b border-slate-200 px-5 py-4"><h2 className="text-base font-semibold text-slate-900">Local accounts</h2><p className="mt-1 text-sm text-slate-500">Accounts available in this workspace</p></div>
              <div className="divide-y divide-slate-100">
                {users.map((user) => <div key={user.uid} className="flex items-center justify-between gap-4 px-5 py-4"><div className="min-w-0"><strong className="block truncate text-sm font-semibold text-slate-900">{user.name}</strong><span className="block truncate text-xs text-slate-500">{user.email}</span></div><div className="text-right"><span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-semibold capitalize text-slate-700">{user.role}</span><span className="mt-1.5 block text-[11px] text-slate-400">{formatDate(user.createdAt)}</span></div></div>)}
              </div>
            </section>
          </div>

          <section id="global-scans-archive">
            <div className="mb-3"><h2 className="text-lg font-semibold text-slate-900">All local records</h2><p className="mt-1 text-sm text-slate-500">Research outputs across local accounts</p></div>
            <ScanTable scans={stats.recentScans} showPhysicianColumn />
          </section>
        </div>
      )}
    </div>
  );
};
