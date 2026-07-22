import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { StatsCard } from '../components/StatsCard';
import { ScanTable } from '../components/ScanTable';
import { AdminStats, User } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, ClipboardList, AlertTriangle, CheckCircle, Users, Calendar, Loader2, AlertCircle, TrendingUp } from 'lucide-react';

export const AdminDashboard: React.FC = () => {
  const { getAuthHeaders } = useAuth();
  
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [usersList, setUsersList] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAdminData = async () => {
      try {
        setLoading(true);
        const headers = getAuthHeaders();
        
        // Fetch stats & users list in parallel
        const [statsRes, usersRes] = await Promise.all([
          fetch('/api/dashboard/stats', { headers }),
          fetch('/api/admin/users', { headers })
        ]);

        if (!statsRes.ok || !usersRes.ok) {
          throw new Error('Administrative database sync failed. Verification credentials required.');
        }

        const statsData = await statsRes.json();
        const usersData = await usersRes.json();

        setStats(statsData);
        setUsersList(usersData);
      } catch (err: any) {
        setError(err.message || 'Error loading clinical records.');
      } finally {
        setLoading(false);
      }
    };

    fetchAdminData();
  }, []);

  const formatDate = (isoStr: string) => {
    try {
      return new Date(isoStr).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      });
    } catch (e) {
      return isoStr;
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" id="admin-dashboard-page">
      <div className="space-y-2 mb-8">
        <h1 className="font-sans font-extrabold text-slate-900 tracking-tight text-3xl">
          System Analytics Workspace
        </h1>
        <p className="text-sm text-slate-500">
          Global clinical deployment surveillance dashboard. Review patient throughput, abnormal diagnosis distributions, and active physician accounts.
        </p>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-24 space-y-3" id="loading-spinner">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
          <span className="text-sm text-slate-500 font-medium">Loading administrative modules...</span>
        </div>
      ) : error ? (
        <div className="bg-red-50 border border-red-100/50 p-5 rounded-2xl flex items-start space-x-3 text-red-800 text-sm max-w-xl mx-auto" id="error-display">
          <AlertCircle className="h-5 w-5 text-red-600 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold block text-red-900 mb-1">Administrative Privilege Verification Failed</span>
            <p className="leading-relaxed">{error}</p>
          </div>
        </div>
      ) : !stats ? (
        <div className="text-center text-slate-500 py-12">
          Failed to load administrative analytics.
        </div>
      ) : (
        <div className="space-y-8 animate-fade-in">
          {/* Stats Indicators Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            <StatsCard
              title="Total Diagnostics Run"
              value={stats.totalScans}
              icon={<ClipboardList className="h-5 w-5" />}
              colorClass="bg-blue-50 text-blue-600 border border-blue-100/40"
              description="Ultrasound scan queries"
            />
            <StatsCard
              title="Normal Scans Index"
              value={stats.normalCount}
              icon={<CheckCircle className="h-5 w-5" />}
              colorClass="bg-emerald-50 text-emerald-600 border border-emerald-100/40"
              description={`${((stats.normalCount / (stats.totalScans || 1)) * 100).toFixed(1)}% of throughput`}
            />
            <StatsCard
              title="Abnormal Scans (NAFLD)"
              value={stats.abnormalCount}
              icon={<AlertTriangle className="h-5 w-5 animate-pulse" />}
              colorClass="bg-red-50 text-red-600 border border-red-100/40"
              description={`${((stats.abnormalCount / (stats.totalScans || 1)) * 100).toFixed(1)}% of throughput`}
            />
            <StatsCard
              title="Active Clinician Accounts"
              value={usersList.length}
              icon={<Users className="h-5 w-5" />}
              colorClass="bg-indigo-50 text-indigo-600 border border-indigo-100/40"
              description="Licensed doctors in directory"
            />
          </div>

          {/* Chart & Users split */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            {/* Left: Scans timeline LineChart */}
            <div className="lg:col-span-8 bg-white border border-slate-200 p-5 rounded-xl shadow-sm flex flex-col justify-between" id="analytics-timeline-panel">
              <div className="flex items-center justify-between mb-4">
                <div className="space-y-1">
                  <h3 className="text-base font-bold text-slate-800 flex items-center space-x-1.5">
                    <TrendingUp className="h-5 w-5 text-blue-600" />
                    <span>Clinical Throughput Timeline</span>
                  </h3>
                  <p className="text-xs text-slate-500">Scan submission counts grouped daily</p>
                </div>
              </div>

              <div className="w-full h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={stats.scansByDay}
                    margin={{ top: 10, right: 10, left: -20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis 
                      dataKey="date" 
                      tickFormatter={(d) => d.split('-')[2] + '/' + d.split('-')[1]} // display dd/mm format
                      tick={{ fill: '#64748b', fontSize: 10 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis 
                      tick={{ fill: '#64748b', fontSize: 10 }}
                      tickLine={false}
                      axisLine={false}
                      allowDecimals={false}
                    />
                    <Tooltip 
                      contentStyle={{ background: '#0f172a', border: 'none', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                      labelStyle={{ color: '#94a3b8', fontWeight: 'bold' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="count" 
                      name="Scans Executed"
                      stroke="#2563eb" 
                      strokeWidth={2.5}
                      dot={{ r: 3, fill: '#2563eb', strokeWidth: 0 }}
                      activeDot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Right: Licensed Clinicians Directory */}
            <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden flex flex-col justify-between" id="clinician-directory-panel">
              <div className="p-5 border-b border-slate-200">
                <h3 className="text-base font-bold text-slate-800">Licensed Clinician Registry</h3>
                <p className="text-xs text-slate-500">Total authorized staff in organization</p>
              </div>

              <div className="divide-y divide-slate-200 overflow-y-auto max-h-[250px] flex-grow">
                {usersList.length > 0 ? (
                  usersList.map((usr) => (
                    <div key={usr.uid} className="p-4 flex items-center justify-between hover:bg-slate-50/50">
                      <div className="space-y-0.5 truncate max-w-[170px]">
                        <span className="font-sans font-semibold text-slate-800 text-xs block truncate">{usr.name}</span>
                        <span className="font-mono text-[10px] text-slate-400 block truncate">{usr.email}</span>
                      </div>
                      <div className="text-right">
                        <span className={`inline-block px-2 py-0.5 rounded-full text-[9px] uppercase font-bold tracking-wider ${
                          usr.role === 'admin' 
                            ? 'bg-indigo-100 text-indigo-700' 
                            : 'bg-emerald-100 text-emerald-700'
                        }`}>
                          {usr.role}
                        </span>
                        <span className="block text-[8px] text-slate-400 mt-0.5">{formatDate(usr.createdAt)}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="py-12 text-center text-xs text-slate-400">
                    No users registered.
                  </div>
                )}
              </div>

              <div className="p-4 bg-slate-50 border-t border-slate-200 text-center text-[10px] text-slate-500 font-mono">
                Licensed Healthcare Registry
              </div>
            </div>
          </div>

          {/* Recent Scans Directory Table */}
          <div className="space-y-3" id="global-scans-archive">
            <h3 className="text-base font-bold text-slate-800">Global Patients Directory</h3>
            <ScanTable scans={stats.recentScans} showPhysicianColumn={true} />
          </div>
        </div>
      )}
    </div>
  );
};
