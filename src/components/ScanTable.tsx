import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowDownAZ, ArrowRight, FileSearch, Filter, Search } from 'lucide-react';
import { Scan } from '../types';

interface ScanTableProps {
  scans: Scan[];
  showPhysicianColumn?: boolean;
}

const normalisedPrediction = (scan: Scan): 'Normal' | 'Abnormal' => {
  if (scan.schemaVersion === 4) return (scan.finalStage ?? 0) === 0 ? 'Normal' : 'Abnormal';
  if (scan.binaryResult === 'NAFLD' || scan.prediction === 'Abnormal') return 'Abnormal';
  return 'Normal';
};

const displayLabel = (scan: Scan) => scan.schemaVersion === 4 && scan.finalStageLabel
  ? scan.finalStageLabel
  : scan.schemaVersion === 3 && scan.finalLabel
    ? scan.finalLabel
    : scan.prediction ?? 'Unknown';

const badgeClass = (scan: Scan) => {
  const stage = scan.schemaVersion === 4 ? scan.finalStage : undefined;
  if (stage === 3) return 'border-rose-200 bg-rose-50 text-rose-800';
  if (stage === 2) return 'border-orange-200 bg-orange-50 text-orange-800';
  if (stage === 1) return 'border-amber-200 bg-amber-50 text-amber-800';
  if (normalisedPrediction(scan) === 'Abnormal') return 'border-amber-200 bg-amber-50 text-amber-800';
  return 'border-emerald-200 bg-emerald-50 text-emerald-800';
};

const formatDate = (value: string) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
};

export const ScanTable: React.FC<ScanTableProps> = ({ scans, showPhysicianColumn = false }) => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [predictionFilter, setPredictionFilter] = useState<'All' | 'Normal' | 'Abnormal'>('All');
  const [genderFilter, setGenderFilter] = useState<'All' | 'Male' | 'Female' | 'Other'>('All');
  const [sortBy, setSortBy] = useState<'date' | 'age' | 'confidence'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const filteredScans = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    return [...scans]
      .filter((scan) => !query || scan.patientName.toLowerCase().includes(query) || scan.id.toLowerCase().includes(query) || (showPhysicianColumn && scan.doctorName.toLowerCase().includes(query)))
      .filter((scan) => predictionFilter === 'All' || normalisedPrediction(scan) === predictionFilter)
      .filter((scan) => genderFilter === 'All' || scan.patientGender === genderFilter)
      .sort((a, b) => {
        const comparison = sortBy === 'date'
          ? new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
          : sortBy === 'age'
            ? a.patientAge - b.patientAge
            : (a.confidence ?? 0) - (b.confidence ?? 0);
        return sortOrder === 'desc' ? -comparison : comparison;
      });
  }, [scans, searchTerm, predictionFilter, genderFilter, sortBy, sortOrder, showPhysicianColumn]);

  const updateSort = (field: typeof sortBy) => {
    if (field === sortBy) setSortOrder((current) => current === 'asc' ? 'desc' : 'asc');
    else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const filtersActive = Boolean(searchTerm || predictionFilter !== 'All' || genderFilter !== 'All');
  const clearFilters = () => {
    setSearchTerm('');
    setPredictionFilter('All');
    setGenderFilter('All');
  };

  return (
    <section className="surface-panel overflow-hidden rounded-xl" id="scan-table-container" aria-label="Research records">
      <div className="border-b border-slate-200 bg-slate-50/70 p-4 sm:p-5">
        <div className="grid gap-3 lg:grid-cols-[minmax(240px,1fr)_auto_auto]">
          <label className="relative block">
            <span className="sr-only">Search records</span>
            <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" aria-hidden="true" />
            <input id="input-table-search" type="search" value={searchTerm} onChange={(event) => setSearchTerm(event.target.value)} className="field-control pl-10" placeholder={showPhysicianColumn ? 'Search case, record ID, or reviewer' : 'Search case alias or record ID'} />
          </label>
          <label className="flex items-center gap-2"><Filter className="h-4 w-4 text-slate-400" /><span className="sr-only">Classification filter</span><select id="select-filter-prediction" value={predictionFilter} onChange={(event) => setPredictionFilter(event.target.value as any)} className="field-control min-w-44"><option value="All">All classifications</option><option value="Normal">Non-NAFLD</option><option value="Abnormal">NAFLD stages</option></select></label>
          <label><span className="sr-only">Sex filter</span><select id="select-filter-gender" value={genderFilter} onChange={(event) => setGenderFilter(event.target.value as any)} className="field-control min-w-36"><option value="All">All sexes</option><option value="Male">Male</option><option value="Female">Female</option><option value="Other">Other</option></select></label>
        </div>
      </div>

      {filteredScans.length === 0 ? (
        <div className="px-6 py-16 text-center">
          <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-slate-100 text-slate-500"><FileSearch className="h-5 w-5" /></span>
          <h3 className="mt-4 text-base font-semibold text-slate-900">{scans.length === 0 ? 'No research records yet' : 'No records match these filters'}</h3>
          <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">{scans.length === 0 ? 'Run a new analysis to create the first local record. Source image bytes will remain unretained in the current configuration.' : 'Clear the filters or try a different case alias or record ID.'}</p>
          <button type="button" onClick={scans.length === 0 ? () => navigate('/') : clearFilters} className="mt-5 inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 hover:border-brand-300 hover:text-brand-700">{scans.length === 0 ? 'Start an analysis' : 'Clear filters'}<ArrowRight className="h-4 w-4" /></button>
        </div>
      ) : (
        <>
          <div className="hidden overflow-x-auto md:block">
            <table className="w-full text-left" id="scans-data-table">
              <thead className="border-b border-slate-200 bg-white text-xs font-semibold text-slate-500">
                <tr>
                  <th className="px-5 py-3.5">Case</th>
                  <th className="px-5 py-3.5"><button type="button" onClick={() => updateSort('age')} className="flex items-center gap-1 hover:text-slate-900">Age / sex<ArrowDownAZ className="h-3.5 w-3.5" /></button></th>
                  {showPhysicianColumn && <th className="px-5 py-3.5">Reviewer</th>}
                  <th className="px-5 py-3.5"><button type="button" onClick={() => updateSort('date')} className="flex items-center gap-1 hover:text-slate-900">Created<ArrowDownAZ className="h-3.5 w-3.5" /></button></th>
                  <th className="px-5 py-3.5">Model output</th>
                  <th className="px-5 py-3.5 text-right"><span className="sr-only">Open record</span></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredScans.map((scan) => (
                  <tr key={scan.id} onClick={() => navigate(`/scan/${scan.id}`)} className="cursor-pointer transition hover:bg-brand-50/50">
                    <td className="px-5 py-4"><strong className="block text-sm font-semibold text-slate-900">{scan.patientName}</strong><span className="mt-0.5 block max-w-48 truncate font-mono text-[11px] text-slate-400">{scan.id}</span></td>
                    <td className="px-5 py-4 text-sm text-slate-600">{scan.patientAge} / {scan.patientGender}</td>
                    {showPhysicianColumn && <td className="px-5 py-4 text-sm text-slate-600">{scan.doctorName}</td>}
                    <td className="px-5 py-4 text-sm text-slate-600">{formatDate(scan.timestamp)}</td>
                    <td className="px-5 py-4"><span className={`inline-flex rounded-md border px-2.5 py-1 text-xs font-semibold ${badgeClass(scan)}`}>{displayLabel(scan)}</span></td>
                    <td className="px-5 py-4 text-right"><button type="button" onClick={(event) => { event.stopPropagation(); navigate(`/scan/${scan.id}`); }} className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-white hover:text-brand-700" aria-label={`Open ${scan.patientName} record`}><ArrowRight className="h-4 w-4" /></button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="divide-y divide-slate-100 md:hidden">
            {filteredScans.map((scan) => (
              <button key={scan.id} type="button" onClick={() => navigate(`/scan/${scan.id}`)} className="block w-full p-4 text-left transition hover:bg-brand-50/50">
                <div className="flex items-start justify-between gap-3"><div className="min-w-0"><strong className="block truncate text-sm font-semibold text-slate-900">{scan.patientName}</strong><span className="mt-0.5 block truncate font-mono text-[11px] text-slate-400">{scan.id}</span></div><span className={`shrink-0 rounded-md border px-2 py-1 text-xs font-semibold ${badgeClass(scan)}`}>{displayLabel(scan)}</span></div>
                <div className="mt-3 flex items-center justify-between text-xs text-slate-500"><span>{scan.patientAge} / {scan.patientGender}</span><span>{formatDate(scan.timestamp)}</span></div>
              </button>
            ))}
          </div>
        </>
      )}

      <footer className="flex items-center justify-between border-t border-slate-200 bg-slate-50/70 px-4 py-3 text-xs text-slate-500 sm:px-5"><span>{filteredScans.length} of {scans.length} records</span>{filtersActive && <button type="button" onClick={clearFilters} className="font-semibold text-brand-700 hover:text-brand-600">Clear filters</button>}</footer>
    </section>
  );
};
