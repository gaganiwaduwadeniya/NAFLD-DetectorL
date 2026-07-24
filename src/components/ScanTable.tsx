import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Scan } from '../types';
import { Search, Eye, ArrowUpDown, ChevronRight, CheckCircle, AlertTriangle } from 'lucide-react';

interface ScanTableProps {
  scans: Scan[];
  showPhysicianColumn?: boolean;
}

export const ScanTable: React.FC<ScanTableProps> = ({ scans, showPhysicianColumn = false }) => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [predictionFilter, setPredictionFilter] = useState<'All' | 'Normal' | 'Abnormal'>('All');
  const [genderFilter, setGenderFilter] = useState<'All' | 'Male' | 'Female' | 'Other'>('All');
  const [sortBy, setSortBy] = useState<'date' | 'age' | 'confidence'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Derive a normalised prediction string for filtering regardless of schema version
  const normalisedPrediction = (s: Scan): 'Normal' | 'Abnormal' => {
    if (s.prediction === 'Normal') return 'Normal';
    if (s.prediction === 'Abnormal') return 'Abnormal';
    // V3 records: derive from binaryResult
    if (s.binaryResult === 'Non-NAFLD') return 'Normal';
    if (s.binaryResult === 'NAFLD') return 'Abnormal';
    return 'Normal';
  };

  // Display label shown in the Diagnostic Output column
  const displayLabel = (s: Scan): string => {
    if (s.schemaVersion === 3 && s.finalLabel) return s.finalLabel;
    return s.prediction ?? 'Unknown';
  };

  const isAbnormalScan = (s: Scan): boolean => normalisedPrediction(s) === 'Abnormal';

  const filteredAndSortedScans = useMemo(() => {
    let result = [...scans];

    // Search filter
    if (searchTerm.trim() !== '') {
      const lowerSearch = searchTerm.toLowerCase();
      result = result.filter(
        (s) =>
          s.patientName.toLowerCase().includes(lowerSearch) ||
          s.id.toLowerCase().includes(lowerSearch) ||
          (showPhysicianColumn && s.doctorName.toLowerCase().includes(lowerSearch))
      );
    }

    // Prediction filter (works for both V3 and legacy)
    if (predictionFilter !== 'All') {
      result = result.filter((s) => normalisedPrediction(s) === predictionFilter);
    }

    // Gender filter
    if (genderFilter !== 'All') {
      result = result.filter((s) => s.patientGender === genderFilter);
    }

    // Sort
    result.sort((a, b) => {
      let comparison = 0;
      if (sortBy === 'date') {
        comparison = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      } else if (sortBy === 'age') {
        comparison = a.patientAge - b.patientAge;
      } else if (sortBy === 'confidence') {
        comparison = (a.confidence ?? 0) - (b.confidence ?? 0);
      }
      return sortOrder === 'desc' ? -comparison : comparison;
    });

    return result;
  }, [scans, searchTerm, predictionFilter, genderFilter, sortBy, sortOrder, showPhysicianColumn]);

  const toggleSort = (field: 'date' | 'age' | 'confidence') => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const formatDate = (isoStr: string) => {
    try {
      const date = new Date(isoStr);
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch (e) {
      return isoStr;
    }
  };

  // Confidence/prob display string: V3 shows "X% prob", legacy shows "X% conf"
  const probDisplay = (s: Scan): string => {
    if (s.confidence === undefined || s.confidence === null) return '';
    const suffix = s.schemaVersion === 3 ? 'prob' : 'conf';
    return `(${s.confidence.toFixed(1)}% ${suffix})`;
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden" id="scan-table-container">
      {/* Table Action Controls */}
      <div className="p-5 border-b border-slate-200 bg-slate-50/50 space-y-4 sm:space-y-0 sm:flex sm:items-center sm:justify-between">
        {/* Search Input */}
        <div className="relative max-w-sm w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 h-4 w-4" />
          <input
            type="text"
            placeholder="Search by Patient Name, ID or Doctor..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-4 py-2 border border-slate-200 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/10 focus:border-blue-500"
            id="input-table-search"
          />
        </div>

        {/* Filter Badges */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Prediction Dropdown */}
          <div className="flex items-center space-x-1.5">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Classification:</span>
            <select
              value={predictionFilter}
              onChange={(e) => setPredictionFilter(e.target.value as any)}
              className="px-2.5 py-1.5 border border-slate-200 bg-white rounded-lg text-xs font-semibold text-slate-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
              id="select-filter-prediction"
            >
              <option value="All">All Outputs</option>
              <option value="Normal">Non-NAFLD / Normal</option>
              <option value="Abnormal">NAFLD / Abnormal</option>
            </select>
          </div>

          {/* Gender Dropdown */}
          <div className="flex items-center space-x-1.5">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Gender:</span>
            <select
              value={genderFilter}
              onChange={(e) => setGenderFilter(e.target.value as any)}
              className="px-2.5 py-1.5 border border-slate-200 bg-white rounded-lg text-xs font-semibold text-slate-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
              id="select-filter-gender"
            >
              <option value="All">All Genders</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>
        </div>
      </div>

      {/* Data Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse" id="scans-data-table">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-slate-400 text-[10px] font-bold uppercase tracking-wider">
              <th className="py-3 px-5">Patient</th>
              <th className="py-3 px-5 cursor-pointer hover:bg-slate-50" onClick={() => toggleSort('age')}>
                <div className="flex items-center space-x-1">
                  <span>Demographics</span>
                  <ArrowUpDown className="h-3 w-3 text-slate-400" />
                </div>
              </th>
              {showPhysicianColumn && <th className="py-3 px-5">Clinician</th>}
              <th className="py-3 px-5 cursor-pointer hover:bg-slate-50" onClick={() => toggleSort('date')}>
                <div className="flex items-center space-x-1">
                  <span>Scan Date</span>
                  <ArrowUpDown className="h-3 w-3 text-slate-400" />
                </div>
              </th>
              <th className="py-3 px-5 cursor-pointer hover:bg-slate-50" onClick={() => toggleSort('confidence')}>
                <div className="flex items-center space-x-1">
                  <span>Diagnostic Output</span>
                  <ArrowUpDown className="h-3 w-3 text-slate-400" />
                </div>
              </th>
              <th className="py-3 px-5 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 text-sm">
            {filteredAndSortedScans.length > 0 ? (
              filteredAndSortedScans.map((scan) => (
                <tr
                  key={scan.id}
                  onClick={() => navigate(`/scan/${scan.id}`)}
                  className="hover:bg-slate-50/70 transition-colors cursor-pointer group"
                >
                  {/* Patient Name & Scan ID */}
                  <td className="py-3.5 px-5">
                    <div className="font-sans font-semibold text-slate-800">{scan.patientName}</div>
                    <div className="font-mono text-[10px] text-slate-400 uppercase tracking-tight">ID: {scan.id}</div>
                  </td>

                  {/* Patient Age & Gender */}
                  <td className="py-3.5 px-5">
                    <div className="text-slate-700">{scan.patientAge} yrs</div>
                    <div className="text-slate-400 text-xs">{scan.patientGender}</div>
                  </td>

                  {/* Consulting Doctor if admin */}
                  {showPhysicianColumn && (
                    <td className="py-3.5 px-5">
                      <div className="font-medium text-slate-700">{scan.doctorName}</div>
                      <div className="text-[10px] text-slate-400">ID: {scan.doctorId}</div>
                    </td>
                  )}

                  {/* Scan Date & Time */}
                  <td className="py-3.5 px-5 text-slate-600">
                    <div>{formatDate(scan.timestamp).split(',')[0]}</div>
                    <div className="text-slate-400 text-xs">{formatDate(scan.timestamp).split(',')[1]}</div>
                  </td>

                  {/* Diagnostic Label & probability */}
                  <td className="py-3.5 px-5">
                    <div className="flex items-center space-x-2">
                      {isAbnormalScan(scan) ? (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold bg-red-50 text-red-700 border border-red-100/40">
                          <AlertTriangle className="h-3 w-3 mr-1" />
                          {displayLabel(scan)}
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-100/40">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          {displayLabel(scan)}
                        </span>
                      )}
                      {scan.confidence !== undefined && (
                        <span className="text-xs font-semibold text-slate-500 font-mono">
                          {probDisplay(scan)}
                        </span>
                      )}
                    </div>
                  </td>

                  {/* Row Action */}
                  <td className="py-3.5 px-5 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/scan/${scan.id}`);
                      }}
                      className="p-1.5 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={showPhysicianColumn ? 6 : 5} className="py-12 text-center text-slate-400 font-sans">
                  <div className="max-w-xs mx-auto space-y-2">
                    <p className="text-sm font-semibold text-slate-600">No Scans Found</p>
                    <p className="text-xs text-slate-400">Try adjusting search or filter criteria.</p>
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Table Footer */}
      <div className="p-4 bg-slate-50/50 border-t border-slate-200 flex justify-between items-center text-xs text-slate-500">
        <span>Showing {filteredAndSortedScans.length} of {scans.length} total records</span>
        <span className="font-mono">NAFLD-Detector V3</span>
      </div>
    </div>
  );
};
