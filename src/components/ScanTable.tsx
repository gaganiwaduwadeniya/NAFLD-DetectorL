import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Scan } from '../types';
import { Search, Eye, Filter, ArrowUpDown, ChevronRight, CheckCircle, AlertTriangle } from 'lucide-react';

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

  // Filter and sort logic combined
  const filteredAndSortedScans = useMemo(() => {
    let result = [...scans];

    // Search Term Filter
    if (searchTerm.trim() !== '') {
      const lowerSearch = searchTerm.toLowerCase();
      result = result.filter(
        (s) =>
          s.patientName.toLowerCase().includes(lowerSearch) ||
          s.id.toLowerCase().includes(lowerSearch) ||
          (showPhysicianColumn && s.doctorName.toLowerCase().includes(lowerSearch))
      );
    }

    // Prediction Filter
    if (predictionFilter !== 'All') {
      result = result.filter((s) => s.prediction === predictionFilter);
    }

    // Gender Filter
    if (genderFilter !== 'All') {
      result = result.filter((s) => s.patientGender === genderFilter);
    }

    // Sort Logic
    result.sort((a, b) => {
      let comparison = 0;
      if (sortBy === 'date') {
        comparison = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      } else if (sortBy === 'age') {
        comparison = a.patientAge - b.patientAge;
      } else if (sortBy === 'confidence') {
        comparison = a.confidence - b.confidence;
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
              <option value="All">All Diagnostic Outputs</option>
              <option value="Normal">Normal Only</option>
              <option value="Abnormal">Abnormal Only</option>
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

      {/* Actual Data Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse" id="scans-data-table">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-slate-400 text-[10px] font-bold uppercase tracking-wider">
              <th className="py-3 px-5">Patient Dossier</th>
              <th className="py-3 px-5 cursor-pointer hover:bg-slate-50" onClick={() => toggleSort('age')}>
                <div className="flex items-center space-x-1">
                  <span>Demographics</span>
                  <ArrowUpDown className="h-3 w-3 text-slate-400" />
                </div>
              </th>
              {showPhysicianColumn && <th className="py-3 px-5">Consulting Doctor</th>}
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
                    <div className="text-slate-700">{scan.patientAge} Years Old</div>
                    <div className="text-slate-400 text-xs">{scan.patientGender}</div>
                  </td>

                  {/* Consulting Doctor if admin mode */}
                  {showPhysicianColumn && (
                    <td className="py-3.5 px-5">
                      <div className="font-medium text-slate-700">{scan.doctorName}</div>
                      <div className="text-[10px] text-slate-400 uppercase">Clinician ID: {scan.doctorId}</div>
                    </td>
                  )}

                  {/* Scan Date & Time */}
                  <td className="py-3.5 px-5 text-slate-600">
                    <div>{formatDate(scan.timestamp).split(',')[0]}</div>
                    <div className="text-slate-400 text-xs">{formatDate(scan.timestamp).split(',')[1]}</div>
                  </td>

                  {/* Prediction Label & Confidence Badge */}
                  <td className="py-3.5 px-5">
                    <div className="flex items-center space-x-2">
                      {scan.prediction === 'Normal' ? (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-100/40">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Normal
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold bg-red-50 text-red-700 border border-red-100/40">
                          <AlertTriangle className="h-3 w-3 mr-1" />
                          Abnormal
                        </span>
                      )}
                      <span className="text-xs font-semibold text-slate-500 font-mono">
                        ({scan.confidence.toFixed(1)}% Conf)
                      </span>
                    </div>
                  </td>

                  {/* Table Row Action Icon */}
                  <td className="py-3.5 px-5 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/scan/${scan.id}`);
                      }}
                      className="p-1.5 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all"
                    >
                      <ChevronRight className="h-4.5 w-4.5" />
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={showPhysicianColumn ? 6 : 5} className="py-12 text-center text-slate-400 font-sans">
                  <div className="max-w-xs mx-auto space-y-2">
                    <p className="text-sm font-semibold text-slate-600">No Patient Scans Found</p>
                    <p className="text-xs text-slate-400">Try modifying search tags or prediction classification filters.</p>
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      {/* Table Footer Stats */}
      <div className="p-4 bg-slate-50/50 border-t border-slate-200 flex justify-between items-center text-xs text-slate-500">
        <span>Showing {filteredAndSortedScans.length} of {scans.length} total patient records</span>
        <span className="font-mono">NAFLD-DB V1.0</span>
      </div>
    </div>
  );
};
