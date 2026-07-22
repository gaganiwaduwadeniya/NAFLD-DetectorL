import React from 'react';

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  description?: string;
  colorClass?: string;
}

export const StatsCard: React.FC<StatsCardProps> = ({ 
  title, 
  value, 
  icon, 
  description, 
  colorClass = 'bg-blue-50 text-blue-600' 
}) => {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm flex items-center justify-between" id={`stats-card-${title.toLowerCase().replace(/\s+/g, '-')}`}>
      <div className="space-y-1.5">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
          {title}
        </span>
        <span className="text-3xl font-bold text-slate-800 block tracking-tight">
          {value}
        </span>
        {description && (
          <span className="text-xs text-slate-500 block">
            {description}
          </span>
        )}
      </div>

      <div className={`p-3.5 rounded-xl ${colorClass}`}>
        {icon}
      </div>
    </div>
  );
};
