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
    <div className="surface-panel flex items-start justify-between rounded-xl p-5" id={`stats-card-${title.toLowerCase().replace(/\s+/g, '-')}`}>
      <div>
        <span className="block text-sm font-medium text-slate-600">
          {title}
        </span>
        <span className="mt-2 block text-3xl font-semibold tracking-tight text-ink-950 tabular-nums">
          {value}
        </span>
        {description && (
          <span className="mt-1 block text-xs text-slate-500">
            {description}
          </span>
        )}
      </div>

      <div className={`rounded-lg p-2.5 ${colorClass}`}>
        {icon}
      </div>
    </div>
  );
};
