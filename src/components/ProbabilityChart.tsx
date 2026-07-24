import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Cell, ResponsiveContainer, LabelList, ReferenceLine } from 'recharts';
import { ScanProbabilities } from '../types';

interface ProbabilityChartProps {
  probabilities: ScanProbabilities;
  // V3 fields (0-1 scale); when present, override the legacy 0-100 display
  binaryProbNafld?: number;
  binaryProbNonNafld?: number;
  binaryThreshold?: number;
}

export const ProbabilityChart: React.FC<ProbabilityChartProps> = ({
  probabilities,
  binaryProbNafld,
  binaryProbNonNafld,
  binaryThreshold,
}) => {
  const isV3 = binaryProbNafld !== undefined && binaryProbNonNafld !== undefined;

  if (isV3) {
    // V3 mode — 0–1 scale, annotated threshold
    const data = [
      { name: 'Non-NAFLD', value: parseFloat((binaryProbNonNafld!).toFixed(4)) },
      { name: 'NAFLD',     value: parseFloat((binaryProbNafld!).toFixed(4)) },
    ];
    const colors: Record<string, string> = {
      'Non-NAFLD': '#10b981',
      'NAFLD': '#ef4444',
    };

    return (
      <div className="w-full h-44" id="probability-chart-container">
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
          Ensemble Probability (0–1)
          {binaryThreshold !== undefined && (
            <span className="ml-2 font-mono text-slate-300 normal-case">
              │ threshold {binaryThreshold.toFixed(4)}
            </span>
          )}
        </h4>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 5, right: 50, left: -10, bottom: 5 }}
          >
            <XAxis
              type="number"
              domain={[0, 1]}
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              tickFormatter={(v) => v.toFixed(1)}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              dataKey="name"
              type="category"
              tick={{ fill: '#475569', fontSize: 11, fontWeight: 500 }}
              axisLine={false}
              tickLine={false}
              width={72}
            />
            {binaryThreshold !== undefined && (
              <ReferenceLine
                x={binaryThreshold}
                stroke="#94a3b8"
                strokeDasharray="4 3"
                strokeWidth={1.5}
              />
            )}
            <Bar dataKey="value" barSize={16} radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={colors[entry.name] ?? '#64748b'}
                />
              ))}
              <LabelList
                dataKey="value"
                position="right"
                formatter={(v: number) => v.toFixed(3)}
                style={{ fill: '#334155', fontSize: 11, fontWeight: 600 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // Legacy mode — 0–100 % scale
  const data = [
    { name: 'Normal',   value: parseFloat(probabilities.Normal.toFixed(1)) },
    { name: 'Abnormal', value: parseFloat(probabilities.Abnormal.toFixed(1)) },
  ];
  const colors: Record<string, string> = {
    Normal:   '#10b981',
    Abnormal: '#ef4444',
  };

  return (
    <div className="w-full h-44" id="probability-chart-container">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
        Probability Distribution (%)
      </h4>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 35, left: -20, bottom: 5 }}
        >
          <XAxis type="number" domain={[0, 100]} hide />
          <YAxis
            dataKey="name"
            type="category"
            tick={{ fill: '#475569', fontSize: 11, fontWeight: 500 }}
            axisLine={false}
            tickLine={false}
          />
          <Bar dataKey="value" barSize={16} radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.name === 'Normal' ? colors.Normal : colors.Abnormal}
              />
            ))}
            <LabelList
              dataKey="value"
              position="right"
              formatter={(value: number) => `${value}%`}
              style={{ fill: '#334155', fontSize: 11, fontWeight: 600 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
