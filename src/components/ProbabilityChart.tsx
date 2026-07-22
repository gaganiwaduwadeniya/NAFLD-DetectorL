import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Cell, ResponsiveContainer, LabelList } from 'recharts';
import { ScanProbabilities } from '../types';

interface ProbabilityChartProps {
  probabilities: ScanProbabilities;
}

export const ProbabilityChart: React.FC<ProbabilityChartProps> = ({ probabilities }) => {
  const data = [
    { name: 'Normal', value: parseFloat(probabilities.Normal.toFixed(1)) },
    { name: 'Abnormal', value: parseFloat(probabilities.Abnormal.toFixed(1)) },
  ];

  const colors = {
    Normal: '#10b981', // green-500
    Abnormal: '#ef4444', // red-500
  };

  return (
    <div className="w-full h-44" id="probability-chart-container">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Probability Distribution (%)</h4>
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
