import React from 'react';
import { ResponsiveContainer, Tooltip, PieChart, Pie, Cell } from 'recharts';
import { Activity, Clock, Zap } from 'lucide-react';

interface Props {
  labels: string[];
  cmi: number;
  latency: number;
}

export const MetricsDashboard: React.FC<Props> = ({ labels, cmi, latency }) => {
  if (!labels || labels.length === 0) return null;

  const teCount = labels.filter(l => l === 'TE').length;
  const enCount = labels.filter(l => l === 'EN').length;
  const otherCount = labels.filter(l => l === 'OTHER').length;

  const data = [
    { name: 'Telugu', value: teCount, color: '#3b82f6' },
    { name: 'English', value: enCount, color: '#22c55e' },
    { name: 'Other', value: otherCount, color: '#94a3b8' },
  ].filter(d => d.value > 0);

  return (
    <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* CMI Score Card */}
      <div className="glass p-5 rounded-xl flex items-center justify-between">
        <div>
          <h4 className="text-sm font-medium text-text-muted mb-1 flex items-center gap-2">
            <Activity size={16} className="text-primary" />
            Code Mixing Index (CMI)
          </h4>
          <div className="text-3xl font-bold">{cmi.toFixed(1)}</div>
        </div>
        <div className="text-right">
          <h4 className="text-sm font-medium text-text-muted mb-1 flex items-center gap-2 justify-end">
            <Clock size={16} className="text-secondary" />
            Latency
          </h4>
          <div className="text-xl font-semibold">{latency.toFixed(2)}s</div>
        </div>
      </div>

      {/* Distribution Chart */}
      <div className="glass p-4 rounded-xl flex items-center">
        <div className="w-1/2 h-32">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={30}
                outerRadius={45}
                paddingAngle={5}
                dataKey="value"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)', borderRadius: '8px' }}
                itemStyle={{ color: 'var(--text)' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="w-1/2">
          <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Zap size={16} className="text-yellow-500" />
            Token Distribution
          </h4>
          <div className="space-y-2">
            {data.map((d, i) => (
              <div key={i} className="flex justify-between text-xs">
                <span className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }}></div>
                  {d.name}
                </span>
                <span className="font-semibold">{d.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
