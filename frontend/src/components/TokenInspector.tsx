import React, { useState } from 'react';

interface Props {
  tokens: string[];
  labels: string[];
}

const getLabelColor = (label: string) => {
  if (label === 'TE') return 'text-token-te bg-token-te/10 border-token-te/30';
  if (label === 'EN') return 'text-token-en bg-token-en/10 border-token-en/30';
  return 'text-token-other bg-token-other/10 border-token-other/30';
};

const getLabelName = (label: string) => {
  if (label === 'TE') return 'Telugu';
  if (label === 'EN') return 'English';
  return 'Other/Punctuation';
};

export const TokenInspector: React.FC<Props> = ({ tokens, labels }) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!tokens || tokens.length === 0) return null;

  return (
    <div className="mt-4 p-4 rounded-xl border border-[var(--border)] bg-[var(--background)]">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-sm">Language Visualization</h3>
        <div className="flex gap-4 text-xs">
          <span className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-token-te"></div> Telugu</span>
          <span className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-token-en"></div> English</span>
          <span className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-token-other"></div> Universal</span>
        </div>
      </div>
      
      <div className="flex flex-wrap gap-1.5 leading-loose">
        {tokens.map((token, idx) => {
          const label = labels[idx] || 'OTHER';
          const colorClass = getLabelColor(label);
          const isHovered = hoveredIdx === idx;
          
          return (
            <div 
              key={idx} 
              className="relative inline-block"
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              <span className={`px-1.5 py-0.5 rounded border cursor-default transition-all duration-200 ${colorClass} ${isHovered ? 'ring-2 ring-offset-1 ring-primary/50' : ''}`}>
                {token}
              </span>
              
              {isHovered && (
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-10 w-max px-3 py-1.5 bg-gray-900 text-white text-xs rounded shadow-lg pointer-events-none">
                  <div className="font-semibold">{token}</div>
                  <div className="text-gray-300">{getLabelName(label)}</div>
                  <div className="text-gray-400 mt-0.5" style={{ fontSize: '10px' }}>Confidence: 100%</div>
                  <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
