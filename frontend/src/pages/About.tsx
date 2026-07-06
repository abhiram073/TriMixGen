import React from 'react';

export const About: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto py-10 px-4">
      <h1 className="text-4xl font-bold mb-8 text-primary">Architecture Overview</h1>
      
      <div className="glass p-8 rounded-2xl mb-8 space-y-6">
        <h2 className="text-2xl font-semibold border-b border-[var(--border)] pb-2">The Curriculum Pipeline</h2>
        <p className="text-text-muted leading-relaxed">
          TriMixGen utilizes a progressive curriculum learning strategy on top of <code className="bg-[var(--background)] px-1 py-0.5 rounded">google/mt5-small</code>. 
          The model prevents catastrophic forgetting by utilizing a dual-replay mechanism.
        </p>
        
        <ul className="list-disc list-inside space-y-3 text-text-muted">
          <li><strong>GEN_001:</strong> Aligned model semantics to Romanized syntactic structures using Alpaca.</li>
          <li><strong>GEN_002:</strong> Adapted statistical distributions of natural social media code-mixing.</li>
          <li><strong>GEN_003:</strong> Implemented multi-attribute prompts for Sentiment, Formality, and Lexical Density.</li>
        </ul>
      </div>
      
      <div className="glass p-8 rounded-2xl space-y-6">
        <h2 className="text-2xl font-semibold border-b border-[var(--border)] pb-2">Language Identification (LID)</h2>
        <p className="text-text-muted leading-relaxed">
          Token-level predictions are powered by an <code className="bg-[var(--background)] px-1 py-0.5 rounded">IndicBERT</code> Sequence Classification head.
          The Code-Mixing Index (CMI) is calculated dynamically based on these predictions to measure cross-lingual density.
        </p>
      </div>
    </div>
  );
};
