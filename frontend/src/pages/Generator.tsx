import React, { useState } from 'react';
import { useGenerate } from '../hooks/useGenerate';
import { useHistory } from '../hooks/useHistory';
import { PromptGallery } from '../components/PromptGallery';
import { TokenInspector } from '../components/TokenInspector';
import { MetricsDashboard } from '../components/MetricsDashboard';
import { ExportButtons } from '../components/ExportButtons';
import { Send, Loader2, RefreshCw } from 'lucide-react';

export const Generator: React.FC = () => {
  const { generate, loading, error, result, reset } = useGenerate();
  const { history, addHistory, clearHistory } = useHistory();

  const [prompt, setPrompt] = useState('');
  const [style, setStyle] = useState('neutral');
  const [englishUsage, setEnglishUsage] = useState('auto');
  const [temperature, setTemperature] = useState(0.8);

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!prompt.trim()) return;

    try {
      const data = await generate({ prompt, style, english_usage: englishUsage, temperature });
      addHistory({
        prompt,
        generated_text: data.generated_text,
        cmi: data.cmi
      });
    } catch (err) {
      // Error handled by hook
    }
  };

  return (
    <div className="flex flex-col lg:flex-row gap-8">
      {/* Left Column: Controls */}
      <div className="w-full lg:w-1/3 flex flex-col gap-6">
        <div className="glass p-6 rounded-2xl">
          <h2 className="text-xl font-bold mb-4">Generation Parameters</h2>
          
          <form onSubmit={handleSubmit} className="flex flex-col gap-5">
            <div>
              <label className="block text-sm font-medium mb-2">Instruction Prompt</label>
              <textarea 
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                placeholder="e.g. Write a review for the movie Kalki 2898 AD"
                className="w-full p-3 rounded-lg border border-[var(--border)] bg-[var(--background)] focus:ring-2 focus:ring-primary focus:border-transparent resize-none"
                rows={4}
                required
                maxLength={500}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Style</label>
                <select 
                  value={style}
                  onChange={e => setStyle(e.target.value)}
                  className="w-full p-2.5 rounded-lg border border-[var(--border)] bg-[var(--background)] focus:ring-2 focus:ring-primary"
                >
                  <option value="neutral">Neutral</option>
                  <option value="positive">Positive</option>
                  <option value="negative">Negative</option>
                  <option value="formal">Formal</option>
                  <option value="informal">Informal</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">English Density</label>
                <select 
                  value={englishUsage}
                  onChange={e => setEnglishUsage(e.target.value)}
                  className="w-full p-2.5 rounded-lg border border-[var(--border)] bg-[var(--background)] focus:ring-2 focus:ring-primary"
                >
                  <option value="auto">Auto</option>
                  <option value="high">High English</option>
                  <option value="low">Low English</option>
                </select>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="block text-sm font-medium">Temperature</label>
                <span className="text-xs font-mono bg-[var(--background)] px-2 py-0.5 rounded border border-[var(--border)]">{temperature.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0.1" 
                max="2.0" 
                step="0.1"
                value={temperature}
                onChange={e => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-primary"
              />
            </div>

            <button 
              type="submit"
              disabled={loading || !prompt.trim()}
              className="w-full mt-2 py-3 bg-primary hover:bg-indigo-600 disabled:bg-primary/50 text-white font-semibold rounded-xl flex justify-center items-center gap-2 transition-colors"
            >
              {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
              {loading ? 'Generating...' : 'Generate Text'}
            </button>
          </form>
        </div>

        <PromptGallery onSelect={(p) => setPrompt(p)} />
      </div>

      {/* Right Column: Output & Metrics */}
      <div className="w-full lg:w-2/3 flex flex-col gap-6">
        <div className="glass p-6 rounded-2xl min-h-[400px] flex flex-col">
          <div className="flex justify-between items-center mb-4 border-b border-[var(--border)] pb-4">
            <h2 className="text-xl font-bold">Output</h2>
            {result && (
              <button onClick={reset} className="text-sm flex items-center gap-1.5 text-text-muted hover:text-primary transition-colors">
                <RefreshCw size={14} /> Reset
              </button>
            )}
          </div>

          {error && (
            <div className="p-4 rounded-lg bg-red-500/10 text-red-500 border border-red-500/20 mb-4">
              <strong>Error:</strong> {error}
            </div>
          )}

          {!result && !loading && !error && (
            <div className="flex-1 flex flex-col items-center justify-center text-text-muted opacity-50">
              <SparklesIcon />
              <p className="mt-4">Enter a prompt to generate code-mixed text.</p>
            </div>
          )}

          {loading && (
            <div className="flex-1 flex flex-col items-center justify-center text-primary">
              <Loader2 size={40} className="animate-spin mb-4" />
              <p className="font-medium animate-pulse">Running Inference Pipeline...</p>
            </div>
          )}

          {result && (
            <div className="flex-1 animate-in fade-in duration-500">
              <div className="p-4 bg-[var(--background)] rounded-xl border border-[var(--border)] text-lg leading-relaxed shadow-inner">
                {result.generated_text}
              </div>
              
              <ExportButtons data={result} />
              
              <TokenInspector tokens={result.language_tags ? result.generated_text.split(" ") : []} labels={result.language_tags} />
              
              <MetricsDashboard 
                labels={result.language_tags} 
                cmi={result.cmi} 
                latency={result.latency} 
              />
            </div>
          )}
        </div>

        {/* History Panel */}
        {history.length > 0 && (
          <div className="glass p-6 rounded-2xl">
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-bold">Recent Generations</h3>
              <button onClick={clearHistory} className="text-xs text-red-500 hover:underline">Clear History</button>
            </div>
            <div className="space-y-3">
              {history.map(item => (
                <div key={item.id} className="p-3 bg-[var(--background)] border border-[var(--border)] rounded-lg text-sm flex justify-between items-start gap-4">
                  <div className="overflow-hidden">
                    <p className="font-medium truncate text-text-muted mb-1">{item.prompt}</p>
                    <p className="truncate">{item.generated_text}</p>
                  </div>
                  <div className="flex flex-col items-end shrink-0">
                    <span className="font-mono text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">CMI: {item.cmi.toFixed(1)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const SparklesIcon = () => (
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
    <path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/>
  </svg>
);
