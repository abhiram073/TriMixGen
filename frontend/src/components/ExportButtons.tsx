import React from 'react';
import { Copy, Download, FileJson } from 'lucide-react';
import type { GenerateResponse } from '../services/api';

interface Props {
  data: GenerateResponse | null;
}

export const ExportButtons: React.FC<Props> = ({ data }) => {
  if (!data) return null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(data.generated_text);
      // We could add a toast notification here
    } catch (e) {
      console.error("Failed to copy", e);
    }
  };

  const downloadTxt = () => {
    const blob = new Blob([data.generated_text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trimixgen_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadJson = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trimixgen_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex gap-2 justify-end mt-4">
      <button 
        onClick={handleCopy}
        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 rounded transition-colors"
        aria-label="Copy output text"
      >
        <Copy size={14} /> Copy
      </button>
      <button 
        onClick={downloadTxt}
        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 rounded transition-colors"
        aria-label="Download TXT"
      >
        <Download size={14} /> TXT
      </button>
      <button 
        onClick={downloadJson}
        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 rounded transition-colors"
        aria-label="Download JSON"
      >
        <FileJson size={14} /> JSON
      </button>
    </div>
  );
};
