import { useState, useEffect } from 'react';

export interface HistoryItem {
  id: string;
  prompt: string;
  generated_text: string;
  cmi: number;
  timestamp: string;
}

export function useHistory(maxItems = 10) {
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // Load from local storage on mount
  useEffect(() => {
    const saved = localStorage.getItem('trimixgen_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse history");
      }
    }
  }, []);

  const addHistory = (item: Omit<HistoryItem, 'id' | 'timestamp'>) => {
    const newItem: HistoryItem = {
      ...item,
      id: Date.now().toString(),
      timestamp: new Date().toISOString()
    };
    
    setHistory(prev => {
      const updated = [newItem, ...prev].slice(0, maxItems);
      localStorage.setItem('trimixgen_history', JSON.stringify(updated));
      return updated;
    });
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('trimixgen_history');
  };

  return { history, addHistory, clearHistory };
}
