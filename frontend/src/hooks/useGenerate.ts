import { useState } from 'react';
import { trimixgenAPI, type GenerateRequest, type GenerateResponse } from '../services/api';

export function useGenerate() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateResponse | null>(null);

  const generate = async (request: GenerateRequest) => {
    setLoading(true);
    setError(null);
    try {
      const data = await trimixgenAPI.generate(request);
      setResult(data);
      return data;
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Generation failed';
      setError(msg);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
  };

  return { generate, loading, error, result, reset };
}
