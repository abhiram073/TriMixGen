import { useState, useEffect } from 'react';
import { trimixgenAPI, type HealthResponse } from '../services/api';

export function useModelStatus(intervalMs = 5000) {
  const [status, setStatus] = useState<HealthResponse | null>(null);
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    let mounted = true;
    
    const fetchStatus = async () => {
      try {
        const data = await trimixgenAPI.getHealth();
        if (mounted) {
          setStatus(data);
          setIsError(false);
        }
      } catch (err) {
        if (mounted) {
          setIsError(true);
          setStatus(null);
        }
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, intervalMs);
    
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [intervalMs]);

  return { status, isError };
}
