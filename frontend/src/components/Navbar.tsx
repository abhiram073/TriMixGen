import React from 'react';
import { useModelStatus } from '../hooks/useModelStatus';
import { Activity, ServerCrash, CheckCircle2 } from 'lucide-react';
import { Link } from 'react-router-dom';

export const Navbar: React.FC = () => {
  const { status, isError } = useModelStatus();

  return (
    <nav className="glass sticky top-0 z-50 flex items-center justify-between px-6 py-4">
      <div className="flex items-center gap-6">
        <Link to="/" className="text-xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
          TriMixGen
        </Link>
        <div className="hidden md:flex gap-4">
          <Link to="/" className="text-sm font-medium text-text-muted hover:text-text transition-colors">Home</Link>
          <Link to="/generate" className="text-sm font-medium text-text-muted hover:text-text transition-colors">Generator</Link>
          <Link to="/about" className="text-sm font-medium text-text-muted hover:text-text transition-colors">About</Link>
        </div>
      </div>
      
      <div className="flex items-center gap-3">
        {isError ? (
          <div className="flex items-center gap-2 text-red-500 bg-red-500/10 px-3 py-1.5 rounded-full text-xs font-medium" role="status" aria-label="Backend Offline">
            <ServerCrash size={14} />
            <span className="hidden sm:inline">Offline</span>
          </div>
        ) : !status ? (
          <div className="flex items-center gap-2 text-yellow-500 bg-yellow-500/10 px-3 py-1.5 rounded-full text-xs font-medium" role="status" aria-label="Backend Loading">
            <Activity size={14} className="animate-pulse" />
            <span className="hidden sm:inline">Connecting</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-green-500 bg-green-500/10 px-3 py-1.5 rounded-full text-xs font-medium" role="status" aria-label="Backend Online">
            <CheckCircle2 size={14} />
            <span className="hidden sm:inline">Connected</span>
          </div>
        )}
      </div>
    </nav>
  );
};
