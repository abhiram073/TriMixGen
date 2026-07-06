import React from 'react';
import { Navbar } from './Navbar';
import { Outlet } from 'react-router-dom';

export const Layout: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-[var(--background)]">
      <Navbar />
      <main className="flex-1 container mx-auto px-4 py-8 max-w-7xl">
        <Outlet />
      </main>
      <footer className="border-t border-[var(--border)] py-6 text-center text-sm text-text-muted">
        TriMixGen &copy; {new Date().getFullYear()} - Advanced Agentic Coding
      </footer>
    </div>
  );
};
