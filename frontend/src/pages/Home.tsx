import React from 'react';
import { Link } from 'react-router-dom';
import { Bot, Sparkles, Code2 } from 'lucide-react';

export const Home: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] text-center max-w-3xl mx-auto px-4">
      <div className="mb-8 relative">
        <div className="absolute inset-0 bg-primary blur-[100px] opacity-20 rounded-full"></div>
        <Bot size={80} className="text-primary relative z-10" />
      </div>
      
      <h1 className="text-5xl font-extrabold tracking-tight mb-6">
        Generate <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">Telugu-English</span> Code-Mixed Text
      </h1>
      
      <p className="text-xl text-text-muted mb-10 leading-relaxed">
        TriMixGen is an advanced curriculum-trained generation model. It seamlessly controls Sentiment, Formality, and Code-Mixing density natively in Romanized Telugu.
      </p>
      
      <div className="flex flex-col sm:flex-row gap-4 mb-16">
        <Link 
          to="/generate" 
          className="flex items-center justify-center gap-2 px-8 py-4 bg-primary hover:bg-indigo-500 text-white font-semibold rounded-xl transition-all hover:scale-105 shadow-lg shadow-primary/30"
        >
          <Sparkles size={20} />
          Launch Generator
        </Link>
        <Link 
          to="/about" 
          className="flex items-center justify-center gap-2 px-8 py-4 bg-[var(--surface)] hover:bg-gray-100 dark:hover:bg-gray-800 text-[var(--text)] font-semibold rounded-xl border border-[var(--border)] transition-all"
        >
          <Code2 size={20} />
          View Architecture
        </Link>
      </div>
    </div>
  );
};
