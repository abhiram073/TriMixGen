import React from 'react';

const PROMPTS = [
  { label: 'Movie Review (Positive)', prompt: 'Write a positive Telugu-English movie review.' },
  { label: 'Restaurant (Negative)', prompt: 'Write a negative Telugu-English restaurant review.' },
  { label: 'College Exams (Casual)', prompt: 'Use a casual tone to discuss college exams in Telugu-English.' },
  { label: 'Tech Discussion', prompt: 'Discuss the new smartphone in Telugu-English.' },
  { label: 'Social Media Reply', prompt: 'Reply casually in code-mixed Telugu to this tweet.' },
];

interface Props {
  onSelect: (prompt: string) => void;
}

export const PromptGallery: React.FC<Props> = ({ onSelect }) => {
  return (
    <div className="mb-6">
      <h3 className="text-sm font-semibold text-text-muted mb-3 uppercase tracking-wider">Example Prompts</h3>
      <div className="flex flex-wrap gap-2">
        {PROMPTS.map((p, idx) => (
          <button
            key={idx}
            onClick={() => onSelect(p.prompt)}
            className="text-xs font-medium px-3 py-1.5 rounded-full border border-[var(--border)] bg-[var(--surface)] hover:border-primary hover:text-primary transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1"
            aria-label={`Select prompt: ${p.label}`}
          >
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
};
