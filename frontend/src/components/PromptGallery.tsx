import React from 'react';

const PROMPTS = [
  { label: 'Movie Review (Positive)', s1: 'Film bahut badhiya thi, mujhe pasand aayi.', s2: 'Ami chhobita dekhe khub anondo peyechhi.', s3: 'The movie was really great, I enjoyed it.' },
  { label: 'Restaurant (Negative)', s1: 'Khana bilkul bakwas tha aur service kharab thi.', s2: 'Khabar akdom bhalo chhilo na ar service kharap chhilo.', s3: 'The food was terrible and the service was bad.' },
  { label: 'College Exams (Casual)', s1: 'Bhai, kal exam hai aur maine kuch nahi padha.', s2: 'Bhai, kal porikkha ar ami kichui porini.', s3: 'Bro, there is an exam tomorrow and I have not studied anything.' },
];

interface Props {
  onSelect: (s1: string, s2: string, s3: string) => void;
}

export const PromptGallery: React.FC<Props> = ({ onSelect }) => {
  return (
    <div className="mb-6">
      <h3 className="text-sm font-semibold text-text-muted mb-3 uppercase tracking-wider">Example Prompts</h3>
      <div className="flex flex-wrap gap-2">
        {PROMPTS.map((p, idx) => (
          <button
            key={idx}
            onClick={() => onSelect(p.s1, p.s2, p.s3)}
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
