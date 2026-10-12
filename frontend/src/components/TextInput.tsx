import React from 'react';
import { Type, X, Sparkles } from 'lucide-react';

interface TextInputProps {
  text: string;
  onChange: (val: string) => void;
  disabled?: boolean;
}

const PRESET_TEXTS = [
  { label: 'Happy', text: 'I am absolutely thrilled and overjoyed with these incredible results!' },
  { label: 'Sad', text: 'I feel so devastated, lonely, and hopeless after losing everything.' },
  { label: 'Angry', text: 'This outrageous injustice makes me furious and totally enraged!' },
  { label: 'Fear', text: 'I am terrified and trembling with fear, something dreadful is coming.' },
  { label: 'Disgust', text: 'That putrid, rotten smell is sickening and completely revolting.' },
  { label: 'Surprise', text: 'I was utterly stunned and astounded when the door suddenly burst open!' },
];

export const TextInput: React.FC<TextInputProps> = ({
  text,
  onChange,
  disabled = false,
}) => {
  const charCount = text.length;
  const maxChars = 10000;

  return (
    <div className="input-card">
      <div className="input-card-header">
        <div className="flex-center gap-2">
          <Type size={18} className="text-indigo" />
          <h3 className="card-title">Text Modality</h3>
        </div>
        <div className="flex-center gap-2">
          <span className="char-counter">
            {charCount.toLocaleString()} / {maxChars.toLocaleString()}
          </span>
          {text && (
            <button
              type="button"
              className="icon-btn"
              onClick={() => onChange('')}
              title="Clear text"
              disabled={disabled}
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      <textarea
        className="text-textarea"
        placeholder="Enter text describing an emotional state or experience..."
        value={text}
        onChange={(e) => onChange(e.target.value)}
        maxLength={maxChars}
        disabled={disabled}
        rows={4}
      />

      <div className="presets-row">
        <span className="presets-label">
          <Sparkles size={13} className="text-amber" /> Presets:
        </span>
        <div className="presets-buttons">
          {PRESET_TEXTS.map((p) => (
            <button
              key={p.label}
              type="button"
              className="preset-btn"
              onClick={() => onChange(p.text)}
              disabled={disabled}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
