import React, { useRef } from 'react';
import { Mic, Upload, X, Music, CheckCircle2 } from 'lucide-react';

interface AudioUploaderProps {
  file: File | null;
  onFileSelect: (file: File | null) => void;
  disabled?: boolean;
}

export const AudioUploader: React.FC<AudioUploaderProps> = ({
  file,
  onFileSelect,
  disabled = false,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      const validExtensions = ['.wav', '.mp3', '.ogg', '.flac'];
      const ext = '.' + droppedFile.name.split('.').pop()?.toLowerCase();
      if (validExtensions.includes(ext)) {
        onFileSelect(droppedFile);
      }
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div className="input-card">
      <div className="input-card-header">
        <div className="flex-center gap-2">
          <Mic size={18} className="text-violet" />
          <h3 className="card-title">Audio / Speech Modality</h3>
        </div>
        {file && (
          <button
            type="button"
            className="icon-btn"
            onClick={() => onFileSelect(null)}
            title="Remove audio file"
            disabled={disabled}
          >
            <X size={14} />
          </button>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".wav,.mp3,.ogg,.flac,audio/*"
        onChange={handleFileChange}
        style={{ display: 'none' }}
        disabled={disabled}
      />

      {!file ? (
        <div
          className="upload-dropzone"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <Upload size={28} className="dropzone-icon" />
          <p className="dropzone-primary-text">Click to browse or drag & drop audio</p>
          <p className="dropzone-sub-text">Supported: WAV (16 kHz recommended), MP3, OGG, FLAC (Max 25 MB)</p>
          <span className="dataset-note">Evaluated on Emo-DB CNN Acoustic Classifier (4 classes: Neutral, Angry, Happy, Sad)</span>
        </div>
      ) : (
        <div className="file-preview-card">
          <div className="file-info-row">
            <div className="file-badge">
              <Music size={20} className="text-violet" />
            </div>
            <div className="file-details">
              <p className="file-name">{file.name}</p>
              <p className="file-size">{formatFileSize(file.size)}</p>
            </div>
            <CheckCircle2 size={18} className="text-emerald" />
          </div>
          <audio controls className="audio-player" src={URL.createObjectURL(file)} />
        </div>
      )}
    </div>
  );
};
