import React, { useRef } from 'react';
import { Eye, Upload, X, Image as ImageIcon } from 'lucide-react';

interface ImageUploaderProps {
  file: File | null;
  onFileSelect: (file: File | null) => void;
  disabled?: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
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
      const validExtensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'];
      const ext = '.' + droppedFile.name.split('.').pop()?.toLowerCase();
      if (validExtensions.includes(ext)) {
        onFileSelect(droppedFile);
      }
    }
  };

  return (
    <div className="input-card vision-card">
      <div className="input-card-header">
        <div className="flex-center gap-2">
          <Eye size={18} className="text-cyan" />
          <h3 className="card-title">Vision Modality</h3>
          <span className="badge-ready">AlexNet Trained (7-Class)</span>
        </div>
        {file && (
          <button
            type="button"
            className="icon-btn"
            onClick={() => onFileSelect(null)}
            title="Remove image"
            disabled={disabled}
          >
            <X size={14} />
          </button>
        )}
      </div>

      {/* Pipeline Description Notice */}
      <div className="vision-notice-banner vision-active-banner">
        <ImageIcon size={18} className="text-cyan flex-shrink-0" />
        <div className="notice-content">
          <strong>Vision Emotion Pipeline (Trained AlexNet):</strong>
          <p>
            Faces detected via OpenCV Haar Cascade, centered & normalized to 224&times;224 RGB crops, and classified into 7 core emotions using genuine trained weights (<code>model_image_new50.pth</code>).
          </p>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,.bmp,image/*"
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
          <Upload size={28} className="dropzone-icon text-cyan" />
          <p className="dropzone-primary-text">Upload facial image for emotion classification</p>
          <p className="dropzone-sub-text">JPG, PNG, WebP (Max 10 MB)</p>
          <span className="dataset-note">
            Full 7-class Ekman facial emotion recognition with confidence and probability distribution.
          </span>
        </div>
      ) : (
        <div className="image-preview-wrapper">
          <img
            src={URL.createObjectURL(file)}
            alt="Facial input preview"
            className="image-preview"
          />
          <div className="image-preview-meta">
            <span className="file-name flex-center gap-1">
              <ImageIcon size={14} /> {file.name}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
