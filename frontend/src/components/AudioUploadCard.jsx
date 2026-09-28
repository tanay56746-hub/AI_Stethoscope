import React, { useState, useRef } from 'react';

/**
 * AudioUploadCard Component
 * Prominent card supporting drag-and-drop WAV upload, file browsing,
 * sample selection, and audio metadata preview.
 */
export default function AudioUploadCard({
  onAnalyze,
  analyzing,
  selectedFile,
  setSelectedFile,
  selectedSample,
  setSelectedSample,
  demoSamples,
}) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (file) => {
    setUploadError(null);
    if (!file.name.toLowerCase().endsWith('.wav')) {
      setUploadError('Invalid file format. Please select a valid WAV (.wav) audio recording.');
      return;
    }

    setSelectedSample(null);
    setSelectedFile(file);
  };

  const handleSelectSample = (sample) => {
    setUploadError(null);
    setSelectedFile(null);
    setSelectedSample(sample);
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '—';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title-group">
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--medical-teal)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
          </svg>
          <div className="card-title">Audio Input</div>
        </div>
        <span className="card-badge">PCM WAV</span>
      </div>

      {uploadError && (
        <div className="alert-box alert-error">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{uploadError}</span>
        </div>
      )}

      {/* Drag & Drop Upload Zone */}
      <div
        className={`upload-dropzone ${isDragActive ? 'drag-active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,audio/wav"
          className="file-input-hidden"
          onChange={handleFileInput}
        />
        <div className="dropzone-icon">
          <svg
            width="36"
            height="36"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>
        <div className="dropzone-text-primary">Drag & drop WAV audio recording here</div>
        <div className="dropzone-text-subtle">or click to browse local files</div>
      </div>

      {/* Demo Sample Selector */}
      {demoSamples && demoSamples.length > 0 && (
        <div className="demo-sample-section">
          <div className="demo-sample-title">Or test with verified project sample:</div>
          {demoSamples.map((sample) => (
            <button
              key={sample.id}
              type="button"
              className="sample-button"
              onClick={() => handleSelectSample(sample)}
              disabled={analyzing}
            >
              <span>{sample.name}</span>
              <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)' }}>Select Sample →</span>
            </button>
          ))}
        </div>
      )}

      {/* Selected Audio Box */}
      {(selectedFile || selectedSample) && (
        <div className="selected-audio-card">
          <div className="selected-audio-header">
            <span style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-muted)' }}>
              Selected Audio:
            </span>
            <span className="spec-chip">WAV</span>
          </div>

          <div className="selected-audio-name">
            {selectedFile ? selectedFile.name : selectedSample.filename}
          </div>

          <div className="selected-audio-specs" style={{ marginTop: '8px' }}>
            <span className="spec-chip">
              Size: {selectedFile ? formatFileSize(selectedFile.size) : formatFileSize(selectedSample.size_bytes)}
            </span>
            <span className="spec-chip">5.0 sec window</span>
            <span className="spec-chip">16 kHz target</span>
          </div>
        </div>
      )}

      {/* Analyze Button */}
      <button
        type="button"
        className="btn-analyze"
        disabled={analyzing || (!selectedFile && !selectedSample)}
        onClick={onAnalyze}
      >
        {analyzing ? (
          <>
            <span className="spinner" />
            <span>ANALYZING AUDIO...</span>
          </>
        ) : (
          <>
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            <span>ANALYZE AUDIO</span>
          </>
        )}
      </button>
    </div>
  );
}
