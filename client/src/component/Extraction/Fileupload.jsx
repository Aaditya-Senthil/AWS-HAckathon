import React, { useState, useRef, useMemo, useCallback, useEffect } from 'react';
import '../../Styles/Fileupload.css'; 

const FileUpload = React.memo(({ onFilesSelected, value = [] }) => {
  const [files, setFiles] = useState(value);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const fileUrl = useMemo(() => {
    if (files.length > 0) {
      return URL.createObjectURL(files[0]);
    }
    return null;
  }, [files]);


  useEffect(() => {
     setFiles(value);
  }, [value]);


  useEffect(() => {
    return () => {
      if (fileUrl) {
        URL.revokeObjectURL(fileUrl);
      }
    };
  }, [fileUrl]);



  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  }, []);

  const handleFileInput = useCallback((e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
    }
  }, []);

  const handleFiles = useCallback((fileList) => {
    const newFiles = Array.from(fileList);
    setFiles(newFiles);
    if (typeof onFilesSelected === "function") {
      onFilesSelected(newFiles);
    }
  }, [onFilesSelected]);

  const handleFileClick = useCallback(() => {
    fileInputRef.current.click();
  }, []);

  const removeFile = useCallback(() => {
    setFiles([]);
    if (typeof onFilesSelected === "function") {
      onFilesSelected([]);
    }
  }, [onFilesSelected]);

  if (files.length > 0) {
    const file = files[0];

    return (
      <div className="file-upload-container">
        <div className="file-upload-header">
          <h3>{file.name}</h3>
          <button className="file-upload-remove-btn" onClick={removeFile}>
            Remove
          </button>
        </div>

        <div className="file-upload-viewer">
          {file.type === 'application/pdf' ? (
            <iframe
              src={fileUrl}
              className="file-upload-pdf-frame"
              title="PDF Preview"
            />
          ) : (
            <div className="file-upload-preview">
              <div className="file-upload-icon">📄</div>
              <p>File: {file.name}</p>
              <p>Size: {(file.size / 1024 / 1024).toFixed(2)} MB</p>
              <p>Type: {file.type}</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div
      className={`file-upload-area ${isDragging ? 'file-upload-dragging' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleFileClick}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileInput}
        accept=".pdf,.doc,.docx,.txt,.png,.jpg,.jpeg"
        style={{ display: 'none' }}
      />
      <div className="file-upload-content">
        <div className="file-upload-upload-icon">
          <svg
            width="64"
            height="64"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#3b82f6"
            strokeWidth="1.5"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7,10 12,15 17,10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        </div>
        <h3>Drag and drop files here, or click to select files</h3>
        <p>
          Upload a single file to verify your extraction, or run a bulk extraction on multiple files asynchronously.
        </p>
      </div>
    </div>
  );
});

FileUpload.displayName = 'FileUpload';

export default FileUpload;
