
import React, { useState } from 'react';
import axios from 'axios';

const DocumentUpload = ({ onAnalysisComplete, onUploadStart, loading }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
      const allowedExtensions = ['.pdf', '.docx', '.txt'];
      const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

      if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
        setError('Please select a PDF, DOCX, or TXT file.');
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first.');
      return;
    }

    onUploadStart();
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      onAnalysisComplete(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during analysis.');
      onAnalysisComplete(null);
    }
  };

  return (
    <div className="text-center">
      <h4 className="mb-4">Upload Contract Document</h4>
      <p className="text-muted mb-4">Select a PDF, DOCX, or TXT file to extract key clauses and identify compliance risks</p>

      <div className="mb-4">
        <input
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleFileSelect}
          className="form-control form-control-lg"
          disabled={loading}
        />
        {selectedFile && (
          <small className="text-success mt-1 d-block">
            Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
          </small>
        )}
      </div>

      {error && (
        <div className="alert alert-danger mb-3">
          {error}
        </div>
      )}

      <button
        onClick={handleUpload}
        disabled={!selectedFile || loading}
        className="btn btn-primary btn-lg px-5"
      >
        {loading ? (
          <>
            <span className="spinner-border spinner-border-sm me-2" role="status"></span>
            Analyzing...
          </>
        ) : (
          'Extract Key Clauses & Analyze Risks'
        )}
      </button>
    </div>
  );
};

export default DocumentUpload;
