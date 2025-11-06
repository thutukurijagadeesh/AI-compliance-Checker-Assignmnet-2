import React, { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import DocumentUpload from './components/DocumentUpload';
import AnalysisResults from './components/AnalysisResults';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');

  const loadingMessages = [
    "🔍 Scanning your document for compliance magic...",
    "⚖️ Analyzing clauses and identifying risks...",
    "🛡️ Ensuring GDPR and HIPAA compliance...",
    "📋 Extracting key terms and recommendations...",
    "✨ Almost there! Finalizing your compliance report..."
  ];

  useEffect(() => {
    let interval;
    if (loading) {
      setLoadingMessage(loadingMessages[Math.floor(Math.random() * loadingMessages.length)]);
      interval = setInterval(() => {
        setLoadingMessage(loadingMessages[Math.floor(Math.random() * loadingMessages.length)]);
      }, 2000); // Change message every 2 seconds
    }
    return () => clearInterval(interval);
  }, [loading]);

  const handleAnalysisComplete = (analysisResults) => {
    setResults(analysisResults);
    setLoading(false);
  };

  const handleUploadStart = () => {
    setLoading(true);
    setResults(null);
  };

  return (
    <div className="App">
      <div className="hero-section">
        <div className="container text-center">
          <h1 className="main-heading">AI Powered Compliance Checker</h1>
          <p className="subtitle">Regulatory compliance analysis for contracts</p>
          <div className="d-flex justify-content-center mt-4">
            <button className="btn me-3 hipaa-btn" disabled>HIPAA</button>
            <button className="btn gdpr-btn" disabled>GDPR</button>
          </div>
        </div>
      </div>

      <div className="container mt-4">
        <div className="row justify-content-center">
          <div className="col-lg-10">
            <div className="card">
              <div className="card-body p-4">
                <DocumentUpload
                  onAnalysisComplete={handleAnalysisComplete}
                  onUploadStart={handleUploadStart}
                  loading={loading}
                />

                {loading && (
                  <div className="text-center mt-4">
                    <div className="spinner-border text-primary" role="status" style={{ width: '3rem', height: '3rem' }}>
                      <span className="visually-hidden">Analyzing...</span>
                    </div>
                    <p className="mt-3 text-primary fw-bold" style={{ fontSize: '1.1rem', animation: 'pulse 2s infinite' }}>
                      {loadingMessage}
                    </p>
                    <p className="text-muted">This may take a few seconds...</p>
                  </div>
                )}

                {results && !loading && (
                  <AnalysisResults results={results} />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
