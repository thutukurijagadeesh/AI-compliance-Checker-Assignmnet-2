
import React from 'react';
import axios from 'axios';
import { Bar, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend as ChartLegend,
  ArcElement,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  ChartTooltip,
  ChartLegend,
  ArcElement
);

const AnalysisResults = ({ results }) => {
  if (!results) return null;

  // Check if document is not a contract
  if (results.is_contract === false) {
    return (
      <div className="mt-5">
        <div className="alert alert-warning" role="alert">
          <h4 className="alert-heading">Document Type Not Supported</h4>
          <p>{results.message}</p>
          <hr />
          <p className="mb-0">Please upload legal contract or agreement documents for compliance analysis.</p>
        </div>
      </div>
    );
  }

  // Function to highlight key terms in clauses
  const highlightKeyTerms = (text) => {
    const keyTerms = [
      'data processing', 'personal data', 'privacy', 'consent', 'breach',
      'security', 'retention', 'transfer', 'rights', 'liability',
      'termination', 'confidentiality', 'intellectual property',
      'gdpr', 'hipaa', 'compliance', 'regulation'
    ];

    let highlightedText = text;
    keyTerms.forEach(term => {
      const regex = new RegExp(`(${term})`, 'gi');
      highlightedText = highlightedText.replace(regex, '<mark>$1</mark>');
    });

    return <span dangerouslySetInnerHTML={{ __html: highlightedText }} />;
  };

  // Function to handle document download
  const handleDownload = async () => {
    if (!results.updated_filename) return;

    try {
      const response = await axios.get(`http://localhost:8000/download/${results.updated_filename}`, {
        responseType: 'blob',
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', results.updated_filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading file:', error);
      alert('Error downloading the updated document. Please try again.');
    }
  };

  // Prepare data for charts
  const riskSeverityData = results.risks ? results.risks.reduce((acc, risk) => {
    const severity = risk.severity || 'Medium';
    acc[severity] = (acc[severity] || 0) + 1;
    return acc;
  }, {}) : {};

  const riskChartData = Object.entries(riskSeverityData).map(([severity, count]) => ({
    name: severity,
    value: count,
    color: severity === 'High' ? '#dc3545' : severity === 'Medium' ? '#ffc107' : '#28a745'
  }));

  const summaryData = [
    { name: 'Key Clauses', value: results.key_clauses ? results.key_clauses.length : 0, color: '#007bff' },
    { name: 'Risks', value: results.risks ? results.risks.length : 0, color: '#dc3545' },
    { name: 'Missing Clauses', value: results.missing_clauses ? results.missing_clauses.length : 0, color: '#ffc107' },
    { name: 'Recommendations', value: results.recommended_clauses ? results.recommended_clauses.length : 0, color: '#28a745' }
  ];

  return (
    <div className="mt-5">
      <h3 className="mb-4">Analysis Results for {results.filename}</h3>

      {/* Download Button */}
      {results.updated_filename && (
        <div className="mb-4">
          <button
            className="btn btn-success btn-lg"
            onClick={handleDownload}
          >
            <i className="bi bi-download me-2"></i>
            Download Updated Document with High-Risk Issues and Recommendations
          </button>
          <p className="text-muted mt-2">
            The updated document includes the original content plus sections for high-risk compliance issues identified and recommended clauses to address them.
          </p>
        </div>
      )}

      {/* Charts Section */}
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Analysis Overview</h5>
            </div>
            <div className="card-body">
              <Bar
                data={{
                  labels: summaryData.map(item => item.name),
                  datasets: [{
                    label: 'Count',
                    data: summaryData.map(item => item.value),
                    backgroundColor: summaryData.map(item => item.color),
                    borderColor: summaryData.map(item => item.color),
                    borderWidth: 1,
                  }],
                }}
                options={{
                  responsive: true,
                  plugins: {
                    legend: {
                      position: 'top',
                    },
                    title: {
                      display: true,
                      text: 'Analysis Metrics Overview',
                    },
                  },
                }}
              />
            </div>
          </div>
        </div>

        <div className="col-md-6">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Risk Severity Distribution</h5>
            </div>
            <div className="card-body">
              <Pie
                data={{
                  labels: riskChartData.map(item => item.name),
                  datasets: [{
                    data: riskChartData.map(item => item.value),
                    backgroundColor: riskChartData.map(item => item.color),
                    borderColor: riskChartData.map(item => item.color),
                    borderWidth: 1,
                  }],
                }}
                options={{
                  responsive: true,
                  plugins: {
                    legend: {
                      position: 'top',
                    },
                    title: {
                      display: true,
                      text: 'Risk Severity Breakdown',
                    },
                  },
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Key Clauses Extracted</h5>
            </div>
            <div className="card-body" style={{ maxHeight: '400px', overflowY: 'auto' }}>
              {results.key_clauses && results.key_clauses.length > 0 ? (
                <div className="list-group list-group-flush">
                  {results.key_clauses.map((clause, index) => (
                    <div key={index} className="list-group-item">
                      <small className="text-muted">Clause {index + 1}:</small>
                      <p className="mb-0">{highlightKeyTerms(clause)}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted">No key clauses identified in the document.</p>
              )}
            </div>
          </div>
        </div>

        <div className="col-md-6">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Compliance Risks Identified</h5>
            </div>
            <div className="card-body" style={{ maxHeight: '400px', overflowY: 'auto' }}>
              {results.risks && results.risks.length > 0 ? (
                <div className="list-group list-group-flush">
                  {results.risks.map((risk, index) => (
                    <div key={index} className="list-group-item">
                      <div className="d-flex justify-content-between align-items-start">
                        <div>
                          <strong className="text-danger">{risk.risk}</strong>
                          <p className="mb-1 small">{risk.description}</p>
                        </div>
                        <span className={`badge ${risk.severity === 'High' ? 'bg-danger' : 'bg-warning'}`}>
                          {risk.severity} Risk
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-success">No major compliance risks identified.</p>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="row mt-4">
        <div className="col-md-6">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Missing Clauses</h5>
            </div>
            <div className="card-body" style={{ maxHeight: '400px', overflowY: 'auto' }}>
              {results.missing_clauses && results.missing_clauses.length > 0 ? (
                <div className="list-group list-group-flush">
                  {results.missing_clauses.map((clause, index) => (
                    <div key={index} className="list-group-item">
                      <div className="d-flex justify-content-between align-items-start">
                        <div>
                          <strong className="text-warning">{clause.clause}</strong>
                          <p className="mb-1 small">{clause.description}</p>
                        </div>
                        <span className={`badge ${clause.importance === 'High' ? 'bg-danger' : 'bg-warning'}`}>
                          {clause.importance} Importance
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-success">All standard compliance clauses appear to be present.</p>
              )}
            </div>
          </div>
        </div>

        <div className="col-md-6">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Recommended Clauses</h5>
            </div>
            <div className="card-body" style={{ maxHeight: '400px', overflowY: 'auto' }}>
              {results.recommended_clauses && results.recommended_clauses.length > 0 ? (
                <div className="list-group list-group-flush">
                  {results.recommended_clauses.map((clause, index) => (
                    <div key={index} className="list-group-item">
                      <small className="text-muted">Recommendation {index + 1}:</small>
                      <p className="mb-0">{highlightKeyTerms(clause)}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted">No specific recommendations generated based on the analysis.</p>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4">
        <div className="card">
          <div className="card-body">
            <h6>Document Summary</h6>
            <div className="row">
              <div className="col-sm-6">
                <p className="mb-1"><strong>Filename:</strong> {results.filename}</p>
                <p className="mb-0"><strong>Text Length:</strong> {results.text_length} characters</p>
              </div>
              <div className="col-sm-6">
                <p className="mb-1"><strong>Key Clauses Found:</strong> {results.key_clauses ? results.key_clauses.length : 0}</p>
                <p className="mb-1"><strong>Risks Identified:</strong> {results.risks ? results.risks.length : 0}</p>
                <p className="mb-1"><strong>Missing Clauses:</strong> {results.missing_clauses ? results.missing_clauses.length : 0}</p>
                <p className="mb-0"><strong>Recommendations:</strong> {results.recommended_clauses ? results.recommended_clauses.length : 0}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;
