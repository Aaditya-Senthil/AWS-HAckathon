import React, { useState, useEffect } from 'react';
import { useUser } from '../../UserContext';

const DocumentHistoryTable = () => {
  const { userId } = useUser();
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expandedRows, setExpandedRows] = useState(new Set());

  useEffect(() => {
    fetchDocuments();
  }, [userId]);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(
        `https://2uz1vqjee9.execute-api.ap-south-1.amazonaws.com/Prod/history?user_id=${userId}`
      );
      const data = await response.json();
      setDocuments(data.data || []);
      console.log('Fetched documents:', data);
      console.log('Fetched results:', data.data);
      data.data.forEach((doc, index) => {
        console.log(`Document ${index + 1}:`, doc.fileName);
        console.log('Extracted data:', doc.data?.data);
      });
    } catch (error) {
      console.error('Error fetching documents:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleRowExpansion = (docId) => {
    setExpandedRows(prev => {
      const newSet = new Set(prev);
      if (newSet.has(docId)) {
        newSet.delete(docId);
      } else {
        newSet.add(docId);
      }
      return newSet;
    });
  };

  const getExtractedFields = (doc) => {
    const extractedData = doc.data?.data;
    if (!extractedData || typeof extractedData !== 'object') return [];
    return Object.keys(extractedData);
  };

  const renderExtractionResults = (doc) => {
    const extractedData = doc.data?.data;
    if (!extractedData) {
      return <div className="extraction-history__no-results">No extraction results available</div>;
    }

    return (
      <div className="extraction-history__results-container">
        <h4 className="extraction-history__results-title">Extracted Information for {doc.fileName}:</h4>
        <div className="extraction-history__results-grid">
          {Object.entries(extractedData).map(([key, value]) => (
            <div key={key} className="extraction-history__result-item">
              <div className="extraction-history__result-label">{key}:</div>
              <div className="extraction-history__result-value">{value}</div>
            </div>
          ))}
        </div>
        {doc.data?.extraction_metadata && (
          <div className="extraction-history__metadata">
            <h5 className="extraction-history__metadata-title">Processing Details:</h5>
            <div className="extraction-history__metadata-stats">
              <span className="extraction-history__metadata-stat">Pages: {doc.data.extraction_metadata.usage?.num_pages_extracted || 'N/A'}</span>
              <span className="extraction-history__metadata-stat">Document Tokens: {doc.data.extraction_metadata.usage?.num_document_tokens || 'N/A'}</span>
              <span className="extraction-history__metadata-stat">Output Tokens: {doc.data.extraction_metadata.usage?.num_output_tokens || 'N/A'}</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  const handleAction = async (action, doc) => {
    switch (action) {
      case 'download':
        window.open(doc.documentUrl, '_blank');
        break;
        
      case 'delete':
        if (window.confirm('Are you sure you want to delete this document?')) {
          console.log('Delete document:', doc);
          // Implement delete API call here
        }
        break;
    }
  };

  if (loading) {
    return (
      <div className="extraction-history__loading-container">
        <div className="extraction-history__loading-spinner"></div>
      </div>
    );
  }

  return (
    <div className="extraction-history">
      <div className="extraction-history__header">
        <h2 className="extraction-history__title">Extraction History</h2>
        <p className="extraction-history__subtitle">View your document extraction results</p>
      </div>

      <div className="extraction-history__table-container">
        <table className="extraction-history__table">
          <thead className="extraction-history__table-header">
            <tr>
              <th className="extraction-history__table-th">File Name</th>
              <th className="extraction-history__table-th">Extracted Fields</th>
              <th className="extraction-history__table-th">Status</th>
              <th className="extraction-history__table-th">Actions</th>
            </tr>
          </thead>
          <tbody className="extraction-history__table-body">
            {documents.map((doc, index) => (
              <React.Fragment key={doc.id}>
                <tr className="extraction-history__table-row">
                  <td className="extraction-history__table-td">
                    <div className="extraction-history__file-name-info">
                      <span className="extraction-history__file-name-text">{doc.fileName}</span>
                      <span className="extraction-history__file-date">{doc.createdAt}</span>
                    </div>
                  </td>
                  <td className="extraction-history__table-td">
                    <div className="extraction-history__extracted-fields">
                      {getExtractedFields(doc).map((field, idx) => (
                        <span key={idx} className="extraction-history__field-tag">
                          {field}
                        </span>
                      ))}
                      {getExtractedFields(doc).length === 0 && (
                        <span className="extraction-history__no-fields">No fields extracted</span>
                      )}
                    </div>
                  </td>
                  <td className="extraction-history__table-td">
                    <span
                      className={`extraction-history__status-badge ${
                        doc.status === 'SUCCESS'
                          ? 'extraction-history__status-badge--success'
                          : doc.status === 'PENDING'
                          ? 'extraction-history__status-badge--pending'
                          : 'extraction-history__status-badge--error'
                      }`}
                    >
                      {doc.status}
                    </span>
                  </td>
                  <td className="extraction-history__table-td">
                    <div className="extraction-history__actions-buttons">
                      <button
                        onClick={() => handleAction('download', doc)}
                        className="extraction-history__action-btn extraction-history__download-btn"
                        title="Download PDF"
                      >
                        📄
                      </button>
                      {doc.status === 'SUCCESS' && (
                        <button
                          onClick={() => toggleRowExpansion(doc.id)}
                          className={`extraction-history__action-btn extraction-history__view-btn ${
                            expandedRows.has(doc.id) ? 'extraction-history__view-btn--active' : ''
                          }`}
                          title={expandedRows.has(doc.id) ? 'Hide Results' : 'View Results'}
                        >
                          {expandedRows.has(doc.id) ? '▼' : '▶'}
                        </button>
                      )}
                    </div>
                  </td>
                </tr>

                {expandedRows.has(doc.id) && (
                  <tr className="extraction-history__expanded-row">
                    <td colSpan="4" className="extraction-history__expanded-content">
                      {renderExtractionResults(doc)}
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>

      {documents.length === 0 && (
        <div className="extraction-history__empty-state">
          <div className="extraction-history__empty-icon">📄</div>
          <p className="extraction-history__empty-message">No documents found</p>
          <p className="extraction-history__empty-subtitle">Start by extracting data from your first document</p>
        </div>
      )}
    </div>
  );
};

export default DocumentHistoryTable;