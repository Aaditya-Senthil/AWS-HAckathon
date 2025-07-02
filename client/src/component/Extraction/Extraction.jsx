import React, { useState } from "react";
import { useUser } from '../../UserContext';
import FileUpload from "./Fileupload";

const Extraction = () => {
  const { userId } = useUser();
  
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [newField, setNewField] = useState({
    FieldName: "",
    Datatype: "string",
  });
  const [fields, setFields] = useState([]);
  const [extractionResult, setExtractionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");

  const EXTRACTION_API = "https://fn3yrpr3gl.execute-api.ap-south-1.amazonaws.com/Production/extraction";
  const STATUS_API_BASE = "https://fn3yrpr3gl.execute-api.ap-south-1.amazonaws.com/Production/status";

  // Field management
  const handleAddField = () => {
    if (newField.FieldName.trim()) {
      setFields([...fields, { name: newField.FieldName, type: newField.Datatype }]);
      setNewField({ FieldName: "", Datatype: "string" });
    }
  };

  const handleRemoveField = (index) => {
    setFields(fields.filter((_, i) => i !== index));
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewField(prev => ({ ...prev, [name]: value }));
  };

  // Generate schema
  const generateSchema = () => {
    const properties = {};
    fields.forEach(field => {
      properties[field.name] = { type: field.type };
    });
    return {
      additionalProperties: false,
      properties,
      required: fields.map(f => f.name),
      type: "object"
    };
  };

  // Simple API call
  const makeApiCall = async (url, method, body = null) => {
    const config = {
      method,
      headers: {
        'Content-Type': 'application/json',
        'user-id': userId
      },
      mode: 'cors',
    };

    if (body) {
      config.body = JSON.stringify(body);
    }

    const response = await fetch(url, config);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  };

  // Check status
  const checkStatus = async (jobId) => {
    try {
      const data = await makeApiCall(`${STATUS_API_BASE}/${jobId}`, 'GET');
      console.log('Status response:', data);
      
      if (data.success && data.overall_status === 'SUCCESS' && data.data) {
        setExtractionResult(data.data);
        setIsLoading(false);
        setStatusMessage('Extraction completed!');
        return true; // Stop polling
      } else if (data.success && data.overall_status === 'FAILED') {
        setIsLoading(false);
        setStatusMessage('Extraction failed. Please try again.');
        return true; // Stop polling
      } else {
        setStatusMessage(`Status: ${data.overall_status || 'Processing'}`);
        return false; // Continue polling
      }
    } catch (error) {
      console.error('Status check failed:', error);
      setStatusMessage(`Status check failed: ${error.message}`);
      return false;
    }
  };

  // Simple polling
  const startPolling = (jobId) => {
    let attempts = 0;
    const maxAttempts = 4;
    
    const poll = async () => {
      attempts++;
      console.log(`Poll attempt ${attempts}/${maxAttempts}`);
      
      const shouldStop = await checkStatus(jobId);
      
      if (shouldStop || attempts >= maxAttempts) {
        if (attempts >= maxAttempts) {
          setIsLoading(false);
          setStatusMessage('Timeout: Please check status manually.');
        }
        return;
      }
      
      // Continue polling
      setTimeout(poll, 5000);
    };
    
    // Start first poll after 3 seconds
    setTimeout(poll, 3000);
  };

  // Start extraction
  const handleExtraction = async () => {
    if (!uploadedFiles.length) {
      alert("Please upload a file first");
      return;
    }

    if (fields.length === 0) {
      alert("Please add at least one field to extract");
      return;
    }

    setIsLoading(true);
    setExtractionResult(null);
    setStatusMessage("Starting extraction...");
    setCurrentJobId(null);

    try {
      const file = uploadedFiles[0];
      const fileContent = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      const base64Content = fileContent.split(',')[1];

      const payload = {
        filename: file.name,
        file_content: base64Content,
        user_id: userId,
        config: {
          extraction_mode: "ACCURATE",
          data_schema: generateSchema(),
          handle_missing: false
        }
      };

      const data = await makeApiCall(EXTRACTION_API, 'POST', payload);
      
      if (data.job_id) {
        setCurrentJobId(data.job_id);
        setStatusMessage(`Extraction started. Job ID: ${data.job_id}`);
        startPolling(data.job_id);
      } else {
        throw new Error('No job ID received');
      }
    } catch (error) {
      console.error('Extraction failed:', error);
      setStatusMessage(`Extraction failed: ${error.message}`);
      setIsLoading(false);
      alert(`Extraction failed: ${error.message}`);
    }
  };

  // Manual status check
  const handleStatusCheck = () => {
    if (currentJobId) {
      checkStatus(currentJobId);
    } else {
      alert("No active job to check");
    }
  };

  // Reset
  const handleReset = () => {
    setIsLoading(false);
    setExtractionResult(null);
    setStatusMessage("");
    setCurrentJobId(null);
  };

  return (
<div className="extraction">
  <div className="extraction__tabs">
    <button className="extraction__tab extraction__tab--active">Schema</button>
  </div>

  <div className="extraction__content">
    <div className="extraction__panel--left">
      <div className="extraction__agent-config">
        <h2>Agent Configuration</h2>

        <div className="extraction__fields">
          {fields.map((field, index) => (
            <div className="extraction__field" key={index}>
              <div className="extraction__field-name">{field.name}</div>
              <div className="extraction__field-type">{field.type}</div>
              <button
                className="extraction__remove-btn"
                onClick={() => handleRemoveField(index)}
              >
                ✕
              </button>
            </div>
          ))}
        </div>

        <div className="extraction__new-field">
          <input
            type="text"
            name="FieldName"
            value={newField.FieldName}
            onChange={handleInputChange}
            placeholder="Enter Field Name"
          />
          <select
            name="Datatype"
            value={newField.Datatype}
            onChange={handleInputChange}
          >
            <option value="string">string</option>
            <option value="number">number</option>
            <option value="boolean">boolean</option>
            <option value="object">object</option>
            <option value="array">array</option>
          </select>
        </div>

        <button className="extraction__add-field-btn" onClick={handleAddField}>
          + Add Field
        </button>

        {(isLoading || statusMessage) && (
          <div className="extraction__status">
            <h3>Extraction Status:</h3>
            {currentJobId && <p><strong>Job ID:</strong> {currentJobId}</p>}
            {statusMessage && <p><strong>Message:</strong> {statusMessage}</p>}
            {isLoading && (
              <div className="extraction__loading">
                <div className="extraction__spinner"></div>
                <span>Processing...</span>
              </div>
            )}
          </div>
        )}

        <div className="extraction__actions">
          <button 
            className="extraction__action-btn"
            onClick={handleExtraction}
            disabled={isLoading}
          >
            {isLoading ? 'Processing...' : 'Start Extraction'}
          </button>

          {currentJobId && (
            <>
              <button 
                className="extraction__status-btn"
                onClick={handleStatusCheck}
                disabled={isLoading}
              >
                Check Status
              </button>
              <button 
                className="extraction__reset-btn"
                onClick={handleReset}
              >
                Reset
              </button>
            </>
          )}
        </div>

        {extractionResult && extractionResult.data && (
          <div className="extraction__results">
            <h3>Extraction Results:</h3>
            <div className="extraction__results-container">
              {Object.entries(extractionResult.data).map(([question, answer]) => (
                <div key={question} className="extraction__result-item">
                  <div className="extraction__question"><strong>{question}</strong></div>
                  <div className="extraction__answer">{answer}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>

    <div className="extraction__panel--right">
      <FileUpload
        onFilesSelected={setUploadedFiles}
        value={uploadedFiles}
      />
    </div>
  </div>
</div>

  );
};

export default Extraction;