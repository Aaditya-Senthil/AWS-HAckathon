import React, { createContext, useContext, useState, useCallback } from 'react';

const UserContext = createContext();

export const useUser = () => useContext(UserContext);

export const UserProvider = ({ children }) => {
  const STATUS_API_BASE = "https://fn3yrpr3gl.execute-api.ap-south-1.amazonaws.com/Production/status";

  const [userId, setUserId] = useState(() => {
    let storedUserId = localStorage.getItem('userId');
    if (!storedUserId) {
      storedUserId = generateUUID();
      localStorage.setItem('userId', storedUserId);
      console.log('Generated new userId:', storedUserId);
    } else {
      console.log('Retrieved existing userId:', storedUserId);
    }
    return storedUserId;
  });

  // Extraction state management in context
  const [extractionJobs, setExtractionJobs] = useState(new Map());
  const [activeJobId, setActiveJobId] = useState(null);

  // Store extraction job data
  const storeExtractionJob = useCallback((jobId, jobData) => {
    setExtractionJobs(prev => {
      const newJobs = new Map(prev);
      newJobs.set(jobId, {
        ...jobData,
        lastUpdated: Date.now()
      });
      return newJobs;
    });
  }, []);

  // Get extraction job data from context (avoiding GET request)
  const getExtractionJob = useCallback((jobId) => {
    return extractionJobs.get(jobId) || null;
  }, [extractionJobs]);

  // Update job status in context
  const updateJobStatus = useCallback((jobId, status, result = null, message = '') => {
    setExtractionJobs(prev => {
      const newJobs = new Map(prev);
      const existingJob = newJobs.get(jobId);
      if (existingJob) {
        newJobs.set(jobId, {
          ...existingJob,
          status,
          result,
          message,
          lastUpdated: Date.now()
        });
      }
      return newJobs;
    });
  }, []);

  // Remove old jobs (cleanup)
  const cleanupOldJobs = useCallback(() => {
    const now = Date.now();
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours
    
    setExtractionJobs(prev => {
      const newJobs = new Map();
      for (const [jobId, jobData] of prev.entries()) {
        if (now - jobData.lastUpdated < maxAge) {
          newJobs.set(jobId, jobData);
        }
      }
      return newJobs;
    });
  }, []);

  // Get all jobs for current user
  const getUserJobs = useCallback(() => {
    return Array.from(extractionJobs.values()).filter(job => job.userId === userId);
  }, [extractionJobs, userId]);

  // Simulate status polling without actual API calls for completed jobs
  const simulateStatusCheck = useCallback((jobId) => {
    const job = getExtractionJob(jobId);
    if (!job) return null;
    
    // If job is already completed, return cached result
    if (job.status === 'SUCCESS' || job.status === 'FAILED') {
      return {
        success: true,
        status: job.status,
        data: job.result,
        message: job.message
      };
    }
    
    // For pending/processing jobs, you might still want to check actual status
    // or implement a mock progression
    return null;
  }, [getExtractionJob]);

  const contextValue = {
    userId,
    setUserId,
    // Extraction management
    extractionJobs,
    activeJobId,
    setActiveJobId,
    storeExtractionJob,
    getExtractionJob, 
    updateJobStatus,
    cleanupOldJobs,
    getUserJobs,
    simulateStatusCheck,
    STATUS_API_BASE
  };

  return (
    <UserContext.Provider value={contextValue}>
      {children}
    </UserContext.Provider>
  );
};

// Dummy UUID generator
const generateUUID = () => {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
};