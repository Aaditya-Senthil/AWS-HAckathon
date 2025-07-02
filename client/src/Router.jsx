import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Extraction from './component/Extraction/Extraction';
import Analytics from './component/Analytics/Analytics';
import Extraction_History from './component/Extraction/Extraction_history';
import Settings from './component/Setting';

const Router = () => {
  return (
<Routes>
  <Route path="/" element={<Extraction />} />
  <Route path="/extraction" element={<Extraction />} />
  <Route path="/extraction/:name" element={<Extraction />} />
  <Route path="/analytics" element={<Analytics />} />
  <Route path="/History" element={<Extraction_History />} />
  <Route path="/settings" element={<Settings />} />
  <Route path="*" element={<Extraction />} /> 
</Routes>

  );
};
export default Router;

