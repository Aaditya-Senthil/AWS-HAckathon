import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import Sidebar from './component/SideBar';
import MainContent from './component/Maincontent';
import { UserProvider } from './UserContext';

import './App.css';

function App() {
  return (
    <UserProvider>
      <Router>
        <div className="app-container">
          <Sidebar />
          <MainContent />
        </div>
      </Router>
    </UserProvider>
  );
}

export default App;

