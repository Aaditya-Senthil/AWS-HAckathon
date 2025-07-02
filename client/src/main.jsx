import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'
import App from './App.jsx'

const root = ReactDOM.createRoot(document.getElementById('root'));

// Initialize theme before rendering
const initializeTheme = () => {
  const savedTheme = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  if (savedTheme) {
    document.documentElement.classList.toggle('theme-dark', savedTheme === 'dark');
  } else {
    // Use system preference if no saved preference
    document.documentElement.classList.toggle('theme-dark', prefersDark);
    localStorage.setItem('theme', prefersDark ? 'dark' : 'light');
  }
};

// Initialize theme immediately
initializeTheme();


root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

