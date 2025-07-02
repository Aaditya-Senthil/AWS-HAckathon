import React, { useState, useEffect } from 'react';

const Settings = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Load theme preference from localStorage on component mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme) {
      setIsDarkMode(savedTheme === 'dark');
      document.documentElement.classList.toggle('theme-dark', savedTheme === 'dark');
    } else {
      // Use system preference if no saved preference
      setIsDarkMode(prefersDark);
      document.documentElement.classList.toggle('theme-dark', prefersDark);
    }
  }, []);

  // Handle theme toggle
  const handleThemeToggle = () => {
    const newTheme = !isDarkMode;
    setIsDarkMode(newTheme);
    
    // Apply theme to document
    document.documentElement.classList.toggle('theme-dark', newTheme);
    
    // Save preference to localStorage
    localStorage.setItem('theme', newTheme ? 'dark' : 'light');
  };

  return (
    <div className="settings">
      <div className="settings__header">
        <h1 className="settings__title">Settings</h1>
        <p className="settings__subtitle">Customize your application preferences</p>
      </div>

      <div className="settings__content">
        <div className="settings__section">
          <h2 className="settings__section-title">Appearance</h2>
          
          <div className="settings__option">
            <div className="settings__option-info">
              <h3 className="settings__option-title">Theme</h3>
              <p className="settings__option-description">
                Choose between light and dark mode for a comfortable viewing experience
              </p>
            </div>
            
            <div className="settings__toggle-container">
              <span className="settings__toggle-label">
                {isDarkMode ? 'Dark' : 'Light'}
              </span>
              <button 
                className={`settings__toggle ${isDarkMode ? 'settings__toggle--active' : ''}`}
                onClick={handleThemeToggle}
                aria-label={`Switch to ${isDarkMode ? 'light' : 'dark'} mode`}
              >
                <span className="settings__toggle-slider"></span>
              </button>
            </div>
          </div>
        </div>

        <div className="settings__section">
          <h2 className="settings__section-title">General</h2>
          
          <div className="settings__option">
            <div className="settings__option-info">
              <h3 className="settings__option-title">Language</h3>
              <p className="settings__option-description">
                Select your preferred language
              </p>
            </div>
            <select className="settings__select">
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
            </select>
          </div>

          <div className="settings__option">
            <div className="settings__option-info">
              <h3 className="settings__option-title">Notifications</h3>
              <p className="settings__option-description">
                Receive notifications about your extractions
              </p>
            </div>
            <button className="settings__toggle">
              <span className="settings__toggle-slider"></span>
            </button>
          </div>
        </div>

        <div className="settings__section">
          <h2 className="settings__section-title">Account</h2>
          
          <div className="settings__option">
            <div className="settings__option-info">
              <h3 className="settings__option-title">Profile</h3>
              <p className="settings__option-description">
                Manage your account information
              </p>
            </div>
            <button className="settings__button settings__button--secondary">
              Edit Profile
            </button>
          </div>

          <div className="settings__option">
            <div className="settings__option-info">
              <h3 className="settings__option-title">API Keys</h3>
              <p className="settings__option-description">
                Manage your API access keys
              </p>
            </div>
            <button className="settings__button settings__button--secondary">
              Manage Keys
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;