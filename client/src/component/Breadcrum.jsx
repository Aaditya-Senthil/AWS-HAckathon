import React, { useState, useEffect } from 'react';
import { Link, useLocation, useParams } from 'react-router-dom';

const Breadcrumb = () => {
  const location = useLocation();
  const params = useParams();
  const [isDark, setIsDark] = useState(false);
  
  const pathnames = location.pathname.split('/').filter(x => x);
  
  // Load theme preference from localStorage on component mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme) {
      setIsDark(savedTheme === 'dark');
      document.documentElement.classList.toggle('theme-dark', savedTheme === 'dark');
    } else {
      // Use system preference if no saved preference
      setIsDark(prefersDark);
      document.documentElement.classList.toggle('theme-dark', prefersDark);
    }
  }, []);

  // Handle theme toggle
  const handleThemeToggle = () => {
    const newTheme = !isDark;
    setIsDark(newTheme);
    
    // Apply theme to document
    document.documentElement.classList.toggle('theme-dark', newTheme);
    
    // Save preference to localStorage
    localStorage.setItem('theme', newTheme ? 'dark' : 'light');
  };
  
  const getCurrentPageTitle = () => {
    if (pathnames.length === 0) return 'Home';
    return pathnames[pathnames.length - 1].charAt(0).toUpperCase() + pathnames[pathnames.length - 1].slice(1);
  };

return (
    <div className="breadcrumb">
      <div className="breadcrumb__header">
        <nav className="breadcrumb__nav">
          {pathnames.length > 0 ? (
            <>
              {pathnames.map((name, index) => {
                const routeTo = `/${pathnames.slice(0, index + 1).join('/')}`;
                const isLast = index === pathnames.length - 1;

                return isLast ? (
                  <span key={name} className="breadcrumb__current">{name.charAt(0).toUpperCase() + name.slice(1)}</span>
                ) : (
                  <React.Fragment key={name}>
                    <Link to={routeTo} className="breadcrumb__link">{name.charAt(0).toUpperCase() + name.slice(1)}</Link>
                    <span className="breadcrumb__separator"> › </span>
                  </React.Fragment>
                );
              })}
            </>
          ) : (
            <span className="breadcrumb__current">Home</span>
          )}
        </nav>
        
        <button 
          className="breadcrumb__theme-toggle"
          onClick={handleThemeToggle}
          aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
        >
          {isDark ? '☀️ Light Mode' : '🌙 Dark Mode'}
        </button>
      </div>
      
      <h1 className="breadcrumb__title">{params.name || getCurrentPageTitle()}</h1>
    </div>
  );
};

export default Breadcrumb;