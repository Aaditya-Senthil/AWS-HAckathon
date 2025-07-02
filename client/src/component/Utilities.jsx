// utils/theme.js
export const initializeTheme = () => {
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

export const toggleTheme = () => {
  const isDark = document.documentElement.classList.contains('theme-dark');
  const newTheme = !isDark;
  
  document.documentElement.classList.toggle('theme-dark', newTheme);
  localStorage.setItem('theme', newTheme ? 'dark' : 'light');
  
  return newTheme;
};

export const getCurrentTheme = () => {
  return document.documentElement.classList.contains('theme-dark') ? 'dark' : 'light';
};