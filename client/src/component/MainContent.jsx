import React from 'react';
import { useLocation } from 'react-router-dom';
import Router from '../Router';
import '../Styles/main.css';
import Breadcrumb from './BreadCrum';


const MainContent = () => {
  const location = useLocation();
  
  return (
    <div className="main-content">
      <div className="main-content__breadcrumb">
        <Breadcrumb />
      </div>
      <div className="main-content__router">
        <Router />
      </div>
    </div>
  );

};

export default MainContent;