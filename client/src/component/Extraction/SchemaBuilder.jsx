import React, { useState } from 'react';
import '../../Styles/SchemaBuilder.css';

const SchemaBuilder = ({ fields, onAddField, onRemoveField, onUpdateField }) => {
  const [newField, setNewField] = useState({ name: '', type: 'string', required: true });
  const [fieldDescription, setFieldDescription] = useState('');

  const handleAddField = () => {
    if (newField.name) {
      onAddField({ ...newField, description: fieldDescription });
      setNewField({ name: '', type: 'string', required: true });
      setFieldDescription('');
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setNewField({
      ...newField,
      [name]: type === 'checkbox' ? checked : value
    });
  };

  return (
    <div className="schema-builder">
      <div className="fields-list">
        {fields.map((field, index) => (
          <div className="field-item" key={index}>
            <div className="field-row">
              <div className="handle"></div>
              <input 
                type="text" 
                value={field.name} 
                onChange={(e) => onUpdateField(index, { ...field, name: e.target.value })}
                placeholder="Field name" 
              />
              <select 
                value={field.type}
                onChange={(e) => onUpdateField(index, { ...field, type: e.target.value })}
              >
                <option value="string">string</option>
                <option value="number">number</option>
                <option value="boolean">boolean</option>
                <option value="object">object</option>
                <option value="array">array</option>
              </select>
              <input 
                type="checkbox" 
                checked={field.required} 
                onChange={(e) => onUpdateField(index, { ...field, required: e.target.checked })}
              />
              <button className="remove-button" onClick={() => onRemoveField(index)}>✕</button>
            </div>
            <input 
              type="text" 
              className="field-description" 
              value={field.description || ''} 
              onChange={(e) => onUpdateField(index, { ...field, description: e.target.value })}
              placeholder="Field description" 
            />
          </div>
        ))}
      </div>
      
      <div className="new-field">
        <div className="field-row">
          <div className="handle"></div>
          <input 
            type="text" 
            name="name"
            value={newField.name} 
            onChange={handleInputChange}
            placeholder="New field name" 
          />
          <select name="type" value={newField.type} onChange={handleInputChange}>
            <option value="string">string</option>
            <option value="number">number</option>
            <option value="boolean">boolean</option>
            <option value="object">object</option>
            <option value="array">array</option>
          </select>
          <input 
            type="checkbox" 
            name="required"
            checked={newField.required} 
            onChange={handleInputChange}
          />
          <button className="add-button" onClick={handleAddField}>+</button>
        </div>
        <input 
          type="text" 
          className="field-description" 
          value={fieldDescription} 
          onChange={(e) => setFieldDescription(e.target.value)}
          placeholder="Field description" 
        />
      </div>
    </div>
  );
};

export default SchemaBuilder;