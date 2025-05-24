'use client';

import { useState } from 'react';
import axios from 'axios';

export default function AskPage() {
  const [question, setQuestion] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const handleInputChange = (event) => {
    setQuestion(event.target.value);
    setMessage(''); // Clear previous messages
    setPrediction(null); // Clear previous prediction
  };

  const handleAsk = async () => {
    if (!question.trim()) {
      setMessage('Please enter a question.');
      return;
    }

    setLoading(true);
    setMessage('');
    setPrediction(null);

    try {
      // Ensure this URL matches your backend address
      const response = await axios.get('http://127.0.0.1:8000/ask', {
        params: { q: question },
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error asking question:', error);
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Ask the Model</h1>
      <p>Enter comma-separated feature values (e.g., 1.2,3.4,5.6)</p>
      <input
        type="text"
        value={question}
        onChange={handleInputChange}
        placeholder="Enter your query (e.g., 1.2,3.4,5.6)"
        style={{ minWidth: '300px', marginRight: '10px' }}
      />
      <button onClick={handleAsk} disabled={!question.trim() || loading}>
        {loading ? 'Getting Prediction...' : 'Ask'}
      </button>
      {message && <p>{message}</p>}
      {prediction !== null && (
        <h2>Prediction: {prediction}</h2>
      )}
    </div>
  );
}
