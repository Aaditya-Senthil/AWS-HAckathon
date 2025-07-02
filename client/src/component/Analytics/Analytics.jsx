import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot } from 'lucide-react';
import Lottie from 'lottie-react';
import animationData from '../../assets/animation-loader.json'; 
import { useUser } from '../../UserContext';

// Particle Component for backdrop
const ParticleBackdrop = () => {
  const canvasRef = useRef(null);
  let particles = [];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const particleCount = 100;

    const colorPalette = ['#c822ff', '#6f2dff', '#1a6dff', '#6dc7ff'];

    const hexToRgb = (hex) => {
      const bigint = parseInt(hex.replace('#', ''), 16);
      return {
        r: (bigint >> 16) & 255,
        g: (bigint >> 8) & 255,
        b: bigint & 255,
      };
    };

    const setCanvasSize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;

      particles = Array.from({ length: particleCount }, () => {
        const targetColor = hexToRgb(
          colorPalette[Math.floor(Math.random() * colorPalette.length)]
        );

        return {
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          dx: (Math.random() - 0.5) * 0.1,
          dy: (Math.random() - 0.5) * 0.1,
          size: Math.random() * 1 + 0.5,
          baseOpacity: Math.random() * 0.25 + 0.1,
          pulsePhase: Math.random() * Math.PI * 2,
          targetColor,
        };
      });
    };

    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);

    const animate = () => {
      const time = Date.now() * 0.002;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let p of particles) {
        p.x += p.dx;
        p.y += p.dy;

        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        const pulse = 0.9 + 0.1 * Math.sin(time + p.pulsePhase);
        const opacity = p.baseOpacity * pulse;

        // Interpolate from white to targetColor based on opacity (lower opacity = more color)
        const factor = 1 - pulse; // closer to 1 at lowest brightness

        const r = Math.round(255 * (1 - factor) + p.targetColor.r * factor);
        const g = Math.round(255 * (1 - factor) + p.targetColor.g * factor);
        const b = Math.round(255 * (1 - factor) + p.targetColor.b * factor);

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${opacity})`;
        ctx.fill();
      }

      requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', setCanvasSize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="particle-backdrop"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 0,
      }}
    />
  );
};






const WelcomeLoader = React.memo(({ isUserTyping }) => {
  return (
    <div className="welcome-state">
      <h1 className="welcome-title">Welcome to Agent-Analytics</h1>
      <p className="welcome-subtitle">Your RAG conversation partner</p>
      <div className="lottie-container">
        <Lottie 
          animationData={animationData} 
          loop 
          autoplay 
          style={{ width: 150, height: 150 }} 
        />
      </div>
      {isUserTyping && (
        <div className="welcome-typing-indicator">
          <div className="welcome-typing-text">Preparing your conversation...</div>
          <div className="welcome-typing-dots">
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
          </div>
        </div>
      )}
    </div>
  );
});


const InteractiveAvatar = () => (
  <Lottie 
    animationData={animationData} 
    loop 
    autoplay 
    style={{ width: 60, height: 60 }} 
  />
);

const ChatInterface = () => {
  const { userId } = useUser();
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isFirstConversation, setIsFirstConversation] = useState(true);
  const [showWelcome, setShowWelcome] = useState(true);
  const [isUserTyping, setIsUserTyping] = useState(false);
  const [showTransition, setShowTransition] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    if (isFirstConversation && showWelcome) {
      setShowTransition(true);
      setIsUserTyping(true);
      setTimeout(() => {
        setShowWelcome(false);
        setIsFirstConversation(false);
        setIsUserTyping(false);
        setShowTransition(false);
      }, 2000);
    }

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
    };

    setMessages(prev => [...prev, userMessage]);
    const currentQuery = inputValue.trim();
    setInputValue('');
    setIsTyping(true);

    const API_URL = ' https://66k86csbva.execute-api.ap-south-1.amazonaws.com/Prod/query'; 

    const Payload = {
      user_id: userId,
      query: currentQuery,
    };

    try {
      const response = await fetch(API_URL, { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(Payload),
      });

      const data = await response.json();

      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: data.response || 'Sorry, I didn’t understand that.', 
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('API Error:', error);
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          type: 'assistant',
          content: 'Oops! Something went wrong. Please try again.',
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
    
    if (showWelcome && isFirstConversation && e.target.value.trim()) {
      setIsUserTyping(true);
    } else if (showWelcome && isFirstConversation && !e.target.value.trim()) {
      setIsUserTyping(false);
    }
    
    // Auto-resize textarea
    const textarea = textareaRef.current;
    textarea.style.height = '50px';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
  };

  const handleNewChat = () => {
    setMessages([]);
    setInputValue('');
    setIsTyping(false);
    setShowWelcome(true);
    setIsFirstConversation(true);
    setIsUserTyping(false);
    setShowTransition(false);
  };

  return (
    <div className="chat-container">
      <ParticleBackdrop />
      
      <header className="chat-header">
        <button className="new-chat-btn" onClick={handleNewChat}>
          Initiate New Chat
        </button>
      </header>

      <div className="messages-container">
        {showWelcome && isFirstConversation ? (
            <WelcomeLoader isUserTyping={isUserTyping} key="welcome-loader" />

        ) : (
          <>
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.type}`}>
                <div className="message-avatar">
                  {message.type === 'user' ? <User size={18} /> : <Bot size={18} />}
                </div>
                <div className="message-content">
                  {message.content}
                </div>
              </div>
            ))}
            
            {isTyping && (
              <div className="message assistant">
                <div className="message-avatar">
                  <Bot size={18} />
                </div>
                <InteractiveAvatar />
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className={`input-container ${showWelcome && isFirstConversation ? 'welcome-input' : ''}`}>
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            className="message-input"
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyPress}
            placeholder="Message Analytics AI..."
            rows={1}
          />
          <button
            onClick={handleSubmit}
            className="send-button"
            disabled={!inputValue.trim()}
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
