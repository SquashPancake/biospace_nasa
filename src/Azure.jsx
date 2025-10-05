import React, { useState, useRef, useEffect } from 'react';
import * as sdk from "microsoft-cognitiveservices-speech-sdk";

const BIOSPACE_LOGO = "/images/biospace-logo.png";

// Chat history storage functions
const CHAT_STORAGE_KEY = 'biospace_chat_history';
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5000";

const loadChatHistory = () => {
  try {
    const stored = localStorage.getItem(CHAT_STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (error) {
    console.error('Error loading chat history:', error);
    return [];
  }
};

const saveChatHistory = (history) => {
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(history));
  } catch (error) {
    console.error('Error saving chat history:', error);
  }
};

const generateChatId = () => `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

function App() {
  // State management
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState(loadChatHistory());
  const [currentChatId, setCurrentChatId] = useState(null);
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [showInfoModal, setShowInfoModal] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [showRobotChat, setShowRobotChat] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [audioBars, setAudioBars] = useState([]);
  const [robotMessages, setRobotMessages] = useState([]);
  const [isBotProcessing, setIsBotProcessing] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [streamingComplete, setStreamingComplete] = useState(false);
  const [pdfUrl, setPdfUrl] = useState(null);
  const [currentlySpeakingId, setCurrentlySpeakingId] = useState(null);
  
  const messagesEndRef = useRef(null);
  const speechSynthesis = window.speechSynthesis;
  const recognitionRef = useRef(null);
  const audioBarsIntervalRef = useRef(null);
  const streamingIntervalRef = useRef(null);
  const azureSynthesizerRef = useRef(null);
  const browserSpeechRef = useRef(null);
  const speechStartTimeRef = useRef(null);

  // Quick suggestions
  const quickSuggestions = [
    "Tell me about microbial experiments on ISS",
    "Show plant growth studies in microgravity", 
    "What are the latest human adaptation findings?",
    "Explain radiation effects on DNA in space"
  ];

  // NEW: Function to send query to backend
  const sendQueryToBackend = async (query, chatHistory = []) => {
    try {
      console.log('Sending to backend:', { query, chatHistory });
      
      const response = await fetch(`${BACKEND_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          history: chatHistory
        })
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      const data = await response.json();
      console.log('Backend response:', data);
      return data;
    } catch (error) {
      console.error('Error calling backend:', error);
      // Fallback response if backend fails
      return {
        response: "I'm having trouble connecting to the research database. Please try again later.",
        pdfs: []
      };
    }
  };

  // NEW: Function to get PDF file from pmc_docs folder
  const getPdfUrl = (fileName, page) => {
    // Assuming your PDF files are in public/pmc_docs folder
    const pdfPath = `/pmc_docs/${fileName}`;
    console.log('Loading PDF from:', pdfPath);
    return {
      url: pdfPath,
      page: page
    };
  };

  // Initialize speech recognition
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setIsListening(false);
        stopAudioBars();
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        stopAudioBars();
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
        stopAudioBars();
      };
    }

    return () => {
      stopAllSpeech();
      stopAudioBars();
      if (streamingIntervalRef.current) {
        clearInterval(streamingIntervalRef.current);
      }
    };
  }, []);

  // Start audio visualization bars
  const startAudioBars = () => {
    stopAudioBars();
    const bars = Array.from({ length: 5 }, () => Math.random() * 40 + 10);
    setAudioBars(bars);
    
    audioBarsIntervalRef.current = setInterval(() => {
      setAudioBars(prev => prev.map(() => Math.random() * 40 + 10));
    }, 100);
  };

  // Stop audio visualization bars
  const stopAudioBars = () => {
    if (audioBarsIntervalRef.current) {
      clearInterval(audioBarsIntervalRef.current);
      audioBarsIntervalRef.current = null;
    }
    setAudioBars([]);
  };

  // Scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage]);

  // Azure TTS Configuration
  const getAzureSynthesizer = () => {
    const speechConfig = sdk.SpeechConfig.fromSubscription(
      import.meta.env.VITE_AZURE_SPEECH_KEY || "your-azure-speech-key",
      import.meta.env.VITE_AZURE_REGION || "eastus"
    );
    
    // Best female neural voices - choose one:
    speechConfig.speechSynthesisVoiceName = "en-US-JennyNeural"; // 🏆 Top quality
    // Alternatives:
    // "en-US-AriaNeural" - Very natural and expressive
    // "en-US-AmberNeural" - Warm and friendly
    // "en-US-AnaNeural" - Young and energetic
    // "en-GB-SoniaNeural" - British, elegant
    // "en-AU-NatashaNeural" - Australian, clear
    
    const audioConfig = sdk.AudioConfig.fromDefaultSpeakerOutput();
    const synthesizer = new sdk.SpeechSynthesizer(speechConfig, audioConfig);
    
    return synthesizer;
  };

  // COMPREHENSIVE SPEECH STOP FUNCTION
  const stopAllSpeech = () => {
    console.log('Stopping all speech...');
    
    // Stop Azure TTS
    if (azureSynthesizerRef.current) {
      try {
        azureSynthesizerRef.current.close();
        azureSynthesizerRef.current = null;
        console.log('Azure synthesizer stopped');
      } catch (error) {
        console.error('Error stopping Azure synthesizer:', error);
      }
    }
    
    // Stop browser TTS
    if (speechSynthesis.speaking) {
      try {
        speechSynthesis.cancel();
        console.log('Browser speech synthesis stopped');
      } catch (error) {
        console.error('Error stopping browser speech synthesis:', error);
      }
    }
    
    // Clear any browser speech reference
    if (browserSpeechRef.current) {
      browserSpeechRef.current = null;
    }
    
    // Update state
    setIsSpeaking(false);
    setCurrentlySpeakingId(null);
    speechStartTimeRef.current = null;
  };

  // Enhanced Text-to-Speech with Azure
  const speakText = async (text, messageId = null) => {
    // If already speaking, stop speech completely
    if (isSpeaking && currentlySpeakingId === messageId) {
      stopAllSpeech();
      return;
    }

    // If speaking something else, stop it first
    if (isSpeaking) {
      stopAllSpeech();
      // Small delay to ensure clean stop
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    const cleanText = text.replace(/\*\*/g, '').replace(/\*/g, '').replace(/## /g, '').replace(/\n/g, '. ');

    try {
      setIsSpeaking(true);
      setCurrentlySpeakingId(messageId);
      speechStartTimeRef.current = Date.now();
      
      // Create new Azure synthesizer
      const synthesizer = getAzureSynthesizer();
      azureSynthesizerRef.current = synthesizer;

      synthesizer.speakTextAsync(
        cleanText,
        result => {
          console.log('Azure TTS completed');
          setIsSpeaking(false);
          setCurrentlySpeakingId(null);
          speechStartTimeRef.current = null;
          if (azureSynthesizerRef.current === synthesizer) {
            synthesizer.close();
            azureSynthesizerRef.current = null;
          }
        },
        error => {
          console.error('Azure TTS error:', error);
          setIsSpeaking(false);
          setCurrentlySpeakingId(null);
          speechStartTimeRef.current = null;
          if (azureSynthesizerRef.current === synthesizer) {
            synthesizer.close();
            azureSynthesizerRef.current = null;
          }
          // Fallback to browser TTS
          fallbackBrowserTTS(cleanText, messageId);
        }
      );

    } catch (error) {
      console.error('Azure TTS failed:', error);
      setIsSpeaking(false);
      setCurrentlySpeakingId(null);
      speechStartTimeRef.current = null;
      if (azureSynthesizerRef.current) {
        azureSynthesizerRef.current.close();
        azureSynthesizerRef.current = null;
      }
      // Fallback to browser TTS
      fallbackBrowserTTS(cleanText, messageId);
    }
  };

  // Fallback browser TTS
  const fallbackBrowserTTS = (text, messageId = null) => {
    // Stop any existing speech first
    if (speechSynthesis.speaking) {
      speechSynthesis.cancel();
    }

    const utterance = new SpeechSynthesisUtterance(text);
    browserSpeechRef.current = utterance;
    
    const voices = speechSynthesis.getVoices();
    const preferredVoices = voices.filter(voice => 
      voice.lang.includes('en') && 
      (voice.name.includes('Female') || voice.name.includes('Zira') || voice.name.includes('Samantha') || voice.name.includes('Karen'))
    );
    
    if (preferredVoices.length > 0) {
      utterance.voice = preferredVoices[0];
    }

    utterance.rate = 0.85;
    utterance.pitch = 1.5;
    utterance.volume = 0.9;

    utterance.onstart = () => {
      setIsSpeaking(true);
      setCurrentlySpeakingId(messageId);
      speechStartTimeRef.current = Date.now();
      console.log('Browser TTS started');
    };
    
    utterance.onend = () => {
      setIsSpeaking(false);
      setCurrentlySpeakingId(null);
      speechStartTimeRef.current = null;
      browserSpeechRef.current = null;
      console.log('Browser TTS ended');
    };
    
    utterance.onerror = (event) => {
      console.error('Browser TTS error:', event);
      setIsSpeaking(false);
      setCurrentlySpeakingId(null);
      speechStartTimeRef.current = null;
      browserSpeechRef.current = null;
    };

    // Add a safety timeout to reset speaking state if something goes wrong
    setTimeout(() => {
      if (isSpeaking && speechStartTimeRef.current && (Date.now() - speechStartTimeRef.current > 30000)) {
        console.log('Safety timeout - resetting speech state');
        stopAllSpeech();
      }
    }, 31000);

    speechSynthesis.speak(utterance);
  };

  // Stop speaking - unified function
  const stopSpeaking = () => {
    stopAllSpeech();
  };

  // Stream text line by line
  const streamText = (text, callback) => {
    if (streamingIntervalRef.current) {
      clearInterval(streamingIntervalRef.current);
    }

    const lines = text.split('\n');
    let currentLine = 0;
    let currentText = '';

    setStreamingMessage('');
    setStreamingComplete(false);

    streamingIntervalRef.current = setInterval(() => {
      if (currentLine < lines.length) {
        currentText += (currentText ? '\n' : '') + lines[currentLine];
        setStreamingMessage(currentText);
        currentLine++;
        scrollToBottom();
      } else {
        clearInterval(streamingIntervalRef.current);
        setStreamingComplete(true);
        if (callback) callback(currentText);
      }
    }, 50);
  };

  // Handle citation click
  const handleCitationClick = (citation) => {
    setSelectedCitation(citation);
    
    // Get the actual PDF file from pmc_docs
    const pdfInfo = getPdfUrl(citation.fileName, citation.page);
    setPdfUrl(pdfInfo);
    
    setRightSidebarOpen(true);
  };

  // Close right sidebar
  const closeRightSidebar = () => {
    setRightSidebarOpen(false);
    setSelectedCitation(null);
    setPdfUrl(null);
  };

  // Microphone button speech recognition - ONLY transcribes to text
  const startVoiceRecognition = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition not supported in this browser. Try Chrome or Edge.");
      return;
    }

    try {
      setIsListening(true);
      startAudioBars();
      
      // Set up recognition for text input only
      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setIsListening(false);
        stopAudioBars();
      };

      recognitionRef.current.start();
    } catch (error) {
      console.error('Error starting speech recognition:', error);
      setIsListening(false);
      stopAudioBars();
    }
  };

  // Stop microphone recognition
  const stopVoiceRecognition = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
      stopAudioBars();
    }
  };

  // Robot chat voice recognition
  const startRobotChat = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition not supported in this browser. Try Chrome or Edge.");
      return;
    }

    try {
      setIsListening(true);
      startAudioBars();
      
      // Set up recognition for robot chat
      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        processRobotChatInput(transcript);
      };

      recognitionRef.current.start();
    } catch (error) {
      console.error('Error starting robot chat:', error);
      setIsListening(false);
      stopAudioBars();
    }
  };

  // Handle robot speech button
  const handleRobotSpeech = () => {
    if (isListening) {
      // Stop listening
      stopVoiceRecognition();
    } else if (isSpeaking) {
      // Stop speaking and start listening
      stopAllSpeech();
      // Small delay to ensure clean stop before starting recognition
      setTimeout(() => {
        startRobotChat();
      }, 100);
    } else {
      // Start listening
      startRobotChat();
    }
  };

  // UPDATED: Process robot chat input with backend integration
  const processRobotChatInput = async (transcript) => {
    setIsListening(false);
    stopAudioBars();

    // Add user message to robot chat
    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: transcript,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    setRobotMessages(prev => [...prev, userMessage]);
    setIsBotProcessing(true);

    try {
      // Prepare chat history for backend
      const chatHistoryForBackend = robotMessages.map(msg => ({
        role: msg.type === 'user' ? 'user' : 'assistant',
        content: msg.text
      }));

      // Send to backend
      const backendResponse = await sendQueryToBackend(transcript, chatHistoryForBackend);

      // Convert backend PDFs to citation format
      const citations = backendResponse.pdfs.map((pdf, index) => ({
        id: Date.now() + index,
        number: index + 1,
        title: `Source: ${pdf.name}`,
        authors: "Research Document",
        source: "NASA Database",
        fileName: pdf.name,
        page: pdf.page
      }));

      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        text: backendResponse.response,
        citations: citations,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      setRobotMessages(prev => [...prev, aiMessage]);
      setIsBotProcessing(false);
      speakText(backendResponse.response, aiMessage.id);

    } catch (error) {
      console.error('Error in robot chat:', error);
      setIsBotProcessing(false);
      
      const errorMessage = {
        id: Date.now() + 1,
        type: 'ai',
        text: "I encountered an error. Please try speaking again.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      setRobotMessages(prev => [...prev, errorMessage]);
      speakText("I encountered an error. Please try speaking again.", errorMessage.id);
    }
  };

  // Close robot chat and save to history - UPDATED TO STOP SPEECH
  const closeRobotChat = () => {
    // Stop all speech when closing robot chat
    stopAllSpeech();
    
    // Save the robot chat conversation to history
    if (robotMessages.length > 0) {
      const chatId = generateChatId();
      const title = robotMessages[0]?.text?.slice(0, 30) + (robotMessages[0]?.text?.length > 30 ? '...' : '') || 'Voice Chat';
      
      const newChat = {
        id: chatId,
        title: title,
        timestamp: "Just now",
        messages: robotMessages,
        type: 'voice'
      };

      const updatedHistory = [newChat, ...chatHistory.slice(0, 19)];
      setChatHistory(updatedHistory);
      saveChatHistory(updatedHistory);
      
      // Set as current chat
      setCurrentChatId(chatId);
      setMessages(robotMessages.map(msg => ({
        id: Date.now() + Math.random(),
        type: msg.type,
        text: msg.text,
        timestamp: msg.timestamp,
        citations: msg.citations || []
      })));
    }
    
    setShowRobotChat(false);
    setIsListening(false);
    setIsBotProcessing(false);
    setRobotMessages([]);
    stopAudioBars();
  };

  // Start new chat - UPDATED TO STOP SPEECH
  const startNewChat = () => {
    stopAllSpeech();
    setMessages([]);
    setCurrentChatId(null);
    setRobotMessages([]);
    setStreamingMessage('');
    setStreamingComplete(false);
    setRightSidebarOpen(false);
    setSelectedCitation(null);
    setPdfUrl(null);
  };

  // Select chat from history
  const selectChat = (chatId) => {
    const chat = chatHistory.find(c => c.id === chatId);
    if (chat) {
      setCurrentChatId(chatId);
      setMessages(chat.messages.map(msg => ({
        ...msg
      })));
    }
  };

  // Delete chat from history
  const deleteChat = (chatId, e) => {
    e.stopPropagation();
    const updatedHistory = chatHistory.filter(chat => chat.id !== chatId);
    setChatHistory(updatedHistory);
    saveChatHistory(updatedHistory);
    
    if (currentChatId === chatId) {
      startNewChat();
    }
  };

  // UPDATED: Handle sending messages with backend integration
  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: input,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setLoading(true);
    setStreamingMessage('');
    setStreamingComplete(false);

    try {
      // Prepare chat history for backend
      const chatHistoryForBackend = messages.map(msg => ({
        role: msg.type === 'user' ? 'user' : 'assistant',
        content: msg.text
      }));

      // Send to backend
      const backendResponse = await sendQueryToBackend(input, chatHistoryForBackend);

      // Convert backend PDFs to citation format
      const citations = backendResponse.pdfs.map((pdf, index) => ({
        id: Date.now() + index,
        number: index + 1,
        title: `Source: ${pdf.name}`,
        authors: "Research Document",
        source: "NASA Database",
        fileName: pdf.name,
        page: pdf.page
      }));

      const aiResponse = backendResponse.response;

      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        text: aiResponse,
        citations: citations,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      // Stream the response line by line
      streamText(aiResponse, (completeText) => {
        const finalAiMessage = {
          ...aiMessage,
          text: completeText
        };
        
        const updatedMessages = [...newMessages, finalAiMessage];
        setMessages(updatedMessages);
        setLoading(false);

        // Update chat history
        if (!currentChatId) {
          const newChatId = generateChatId();
          setCurrentChatId(newChatId);
          const newChat = {
            id: newChatId,
            title: input.slice(0, 30) + (input.length > 30 ? '...' : ''),
            timestamp: "Just now",
            messages: updatedMessages,
            type: 'text'
          };
          
          const updatedHistory = [newChat, ...chatHistory.slice(0, 19)];
          setChatHistory(updatedHistory);
          saveChatHistory(updatedHistory);
        } else {
          const updatedHistory = chatHistory.map(chat => 
            chat.id === currentChatId 
              ? { ...chat, messages: updatedMessages, timestamp: "Just now" }
              : chat
          );
          setChatHistory(updatedHistory);
          saveChatHistory(updatedHistory);
        }
      });

    } catch (error) {
      console.error('Error processing message:', error);
      setLoading(false);
      
      // Show error message to user
      const errorMessage = {
        id: Date.now() + 1,
        type: 'ai',
        text: "Sorry, I encountered an error while processing your request. Please try again.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Format text with basic markdown
  const formatText = (text) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/## (.*?)\n/g, '<h3 style="color: var(--accent-yellow); margin: 1rem 0 0.5rem 0; font-size: 16px; font-weight: 600;">$1</h3>')
      .replace(/\n/g, '<br/>');
  };

  // Render robot chat interface
  const renderRobotChat = () => (
    <div className="robot-chat-overlay">
      <div className="robot-chat-container">
        {/* Close Button */}
        <button 
          className="robot-close-button"
          onClick={closeRobotChat}
          title="Close Voice Chat"
        >
          ×
        </button>

        {/* Robot Icon */}
        <div className="robot-icon-large">
          <img src={BIOSPACE_LOGO} alt="BioSpace AI" />
          {isListening && (
            <div className="listening-animation">
              <div className="pulse-ring"></div>
              <div className="pulse-ring"></div>
              <div className="pulse-ring"></div>
            </div>
          )}
          {isBotProcessing && (
            <div className="processing-animation">
              <div className="processing-dots">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </div>
          )}
        </div>

        {/* Status Indicator */}
        <div className="robot-status">
          {isListening ? (
            <div className="status-listening">
              <div className="audio-bars">
                {audioBars.map((height, index) => (
                  <div 
                    key={index}
                    className="audio-bar"
                    style={{ height: `${height}%` }}
                  />
                ))}
              </div>
              Listening...
            </div>
          ) : isSpeaking ? (
            <div className="status-speaking">
              <div className="speaking-animation">
                <div className="speaking-bar"></div>
                <div className="speaking-bar"></div>
                <div className="speaking-bar"></div>
                <div className="speaking-bar"></div>
                <div className="speaking-bar"></div>
              </div>
              Speaking...
            </div>
          ) : isBotProcessing ? (
            <div className="status-processing">
              <div className="processing-dots">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
              Processing your question...
            </div>
          ) : (
            <div className="status-ready">
              Ready to help with space biology research
            </div>
          )}
        </div>

        {/* Control Buttons */}
        <div className="robot-controls">
          <button 
            className={`robot-speech-button ${isListening ? 'listening' : isSpeaking ? 'speaking' : ''}`}
            onClick={handleRobotSpeech}
            title={isListening ? 'Stop Listening' : isSpeaking ? 'Stop Speaking' : 'Start Speaking'}
          >
            {isListening ? '🛑 Stop' : isSpeaking ? '⏹️ Stop' : '🎤 Speak'}
          </button>
        </div>
      </div>
    </div>
  );

  // Render welcome screen
  const renderWelcomeScreen = () => (
    <div className="welcome-screen">
      <div className="welcome-icon">
        <img src={BIOSPACE_LOGO} alt="BioSpace" style={{ width: '80px', height: '80px', borderRadius: '16px' }} />
      </div>
      <h1 className="welcome-title">BioSpace Explorer</h1>
      <p className="welcome-subtitle">
        Advanced AI interface for NASA space biology research. 
        Access and analyze space biology publications with intelligent citations.
      </p>
      
      <div className="quick-suggestions">
        {quickSuggestions.map((suggestion, index) => (
          <button
            key={index}
            className="suggestion-chip"
            onClick={() => {
              setInput(suggestion);
              setTimeout(handleSend, 100);
            }}
            title={`Ask: ${suggestion}`}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );

  // Render citations
  const renderCitations = (citations) => (
    <div className="citations">
      <div className="citations-title">
        <span>📚</span>
        Sources
      </div>
      <div className="citation-chips">
        {citations.map((citation) => (
          <button
            key={citation.id}
            className="citation-chip"
            onClick={() => handleCitationClick(citation)}
            title={`Open ${citation.title}`}
          >
            <span className="citation-number">{citation.number}</span>
            <span>{citation.title}</span>
          </button>
        ))}
      </div>
    </div>
  );

  // Render message actions - UPDATED TO USE stopAllSpeech and track which message is speaking
  const renderMessageActions = (message) => {
    const isThisMessageSpeaking = isSpeaking && currentlySpeakingId === message.id;
    
    return (
      <div className="message-actions">
        <button
          className={`action-button ${isThisMessageSpeaking ? 'active' : ''}`}
          onClick={() => isThisMessageSpeaking ? stopAllSpeech() : speakText(message.text.replace(/\*\*/g, '').replace(/\*/g, ''), message.id)}
          title={isThisMessageSpeaking ? 'Stop Reading' : 'Read Aloud'}
        >
          {isThisMessageSpeaking ? (
            <>
              <div className="speaking-indicator">
                <div className="speaking-dot"></div>
                <div className="speaking-dot"></div>
                <div className="speaking-dot"></div>
              </div>
              Stop
            </>
          ) : (
            '🔊 Read'
          )}
        </button>
        <button
          className="action-button"
          onClick={() => navigator.clipboard.writeText(message.text)}
          title="Copy Text"
        >
          📋 Copy
        </button>
      </div>
    );
  };

  // UPDATED: Render right sidebar with actual PDF viewer
  const renderRightSidebar = () => (
    <aside className={`sidebar-right ${rightSidebarOpen ? '' : 'collapsed'}`}>
      <div className="sidebar-header">
        <h3 className="sidebar-title">Source Document</h3>
        <button 
          className="sidebar-close"
          onClick={closeRightSidebar}
          title="Close Source Viewer"
        >
          ×
        </button>
      </div>
      
      {selectedCitation && (
        <div className="source-viewer">
          <div className="source-info">
            <h4 className="source-title">{selectedCitation.title}</h4>
            <p className="source-authors">{selectedCitation.authors}</p>
            <p className="source-meta">
              <strong>Source:</strong> {selectedCitation.source}<br/>
              <strong>File:</strong> {selectedCitation.fileName}<br/>
              <strong>Page:</strong> {selectedCitation.page}
            </p>
          </div>
          
          <div className="pdf-viewer">
            {pdfUrl ? (
              <div className="pdf-container">
                <object 
                  data={`${pdfUrl.url}#page=${pdfUrl.page}`}
                  type="application/pdf"
                  width="100%"
                  height="100%"
                >
                  <div className="pdf-fallback">
                    <div className="pdf-icon">📄</div>
                    <p>Unable to display PDF</p>
                    <p className="pdf-info">
                      File: {selectedCitation.fileName}<br/>
                      Page: {selectedCitation.page}
                    </p>
                    <a 
                      href={pdfUrl.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="pdf-download-link"
                    >
                      📥 Download PDF to view
                    </a>
                  </div>
                </object>
              </div>
            ) : (
              <div className="pdf-placeholder">
                <div className="pdf-icon">📄</div>
                <p>Loading PDF...</p>
              </div>
            )}
          </div>
          
          <div className="source-actions">
            {pdfUrl && (
              <a 
                href={pdfUrl.url}
                target="_blank"
                rel="noopener noreferrer"
                className="source-action-btn"
                title="Open PDF in new tab"
              >
                👁️ Open Full PDF
              </a>
            )}
            <button 
              className="source-action-btn" 
              onClick={() => window.open(pdfUrl?.url, '_blank')}
              title="Download PDF"
            >
              📥 Download
            </button>
          </div>
        </div>
      )}
    </aside>
  );

  // Render streaming message
  const renderStreamingMessage = () => (
    <div className="message message-ai">
      <div className="message-header">
        <div className="avatar avatar-ai">
          <img src={BIOSPACE_LOGO} alt="BioSpace AI" style={{ width: '100%', height: '100%', borderRadius: '6px' }} />
        </div>
        <span>BioSpace AI</span>
        <span style={{ color: 'var(--text-muted)', fontSize: '12px', marginLeft: 'auto' }}>
          {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
      <div 
        className="message-content streaming-content"
        dangerouslySetInnerHTML={{ __html: formatText(streamingMessage) }}
      />
      {!streamingComplete && (
        <div className="streaming-cursor">|</div>
      )}
    </div>
  );

  // Render message
  const renderMessage = (message) => (
    <div key={message.id} className={`message ${message.type === 'user' ? 'message-user' : 'message-ai'}`}>
      <div className="message-header">
        <div className={`avatar ${message.type === 'user' ? 'avatar-user' : 'avatar-ai'}`}>
          {message.type === 'ai' ? (
            <img src={BIOSPACE_LOGO} alt="BioSpace AI" style={{ width: '100%', height: '100%', borderRadius: '6px' }} />
          ) : (
            '👤'
          )}
        </div>
        <span>{message.type === 'ai' ? 'BioSpace AI' : 'You'}</span>
        <span style={{ color: 'var(--text-muted)', fontSize: '12px', marginLeft: 'auto' }}>
          {message.timestamp}
        </span>
      </div>
      <div 
        className="message-content"
        dangerouslySetInnerHTML={{ __html: formatText(message.text) }}
      />
      {message.citations && renderCitations(message.citations)}
      {message.type === 'ai' && renderMessageActions(message)}
    </div>
  );

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo" onClick={startNewChat}>
            <div className="logo-icon">
              <img src={BIOSPACE_LOGO} alt="BioSpace" />
            </div>
            <div className="logo-text">BioSpace</div>
          </div>
          
          <div className="header-actions">
            <button 
              className="icon-button"
              onClick={() => setShowInfoModal(true)}
              title="About BioSpace"
            >
              ℹ️
            </button>
          </div>
        </div>
      </header>

      {/* Robot Chat Floating Button */}
      <button 
        className="robot-floating-button"
        onClick={() => setShowRobotChat(true)}
        title="Voice Assistant"
      >
        <img src={BIOSPACE_LOGO} alt="BioSpace AI" />
      </button>

      {/* Robot Chat Interface */}
      {showRobotChat && renderRobotChat()}

      <div className="app-container">
        {/* Sidebar Toggle Button */}
        {!leftSidebarOpen && (
          <button 
            className="sidebar-toggle"
            onClick={() => setLeftSidebarOpen(true)}
            title="Show Chat History"
          >
            💬
          </button>
        )}

        {/* Left Sidebar - Chat History */}
        <aside className={`sidebar-left ${leftSidebarOpen ? '' : 'collapsed'}`}>
          <div className="sidebar-header">
            <h3 className="sidebar-title">Chat History</h3>
            <button 
              className="sidebar-close"
              onClick={() => setLeftSidebarOpen(false)}
              title="Collapse Sidebar"
            >
              ×
            </button>
          </div>
          
          {leftSidebarOpen && (
            <>
              <button className="new-chat-btn" onClick={startNewChat} title="Start New Chat">
                <span>+</span>
                New Chat
              </button>
              
              <div className="history-list">
                {chatHistory.map(chat => (
                  <div
                    key={chat.id}
                    className={`history-item ${currentChatId === chat.id ? 'active' : ''}`}
                    onClick={() => selectChat(chat.id)}
                  >
                    <div className="history-item-content">
                      <div className="chat-preview">
                        <div className="chat-icon">
                          {chat.type === 'voice' ? '🎤' : '💬'}
                        </div>
                        <div className="chat-info">
                          <div className="chat-title">
                            {chat.title}
                          </div>
                          <div className="chat-meta">
                            <span className="chat-time">{chat.timestamp}</span>
                            <span className="chat-type">{chat.type}</span>
                          </div>
                        </div>
                      </div>
                      <button 
                        className="delete-chat-btn"
                        onClick={(e) => deleteChat(chat.id, e)}
                        title="Delete Chat"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                ))}
                
                {chatHistory.length === 0 && (
                  <div className="empty-history">
                    <div className="empty-icon">💬</div>
                    <p>No chats yet</p>
                    <span>Start a conversation to see history</span>
                  </div>
                )}
              </div>
            </>
          )}
        </aside>

        {/* Main Chat Area */}
        <main className="chat-main">
          <div className="messages-container">
            {messages.length === 0 ? (
              renderWelcomeScreen()
            ) : (
              messages.map(renderMessage)
            )}
            
            {loading && streamingMessage && renderStreamingMessage()}
            
            {loading && !streamingMessage && (
              <div className="message message-ai">
                <div className="message-header">
                  <div className="avatar avatar-ai">
                    <img src={BIOSPACE_LOGO} alt="BioSpace AI" style={{ width: '100%', height: '100%', borderRadius: '6px' }} />
                  </div>
                  <span>BioSpace AI</span>
                </div>
                <div className="message-content">
                  <div className="loading">
                    <div className="loading-dots">
                      <div className="loading-dot"></div>
                      <div className="loading-dot"></div>
                      <div className="loading-dot"></div>
                    </div>
                    Searching NASA databases...
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="input-area">
            <div className="input-container">
              {/* Audio visualization bars when listening */}
              {isListening && audioBars.length > 0 ? (
                <div className="audio-visualization">
                  {audioBars.map((height, index) => (
                    <div 
                      key={index}
                      className="audio-bar"
                      style={{ height: `${height}%` }}
                    />
                  ))}
                </div>
              ) : (
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Ask about NASA space biology research..."
                  className="text-input"
                  rows="1"
                />
              )}
              
              <div className="input-actions">
                <button
                  className={`voice-input-button ${isListening ? 'listening' : ''}`}
                  onClick={isListening ? stopVoiceRecognition : startVoiceRecognition}
                  title={isListening ? 'Stop Recording' : 'Voice Input'}
                >
                  🎤
                </button>
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || loading}
                  className="send-button"
                  title="Send Message"
                >
                  ↑
                </button>
              </div>
            </div>
          </div>
        </main>

        {/* Right Sidebar - Source Viewer */}
        {renderRightSidebar()}
      </div>

      {/* Info Modal */}
      {showInfoModal && (
        <div className="modal-overlay" onClick={() => setShowInfoModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title">About BioSpace</div>
              <button 
                className="close-button"
                onClick={() => setShowInfoModal(false)}
                title="Close"
              >
                ✕
              </button>
            </div>
            
            <div className="modal-body">
              <p>
                BioSpace is an AI-powered interface for exploring NASA's space biology 
                research database with intelligent citation tracking and voice features.
              </p>
              
              <h3>Backend Integration</h3>
              <ul>
                <li><strong>Backend URL:</strong> {BACKEND_URL}</li>
                <li><strong>PDF Source:</strong> pmc_docs folder</li>
                <li><strong>Chat History:</strong> Sent with each query</li>
                <li><strong>Real PDFs:</strong> Displayed with specific pages</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;