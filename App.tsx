
import React, { useState, useEffect, useRef } from 'react';
import { Theme, ChatSession, Message, Attachment } from './types';
import Sidebar from './components/Sidebar';
import MathMarkdown from './components/MathMarkdown';
import { geminiService } from './services/geminiService';

const App: React.FC = () => {
  const [chats, setChats] = useState<ChatSession[]>(() => {
    const saved = localStorage.getItem('sigma_math_chats');
    return saved ? JSON.parse(saved) : [];
  });
  const [currentChatId, setCurrentChatId] = useState<string | null>(() => {
    const saved = localStorage.getItem('sigma_math_chats');
    if (saved) {
      const parsed = JSON.parse(saved);
      return parsed.length > 0 ? parsed[0].id : null;
    }
    return null;
  });
  const [theme, setTheme] = useState<Theme>(() => {
    return (localStorage.getItem('sigma_theme') as Theme) || Theme.DARK;
  });
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth > 768);
  const [inputText, setInputText] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    localStorage.setItem('sigma_math_chats', JSON.stringify(chats));
    localStorage.setItem('sigma_theme', theme);
  }, [chats, theme]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chats, currentChatId, isTyping]);

  const currentChat = chats.find(c => c.id === currentChatId);

  const startNewChat = () => {
    const newChat: ChatSession = {
      id: Date.now().toString(),
      title: 'New Calculation',
      messages: [],
      createdAt: Date.now(),
      lastModified: Date.now()
    };
    setChats(prev => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  };

  const deleteChat = (id: string) => {
    setChats(prev => prev.filter(c => c.id !== id));
    if (currentChatId === id) {
      const remaining = chats.filter(c => c.id !== id);
      setCurrentChatId(remaining.length > 0 ? remaining[0].id : null);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newAttachments: Attachment[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.size > 10 * 1024 * 1024) {
        alert("File too large. Max 10MB per file.");
        continue;
      }
      const reader = new FileReader();
      const result = await new Promise<string>((resolve) => {
        reader.onload = (e) => resolve(e.target?.result as string);
        reader.readAsDataURL(file);
      });
      newAttachments.push({
        name: file.name,
        type: file.type,
        data: result
      });
    }
    setAttachments(prev => [...prev, ...newAttachments]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const sendMessage = async () => {
    if (!inputText.trim() && attachments.length === 0) return;
    if (isTyping) return;

    let activeChatId = currentChatId;
    if (!activeChatId) {
      const newChat: ChatSession = {
        id: Date.now().toString(),
        title: inputText.slice(0, 35) || 'Multimodal Query',
        messages: [],
        createdAt: Date.now(),
        lastModified: Date.now()
      };
      setChats(prev => [newChat, ...prev]);
      setCurrentChatId(newChat.id);
      activeChatId = newChat.id;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText,
      attachments: [...attachments],
      timestamp: Date.now()
    };

    setChats(prev => prev.map(chat => 
      chat.id === activeChatId 
      ? { ...chat, messages: [...chat.messages, userMessage], lastModified: Date.now() } 
      : chat
    ));

    const originalInput = inputText;
    const originalAttachments = [...attachments];
    setInputText('');
    setAttachments([]);
    setIsTyping(true);

    try {
      const chatHistory = chats.find(c => c.id === activeChatId)?.messages || [];
      const response = await geminiService.generateMathResponse(chatHistory, originalInput, originalAttachments);
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.text,
        thinking: response.thinking,
        timestamp: Date.now()
      };

      setChats(prev => prev.map(chat => 
        chat.id === activeChatId 
        ? { 
            ...chat, 
            messages: [...chat.messages, assistantMessage], 
            title: chat.messages.length === 0 ? originalInput.slice(0, 35) : chat.title,
            lastModified: Date.now() 
          } 
        : chat
      ));
    } catch (error) {
      console.error(error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I encountered an error processing your mathematical query. Please ensure your expression is valid and try again.",
        timestamp: Date.now()
      };
      setChats(prev => prev.map(chat => 
        chat.id === activeChatId ? { ...chat, messages: [...chat.messages, errorMessage] } : chat
      ));
    } finally {
      setIsTyping(false);
    }
  };

  const toggleRecording = () => {
    if (!('webkitSpeechRecognition' in window)) {
      alert("Speech recognition is not supported in this browser.");
      return;
    }

    if (!isRecording) {
      const recognition = new (window as any).webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInputText(prev => prev + (prev ? ' ' : '') + transcript);
        setIsRecording(false);
      };
      recognition.onerror = () => setIsRecording(false);
      recognition.onend = () => setIsRecording(false);
      recognition.start();
      setIsRecording(true);
    } else {
      setIsRecording(false);
    }
  };

  return (
    <div className={`flex h-screen w-full transition-colors duration-300 ${theme}`}>
      <Sidebar 
        chats={chats} 
        currentChatId={currentChatId} 
        onSelectChat={setCurrentChatId} 
        onNewChat={startNewChat}
        onDeleteChat={deleteChat}
        isOpen={isSidebarOpen}
        theme={theme}
        setTheme={setTheme}
      />

      <main className="flex-1 flex flex-col relative bg-[var(--bg-primary)] overflow-hidden">
        {/* Header */}
        <header className="h-16 flex items-center justify-between px-6 border-b border-gray-200 dark:border-slate-800 glass sticky top-0 z-30">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="p-2 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-xl transition-all active:scale-95"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" /></svg>
            </button>
            <div className="flex flex-col">
              <h1 className="text-xl font-black tracking-tighter bg-gradient-to-r from-blue-600 to-indigo-500 bg-clip-text text-transparent italic">SIGMA MATH</h1>
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></span>
                <span className="text-[10px] uppercase tracking-widest text-gray-400 font-bold">Systems Online</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
             <button onClick={startNewChat} className="hidden md:flex items-center gap-2 text-xs font-bold px-3 py-1.5 bg-blue-600/10 text-blue-600 rounded-lg hover:bg-blue-600 hover:text-white transition-all">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" /></svg>
                RESET
             </button>
          </div>
        </header>

        {/* Chat Area */}
        <div 
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-4 space-y-8 md:p-10 scroll-smooth"
        >
          {!currentChat || currentChat.messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-8 animate-fade-in px-4">
              <div className="relative">
                <div className="absolute inset-0 bg-blue-500 blur-3xl opacity-20 rounded-full animate-pulse"></div>
                <div className="relative w-24 h-24 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-[2rem] flex items-center justify-center shadow-2xl rotate-3">
                  <span className="text-4xl font-black text-white">Σ</span>
                </div>
              </div>
              <div className="max-w-xl">
                <h2 className="text-4xl font-black mb-4 tracking-tight leading-tight">Universal Math Engine.</h2>
                <p className="text-gray-500 dark:text-gray-400 text-lg leading-relaxed">
                  Sigma utilizes specialized reasoning models to solve integration, 
                  multivariable calculus, and theoretical physics problems.
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-3 w-full max-w-3xl">
                {[
                  "$$\\int_{0}^{\\infty} \\frac{\\sin x}{x} dx$$",
                  "Solve: $x^2 + 5x + 6 = 0$",
                  "Graph of $f(x) = \\sin(x)e^{-x}$"
                ].map(suggestion => (
                  <button 
                    key={suggestion}
                    onClick={() => setInputText(suggestion)}
                    className="p-4 bg-white dark:bg-slate-800 border border-gray-100 dark:border-slate-700 rounded-2xl text-sm font-semibold hover:border-blue-500 hover:shadow-xl transition-all text-left flex items-center gap-3"
                  >
                    <MathMarkdown content={suggestion} />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            currentChat.messages.map((msg) => (
              <div 
                key={msg.id} 
                className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-slide-up group`}
              >
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-lg bg-blue-600 flex-shrink-0 mr-3 mt-1 flex items-center justify-center text-white font-bold text-xs">
                    Σ
                  </div>
                )}
                <div className={`max-w-[85%] md:max-w-[80%] rounded-2xl px-6 py-5 shadow-sm relative transition-all ${
                  msg.role === 'user' 
                  ? 'bg-blue-600 text-white rounded-tr-none shadow-blue-500/20' 
                  : 'bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-tl-none'
                }`}>
                  {msg.attachments && msg.attachments.length > 0 && (
                    <div className="flex flex-wrap gap-3 mb-4">
                      {msg.attachments.map((att, idx) => (
                        <div key={idx} className="relative group/att cursor-pointer hover:scale-105 transition-transform">
                          {att.type.startsWith('image/') ? (
                            <img src={att.data} alt={att.name} className="max-h-60 w-auto rounded-xl border-2 border-white/20 shadow-lg" />
                          ) : (
                            <div className="p-3 bg-white/10 rounded-xl flex items-center gap-3 text-sm font-medium">
                              <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                              </div>
                              <span className="truncate max-w-[120px]">{att.name}</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {msg.thinking && (
                    <div className="mb-6">
                      <div className="flex items-center gap-2 mb-3 opacity-60">
                         <div className="w-4 h-px bg-current"></div>
                         <span className="text-[10px] font-black uppercase tracking-widest">Logic Stream</span>
                      </div>
                      <div className="text-sm italic opacity-80 border-l-2 border-blue-500/50 pl-5 py-1 whitespace-pre-wrap font-serif">
                        {msg.thinking}
                      </div>
                      <div className="mt-4 mb-2 h-px bg-gray-200 dark:bg-slate-700"></div>
                    </div>
                  )}

                  <MathMarkdown content={msg.content} />

                  <div className={`mt-4 flex items-center justify-between gap-4 text-[10px] uppercase font-black tracking-widest opacity-40 ${msg.role === 'user' ? 'text-white' : ''}`}>
                    <span>{new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    <div className="flex items-center gap-3 opacity-0 group-hover:opacity-100 transition-opacity">
                      {msg.role === 'assistant' && (
                        <button 
                          onClick={() => geminiService.speakText(msg.content).then(bytes => {
                             if (!bytes) return;
                             const ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
                             const int16 = new Int16Array(bytes.buffer);
                             const buf = ctx.createBuffer(1, int16.length, 24000);
                             buf.getChannelData(0).set(Array.from(int16).map(v => v / 32768));
                             const src = ctx.createBufferSource();
                             src.buffer = buf;
                             src.connect(ctx.destination);
                             src.start();
                          })}
                          className="hover:text-blue-500 flex items-center gap-1"
                        >
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" /></svg>
                          Voice
                        </button>
                      )}
                      <button className="hover:text-blue-500">Copy</button>
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
          {isTyping && (
            <div className="flex justify-start items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-blue-600/20 flex items-center justify-center">
                 <div className="w-1 h-1 bg-blue-600 rounded-full animate-ping"></div>
              </div>
              <div className="bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-2xl px-6 py-4 flex gap-1">
                {[0, 0.15, 0.3].map(d => (
                  <span key={d} style={{ animationDelay: `${d}s` }} className="w-1.5 h-1.5 bg-blue-600 rounded-full animate-bounce"></span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="p-4 md:p-6 glass border-t border-gray-200 dark:border-slate-800">
          <div className="max-w-4xl mx-auto">
            {attachments.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-4">
                {attachments.map((att, i) => (
                  <div key={i} className="flex items-center gap-2 bg-blue-500/10 dark:bg-blue-500/20 px-3 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800/50">
                    <span className="truncate max-w-[100px]">{att.name}</span>
                    <button onClick={() => setAttachments(prev => prev.filter((_, idx) => idx !== i))} className="hover:text-red-500">×</button>
                  </div>
                ))}
                <button onClick={() => setAttachments([])} className="text-[10px] font-black uppercase text-gray-400 hover:text-red-500 ml-2">Clear All</button>
              </div>
            )}
            
            <div className="flex items-end gap-2 bg-white dark:bg-slate-950 border border-gray-200 dark:border-slate-800 rounded-3xl p-2.5 shadow-2xl transition-all focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent">
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                multiple 
                className="hidden" 
              />
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="p-3.5 text-gray-400 hover:text-blue-600 transition-all active:scale-90"
                title="Add Scientific Context"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" /></svg>
              </button>

              <textarea 
                rows={1}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                placeholder="Enter mathematical problem..."
                className="flex-1 bg-transparent border-none focus:ring-0 text-base py-3 px-2 resize-none max-h-48 min-h-[44px] font-medium"
              />

              <button 
                onClick={toggleRecording}
                className={`p-3.5 transition-all rounded-full ${isRecording ? 'text-red-500 bg-red-500/10 animate-pulse' : 'text-gray-400 hover:text-blue-600'}`}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
              </button>

              <button 
                onClick={sendMessage}
                disabled={isTyping || (!inputText.trim() && attachments.length === 0)}
                className="p-3.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 dark:disabled:bg-slate-800 disabled:text-gray-500 text-white rounded-2xl shadow-xl shadow-blue-500/20 transition-all active:scale-95"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" /></svg>
              </button>
            </div>
            <div className="flex justify-center mt-3">
               <span className="text-[9px] font-black uppercase tracking-widest text-gray-400 opacity-60">Computational Reasoning Engine Powered by Gemini 3 Pro</span>
            </div>
          </div>
        </div>
      </main>

      <style>{`
        .animate-fade-in { animation: fadeIn 0.6s cubic-bezier(0.23, 1, 0.32, 1); }
        .animate-slide-up { animation: slideUp 0.4s cubic-bezier(0.23, 1, 0.32, 1); }
        @keyframes fadeIn { from { opacity: 0; transform: scale(0.98); } to { opacity: 1; transform: scale(1); } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
};

export default App;
