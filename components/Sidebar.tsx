
import React from 'react';
import { ChatSession, Theme } from '../types';

interface SidebarProps {
  chats: ChatSession[];
  currentChatId: string | null;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat: (id: string) => void;
  isOpen: boolean;
  theme: Theme;
  setTheme: (t: Theme) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ 
  chats, 
  currentChatId, 
  onSelectChat, 
  onNewChat, 
  onDeleteChat, 
  isOpen,
  theme,
  setTheme
}) => {
  return (
    <aside className={`
      fixed inset-y-0 left-0 z-40 w-72 transform transition-transform duration-300 ease-in-out
      ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      md:relative md:translate-x-0 bg-[var(--sidebar-bg)] border-r border-gray-200 dark:border-slate-700
      flex flex-col
    `}>
      <div className="p-4 border-b border-gray-200 dark:border-slate-700">
        <button 
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2.5 px-4 rounded-xl transition-all shadow-lg"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" /></svg>
          New Calculation
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-4 space-y-1">
        <h3 className="px-3 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">History</h3>
        {chats.length === 0 ? (
          <p className="px-3 text-sm text-gray-400 italic">No history yet</p>
        ) : (
          chats.sort((a,b) => b.lastModified - a.lastModified).map(chat => (
            <div 
              key={chat.id}
              className={`group flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer transition-colors ${
                currentChatId === chat.id 
                ? 'bg-blue-100 dark:bg-slate-700 text-blue-700 dark:text-blue-300' 
                : 'hover:bg-gray-100 dark:hover:bg-slate-800 text-gray-700 dark:text-gray-300'
              }`}
              onClick={() => onSelectChat(chat.id)}
            >
              <svg className="w-4 h-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
              <span className="flex-1 text-sm font-medium truncate">{chat.title || "Untitled Math"}</span>
              <button 
                onClick={(e) => { e.stopPropagation(); onDeleteChat(chat.id); }}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 transition-opacity"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
              </button>
            </div>
          ))
        )}
      </div>

      <div className="p-4 border-t border-gray-200 dark:border-slate-700 space-y-3">
        <div className="flex items-center justify-between gap-1 p-1 bg-gray-100 dark:bg-slate-800 rounded-lg">
          {(Object.values(Theme) as Theme[]).map(t => (
            <button
              key={t}
              onClick={() => setTheme(t)}
              className={`flex-1 py-1 px-2 text-xs font-medium rounded capitalize transition-all ${
                theme === t ? 'bg-white dark:bg-slate-600 shadow-sm text-blue-600 dark:text-white' : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
              }`}
            >
              {t.split('-')[1]}
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
