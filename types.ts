
export enum Theme {
  LIGHT = 'theme-light',
  DARK = 'theme-dark',
  MATH = 'theme-math'
}

export interface Attachment {
  name: string;
  type: string;
  data: string; // base64
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  attachments?: Attachment[];
  thinking?: string;
  timestamp: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  lastModified: number;
}

export interface MathState {
  chats: ChatSession[];
  currentChatId: string | null;
  theme: Theme;
  isSidebarOpen: boolean;
  isLiveMode: boolean;
}
