
import { GoogleGenAI, GenerateContentResponse, Type } from "@google/genai";
import { MATH_MODEL, SYSTEM_INSTRUCTION } from "../constants";
import { Message, Attachment } from "../types";

export class GeminiService {
  private ai: GoogleGenAI;

  constructor() {
    this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
  }

  async generateMathResponse(
    history: Message[],
    currentMessage: string,
    attachments?: Attachment[]
  ): Promise<{ text: string; thinking?: string }> {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
    
    // Construct contents
    const contents = history.map(msg => ({
      role: msg.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: msg.content }]
    }));

    // Add current parts
    const currentParts: any[] = [{ text: currentMessage }];
    
    if (attachments) {
      attachments.forEach(att => {
        currentParts.push({
          inlineData: {
            mimeType: att.type,
            data: att.data.split(',')[1] || att.data // handle potential data: prefix
          }
        });
      });
    }

    contents.push({ role: 'user', parts: currentParts });

    try {
      const response = await ai.models.generateContent({
        model: MATH_MODEL,
        contents: contents as any,
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
          thinkingConfig: { thinkingBudget: 16384 },
          temperature: 0.1, // Lower temperature for precision
        }
      });

      // The SDK currently returns thinking in the response if thinkingConfig is set
      // Some versions might put it in candidates[0].content.parts
      const parts = response.candidates?.[0]?.content?.parts || [];
      const textPart = parts.find(p => p.text)?.text || '';
      const thinkingPart = parts.find(p => (p as any).thought)?.thought || '';

      return {
        text: textPart || response.text || "I couldn't generate a solution. Please try again.",
        thinking: thinkingPart
      };
    } catch (error) {
      console.error("Gemini Error:", error);
      throw error;
    }
  }

  async speakText(text: string): Promise<Uint8Array | null> {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-preview-tts',
        contents: [{ parts: [{ text: `Explain this math problem clearly: ${text}` }] }],
        config: {
          responseModalities: ['AUDIO' as any],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
          }
        }
      });

      const audioData = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (audioData) {
        return this.decodeBase64(audioData);
      }
      return null;
    } catch (e) {
      console.error("TTS Error", e);
      return null;
    }
  }

  private decodeBase64(base64: string): Uint8Array {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  }
}

export const geminiService = new GeminiService();
