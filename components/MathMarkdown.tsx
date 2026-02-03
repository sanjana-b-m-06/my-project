
import React, { useMemo } from 'react';

interface MathMarkdownProps {
  content: string;
  className?: string;
}

const MathMarkdown: React.FC<MathMarkdownProps> = ({ content, className = "" }) => {
  const renderedContent = useMemo(() => {
    if (!content) return null;
    if (!(window as any).katex) return content;

    const katex = (window as any).katex;

    // Helper to render math strings safely
    const renderMath = (text: string, isBlock: boolean) => {
      try {
        return katex.renderToString(text, {
          displayMode: isBlock,
          throwOnError: false,
          trust: true
        });
      } catch (e) {
        console.error("Math render error:", e);
        return text;
      }
    };

    // Regex to find math delimiters: $$...$$ or $...$
    // We handle blocks first to avoid overlapping
    let processed = content;

    // 1. Process block math $$ ... $$
    processed = processed.replace(/\$\$\s*([\s\S]*?)\s*\$\$/g, (_, math) => {
      return `<div class="katex-display-container">${renderMath(math, true)}</div>`;
    });

    // 2. Process inline math $ ... $
    // We use a negative lookbehind/lookahead for common currency symbols if needed, 
    // but standard $...$ is usually fine in math contexts.
    processed = processed.replace(/\$([^\$\n]+?)\$/g, (_, math) => {
      return renderMath(math, false);
    });

    // 3. Basic Markdown-like formatting for the rest
    return processed.split('\n').map((line, idx) => {
      let lineHtml = line;
      
      // Bold **text**
      lineHtml = lineHtml.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      
      // List items
      if (line.trim().startsWith('* ') || line.trim().startsWith('- ')) {
        return <li key={idx} className="ml-5 list-disc mb-1" dangerouslySetInnerHTML={{ __html: lineHtml.trim().substring(2) }} />;
      }
      
      if (/^\d+\. /.test(line.trim())) {
        const contentStart = line.indexOf(' ') + 1;
        return <li key={idx} className="ml-5 list-decimal mb-1" dangerouslySetInnerHTML={{ __html: lineHtml.trim().substring(contentStart) }} />;
      }

      if (!line.trim()) return <br key={idx} />;

      return (
        <p 
          key={idx} 
          className="mb-3 leading-relaxed" 
          dangerouslySetInnerHTML={{ __html: lineHtml }} 
        />
      );
    });
  }, [content]);

  return (
    <div className={`prose dark:prose-invert max-w-none text-inherit ${className}`}>
      {renderedContent}
    </div>
  );
};

export default MathMarkdown;
