import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MarkdownMessageProps {
  content: string;
  className?: string;
}

export const MarkdownMessage: React.FC<MarkdownMessageProps> = ({ content, className = '' }) => {
  return (
    <ReactMarkdown
      className={`markdown-content ${className}`}
      remarkPlugins={[remarkGfm]}
      components={{
        // Headings
        h1: ({ node, ...props }) => <h1 className="text-xl font-bold mt-4 mb-2" {...props} />,
        h2: ({ node, ...props }) => <h2 className="text-lg font-bold mt-3 mb-2" {...props} />,
        h3: ({ node, ...props }) => <h3 className="text-base font-bold mt-2 mb-1" {...props} />,
        
        // Paragraphs
        p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
        
        // Links
        a: ({ node, ...props }) => (
          <a
            className="text-blue-600 hover:text-blue-800 underline"
            target="_blank"
            rel="noopener noreferrer"
            {...props}
          />
        ),
        
        // Lists
        ul: ({ node, ...props }) => <ul className="list-disc list-inside mb-2 space-y-1" {...props} />,
        ol: ({ node, ...props }) => <ol className="list-decimal list-inside mb-2 space-y-1" {...props} />,
        li: ({ node, ...props }) => <li className="ml-2" {...props} />,
        
        // Code
        code: ({ node, inline, className, children, ...props }: any) => {
          if (inline) {
            return (
              <code
                className="bg-gray-100 text-red-600 px-1.5 py-0.5 rounded text-sm font-mono"
                {...props}
              >
                {children}
              </code>
            );
          }
          return (
            <code
              className="block bg-gray-100 text-gray-800 p-3 rounded-lg overflow-x-auto text-sm font-mono my-2"
              {...props}
            >
              {children}
            </code>
          );
        },
        
        // Pre (code blocks)
        pre: ({ node, ...props }) => (
          <pre className="bg-gray-100 p-3 rounded-lg overflow-x-auto my-2" {...props} />
        ),
        
        // Blockquotes
        blockquote: ({ node, ...props }) => (
          <blockquote className="border-l-4 border-gray-300 pl-4 italic my-2" {...props} />
        ),
        
        // Strong/Bold
        strong: ({ node, ...props }) => <strong className="font-bold" {...props} />,
        
        // Emphasis/Italic
        em: ({ node, ...props }) => <em className="italic" {...props} />,
        
        // Horizontal rule
        hr: ({ node, ...props }) => <hr className="my-4 border-gray-300" {...props} />,
        
        // Tables
        table: ({ node, ...props }) => (
          <div className="overflow-x-auto my-2">
            <table className="min-w-full border-collapse border border-gray-300" {...props} />
          </div>
        ),
        thead: ({ node, ...props }) => <thead className="bg-gray-100" {...props} />,
        tbody: ({ node, ...props }) => <tbody {...props} />,
        tr: ({ node, ...props }) => <tr className="border-b border-gray-300" {...props} />,
        th: ({ node, ...props }) => (
          <th className="border border-gray-300 px-3 py-2 text-left font-semibold" {...props} />
        ),
        td: ({ node, ...props }) => (
          <td className="border border-gray-300 px-3 py-2" {...props} />
        ),
        
        // Strikethrough (with remark-gfm)
        del: ({ node, ...props }) => <del className="line-through" {...props} />,
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

