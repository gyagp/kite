import { Children, isValidElement, memo, type ReactNode } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { CopyButton } from './components';

function plainText(children: ReactNode): string {
  return Children.toArray(children)
    .map((child): string => {
      if (typeof child === 'string' || typeof child === 'number') return String(child);
      return isValidElement<{ children?: ReactNode }>(child) ? plainText(child.props.children) : '';
    })
    .join('');
}

const components: Components = {
  a: ({ children, href, title }) => (
    <a href={href} title={title} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
  img: ({ alt }) => <span className="muted">[Image: {alt || 'external image'}]</span>,
  pre: ({ children }) => {
    const code = Children.toArray(children).find((child) => isValidElement(child));
    const className = isValidElement<{ className?: string }>(code) ? code.props.className : '';
    const language = /language-([\w+-]+)/.exec(className || '')?.[1] || 'text';
    return (
      <div className="markdown-code-block">
        <div className="code-block-heading">
          <span>{language}</span>
          <CopyButton label="Copy code" text={plainText(children).replace(/\n$/, '')} />
        </div>
        <pre tabIndex={0} aria-label={`${language} code`}>
          {children}
        </pre>
      </div>
    );
  },
  table: ({ children }) => (
    <div className="markdown-table" role="region" aria-label="Markdown table" tabIndex={0}>
      <table>{children}</table>
    </div>
  ),
};

// Static components keep copy-button state stable while new streamed text arrives.
const Markdown = memo(function Markdown({ content }: { content: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: false, ignoreMissing: true }]]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});

export default Markdown;
