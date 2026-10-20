import { useEffect, useRef, type ReactNode } from 'react';
import { Check, Copy, FolderOpen, LoaderCircle, X } from 'lucide-react';
import { useState } from 'react';
import type { Model } from './types';

export function Spinner() {
  return <LoaderCircle size={16} className="spin" />;
}
export function Badge({ children, tone = '' }: { children: ReactNode; tone?: string }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
export function ModelSelect({
  models,
  value,
  onChange,
  disabled,
}: {
  models: Model[];
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
}) {
  return (
    <select
      aria-label="Select model"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled || !models.length}
    >
      <option value="" disabled>
        Select a model
      </option>
      {models.map((m) => (
        <option key={m.id} value={m.id} disabled={!m.available}>
          {m.name}
          {models.filter((x) => x.name === m.name).length > 1 ? ` · ${m.variant}` : ''}
          {!m.available ? ' (not downloaded)' : ''}
        </option>
      ))}
    </select>
  );
}
export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [state, setState] = useState('');
  return (
    <button
      className="icon-button"
      aria-label={state || label}
      title={state || label}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setState('Copied');
        } catch {
          setState('Could not copy');
        }
        setTimeout(() => setState(''), 1800);
      }}
    >
      {state === 'Copied' ? <Check size={15} /> : <Copy size={15} />}
    </button>
  );
}
export function Modal({
  title,
  children,
  onClose,
  className = '',
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  className?: string;
}) {
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close.current();
    };
    document.addEventListener('keydown', handler);
    const before = document.activeElement as HTMLElement;
    const modal = document.querySelector<HTMLElement>('[role="dialog"]');
    const focusable = () =>
      modal?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex="0"]',
      );
    (modal?.querySelector<HTMLElement>('input, textarea') || focusable()?.[0])?.focus();
    const trap = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;
      const elements = focusable();
      if (!elements?.length) return;
      if (e.shiftKey && document.activeElement === elements[0]) {
        e.preventDefault();
        elements[elements.length - 1].focus();
      } else if (!e.shiftKey && document.activeElement === elements[elements.length - 1]) {
        e.preventDefault();
        elements[0].focus();
      }
    };
    document.addEventListener('keydown', trap);
    return () => {
      document.removeEventListener('keydown', handler);
      document.removeEventListener('keydown', trap);
      before?.focus();
    };
  }, []);
  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <section className={`modal ${className}`} role="dialog" aria-modal="true" aria-label={title}>
        <div className="modal-heading">
          <h2>{title}</h2>
          <button className="icon-button" aria-label="Close dialog" onClick={onClose}>
            <X size={19} />
          </button>
        </div>
        {children}
      </section>
    </div>
  );
}
export function PathInput({
  label,
  value,
  onChange,
  kind = 'folder',
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  kind?: 'folder' | 'runtime' | 'python';
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="path-input">
        <input value={value} onChange={(e) => onChange(e.target.value)} spellCheck={false} />
        {window.kiteDesktop && (
          <button
            type="button"
            className="icon-button"
            title={`Browse for ${label}`}
            aria-label={`Browse for ${label}`}
            onClick={async () => {
              const path = await window.kiteDesktop!.selectPath(kind);
              if (path) onChange(path);
            }}
          >
            <FolderOpen size={17} />
          </button>
        )}
      </div>
    </label>
  );
}
