import { useEffect, useRef, useState } from 'react';
import {
  ArrowDownToLine,
  ArrowUp,
  Brain,
  Check,
  Code2,
  FileText,
  Image,
  MessageSquare,
  Mic,
  Paperclip,
  Plus,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import Markdown from './Markdown';
import { ChatStats } from './ChatStats';
import { api, download, errorText } from './api';
import { Badge, CopyButton, Modal, ModelSelect, Spinner } from './components';
import type { Attachment, Execute, Message, Model, Status, Thread } from './types';

const suggestions = [
  {
    icon: Sparkles,
    title: 'Make something clear',
    text: 'Explain how small language models run locally, in three simple sentences.',
  },
  {
    icon: Code2,
    title: 'Think through some code',
    text: 'Write a Python function that returns the sum of the even numbers in a list. Explain the implementation.',
  },
  {
    icon: FileText,
    title: 'Find the right words',
    text: 'Help me write a short, friendly invitation to a team lunch this Friday.',
  },
  {
    icon: Brain,
    title: 'Follow your curiosity',
    text: 'Why do the northern lights have different colors?',
  },
];

export function Chat({
  models,
  preferred,
  execute,
  busy,
  status,
  initial,
  onNew,
  onModels,
  notify,
  refresh,
  preferences,
}: {
  models: Model[];
  preferred: string;
  execute: Execute;
  busy: boolean;
  status: Status;
  initial: Thread | null;
  onNew: () => void;
  onModels: () => void;
  notify: (s: string) => void;
  refresh: () => Promise<void>;
  preferences?: { max_tokens: number };
}) {
  const [modelId, setModelId] = useState(
    initial?.model || preferred || models.find((m) => m.available)?.id || '',
  );
  const [messages, setMessages] = useState<Message[]>(
    initial?.messages.filter((m) => m.role !== 'system') || [],
  );
  const [threadId, setThreadId] = useState(initial?.id);
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(preferences?.max_tokens ?? 4096);
  const [savingPreferences, setSavingPreferences] = useState(false);
  useEffect(() => {
    if (preferences?.max_tokens != null) setMaxTokens(preferences.max_tokens);
  }, [preferences?.max_tokens]);
  const [topP, setTopP] = useState(0.9);
  const [system, setSystem] = useState(
    initial?.messages.find((m) => m.role === 'system')?.content ||
      'You are a helpful, concise assistant.',
  );
  const [thinking, setThinking] = useState(false);
  const [profile, setProfile] = useState(false);
  const [running, setRunning] = useState(false);
  const [showSettings, setShowSettings] = useState(true);
  const [rename, setRename] = useState(false);
  const [title, setTitle] = useState(initial?.title || 'New conversation');
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [uploading, setUploading] = useState(false);
  const file = useRef<HTMLInputElement>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const scroll = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  const model = models.find((m) => m.id === modelId);
  const hasMedia = model && model.modalities.length > 1;
  useEffect(() => {
    if (follow.current) bottom.current?.scrollIntoView({ behavior: 'instant', block: 'end' });
  }, [messages]);
  const send = async (history = messages, retry = false) => {
    if (busy || !model?.available || (!retry && !input.trim() && !attachments.length)) return;
    const next = retry
      ? history
      : [...history, { role: 'user' as const, content: input.trim(), attachments }];
    const draft = input,
      files = attachments;
    follow.current = true;
    setMessages([...next, { role: 'assistant', content: '', status: 'running' }]);
    setInput('');
    setAttachments([]);
    setRunning(true);
    let started = false;
    try {
      await execute(
        {
          kind: 'chat',
          model: model.id,
          thread_id: threadId,
          messages: [...(system.trim() ? [{ role: 'system', content: system }] : []), ...next],
          temperature,
          max_tokens: maxTokens,
          top_p: topP,
          thinking,
          profile,
        },
        (event) => {
          started = true;
          if (event.type === 'thread') {
            setThreadId(event.id);
            if (!threadId)
              setTitle(next.find((m) => m.role === 'user')?.content.slice(0, 65) || 'Conversation');
          }
          if (event.type === 'delta')
            setMessages((prev) =>
              prev.map((m, i) =>
                i === prev.length - 1 ? { ...m, content: m.content + (event.text || '') } : m,
              ),
            );
          if (event.type === 'metrics')
            setMessages((prev) =>
              prev.map((m, i) => (i === prev.length - 1 ? { ...m, metrics: event.data } : m)),
            );
          if (event.type === 'result')
            setMessages((prev) =>
              prev.map((m, i) =>
                i === prev.length - 1
                  ? {
                      ...m,
                      content: event.data?.text || m.content,
                      metrics: event.data,
                      status: 'completed',
                    }
                  : m,
              ),
            );
          if (event.type === 'error' || event.type === 'cancelled')
            setMessages((prev) =>
              prev.map((m, i) =>
                i === prev.length - 1
                  ? {
                      ...m,
                      status: event.type === 'error' ? 'failed' : 'cancelled',
                      error: event.message,
                    }
                  : m,
              ),
            );
        },
      );
    } catch (e) {
      setMessages((prev) =>
        prev.map((m, i) =>
          i === prev.length - 1 ? { ...m, status: 'failed', error: errorText(e) } : m,
        ),
      );
      if (!started) {
        setInput(draft);
        setAttachments(files);
        setMessages(history);
      }
    } finally {
      setRunning(false);
    }
  };
  const attach = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploading(true);
    try {
      const uploaded: Attachment[] = [];
      if (files.length + attachments.length > 8)
        throw new Error('A message can include at most 8 attachments.');
      for (const f of Array.from(files)) {
        if (f.size > 16 * 1024 * 1024)
          throw new Error('Each attachment must be smaller than 16 MB.');
        const kind = /\.(png|jpe?g)$/i.test(f.name)
          ? 'image'
          : /\.wav$/i.test(f.name)
            ? 'audio'
            : '';
        if (!kind || !model?.modalities.includes(kind))
          throw new Error(
            `This model package cannot process ${kind || 'this file type'}. Check its input modalities in My models.`,
          );
        const data = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result).split(',')[1]);
          reader.onerror = () => reject(new Error('Could not read the attachment.'));
          reader.readAsDataURL(f);
        });
        uploaded.push(await api<Attachment>('/api/attachments', 'POST', { name: f.name, data }));
      }
      setAttachments((prev) => [...prev, ...uploaded]);
    } catch (e) {
      notify(errorText(e));
    } finally {
      setUploading(false);
      if (file.current) file.current.value = '';
    }
  };
  const removeChat = async () => {
    try {
      await api(`/api/threads/${threadId}`, 'DELETE');
      await refresh();
      onNew();
    } catch (e) {
      notify(errorText(e));
    }
  };
  return (
    <div className="chat-page">
      <div className="chat-heading">
        <div>
          <div className="eyebrow">YOUR LOCAL ASSISTANT</div>
          <button
            className="chat-title"
            onClick={() => threadId && setRename(true)}
            title={threadId ? 'Rename conversation' : undefined}
          >
            <h1>{title}</h1>
          </button>
        </div>
        <div className="button-row">
          {messages.length > 0 && (
            <>
              <button
                className="icon-button"
                aria-label="Export conversation"
                title="Export conversation"
                onClick={() =>
                  download('kite-conversation.json', { title, model: modelId, messages })
                }
              >
                <ArrowDownToLine size={17} />
              </button>
              <button
                className="icon-button"
                aria-label="Delete conversation"
                title="Delete conversation"
                disabled={!threadId || busy}
                onClick={() => setConfirmDelete(true)}
              >
                <Trash2 size={16} />
              </button>
            </>
          )}
          <button className="button subtle" onClick={onNew} disabled={busy}>
            <Plus size={15} />
            New chat
          </button>
          <button
            className={`icon-button ${showSettings ? 'active' : ''}`}
            aria-label="Toggle chat settings"
            onClick={() => setShowSettings(!showSettings)}
          >
            <SlidersHorizontal size={18} />
          </button>
        </div>
      </div>
      <div className={`chat-layout ${showSettings ? '' : 'wide'}`}>
        <section className="conversation-panel">
          <div className="chat-model-bar">
            <span className="mini-model">Q</span>
            <ModelSelect
              models={models}
              value={modelId}
              onChange={(id) => {
                setModelId(id);
                setAttachments([]);
              }}
              disabled={busy || messages.length > 0}
            />
            <Badge>{model?.modalities.includes('image') ? 'VISION' : 'TEXT'}</Badge>
            <span className="muted small">
              {status.loaded?.model === modelId ? '● Loaded' : 'Loads on first message'}
            </span>
          </div>
          <div
            className="conversation"
            ref={scroll}
            onScroll={() => {
              if (scroll.current) {
                const s = scroll.current;
                follow.current = s.scrollHeight - s.scrollTop - s.clientHeight < 120;
              }
            }}
          >
            {!messages.length ? (
              <div className="chat-welcome">
                <div className="welcome-mark">
                  <img src="/kite.svg" alt="" />
                </div>
                <h2>
                  A little intelligence.
                  <br />
                  Entirely yours.
                </h2>
                <p>
                  Room to think, write, and explore.
                  <br />
                  Powered by a model on your own computer.
                </p>
                <div className="suggestions">
                  {suggestions.map((s) => (
                    <button key={s.title} disabled={busy} onClick={() => setInput(s.text)}>
                      <s.icon size={18} />
                      <strong>{s.title}</strong>
                      <span>{s.text}</span>
                      <ArrowUp size={14} />
                    </button>
                  ))}
                </div>
                {!models.length && (
                  <button className="button primary" onClick={onModels}>
                    Import a chat model
                  </button>
                )}
              </div>
            ) : (
              <div className="messages">
                {messages.map((m, i) => (
                  <article className={`message ${m.role}`} key={i}>
                    <div className="message-avatar">
                      {m.role === 'assistant' ? <img src="/kite.svg" alt="" /> : 'Y'}
                    </div>
                    <div className="message-body">
                      <div className="message-author">
                        {m.role === 'assistant' ? model?.name || 'Assistant' : 'You'}
                        {m.status === 'running' && <Badge>GENERATING</Badge>}
                      </div>
                      {m.attachments?.map((a) => (
                        <div key={a.id} className="message-attachment">
                          {a.kind === 'image' ? (
                            <img src={`/api/attachments/${a.id}`} alt={a.name} />
                          ) : (
                            <audio controls src={`/api/attachments/${a.id}`} />
                          )}
                          <span>{a.name}</span>
                        </div>
                      ))}
                      <Markdown content={m.content} />
                      {m.status === 'running' && !m.content && (
                        <div className="thinking-dots">
                          <span />
                          <span />
                          <span />
                        </div>
                      )}
                      {m.error && (
                        <p className={`message-error ${m.status === 'cancelled' ? 'muted' : ''}`}>
                          {m.error}
                        </p>
                      )}
                      {m.role === 'assistant' && m.content && (
                        <div className="message-meta">
                          <CopyButton text={m.content} />
                          {m.status === 'cancelled' && <Badge>STOPPED</Badge>}
                        </div>
                      )}
                      {m.role === 'assistant' && m.metrics && (
                        <ChatStats
                          metrics={{
                            ...m.metrics,
                            partial: m.status === 'cancelled' || m.status === 'failed',
                          }}
                          running={m.status === 'running'}
                        />
                      )}
                    </div>
                  </article>
                ))}
                {!busy && messages.length > 1 && messages.at(-1)?.role === 'assistant' && (
                  <button
                    className="text-button regenerate"
                    onClick={() => send(messages.slice(0, -1), true)}
                  >
                    <RefreshCw size={13} />
                    Regenerate response
                  </button>
                )}
              </div>
            )}
            <div ref={bottom} />
          </div>
          <div className="composer-wrap">
            <div
              className="composer"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (!busy) void attach(e.dataTransfer.files);
              }}
            >
              {attachments.length > 0 && (
                <div className="attachment-chips">
                  {attachments.map((a) => (
                    <span key={a.id}>
                      {a.kind === 'image' ? <Image size={13} /> : <Mic size={13} />} {a.name}
                      <button
                        className="icon-button"
                        aria-label={`Remove ${a.name}`}
                        onClick={() => setAttachments(attachments.filter((x) => x.id !== a.id))}
                      >
                        <X size={12} />
                      </button>
                    </span>
                  ))}
                </div>
              )}
              <textarea
                aria-label="Message"
                placeholder={
                  model ? 'Ask anything. Keep it local.' : 'Import a chat model to get started…'
                }
                value={input}
                disabled={busy || !model}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    void send();
                  }
                }}
                rows={3}
              />
              <div className="composer-bottom">
                <div>
                  <input
                    className="hidden"
                    ref={file}
                    type="file"
                    accept={[
                      ...(model?.modalities.includes('image') ? ['.png', '.jpg', '.jpeg'] : []),
                      ...(model?.modalities.includes('audio') ? ['.wav'] : []),
                    ].join(',')}
                    multiple
                    onChange={(e) => attach(e.target.files)}
                  />
                  <button
                    className="icon-button"
                    disabled={busy || uploading || !hasMedia}
                    title={
                      hasMedia
                        ? 'Attach image or WAV audio'
                        : 'This export supports text only. Import a multimodal package to attach media.'
                    }
                    aria-label="Attach media"
                    onClick={() => file.current?.click()}
                  >
                    {uploading ? <Spinner /> : <Paperclip size={17} />}
                  </button>
                  <span className="small muted">
                    {hasMedia ? 'Drop media here' : 'Text-only export'} · Shift + Enter for a new
                    line
                  </span>
                </div>
                <button
                  className="send-button"
                  aria-label="Send message"
                  disabled={
                    busy || uploading || !model?.available || (!input.trim() && !attachments.length)
                  }
                  onClick={() => send()}
                >
                  {running ? <Spinner /> : <ArrowUp size={20} />}
                </button>
              </div>
            </div>
            <p className="composer-disclaimer">
              Local models can make mistakes. Verify important information.
            </p>
          </div>
        </section>
        {showSettings && (
          <aside className="inspector chat-inspector">
            <div className="inspector-title">
              <SlidersHorizontal size={16} />
              Conversation settings
            </div>
            <section className="panel parameters">
              <label className="field">
                <span>System prompt</span>
                <textarea
                  value={system}
                  onChange={(e) => setSystem(e.target.value)}
                  disabled={busy}
                  rows={5}
                  placeholder="How should your assistant behave?"
                />
              </label>
              <label className="field">
                <span>
                  Temperature <b>{temperature.toFixed(1)}</b>
                </span>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step=".1"
                  value={temperature}
                  onChange={(e) => setTemperature(+e.target.value)}
                  disabled={busy}
                />
              </label>
              <p>Lower values are more focused. Zero uses greedy decoding.</p>
              <label className="field">
                <span>Maximum new tokens</span>
                <select
                  value={maxTokens}
                  onChange={async (e) => {
                    const previous = maxTokens;
                    const value = +e.target.value;
                    setMaxTokens(value);
                    setSavingPreferences(true);
                    try {
                      await api('/api/preferences/chat', 'POST', { max_tokens: value });
                      await refresh();
                    } catch (error) {
                      setMaxTokens(previous);
                      notify(errorText(error));
                    } finally {
                      setSavingPreferences(false);
                    }
                  }}
                  disabled={busy || savingPreferences}
                >
                  {[128, 256, 512, 1024, 2048, 4096].map((n) => (
                    <option key={n}>{n}</option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>
                  Top P <b>{topP.toFixed(2)}</b>
                </span>
                <input
                  type="range"
                  min=".05"
                  max="1"
                  step=".05"
                  value={topP}
                  onChange={(e) => setTopP(+e.target.value)}
                  disabled={busy}
                />
              </label>
              <label className="toggle-label">
                <span>
                  <Brain size={15} />
                  Thinking mode
                </span>
                <input
                  type="checkbox"
                  checked={thinking}
                  onChange={(e) => setThinking(e.target.checked)}
                  disabled={busy}
                />
              </label>
              <p>Used when the model’s chat template supports thinking.</p>
              <label className="studio-profile-option">
                <input
                  type="checkbox"
                  checked={profile}
                  onChange={(e) => setProfile(e.target.checked)}
                  disabled={busy}
                />
                <span>
                  <strong>Profile response</strong>
                  <small>Capture a fresh model load and operator trace. Adds overhead.</small>
                </span>
              </label>
            </section>
            <section className="tip-card">
              <MessageSquare size={18} />
              <h4>Give your assistant context</h4>
              <p>
                Your full conversation is included with each message. Start a new chat when you want
                a fresh perspective.
              </p>
            </section>
            <section className="capabilities">
              <h4>MODEL INPUTS</h4>
              {['text', 'image', 'audio'].map((c) => (
                <div key={c}>
                  <span>
                    {c === 'text' ? (
                      <FileText size={14} />
                    ) : c === 'image' ? (
                      <Image size={14} />
                    ) : (
                      <Mic size={14} />
                    )}{' '}
                    {c[0].toUpperCase() + c.slice(1)}
                  </span>
                  <span>
                    {model?.modalities.includes(c) ? (
                      <Check size={14} className="accent-text" />
                    ) : (
                      'Not in export'
                    )}
                  </span>
                </div>
              ))}
            </section>
          </aside>
        )}
      </div>
      {rename && (
        <Modal title="Rename conversation" onClose={() => setRename(false)}>
          <form
            onSubmit={async (e) => {
              e.preventDefault();
              try {
                await api(`/api/threads/${threadId}`, 'POST', { title });
                await refresh();
                setRename(false);
              } catch (err) {
                notify(errorText(err));
              }
            }}
          >
            <label className="field">
              <span>Title</span>
              <input
                value={title}
                maxLength={100}
                onChange={(e) => setTitle(e.target.value)}
                required
              />
            </label>
            <div className="modal-footer">
              <button type="submit" className="button primary">
                Save title
              </button>
            </div>
          </form>
        </Modal>
      )}
      {confirmDelete && (
        <Modal title="Delete this conversation?" onClose={() => setConfirmDelete(false)}>
          <p>The saved messages will be removed from this workspace.</p>
          <div className="modal-footer">
            <button className="button subtle" onClick={() => setConfirmDelete(false)}>
              Keep conversation
            </button>
            <button className="button danger" onClick={removeChat}>
              Delete conversation
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
