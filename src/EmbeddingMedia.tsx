import { useRef } from 'react';
import { Image, Mic, Paperclip, Video, X } from 'lucide-react';
import { api, errorText } from './api';
import { Spinner } from './components';
import type { Attachment } from './types';

export function MediaPreview({
  attachments,
  onRemove,
  disabled,
  compact = false,
}: {
  attachments: Attachment[];
  onRemove?: (id: string) => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  if (!attachments.length) return null;
  return (
    <div className={`embedding-media ${compact ? 'compact' : ''}`}>
      {attachments.map((a) => (
        <figure key={a.id}>
          {a.kind === 'image' ? (
            <img src={`/api/attachments/${a.id}`} alt={a.name} loading="lazy" />
          ) : a.kind === 'audio' ? (
            <div className="audio-preview">
              <Mic size={22} />
              <audio controls preload="metadata" src={`/api/attachments/${a.id}`} />
            </div>
          ) : (
            <video controls playsInline preload="metadata" src={`/api/attachments/${a.id}`} />
          )}
          <figcaption>
            {a.kind === 'image' ? (
              <Image size={12} />
            ) : a.kind === 'audio' ? (
              <Mic size={12} />
            ) : (
              <Video size={12} />
            )}
            <span title={a.name}>{a.name}</span>
            {onRemove && (
              <button
                className="icon-button"
                aria-label={`Remove ${a.name}`}
                disabled={disabled}
                onClick={() => onRemove(a.id)}
              >
                <X size={12} />
              </button>
            )}
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

export function EmbeddingAttach({
  label,
  modalities,
  attachments,
  onChange,
  disabled,
  onUploading,
  notify,
}: {
  label: string;
  modalities: string[];
  attachments: Attachment[];
  onChange: (files: Attachment[]) => void;
  disabled: boolean;
  onUploading: (busy: boolean) => void;
  notify: (s: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const uploading = useRef(false);
  const supported = modalities.filter((m) => m !== 'text');
  const accept = supported
    .flatMap((m) =>
      m === 'image'
        ? ['.png', '.jpg', '.jpeg', '.webp']
        : m === 'audio'
          ? ['.wav', '.flac', '.mp3']
          : ['.mp4', '.mov'],
    )
    .join(',');
  const upload = async (files: FileList | null) => {
    if (!files?.length || uploading.current || disabled) return;
    uploading.current = true;
    onUploading(true);
    try {
      if (attachments.length + files.length > 8)
        throw new Error('Each input can contain up to 8 media files.');
      const added: Attachment[] = [];
      for (const file of Array.from(files)) {
        if (file.size > 64 * 1024 * 1024)
          throw new Error('Each media file must be smaller than 64 MB.');
        const kind = /\.(png|jpe?g|webp)$/i.test(file.name)
          ? 'image'
          : /\.(wav|flac|mp3)$/i.test(file.name)
            ? 'audio'
            : /\.(mp4|mov)$/i.test(file.name)
              ? 'video'
              : '';
        if (!supported.includes(kind))
          throw new Error(`The selected export cannot process ${kind || 'this file type'}.`);
        const data = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result).split(',')[1]);
          reader.onerror = () => reject(new Error('Could not read media file.'));
          reader.readAsDataURL(file);
        });
        added.push(await api<Attachment>('/api/attachments', 'POST', { name: file.name, data }));
      }
      onChange([...attachments, ...added]);
    } catch (e) {
      notify(errorText(e));
    } finally {
      uploading.current = false;
      onUploading(false);
      if (input.current) input.current.value = '';
    }
  };
  return (
    <>
      <input
        className="hidden"
        ref={input}
        type="file"
        aria-label={label}
        accept={accept}
        multiple
        onChange={(e) => upload(e.target.files)}
        disabled={disabled || !supported.length}
      />
      <button
        className="text-button attach-embedding"
        disabled={disabled || !supported.length}
        onClick={() => input.current?.click()}
        title={
          supported.length
            ? 'PNG/JPEG/WebP, WAV/FLAC/MP3, MP4/MOV · up to 64 MB each'
            : 'Choose the multimodal EmbeddingGemma 2 export to attach media'
        }
      >
        {uploading.current ? <Spinner /> : <Paperclip size={13} />}Attach media
      </button>
    </>
  );
}
