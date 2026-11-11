'use client';

export const EMOTION_COLORS = {
  neutral: 'var(--emo-neutral)',
  surprise: 'var(--emo-surprise)',
  fear: 'var(--emo-fear)',
  sadness: 'var(--emo-sadness)',
  joy: 'var(--emo-joy)',
  disgust: 'var(--emo-disgust)',
  anger: 'var(--emo-anger)',
};

export const emotionColor = (emotion) => EMOTION_COLORS[emotion] || 'var(--text-faint)';

export function EmotionDot({ emotion }) {
  return <span className="emotion-dot" style={{ background: emotionColor(emotion) }} aria-hidden="true" />;
}

export function SupportBar({ contributions }) {
  const parts = [
    { label: 'Text', value: contributions.text_pct, color: 'var(--mod-text)' },
    { label: 'Voice', value: contributions.audio_pct, color: 'var(--mod-voice)' },
    { label: 'Face', value: contributions.video_pct, color: 'var(--mod-face)' },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', fontSize: '0.78rem' }}>
        <span style={{ color: 'var(--text-muted)' }}>Support for this prediction</span>
        <span style={{ color: 'var(--text-secondary)' }}>Strongest: {contributions.dominant_modality}</span>
      </div>
      <div className="support-track">
        {parts.map((part) => (
          <div key={part.label} style={{ width: `${part.value}%`, background: part.color }} title={`${part.label}: ${part.value}%`} />
        ))}
      </div>
      <div className="support-legend">
        {parts.map((part) => (
          <span key={part.label}>
            <i className="support-swatch" style={{ background: part.color }} aria-hidden="true" />
            {part.label} <span className="font-mono">{part.value}%</span>
          </span>
        ))}
      </div>
    </div>
  );
}

export function ModalityRow({ name, prediction, note }) {
  const emotion = prediction?.predicted_emotion || 'unknown';
  const hasResult = emotion !== 'unknown';

  return (
    <div className="modality-row">
      <span className="modality-name">{name}</span>
      <span className="modality-emotion">
        <EmotionDot emotion={emotion} />
        {hasResult ? emotion : <span style={{ color: 'var(--text-muted)' }}>No evidence</span>}
      </span>
      <span className="font-mono" style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
        {hasResult ? `${prediction.confidence_pct}%` : '—'}
      </span>
      {note && <span className="modality-note">{note}</span>}
    </div>
  );
}

export function EmptyState({ title, message, actionLabel, onAction }) {
  return (
    <div className="empty-state">
      <h3 className="page-title" style={{ fontSize: '1.6rem' }}>{title}</h3>
      <p>{message}</p>
      {actionLabel && (
        <button className="cinema-btn-primary" onClick={onAction}>
          {actionLabel}
        </button>
      )}
    </div>
  );
}
