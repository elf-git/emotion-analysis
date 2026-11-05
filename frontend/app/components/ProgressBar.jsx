'use client';

export default function ProgressBar({ progress, stage, title = 'Neural Analysis In Progress' }) {
  const safeProgress = Math.min(100, Math.max(0, progress || 0));

  return (
    <div
      style={{
        background: '#121214',
        border: '1px solid #27272a',
        borderRadius: 'var(--radius-md)',
        padding: '16px 20px',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: '600', color: '#f4f4f5' }}>{title}</span>
        </div>
        <span className="font-mono" style={{ fontSize: '0.88rem', fontWeight: '700', color: '#f4f4f5' }}>
          {safeProgress.toFixed(0)}%
        </span>
      </div>

      {/* Progress Track */}
      <div
        style={{
          height: '6px',
          background: '#18181b',
          borderRadius: 'var(--radius-full)',
          overflow: 'hidden',
          marginBottom: '10px',
          border: '1px solid #27272a',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${Math.max(4, safeProgress)}%`,
            background: '#fafafa',
            borderRadius: 'var(--radius-full)',
            transition: 'width 0.2s ease',
          }}
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.76rem', color: '#a1a1aa' }}>
        <span>{stage || 'Processing stream...'}</span>
        <span className="font-mono" style={{ color: '#71717a' }}>
          {safeProgress < 100 ? 'PyTorch 2.x • Faster-Whisper' : 'Ready'}
        </span>
      </div>
    </div>
  );
}
