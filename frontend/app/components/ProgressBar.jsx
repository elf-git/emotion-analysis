'use client';

export default function ProgressBar({ progress, stage, title = 'Analyzing video' }) {
  const safeProgress = Math.min(100, Math.max(0, progress || 0));

  return (
    <div className="cinema-card" style={{ padding: '16px 20px' }} role="status">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '12px', marginBottom: '10px' }}>
        <span className="section-title">{title}</span>
        <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>Estimated progress</span>
      </div>

      <div style={{ height: '6px', background: 'var(--surface-2)', borderRadius: '2px', overflow: 'hidden', marginBottom: '10px' }}>
        <div
          style={{
            height: '100%',
            width: `${Math.max(4, safeProgress)}%`,
            background: 'var(--text-primary)',
            borderRadius: '2px',
            transition: 'width 0.3s ease',
          }}
        />
      </div>

      <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{stage || 'Processing...'}</div>
    </div>
  );
}
