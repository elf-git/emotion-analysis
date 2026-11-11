'use client';

export default function MultilingualTranscript({
  clipResult,
  activeSegmentIndex,
  onSeekToSegment,
}) {
  if (!clipResult || !clipResult.segments?.length) {
    return (
      <div className="cinema-card" style={{ padding: '48px 24px', textAlign: 'center', color: 'var(--text-muted)' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: '600', color: 'var(--text-primary)' }}>No Transcript Available</h3>
        <p style={{ fontSize: '0.85rem', marginTop: '6px' }}>Analyze a video clip to generate timestamped transcripts.</p>
      </div>
    );
  }

  const { scene_metadata, segments } = clipResult;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header Panel */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 className="page-title">Transcript</h1>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '4px' }}>
              What was said and when. Click a time to jump to that moment.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <span className="metric-badge">
              {scene_metadata.detected_language}
            </span>
          </div>
        </div>
      </div>

      {/* Transcript Turns Table / Interactive Cards */}
      <div className="cinema-card" style={{ padding: '22px' }}>
        <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '700', marginBottom: '14px' }}>
          Clickable Timestamps (Seeks Video Player)
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {segments.map((seg, idx) => {
            const isActive = activeSegmentIndex === idx;
            const approachesMatch =
              seg.modality_predictions.text.approach_a_multilingual ===
              seg.modality_predictions.text.approach_b_translated;

            return (
              <div
                key={idx}
                style={{
                  background: isActive ? 'var(--surface-2)' : 'var(--bg-inset)',
                  border: isActive ? '1px solid var(--line-strong)' : '1px solid var(--line)',
                  borderRadius: 'var(--radius-md)',
                  padding: '14px 18px',
                  display: 'grid',
                  gridTemplateColumns: '120px 1fr 200px',
                  gap: '16px',
                  alignItems: 'center',
                  transition: 'all 0.15s ease',
                  cursor: 'pointer',
                }}
                onClick={() => onSeekToSegment(seg, idx)}
              >
                {/* Timestamp & Speaker Tag */}
                <div>
                  <button
                    className="cinema-btn-secondary"
                    style={{
                      fontSize: '0.74rem',
                      padding: '4px 8px',
                      background: 'var(--surface)',
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSeekToSegment(seg, idx);
                    }}
                  >
                    ▶ {seg.timestamp_label}
                  </button>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                    Turn #{idx + 1} • {seg.transcript.language}
                  </div>
                </div>

                {/* Script: Original + English Translation */}
                <div>
                  <div className="font-indic" style={{ fontSize: '0.96rem', color: 'var(--text-primary)', fontWeight: '500', marginBottom: '3px' }}>
                    {seg.transcript.original_text.startsWith('[') ? 'No speech in this segment.' : `“${seg.transcript.original_text}”`}
                  </div>
                  {seg.transcript.translated_text && seg.transcript.translated_text !== seg.transcript.original_text && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      EN: “{seg.transcript.translated_text}”
                    </div>
                  )}
                </div>

                {/* Dual-Approach Comparison Badges */}
                {seg.modality_predictions.text.predicted_emotion !== 'unknown' && <div style={{ textAlign: 'right' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'flex-end' }}>
                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Approach A:</span>
                      <span className="metric-badge" style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
                        {seg.modality_predictions.text.approach_a_multilingual}
                      </span>
                    </div>

                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Approach B:</span>
                      <span className="metric-badge" style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
                        {seg.modality_predictions.text.approach_b_translated}
                      </span>
                    </div>

                    <span
                      style={{
                        fontSize: '0.68rem',
                        color: approachesMatch ? 'var(--text-primary)' : 'var(--text-secondary)',
                        fontWeight: '600',
                      }}
                    >
                      {approachesMatch ? 'Approaches agree' : 'Approaches differ'}
                    </span>
                  </div>
                </div>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
