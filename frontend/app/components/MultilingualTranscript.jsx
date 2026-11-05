'use client';

export default function MultilingualTranscript({
  clipResult,
  activeSegmentIndex,
  onSeekToSegment,
}) {
  if (!clipResult || !clipResult.segments?.length) {
    return (
      <div className="cinema-card" style={{ padding: '48px 24px', textAlign: 'center', color: '#71717a' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#f4f4f5' }}>No Transcript Available</h3>
        <p style={{ fontSize: '0.85rem', marginTop: '6px' }}>Analyze a video clip to generate timestamped transcripts.</p>
      </div>
    );
  }

  const { scene_metadata, segments } = clipResult;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header Panel */}
      <div className="cinema-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h2 style={{ fontSize: '1.2rem', fontWeight: '700', color: '#f4f4f5' }}>
                Multilingual Transcript & Dual-Model Comparison
              </h2>
              <span className="metric-badge">
                IndicTrans2 Aligned
              </span>
            </div>
            <p style={{ color: '#71717a', fontSize: '0.85rem', marginTop: '4px' }}>
              Timestamped original dialogue scripts, English translation alignments, and side-by-side Approach A vs Approach B inference.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <span className="metric-badge">
              Lang: {scene_metadata.detected_language} ({scene_metadata.language_code?.toUpperCase()})
            </span>
          </div>
        </div>
      </div>

      {/* Transcript Turns Table / Interactive Cards */}
      <div className="cinema-card" style={{ padding: '22px' }}>
        <div style={{ fontSize: '0.74rem', color: '#71717a', textTransform: 'uppercase', fontWeight: '700', marginBottom: '14px' }}>
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
                  background: isActive ? '#18181b' : '#09090b',
                  border: isActive ? '1px solid #3f3f46' : '1px solid #27272a',
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
                      background: '#121214',
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSeekToSegment(seg, idx);
                    }}
                  >
                    ▶ {seg.timestamp_label}
                  </button>
                  <div style={{ fontSize: '0.72rem', color: '#71717a', marginTop: '4px' }}>
                    Turn #{idx + 1} • {seg.transcript.language}
                  </div>
                </div>

                {/* Script: Original + English Translation */}
                <div>
                  <div className="font-indic" style={{ fontSize: '0.96rem', color: '#f4f4f5', fontWeight: '500', marginBottom: '3px' }}>
                    "{seg.transcript.original_text}"
                  </div>
                  {seg.transcript.translated_text && (
                    <div style={{ fontSize: '0.8rem', color: '#a1a1aa' }}>
                      EN: "{seg.transcript.translated_text}"
                    </div>
                  )}
                </div>

                {/* Dual-Approach Comparison Badges */}
                <div style={{ textAlign: 'right' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'flex-end' }}>
                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.7rem', color: '#71717a' }}>Approach A:</span>
                      <span className="metric-badge" style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
                        {seg.modality_predictions.text.approach_a_multilingual}
                      </span>
                    </div>

                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.7rem', color: '#71717a' }}>Approach B:</span>
                      <span className="metric-badge" style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
                        {seg.modality_predictions.text.approach_b_translated}
                      </span>
                    </div>

                    <span
                      style={{
                        fontSize: '0.68rem',
                        color: approachesMatch ? '#f4f4f5' : '#a1a1aa',
                        fontWeight: '600',
                      }}
                    >
                      {approachesMatch ? '✓ Full Concordance' : 'Nuance Shift'}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
