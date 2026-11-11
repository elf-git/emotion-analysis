'use client';

export default function ExplainabilitySuite({ clipResult, activeSegmentIndex }) {
  if (!clipResult || !clipResult.segments?.length) {
    return (
      <div className="cinema-card" style={{ padding: '48px 24px', textAlign: 'center', color: 'var(--text-muted)' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: '600', color: 'var(--text-primary)' }}>No Explainability Data Available</h3>
        <p style={{ fontSize: '0.85rem', marginTop: '6px' }}>Run a video analysis or load a sample scene.</p>
      </div>
    );
  }

  const currentSeg = clipResult.segments[activeSegmentIndex] || clipResult.segments[0];

  // Word importance from Integrated Gradients, when the segment has speech
  const tokens = currentSeg.text_details?.token_attributions || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header Panel */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 className="page-title">Explainability</h1>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '4px' }}>
              Why segment {currentSeg.segment_index + 1} ({currentSeg.timestamp_label}) got this result.
            </p>
          </div>

          <span className="metric-badge" style={{ color: 'var(--text-primary)' }}>
            {currentSeg.fused_prediction.predicted_emotion === 'unknown' ? 'No evidence' : `${currentSeg.fused_prediction.predicted_emotion} ${currentSeg.fused_prediction.confidence_pct.toFixed(1)}%`}
          </span>
        </div>
      </div>

      {/* Grid: Visual Grad-CAM + Linguistic Attribution + Acoustic Prosody */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))', gap: '20px' }}>
        {/* 1. Visual Grad-CAM Heatmap */}
        <div className="cinema-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <h3 style={{ fontSize: '0.92rem', fontWeight: '700', color: 'var(--text-primary)' }}>
              Face heatmap (Grad-CAM)
            </h3>
            <span className="metric-badge">
              ViT Patch Tokens
            </span>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'flex-start', marginBottom: '16px' }}>
            {currentSeg.modality_predictions.visual.gradcam_url ? (
              <div style={{ width: '224px', maxWidth: '100%', aspectRatio: '1', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--line)', background: '#000000', flexShrink: 0 }}>
                <img
                  src={currentSeg.modality_predictions.visual.gradcam_url}
                  alt="Grad-CAM Saliency Heatmap"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
              </div>
            ) : (
              <div style={{ width: '224px', maxWidth: '100%', aspectRatio: '1', borderRadius: 'var(--radius-md)', background: 'var(--surface-2)', border: '1px solid var(--line)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.6rem', color: 'var(--text-secondary)' }}>
                No face found
              </div>
            )}

            <div style={{ flex: '1 1 160px', fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              <div>Class: <strong style={{ color: 'var(--text-primary)', textTransform: 'capitalize' }}>{currentSeg.modality_predictions.visual.predicted_emotion}</strong></div>
              <div>Confidence: <strong style={{ color: 'var(--text-primary)' }}>{currentSeg.modality_predictions.visual.confidence_pct}%</strong></div>
              <div style={{ color: 'var(--text-muted)', marginTop: '6px' }}>
                Warmer colours mark image regions that pushed the model toward this emotion. It is an estimate, not proof of why.
              </div>
            </div>
          </div>

          <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', lineHeight: 1.5, background: 'var(--bg-inset)', border: '1px solid var(--line)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
            Calculated from the gradient of the predicted emotion score with respect to the patch tokens of the last transformer block.
          </div>
        </div>

        {/* 2. Word-Level Linguistic Attributions */}
        <div className="cinema-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <h3 style={{ fontSize: '0.92rem', fontWeight: '700', color: 'var(--text-primary)' }}>
              Word importance
            </h3>
            <span className="metric-badge">
              Integrated Gradients
            </span>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
              WORD IMPORTANCE PERCENTAGE:
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {tokens.length === 0 && (
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No word attribution for this segment.</span>
              )}
              {tokens.map((tok, i) => (
                <div
                  key={i}
                  className="font-indic"
                  style={{
                    background: 'var(--surface-2)',
                    border: '1px solid var(--line)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '6px 12px',
                    fontSize: '0.85rem',
                    color: 'var(--text-primary)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                  }}
                >
                  <span>{tok.token}</span>
                  <span className="font-mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    {tok.importance_pct}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', lineHeight: 1.5, background: 'var(--bg-inset)', border: '1px solid var(--line)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
            Integrated Gradients estimates how much each word contributed to the predicted emotion.
          </div>
        </div>

        {/* 3. Acoustic Prosody & Energy Dynamics */}
        <div className="cinema-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <h3 style={{ fontSize: '0.92rem', fontWeight: '700', color: 'var(--text-primary)' }}>
              Measured voice
            </h3>
            <span className="metric-badge">
              Audio Features
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', marginBottom: '16px' }}>
            <div style={{ background: 'var(--bg-inset)', border: '1px solid var(--line)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Fundamental Pitch (F_0)</div>
              <div className="font-mono" style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--text-primary)', marginTop: '2px' }}>
                {currentSeg.modality_predictions.audio.pitch_hz ?? '—'} Hz
              </div>
            </div>

            <div style={{ background: 'var(--bg-inset)', border: '1px solid var(--line)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>RMS Energy / Intensity</div>
              <div className="font-mono" style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--text-primary)', marginTop: '2px' }}>
                {currentSeg.modality_predictions.audio.rms_energy ?? '—'}
              </div>
            </div>
          </div>

          <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', lineHeight: 1.5, background: 'var(--bg-inset)', border: '1px solid var(--line)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
            Acoustic prosody features capture vocal strain, shouting, whisper nuances, and speaking tempo aligned with timestamps.
          </div>
        </div>
      </div>

      {/* Model Limitations & Ethics Advisory */}
      <div
        className="cinema-card"
        style={{
          padding: '18px 22px',
          background: 'var(--surface)',
          border: '1px solid var(--line)',
        }}
      >
        <h4 style={{ fontSize: '0.88rem', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '4px' }}>
          Limits of these explanations
        </h4>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          Linguistic emotion predictions are evaluated within conversational context. Sarcasm, regional idiomatic phrases, 
          and dramatic theatrical performances may produce discrepancies between spoken transcript sentiment and acoustic/facial dynamics. 
          Fusion uses fixed weights, and the support bars show how much each modality backed the final prediction.
        </p>
      </div>
    </div>
  );
}
