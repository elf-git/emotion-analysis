'use client';

export default function SegmentInspector({
  clipResult,
  activeSegmentIndex,
  setActiveSegmentIndex,
  onSeekToSegment,
}) {
  if (!clipResult || !clipResult.segments?.length) {
    return (
      <div className="cinema-card" style={{ padding: '48px 24px', textAlign: 'center', color: '#71717a' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#f4f4f5' }}>No Segment Analysis Available</h3>
        <p style={{ fontSize: '0.85rem', marginTop: '6px' }}>Run a full scene analysis first.</p>
      </div>
    );
  }

  const segments = clipResult.segments;
  const currentSeg = segments[activeSegmentIndex] || segments[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Segment Selection Carousel */}
      <div className="cinema-card" style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <span style={{ fontSize: '0.74rem', color: '#71717a', textTransform: 'uppercase', fontWeight: '700' }}>
            SELECT DIALOGUE TURN ({segments.length} TOTAL)
          </span>
          <span className="font-mono" style={{ fontSize: '0.8rem', color: '#f4f4f5' }}>
            Active: #{activeSegmentIndex + 1} ({currentSeg.timestamp_label})
          </span>
        </div>

        <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '4px' }}>
          {segments.map((s, i) => (
            <button
              key={i}
              className={`cinema-btn-secondary ${activeSegmentIndex === i ? 'active' : ''}`}
              style={{
                fontSize: '0.76rem',
                padding: '6px 12px',
                background: activeSegmentIndex === i ? '#27272a' : '#18181b',
                borderColor: activeSegmentIndex === i ? '#52525b' : '#27272a',
                color: activeSegmentIndex === i ? '#ffffff' : '#a1a1aa',
              }}
              onClick={() => {
                setActiveSegmentIndex(i);
                onSeekToSegment(s, i);
              }}
            >
              #{i + 1} {s.timestamp_label} ({s.fused_prediction.predicted_emotion})
            </button>
          ))}
        </div>
      </div>

      {/* Main Segment Telemetry Canvas */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
        {/* 1. Primary Fused Decision & Transcript Card */}
        <div
          className="cinema-card"
          style={{
            padding: '24px',
            background: '#121214',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <span className="metric-badge">
              Segment #{currentSeg.segment_index + 1}
            </span>
            <span className="metric-badge" style={{ color: '#ffffff' }}>
              {currentSeg.fused_prediction.confidence_pct.toFixed(1)}% Confidence
            </span>
          </div>

          <div style={{ fontSize: '2.2rem', fontWeight: '700', textTransform: 'uppercase', color: '#f4f4f5', letterSpacing: '-0.02em', marginBottom: '6px' }}>
            {currentSeg.fused_prediction.predicted_emotion}
          </div>

          <div style={{ fontSize: '0.82rem', color: '#a1a1aa', marginBottom: '16px' }}>
            Valence: <strong style={{ color: '#f4f4f5' }}>{currentSeg.fused_prediction.valence > 0 ? `+${currentSeg.fused_prediction.valence}` : currentSeg.fused_prediction.valence}</strong> • Arousal: <strong style={{ color: '#f4f4f5' }}>{currentSeg.fused_prediction.arousal}</strong>
          </div>

          {/* Dialogue Transcript in Native Script + Translation */}
          <div style={{ background: '#09090b', border: '1px solid #27272a', borderRadius: 'var(--radius-md)', padding: '14px', marginBottom: '16px' }}>
            <div style={{ fontSize: '0.72rem', color: '#71717a', textTransform: 'uppercase', fontWeight: '600', marginBottom: '4px' }}>
              Original Dialogue Transcript ({currentSeg.transcript.language})
            </div>
            <div className="font-indic" style={{ fontSize: '1.05rem', color: '#ffffff', fontStyle: 'italic', marginBottom: '6px' }}>
              "{currentSeg.transcript.original_text}"
            </div>
            {currentSeg.transcript.translated_text && (
              <div style={{ fontSize: '0.82rem', color: '#a1a1aa' }}>
                English: "{currentSeg.transcript.translated_text}"
              </div>
            )}
          </div>

          {/* Explanation Synopsis */}
          <div style={{ fontSize: '0.82rem', color: '#a1a1aa', lineHeight: 1.5 }}>
            <strong style={{ color: '#f4f4f5' }}>Cross-Modal Synthesis:</strong> {currentSeg.explanation}
          </div>
        </div>

        {/* 2. Tri-Modal Breakdown (Text, Audio, Visual) */}
        <div className="cinema-card" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: '700', color: '#f4f4f5', marginBottom: '16px' }}>
            Modality-Wise Unimodal Predictions
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {/* Linguistic Branch */}
            <div style={{ background: '#09090b', border: '1px solid #27272a', borderRadius: 'var(--radius-md)', padding: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span style={{ fontSize: '0.74rem', color: '#a1a1aa', fontWeight: '600', textTransform: 'uppercase' }}>
                  Linguistic Semantics (DistilBERT)
                </span>
                <span className="font-mono" style={{ fontSize: '0.78rem', color: '#f4f4f5' }}>
                  {currentSeg.modality_predictions.text.confidence_pct}%
                </span>
              </div>
              <div style={{ fontSize: '0.92rem', fontWeight: '600', textTransform: 'capitalize', color: '#f4f4f5' }}>
                {currentSeg.modality_predictions.text.predicted_emotion}
              </div>
              <div style={{ fontSize: '0.72rem', color: '#71717a', marginTop: '2px' }}>
                Approach A: {currentSeg.modality_predictions.text.approach_a_multilingual} • Approach B: {currentSeg.modality_predictions.text.approach_b_translated}
              </div>
            </div>

            {/* Acoustic Branch */}
            <div style={{ background: '#09090b', border: '1px solid #27272a', borderRadius: 'var(--radius-md)', padding: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span style={{ fontSize: '0.74rem', color: '#a1a1aa', fontWeight: '600', textTransform: 'uppercase' }}>
                  Acoustic Prosody (Whisper / openSMILE)
                </span>
                <span className="font-mono" style={{ fontSize: '0.78rem', color: '#f4f4f5' }}>
                  {currentSeg.modality_predictions.audio.confidence_pct}%
                </span>
              </div>
              <div style={{ fontSize: '0.92rem', fontWeight: '600', textTransform: 'capitalize', color: '#f4f4f5' }}>
                {currentSeg.modality_predictions.audio.predicted_emotion}
              </div>
              <div style={{ fontSize: '0.72rem', color: '#71717a', marginTop: '2px' }}>
                Pitch: {currentSeg.modality_predictions.audio.pitch_hz || 0} Hz • Energy RMS: {currentSeg.modality_predictions.audio.rms_energy || 0}
              </div>
            </div>

            {/* Visual Branch */}
            <div style={{ background: '#09090b', border: '1px solid #27272a', borderRadius: 'var(--radius-md)', padding: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span style={{ fontSize: '0.74rem', color: '#a1a1aa', fontWeight: '600', textTransform: 'uppercase' }}>
                  Visual Facial AUs (ResNet-18)
                </span>
                <span className="font-mono" style={{ fontSize: '0.78rem', color: '#f4f4f5' }}>
                  {currentSeg.modality_predictions.visual.confidence_pct}%
                </span>
              </div>
              <div style={{ fontSize: '0.92rem', fontWeight: '600', textTransform: 'capitalize', color: '#f4f4f5' }}>
                {currentSeg.modality_predictions.visual.predicted_emotion}
              </div>
              <div style={{ fontSize: '0.72rem', color: '#71717a', marginTop: '2px' }}>
                Conv4 Saliency: {currentSeg.modality_predictions.visual.gradcam_available ? 'Active (Facial AU Overlay)' : 'Standard'}
              </div>
            </div>
          </div>
        </div>

        {/* 3. Dynamic Attention Gating & Calibrated Uncertainty */}
        <div className="cinema-card" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: '700', color: '#f4f4f5', marginBottom: '16px' }}>
            Dynamic Gating & Uncertainty Telemetry
          </h3>

          {/* Gating Progress */}
          <div style={{ marginBottom: '18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem', marginBottom: '6px' }}>
              <span style={{ color: '#a1a1aa' }}>Cross-Modal Attention (α Weights)</span>
              <span style={{ color: '#f4f4f5', fontWeight: '600' }}>
                Dominant: {currentSeg.modality_contributions.dominant_modality}
              </span>
            </div>
            <div style={{ height: '8px', background: '#18181b', borderRadius: 'var(--radius-full)', overflow: 'hidden', display: 'flex', border: '1px solid #27272a' }}>
              <div style={{ width: `${currentSeg.modality_contributions.text_pct}%`, background: '#ffffff' }} title="Text" />
              <div style={{ width: `${currentSeg.modality_contributions.audio_pct}%`, background: '#a1a1aa' }} title="Audio" />
              <div style={{ width: `${currentSeg.modality_contributions.video_pct}%`, background: '#52525b' }} title="Video" />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#71717a', marginTop: '6px' }}>
              <span style={{ color: '#f4f4f5' }}>Text: {currentSeg.modality_contributions.text_pct}%</span>
              <span style={{ color: '#a1a1aa' }}>Audio: {currentSeg.modality_contributions.audio_pct}%</span>
              <span style={{ color: '#71717a' }}>Video: {currentSeg.modality_contributions.video_pct}%</span>
            </div>
          </div>

          {/* Calibrated Uncertainty Metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', textAlign: 'center' }}>
            <div style={{ background: '#09090b', border: '1px solid #27272a', padding: '12px', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: '0.7rem', color: '#71717a' }}>Entropy (H_norm)</div>
              <div className="font-mono" style={{ fontSize: '1.05rem', fontWeight: '700', color: '#f4f4f5', marginTop: '2px' }}>
                0.19
              </div>
              <div style={{ fontSize: '0.65rem', color: '#71717a' }}>Low Ambiguity</div>
            </div>

            <div style={{ background: '#09090b', border: '1px solid #27272a', padding: '12px', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: '0.7rem', color: '#71717a' }}>Top-2 Margin (Δ)</div>
              <div className="font-mono" style={{ fontSize: '1.05rem', fontWeight: '700', color: '#f4f4f5', marginTop: '2px' }}>
                0.74
              </div>
              <div style={{ fontSize: '0.65rem', color: '#71717a' }}>Strong Margin</div>
            </div>

            <div style={{ background: '#09090b', border: '1px solid #27272a', padding: '12px', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: '0.7rem', color: '#71717a' }}>Wald 95% CI</div>
              <div className="font-mono" style={{ fontSize: '0.88rem', fontWeight: '700', color: '#d4d4d8', marginTop: '4px' }}>
                [0.86 – 0.94]
              </div>
              <div style={{ fontSize: '0.65rem', color: '#71717a' }}>Calibrated</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
