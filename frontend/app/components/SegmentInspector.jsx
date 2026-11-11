'use client';

import { useRef } from 'react';
import { MOTION_OK, gsap, useGSAP } from '../lib/gsap';
import { EmptyState, ModalityRow, SupportBar, emotionColor } from './EmotionUI';

export default function SegmentInspector({
  clipResult,
  activeSegmentIndex,
  setActiveSegmentIndex,
  onSeekToSegment,
  onNewAnalysis,
}) {
  const root = useRef(null);

  useGSAP(
    () => {
      if (!clipResult?.segments?.length) return undefined;
      const mm = gsap.matchMedia();
      mm.add(MOTION_OK, () => {
        gsap.from('.inspect-card', { autoAlpha: 0, y: 18, duration: 0.5, stagger: 0.08, ease: 'power3.out' });
      });
      return () => mm.revert();
    },
    { scope: root, dependencies: [activeSegmentIndex, clipResult] }
  );

  if (!clipResult || !clipResult.segments?.length) {
    return (
      <EmptyState
        title="No segments to inspect"
        message="Analyze a video first, then open any segment here."
        actionLabel="Analyze a video"
        onAction={onNewAnalysis}
      />
    );
  }

  const segments = clipResult.segments;
  const currentSeg = segments[activeSegmentIndex] || segments[0];
  const emotion = currentSeg.fused_prediction.predicted_emotion;
  const hasResult = emotion !== 'unknown';
  const hasSpeech = !currentSeg.transcript.original_text.startsWith('[');
  const predictions = currentSeg.modality_predictions;
  const translated = currentSeg.transcript.translated_text;
  const valence = currentSeg.fused_prediction.valence;
  const uncertainty = currentSeg.uncertainty;

  return (
    <div ref={root} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <header>
        <h1 className="page-title">Segment inspector</h1>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
          See what each model said for one moment of the scene.
        </p>
      </header>

      <div className="segment-picker" role="group" aria-label="Choose a segment">
        {segments.map((seg, i) => {
          const segEmotion = seg.fused_prediction.predicted_emotion;
          return (
            <button
              key={i}
              className={`segment-pick ${activeSegmentIndex === i ? 'is-active' : ''}`}
              style={{ borderTopColor: emotionColor(segEmotion) }}
              aria-pressed={activeSegmentIndex === i}
              onClick={() => {
                setActiveSegmentIndex(i);
                onSeekToSegment(seg, i);
              }}
            >
              <span className="font-mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{seg.timestamp_label}</span>
              <strong>{segEmotion === 'unknown' ? 'No evidence' : segEmotion}</strong>
            </button>
          );
        })}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))', gap: '20px', alignItems: 'start' }}>
        <section className="cinema-card inspect-card" style={{ padding: '24px', borderTop: `3px solid ${emotionColor(emotion)}` }}>
          <span className="eyebrow">Fused result</span>
          <div className="font-display" style={{ fontSize: '2.8rem', lineHeight: 1.15, textTransform: 'capitalize', margin: '4px 0' }}>
            {hasResult ? emotion : 'No evidence'}
          </div>
          {hasResult && (
            <div className="font-mono" style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              {currentSeg.fused_prediction.confidence_pct.toFixed(1)}% · valence {valence > 0 ? `+${valence}` : valence} · arousal {currentSeg.fused_prediction.arousal}
            </div>
          )}

          <div style={{ padding: '14px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-inset)', border: '1px solid var(--line)', marginTop: '12px', marginBottom: '14px' }}>
            <div className="eyebrow" style={{ marginBottom: '4px' }}>Transcript · {currentSeg.transcript.language}</div>
            <p className="font-indic" style={{ fontSize: '1.02rem', color: hasSpeech ? 'var(--text-primary)' : 'var(--text-muted)' }}>
              {hasSpeech ? `“${currentSeg.transcript.original_text}”` : 'No speech in this segment.'}
            </p>
            {hasSpeech && translated && translated !== currentSeg.transcript.original_text && (
              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '6px' }}>English: “{translated}”</p>
            )}
          </div>

          <p style={{ fontSize: '0.86rem', lineHeight: 1.55, color: 'var(--text-secondary)' }}>{currentSeg.explanation}</p>
        </section>

        <section className="cinema-card inspect-card" style={{ padding: '24px' }}>
          <h2 className="section-title" style={{ marginBottom: '12px' }}>What each model said</h2>
          <div style={{ borderBottom: '1px solid var(--line)' }}>
            <ModalityRow
              name="Text"
              prediction={predictions.text}
              note={predictions.text?.predicted_emotion !== 'unknown' ? 'DistilRoBERTa' : 'No speech to read'}
            />
            <ModalityRow
              name="Voice"
              prediction={predictions.audio}
              note={`wav2vec 2.0 · pitch ${predictions.audio?.pitch_hz ?? '—'} Hz · energy ${predictions.audio?.rms_energy ?? '—'}`}
            />
            <ModalityRow
              name="Face"
              prediction={predictions.visual}
              note={`Vision Transformer · faces in ${predictions.visual?.faces_detected ?? '—'} of ${predictions.visual?.frames_sampled ?? '—'} frames`}
            />
          </div>

          {hasResult && (
            <div style={{ marginTop: '18px' }}>
              <SupportBar contributions={currentSeg.modality_contributions} />
            </div>
          )}
        </section>

        <section className="cinema-card inspect-card" style={{ padding: '24px' }}>
          <h2 className="section-title" style={{ marginBottom: '4px' }}>How sure is it?</h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '14px' }}>
            These scores describe this prediction. They are not calibrated accuracy.
          </p>
          <div className="metric-tiles">
            <div className="metric-tile">
              <span className="eyebrow">Entropy</span>
              <div className="metric-tile-value">{hasResult ? uncertainty?.entropy?.toFixed(2) ?? '—' : '—'}</div>
              <small>0 is sure, 1 is unsure</small>
            </div>
            <div className="metric-tile">
              <span className="eyebrow">Margin</span>
              <div className="metric-tile-value">{hasResult ? uncertainty?.top2_margin?.toFixed(2) ?? '—' : '—'}</div>
              <small>Gap to second place</small>
            </div>
            <div className="metric-tile">
              <span className="eyebrow">Level</span>
              <div className="metric-tile-value">{hasResult ? uncertainty?.uncertainty_level?.split(' ')[0] ?? '—' : '—'}</div>
              <small>Overall uncertainty</small>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
