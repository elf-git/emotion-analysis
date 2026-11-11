'use client';

import { useRef } from 'react';
import { MOTION_OK, SplitText, createScrubber, gsap, useGSAP } from '../lib/gsap';
import { EmotionDot, EmptyState, ModalityRow, SupportBar, emotionColor } from './EmotionUI';

const formatTime = (seconds) => {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, '0')}`;
};

export default function SceneDashboard({
  clipResult,
  videoPlayerRef,
  videoPreviewUrl,
  currentPlaybackTime,
  activeSegmentIndex,
  onSeekToSegment,
  onSelectSegment,
  onTimeUpdate,
  onNewAnalysis,
}) {
  const root = useRef(null);
  const track = useRef(null);
  const playhead = useRef(null);

  // Scene entrance
  useGSAP(
    () => {
      if (!clipResult) return undefined;
      const mm = gsap.matchMedia();
      mm.add(MOTION_OK, () => {
        const word = SplitText.create('.dominant-word', { type: 'chars' });
        gsap
          .timeline({ defaults: { ease: 'power3.out' } })
          .from('.scene-header .metric-badge, .scene-header .eyebrow', { autoAlpha: 0, y: 10, duration: 0.45, stagger: 0.04 })
          .from(word.chars, { autoAlpha: 0, yPercent: 60, duration: 0.6, stagger: 0.03, ease: 'expo.out' }, '-=0.25')
          .from('.scene-stats > div', { autoAlpha: 0, y: 14, duration: 0.5, stagger: 0.06 }, '<0.1')
          .from('.cinema-player-box', { autoAlpha: 0, scale: 0.985, duration: 0.7 }, '<')
          .from('.timeline-block', {
            scaleX: 0,
            transformOrigin: 'left center',
            duration: 0.6,
            stagger: { amount: 0.6 },
            ease: 'power2.inOut',
            clearProps: 'transform',
          }, '<0.15')
          .from('.flow > span', { autoAlpha: 0, x: -10, duration: 0.4, stagger: { amount: 0.4 } }, '<0.3')
          .from('.dist-fill', {
            scaleX: 0,
            transformOrigin: 'left center',
            duration: 0.9,
            stagger: 0.05,
            ease: 'expo.out',
          }, '<');
      });
      return () => mm.revert();
    },
    { scope: root, dependencies: [clipResult] }
  );

  // Refresh the panel when the active segment changes
  useGSAP(
    () => {
      if (!clipResult) return undefined;
      const mm = gsap.matchMedia();
      mm.add(MOTION_OK, () => {
        gsap.from('.segment-focus > *', { autoAlpha: 0, y: 8, duration: 0.35, stagger: 0.03, ease: 'power2.out' });
      });
      return () => mm.revert();
    },
    { scope: root, dependencies: [activeSegmentIndex] }
  );

  // Dragging the playhead scrubs the real video
  useGSAP(
    () => {
      if (!playhead.current || !track.current) return undefined;
      const scrubber = createScrubber(playhead.current, track.current, {
        onPress: () => videoPlayerRef.current?.pause(),
        onScrub: (progress) => {
          const video = videoPlayerRef.current;
          if (video) video.currentTime = progress * (clipResult?.scene_metadata?.duration_sec || 1);
        },
      });
      return () => scrubber.kill();
    },
    { dependencies: [clipResult, videoPreviewUrl] }
  );

  if (!clipResult) {
    return (
      <EmptyState
        title="No scene analyzed yet"
        message="Analyze a video to see its emotion timeline here."
        actionLabel="Analyze a video"
        onAction={onNewAnalysis}
      />
    );
  }

  const { scene_metadata, scene_summary, segments } = clipResult;
  const activeSegment = segments[activeSegmentIndex] || segments[0];
  const duration = scene_metadata.duration_sec || 1;
  const dominant = scene_summary.dominant_emotion;
  const valence = scene_summary.average_valence;
  const playheadPct = Math.min(100, (currentPlaybackTime / duration) * 100);

  const distribution = Object.entries(scene_summary.emotion_distribution).sort(
    ([, a], [, b]) => b.percentage - a.percentage
  );
  const presentEmotions = [...new Set(segments.map((seg) => seg.fused_prediction.predicted_emotion))];
  const noEvidenceCount = segments.filter((seg) => seg.fused_prediction.predicted_emotion === 'unknown').length;
  const activeEmotion = activeSegment.fused_prediction.predicted_emotion;
  const predictions = activeSegment.modality_predictions;
  const hasSpeech = !activeSegment.transcript.original_text.startsWith('[');
  const translated = activeSegment.transcript.translated_text;

  return (
    <div ref={root} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <header className="scene-header">
        <div style={{ minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '10px' }}>
            <span className="metric-badge">{scene_metadata.detected_language || 'Multilingual'}</span>
            <span className={`metric-badge ${clipResult.is_live_inference ? '' : 'is-warning'}`}>
              {clipResult.is_live_inference ? 'Pretrained model output' : 'Scripted sample, not model output'}
            </span>
          </div>
          <div className="eyebrow" style={{ marginBottom: '6px', overflowWrap: 'anywhere' }}>
            {scene_metadata.filename || 'Scene'}
          </div>
          <div className="dominant-word" style={{ color: emotionColor(dominant) === 'var(--text-faint)' ? 'var(--text-primary)' : emotionColor(dominant) }}>
            {dominant.charAt(0).toUpperCase() + dominant.slice(1)}
          </div>
        </div>

        <div className="scene-stats">
          <div>
            <div className="eyebrow">Valence</div>
            <div className="scene-stat-value">{valence > 0 ? `+${valence}` : valence}</div>
          </div>
          <div>
            <div className="eyebrow">Arousal</div>
            <div className="scene-stat-value">{scene_summary.average_arousal}</div>
          </div>
          <div>
            <div className="eyebrow">Duration</div>
            <div className="scene-stat-value">{scene_metadata.duration_sec.toFixed(1)}s</div>
          </div>
          <div>
            <div className="eyebrow">Segments</div>
            <div className="scene-stat-value">{scene_summary.total_segments}</div>
          </div>
        </div>
      </header>

      <div className="dash-grid">
        <div className="dash-col">
          <div className="cinema-player-box">
            {videoPreviewUrl ? (
              <video ref={videoPlayerRef} src={videoPreviewUrl} controls playsInline onTimeUpdate={onTimeUpdate} />
            ) : (
              <iframe
                src={clipResult.embed_url || `https://www.youtube.com/embed/${clipResult.video_id || 'v77IOwXg3B4'}?enablejsapi=1&autoplay=0&rel=0`}
                title={scene_metadata.filename || 'Movie scene player'}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            )}
          </div>

          <section className="cinema-card" style={{ padding: '18px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '12px', flexWrap: 'wrap' }}>
              <h2 className="section-title">Emotion timeline</h2>
              <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                {videoPreviewUrl ? 'Click a segment or drag the playhead' : 'Click a segment to jump to it'}
              </span>
            </div>

            <div ref={track} className="timeline-track">
              {segments.map((seg, idx) => {
                const emotion = seg.fused_prediction.predicted_emotion;
                return (
                  <button
                    key={idx}
                    className={`timeline-block ${activeSegmentIndex === idx ? 'is-active' : ''} ${emotion === 'unknown' ? 'is-unknown' : ''}`}
                    style={{
                      flexGrow: Math.max(0.1, seg.end_time_sec - seg.start_time_sec),
                      flexBasis: 0,
                      ...(emotion === 'unknown' ? {} : { background: emotionColor(emotion) }),
                    }}
                    onClick={() => onSeekToSegment(seg, idx)}
                    title={`${seg.timestamp_label}: ${emotion}`}
                    aria-label={`Segment ${idx + 1}, ${seg.timestamp_label}, ${emotion}`}
                  />
                );
              })}
              {videoPreviewUrl && (
                <div
                  ref={playhead}
                  className="timeline-playhead"
                  style={{ left: `${playheadPct}%` }}
                  aria-hidden="true"
                />
              )}
            </div>

            <div className="timeline-scale">
              <span>0:00.0</span>
              <span>{formatTime(duration)}</span>
            </div>

            <div className="legend" style={{ marginTop: '12px' }}>
              {presentEmotions.map((emotion) => (
                <span key={emotion}>
                  <EmotionDot emotion={emotion} />
                  {emotion === 'unknown' ? 'No evidence' : emotion}
                </span>
              ))}
            </div>
          </section>

          <section>
            <div className="eyebrow" style={{ marginBottom: '10px' }}>Emotion flow</div>
            <div className="flow">
              {scene_summary.emotion_transitions?.map((emotion, idx) => (
                <span key={idx} style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                  <button
                    className={`flow-chip ${activeSegmentIndex === idx ? 'is-active' : ''}`}
                    onClick={() => onSeekToSegment(segments[idx] || segments[0], idx)}
                  >
                    <EmotionDot emotion={emotion} />
                    {emotion === 'unknown' ? 'No evidence' : emotion}
                  </button>
                  {idx < scene_summary.emotion_transitions.length - 1 && (
                    <span className="flow-arrow" aria-hidden="true">→</span>
                  )}
                </span>
              ))}
            </div>
          </section>
        </div>

        <div className="dash-col">
          <section className="cinema-card segment-focus" style={{ padding: '20px', borderTop: `3px solid ${emotionColor(activeEmotion)}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '12px', marginBottom: '4px' }}>
              <span className="eyebrow">Segment {activeSegment.segment_index + 1} of {segments.length}</span>
              <span className="font-mono" style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                {activeSegment.timestamp_label}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px', marginBottom: '10px' }}>
              <span className="font-display" style={{ fontSize: '2rem', lineHeight: 1.2, textTransform: 'capitalize' }}>
                {activeEmotion === 'unknown' ? 'No evidence' : activeEmotion}
              </span>
              {activeEmotion !== 'unknown' && (
                <span className="font-mono" style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  {activeSegment.fused_prediction.confidence_pct.toFixed(1)}%
                </span>
              )}
            </div>

            <p className="font-indic" style={{ fontSize: '0.98rem', lineHeight: 1.5, color: hasSpeech ? 'var(--text-primary)' : 'var(--text-muted)', marginBottom: '6px' }}>
              {hasSpeech ? `“${activeSegment.transcript.original_text}”` : 'No speech in this segment.'}
            </p>
            {hasSpeech && translated && translated !== activeSegment.transcript.original_text && (
              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                English: “{translated}”
              </p>
            )}

            <div style={{ marginTop: '14px', borderBottom: '1px solid var(--line)' }}>
              <ModalityRow name="Text" prediction={predictions.text} />
              <ModalityRow name="Voice" prediction={predictions.audio} />
              <ModalityRow name="Face" prediction={predictions.visual} />
            </div>

            {activeEmotion !== 'unknown' && (
              <div style={{ marginTop: '14px' }}>
                <SupportBar contributions={activeSegment.modality_contributions} />
              </div>
            )}

            <button
              className="cinema-btn-secondary"
              style={{ width: '100%', marginTop: '16px' }}
              onClick={() => onSelectSegment(activeSegmentIndex)}
            >
              Inspect this segment
            </button>
          </section>

          <section className="cinema-card" style={{ padding: '20px' }}>
            <h2 className="section-title" style={{ marginBottom: '14px' }}>Scene distribution</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '9px' }}>
              {distribution.map(([emotion, info]) => (
                <div key={emotion} className="dist-row">
                  <span style={{ textTransform: 'capitalize', color: info.count > 0 ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                    {emotion}
                  </span>
                  <div className="dist-track">
                    <div className="dist-fill" style={{ width: `${info.percentage}%`, background: emotionColor(emotion) }} />
                  </div>
                  <span className="font-mono" style={{ textAlign: 'right', color: info.count > 0 ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                    {info.percentage.toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
            {noEvidenceCount > 0 && (
              <p style={{ marginTop: '12px', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                {noEvidenceCount} of {segments.length} segments had no usable evidence, so the total is below 100%.
              </p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
