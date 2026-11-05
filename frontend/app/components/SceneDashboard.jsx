'use client';

export default function SceneDashboard({
  clipResult,
  videoPlayerRef,
  videoPreviewUrl,
  currentPlaybackTime,
  activeSegmentIndex,
  onSeekToSegment,
  onSelectSegment,
  onTimeUpdate,
}) {
  if (!clipResult) {
    return (
      <div className="cinema-card" style={{ padding: '60px 24px', textAlign: 'center', color: '#71717a' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#f4f4f5' }}>No Full-Scene Analysis Loaded</h3>
        <p style={{ fontSize: '0.85rem', marginTop: '6px' }}>
          Upload a video in the <strong>Video Upload</strong> tab or load a sample scene to inspect the timeline.
        </p>
      </div>
    );
  }

  const { scene_metadata, scene_summary, segments } = clipResult;
  const activeSegment = segments[activeSegmentIndex] || segments[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Top Scene Telemetry Banner */}
      <div
        className="cinema-card"
        style={{
          padding: '20px 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '16px',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
            <span style={{ fontSize: '0.72rem', color: '#71717a', textTransform: 'uppercase', letterSpacing: '0.04em', fontWeight: '700' }}>
              SCENE ANALYTICS
            </span>
            <span className="metric-badge">
              {scene_metadata.detected_language || 'Multilingual'}
            </span>
            <span className="metric-badge">
              {clipResult.is_live_inference ? 'Live Neural Pipeline' : 'Reference Demo'}
            </span>
          </div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: '700', color: '#f4f4f5' }}>
            {scene_metadata.filename || 'Full-Clip Cinema Scene'}
          </h2>
        </div>

        {/* Scene Summary Metrics */}
        <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.72rem', color: '#71717a', textTransform: 'uppercase' }}>Dominant Emotion</div>
            <div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#f4f4f5', textTransform: 'uppercase' }}>
              {scene_summary.dominant_emotion}
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.72rem', color: '#71717a', textTransform: 'uppercase' }}>Valence / Arousal</div>
            <div className="font-mono" style={{ fontSize: '1.05rem', fontWeight: '600', color: '#d4d4d8' }}>
              {scene_summary.average_valence > 0 ? `+${scene_summary.average_valence}` : scene_summary.average_valence} / {scene_summary.average_arousal}
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.72rem', color: '#71717a', textTransform: 'uppercase' }}>Duration / Segments</div>
            <div className="font-mono" style={{ fontSize: '1.05rem', fontWeight: '600', color: '#f4f4f5' }}>
              {scene_metadata.duration_sec.toFixed(1)}s ({scene_summary.total_segments} turns)
            </div>
          </div>
        </div>
      </div>

      {/* Main Studio Grid: Video Player + Synchronized Scrubber */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(360px, 1.3fr) minmax(320px, 1fr)', gap: '20px', alignItems: 'start' }}>
        {/* Left Column: Player & Interactive Timeline Scrubber */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Cinema Player Box */}
          <div className="cinema-card" style={{ padding: '16px', background: '#09090b' }}>
            <div className="cinema-player-box" style={{ position: 'relative', width: '100%', minHeight: '340px', background: '#000000', borderRadius: '8px', overflow: 'hidden' }}>
              {videoPreviewUrl ? (
                <video
                  ref={videoPlayerRef}
                  src={videoPreviewUrl}
                  controls
                  playsInline
                  onTimeUpdate={onTimeUpdate}
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
              ) : (
                <iframe
                  src={clipResult.embed_url || `https://www.youtube.com/embed/${clipResult.video_id || 'v77IOwXg3B4'}?enablejsapi=1&autoplay=0&rel=0`}
                  title={scene_metadata.filename || 'Movie Scene Player'}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  style={{ width: '100%', height: '100%', minHeight: '340px', border: 'none' }}
                />
              )}
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px', fontSize: '0.78rem', color: '#a1a1aa' }}>
              <span>
                Playback Timecode:{' '}
                <strong className="font-mono" style={{ color: '#f4f4f5' }}>
                  {currentPlaybackTime.toFixed(2)}s
                </strong>{' '}
                / {scene_metadata.duration_sec.toFixed(1)}s
              </span>
              <span>
                Active Segment: <strong>#{activeSegmentIndex + 1}</strong> ({activeSegment.timestamp_label})
              </span>
            </div>
          </div>

          {/* Synchronized Timeline Strip Component */}
          <div className="timeline-strip-container">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.78rem' }}>
              <span style={{ fontWeight: '600', color: '#f4f4f5' }}>
                Synchronized Emotion Timeline (Click to Seek)
              </span>
              <span style={{ color: '#71717a' }}>
                {scene_summary.total_segments} Dialogue Segments
              </span>
            </div>

            {/* Interactive Visual Bar Strip (monochrome / zinc tones) */}
            <div className="timeline-strip-track">
              {segments.map((seg, idx) => {
                const totalDuration = scene_metadata.duration_sec || 1;
                const segDuration = seg.end_time_sec - seg.start_time_sec;
                const widthPct = Math.max(5, (segDuration / totalDuration) * 100);
                const isActive = activeSegmentIndex === idx;

                return (
                  <div
                    key={idx}
                    className={`timeline-segment-block ${isActive ? 'is-active' : ''}`}
                    style={{
                      width: `${widthPct}%`,
                      background: isActive ? '#ffffff' : (idx % 2 === 0 ? '#3f3f46' : '#27272a'),
                    }}
                    onClick={() => onSeekToSegment(seg, idx)}
                    title={`Segment #${idx + 1}: ${seg.timestamp_label} (${seg.fused_prediction.predicted_emotion})`}
                  />
                );
              })}
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: '#71717a', marginTop: '4px' }}>
              <span>0.00s</span>
              <span>Active segment highlighted in white</span>
              <span>{scene_metadata.duration_sec.toFixed(1)}s</span>
            </div>
          </div>

          {/* Emotion Transitions Flow */}
          <div className="cinema-card" style={{ padding: '16px' }}>
            <div style={{ fontSize: '0.74rem', color: '#71717a', textTransform: 'uppercase', fontWeight: '600', marginBottom: '10px' }}>
              Chronological Emotion Transition Flow
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '8px' }}>
              {scene_summary.emotion_transitions?.map((emo, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span
                    className="metric-badge"
                    style={{
                      cursor: 'pointer',
                      background: activeSegmentIndex === idx ? '#27272a' : '#18181b',
                      color: activeSegmentIndex === idx ? '#ffffff' : '#d4d4d8',
                      borderColor: activeSegmentIndex === idx ? '#52525b' : '#27272a',
                    }}
                    onClick={() => onSeekToSegment(segments[idx] || segments[0], idx)}
                  >
                    #{idx + 1} {emo}
                  </span>
                  {idx < scene_summary.emotion_transitions.length - 1 && (
                    <span style={{ color: '#52525b', fontSize: '0.8rem' }}>➔</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Scene Emotion Distribution & Active Segment Card */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* 7-Class Scene Emotion Distribution */}
          <div className="cinema-card" style={{ padding: '20px' }}>
            <h3 style={{ fontSize: '0.92rem', fontWeight: '600', color: '#f4f4f5', marginBottom: '14px' }}>
              Scene-Level Emotion Distribution
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '9px' }}>
              {Object.entries(scene_summary.emotion_distribution).map(([emo, info]) => (
                <div key={emo} style={{ display: 'grid', gridTemplateColumns: '75px 1fr 45px', alignItems: 'center', gap: '10px', fontSize: '0.82rem' }}>
                  <span style={{ textTransform: 'capitalize', color: '#a1a1aa' }}>{emo}</span>
                  <div style={{ height: '6px', background: '#18181b', borderRadius: 'var(--radius-full)', overflow: 'hidden', border: '1px solid #27272a' }}>
                    <div
                      style={{
                        width: `${info.percentage}%`,
                        height: '100%',
                        background: info.percentage > 0 ? '#fafafa' : 'transparent',
                        borderRadius: 'var(--radius-full)',
                        transition: 'width 0.3s ease',
                      }}
                    />
                  </div>
                  <span className="font-mono" style={{ textAlign: 'right', color: info.count > 0 ? '#f4f4f5' : '#71717a' }}>
                    {info.percentage.toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Active Segment Summary Quickview */}
          <div
            className="cinema-card"
            style={{
              padding: '20px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <span className="metric-badge">
                Turn #{activeSegment.segment_index + 1} ({activeSegment.timestamp_label})
              </span>
              <span className="metric-badge" style={{ color: '#ffffff' }}>
                {activeSegment.fused_prediction.predicted_emotion.toUpperCase()} ({activeSegment.fused_prediction.confidence_pct.toFixed(1)}%)
              </span>
            </div>

            <div className="font-indic" style={{ fontSize: '1rem', fontStyle: 'italic', color: '#ffffff', marginBottom: '8px', lineHeight: 1.5 }}>
              "{activeSegment.transcript.original_text}"
            </div>

            {activeSegment.transcript.translated_text && (
              <div style={{ fontSize: '0.82rem', color: '#a1a1aa', marginBottom: '16px' }}>
                EN: "{activeSegment.transcript.translated_text}"
              </div>
            )}

            {/* Modality Dynamic Gating Progress Bar (Clean Monochrome) */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: '#71717a', marginBottom: '6px' }}>
                <span>Dynamic Attention Weights</span>
                <span style={{ color: '#f4f4f5' }}>Driver: {activeSegment.modality_contributions.dominant_modality}</span>
              </div>
              <div style={{ height: '6px', background: '#18181b', borderRadius: 'var(--radius-full)', overflow: 'hidden', display: 'flex', border: '1px solid #27272a' }}>
                <div style={{ width: `${activeSegment.modality_contributions.text_pct}%`, background: '#ffffff' }} title="Text" />
                <div style={{ width: `${activeSegment.modality_contributions.audio_pct}%`, background: '#a1a1aa' }} title="Audio" />
                <div style={{ width: `${activeSegment.modality_contributions.video_pct}%`, background: '#52525b' }} title="Video" />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#71717a', marginTop: '6px' }}>
                <span style={{ color: '#f4f4f5' }}>Text: {activeSegment.modality_contributions.text_pct}%</span>
                <span style={{ color: '#a1a1aa' }}>Audio: {activeSegment.modality_contributions.audio_pct}%</span>
                <span style={{ color: '#71717a' }}>Video: {activeSegment.modality_contributions.video_pct}%</span>
              </div>
            </div>

            <button
              className="cinema-btn-secondary"
              style={{ width: '100%', marginTop: '16px', fontSize: '0.82rem' }}
              onClick={() => onSelectSegment(activeSegmentIndex)}
            >
              Open Segment Deep Inspector ➔
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
