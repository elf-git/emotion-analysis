'use client';

import { useState } from 'react';
import ProgressBar from './ProgressBar';

export default function VideoUploader({
  videoFile,
  setVideoFile,
  videoPreviewUrl,
  setVideoPreviewUrl,
  samplingInterval,
  setSamplingInterval,
  segmentDuration,
  setSegmentDuration,
  language,
  setLanguage,
  translate,
  setTranslate,
  onAnalyze,
  isAnalyzing,
  analysisProgress = 0,
  analysisStage = '',
  onLoadDemo,
  onAnalyzeYouTube,
  youtubeInputUrl: externalYoutubeUrl,
  setYoutubeInputUrl: setExternalYoutubeUrl,
  selectedScene = null,
  setSelectedScene = null,
}) {
  const [inputMode, setInputMode] = useState('youtube'); // 'youtube' or 'upload'
  const [internalYoutubeUrl, setInternalYoutubeUrl] = useState('https://www.youtube.com/watch?v=yM3kO4_88Q8');
  const youtubeInputUrl = externalYoutubeUrl !== undefined ? externalYoutubeUrl : internalYoutubeUrl;
  const setYoutubeInputUrl = setExternalYoutubeUrl || setInternalYoutubeUrl;

  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelected = (file) => {
    setVideoFile(file);
    setVideoPreviewUrl(URL.createObjectURL(file));
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Upload & Configuration Panel */}
      <div className="cinema-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '14px', marginBottom: '20px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h2 style={{ fontSize: '1.2rem', fontWeight: '700', color: '#f4f4f5' }}>
                Video Pipeline: YouTube Stream or Local Upload
              </h2>
              <span className="metric-badge">
                yt-dlp Stream Engine
              </span>
            </div>
            <p style={{ color: '#71717a', fontSize: '0.85rem', marginTop: '4px' }}>
              Analyze video scenes directly from YouTube URLs or upload local video files for tri-modal emotion classification.
            </p>
          </div>

          {/* Mode Switcher Tabs */}
          <div style={{ display: 'flex', gap: '4px', background: '#18181b', padding: '3px', borderRadius: 'var(--radius-md)', border: '1px solid #27272a' }}>
            <button
              type="button"
              className={`studio-nav-btn ${inputMode === 'youtube' ? 'active' : ''}`}
              style={{ padding: '5px 12px', fontSize: '0.8rem' }}
              onClick={() => setInputMode('youtube')}
            >
              YouTube URL / ID
            </button>
            <button
              type="button"
              className={`studio-nav-btn ${inputMode === 'upload' ? 'active' : ''}`}
              style={{ padding: '5px 12px', fontSize: '0.8rem' }}
              onClick={() => setInputMode('upload')}
            >
              Upload Local File
            </button>
          </div>
        </div>

        {/* MODE A: YOUTUBE URL INPUT */}
        {inputMode === 'youtube' && (
          <div
            style={{
              background: '#09090b',
              border: '1px solid #27272a',
              borderRadius: 'var(--radius-md)',
              padding: '20px',
              marginBottom: '20px',
            }}
          >
            {/* Auto-populated Selected Scene Preview if arriving from Discovery */}
            {selectedScene && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '14px',
                  padding: '12px 16px',
                  background: '#121214',
                  border: '1px solid #27272a',
                  borderRadius: 'var(--radius-md)',
                  marginBottom: '18px',
                }}
              >
                <div style={{ position: 'relative', width: '100px', height: '56px', borderRadius: '4px', overflow: 'hidden', flexShrink: 0, background: '#000000', border: '1px solid #27272a' }}>
                  <img
                    src={selectedScene.thumbnail_url || `https://img.youtube.com/vi/${selectedScene.video_id}/hqdefault.jpg`}
                    alt={selectedScene.title}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '2px' }}>
                    <span className="metric-badge">
                      {selectedScene.language_name || selectedScene.language_code?.toUpperCase() || 'HINDI'}
                    </span>
                    <span style={{ fontSize: '0.74rem', color: '#71717a' }}>
                      {selectedScene.movie_name || selectedScene.channel_name || 'Selected Movie Scene'}
                    </span>
                  </div>
                  <h4 style={{ fontSize: '0.9rem', fontWeight: '600', color: '#f4f4f5', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {selectedScene.title}
                  </h4>
                </div>
                <button
                  type="button"
                  className="cinema-btn-primary"
                  style={{ padding: '7px 14px', fontSize: '0.82rem', flexShrink: 0 }}
                  onClick={() => onAnalyzeYouTube(youtubeInputUrl, selectedScene)}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? 'Processing...' : 'Analyze This Scene'}
                </button>
              </div>
            )}

            <div style={{ marginBottom: '14px' }}>
              <label style={{ fontSize: '0.75rem', color: '#a1a1aa', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '6px' }}>
                Paste YouTube Video URL or 11-Character Video ID
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 180px', gap: '10px' }}>
                <input
                  type="text"
                  className="cinema-input font-mono"
                  placeholder="https://www.youtube.com/watch?v=yoZLUnaC3JE or video ID"
                  value={youtubeInputUrl}
                  onChange={(e) => setYoutubeInputUrl(e.target.value)}
                  style={{ fontSize: '0.88rem' }}
                />
                <button
                  type="button"
                  className="cinema-btn-primary"
                  disabled={isAnalyzing || !youtubeInputUrl.trim()}
                  onClick={() => onAnalyzeYouTube(youtubeInputUrl.trim(), selectedScene)}
                >
                  {isAnalyzing ? 'Processing...' : 'Analyze YouTube'}
                </button>
              </div>
            </div>

            {/* Quick Preset Buttons */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', paddingTop: '12px', borderTop: '1px solid #18181b' }}>
              <span style={{ fontSize: '0.72rem', color: '#71717a', fontWeight: '600' }}>
                SAMPLE SCENES:
              </span>
              <button
                type="button"
                className="cinema-btn-secondary"
                style={{ fontSize: '0.74rem', padding: '4px 10px' }}
                onClick={() => setYoutubeInputUrl('https://www.youtube.com/watch?v=yM3kO4_88Q8')}
              >
                Kumbalangi Nights (ML)
              </button>
              <button
                type="button"
                className="cinema-btn-secondary"
                style={{ fontSize: '0.74rem', padding: '4px 10px' }}
                onClick={() => setYoutubeInputUrl('https://www.youtube.com/watch?v=S1XpD0r4m38')}
              >
                3 Idiots Hospital (HI)
              </button>
              <button
                type="button"
                className="cinema-btn-secondary"
                style={{ fontSize: '0.74rem', padding: '4px 10px' }}
                onClick={() => setYoutubeInputUrl('https://www.youtube.com/watch?v=t9yN821PqM8')}
              >
                Vikram Vedha (TA)
              </button>
              <button
                type="button"
                className="cinema-btn-secondary"
                style={{ fontSize: '0.74rem', padding: '4px 10px' }}
                onClick={() => setYoutubeInputUrl('https://www.youtube.com/watch?v=v77IOwXg3B4')}
              >
                Interstellar (EN)
              </button>
            </div>
          </div>
        )}

        {/* MODE B: FILE DROPZONE */}
        {inputMode === 'upload' && (
          <div style={{ display: 'grid', gridTemplateColumns: videoPreviewUrl ? '1fr 1fr' : '1fr', gap: '16px', marginBottom: '20px' }}>
            <div
              className={`cinema-dropzone ${dragActive ? 'drag-active' : ''} ${videoFile ? 'has-file' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => document.getElementById('cinema-video-file-input').click()}
            >
              <input
                id="cinema-video-file-input"
                type="file"
                accept="video/*"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileSelected(e.target.files[0]);
                  }
                }}
              />
              <div style={{ fontSize: '1.8rem', color: '#a1a1aa' }}>📁</div>
              <div>
                <div style={{ fontWeight: '600', fontSize: '0.92rem', color: '#f4f4f5', marginBottom: '2px' }}>
                  {videoFile ? videoFile.name : 'Select or drop video file here'}
                </div>
                <div style={{ fontSize: '0.78rem', color: '#71717a' }}>
                  {videoFile
                    ? `${(videoFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for analysis`
                    : 'Supports MP4, WebM, MOV up to 100MB'}
                </div>
              </div>
              {videoFile && (
                <span className="metric-badge">
                  File Selected
                </span>
              )}
            </div>

            {videoPreviewUrl && (
              <div className="cinema-card" style={{ padding: '12px', background: '#09090b', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', borderRadius: 'var(--radius-sm)', overflow: 'hidden', background: '#000000' }}>
                  <video src={videoPreviewUrl} controls playsInline style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px', fontSize: '0.75rem', color: '#71717a' }}>
                  <span>Local Preview Player</span>
                  <button
                    type="button"
                    style={{ background: 'transparent', border: 'none', color: '#a1a1aa', cursor: 'pointer', fontSize: '0.75rem' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      setVideoFile(null);
                      setVideoPreviewUrl(null);
                    }}
                  >
                    Remove File ✕
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Configuration Parameters Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px', alignItems: 'end', marginBottom: '20px' }}>
          <div>
            <label style={{ fontSize: '0.74rem', color: '#71717a', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
              Frame Sampling Rate ({samplingInterval}s)
            </label>
            <select className="cinema-select" value={samplingInterval} onChange={(e) => setSamplingInterval(e.target.value)}>
              <option value="0.5">0.5s (High Resolution)</option>
              <option value="1.0">1.0s (Standard Balanced)</option>
              <option value="1.5">1.5s (Accelerated Preview)</option>
              <option value="2.0">2.0s (Keyframe Sparse)</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.74rem', color: '#71717a', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
              Segment Duration ({segmentDuration}s)
            </label>
            <select className="cinema-select" value={segmentDuration} onChange={(e) => setSegmentDuration(e.target.value)}>
              <option value="2.0">2.0 Seconds (Fast Utterances)</option>
              <option value="3.0">3.0 Seconds (Recommended)</option>
              <option value="4.0">4.0 Seconds (Extended Turns)</option>
              <option value="5.0">5.0 Seconds (Narrative Beats)</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.74rem', color: '#71717a', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
              Spoken Language
            </label>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                height: '42px',
                padding: '0 12px',
                background: '#09090b',
                border: '1px solid #27272a',
                borderRadius: 'var(--radius-md)',
                color: '#f4f4f5',
                fontSize: '0.80rem',
                fontFamily: 'var(--font-mono)',
              }}
            >
              <span style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: '#22c55e' }}></span>
              <span>Auto-Detect (Groq Whisper Large V3)</span>
            </div>
          </div>

          <div>
            <label style={{ fontSize: '0.74rem', color: '#71717a', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
              IndicTrans2 Translation
            </label>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                height: '42px',
                padding: '0 12px',
                background: '#09090b',
                border: '1px solid #27272a',
                borderRadius: 'var(--radius-md)',
              }}
            >
              <input
                type="checkbox"
                id="cinema-translate-toggle"
                checked={translate}
                onChange={(e) => setTranslate(e.target.checked)}
                style={{ width: '15px', height: '15px', accentColor: '#fafafa', cursor: 'pointer' }}
              />
              <label htmlFor="cinema-translate-toggle" style={{ fontSize: '0.84rem', color: '#f4f4f5', cursor: 'pointer' }}>
                Translate to English
              </label>
            </div>
          </div>
        </div>

        {/* Action Button & Processing Telemetry with Loading Bar */}
        <div>
          {isAnalyzing ? (
            <ProgressBar progress={analysisProgress} stage={analysisStage} />
          ) : (
            <div>
              {inputMode === 'upload' ? (
                videoFile ? (
                  <button
                    className="cinema-btn-primary"
                    style={{ width: '100%', padding: '12px', fontSize: '0.92rem' }}
                    onClick={onAnalyze}
                  >
                    Analyze Uploaded Video File ({videoFile.name})
                  </button>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '10px' }}>
                    <button
                      type="button"
                      className="cinema-btn-secondary"
                      style={{ padding: '12px', fontSize: '0.88rem' }}
                      onClick={() => document.getElementById('cinema-video-file-input').click()}
                    >
                      Choose Local Video File (.mp4, .webm)
                    </button>
                    <button
                      type="button"
                      className="cinema-btn-primary"
                      style={{ padding: '12px', fontSize: '0.88rem' }}
                      onClick={() => onLoadDemo('ml')}
                    >
                      Run Instant Demo Scene
                    </button>
                  </div>
                )
              ) : (
                <button
                  className="cinema-btn-primary"
                  style={{ width: '100%', padding: '12px', fontSize: '0.92rem' }}
                  disabled={!youtubeInputUrl.trim() || isAnalyzing}
                  onClick={() => onAnalyzeYouTube(youtubeInputUrl.trim(), selectedScene)}
                >
                  {selectedScene
                    ? `Run Neural Pipeline on ${selectedScene.movie_name || selectedScene.title?.slice(0, 35)}`
                    : 'Analyze YouTube Video with Neural Pipeline'}
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
