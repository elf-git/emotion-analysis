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
  const [internalYoutubeUrl, setInternalYoutubeUrl] = useState('');
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
            <h1 className="page-title">New analysis</h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '4px' }}>
              Paste a YouTube link or upload a clip. Short scenes with clear faces and speech work best.
            </p>
          </div>

          {/* Mode Switcher Tabs */}
          <div className="segmented">
            <button
              type="button"
              className={`studio-nav-btn ${inputMode === 'youtube' ? 'active' : ''}`}
                            onClick={() => setInputMode('youtube')}
            >
              YouTube link
            </button>
            <button
              type="button"
              className={`studio-nav-btn ${inputMode === 'upload' ? 'active' : ''}`}
                            onClick={() => setInputMode('upload')}
            >
              Upload file
            </button>
          </div>
        </div>

        {/* MODE A: YOUTUBE URL INPUT */}
        {inputMode === 'youtube' && (
          <div
            style={{
              background: 'var(--bg-inset)',
              border: '1px solid var(--line)',
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
                  background: 'var(--surface)',
                  border: '1px solid var(--line)',
                  borderRadius: 'var(--radius-md)',
                  marginBottom: '18px',
                }}
              >
                <div style={{ position: 'relative', width: '100px', height: '56px', borderRadius: '4px', overflow: 'hidden', flexShrink: 0, background: '#000000', border: '1px solid var(--line)' }}>
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
                    <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                      {selectedScene.movie_name || selectedScene.channel_name || 'Selected Movie Scene'}
                    </span>
                  </div>
                  <h4 style={{ fontSize: '0.9rem', fontWeight: '600', color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
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

            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '6px' }}>
                YouTube link or video ID
              </label>
              <div className="url-row">
                <input
                  type="text"
                  className="cinema-input font-mono"
                  placeholder="https://www.youtube.com/watch?v=..."
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
                  {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                </button>
              </div>
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
              <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 16V4" />
                <path d="m7 9 5-5 5 5" />
                <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
              </svg>
              <div>
                <div style={{ fontWeight: '600', fontSize: '0.92rem', color: 'var(--text-primary)', marginBottom: '2px' }}>
                  {videoFile ? videoFile.name : 'Choose a video or drop it here'}
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {videoFile
                    ? `${(videoFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for analysis`
                    : 'MP4, WebM, MOV, AVI or MKV, up to 30 MB'}
                </div>
              </div>
              {videoFile && (
                <span className="metric-badge">
                  File Selected
                </span>
              )}
            </div>

            {videoPreviewUrl && (
              <div className="cinema-card" style={{ padding: '12px', background: 'var(--bg-inset)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', borderRadius: 'var(--radius-sm)', overflow: 'hidden', background: '#000000' }}>
                  <video src={videoPreviewUrl} controls playsInline style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  <span>Preview</span>
                  <button
                    type="button"
                    style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '0.75rem' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      setVideoFile(null);
                      setVideoPreviewUrl(null);
                    }}
                  >
                    Remove file
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Configuration Parameters Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px', alignItems: 'end', marginBottom: '20px' }}>
          <div>
            <label style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
              Frame sampling
            </label>
            <select className="cinema-select" value={samplingInterval} onChange={(e) => setSamplingInterval(e.target.value)}>
              <option value="0.5">Every 0.5 seconds</option>
              <option value="1.0">Every 1 second</option>
              <option value="1.5">Every 1.5 seconds</option>
              <option value="2.0">Every 2 seconds</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
              Segment length
            </label>
            <select className="cinema-select" value={segmentDuration} onChange={(e) => setSegmentDuration(e.target.value)}>
              <option value="2.0">2 seconds</option>
              <option value="3.0">3 seconds (recommended)</option>
              <option value="4.0">4 seconds</option>
              <option value="5.0">5 seconds</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
              Spoken Language
            </label>
            <select className="cinema-select" value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="auto">Detect automatically</option>
              <option value="en">English</option>
              <option value="hi">Hindi (हिन्दी)</option>
              <option value="ml">Malayalam (മലയാളം)</option>
              <option value="ta">Tamil (தமிழ்)</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
              Translation
            </label>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                height: '42px',
                padding: '0 12px',
                background: 'var(--bg-inset)',
                border: '1px solid var(--line)',
                borderRadius: 'var(--radius-md)',
              }}
            >
              <input
                type="checkbox"
                id="cinema-translate-toggle"
                checked={translate}
                onChange={(e) => setTranslate(e.target.checked)}
                style={{ width: '15px', height: '15px', accentColor: '#f2efe8', cursor: 'pointer' }}
              />
              <label htmlFor="cinema-translate-toggle" style={{ fontSize: '0.84rem', color: 'var(--text-primary)', cursor: 'pointer' }}>
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
                    Analyze {videoFile.name}
                  </button>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '10px' }}>
                    <button
                      type="button"
                      className="cinema-btn-secondary"
                      style={{ padding: '12px', fontSize: '0.88rem' }}
                      onClick={() => document.getElementById('cinema-video-file-input').click()}
                    >
                      Choose a video file
                    </button>
                    <button
                      type="button"
                      className="cinema-btn-secondary"
                      style={{ padding: '12px', fontSize: '0.88rem' }}
                      onClick={() => onLoadDemo('ml')}
                    >
                      View Scripted Sample
                    </button>
                  </div>
                )
              ) : null}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
