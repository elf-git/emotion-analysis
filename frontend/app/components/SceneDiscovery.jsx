'use client';

import { useState } from 'react';
import ProgressBar from './ProgressBar';

export default function SceneDiscovery({
  sceneResults,
  onSearch,
  isSearching,
  searchMeta,
  onSelectSceneForAnalysis,
  analyzingVideoId = null,
  analysisProgress = 0,
  analysisStage = '',
}) {
  const [movieName, setMovieName] = useState('');
  const [sceneDesc, setSceneDesc] = useState('');
  const [characterName, setCharacterName] = useState('');
  const [language, setLanguage] = useState('all');
  const [emotion, setEmotion] = useState('all');
  const [activeIframeScene, setActiveIframeScene] = useState(null);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    onSearch({
      movie: movieName,
      scene: sceneDesc,
      character: characterName,
      language,
      emotion,
    });
  };

  const handleApplyPreset = (pMovie, pScene, pLang, pEmo) => {
    setMovieName(pMovie);
    setSceneDesc(pScene);
    setLanguage(pLang);
    setEmotion(pEmo);
    onSearch({
      movie: pMovie,
      scene: pScene,
      character: '',
      language: pLang,
      emotion: pEmo,
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Search Filter Panel */}
      <div className="cinema-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px', marginBottom: '18px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 className="page-title">Scene discovery</h1>
              <span className="metric-badge">
                YouTube Index
              </span>
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '4px' }}>
              Search for movie scenes by title, moment, character, language or emotion.
            </p>
          </div>

          {searchMeta && (
            <span className="metric-badge">
              Source: {searchMeta.source}
            </span>
          )}
        </div>

        {/* Form Controls */}
        <form onSubmit={handleSubmit}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginBottom: '16px' }}>
            <div>
              <label style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
                Movie Title
              </label>
              <input
                type="text"
                className="cinema-input"
                placeholder="e.g. Kumbalangi Nights, 3 Idiots..."
                value={movieName}
                onChange={(e) => setMovieName(e.target.value)}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
                Scene Description
              </label>
              <input
                type="text"
                className="cinema-input"
                placeholder="e.g. dinner table confrontation..."
                value={sceneDesc}
                onChange={(e) => setSceneDesc(e.target.value)}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
                Character Name
              </label>
              <input
                type="text"
                className="cinema-input"
                placeholder="e.g. Shammi, Rancho..."
                value={characterName}
                onChange={(e) => setCharacterName(e.target.value)}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
                Language
              </label>
              <select className="cinema-select" value={language} onChange={(e) => setLanguage(e.target.value)}>
                <option value="all">All Languages</option>
                <option value="en">English</option>
                <option value="hi">Hindi (हिन्दी)</option>
                <option value="ml">Malayalam (മലയാളം)</option>
                <option value="ta">Tamil (தமிழ்)</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '5px' }}>
                Emotion Filter
              </label>
              <select className="cinema-select" value={emotion} onChange={(e) => setEmotion(e.target.value)}>
                <option value="all">All Emotions</option>
                <option value="joy">Joy / Celebration</option>
                <option value="anger">Anger / Confrontation</option>
                <option value="sadness">Sadness / Grief</option>
                <option value="surprise">Surprise / Shock</option>
                <option value="fear">Fear / Suspense</option>
                <option value="disgust">Disgust / Contempt</option>
                <option value="neutral">Neutral / Conversational</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
            {/* Quick Presets */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: '600' }}>
                PRESETS:
              </span>
              <button
                type="button"
                className="cinema-btn-secondary"
                style={{ fontSize: '0.75rem', padding: '4px 10px' }}
                onClick={() => handleApplyPreset('Kumbalangi Nights', 'Shammi dinner scene', 'ml', 'anger')}
              >
                Kumbalangi Nights (ML)
              </button>
              <button
                type="button"
                className="cinema-btn-secondary"
                style={{ fontSize: '0.75rem', padding: '4px 10px' }}
                onClick={() => handleApplyPreset('3 Idiots', 'Rancho rooftop conversation', 'hi', 'anger')}
              >
                3 Idiots (HI)
              </button>
              <button
                type="button"
                className="cinema-btn-secondary"
                style={{ fontSize: '0.75rem', padding: '4px 10px' }}
                onClick={() => handleApplyPreset('Super Deluxe', 'Shilpa emotional revelation', 'ta', 'sadness')}
              >
                Super Deluxe (TA)
              </button>
              <button
                type="button"
                className="cinema-btn-secondary"
                style={{ fontSize: '0.75rem', padding: '4px 10px' }}
                onClick={() => handleApplyPreset('Interstellar', 'Cooper watches 23 years messages', 'en', 'sadness')}
              >
                Interstellar (EN)
              </button>
            </div>

            <button
              type="submit"
              className="cinema-btn-primary"
              disabled={isSearching}
              style={{ minWidth: '150px' }}
            >
              {isSearching ? 'Searching...' : 'Search Scenes'}
            </button>
          </div>
        </form>
      </div>
      {searchMeta?.source === 'curated_mock_catalogue' && (
        <div className="cinema-card" style={{ padding: '12px 16px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--warn)', fontWeight: 600 }}>Sample list, not live search.</strong>{' '}
          Add a SerpApi key to the .env file for real results. Many sample links no longer work, so paste your own link in New Analysis.
        </div>
      )}


      {/* Structured Search Results Grid */}
      {sceneResults.length > 0 ? (
        <div className="youtube-discovery-grid">
          {sceneResults.map((scene, idx) => (
            <div key={idx} className="cinema-card" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
              <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', background: '#000000', borderBottom: '1px solid var(--line)' }}>
                <img
                  src={scene.thumbnail_url}
                  alt={scene.title}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  loading="lazy"
                />
                {scene.duration && (
                  <span
                    style={{
                      position: 'absolute',
                      bottom: '6px',
                      right: '6px',
                      background: 'rgba(0, 0, 0, 0.85)',
                      color: 'var(--text-primary)',
                      fontSize: '0.7rem',
                      fontWeight: '600',
                      padding: '2px 6px',
                      borderRadius: '3px',
                    }}
                  >
                    {scene.duration}
                  </span>
                )}
                <span
                  style={{
                    position: 'absolute',
                    top: '6px',
                    left: '6px',
                    background: 'var(--surface)',
                    color: 'var(--text-soft)',
                    border: '1px solid var(--line)',
                    fontSize: '0.68rem',
                    fontWeight: '600',
                    padding: '2px 6px',
                    borderRadius: 'var(--radius-sm)',
                  }}
                >
                  {scene.language_code?.toUpperCase() || 'EN'}
                </span>
              </div>

              <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', flex: 1, justifyContent: 'space-between' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span className="metric-badge">
                      {scene.emotion_tag ? scene.emotion_tag.toUpperCase() : 'SCENE'}
                    </span>
                    <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                      {scene.channel_name}
                    </span>
                  </div>

                  <h3
                    style={{
                      fontSize: '0.9rem',
                      fontWeight: '600',
                      lineHeight: 1.4,
                      marginBottom: '6px',
                      color: 'var(--text-primary)',
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                    }}
                  >
                    {scene.title}
                  </h3>

                  {scene.description && (
                    <p
                      style={{
                        fontSize: '0.76rem',
                        color: 'var(--text-secondary)',
                        lineHeight: 1.4,
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                      }}
                    >
                      {scene.description}
                    </p>
                  )}
                </div>

                {analyzingVideoId === scene.video_id ? (
                  <div style={{ marginTop: '12px' }}>
                    <ProgressBar progress={analysisProgress} stage={analysisStage} title="Running Neural Pipeline" />
                  </div>
                ) : (
                  <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                    <button
                      className="cinema-btn-primary"
                      style={{ flex: 1, padding: '7px', fontSize: '0.8rem' }}
                      onClick={() => {
                        setActiveIframeScene(scene);
                        if (onSelectSceneForAnalysis) onSelectSceneForAnalysis(scene, false);
                      }}
                    >
                      Watch Preview
                    </button>
                    <button
                      className="cinema-btn-secondary"
                      style={{ padding: '7px 12px', fontSize: '0.8rem' }}
                      title="Send to Neural Pipeline"
                      onClick={() => onSelectSceneForAnalysis(scene, true)}
                    >
                      Analyze ➔
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="cinema-card" style={{ padding: '48px 24px', textAlign: 'center', color: 'var(--text-muted)' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: '600', color: 'var(--text-primary)' }}>Search to find scenes</h3>
          <p style={{ fontSize: '0.82rem', marginTop: '4px' }}>
            Enter a movie or moment above, or pick a preset.
          </p>
        </div>
      )}

      {/* Official YouTube IFrame Player Modal */}
      {activeIframeScene && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px',
          }}
          onClick={() => setActiveIframeScene(null)}
        >
          <div
            className="cinema-card"
            style={{ maxWidth: '860px', width: '100%', overflow: 'hidden', background: 'var(--surface)' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ position: 'relative', width: '100%', paddingBottom: '56.25%', background: '#000000' }}>
              <iframe
                src={`${activeIframeScene.embed_url || `https://www.youtube.com/embed/${activeIframeScene.video_id}`}?enablejsapi=1&autoplay=1`}
                title={activeIframeScene.title}
                style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', border: 0 }}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>

            <div style={{ padding: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '16px', marginBottom: '12px' }}>
                <div>
                  <h3 style={{ fontSize: '1.05rem', fontWeight: '700', color: 'var(--text-primary)' }}>{activeIframeScene.title}</h3>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                    Channel: <strong>{activeIframeScene.channel_name}</strong> • Duration: {activeIframeScene.duration || 'N/A'} • Lang: {activeIframeScene.language_code?.toUpperCase()}
                  </div>
                </div>
                <button className="cinema-btn-secondary" onClick={() => setActiveIframeScene(null)}>
                  ✕ Close
                </button>
              </div>

              {/* Action */}
              <div
                style={{
                  background: 'var(--surface-2)',
                  border: '1px solid var(--line)',
                  borderRadius: 'var(--radius-md)',
                  padding: '12px 16px',
                  fontSize: '0.82rem',
                  color: 'var(--text-secondary)',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: '10px',
                }}
              >
                <div>
                  Ready to run multimodal emotion analysis on this scene?
                </div>
                <button
                  className="cinema-btn-primary"
                  style={{ fontSize: '0.8rem', padding: '6px 14px' }}
                  onClick={() => {
                    const sceneToAnalyze = activeIframeScene;
                    setActiveIframeScene(null);
                    onSelectSceneForAnalysis(sceneToAnalyze);
                  }}
                >
                  Analyze in Video Pipeline ➔
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
