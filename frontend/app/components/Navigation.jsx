'use client';

import { useEffect, useState } from 'react';
import { EMOTION_COLORS, emotionColor } from './EmotionUI';

const LINKS = [
  { id: 'landing', label: 'Home' },
  { id: 'upload', label: 'New Analysis' },
  { id: 'discovery', label: 'Discover' },
  { id: 'dashboard', label: 'Dashboard', needsResult: true },
  { id: 'segment', label: 'Segments', needsResult: true },
  { id: 'explainability', label: 'Explainability', needsResult: true },
  { id: 'transcript', label: 'Transcript', needsResult: true },
  { id: 'evaluation', label: 'Evaluation' },
];

function statusFor(apiHealth) {
  if (apiHealth.status !== 'healthy') return { label: 'Connecting', color: 'var(--warn)' };
  if (!apiHealth.models_loaded) return { label: 'Models missing', color: 'var(--danger)' };
  return { label: 'Models ready', color: 'var(--good)' };
}

export default function Navigation({ activeScreen, setActiveScreen, apiHealth, resultEmotion }) {
  const status = statusFor(apiHealth);
  const [scrolled, setScrolled] = useState(false);

  // The bar is see-through over the home banner and turns solid once you scroll
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 30);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const isSolid = scrolled || activeScreen !== 'landing';

  return (
    <header className={`topnav ${isSolid ? 'is-solid' : ''}`}>
      <button className="brand" onClick={() => setActiveScreen('landing')} aria-label="CinemaAI home">
        <span className="brand-mark" aria-hidden="true">
          {Object.values(EMOTION_COLORS).map((color) => (
            <i key={color} style={{ background: color }} />
          ))}
        </span>
        <span className="brand-name">CinemaAI</span>
      </button>

      <nav className="topnav-links" aria-label="Main">
        {LINKS.map((link) => (
          <button
            key={link.id}
            className={`topnav-link ${activeScreen === link.id ? 'active' : ''} ${link.needsResult && !resultEmotion ? 'is-empty' : ''}`}
            aria-current={activeScreen === link.id ? 'page' : undefined}
            onClick={() => setActiveScreen(link.id)}
          >
            {link.label}
            {link.id === 'dashboard' && resultEmotion && (
              <span className="nav-result-dot" style={{ background: emotionColor(resultEmotion) }} aria-hidden="true" />
            )}
          </button>
        ))}
      </nav>

      <div className="status-pill" role="status">
        <span className="status-dot" style={{ background: status.color }} aria-hidden="true" />
        {status.label}
      </div>
    </header>
  );
}
