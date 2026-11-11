'use client';

import { useRef, useState } from 'react';
import { emotionColor } from './EmotionUI';

export function Row({ title, children }) {
  const track = useRef(null);

  const scroll = (direction) => {
    const el = track.current;
    el.scrollBy({ left: direction * el.clientWidth * 0.8, behavior: 'smooth' });
  };

  return (
    <section className="row">
      <h2 className="row-title">{title}</h2>
      <div className="row-viewport">
        <button className="row-arrow is-left" onClick={() => scroll(-1)} aria-label={`Scroll ${title} left`}>‹</button>
        <div ref={track} className="row-track">
          {children}
        </div>
        <button className="row-arrow is-right" onClick={() => scroll(1)} aria-label={`Scroll ${title} right`}>›</button>
      </div>
    </section>
  );
}

export function EmotionStrip({ strip, className }) {
  return (
    <div className={className} aria-hidden="true">
      {strip.map(([emotion, duration], i) => (
        <i key={i} style={{ flexGrow: Math.max(0.1, duration), flexBasis: 0, background: emotion === 'unknown' ? 'var(--line-strong)' : emotionColor(emotion) }} />
      ))}
    </div>
  );
}

const timeAgo = (iso) => {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return `${Math.round(hours / 24)} d ago`;
};

export function HistoryTile({ item, onOpen }) {
  const hasEmotion = !['unknown', 'mixed'].includes(item.dominant);
  return (
    <button className="tile" onClick={() => onOpen(item)}>
      <div className="tile-media">
        {item.thumb && <img src={item.thumb} alt="" loading="lazy" />}
        <span className="tile-tag">{item.kind === 'youtube' ? 'YOUTUBE' : 'UPLOAD'}</span>
        <span
          className="tile-badge"
          style={{ background: hasEmotion ? emotionColor(item.dominant) : 'var(--text-soft)' }}
        >
          {item.dominant === 'unknown' ? 'No evidence' : item.dominant}
        </span>
        <EmotionStrip strip={item.strip} className="tile-strip" />
      </div>
      <div className="tile-body">
        <div className="tile-title">{item.title}</div>
        <div className="tile-meta">
          {item.segments} segments · {item.duration.toFixed(0)}s · {timeAgo(item.date)}
        </div>
      </div>
    </button>
  );
}

// YouTube serves a 120 px wide grey image for deleted videos, so those cards hide themselves.
export function SceneTile({ scene, onOpen }) {
  const [isDead, setIsDead] = useState(false);
  if (isDead) return null;

  return (
    <button className="tile" onClick={() => onOpen(scene)}>
      <div className="tile-media">
        <img
          src={scene.thumbnail_url || `https://i.ytimg.com/vi/${scene.video_id}/hqdefault.jpg`}
          alt=""
          loading="lazy"
          onLoad={(e) => e.currentTarget.naturalWidth <= 120 && setIsDead(true)}
          onError={() => setIsDead(true)}
        />
        <span className="tile-tag">{(scene.language || 'en').toUpperCase()}</span>
      </div>
      <div className="tile-body">
        <div className="tile-title">{scene.title}</div>
        <div className="tile-meta">{scene.movie_name || scene.channel_name} · {scene.duration}</div>
      </div>
    </button>
  );
}
