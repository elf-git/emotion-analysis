'use client';

import { useRef, useState } from 'react';
import { MOTION_OK, ScrollTrigger, SplitText, createScrubber, gsap, useGSAP } from '../lib/gsap';
import { emotionColor } from './EmotionUI';
import { EmotionStrip, HistoryTile, Row, SceneTile } from './Rows';

// Hand-written example of three readings disagreeing. Not a model result.
const DEMO_SCENE = [
  { end: 0.3, fused: 'neutral', words: 'neutral', voice: 'neutral', face: 'neutral', line: '“Sit down. Let’s talk.”' },
  { end: 0.75, fused: 'anger', words: 'neutral', voice: 'anger', face: 'anger', line: '“I said I’m fine.”' },
  { end: 1, fused: 'sadness', words: 'neutral', voice: 'sadness', face: 'sadness', line: '“It doesn’t matter now.”' },
];

const DEMO_LANES = [
  { key: 'words', label: 'Words' },
  { key: 'voice', label: 'Voice' },
  { key: 'face', label: 'Face' },
  { key: 'fused', label: 'Fused', isFused: true },
];

const EMOTIONS = [
  { name: 'Neutral', key: 'neutral', text: 'No clear emotion stands out.' },
  { name: 'Joy', key: 'joy', text: 'Happiness and positive excitement.' },
  { name: 'Surprise', key: 'surprise', text: 'A reaction to something unexpected.' },
  { name: 'Sadness', key: 'sadness', text: 'Sad or distressed expression.' },
  { name: 'Fear', key: 'fear', text: 'Fearful or anxious expression.' },
  { name: 'Disgust', key: 'disgust', text: 'Aversion or repulsion.' },
  { name: 'Anger', key: 'anger', text: 'Angry or irritated expression.' },
];

const STEPS = [
  { title: 'Extract', text: 'FFmpeg separates the audio, Whisper transcribes speech with timestamps, and frames are sampled from the video.' },
  { title: 'Read', text: 'Three models work on each time segment: one for words, one for voice, and one for faces.' },
  { title: 'Combine', text: 'Late fusion merges the probabilities. A missing signal is left out and never treated as neutral.' },
  { title: 'Explain', text: 'Integrated Gradients scores each word, Grad-CAM maps the face, and pitch and energy are measured.' },
];

const MODELS = [
  { part: 'Words', model: 'DistilRoBERTa', covers: '7 emotions', limit: 'English only. Other languages go through translation.' },
  { part: 'Face', model: 'Vision Transformer + OpenCV', covers: '7 emotions', limit: 'Reads the largest face, which may not be the speaker.' },
  { part: 'Voice', model: 'wav2vec 2.0', covers: '4 emotions', limit: 'Unreliable on film audio, so it has a low weight.' },
  { part: 'Speech', model: 'Faster-Whisper small', covers: 'Speech to text', limit: 'Weaker on Malayalam and Tamil.' },
];

const AMBIENT = [
  { emotion: 'anger', left: '58%', top: '-12%' },
  { emotion: 'sadness', left: '76%', top: '30%' },
  { emotion: 'fear', left: '40%', top: '48%' },
  { emotion: 'joy', left: '88%', top: '-20%' },
];

const segmentAt = (progress) => DEMO_SCENE.findIndex((segment) => progress <= segment.end);
const capitalize = (text) => text.charAt(0).toUpperCase() + text.slice(1);

function SceneDemo() {
  const root = useRef(null);
  const track = useRef(null);
  const handle = useRef(null);
  const word = useRef(null);
  const position = useRef(0.08);
  const sweep = useRef(null);
  const [active, setActive] = useState(0);

  const moveTo = (progress) => {
    position.current = progress;
    gsap.set(handle.current, { left: `${progress * 100}%` });
    handle.current.setAttribute('aria-valuenow', Math.round(progress * 100));
    const next = segmentAt(progress);
    setActive((current) => (current === next ? current : next));
  };

  const takeControl = () => sweep.current?.kill();

  useGSAP(
    () => {
      moveTo(position.current);
      const scrubber = createScrubber(track.current, track.current, { onPress: takeControl, onScrub: moveTo });

      const mm = gsap.matchMedia();
      mm.add(MOTION_OK, () => {
        gsap
          .timeline({ delay: 0.5 })
          .from('.demo-card', { autoAlpha: 0, y: 30, duration: 0.9, ease: 'expo.out' })
          .from('.lane > div', { scaleX: 0, transformOrigin: 'left center', duration: 0.7, stagger: 0.05, ease: 'power3.out' }, '-=0.5');

        const state = { progress: position.current };
        sweep.current = gsap.to(state, {
          progress: 0.94,
          duration: 7,
          delay: 1.6,
          ease: 'sine.inOut',
          repeat: -1,
          yoyo: true,
          onUpdate: () => moveTo(state.progress),
        });
      });

      return () => {
        scrubber.kill();
        mm.revert();
      };
    },
    { scope: root }
  );

  useGSAP(
    () => {
      const mm = gsap.matchMedia();
      mm.add(MOTION_OK, () => {
        gsap.fromTo(word.current, { yPercent: 55, autoAlpha: 0 }, { yPercent: 0, autoAlpha: 1, duration: 0.5, ease: 'power3.out' });
      });
      return () => mm.revert();
    },
    { scope: root, dependencies: [active] }
  );

  const onKeyDown = (event) => {
    const step = { ArrowLeft: -0.05, ArrowRight: 0.05 }[event.key];
    if (!step) return;
    event.preventDefault();
    takeControl();
    moveTo(gsap.utils.clamp(0, 1, position.current + step));
  };

  const scene = DEMO_SCENE[active];

  return (
    <div ref={root}>
      <figure className="cinema-card demo-card">
        <figcaption style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', marginBottom: '10px' }}>
          <span className="section-title">Drag through this scene</span>
          <span className="metric-badge">Illustration</span>
        </figcaption>

        <div className="demo-readout">
          <div className="demo-word-mask">
            <div ref={word} className="demo-word" style={{ color: emotionColor(scene.fused) }}>
              {scene.fused}
            </div>
          </div>
          <p className="demo-line">{scene.line}</p>
        </div>

        <div className="demo-grid">
          <div className="demo-labels">
            {DEMO_LANES.map((lane) => (
              <span key={lane.key} className={lane.isFused ? 'is-fused' : ''}>{lane.label}</span>
            ))}
          </div>
          <div ref={track} className="demo-track">
            {DEMO_LANES.map((lane) => (
              <div key={lane.key} className={`lane ${lane.isFused ? 'is-fused' : ''}`}>
                {DEMO_SCENE.map((segment, i) => (
                  <div
                    key={i}
                    style={{
                      flexGrow: segment.end - (DEMO_SCENE[i - 1]?.end ?? 0),
                      background: emotionColor(segment[lane.key]),
                      opacity: i === active ? 1 : 0.45,
                    }}
                  />
                ))}
              </div>
            ))}
            <div
              ref={handle}
              className="demo-handle"
              role="slider"
              tabIndex={0}
              aria-label="Scene position"
              aria-valuemin={0}
              aria-valuemax={100}
              onKeyDown={onKeyDown}
            />
          </div>
        </div>
      </figure>
    </div>
  );
}

function Backdrop({ latest }) {
  const [src, setSrc] = useState(
    latest?.kind === 'youtube' ? `https://i.ytimg.com/vi/${latest.videoId}/maxresdefault.jpg` : latest?.thumb
  );

  if (!latest || !src) {
    return (
      <div className="billboard-ambient" aria-hidden="true">
        {AMBIENT.map((blob) => (
          <span key={blob.emotion} style={{ left: blob.left, top: blob.top, background: emotionColor(blob.emotion) }} />
        ))}
      </div>
    );
  }

  return (
    <div
      className="billboard-backdrop"
      style={{ backgroundImage: `url("${src}")` }}
      aria-hidden="true"
    >
      {latest.kind === 'youtube' && (
        <img
          src={src}
          alt=""
          style={{ display: 'none' }}
          onLoad={(e) => e.currentTarget.naturalWidth <= 120 && setSrc(latest.thumb)}
          onError={() => setSrc(latest.thumb)}
        />
      )}
    </div>
  );
}

export default function LandingHero({ onNavigate, history = [], scenes = [], onOpenHistory, onOpenScene }) {
  const root = useRef(null);
  const latest = history[0];

  useGSAP(
    () => {
      const mm = gsap.matchMedia();

      mm.add(MOTION_OK, () => {
        const backdrop = gsap.utils.toArray('.billboard-backdrop, .billboard-ambient');
        const photo = gsap.utils.toArray('.billboard-backdrop');
        if (photo.length) {
          gsap.fromTo(photo, { scale: 1.2, autoAlpha: 0 }, { scale: 1.08, autoAlpha: 1, duration: 2.2, ease: 'power2.out' });
        }
        gsap.utils.toArray('.billboard-ambient span').forEach((blob, i) => {
          gsap.to(blob, {
            xPercent: gsap.utils.random(-25, 25),
            yPercent: gsap.utils.random(-20, 20),
            scale: gsap.utils.random(0.85, 1.2),
            duration: gsap.utils.random(7, 11),
            delay: i * 0.4,
            ease: 'sine.inOut',
            repeat: -1,
            yoyo: true,
          });
        });

        const split = SplitText.create('.billboard-title', {
          type: 'lines',
          mask: 'lines',
          autoSplit: true,
          onSplit: (self) => gsap.from(self.lines, { yPercent: 110, duration: 1.1, stagger: 0.12, ease: 'expo.out' }),
        });
        gsap.from('.billboard-rise', { autoAlpha: 0, y: 22, duration: 0.8, stagger: 0.1, delay: 0.4, ease: 'power3.out' });

        // Billboard drifts and dims as you scroll past it
        gsap.to(backdrop, {
          yPercent: 12,
          autoAlpha: 0.35,
          ease: 'none',
          scrollTrigger: { trigger: '.billboard', start: 'top top', end: 'bottom top', scrub: true },
        });

        gsap.utils.toArray('.row').forEach((row) => {
          gsap.from(row.querySelectorAll('.row-title, .row-track > *'), {
            autoAlpha: 0,
            x: 40,
            duration: 0.8,
            stagger: 0.06,
            ease: 'power3.out',
            scrollTrigger: { trigger: row, start: 'top 88%' },
          });
        });

        gsap.from('.models-card', {
          autoAlpha: 0,
          y: 40,
          duration: 0.9,
          ease: 'power3.out',
          scrollTrigger: { trigger: '.models-card', start: 'top 88%' },
        });

        return () => split.revert();
      });

      document.fonts?.ready.then(() => ScrollTrigger.refresh());
      return () => mm.revert();
    },
    { scope: root, dependencies: [latest?.key] }
  );

  return (
    <div ref={root}>
      <section className="billboard">
        <Backdrop key={latest?.key || 'ambient'} latest={latest} />

        <div className="billboard-inner">
          {latest ? (
            <div key={`latest-${latest.key}`}>
              <span className="eyebrow billboard-rise">Last analyzed</span>
              <h1 className="billboard-title">{latest.title}</h1>
              <div className="billboard-meta billboard-rise">
                <span
                  className="tile-badge"
                  style={{
                    position: 'static',
                    background: ['unknown', 'mixed'].includes(latest.dominant) ? 'var(--text-soft)' : emotionColor(latest.dominant),
                  }}
                >
                  {latest.dominant === 'unknown' ? 'No evidence' : latest.dominant}
                </span>
                <span>{latest.segments} segments</span>
                <span>{latest.duration.toFixed(1)}s</span>
                <span>{latest.language}</span>
              </div>
              <EmotionStrip strip={latest.strip} className="billboard-strip billboard-rise" />
              <div className="billboard-rise" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                <button className="btn-play" onClick={() => onOpenHistory(latest)}>
                  <span aria-hidden="true">▶</span> Open results
                </button>
                <button className="btn-ghost" onClick={() => onNavigate('upload')}>
                  <span aria-hidden="true">+</span> New analysis
                </button>
              </div>
            </div>
          ) : (
            <div key="intro">
              <span className="eyebrow billboard-rise">Explainable Multimodal Emotion Recognition</span>
              <h1 className="billboard-title">
                The words said fine. <em>The face didn’t.</em>
              </h1>
              <p className="hero-lede billboard-rise">
                CinemaAI reads a film scene through dialogue, voice and facial expression, then shows how the emotion
                changes moment by moment and why each result was reached.
              </p>
              <div className="billboard-rise" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                <button className="btn-play" onClick={() => onNavigate('upload')}>
                  <span aria-hidden="true">▶</span> Analyze a video
                </button>
                <button className="btn-ghost" onClick={() => onNavigate('discovery')}>
                  Browse scenes
                </button>
              </div>
            </div>
          )}

          <SceneDemo />
        </div>
      </section>

      <div className="rows">
        {history.length > 0 && (
          <Row title="Recently analyzed">
            {history.map((item) => (
              <HistoryTile key={item.key} item={item} onOpen={onOpenHistory} />
            ))}
          </Row>
        )}

        <Row title="Seven emotions">
          {EMOTIONS.map((emotion, i) => (
            <div key={emotion.key} className="poster" style={{ background: emotionColor(emotion.key) }}>
              <span className="poster-index">0{i + 1}</span>
              <div className="poster-word">{emotion.name}</div>
              <p>{emotion.text}</p>
            </div>
          ))}
        </Row>

        <Row title="How it works">
          {STEPS.map((step, i) => (
            <div key={step.title} className="rank-tile">
              <span className="rank-number" aria-hidden="true">{i + 1}</span>
              <div className="rank-card">
                <h3>{step.title}</h3>
                <p>{step.text}</p>
              </div>
            </div>
          ))}
        </Row>

        {scenes.length > 0 && (
          <Row title="From the scene library">
            {scenes.map((scene) => (
              <SceneTile key={scene.video_id} scene={scene} onOpen={onOpenScene} />
            ))}
          </Row>
        )}

        <section className="cinema-card models-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '12px', flexWrap: 'wrap', marginBottom: '16px' }}>
            <h2 className="row-title">Models in use</h2>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Public pretrained models, not trained by this project. Accuracy per language has not been measured.
            </span>
          </div>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Part</th>
                  <th>Model</th>
                  <th>Covers</th>
                  <th>Known limit</th>
                </tr>
              </thead>
              <tbody>
                {MODELS.map((row) => (
                  <tr key={row.part}>
                    <td>{row.part}</td>
                    <td>{row.model}</td>
                    <td>{row.covers}</td>
                    <td>{row.limit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}

export { capitalize };
