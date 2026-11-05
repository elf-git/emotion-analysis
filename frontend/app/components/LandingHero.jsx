'use client';

export default function LandingHero({ onNavigate }) {
  const capabilities = [
    {
      title: 'Tri-Modal Dynamic Gating',
      desc: 'Learns soft attention weights (α_T, α_A, α_V) dynamically per conversational turn to prioritize the most salient communicative modality.',
      tag: 'Fusion',
    },
    {
      title: 'Multilingual Indic NLP',
      desc: 'Dual-approach linguistic processing: native Dravidian/Indo-Aryan morphological stemming alongside IndicTrans2 neural translation.',
      tag: 'Linguistics',
    },
    {
      title: 'Visual Grad-CAM Heatmaps',
      desc: 'Spatial attention visualization overlaying ResNet-18 Conv4 activation layers onto facial muscle Action Units and gaze regions.',
      tag: 'Vision',
    },
    {
      title: 'Acoustic Prosody & Speech',
      desc: 'Faster-Whisper speech demuxing paired with openSMILE-aligned fundamental pitch (F_0) contours and RMS energy dynamics.',
      tag: 'Audio',
    },
    {
      title: 'Calibrated Uncertainty',
      desc: 'Information-theoretic Shannon entropy, Top-2 margin separation, and Wald 95% confidence intervals quantify model ambiguity.',
      tag: 'Metrics',
    },
    {
      title: 'Synchronized Cinema Timeline',
      desc: 'End-to-end full-scene chronological playback with turn-by-turn emotion state tracking and valence-arousal drift metrics.',
      tag: 'Chronology',
    },
  ];

  const languages = [
    { code: 'EN', name: 'English', script: 'Latin', benchmarkF1: '65.1%' },
    { code: 'HI', name: 'Hindi', script: 'Devanagari (हिन्दी)', benchmarkF1: '78.5%' },
    { code: 'ML', name: 'Malayalam', script: 'Malayalam (മലയാളം)', benchmarkF1: '76.8%' },
    { code: 'TA', name: 'Tamil', script: 'Tamil (தமிழ்)', benchmarkF1: '77.2%' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Hero Canvas */}
      <div
        className="cinema-card"
        style={{
          padding: '40px 32px',
          background: '#121214',
          position: 'relative',
        }}
      >
        <div style={{ maxWidth: '820px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
            <span className="metric-badge">
              Academic Research Workstation
            </span>
            <span style={{ fontSize: '0.8rem', color: '#71717a' }}>
              M.Sc. Artificial Intelligence & Machine Learning
            </span>
          </div>

          <h1
            style={{
              fontSize: '2.4rem',
              fontWeight: '700',
              lineHeight: 1.2,
              letterSpacing: '-0.02em',
              marginBottom: '14px',
              color: '#f4f4f5',
            }}
          >
            Explainable Multimodal Emotion & Communication Analysis
          </h1>

          <p
            style={{
              fontSize: '1rem',
              lineHeight: 1.6,
              color: '#a1a1aa',
              marginBottom: '24px',
            }}
          >
            A cinematic AI workstation evaluating conversational communication dynamics through tri-modal fusion: 
            {' '}<strong>Linguistic semantics</strong> (DistilBERT & IndicTrans2), 
            {' '}<strong>Acoustic prosody</strong> (Faster-Whisper & openSMILE), and 
            {' '}<strong>Facial expressions</strong> (ResNet-18 with Grad-CAM saliency).
          </p>

          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <button
              className="cinema-btn-primary"
              onClick={() => onNavigate('upload')}
            >
              Analyze Video Clip
            </button>
            <button
              className="cinema-btn-secondary"
              onClick={() => onNavigate('discovery')}
            >
              Discover Movie Scenes
            </button>
            <button
              className="cinema-btn-secondary"
              onClick={() => onNavigate('evaluation')}
            >
              View Ablation Benchmark
            </button>
          </div>
        </div>
      </div>

      {/* Core Architectural Pillars */}
      <div>
        <div style={{ marginBottom: '14px' }}>
          <h2 style={{ fontSize: '1.15rem', fontWeight: '700', letterSpacing: '-0.01em', color: '#f4f4f5' }}>
            System Architecture & Explainable Pillars
          </h2>
          <p style={{ color: '#71717a', fontSize: '0.84rem' }}>
            Modular deep learning pipelines ensuring transparent attribution from raw pixels and waveforms to conversational emotion states.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '14px' }}>
          {capabilities.map((cap, i) => (
            <div key={i} className="cinema-card" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <h3 style={{ fontSize: '0.95rem', fontWeight: '600', color: '#f4f4f5' }}>
                  {cap.title}
                </h3>
                <span className="metric-badge" style={{ fontSize: '0.68rem' }}>
                  {cap.tag}
                </span>
              </div>
              <p style={{ fontSize: '0.82rem', color: '#a1a1aa', lineHeight: 1.5 }}>
                {cap.desc}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Supported Languages Matrix */}
      <div className="cinema-card" style={{ padding: '22px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: '700', color: '#f4f4f5' }}>
              Multilingual Corpus Support (Indic & English)
            </h3>
            <p style={{ fontSize: '0.8rem', color: '#71717a' }}>
              Evaluated on Dravidian and Indo-Aryan dialects with IndicTrans2 alignment.
            </p>
          </div>
          <span className="metric-badge">
            IndicTrans2 + Faster-Whisper
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
          {languages.map((l) => (
            <div
              key={l.code}
              style={{
                background: '#18181b',
                border: '1px solid #27272a',
                borderRadius: 'var(--radius-md)',
                padding: '14px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span className="metric-badge" style={{ padding: '2px 6px' }}>
                  {l.code}
                </span>
                <span className="font-mono" style={{ fontSize: '0.8rem', color: '#f4f4f5', fontWeight: '600' }}>
                  F1: {l.benchmarkF1}
                </span>
              </div>
              <div style={{ fontWeight: '600', fontSize: '0.92rem', color: '#f4f4f5' }}>{l.name}</div>
              <div className="font-indic" style={{ fontSize: '0.8rem', color: '#71717a', marginTop: '2px' }}>
                {l.script}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
