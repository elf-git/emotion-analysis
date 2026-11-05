'use client';

export default function Navigation({ activeScreen, setActiveScreen, apiHealth }) {
  const screens = [
    { id: 'landing', label: 'Overview' },
    { id: 'discovery', label: 'Scene Discovery' },
    { id: 'upload', label: 'Video Upload' },
    { id: 'dashboard', label: 'Scene Dashboard' },
    { id: 'segment', label: 'Segment Inspector' },
    { id: 'explainability', label: 'Explainability & CAM' },
    { id: 'transcript', label: 'Multilingual Script' },
    { id: 'evaluation', label: 'Evaluation & Ablation' },
  ];

  return (
    <header style={{ marginBottom: '24px' }}>
      {/* Top Telemetry & Status Bar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          paddingBottom: '16px',
          borderBottom: '1px solid #27272a',
          marginBottom: '16px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: 'var(--radius-sm)',
              background: '#18181b',
              border: '1px solid #27272a',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '0.95rem',
              color: '#f4f4f5',
              fontWeight: '700',
            }}
          >
            M
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontWeight: '700', fontSize: '1rem', letterSpacing: '-0.02em', color: '#f4f4f5' }}>
                MELD CinemaAI
              </span>
              <span className="metric-badge">
                M.Sc. AI & ML Thesis
              </span>
            </div>
            <div style={{ fontSize: '0.76rem', color: '#71717a' }}>
              Multimodal Emotion Recognition & Communication Analysis
            </div>
          </div>
        </div>

        {/* Backend & Modality Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span className="metric-badge">
            DistilBERT + IndicTrans2
          </span>
          <span className="metric-badge">
            Groq Whisper Large V3 + openSMILE
          </span>
          <span className="metric-badge">
            ResNet-18 Grad-CAM
          </span>
          <span
            className="metric-badge"
            style={{
              color: apiHealth.status === 'healthy' ? '#e4e4e7' : '#a1a1aa',
            }}
          >
            ● {apiHealth.status === 'healthy' ? 'Neural Engine Active' : 'Connecting API...'}
          </span>
        </div>
      </div>

      {/* Main Studio Navigation Tab Bar */}
      <nav className="studio-nav">
        {screens.map((s) => (
          <button
            key={s.id}
            className={`studio-nav-btn ${activeScreen === s.id ? 'active' : ''}`}
            onClick={() => setActiveScreen(s.id)}
          >
            <span>{s.label}</span>
          </button>
        ))}
      </nav>
    </header>
  );
}
