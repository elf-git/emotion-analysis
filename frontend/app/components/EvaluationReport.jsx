'use client';

export default function EvaluationReport({ benchmarkData }) {
  const languageBenchmarks = [
    { lang: 'English (MELD Test)', samples: 2610, macroF1: '54.2%', weightedF1: '65.1%', topEmotion: 'Joy (72.4%)' },
    { lang: 'Hindi (Devanagari)', samples: 180, macroF1: '68.4%', weightedF1: '78.5%', topEmotion: 'Anger (84.1%)' },
    { lang: 'Malayalam (മലയാളം)', samples: 150, macroF1: '65.8%', weightedF1: '76.8%', topEmotion: 'Sadness (81.0%)' },
    { lang: 'Tamil (தமிழ்)', samples: 150, macroF1: '66.2%', weightedF1: '77.2%', topEmotion: 'Surprise (80.5%)' },
  ];

  // Normalized 7-class confusion matrix sample
  const emotions = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'];
  const confusionMatrix = [
    [0.72, 0.05, 0.02, 0.08, 0.06, 0.02, 0.05], // neutral
    [0.08, 0.69, 0.06, 0.02, 0.09, 0.01, 0.05], // surprise
    [0.05, 0.12, 0.58, 0.11, 0.02, 0.04, 0.08], // fear
    [0.10, 0.02, 0.04, 0.68, 0.02, 0.03, 0.11], // sadness
    [0.09, 0.08, 0.01, 0.02, 0.76, 0.01, 0.03], // joy
    [0.04, 0.02, 0.03, 0.06, 0.01, 0.62, 0.22], // disgust
    [0.06, 0.03, 0.03, 0.05, 0.02, 0.09, 0.72], // anger
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Unverified results warning */}
      <div className="cinema-card" style={{ padding: '18px 24px', borderColor: 'var(--danger)' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: '700', color: 'var(--danger)' }}>
          These numbers are unverified. Do not present them as results.
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '6px', lineHeight: 1.5 }}>
          Every figure on this page is a fixed value from the repository. They do not describe the pretrained models
          this app runs, and no trained weights exist to reproduce them. The only measured results saved in the repository
          are 1 to 2 epoch smoke tests, with test accuracy of 31.0% (text), 16.5% (audio) and 26.5% (video).
        </p>
      </div>

      {/* Header Banner */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 className="page-title">Evaluation</h1>
              <span className="metric-badge">
                MELD Official Splits (2,610 Utterances)
              </span>
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '4px' }}>
              Repository figures for unimodal baselines, late fusion and multilingual benchmarks. Unverified.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '20px' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Claimed Top F1 (Unverified)</div>
              <div className="font-mono" style={{ fontSize: '1.15rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                94.72% (T+V)
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Claimed Tri-Modal F1 (Unverified)</div>
              <div className="font-mono" style={{ fontSize: '1.15rem', fontWeight: '700', color: 'var(--text-secondary)' }}>
                87.05% (T+A+V)
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Modality Ablation Experiments Table */}
      <div className="cinema-card" style={{ padding: '22px' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '14px' }}>
          Modality Ablation Study (MELD Test Set)
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--line)', textAlign: 'left', color: 'var(--text-muted)' }}>
                <th style={{ padding: '8px 10px' }}>Configuration</th>
                <th style={{ padding: '8px 10px' }}>Channels</th>
                <th style={{ padding: '8px 10px' }}>Weights Scheme</th>
                <th style={{ padding: '8px 10px', textAlign: 'right' }}>Test Accuracy</th>
                <th style={{ padding: '8px 10px', textAlign: 'right' }}>Macro F1</th>
                <th style={{ padding: '8px 10px', textAlign: 'right' }}>Weighted F1</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkData.map((row, idx) => {
                const isTop = row.weighted_f1 >= 85.0;
                return (
                  <tr
                    key={idx}
                    style={{
                      borderBottom: '1px solid var(--surface-2)',
                      background: isTop ? 'var(--surface-2)' : 'transparent',
                    }}
                  >
                    <td style={{ padding: '8px 10px', fontWeight: isTop ? '600' : '400', color: isTop ? 'var(--text-primary)' : 'var(--text-soft)' }}>
                      {row.configuration}
                    </td>
                    <td style={{ padding: '8px 10px', color: 'var(--text-secondary)' }}>{row.modality}</td>
                    <td style={{ padding: '8px 10px', color: 'var(--text-muted)', fontSize: '0.8rem' }}>{row.weights}</td>
                    <td className="font-mono" style={{ padding: '8px 10px', textAlign: 'right', color: 'var(--text-secondary)' }}>{row.test_acc.toFixed(2)}%</td>
                    <td className="font-mono" style={{ padding: '8px 10px', textAlign: 'right', color: 'var(--text-secondary)' }}>{row.macro_f1.toFixed(2)}%</td>
                    <td className="font-mono" style={{ padding: '8px 10px', textAlign: 'right', fontWeight: '600', color: isTop ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                      {row.weighted_f1.toFixed(2)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Grid: Multilingual Performance Matrix + Confusion Matrix */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
        {/* Language-Wise Benchmark */}
        <div className="cinema-card" style={{ padding: '22px' }}>
          <h3 style={{ fontSize: '0.92rem', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '14px' }}>
            Multilingual Language-Wise Performance
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {languageBenchmarks.map((lb) => (
              <div
                key={lb.lang}
                style={{
                  background: 'var(--bg-inset)',
                  border: '1px solid var(--line)',
                  borderRadius: 'var(--radius-md)',
                  padding: '10px 14px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div>
                  <div style={{ fontWeight: '600', fontSize: '0.88rem', color: 'var(--text-primary)' }}>{lb.lang}</div>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                    {lb.samples} samples • Peak: {lb.topEmotion}
                  </div>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <div className="font-mono" style={{ fontWeight: '600', color: 'var(--text-primary)', fontSize: '0.9rem' }}>
                    {lb.weightedF1}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    Macro: {lb.macroF1}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 7-Class Normalized Confusion Matrix */}
        <div className="cinema-card" style={{ padding: '22px' }}>
          <h3 style={{ fontSize: '0.92rem', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '14px' }}>
            Normalized Emotion Confusion Matrix
          </h3>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.72rem' }}>
              <thead>
                <tr>
                  <th style={{ padding: '6px', textAlign: 'left', color: 'var(--text-muted)' }}>Pred ➔</th>
                  {emotions.map((e) => (
                    <th key={e} style={{ padding: '6px', textTransform: 'capitalize', color: 'var(--text-muted)', textAlign: 'center' }}>
                      {e.substring(0, 3)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {confusionMatrix.map((row, rIdx) => (
                  <tr key={rIdx}>
                    <td style={{ padding: '6px', fontWeight: '600', textTransform: 'capitalize', color: 'var(--text-secondary)' }}>
                      {emotions[rIdx].substring(0, 3)}
                    </td>
                    {row.map((val, cIdx) => {
                      const isDiag = rIdx === cIdx;
                      return (
                        <td
                          key={cIdx}
                          className="font-mono"
                          style={{
                            padding: '6px',
                            textAlign: 'center',
                            background: isDiag ? 'rgba(255, 255, 255, 0.12)' : 'transparent',
                            color: isDiag ? 'var(--text-primary)' : 'var(--text-muted)',
                            fontWeight: isDiag ? '600' : '400',
                            borderRadius: '2px',
                          }}
                        >
                          {(val * 100).toFixed(0)}%
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
