import "./globals.css";

export const metadata = {
  title: "Explainable Multimodal Emotion Recognition | MELD AI & IndicMultilingual",
  description: "Advanced Multimodal Emotion Recognition with Cross-Modal Attention, Modality Gating, SerpApi Movie Scene Discovery, and Indic Multilingual Full-Clip Analysis.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Noto+Sans+Devanagari:wght@400;600;700&family=Noto+Sans+Malayalam:wght@400;600;700&family=Noto+Sans+Tamil:wght@400;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
