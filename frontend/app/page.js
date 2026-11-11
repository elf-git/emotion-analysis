'use client';

import { useState, useEffect, useRef } from 'react';
import { MOTION_OK, ScrollTrigger, gsap, useGSAP } from './lib/gsap';
import Navigation from './components/Navigation';
import LandingHero from './components/LandingHero';
import SceneDiscovery from './components/SceneDiscovery';
import VideoUploader from './components/VideoUploader';
import SceneDashboard from './components/SceneDashboard';
import SegmentInspector from './components/SegmentInspector';
import ExplainabilitySuite from './components/ExplainabilitySuite';
import MultilingualTranscript from './components/MultilingualTranscript';
import EvaluationReport from './components/EvaluationReport';

const HISTORY_KEY = 'cinemaai.history.v1';
const MAX_HISTORY = 12;

// Keeps only what the home page cards need from a full analysis
function summarizeAnalysis(data, extra) {
  const segments = data.segments || [];
  return {
    ...extra,
    dominant: data.scene_summary?.dominant_emotion || 'unknown',
    strip: segments.map((seg) => [seg.fused_prediction.predicted_emotion, seg.end_time_sec - seg.start_time_sec]),
    segments: segments.length,
    duration: data.scene_metadata?.duration_sec || 0,
    language: data.scene_metadata?.detected_language || '',
    date: new Date().toISOString(),
  };
}

// Grabs one small frame from an uploaded video to use as its card image
function captureFrame(url) {
  return new Promise((resolve) => {
    if (!url) return resolve(null);
    const video = document.createElement('video');
    const done = (value) => {
      video.removeAttribute('src');
      resolve(value);
    };
    video.muted = true;
    video.preload = 'auto';
    video.src = url;
    video.addEventListener('loadeddata', () => {
      video.currentTime = Math.min(1, (video.duration || 2) / 2);
    });
    video.addEventListener('seeked', () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = 480;
        canvas.height = Math.round((480 * video.videoHeight) / (video.videoWidth || 480)) || 270;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        done(canvas.toDataURL('image/jpeg', 0.72));
      } catch {
        done(null);
      }
    });
    video.addEventListener('error', () => done(null));
    setTimeout(() => done(null), 4000);
  });
}

// Hand-written sample scenes. These are scripted examples, not model output.
const CURATED_SCENE_ANALYSES = {
  ml: {
    status: 'success',
    scene_metadata: {
      filename: 'kumbalangi_nights_dinner_table_confrontation.mp4',
      duration_sec: 14.5,
      fps: 25.0,
      total_frames_sampled: 15,
      audio_extracted: true,
      detected_language: 'Malayalam',
      language_code: 'ml',
    },
    scene_summary: {
      dominant_emotion: 'anger',
      dominant_emotion_color: '#f87171',
      total_segments: 4,
      average_valence: -0.68,
      average_arousal: 0.82,
      emotion_transitions: ['neutral', 'surprise', 'anger', 'anger'],
      emotion_distribution: {
        anger: { count: 2, percentage: 50.0, color: '#f87171' },
        surprise: { count: 1, percentage: 25.0, color: '#38bdf8' },
        neutral: { count: 1, percentage: 25.0, color: '#94a3b8' },
        joy: { count: 0, percentage: 0.0, color: '#fbbf24' },
        sadness: { count: 0, percentage: 0.0, color: '#60a5fa' },
        fear: { count: 0, percentage: 0.0, color: '#c084fc' },
        disgust: { count: 0, percentage: 0.0, color: '#34d399' },
      },
    },
    segments: [
      {
        segment_index: 0,
        start_time_sec: 0.0,
        end_time_sec: 3.5,
        timestamp_label: '00:00 - 00:03.5',
        transcript: {
          original_text: 'ഇവിടെ എന്താണ് നടക്കുന്നത്?',
          language: 'Malayalam',
          language_code: 'ml',
          translated_text: 'What is happening here?',
        },
        fused_prediction: {
          predicted_emotion: 'neutral',
          emotion_color: '#94a3b8',
          confidence_pct: 82.4,
          valence: 0.05,
          arousal: 0.2,
        },
        modality_contributions: {
          text_pct: 42.0,
          audio_pct: 28.0,
          video_pct: 30.0,
          dominant_modality: 'Linguistic (Text)',
        },
        modality_predictions: {
          text: {
            predicted_emotion: 'neutral',
            confidence_pct: 79.5,
            approach_a_multilingual: 'neutral',
            approach_b_translated: 'neutral',
          },
          audio: {
            predicted_emotion: 'neutral',
            confidence_pct: 74.0,
            pitch_hz: 142.5,
            rms_energy: 0.038,
          },
          visual: {
            predicted_emotion: 'neutral',
            confidence_pct: 85.0,
            gradcam_available: true,
            gradcam_url: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"><rect width="120" height="120" fill="%231e293b"/><circle cx="60" cy="50" r="28" fill="%2394a3b8" opacity="0.6"/><text x="60" y="66" font-size="12" fill="%23ffffff" text-anchor="middle">Neutral Face</text></svg>',
          },
        },
        explanation: 'Restrained conversational inquiry with baseline acoustic pitch.',
      },
      {
        segment_index: 1,
        start_time_sec: 3.5,
        end_time_sec: 7.0,
        timestamp_label: '00:03.5 - 00:07.0',
        transcript: {
          original_text: 'നിങ്ങൾ എന്നോട് ചോദിക്കാതെ ഇത് ചെയ്തോ?!',
          language: 'Malayalam',
          language_code: 'ml',
          translated_text: 'You did this without asking me?!',
        },
        fused_prediction: {
          predicted_emotion: 'surprise',
          emotion_color: '#38bdf8',
          confidence_pct: 88.6,
          valence: -0.3,
          arousal: 0.75,
        },
        modality_contributions: {
          text_pct: 35.0,
          audio_pct: 45.0,
          video_pct: 20.0,
          dominant_modality: 'Acoustic (Speech Prosody)',
        },
        modality_predictions: {
          text: {
            predicted_emotion: 'surprise',
            confidence_pct: 84.0,
            approach_a_multilingual: 'surprise',
            approach_b_translated: 'surprise',
          },
          audio: {
            predicted_emotion: 'surprise',
            confidence_pct: 91.2,
            pitch_hz: 248.0,
            rms_energy: 0.089,
          },
          visual: {
            predicted_emotion: 'surprise',
            confidence_pct: 82.5,
            gradcam_available: true,
            gradcam_url: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"><rect width="120" height="120" fill="%231e293b"/><circle cx="60" cy="50" r="32" fill="%2338bdf8" opacity="0.7"/><circle cx="50" cy="45" r="5" fill="%23ffffff"/><circle cx="70" cy="45" r="5" fill="%23ffffff"/><ellipse cx="60" cy="65" rx="10" ry="14" fill="%23ffffff"/><text x="60" y="105" font-size="10" fill="%2338bdf8" text-anchor="middle">Grad-CAM AU1+2</text></svg>',
          },
        },
        explanation: 'Sharp pitch surge (248 Hz) with open mouth and raised brow activations.',
      },
      {
        segment_index: 2,
        start_time_sec: 7.0,
        end_time_sec: 11.0,
        timestamp_label: '00:07.0 - 00:11.0',
        transcript: {
          original_text: 'ഇത് എൻ്റെ വീടാണ്, എൻ്റെ തീരുമാനമാണ്!',
          language: 'Malayalam',
          language_code: 'ml',
          translated_text: 'This is my house, this is my decision!',
        },
        fused_prediction: {
          predicted_emotion: 'anger',
          emotion_color: '#f87171',
          confidence_pct: 94.3,
          valence: -0.85,
          arousal: 0.92,
        },
        modality_contributions: {
          text_pct: 38.0,
          audio_pct: 32.0,
          video_pct: 30.0,
          dominant_modality: 'Linguistic (Text)',
        },
        modality_predictions: {
          text: {
            predicted_emotion: 'anger',
            confidence_pct: 95.0,
            approach_a_multilingual: 'anger',
            approach_b_translated: 'anger',
          },
          audio: {
            predicted_emotion: 'anger',
            confidence_pct: 91.5,
            pitch_hz: 232.0,
            rms_energy: 0.115,
          },
          visual: {
            predicted_emotion: 'anger',
            confidence_pct: 93.0,
            gradcam_available: true,
            gradcam_url: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"><rect width="120" height="120" fill="%231e293b"/><circle cx="60" cy="50" r="30" fill="%23f87171" opacity="0.8"/><line x1="45" y1="40" x2="58" y2="48" stroke="%23ffffff" stroke-width="3"/><line x1="75" y1="40" x2="62" y2="48" stroke="%23ffffff" stroke-width="3"/><ellipse cx="60" cy="70" rx="14" ry="6" fill="%23ffffff"/><text x="60" y="105" font-size="10" fill="%23f87171" text-anchor="middle">Grad-CAM AU4+24</text></svg>',
          },
        },
        explanation: 'Aggressive brow furrowing (AU4), elevated voice energy, and confrontational dialogue.',
      },
      {
        segment_index: 3,
        start_time_sec: 11.0,
        end_time_sec: 14.5,
        timestamp_label: '00:11.0 - 00:14.5',
        transcript: {
          original_text: 'ഇനി ആരും ഒന്നും മിണ്ടിപ്പോകരുത്!',
          language: 'Malayalam',
          language_code: 'ml',
          translated_text: 'Nobody utter another word!',
        },
        fused_prediction: {
          predicted_emotion: 'anger',
          emotion_color: '#f87171',
          confidence_pct: 91.8,
          valence: -0.9,
          arousal: 0.88,
        },
        modality_contributions: {
          text_pct: 36.0,
          audio_pct: 34.0,
          video_pct: 30.0,
          dominant_modality: 'Linguistic (Text)',
        },
        modality_predictions: {
          text: {
            predicted_emotion: 'anger',
            confidence_pct: 93.4,
            approach_a_multilingual: 'anger',
            approach_b_translated: 'anger',
          },
          audio: {
            predicted_emotion: 'anger',
            confidence_pct: 88.0,
            pitch_hz: 215.0,
            rms_energy: 0.095,
          },
          visual: {
            predicted_emotion: 'anger',
            confidence_pct: 90.5,
            gradcam_available: true,
            gradcam_url: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"><rect width="120" height="120" fill="%231e293b"/><circle cx="60" cy="50" r="30" fill="%23f87171" opacity="0.7"/><ellipse cx="60" cy="68" rx="12" ry="5" fill="%23ffffff"/><text x="60" y="105" font-size="10" fill="%23f87171" text-anchor="middle">Grad-CAM Jaw Clench</text></svg>',
          },
        },
        explanation: 'Authoritative ultimatum with sustained acoustic energy and clenching facial activation.',
      },
    ],
  },

  hi: {
    status: 'success',
    scene_metadata: {
      filename: '3_idiots_hospital_awakening_scene.mp4',
      duration_sec: 15.0,
      fps: 25.0,
      total_frames_sampled: 16,
      audio_extracted: true,
      detected_language: 'Hindi',
      language_code: 'hi',
    },
    scene_summary: {
      dominant_emotion: 'joy',
      dominant_emotion_color: '#fbbf24',
      total_segments: 4,
      average_valence: 0.74,
      average_arousal: 0.85,
      emotion_transitions: ['neutral', 'surprise', 'joy', 'joy'],
      emotion_distribution: {
        joy: { count: 2, percentage: 50.0, color: '#fbbf24' },
        surprise: { count: 1, percentage: 25.0, color: '#38bdf8' },
        neutral: { count: 1, percentage: 25.0, color: '#94a3b8' },
        anger: { count: 0, percentage: 0.0, color: '#f87171' },
        sadness: { count: 0, percentage: 0.0, color: '#60a5fa' },
        fear: { count: 0, percentage: 0.0, color: '#c084fc' },
        disgust: { count: 0, percentage: 0.0, color: '#34d399' },
      },
    },
    segments: [
      {
        segment_index: 0,
        start_time_sec: 0.0,
        end_time_sec: 3.8,
        timestamp_label: '00:00 - 00:03.8',
        transcript: {
          original_text: 'राजू, क्या तुम मुझे सुन पा रहे हो?',
          language: 'Hindi',
          language_code: 'hi',
          translated_text: 'Raju, can you hear me?',
        },
        fused_prediction: {
          predicted_emotion: 'neutral',
          emotion_color: '#94a3b8',
          confidence_pct: 84.1,
          valence: 0.1,
          arousal: 0.3,
        },
        modality_contributions: { text_pct: 45.0, audio_pct: 25.0, video_pct: 30.0, dominant_modality: 'Linguistic (Text)' },
        modality_predictions: {
          text: { predicted_emotion: 'neutral', confidence_pct: 85.0, approach_a_multilingual: 'neutral', approach_b_translated: 'neutral' },
          audio: { predicted_emotion: 'neutral', confidence_pct: 78.0, pitch_hz: 155.0, rms_energy: 0.042 },
          visual: { predicted_emotion: 'neutral', confidence_pct: 86.0, gradcam_available: true },
        },
        explanation: 'Low arousal conversational inquiry at bedside.',
      },
      {
        segment_index: 1,
        start_time_sec: 3.8,
        end_time_sec: 7.5,
        timestamp_label: '00:03.8 - 00:07.5',
        transcript: {
          original_text: 'अरे देखो! उसकी उँगलियाँ हिलीं!',
          language: 'Hindi',
          language_code: 'hi',
          translated_text: 'Hey look! His fingers just moved!',
        },
        fused_prediction: {
          predicted_emotion: 'surprise',
          emotion_color: '#38bdf8',
          confidence_pct: 92.4,
          valence: 0.5,
          arousal: 0.88,
        },
        modality_contributions: { text_pct: 30.0, audio_pct: 45.0, video_pct: 25.0, dominant_modality: 'Acoustic (Audio)' },
        modality_predictions: {
          text: { predicted_emotion: 'surprise', confidence_pct: 90.0, approach_a_multilingual: 'surprise', approach_b_translated: 'surprise' },
          audio: { predicted_emotion: 'surprise', confidence_pct: 94.0, pitch_hz: 275.0, rms_energy: 0.098 },
          visual: { predicted_emotion: 'surprise', confidence_pct: 89.0, gradcam_available: true },
        },
        explanation: 'Vocal pitch surge and wide open eye action units upon sudden finger movement.',
      },
      {
        segment_index: 2,
        start_time_sec: 7.5,
        end_time_sec: 11.2,
        timestamp_label: '00:07.5 - 00:11.2',
        transcript: {
          original_text: 'राजू भाई होश में आ गया! ऑल इज़ वैल!',
          language: 'Hindi',
          language_code: 'hi',
          translated_text: 'Raju is awake! All is well!',
        },
        fused_prediction: {
          predicted_emotion: 'joy',
          emotion_color: '#fbbf24',
          confidence_pct: 96.0,
          valence: 0.95,
          arousal: 0.92,
        },
        modality_contributions: { text_pct: 35.0, audio_pct: 35.0, video_pct: 30.0, dominant_modality: 'Linguistic + Acoustic' },
        modality_predictions: {
          text: { predicted_emotion: 'joy', confidence_pct: 97.0, approach_a_multilingual: 'joy', approach_b_translated: 'joy' },
          audio: { predicted_emotion: 'joy', confidence_pct: 95.0, pitch_hz: 260.0, rms_energy: 0.11 },
          visual: { predicted_emotion: 'joy', confidence_pct: 96.0, gradcam_available: true },
        },
        explanation: 'Euphoric celebration with broad smiles, laughing acoustics, and iconic dialogue.',
      },
      {
        segment_index: 3,
        start_time_sec: 11.2,
        end_time_sec: 15.0,
        timestamp_label: '00:11.2 - 00:15.0',
        transcript: {
          original_text: 'हम सब हमेशा साथ रहेंगे, तू चिंता मत कर!',
          language: 'Hindi',
          language_code: 'hi',
          translated_text: 'We will always be together, do not worry!',
        },
        fused_prediction: {
          predicted_emotion: 'joy',
          emotion_color: '#fbbf24',
          confidence_pct: 93.5,
          valence: 0.88,
          arousal: 0.78,
        },
        modality_contributions: { text_pct: 40.0, audio_pct: 30.0, video_pct: 30.0, dominant_modality: 'Linguistic (Text)' },
        modality_predictions: {
          text: { predicted_emotion: 'joy', confidence_pct: 94.0, approach_a_multilingual: 'joy', approach_b_translated: 'joy' },
          audio: { predicted_emotion: 'joy', confidence_pct: 91.0, pitch_hz: 220.0, rms_energy: 0.08 },
          visual: { predicted_emotion: 'joy', confidence_pct: 93.0, gradcam_available: true },
        },
        explanation: 'Warm communal reassurance and high positive valence.',
      },
    ],
  },

  ta: {
    status: 'success',
    scene_metadata: {
      filename: 'super_deluxe_emotional_revelation.mp4',
      duration_sec: 14.0,
      fps: 25.0,
      total_frames_sampled: 14,
      audio_extracted: true,
      detected_language: 'Tamil',
      language_code: 'ta',
    },
    scene_summary: {
      dominant_emotion: 'sadness',
      dominant_emotion_color: '#60a5fa',
      total_segments: 4,
      average_valence: -0.62,
      average_arousal: 0.72,
      emotion_transitions: ['neutral', 'sadness', 'surprise', 'sadness'],
      emotion_distribution: {
        sadness: { count: 2, percentage: 50.0, color: '#60a5fa' },
        surprise: { count: 1, percentage: 25.0, color: '#38bdf8' },
        neutral: { count: 1, percentage: 25.0, color: '#94a3b8' },
        joy: { count: 0, percentage: 0.0, color: '#fbbf24' },
        anger: { count: 0, percentage: 0.0, color: '#f87171' },
        fear: { count: 0, percentage: 0.0, color: '#c084fc' },
        disgust: { count: 0, percentage: 0.0, color: '#34d399' },
      },
    },
    segments: [
      {
        segment_index: 0,
        start_time_sec: 0.0,
        end_time_sec: 3.5,
        timestamp_label: '00:00 - 00:03.5',
        transcript: {
          original_text: 'என்னை உனக்கு அடையாளம் தெரிகிறதா?',
          language: 'Tamil',
          language_code: 'ta',
          translated_text: 'Do you recognize who I am?',
        },
        fused_prediction: { predicted_emotion: 'neutral', emotion_color: '#94a3b8', confidence_pct: 81.0, valence: -0.1, arousal: 0.3 },
        modality_contributions: { text_pct: 40.0, audio_pct: 30.0, video_pct: 30.0, dominant_modality: 'Linguistic (Text)' },
        modality_predictions: {
          text: { predicted_emotion: 'neutral', confidence_pct: 82.0, approach_a_multilingual: 'neutral', approach_b_translated: 'neutral' },
          audio: { predicted_emotion: 'neutral', confidence_pct: 79.0, pitch_hz: 148.0, rms_energy: 0.04 },
          visual: { predicted_emotion: 'neutral', confidence_pct: 82.0, gradcam_available: true },
        },
        explanation: 'Hesitant emotional opening with controlled pitch.',
      },
      {
        segment_index: 1,
        start_time_sec: 3.5,
        end_time_sec: 7.0,
        timestamp_label: '00:03.5 - 00:07.0',
        transcript: {
          original_text: 'இத்தனை வருடங்களாக நான் பட்ட வலி உனக்கு புரியாது...',
          language: 'Tamil',
          language_code: 'ta',
          translated_text: 'You cannot comprehend the pain I endured all these years...',
        },
        fused_prediction: { predicted_emotion: 'sadness', emotion_color: '#60a5fa', confidence_pct: 91.5, valence: -0.75, arousal: 0.65 },
        modality_contributions: { text_pct: 38.0, audio_pct: 34.0, video_pct: 28.0, dominant_modality: 'Linguistic (Text)' },
        modality_predictions: {
          text: { predicted_emotion: 'sadness', confidence_pct: 93.0, approach_a_multilingual: 'sadness', approach_b_translated: 'sadness' },
          audio: { predicted_emotion: 'sadness', confidence_pct: 90.0, pitch_hz: 135.0, rms_energy: 0.035 },
          visual: { predicted_emotion: 'sadness', confidence_pct: 91.0, gradcam_available: true },
        },
        explanation: 'Deep vocal tremor and downcast gaze conveying profound grief.',
      },
      {
        segment_index: 2,
        start_time_sec: 7.0,
        end_time_sec: 10.5,
        timestamp_label: '00:07.0 - 00:10.5',
        transcript: {
          original_text: 'நீ எப்படி இந்த நிலைக்கு மாறினாய்?!',
          language: 'Tamil',
          language_code: 'ta',
          translated_text: 'How did you change into this condition?!',
        },
        fused_prediction: { predicted_emotion: 'surprise', emotion_color: '#38bdf8', confidence_pct: 88.0, valence: -0.4, arousal: 0.82 },
        modality_contributions: { text_pct: 30.0, audio_pct: 42.0, video_pct: 28.0, dominant_modality: 'Acoustic (Audio)' },
        modality_predictions: {
          text: { predicted_emotion: 'surprise', confidence_pct: 86.0, approach_a_multilingual: 'surprise', approach_b_translated: 'surprise' },
          audio: { predicted_emotion: 'surprise', confidence_pct: 91.0, pitch_hz: 242.0, rms_energy: 0.085 },
          visual: { predicted_emotion: 'surprise', confidence_pct: 85.0, gradcam_available: true },
        },
        explanation: 'Involuntary shock and pitch jump facing transformation revelation.',
      },
      {
        segment_index: 3,
        start_time_sec: 10.5,
        end_time_sec: 14.0,
        timestamp_label: '00:10.5 - 00:14.0',
        transcript: {
          original_text: 'ஆனால் நீ இன்னும் என் தாய் தான்...',
          language: 'Tamil',
          language_code: 'ta',
          translated_text: 'Yet you are still my mother...',
        },
        fused_prediction: { predicted_emotion: 'sadness', emotion_color: '#60a5fa', confidence_pct: 90.0, valence: -0.5, arousal: 0.55 },
        modality_contributions: { text_pct: 42.0, audio_pct: 28.0, video_pct: 30.0, dominant_modality: 'Linguistic (Text)' },
        modality_predictions: {
          text: { predicted_emotion: 'sadness', confidence_pct: 92.0, approach_a_multilingual: 'sadness', approach_b_translated: 'sadness' },
          audio: { predicted_emotion: 'sadness', confidence_pct: 88.0, pitch_hz: 140.0, rms_energy: 0.038 },
          visual: { predicted_emotion: 'sadness', confidence_pct: 89.0, gradcam_available: true },
        },
        explanation: 'Subdued emotional acceptance with tearful ocular activation.',
      },
    ],
  },

  en: {
    status: 'success',
    scene_metadata: {
      filename: 'interstellar_cooper_23_years_messages.mp4',
      duration_sec: 16.0,
      fps: 24.0,
      total_frames_sampled: 16,
      audio_extracted: true,
      detected_language: 'English',
      language_code: 'en',
    },
    scene_summary: {
      dominant_emotion: 'sadness',
      dominant_emotion_color: '#60a5fa',
      total_segments: 4,
      average_valence: -0.78,
      average_arousal: 0.84,
      emotion_transitions: ['neutral', 'sadness', 'sadness', 'sadness'],
      emotion_distribution: {
        sadness: { count: 3, percentage: 75.0, color: '#60a5fa' },
        neutral: { count: 1, percentage: 25.0, color: '#94a3b8' },
        joy: { count: 0, percentage: 0.0, color: '#fbbf24' },
        surprise: { count: 0, percentage: 0.0, color: '#38bdf8' },
        anger: { count: 0, percentage: 0.0, color: '#f87171' },
        fear: { count: 0, percentage: 0.0, color: '#c084fc' },
        disgust: { count: 0, percentage: 0.0, color: '#34d399' },
      },
    },
    segments: [
      {
        segment_index: 0,
        start_time_sec: 0.0,
        end_time_sec: 4.0,
        timestamp_label: '00:00 - 00:04.0',
        transcript: {
          original_text: 'Playback transmission: Message log entry 1...',
          language: 'English',
          language_code: 'en',
          translated_text: 'Playback transmission: Message log entry 1...',
        },
        fused_prediction: { predicted_emotion: 'neutral', emotion_color: '#94a3b8', confidence_pct: 88.0, valence: 0.0, arousal: 0.2 },
        modality_contributions: { text_pct: 50.0, audio_pct: 20.0, video_pct: 30.0, dominant_modality: 'Linguistic (Text)' },
        modality_predictions: {
          text: { predicted_emotion: 'neutral', confidence_pct: 90.0, approach_a_multilingual: 'neutral', approach_b_translated: 'neutral' },
          audio: { predicted_emotion: 'neutral', confidence_pct: 85.0, pitch_hz: 120.0, rms_energy: 0.02 },
          visual: { predicted_emotion: 'neutral', confidence_pct: 88.0, gradcam_available: true },
        },
        explanation: 'Robotic automated log prompt with baseline subject monitoring.',
      },
      {
        segment_index: 1,
        start_time_sec: 4.0,
        end_time_sec: 8.0,
        timestamp_label: '00:04.0 - 00:08.0',
        transcript: {
          original_text: 'Dad... today is my birthday. And it is a special one...',
          language: 'English',
          language_code: 'en',
          translated_text: 'Dad... today is my birthday. And it is a special one...',
        },
        fused_prediction: { predicted_emotion: 'sadness', emotion_color: '#60a5fa', confidence_pct: 92.5, valence: -0.7, arousal: 0.75 },
        modality_contributions: { text_pct: 32.0, audio_pct: 34.0, video_pct: 34.0, dominant_modality: 'Visual Facial Saliency' },
        modality_predictions: {
          text: { predicted_emotion: 'sadness', confidence_pct: 88.0, approach_a_multilingual: 'sadness', approach_b_translated: 'sadness' },
          audio: { predicted_emotion: 'sadness', confidence_pct: 92.0, pitch_hz: 165.0, rms_energy: 0.06 },
          visual: { predicted_emotion: 'sadness', confidence_pct: 96.0, gradcam_available: true },
        },
        explanation: 'Cooper begins trembling as realization of lost decades hits.',
      },
      {
        segment_index: 2,
        start_time_sec: 8.0,
        end_time_sec: 12.0,
        timestamp_label: '00:08.0 - 00:12.0',
        transcript: {
          original_text: 'Because today, I am the same age you were when you left.',
          language: 'English',
          language_code: 'en',
          translated_text: 'Because today, I am the same age you were when you left.',
        },
        fused_prediction: { predicted_emotion: 'sadness', emotion_color: '#60a5fa', confidence_pct: 97.2, valence: -0.92, arousal: 0.94 },
        modality_contributions: { text_pct: 30.0, audio_pct: 35.0, video_pct: 35.0, dominant_modality: 'Visual + Acoustic' },
        modality_predictions: {
          text: { predicted_emotion: 'sadness', confidence_pct: 95.0, approach_a_multilingual: 'sadness', approach_b_translated: 'sadness' },
          audio: { predicted_emotion: 'sadness', confidence_pct: 96.0, pitch_hz: 180.0, rms_energy: 0.09 },
          visual: { predicted_emotion: 'sadness', confidence_pct: 98.0, gradcam_available: true },
        },
        explanation: 'Extreme facial muscle breakdown (AU1, AU4, AU15) with open weeping and high grief arousal.',
      },
      {
        segment_index: 3,
        start_time_sec: 12.0,
        end_time_sec: 16.0,
        timestamp_label: '00:12.0 - 00:16.0',
        transcript: {
          original_text: 'So it would be a real good time for you to come back now.',
          language: 'English',
          language_code: 'en',
          translated_text: 'So it would be a real good time for you to come back now.',
        },
        fused_prediction: { predicted_emotion: 'sadness', emotion_color: '#60a5fa', confidence_pct: 95.8, valence: -0.88, arousal: 0.86 },
        modality_contributions: { text_pct: 35.0, audio_pct: 32.0, video_pct: 33.0, dominant_modality: 'Linguistic (Text)' },
        modality_predictions: {
          text: { predicted_emotion: 'sadness', confidence_pct: 96.0, approach_a_multilingual: 'sadness', approach_b_translated: 'sadness' },
          audio: { predicted_emotion: 'sadness', confidence_pct: 94.0, pitch_hz: 155.0, rms_energy: 0.07 },
          visual: { predicted_emotion: 'sadness', confidence_pct: 96.0, gradcam_available: true },
        },
        explanation: 'Heartbreaking plea under sustained ocular tears and vocal sob catch.',
      },
    ],
  },
};

export default function Home() {
  const [activeScreen, setActiveScreen] = useState('landing');
  const [apiHealth, setApiHealth] = useState({ status: 'connecting', models_loaded: false });
  const [errorMessage, setErrorMessage] = useState(null);

  // Scene Discovery State
  const [sceneResults, setSceneResults] = useState([]);
  const [searchMeta, setSearchMeta] = useState(null);
  const [isSearching, setIsSearching] = useState(false);

  // Video Upload & Pipeline Configuration State
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [selectedScene, setSelectedScene] = useState(null);
  const [selectedYouTubeUrl, setSelectedYouTubeUrl] = useState('');
  const [samplingInterval, setSamplingInterval] = useState('1.0');
  const [segmentDuration, setSegmentDuration] = useState('3.0');
  const [language, setLanguage] = useState('auto');
  const [translate, setTranslate] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisStage, setAnalysisStage] = useState('');
  const [analyzingVideoId, setAnalyzingVideoId] = useState(null);

  // Full-Clip Scene Analysis Result & Interactive Playback
  const [clipResult, setClipResult] = useState(null);
  const [activeSegmentIndex, setActiveSegmentIndex] = useState(0);
  const [currentPlaybackTime, setCurrentPlaybackTime] = useState(0);
  const videoPlayerRef = useRef(null);

  // Benchmark Data
  const [benchmarkData, setBenchmarkData] = useState([]);
  const mainRef = useRef(null);
  const [history, setHistory] = useState([]);
  const [featuredScenes, setFeaturedScenes] = useState([]);

  // Recent analyses are kept in this browser only
  useEffect(() => {
    try {
      const saved = JSON.parse(window.localStorage.getItem(HISTORY_KEY) || '[]');
      // Browser storage only exists after mount, so this cannot be the initial state
      // eslint-disable-next-line react-hooks/set-state-in-effect
      if (Array.isArray(saved)) setHistory(saved);
    } catch {
      // Storage can be unavailable, for example in private windows
    }
  }, []);

  const rememberAnalysis = (item) => {
    setHistory((current) => {
      const next = [item, ...current.filter((entry) => entry.key !== item.key)].slice(0, MAX_HISTORY);
      try {
        window.localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      } catch {
        // Keep the in-memory list even if it cannot be saved
      }
      return next;
    });
  };
  const isFirstScreen = useRef(true);

  // Screen change transition
  useGSAP(
    () => {
      window.scrollTo(0, 0);
      if (isFirstScreen.current) {
        isFirstScreen.current = false;
        return undefined;
      }
      const mm = gsap.matchMedia();
      mm.add(MOTION_OK, () => {
        gsap.fromTo(
          mainRef.current.children,
          { autoAlpha: 0, y: 16 },
          { autoAlpha: 1, y: 0, duration: 0.45, ease: 'power3.out', clearProps: 'transform,opacity,visibility' }
        );
      });
      ScrollTrigger.refresh();
      return () => mm.revert();
    },
    { dependencies: [activeScreen] }
  );

  // Load initial health, featured scenes & benchmark data on mount
  useEffect(() => {
    // The backend can take a while to start, so keep checking until it answers
    let healthTimer;
    const checkHealth = () => {
      fetch('/api/health')
        .then((res) => {
          if (!res.ok) throw new Error('Backend not ready');
          return res.json();
        })
        .then((data) => setApiHealth(data))
        .catch(() => {
          setApiHealth({ status: 'offline', models_loaded: false });
          healthTimer = setTimeout(checkHealth, 3000);
        });
    };
    checkHealth();

    fetch('/api/benchmark')
      .then((res) => res.json())
      .then((data) => setBenchmarkData(data.benchmark || []))
      .catch(() => {});

    fetch('/api/scenes/featured')
      .then((res) => res.json())
      .then((data) => {
        if (data.featured) {
          setFeaturedScenes(Object.values(data.featured).flat());
          setSceneResults(data.featured);
          setSearchMeta({
            query_used: 'Featured Movie Scenes (Multilingual)',
            total_results: data.total_featured,
            source: 'curated_multilingual_archive',
          });
        }
      })
      .catch(() => {});

    return () => clearTimeout(healthTimer);
  }, []);

  // Cleanup object URL on unmount
  useEffect(() => {
    return () => {
      if (videoPreviewUrl && videoPreviewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(videoPreviewUrl);
      }
    };
  }, [videoPreviewUrl]);

  // Video playback time synchronization with emotion segments
  const handleTimeUpdate = () => {
    if (!videoPlayerRef.current || !clipResult?.segments?.length) return;
    const curTime = videoPlayerRef.current.currentTime;
    setCurrentPlaybackTime(curTime);

    const segIdx = clipResult.segments.findIndex(
      (s) => curTime >= s.start_time_sec && curTime <= s.end_time_sec + 0.3
    );
    if (segIdx !== -1 && segIdx !== activeSegmentIndex) {
      setActiveSegmentIndex(segIdx);
    }
  };

  // Seek video player to specific segment
  const handleSeekToSegment = (seg, idx) => {
    setActiveSegmentIndex(idx);
    if (videoPlayerRef.current) {
      videoPlayerRef.current.currentTime = seg.start_time_sec;
      videoPlayerRef.current.play().catch(() => {});
    }
  };

  // SerpApi Search Handler
  const handleSearchScenes = async ({ movie, scene, character, language: lang, emotion: emo }) => {
    setIsSearching(true);
    setErrorMessage(null);

    try {
      const params = new URLSearchParams();
      if (movie && movie.trim()) params.append('movie', movie.trim());
      if (scene && scene.trim()) params.append('scene', scene.trim());
      if (character && character.trim()) params.append('character', character.trim());
      if (lang && lang !== 'all') params.append('language', lang);
      if (emo && emo !== 'all') params.append('emotion', emo);

      const res = await fetch(`/api/scenes/search?${params.toString()}`);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Scene search failed');
      }

      const data = await res.json();
      setSceneResults(data.results || []);
      setSearchMeta({
        query_used: data.query_used,
        total_results: data.total_results,
        source: data.source,
      });
    } catch (err) {
      setErrorMessage(err.message || 'Error searching movie scenes.');
    } finally {
      setIsSearching(false);
    }
  };

  // Progress Simulation Helper for animated loading bar
  const startProgressSimulation = (isYouTube = false) => {
    setAnalysisProgress(12);
    setAnalysisStage(isYouTube ? '[Phase 1/5] Connecting to YouTube stream with yt-dlp...' : '[Phase 1/5] Reading video container frames...');

    const timer = setInterval(() => {
      setAnalysisProgress((prev) => {
        if (prev < 28) {
          setAnalysisStage(isYouTube ? '[Phase 1/5] Downloading YouTube stream buffer...' : '[Phase 1/5] Inspecting video container...');
          return prev + 4;
        } else if (prev < 52) {
          setAnalysisStage('[Phase 2/5] Demuxing 16kHz audio track via FFmpeg...');
          return prev + 3;
        } else if (prev < 74) {
          setAnalysisStage('[Phase 3/5] Faster-Whisper multilingual speech-to-text...');
          return prev + 3;
        } else if (prev < 88) {
          setAnalysisStage('[Phase 4/5] Detecting faces and scoring expressions...');
          return prev + 2;
        } else if (prev < 96) {
          setAnalysisStage('[Phase 5/5] Fusing text, voice and face predictions...');
          return prev + 1;
        }
        return prev;
      });
    }, 350);

    return timer;
  };

  // Direct YouTube Analysis using yt-dlp on the backend
  const handleAnalyzeYouTube = async (urlOrId, sceneData = null) => {
    if (!urlOrId) return;
    setIsAnalyzing(true);
    setErrorMessage(null);
    setAnalyzingVideoId(urlOrId);

    const progressTimer = startProgressSimulation(true);

    try {
      const payload = {
        video_url: urlOrId,
        title: sceneData?.title || null,
        movie_name: sceneData?.movie_name || null,
        character: sceneData?.character || null,
        scene_description: sceneData?.scene_description || sceneData?.description || null,
        emotion_tag: sceneData?.emotion || sceneData?.emotion_tag || null,
        thumbnail_url: sceneData?.thumbnail_url || null,
        embed_url: sceneData?.embed_url || null,
        language_override: sceneData?.language || sceneData?.language_code || (language !== 'auto' ? language : null),
        translate_to_english: translate,
        frame_sampling_interval_sec: parseFloat(samplingInterval) || 1.0,
        segment_interval_sec: parseFloat(segmentDuration) || 3.0,
      };

      const res = await fetch('/api/scenes/analyze-youtube', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `API status ${res.status}` }));
        throw new Error(err.detail || 'YouTube video analysis failed');
      }

      const data = await res.json();
      setAnalysisProgress(100);
      setAnalysisStage('✓ Analysis complete. Finalizing timeline...');

      if (data.video_stream_url) {
        setVideoPreviewUrl(data.video_stream_url);
      } else {
        setVideoPreviewUrl(null);
      }

      setClipResult(data);
      setActiveSegmentIndex(0);
      if (data.video_id) {
        rememberAnalysis(
          summarizeAnalysis(data, {
            key: `youtube:${data.video_id}`,
            kind: 'youtube',
            videoId: data.video_id,
            title: data.title || sceneData?.title || `YouTube clip ${data.video_id}`,
            thumb: `https://i.ytimg.com/vi/${data.video_id}/hqdefault.jpg`,
          })
        );
      }

      setTimeout(() => {
        setActiveScreen('dashboard');
      }, 500);
    } catch (err) {
      setErrorMessage(err.message || 'YouTube video analysis failed.');
    } finally {
      clearInterval(progressTimer);
      setIsAnalyzing(false);
      setAnalyzingVideoId(null);
    }
  };

  // Full-Clip Neural Analysis Handler for uploaded file
  const handleAnalyzeVideo = async () => {
    if (!videoFile) {
      setErrorMessage('Choose a video file first.');
      return;
    }

    setIsAnalyzing(true);
    setErrorMessage(null);

    const progressTimer = startProgressSimulation(false);

    try {
      const formData = new FormData();
      formData.append('video', videoFile);
      formData.append('video_file', videoFile);
      formData.append('frame_sampling_interval_sec', samplingInterval);
      formData.append('segment_interval_sec', segmentDuration);
      if (language !== 'auto') {
        formData.append('language_override', language);
      }
      formData.append('translate_to_english', translate ? 'true' : 'false');
      formData.append('include_translation', translate ? 'true' : 'false');

      const res = await fetch('/api/scenes/analyze-clip', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `API status ${res.status}` }));
        throw new Error(err.detail || 'Full-clip video analysis failed');
      }

      const data = await res.json();
      setAnalysisProgress(100);
      setAnalysisStage('✓ Analysis complete. Finalizing timeline...');

      setClipResult(data);
      setActiveSegmentIndex(0);
      rememberAnalysis(
        summarizeAnalysis(data, {
          key: `upload:${videoFile.name}:${videoFile.size}`,
          kind: 'upload',
          title: videoFile.name.replace(/\.[^.]+$/, ''),
          thumb: await captureFrame(videoPreviewUrl),
        })
      );

      setTimeout(() => {
        setActiveScreen('dashboard');
      }, 500);
    } catch (err) {
      setErrorMessage(err.message || 'Video analysis failed.');
    } finally {
      clearInterval(progressTimer);
      setIsAnalyzing(false);
    }
  };

  // Load Curated Sample Demo Scene
  const handleLoadDemo = (langCode = 'ml') => {
    const demo = CURATED_SCENE_ANALYSES[langCode] || CURATED_SCENE_ANALYSES.ml;
    setClipResult(demo);
    setActiveSegmentIndex(0);
    setActiveScreen('dashboard');
  };

  // Reopening a recent analysis: YouTube clips are cached, so they re-run quickly
  const handleOpenHistory = (item) => {
    setActiveScreen('upload');
    if (item.kind === 'youtube') {
      setSelectedScene(null);
      setSelectedYouTubeUrl(`https://www.youtube.com/watch?v=${item.videoId}`);
      handleAnalyzeYouTube(item.videoId);
    } else {
      setErrorMessage(`To reopen "${item.title}", upload the file again.`);
    }
  };

  const handleOpenScene = (scene) => {
    setActiveScreen('upload');
    handleSelectSceneForAnalysis(scene);
  };

  // When clicking "Analyze ➔" or previewing ANY discovered scene card:
  const handleSelectSceneForAnalysis = (scene, triggerAnalysis = true) => {
    setSelectedScene(scene);
    const targetUrl = scene?.video_url || (scene?.video_id ? `https://www.youtube.com/watch?v=${scene.video_id}` : null);
    if (targetUrl) {
      setSelectedYouTubeUrl(targetUrl);
      if (triggerAnalysis) {
        handleAnalyzeYouTube(targetUrl, scene);
      }
    } else {
      if (triggerAnalysis) {
        setErrorMessage('This scene has no video link to analyze.');
      }
    }
  };

  return (
    <div className="app-shell">
      <Navigation
        activeScreen={activeScreen}
        setActiveScreen={setActiveScreen}
        apiHealth={apiHealth}
        resultEmotion={clipResult?.scene_summary?.dominant_emotion}
      />

      <main ref={mainRef} className={`app-content ${activeScreen === 'landing' ? 'is-home' : ''}`}>

      {/* Global Error Notification */}
      {errorMessage && (
        <div className="alert" role="alert">
          <span>{errorMessage}</span>
          <button onClick={() => setErrorMessage(null)} aria-label="Dismiss">✕</button>
        </div>
      )}

      {/* ========================================================= */}
      {/* SCREEN 1: LANDING OVERVIEW                                */}
      {/* ========================================================= */}
      {activeScreen === 'landing' && (
        <LandingHero
          onNavigate={(screen) => setActiveScreen(screen)}
          history={history}
          scenes={featuredScenes}
          onOpenHistory={handleOpenHistory}
          onOpenScene={handleOpenScene}
        />
      )}

      {/* ========================================================= */}
      {/* SCREEN 2: MOVIE SCENE DISCOVERY                           */}
      {/* ========================================================= */}
      {activeScreen === 'discovery' && (
        <SceneDiscovery
          sceneResults={sceneResults}
          onSearch={handleSearchScenes}
          isSearching={isSearching}
          searchMeta={searchMeta}
          onSelectSceneForAnalysis={handleSelectSceneForAnalysis}
          analyzingVideoId={analyzingVideoId}
          analysisProgress={analysisProgress}
          analysisStage={analysisStage}
        />
      )}

      {/* ========================================================= */}
      {/* SCREEN 3: VIDEO UPLOAD & CONFIGURATION                    */}
      {/* ========================================================= */}
      {activeScreen === 'upload' && (
        <VideoUploader
          videoFile={videoFile}
          setVideoFile={setVideoFile}
          videoPreviewUrl={videoPreviewUrl}
          setVideoPreviewUrl={setVideoPreviewUrl}
          samplingInterval={samplingInterval}
          setSamplingInterval={setSamplingInterval}
          segmentDuration={segmentDuration}
          setSegmentDuration={setSegmentDuration}
          language={language}
          setLanguage={setLanguage}
          translate={translate}
          setTranslate={setTranslate}
          onAnalyze={handleAnalyzeVideo}
          isAnalyzing={isAnalyzing}
          analysisProgress={analysisProgress}
          analysisStage={analysisStage}
          onLoadDemo={handleLoadDemo}
          onAnalyzeYouTube={handleAnalyzeYouTube}
          youtubeInputUrl={selectedYouTubeUrl}
          setYoutubeInputUrl={setSelectedYouTubeUrl}
          selectedScene={selectedScene}
          setSelectedScene={setSelectedScene}
        />
      )}

      {/* ========================================================= */}
      {/* SCREEN 4: FULL-SCENE DASHBOARD                            */}
      {/* ========================================================= */}
      {activeScreen === 'dashboard' && (
        <SceneDashboard
          clipResult={clipResult}
          videoPlayerRef={videoPlayerRef}
          videoPreviewUrl={videoPreviewUrl}
          currentPlaybackTime={currentPlaybackTime}
          activeSegmentIndex={activeSegmentIndex}
          onSeekToSegment={handleSeekToSegment}
          onSelectSegment={(idx) => {
            setActiveSegmentIndex(idx);
            setActiveScreen('segment');
          }}
          onTimeUpdate={handleTimeUpdate}
          onNewAnalysis={() => setActiveScreen('upload')}
        />
      )}

      {/* ========================================================= */}
      {/* SCREEN 5: SEGMENT INSPECTOR                               */}
      {/* ========================================================= */}
      {activeScreen === 'segment' && (
        <SegmentInspector
          clipResult={clipResult}
          activeSegmentIndex={activeSegmentIndex}
          setActiveSegmentIndex={setActiveSegmentIndex}
          onSeekToSegment={handleSeekToSegment}
          onNewAnalysis={() => setActiveScreen('upload')}
        />
      )}

      {/* ========================================================= */}
      {/* SCREEN 6: EXPLAINABILITY SUITE (GRAD-CAM)                 */}
      {/* ========================================================= */}
      {activeScreen === 'explainability' && (
        <ExplainabilitySuite
          clipResult={clipResult}
          activeSegmentIndex={activeSegmentIndex}
        />
      )}

      {/* ========================================================= */}
      {/* SCREEN 7: MULTILINGUAL TRANSCRIPT                         */}
      {/* ========================================================= */}
      {activeScreen === 'transcript' && (
        <MultilingualTranscript
          clipResult={clipResult}
          activeSegmentIndex={activeSegmentIndex}
          onSeekToSegment={handleSeekToSegment}
        />
      )}

      {/* ========================================================= */}
      {/* SCREEN 8: EVALUATION & ABLATION BENCHMARK                 */}
      {/* ========================================================= */}
      {activeScreen === 'evaluation' && (
        <EvaluationReport benchmarkData={benchmarkData} />
      )}
      </main>
    </div>
  );
}
