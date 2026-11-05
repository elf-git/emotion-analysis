# Multilingual Emotion Recognition Benchmark Report

Empirical evaluation of **English, Hindi, Malayalam, and Tamil** conversational emotion recognition.
Comparing **Approach A (Direct Multilingual Model)** against **Approach B (Translation-Based Model)**.

> [!IMPORTANT]
> **Dataset Scope & Limitation Disclosure:**
> - Evaluated on language-specific curated emotional conversational datasets.
> - Performance claims are strictly empirical and evaluated per language; no English MELD performance is extrapolated directly to Indic languages.
> - Dialectal variations and nuanced colloquial phrasing may exhibit variation outside the benchmark distribution.

## Summary Results Table

| Language | Code | Samples | Approach A Acc | Approach A Macro-F1 | Approach A W-F1 | Approach B Acc | Approach B Macro-F1 | Approach B W-F1 | Agreement |
|---|---|---|---|---|---|---|---|---|---|
| **English** | `en` | 14 | 92.9% | 92.4% | 92.4% | 78.6% | 79.6% | 79.6% | 71.4% |
| **Hindi** | `hi` | 14 | 100.0% | 100.0% | 100.0% | 64.3% | 63.5% | 63.5% | 64.3% |
| **Malayalam** | `ml` | 14 | 71.4% | 73.8% | 73.8% | 50.0% | 43.3% | 43.3% | 78.6% |
| **Tamil** | `ta` | 14 | 71.4% | 67.2% | 67.2% | 35.7% | 33.0% | 33.0% | 50.0% |

## Detailed Per-Language Analysis

### English (`en`)
- **Sample Size**: 14 utterances across 7 emotions (`neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`)
- **Approach A (Direct Multilingual)**: Accuracy = 92.86%, Macro-F1 = 92.38%, Weighted-F1 = 92.38%
- **Approach B (Translation-Based)**: Accuracy = 78.57%, Macro-F1 = 79.59%, Weighted-F1 = 79.59%
- **Cross-Approach Agreement**: 71.43%

#### Per-Class F1 Breakdown (Approach A vs Approach B):
| Emotion | Approach A F1 | Approach B F1 | Support |
|---|---|---|---|
| Neutral | 100.0% | 57.1% | 2 |
| Surprise | 80.0% | 100.0% | 2 |
| Fear | 100.0% | 66.7% | 2 |
| Sadness | 100.0% | 100.0% | 2 |
| Joy | 66.7% | 66.7% | 2 |
| Disgust | 100.0% | 100.0% | 2 |
| Anger | 100.0% | 66.7% | 2 |

### Hindi (`hi`)
- **Sample Size**: 14 utterances across 7 emotions (`neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`)
- **Approach A (Direct Multilingual)**: Accuracy = 100.00%, Macro-F1 = 100.00%, Weighted-F1 = 100.00%
- **Approach B (Translation-Based)**: Accuracy = 64.29%, Macro-F1 = 63.49%, Weighted-F1 = 63.49%
- **Cross-Approach Agreement**: 64.29%

#### Per-Class F1 Breakdown (Approach A vs Approach B):
| Emotion | Approach A F1 | Approach B F1 | Support |
|---|---|---|---|
| Neutral | 100.0% | 44.4% | 2 |
| Surprise | 100.0% | 0.0% | 2 |
| Fear | 100.0% | 66.7% | 2 |
| Sadness | 100.0% | 66.7% | 2 |
| Joy | 100.0% | 100.0% | 2 |
| Disgust | 100.0% | 100.0% | 2 |
| Anger | 100.0% | 66.7% | 2 |

### Malayalam (`ml`)
- **Sample Size**: 14 utterances across 7 emotions (`neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`)
- **Approach A (Direct Multilingual)**: Accuracy = 71.43%, Macro-F1 = 73.81%, Weighted-F1 = 73.81%
- **Approach B (Translation-Based)**: Accuracy = 50.00%, Macro-F1 = 43.29%, Weighted-F1 = 43.29%
- **Cross-Approach Agreement**: 78.57%

#### Per-Class F1 Breakdown (Approach A vs Approach B):
| Emotion | Approach A F1 | Approach B F1 | Support |
|---|---|---|---|
| Neutral | 50.0% | 36.4% | 2 |
| Surprise | 66.7% | 0.0% | 2 |
| Fear | 100.0% | 100.0% | 2 |
| Sadness | 66.7% | 0.0% | 2 |
| Joy | 66.7% | 66.7% | 2 |
| Disgust | 66.7% | 0.0% | 2 |
| Anger | 100.0% | 100.0% | 2 |

### Tamil (`ta`)
- **Sample Size**: 14 utterances across 7 emotions (`neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`)
- **Approach A (Direct Multilingual)**: Accuracy = 71.43%, Macro-F1 = 67.21%, Weighted-F1 = 67.21%
- **Approach B (Translation-Based)**: Accuracy = 35.71%, Macro-F1 = 32.97%, Weighted-F1 = 32.97%
- **Cross-Approach Agreement**: 50.00%

#### Per-Class F1 Breakdown (Approach A vs Approach B):
| Emotion | Approach A F1 | Approach B F1 | Support |
|---|---|---|---|
| Neutral | 57.1% | 30.8% | 2 |
| Surprise | 100.0% | 66.7% | 2 |
| Fear | 66.7% | 0.0% | 2 |
| Sadness | 100.0% | 0.0% | 2 |
| Joy | 66.7% | 0.0% | 2 |
| Disgust | 0.0% | 66.7% | 2 |
| Anger | 80.0% | 66.7% | 2 |

## Methodological Insights

1. **Direct Native Script Model (Approach A)** preserves culturally specific exclamations (e.g. *ഛീ*, *அடங்கு*, *शाबाश*) that generic machine translation can smooth away.
2. **Translation-Based Model (Approach B)** leverages the extensive conversational knowledge of the MELD-trained DistilBERT model once the utterance is accurately mapped into English.
3. **Ensemble & Consensus Strategy**: When both approaches agree, prediction confidence is high (entropy < 0.25). When they diverge, the system prioritizes the native script model for Indic languages to safeguard cultural sentiment.

---
*Benchmark generated automatically via `ml/evaluation/evaluate_multilingual.py`*