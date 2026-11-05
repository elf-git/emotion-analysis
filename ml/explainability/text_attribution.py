"""
text_attribution.py - Word-Level Saliency & Integrated Gradients for Text
Explains text emotion predictions by identifying the exact words/tokens
that drove the emotional classification.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class TextAttributor:
    """
    Computes token-level and word-level attribution scores using
    Integrated Gradients and Attention Pooling Attribution.
    """

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def attribute_attention(self, text: str, max_seq_length: int = 128) -> Dict:
        """
        Extract token attribution scores directly from model attention pooling head.
        """
        encoding = self.tokenizer(
            text,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.model(input_ids, attention_mask)

        logits = out["logits"].squeeze(0)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        pred_idx = int(probs.argmax())

        # Attention weights if provided by model
        if "token_importance" in out:
            importance = out["token_importance"].squeeze(0).cpu().numpy()
        else:
            importance = np.ones(input_ids.shape[1], dtype=np.float32) / input_ids.shape[1]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())

        # Filter special/padding tokens
        clean_tokens, clean_scores = [], []
        special = {self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token, "<pad>", "<s>", "</s>", "[PAD]", "[CLS]", "[SEP]"}
        for tok, score in zip(tokens, importance):
            if tok not in special:
                clean_tokens.append(tok)
                clean_scores.append(float(score))

        # Normalize clean scores
        total_s = sum(clean_scores) + 1e-9
        normalized_scores = [s / total_s for s in clean_scores]

        return {
            "text": text,
            "predicted_class": pred_idx,
            "probabilities": probs.tolist(),
            "confidence": float(probs[pred_idx]),
            "tokens": clean_tokens,
            "attribution_scores": normalized_scores,
        }

    def attribute_integrated_gradients(
        self,
        text: str,
        target_class: Optional[int] = None,
        steps: int = 20,
        max_seq_length: int = 128,
    ) -> Dict:
        """
        Compute Integrated Gradients attribution for input text tokens.
        """
        encoding = self.tokenizer(
            text,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get embedding layer of the encoder
        encoder = getattr(self.model, "encoder", self.model)
        if hasattr(encoder, "embeddings"):
            embedding_layer = encoder.embeddings.word_embeddings
        elif hasattr(encoder, "get_input_embeddings"):
            embedding_layer = encoder.get_input_embeddings()
        else:
            # Fallback to attention attribution if embeddings not directly accessible
            return self.attribute_attention(text, max_seq_length)

        input_embeds = embedding_layer(input_ids).detach()
        baseline_embeds = torch.zeros_like(input_embeds)

        # First forward pass to determine prediction
        with torch.no_grad():
            out = self.model(input_ids, attention_mask)
            pred_idx = int(out["logits"].argmax(dim=-1).item()) if target_class is None else target_class
            confidence = float(F.softmax(out["logits"], dim=-1)[0, pred_idx].item())

        # Riemann integration along interpolation path
        scaled_grads = []
        for step in range(1, steps + 1):
            alpha = float(step) / steps
            interp_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
            interp_embeds.requires_grad_(True)

            # Forward with inputs_embeds
            if hasattr(self.model, "forward_with_embeds"):
                logits = self.model.forward_with_embeds(interp_embeds, attention_mask)["logits"]
            else:
                # Approximate via attention fallback if custom embed forward not available
                return self.attribute_attention(text, max_seq_length)

            target_score = logits[0, pred_idx]
            self.model.zero_grad()
            target_score.backward()

            grad = interp_embeds.grad.detach()
            scaled_grads.append(grad)

        avg_grads = torch.stack(scaled_grads).mean(dim=0) # (1, seq_len, dim)
        delta = input_embeds - baseline_embeds
        ig = (delta * avg_grads).sum(dim=-1).squeeze(0) # (seq_len,)
        ig_scores = F.relu(ig).cpu().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
        special = {self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token, "<pad>", "<s>", "</s>", "[PAD]", "[CLS]", "[SEP]"}
        
        clean_tokens, clean_scores = [], []
        for tok, score in zip(tokens, ig_scores):
            if tok not in special:
                clean_tokens.append(tok)
                clean_scores.append(float(score))

        total = sum(clean_scores) + 1e-9
        normalized_scores = [s / total for s in clean_scores]

        return {
            "text": text,
            "predicted_class": pred_idx,
            "confidence": confidence,
            "tokens": clean_tokens,
            "attribution_scores": normalized_scores,
        }


def explain_text_utterance(
    text: str,
    attributions: Dict,
    emotion_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> str:
    """
    Format attribution results into an interpretable ASCII/text report.
    """
    if emotion_names is None:
        emotion_names = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]

    pred_idx = attributions["predicted_class"]
    pred_label = emotion_names[pred_idx] if pred_idx < len(emotion_names) else str(pred_idx)
    conf = attributions.get("confidence", 0.0)

    tokens = attributions["tokens"]
    scores = attributions["attribution_scores"]

    ranked = sorted(zip(tokens, scores), key=lambda x: x[1], reverse=True)[:top_k]

    report = []
    report.append(f"Utterance: \"{text}\"")
    report.append(f"Prediction: {pred_label.upper()} (Confidence: {conf*100:.1f}%)")
    report.append("Top Influential Words / Tokens:")
    for tok, score in ranked:
        bar = "█" * int(score * 30)
        report.append(f"  {tok:<14} | {score:6.4f}  {bar}")

    return "\n".join(report)
