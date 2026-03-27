from typing import List, Dict, Any
import re
import math


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())


def _compute_bleu(prediction: str, reference: str, n: int = 4) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens:
        return 0.0
    scores = []
    for k in range(1, n + 1):
        pred_ngrams = [tuple(pred_tokens[i:i + k]) for i in range(len(pred_tokens) - k + 1)]
        ref_ngrams = [tuple(ref_tokens[i:i + k]) for i in range(len(ref_tokens) - k + 1)]
        if not pred_ngrams:
            scores.append(0.0)
            continue
        ref_set = {}
        for ng in ref_ngrams:
            ref_set[ng] = ref_set.get(ng, 0) + 1
        clipped = 0
        pred_count = {}
        for ng in pred_ngrams:
            pred_count[ng] = pred_count.get(ng, 0) + 1
        for ng, count in pred_count.items():
            clipped += min(count, ref_set.get(ng, 0))
        precision = clipped / len(pred_ngrams) if pred_ngrams else 0.0
        scores.append(precision)
    if all(s == 0 for s in scores):
        return 0.0
    log_avg = sum(math.log(s + 1e-10) for s in scores) / len(scores)
    bp = min(1.0, len(pred_tokens) / max(len(ref_tokens), 1))
    return bp * math.exp(log_avg)


def _compute_rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / m if m > 0 else 0.0
    recall = lcs / n if n > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_bert_score_f1_approx(prediction: str, reference: str) -> float:
    pred_tokens = set(_tokenize(prediction))
    ref_tokens = set(_tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class GenerationEvaluator:
    def __init__(self):
        pass

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        bleu_scores = [_compute_bleu(p, r) for p, r in zip(predictions, references)]
        rouge_scores = [_compute_rouge_l(p, r) for p, r in zip(predictions, references)]
        bert_scores = [_compute_bert_score_f1_approx(p, r) for p, r in zip(predictions, references)]
        return {
            "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
            "bert_score_f1": sum(bert_scores) / len(bert_scores) if bert_scores else 0.0,
        }
