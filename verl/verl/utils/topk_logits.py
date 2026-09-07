"""Memory-bounded top-k helpers shared by actor and teacher paths."""

import torch

# Bound FP32 vocabulary-sized temporaries independently of model batch size.
TOP_K_LOGITS_CHUNK_ROWS = 1024
TOP_K_LOGITS_CHUNK_BYTES = 512 * 1024 * 1024


def top_k_logits_chunk_rows(logits: torch.Tensor) -> int:
    """Return a safe row chunk size for FP32 vocab post-processing."""
    fp32_bytes = torch.finfo(torch.float32).bits // 8
    rows_by_bytes = TOP_K_LOGITS_CHUNK_BYTES // (logits.shape[-1] * fp32_bytes)
    return max(1, min(TOP_K_LOGITS_CHUNK_ROWS, rows_by_bytes))


def chunked_top_k_log_probs(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k ids and normalized log-probs without full ``logits.float()``."""
    if logits.ndim != 2:
        raise ValueError(f"expected [rows, vocab] logits, got {tuple(logits.shape)}")
    top_k = min(int(top_k), logits.shape[-1])
    chunk_rows = top_k_logits_chunk_rows(logits)
    ids_parts, log_prob_parts = [], []
    for start in range(0, logits.shape[0], chunk_rows):
        chunk = logits[start : start + chunk_rows]
        chunk_fp32 = chunk.float()
        top_logits, top_ids = torch.topk(chunk, k=top_k, dim=-1)
        top_log_probs = top_logits.float() - torch.logsumexp(chunk_fp32, dim=-1, keepdim=True)
        ids_parts.append(top_ids.to(torch.int64))
        log_prob_parts.append(top_log_probs)
    return torch.cat(ids_parts, dim=0), torch.cat(log_prob_parts, dim=0)
