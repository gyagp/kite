"""Explicit wall-clock boundaries for native embedding and generation metrics."""

EMBEDDING_SUM_FIELDS = (
    "tokens",
    "padded_tokens",
    "input_count",
    "batch_count",
    "inference_ms",
    "preparation_ms",
    "postprocessing_ms",
    "total_ms",
)


def search_embedding_metrics(query, documents):
    """Sum sequential query/document work before deriving throughput."""
    return {
        **{key: query[key] + documents[key] for key in EMBEDDING_SUM_FIELDS},
        "search_phases": {
            name: {key: result[key] for key in EMBEDDING_SUM_FIELDS}
            for name, result in (("query", query), ("documents", documents))
        },
        "inputs": documents.get("inputs", []),
        "query_inputs": query.get("inputs", []),
    }


def embedding_performance(result, *, request_ms, setup_ms, model_load_ms):
    inference_ms = result["inference_ms"]
    return {
        "request_ms": request_ms,
        "setup_ms": setup_ms,
        "model_load_ms": model_load_ms,
        "inference_tokens_per_second": result["tokens"] * 1000 / inference_ms
        if inference_ms > 0
        else None,
        "inference_inputs_per_second": result["input_count"] * 1000 / inference_ms
        if inference_ms > 0
        else None,
        "timing_method": "embedding-graph-execution-and-cpu-readback; host-wall-clock",
    }


def generation_metrics(
    *,
    prompt_tokens,
    completion_tokens,
    first_step_tokens,
    started,
    prefill_finished,
    first_token,
    last_token,
    observed,
    input_preparation_ms=0.0,
    generator_setup_ms=0.0,
):
    prefill_seconds = max(0.0, prefill_finished - started)
    decode_tokens = max(0, completion_tokens - first_step_tokens)
    decode_seconds = (
        max(0.0, last_token - first_token)
        if first_token is not None and last_token is not None
        else 0.0
    )
    decode_rate = decode_tokens / decode_seconds if decode_tokens and decode_seconds > 0 else None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "first_step_tokens": first_step_tokens,
        "prefill_ms": prefill_seconds * 1000,
        "prefill_tokens_per_second": prompt_tokens / prefill_seconds
        if prefill_seconds > 0
        else None,
        "ttft_ms": (first_token - started) * 1000 if first_token is not None else None,
        "decode_tokens": decode_tokens,
        "decode_ms": decode_seconds * 1000,
        "decode_tokens_per_second": decode_rate,
        "decode_ms_per_token": decode_seconds * 1000 / decode_tokens if decode_tokens else None,
        "tokens_per_second": decode_rate,
        "total_ms": max(0.0, observed - started) * 1000,
        "input_preparation_ms": input_preparation_ms,
        "generator_setup_ms": generator_setup_ms,
        "timing_method": "synchronized-prefill-logits-readback; first-to-last-token-decode-wall-time",
    }
