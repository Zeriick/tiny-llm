import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def _validate_max_tokens(max_tokens: int) -> None:
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 0
    ):
        raise ValueError("max_tokens must be a non-negative integer")


def _release_kv_cache(kv_cache):
    if kv_cache is None:
        return
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
    max_tokens: int = 256,
) -> None:
    _validate_max_tokens(max_tokens)
    if max_tokens == 0:
        return

    def _step(model, y):
        logits = model(y[None])
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        if sampler is None:
            return mx.argmax(logprobs, axis=-1)
        return sampler(logprobs)

    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    if tokens.size == 0:
        raise ValueError("prompt must encode to at least one token")
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    for _ in range(max_tokens):
        token = _step(model, tokens)
        mx.eval(token)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)

    detokenizer.finalize()
    print(detokenizer.last_segment, end="", flush=True)


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int = 256,
) -> str:
    _validate_max_tokens(max_tokens)
    if max_tokens == 0:
        return ""

    kv_cache = model.create_kv_cache()

    def _step(model, y, offset, kv_cache):
        logits = model(y[None], offset, kv_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        token = mx.argmax(logprobs, axis=-1)
        return token, logprobs.squeeze(0)

    try:
        tokens = mx.array(
            tokenizer.encode(prompt, add_special_tokens=False), dtype=mx.int32
        )
        if tokens.size == 0:
            raise ValueError("prompt must encode to at least one token")
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        offset = 0
        emitted = 0

        while emitted < max_tokens:
            token, _ = _step(model, tokens, offset, kv_cache)
            mx.eval(token)
            if token.item() == tokenizer.eos_token_id:
                break
            detokenizer.add_token(token.item())
            print(detokenizer.last_segment, end="", flush=True)
            emitted += 1
            if emitted == max_tokens:
                break
            offset += tokens.size
            tokens = token

        detokenizer.finalize()
        print(detokenizer.last_segment, end="", flush=True)
        return detokenizer.text
    finally:
        _release_kv_cache(kv_cache)


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
    proposal_length: int = 4,
    max_tokens: int = 256,
) -> str:
    pass
