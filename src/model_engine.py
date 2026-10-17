"""Model adapter using only the SDK public inference API."""


class Model:
    """Own the SDK instance and cache token decoding."""

    def __init__(self) -> None:
        """Load the required model after input validation."""
        from llm_sdk import Small_LLM_Model

        self.sdk = Small_LLM_Model()
        self._decoded: dict[int, str] = {}

    def get_logits(self, prompt_message: str,
                   previous_tokens: str) -> list[float]:
        """Return next-token logits using the non-thinking chat format."""
        prompt = (
            "<|im_start|>user\n" + prompt_message
            + "<|im_end|>\n<|im_start|>assistant\n"
            + "<think>\n\n</think>\n\n" + previous_tokens
        )
        ids = self.sdk.encode(prompt).tolist()[0]
        return list(self.sdk.get_logits_from_input_ids(ids))

    def decode_token(self, token_id: int) -> str:
        """Decode a token once and reuse its cached text."""
        if token_id not in self._decoded:
            value = self.sdk.decode([token_id])
            if not isinstance(value, str):
                raise TypeError("SDK decode must return text")
            self._decoded[token_id] = value
        return self._decoded[token_id]
