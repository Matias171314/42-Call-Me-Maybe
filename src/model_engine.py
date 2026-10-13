"""Validated adapter using only the SDK public inference API."""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class Model(BaseModel):
    """Own the SDK instance and cache token decoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    sdk: Any = Field(exclude=True)
    _decoded: dict[int, str] = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        """Load the required model lazily after input validation."""
        from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
        super().__init__(sdk=Small_LLM_Model(**kwargs))

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
        """Decode a token through a bounded shared cache."""
        if token_id not in self._decoded:
            value = self.sdk.decode([token_id])
            if not isinstance(value, str):
                raise TypeError("SDK decode must return text")
            self._decoded[token_id] = value
        return self._decoded[token_id]
