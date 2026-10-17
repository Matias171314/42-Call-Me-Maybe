"""Generate function names and primitive values with token constraints."""
import json
import math
from typing import Callable

import numpy as np

from src.model_engine import Model


class ConstrainedDecodingError(Exception):
    """Report that generation cannot produce a valid value."""


class ConstrainedDecoder:
    """Select the highest-scoring continuation allowed by the current type."""

    def __init__(self, model: Model) -> None:
        """Reuse the model loaded by the application."""
        self._model = model

    def _select_valid_token(self, logits: list[float],
                            validator: Callable[[str], bool]) -> str:
        """Mask rejected tokens until the best remaining token is valid."""
        scores = np.asarray(logits, dtype=float).copy()
        while scores.size:
            token_id = int(np.argmax(scores))
            if not np.isfinite(scores[token_id]):
                break
            text = self._model.decode_token(token_id)
            if text and validator(text):
                return text
            scores[token_id] = -np.inf
        raise ConstrainedDecodingError("No valid token is available")

    def select_from_candidates(self, prompt_message: str,
                               candidates: list[str],
                               previous_tokens: str = "",
                               terminators: tuple[str, ...] = ("\n",),
                               max_tokens: int = 32) -> str:
        """Choose a candidate while preserving names with shared prefixes."""
        choices = list(dict.fromkeys(candidates))
        if not choices or any(not choice for choice in choices):
            raise ConstrainedDecodingError("Candidates must be nonempty")
        generated = ""

        def valid(token: str) -> bool:
            text = (generated + token).lstrip(" \t")
            return any(
                choice.startswith(text)
                or (text.startswith(choice)
                    and text[len(choice):].startswith(terminators))
                for choice in choices
            )

        for _ in range(max_tokens):
            prefix = generated.lstrip(" \t")
            remaining = [choice for choice in choices
                         if choice.startswith(prefix)]
            if prefix and len(remaining) == 1:
                return remaining[0]
            logits = self._model.get_logits(
                prompt_message, previous_tokens + generated,
            )
            generated += self._select_valid_token(logits, valid)
            text = generated.lstrip(" \t")
            for choice in choices:
                if text == choice and not any(
                    other != choice and other.startswith(choice)
                    for other in choices
                ):
                    return choice
                if (text.startswith(choice)
                        and text[len(choice):].startswith(terminators)):
                    return choice
        raise ConstrainedDecodingError("Candidate token limit exceeded")

    @staticmethod
    def _is_valid_number_prefix(value: str) -> bool:
        """Accept JSON numbers and incomplete parts such as '-' or '2e+'."""
        if not value or value == "-":
            return True
        index = 1 if value.startswith("-") else 0
        if value[index] == "0":
            index += 1
        elif value[index] in "123456789":
            while index < len(value) and value[index] in "0123456789":
                index += 1
        else:
            return False

        if index < len(value) and value[index] == ".":
            index += 1
            start = index
            while index < len(value) and value[index] in "0123456789":
                index += 1
            if index == start:
                return index == len(value)

        if index < len(value) and value[index] in "eE":
            index += 1
            if index < len(value) and value[index] in "+-":
                index += 1
            while index < len(value) and value[index] in "0123456789":
                index += 1
        return index == len(value)

    @classmethod
    def _is_complete_number(cls, value: str) -> bool:
        """Distinguish complete JSON numbers from unfinished prefixes."""
        if not cls._is_valid_number_prefix(value):
            return False
        try:
            number = json.loads(value)
        except ValueError:
            return False
        return type(number) in (int, float)

    @staticmethod
    def _split_at_terminator(text: str,
                             terminators: tuple[str, ...]) -> tuple[str, bool]:
        """Separate a value from the first generated output delimiter."""
        positions = [text.find(end) for end in terminators
                     if end and end in text]
        if positions:
            return text[:min(positions)].strip(), True
        return text.strip(), False

    def generate_number(self, prompt_message: str, previous_tokens: str = "",
                        terminators: tuple[str, ...] = (",", "}", "\n"),
                        max_tokens: int = 32,
                        integer: bool = False) -> float | int:
        """Generate a finite number, retaining exact integer precision."""
        generated = ""

        def valid(token: str) -> bool:
            value, done = self._split_at_terminator(
                generated + token, terminators,
            )
            if integer and any(char in value for char in ".eE"):
                return False
            if done:
                return self._is_complete_number(value)
            return self._is_valid_number_prefix(value)

        for _ in range(max_tokens):
            logits = self._model.get_logits(
                prompt_message, previous_tokens + generated,
            )
            generated += self._select_valid_token(logits, valid)
            value, done = self._split_at_terminator(generated, terminators)
            if done:
                if integer:
                    return int(value)
                number = float(value)
                if not math.isfinite(number):
                    raise ConstrainedDecodingError("Number is not finite")
                return number
        raise ConstrainedDecodingError("Number token limit exceeded")

    @staticmethod
    def _find_unescaped_quote(text: str) -> int | None:
        """Find the string terminator, ignoring escaped quotes."""
        escaped = False
        for index, char in enumerate(text):
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return index
        return None

    @staticmethod
    def _is_valid_json_string_prefix(text: str) -> bool:
        """Validate JSON string content, allowing unfinished final escapes."""
        index = 0
        while index < len(text):
            char = text[index]
            index += 1
            if char == '"' or ord(char) < 32:
                return False
            if char != "\\":
                continue
            if index == len(text):
                return True
            escape = text[index]
            index += 1
            if escape in '"\\/bfnrt':
                continue
            if escape != "u":
                return False
            for _ in range(4):
                if index == len(text):
                    return True
                if text[index] not in "0123456789abcdefABCDEF":
                    return False
                index += 1
        return True

    def generate_string(self, prompt_message: str, previous_tokens: str = "",
                        max_tokens: int = 128,
                        allow_empty: bool = True) -> str:
        """Generate JSON string content after the caller's opening quote."""
        generated = ""

        def valid(token: str) -> bool:
            text = generated + token
            end = self._find_unescaped_quote(text)
            if end is None:
                return self._is_valid_json_string_prefix(text)
            try:
                value = json.loads('"' + text[:end] + '"')
            except ValueError:
                return False
            return isinstance(value, str) and (allow_empty or value != "")

        for _ in range(max_tokens):
            logits = self._model.get_logits(
                prompt_message, previous_tokens + generated,
            )
            generated += self._select_valid_token(logits, valid)
            end = self._find_unescaped_quote(generated)
            if end is not None:
                value = json.loads('"' + generated[:end] + '"')
                if isinstance(value, str):
                    return value
        raise ConstrainedDecodingError("String token limit exceeded")
