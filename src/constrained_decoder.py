import json
import re
import math
import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing import Callable, ClassVar, List, Optional, Sequence, Tuple

from src.model_engine import Model


class ConstrainedDecodingError(Exception):
    """Custom exception for constrained decoding errors."""

    pass


class ConstrainedDecoder(BaseModel):
    """Perform token-level constrained generation using LLM logits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _model: Model = PrivateAttr()

    _NUMBER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"-?(?:0|[1-9]\d*)"
        r"(?:\.\d+)?"
        r"(?:[eE][+-]?\d+)?"
    )

    def __init__(self, model: Model) -> None:
        """
        Initialize the constrained decoder.

        Args:
            model (Model): LLM model used for token generation.
        """
        super().__init__()
        self._model = model

    @staticmethod
    def _rank_token_ids(logits: List[float]) -> List[int]:
        """
        Rank vocabulary token IDs by descending logit.

        Args:
            logits (List[float]): Model logits.

        Returns:
            List[int]: Token IDs ordered from highest to lowest logit.
        """
        return sorted(
            range(len(logits)),
            key=lambda token_id: logits[token_id],
            reverse=True,
        )

    def _select_valid_token(
        self,
        logits: List[float],
        validator: Callable[[str], bool],
    ) -> Tuple[int, str]:
        """
        Select the highest-logit token satisfying a constraint.

        Trying tokens from highest to lowest logit until one is valid is
        equivalent to masking invalid token logits with negative infinity
        and selecting the argmax.

        Args:
            logits (List[float]): Current model logits.
            validator (Callable[[str], bool]):
                Constraint applied to decoded token text.

        Returns:
            Tuple[int, str]: Selected token ID and its decoded text.

        Raises:
            ConstrainedDecodingError: If no valid token exists.
        """
        scores = np.asarray(logits)
        # Examine small top-k groups first, preserving exact greedy order.
        for size in dict.fromkeys(min(n, len(scores))
                                  for n in (32, 256, 2048, len(scores))):
            ids = np.argpartition(scores, -size)[-size:]
            ids = ids[np.argsort(scores[ids])[::-1]]
            for raw_id in ids:
                token_id = int(raw_id)
                token_text = self._model.decode_token(token_id)
                if token_text and validator(token_text):
                    return token_id, token_text

        raise ConstrainedDecodingError(
            "No token satisfies the active decoding constraint."
        )

    @staticmethod
    def _candidate_prefix_is_valid(
        value: str,
        candidates: Sequence[str],
        terminators: Sequence[str],
    ) -> bool:
        """
        Check whether text can still produce one allowed candidate.

        Args:
            value (str): Generated candidate prefix.
            candidates (Sequence[str]): Allowed final candidates.
            terminators (Sequence[str]): Allowed output terminators.

        Returns:
            bool: True if generation may continue.
        """
        for candidate in candidates:
            if candidate.startswith(value):
                return True

            if value.startswith(candidate):
                remainder = value[len(candidate):]

                if not remainder:
                    return True

                if any(
                    remainder.startswith(terminator)
                    for terminator in terminators
                ):
                    return True

        return False

    @staticmethod
    def _completed_candidate(
        value: str,
        candidates: Sequence[str],
        terminators: Sequence[str],
    ) -> Optional[str]:
        """
        Return the completed candidate if generation has finished.

        Args:
            value (str): Current generated text.
            candidates (Sequence[str]): Allowed candidates.
            terminators (Sequence[str]): Allowed terminators.

        Returns:
            Optional[str]: Completed candidate or None.
        """
        for candidate in candidates:
            if value == candidate:
                longer_candidate_exists = any(
                    other != candidate
                    and other.startswith(candidate)
                    for other in candidates
                )

                if not longer_candidate_exists:
                    return candidate

            if value.startswith(candidate):
                remainder = value[len(candidate):]

                if any(
                    remainder.startswith(terminator)
                    for terminator in terminators
                ):
                    return candidate

        return None

    def select_from_candidates(
        self,
        prompt_message: str,
        candidates: Sequence[str],
        previous_tokens: str = "",
        terminators: Sequence[str] = ("\n",),
        max_tokens: int = 32,
    ) -> str:
        """
        Select exactly one value from a finite candidate set.

        Args:
            prompt_message (str): Instruction sent to the LLM.
            candidates (Sequence[str]): Allowed output values.
            previous_tokens (str): Output already generated before this value.
            terminators (Sequence[str]): Valid value terminators.
            max_tokens (int): Maximum generated tokens.

        Returns:
            str: Candidate selected by the LLM.

        Raises:
            ConstrainedDecodingError: If generation fails.
        """
        if not candidates:
            raise ConstrainedDecodingError(
                "At least one candidate is required."
            )

        unique_candidates = list(dict.fromkeys(candidates))

        if any(not candidate for candidate in unique_candidates):
            raise ConstrainedDecodingError(
                "Candidates cannot be empty strings."
            )

        generated = ""

        for _ in range(max_tokens):
            prefix = generated.lstrip(" \t")
            remaining = [c for c in unique_candidates if c.startswith(prefix)]
            if prefix and len(remaining) == 1:
                return remaining[0]
            logits = self._model.get_logits(
                prompt_message,
                previous_tokens + generated,
            )

            def validator(token_text: str) -> bool:
                raw_candidate = generated + token_text

                candidate = raw_candidate.lstrip(" \t")

                return self._candidate_prefix_is_valid(
                    candidate,
                    unique_candidates,
                    terminators,
                )

            _, token_text = self._select_valid_token(
                logits,
                validator,
            )

            generated += token_text

            normalized = generated.lstrip(" \t")

            completed = self._completed_candidate(
                normalized,
                unique_candidates,
                terminators,
            )

            if completed is not None:
                return completed

        raise ConstrainedDecodingError(
            "Maximum candidate generation length exceeded."
        )

    @staticmethod
    def _split_at_terminator(
        value: str,
        terminators: Sequence[str],
    ) -> Tuple[str, bool]:
        """
        Split text at the earliest configured terminator.

        Args:
            value (str): Generated text.
            terminators (Sequence[str]): Possible terminators.

        Returns:
            Tuple[str, bool]: Text before terminator and whether one
                was found.
        """
        positions = [
            value.find(terminator)
            for terminator in terminators
            if terminator and value.find(terminator) >= 0
        ]

        if not positions:
            return value, False

        position = min(positions)

        return value[:position], True

    @classmethod
    def _is_complete_number(cls, value: str) -> bool:
        """
        Check whether text represents a complete valid number.

        Args:
            value (str): Text to validate.

        Returns:
            bool: True if the number is complete.
        """
        return cls._NUMBER_PATTERN.fullmatch(value) is not None

    @staticmethod
    def _is_valid_number_prefix(value: str) -> bool:
        """
        Check whether text may still become a valid number.

        Args:
            value (str): Generated numeric prefix.

        Returns:
            bool: True if the prefix remains valid.
        """
        if value == "":
            return True

        index = 0
        length = len(value)

        if value[index] == "-":
            index += 1

            if index == length:
                return True

        if index >= length:
            return True

        if value[index] == "0":
            index += 1

            if (
                index < length
                and value[index] in "0123456789"
            ):
                return False

        elif value[index] in "0123456789":
            while (
                index < length
                and value[index] in "0123456789"
            ):
                index += 1

        else:
            return False

        if (
            index < length
            and value[index] == "."
        ):
            index += 1
            start_fraction = index
            while (
                index < length
                and value[index] in "0123456789"
            ):
                index += 1
            if index == length:
                return True
            if index == start_fraction:
                return False

        if (
            index < length
            and value[index] in "eE"
        ):
            index += 1

            if index == length:
                return True

            if value[index] in "+-":
                index += 1

                if index == length:
                    return True

            if value[index] not in "0123456789":
                return False

            while (
                index < length
                and value[index] in "0123456789"
            ):
                index += 1

        return index == length

    def generate_number(
        self,
        prompt_message: str,
        previous_tokens: str = "",
        terminators: Sequence[str] = (",", "}", "\n"),
        max_tokens: int = 32,
        integer: bool = False,
    ) -> float | int:
        """
        Generate a number while enforcing numeric syntax.

        Leading and trailing whitespace produced by the tokenizer is
        tolerated, while the numeric content itself remains constrained.

        Args:
            prompt_message (str): Instruction sent to the LLM.
            previous_tokens (str): Output generated before this value.
            terminators (Sequence[str]): Valid value terminators.
            max_tokens (int): Maximum generated tokens.

        Returns:
            float: Generated numeric value.

        Raises:
            ConstrainedDecodingError: If no valid number is produced.
        """
        generated = ""

        for _ in range(max_tokens):
            logits = self._model.get_logits(
                prompt_message,
                previous_tokens + generated,
            )

            def validator(token_text: str) -> bool:
                candidate = generated + token_text

                value, terminated = self._split_at_terminator(
                    candidate,
                    terminators,
                )

                normalized = value.strip()
                if integer and any(c in normalized for c in ".eE"):
                    return False

                if terminated:
                    return self._is_complete_number(
                        normalized
                    )

                return self._is_valid_number_prefix(
                    normalized
                )

            _, token_text = self._select_valid_token(
                logits,
                validator,
            )

            generated += token_text

            value, terminated = self._split_at_terminator(
                generated,
                terminators,
            )

            normalized = value.strip()

            if (
                terminated
                and self._is_complete_number(normalized)
            ):
                return self._finite_number(normalized, integer)

        normalized = generated.strip()

        if self._is_complete_number(normalized):
            return self._finite_number(normalized, integer)

        raise ConstrainedDecodingError(
            "Maximum numeric generation length exceeded."
        )

    @staticmethod
    def _finite_number(value: str, integer: bool = False) -> float | int:
        """Reject values that JSON cannot represent."""
        number = int(value) if integer else float(value)
        if not integer and not math.isfinite(number):
            raise ConstrainedDecodingError("Number is outside finite range")
        return number

    @staticmethod
    def _find_unescaped_quote(value: str) -> Optional[int]:
        """
        Find the first unescaped double quote.

        Args:
            value (str): Generated JSON string content.

        Returns:
            Optional[int]: Quote position or None.
        """
        for index, character in enumerate(value):
            if character != '"':
                continue

            backslashes = 0
            previous = index - 1

            while (
                previous >= 0
                and value[previous] == "\\"
            ):
                backslashes += 1
                previous -= 1

            if backslashes % 2 == 0:
                return index

        return None

    @staticmethod
    def _is_valid_json_string_prefix(value: str) -> bool:
        """
        Validate an incomplete JSON string value.

        Args:
            value (str): Generated content without opening quote.

        Returns:
            bool: True if the prefix can still form a valid JSON string.
        """
        index = 0

        while index < len(value):
            character = value[index]

            if ord(character) < 0x20:
                return False

            if character == '"':
                return False

            if character != "\\":
                index += 1
                continue

            index += 1

            if index == len(value):
                return True

            escape = value[index]

            if escape in '"\\/bfnrt':
                index += 1
                continue

            if escape != "u":
                return False

            index += 1

            remaining = len(value) - index
            digits_to_check = min(4, remaining)

            for offset in range(digits_to_check):
                if value[index + offset] not in (
                    "0123456789abcdefABCDEF"
                ):
                    return False

            if remaining < 4:
                return True

            index += 4

        return True

    @staticmethod
    def _decode_json_string(value: str) -> Optional[str]:
        """
        Decode raw JSON string content.

        Args:
            value (str): Content between JSON quotes.

        Returns:
            Optional[str]: Decoded string or None if invalid.
        """
        try:
            decoded = json.loads(
                f'"{value}"'
            )
        except json.JSONDecodeError:
            return None

        if not isinstance(decoded, str):
            return None

        return decoded

    def generate_string(
        self,
        prompt_message: str,
        previous_tokens: str = "",
        max_tokens: int = 128,
        allow_empty: bool = True,
    ) -> str:
        """
        Generate the content of a JSON string.

        The caller must include the opening double quote in
        previous_tokens. Generation stops on the first unescaped
        closing quote.

        Args:
            prompt_message (str): Instruction sent to the LLM.
            previous_tokens (str): Output including the opening quote.
            max_tokens (int): Maximum generated tokens.
            allow_empty (bool): Whether an empty string is accepted.

        Returns:
            str: Decoded generated string.

        Raises:
            ConstrainedDecodingError: If string generation fails.
        """
        generated = ""

        for _ in range(max_tokens):
            logits = self._model.get_logits(
                prompt_message,
                previous_tokens + generated,
            )

            def validator(token_text: str) -> bool:
                candidate = generated + token_text

                quote_position = self._find_unescaped_quote(
                    candidate
                )

                if quote_position is not None:
                    raw_value = candidate[:quote_position]

                    decoded = self._decode_json_string(
                        raw_value
                    )

                    if decoded is None:
                        return False

                    if not allow_empty and decoded == "":
                        return False

                    return True

                return self._is_valid_json_string_prefix(
                    candidate
                )

            _, token_text = self._select_valid_token(
                logits,
                validator,
            )

            generated += token_text

            quote_position = self._find_unescaped_quote(
                generated
            )

            if quote_position is None:
                continue

            raw_value = generated[:quote_position]

            decoded = self._decode_json_string(
                raw_value
            )

            if decoded is None:
                raise ConstrainedDecodingError(
                    "Generated an invalid JSON string."
                )

            if not allow_empty and decoded == "":
                raise ConstrainedDecodingError(
                    "Generated string cannot be empty."
                )

            return decoded

        raise ConstrainedDecodingError(
            "Maximum string generation length exceeded."
        )
