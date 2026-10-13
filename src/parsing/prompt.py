import json
from typing import Any, List

from pydantic import BaseModel, ValidationError, ConfigDict


class PromptError(Exception):
    """Custom exception for errors during prompts parsing."""
    pass


class Prompt(BaseModel):
    """
    Model representing a prompt entry.

    Attributes:
        prompt (str): The natural language query or instruction.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str


def get_prompts(file_path: str) -> List[Prompt]:
    """
    Read and validate prompts from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing prompts.

    Returns:
        List[Prompt]: Validated prompt entries.

    Raises:
        PromptError: If the file cannot be read, contains invalid JSON,
            has an invalid top-level structure, or contains prompt entries
            that do not match the expected schema.
    """

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data_prompts: Any = json.load(file)

    except FileNotFoundError as error:
        raise PromptError(
            f"File Not Found: The prompts file at '{file_path}' "
            "does not exist."
        ) from error

    except json.JSONDecodeError as error:
        raise PromptError(
            f"JSON Decode Error: The file '{file_path}' contains "
            f"invalid JSON data. Details: {error}"
        ) from error

    except PermissionError as error:
        raise PromptError(
            "Permission Denied: Lacking read permissions for "
            f"the file '{file_path}'."
        ) from error

    except (OSError, UnicodeError) as error:
        raise PromptError(
            f"File Error: Could not read '{file_path}'. "
            f"Details: {error}"
        ) from error

    if not isinstance(data_prompts, list):
        raise PromptError(
            "Invalid Format: The top-level JSON structure must be a list "
            "of prompts."
        )

    validated_prompts: List[Prompt] = []

    for index, item in enumerate(data_prompts):
        try:
            valid_prompt = Prompt.model_validate(item)
            validated_prompts.append(valid_prompt)

        except ValidationError as error:
            raise PromptError(
                f"Validation Error: Prompt at index {index} does not "
                f"match the expected schema.\nDetails: {error}"
            ) from error

    return validated_prompts
