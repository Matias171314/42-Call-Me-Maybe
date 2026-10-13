import json
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ValidationError, ConfigDict, Field, TypeAdapter


class FunctionDefinitionError(Exception):
    """Custom exception for errors during function definitions parsing."""
    pass


class ParameterInfo(BaseModel):
    """
    Model representing the type information of a parameter or return value.

    Attributes:
        type (str): The expected data type (e.g., 'string', 'number').
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["number", "integer", "string", "boolean", "array",
                  "object", "null"]
    description: str = ""
    enum: List[Any] | None = None


class FunctionDefinition(BaseModel):
    """
    Model representing the complete definition of a function.

    Attributes:
        name (str): The name of the function.
        description (str): A brief description of what the function does.
        parameters (Dict[str, ParameterInfo]): Dictionary mapping parameter
            names to their type information.
        returns (ParameterInfo): The expected return type of the function.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    description: str
    parameters: Dict[str, ParameterInfo]
    returns: ParameterInfo


def get_function_definitions(
    file_path: str
) -> List[FunctionDefinition]:
    """
    Read and validate function definitions from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing function definitions.

    Returns:
        List[FunctionDefinition]: Validated function definitions.

    Raises:
        FunctionDefinitionError: If the file cannot be read, contains invalid
            JSON, has an invalid top-level structure, or contains function
            definitions that do not match the expected schema.
    """

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data_functions: Any = json.load(file)

    except FileNotFoundError as error:
        raise FunctionDefinitionError(
            "File Not Found: The function definitions file at "
            f"'{file_path}' does not exist."
        ) from error

    except json.JSONDecodeError as error:
        raise FunctionDefinitionError(
            f"JSON Decode Error: The file '{file_path}' contains "
            f"invalid JSON data. Details: {error}"
        ) from error

    except PermissionError as error:
        raise FunctionDefinitionError(
            "Permission Denied: Lacking read permissions for "
            f"the file '{file_path}'."
        ) from error

    except (OSError, UnicodeError) as error:
        raise FunctionDefinitionError(
            f"File Error: Could not read '{file_path}'. "
            f"Details: {error}"
        ) from error

    if not isinstance(data_functions, list):
        raise FunctionDefinitionError(
            "Invalid Format: The top-level JSON structure must be a list "
            "of functions."
        )

    validated_functions: List[FunctionDefinition] = []

    for index, function in enumerate(data_functions):
        try:
            valid_function = FunctionDefinition.model_validate(function)
            validated_functions.append(valid_function)

        except ValidationError as error:
            if isinstance(function, dict):
                fail_name = function.get("name", "Unknown")
            else:
                fail_name = "Unknown"

            raise FunctionDefinitionError(
                f"Validation Error: Function at index {index} "
                f"('{fail_name}') does not match the expected schema.\n"
                f"Details: {error}"
            ) from error

    if not validated_functions:
        raise FunctionDefinitionError("At least one function is required")
    names = [f.name for f in validated_functions]
    if len(names) != len(set(names)):
        raise FunctionDefinitionError("Duplicate function names")
    for function in validated_functions:
        for parameter in function.parameters.values():
            supported = {"string", "number", "integer", "boolean"}
            if parameter.type not in supported:
                raise FunctionDefinitionError(
                    f"Unsupported argument type: {parameter.type}"
                )
            if parameter.enum == []:
                raise FunctionDefinitionError("Enum cannot be empty")
            expected = {"string": str, "number": float,
                        "integer": int, "boolean": bool}[parameter.type]
            if parameter.enum is not None:
                try:
                    for value in parameter.enum:
                        TypeAdapter(expected).validate_python(
                            value, strict=True
                        )
                        json.dumps(value, allow_nan=False)
                except (ValidationError, ValueError) as error:
                    raise FunctionDefinitionError(
                        "Enum values must match the declared type"
                    ) from error
    return validated_functions
