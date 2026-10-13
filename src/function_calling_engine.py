import json
import unicodedata
import re
from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, PrivateAttr, TypeAdapter

from src.constrained_decoder import (
    ConstrainedDecoder,
    ConstrainedDecodingError,
)
from src.model_engine import Model
from src.parsing.function_definition import FunctionDefinition
from src.parsing.prompt import Prompt


ParameterValue = float | int | str | bool


class FunctionCallingError(Exception):
    """Custom exception for function-calling processing errors."""

    pass


class FunctionCallingEngine(BaseModel):
    """Convert natural-language prompts into structured function calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _prompts: List[Prompt] = PrivateAttr()
    _function_definitions: List[FunctionDefinition] = PrivateAttr()
    _decoder: ConstrainedDecoder = PrivateAttr()
    _functions_by_name: Dict[str, FunctionDefinition] = PrivateAttr()

    def __init__(
        self,
        prompts: List[Prompt],
        function_definitions: List[FunctionDefinition],
        model: Model,
    ) -> None:
        """
        Initialize the function-calling engine.

        Args:
            prompts (List[Prompt]): Prompts to process.
            function_definitions (List[FunctionDefinition]):
                Available function definitions.
            model (Model): Language model used for generation.

        Raises:
            FunctionCallingError: If no functions are available or
                duplicate function names are found.
        """
        super().__init__()
        if not function_definitions:
            raise FunctionCallingError(
                "At least one function definition is required."
            )

        self._prompts = prompts
        self._function_definitions = function_definitions
        self._decoder = ConstrainedDecoder(model)

        self._functions_by_name: Dict[
            str,
            FunctionDefinition,
        ] = {
            function.name: function
            for function in function_definitions
        }

        if (
            len(self._functions_by_name)
            != len(function_definitions)
        ):
            raise FunctionCallingError(
                "Duplicate function names are not allowed."
            )

    def process(self) -> List[Dict[str, Any]]:
        """
        Process all input prompts.

        Returns:
            List[Dict[str, Any]]: Generated function calls.

        Raises:
            FunctionCallingError: If a prompt cannot be processed.
        """
        output: List[Dict[str, Any]] = []

        for index, prompt in enumerate(self._prompts):
            try:
                function = self.select_function(prompt)

                parameters = self.generate_parameters(
                    prompt,
                    function,
                )

            except (
                FunctionCallingError,
                ConstrainedDecodingError,
            ) as error:
                raise FunctionCallingError(
                    f"Could not process prompt at index {index}: "
                    f"'{prompt.prompt}'. Details: {error}"
                ) from error

            result: Dict[str, Any] = {
                "prompt": prompt.prompt,
                "name": function.name,
                "parameters": parameters,
            }

            output.append(result)
            print(f"[INFO] Processed {index + 1}/{len(self._prompts)}")

        return output

    def _build_function_selection_prompt(
        self,
        prompt: Prompt,
    ) -> str:
        """
        Build the prompt used to select a function.

        Args:
            prompt (Prompt): User request.

        Returns:
            str: Function-selection prompt.
        """
        available_functions = "\n".join(
            (
                f"- {function.name}: "
                f"{function.description}"
            )
            for function in self._function_definitions
        )

        return (
            "Select the function that best matches the user's "
            "request.\n\n"
            "Available functions:\n"
            f"{available_functions}\n\n"
            "User request:\n"
            f"{prompt.prompt}\n\n"
            "Return only the exact function name. "
            "If none of the functions can satisfy the request, "
            "return NO_MATCH."
        )

    def select_function(
        self,
        prompt: Prompt,
    ) -> FunctionDefinition:
        """
        Select the most appropriate function.

        The model chooses semantically between the available
        functions while constrained decoding guarantees that only
        a valid function name can be produced.

        Args:
            prompt (Prompt): User request.

        Returns:
            FunctionDefinition: Selected function.

        Raises:
            FunctionCallingError: If the selected function cannot
                be resolved.
        """
        prompt_message = (
            self._build_function_selection_prompt(
                prompt
            )
        )

        candidates = [
            function.name
            for function in self._function_definitions
        ] + ["NO_MATCH"]

        selected_name = (
            self._decoder.select_from_candidates(
                prompt_message=prompt_message,
                candidates=candidates,
            )
        )

        if selected_name == "NO_MATCH":
            confirmation = self._decoder.select_from_candidates(
                "This is offline function selection, not execution. "
                "Can any listed function represent this request? "
                "Ignore permissions and runtime availability. "
                "Answer true or false.\n"
                + prompt_message,
                ["true", "false"],
            )
            if confirmation == "false":
                raise FunctionCallingError(
                    "No available function matches request"
                )
            selected_name = self._decoder.select_from_candidates(
                prompt_message.split("If none of the functions")[0],
                candidates[:-1],
            )

        selected_function = (
            self._functions_by_name.get(
                selected_name
            )
        )

        if selected_function is None:
            raise FunctionCallingError(
                f"Unknown function selected: "
                f"'{selected_name}'."
            )

        return selected_function

    @staticmethod
    def _format_parameter_value(
        value: ParameterValue,
    ) -> str:
        """
        Format a generated value for subsequent model context.

        Args:
            value (ParameterValue): Generated parameter value.

        Returns:
            str: Formatted parameter value.
        """
        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, str):
            return json.dumps(
                value,
                ensure_ascii=False,
            )

        return str(value)

    def _build_previous_generation(
        self,
        generated_parameters: Dict[str, Any],
        current_parameter: str,
        current_type: str,
    ) -> str:
        """
        Build the partial argument context for generation.

        Args:
            generated_parameters (Dict[str, Any]):
                Parameters already generated.
            current_parameter (str): Parameter being generated.
            current_type (str): Expected parameter type.

        Returns:
            str: Partial argument context.
        """
        previous_generation = "{"

        for name, value in generated_parameters.items():
            previous_generation += (
                f"{json.dumps(name)}: "
                f"{self._format_parameter_value(value)}"
                ", "
            )

        previous_generation += (
            f'{json.dumps(current_parameter)}: '
        )

        if current_type == "string":
            previous_generation += '"'

        return previous_generation

    @staticmethod
    def _build_parameter_prompt(
        prompt: Prompt,
        function: FunctionDefinition,
        parameter_name: str,
        parameter_type: str,
    ) -> str:
        """
        Build the prompt used to generate one function argument.

        Args:
            prompt (Prompt): Original user request.
            function (FunctionDefinition): Selected function.
            parameter_name (str): Parameter being generated.
            parameter_type (str): Expected parameter type.

        Returns:
            str: Parameter-generation prompt.
        """
        argument_types = {
            k: v.model_dump(exclude_defaults=True, exclude_none=True)
            for k, v in function.parameters.items()
        }
        guidance = "Copy literal text exactly, preserving embedded quotes."
        role = parameter_name.lower()
        if "regex" in role or "pattern" in role:
            guidance = (
                "Return the shortest regex matching the requested targets, "
                "not the replacement. Match individual occurrences. "
                'Example JSON regex for whitespace: "\\\\s+". '
                "Double regex backslashes in JSON."
            )
        elif "replacement" in role:
            guidance = (
                "Return the replacement text, not the matching pattern. "
                "If a character is named, return that character: "
                "asterisk means *, percent sign means %. "
                "Use one character unless repetition is requested."
            )
        return (
            "Extract one argument. Never calculate the function result.\n"
            f"Function: {function.name}: {function.description}\n"
            f"Arguments: {json.dumps(argument_types)}\n"
            f"Request: {prompt.prompt}\n"
            f"Argument to extract: {parameter_name} ({parameter_type}).\n"
            f"{guidance} Return JSON."
        )

    def _generate_parameter_value(
        self,
        prompt_message: str,
        previous_generation: str,
        parameter_name: str,
        parameter_type: str,
    ) -> ParameterValue:
        """
        Generate one value according to its schema type.

        Args:
            prompt_message (str): Semantic generation prompt.
            previous_generation (str): Partial argument context.
            parameter_name (str): Parameter being generated.
            parameter_type (str): Expected parameter type.

        Returns:
            ParameterValue: Generated and constrained value.

        Raises:
            FunctionCallingError: If the type is unsupported.
        """
        if parameter_type in ("number", "integer"):
            value = self._decoder.generate_number(
                prompt_message=prompt_message,
                previous_tokens=previous_generation,
                integer=parameter_type == "integer",
                terminators=(
                    "\n",
                    ",",
                    "}",
                ),
            )

            if parameter_type == "integer":
                if isinstance(value, float) and not value.is_integer():
                    raise FunctionCallingError("Expected an integer argument")
                return int(value)
            return value

        if parameter_type == "string":
            return self._decoder.generate_string(
                prompt_message=prompt_message,
                previous_tokens=previous_generation,
            )

        if parameter_type == "boolean":
            generated_boolean = (
                self._decoder.select_from_candidates(
                    prompt_message=prompt_message,
                    candidates=[
                        "true",
                        "false",
                    ],
                    previous_tokens=previous_generation,
                    terminators=(
                        "\n",
                        ",",
                        "}",
                    ),
                )
            )

            return generated_boolean == "true"

        raise FunctionCallingError(
            "Unsupported parameter type "
            f"'{parameter_type}' for parameter "
            f"'{parameter_name}'."
        )

    @staticmethod
    def _restore_literal(request: str, value: str) -> str:
        """Recover embedded quotes omitted from a colon-delimited literal."""
        if ":" not in request:
            return value
        literal = request.split(":", 1)[1].strip()
        if (len(literal) >= 2 and literal[0] in "\"'"
                and literal[-1] == literal[0]):
            return value
        without_quotes = literal.replace('"', '').replace("'", '')
        generated_text = value.replace('"', '').replace("'", '')
        if literal != value and without_quotes == generated_text:
            return literal
        return value

    @staticmethod
    def _path_candidates(request: str) -> list[str]:
        """Extract literal path spans, preserving Windows separators."""
        pattern = re.compile(r"(?:[A-Za-z]:[\\/]|/|\.\.?/)[^\s\"']+")
        quoted_spans = list(re.finditer(r"([\"'])(.*?)\1", request))
        quoted = [m.group(2) for m in quoted_spans
                  if pattern.match(m.group(2))]
        bare = [m.group() for m in pattern.finditer(request)
                if not any(q.start() < m.start() < q.end()
                           for q in quoted_spans)]
        values = quoted + bare
        return list(dict.fromkeys(values))

    def generate_parameters(
        self,
        prompt: Prompt,
        function: FunctionDefinition,
    ) -> Dict[str, Any]:
        """
        Generate all parameters required by a function.

        Parameters are generated sequentially according to the
        function schema. Previously generated values are supplied
        as context for subsequent parameters.

        Args:
            prompt (Prompt): Original user request.
            function (FunctionDefinition): Selected function.

        Returns:
            Dict[str, Any]: Generated parameter values.
        """
        parameters: Dict[str, Any] = {}

        for (
            parameter_name,
            parameter_info,
        ) in function.parameters.items():
            parameter_type = parameter_info.type

            prompt_message = (
                self._build_parameter_prompt(
                    prompt=prompt,
                    function=function,
                    parameter_name=parameter_name,
                    parameter_type=parameter_type,
                )
            )

            previous_generation = (
                self._build_previous_generation(
                    generated_parameters=parameters,
                    current_parameter=parameter_name,
                    current_type=parameter_type,
                )
            )

            if parameter_info.enum is not None:
                literal = self._decoder.select_from_candidates(
                    prompt_message, [json.dumps(v) for v in
                                     parameter_info.enum],
                    previous_tokens=previous_generation.rstrip('"'),
                    terminators=(",", "}", "\n"),
                )
                value = json.loads(literal)
            elif (parameter_type == "string"
                  and "path" in parameter_name.lower()
                  and self._path_candidates(prompt.prompt)):
                literal = self._decoder.select_from_candidates(
                    prompt_message,
                    [json.dumps(path) for path in
                     self._path_candidates(prompt.prompt)],
                    previous_tokens=previous_generation[:-1],
                    terminators=(",", "}", "\n"), max_tokens=128,
                )
                value = json.loads(literal)
            else:
                value = self._generate_parameter_value(
                    prompt_message=prompt_message,
                    previous_generation=previous_generation,
                    parameter_name=parameter_name,
                    parameter_type=parameter_type,
                )

            if parameter_info.enum is None and isinstance(value, str):
                value = self._restore_literal(prompt.prompt, value)
                if ("path" in parameter_name.lower()
                        and value != value.strip()
                        and f'"{value}"' not in prompt.prompt
                        and f"'{value}'" not in prompt.prompt):
                    value = value.strip()

            if (parameter_info.enum is None and isinstance(value, str)
                    and "replacement" in parameter_name.lower()
                    and value.lower() in prompt.prompt.lower()
                    and f"'{value}'" not in prompt.prompt
                    and f'"{value}"' not in prompt.prompt):
                # Recover named characters through the Unicode standard,
                # while preserving explicitly quoted replacement words.
                try:
                    value = unicodedata.lookup(value.upper())
                except KeyError:
                    pass

            expected = {"string": str, "number": float,
                        "integer": int, "boolean": bool}[parameter_type]
            value = TypeAdapter(expected).validate_python(value, strict=True)
            if (parameter_info.enum is not None
                    and value not in parameter_info.enum):
                raise FunctionCallingError("Argument outside allowed enum")
            parameters[parameter_name] = value

        return parameters
