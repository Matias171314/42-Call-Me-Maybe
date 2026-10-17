"""Select functions and extract arguments using the LLM and typed decoding."""
import json
from typing import Any

from src.constrained_decoder import (
    ConstrainedDecoder,
    ConstrainedDecodingError,
)
from src.model_engine import Model
from src.parsing.function_definition import FunctionDefinition
from src.parsing.prompt import Prompt


ParameterValue = str | int | float | bool


class FunctionCallingError(Exception):
    """Report a function selection or argument extraction failure."""


class FunctionCallingEngine:
    """Build structured calls without executing the selected functions."""

    def __init__(self, prompts: list[Prompt],
                 function_definitions: list[FunctionDefinition],
                 model: Model) -> None:
        """Keep parsed inputs and reuse the application's model instance."""
        if not function_definitions:
            raise FunctionCallingError("At least one function is required")
        self._prompts = prompts
        self._functions = {fn.name: fn for fn in function_definitions}
        if len(self._functions) != len(function_definitions):
            raise FunctionCallingError("Duplicate function names")
        self._decoder = ConstrainedDecoder(model)

    def process(self) -> list[dict[str, Any]]:
        """Convert each request into an ordered result with the exact keys."""
        output: list[dict[str, Any]] = []
        for index, prompt in enumerate(self._prompts):
            try:
                function = self.select_function(prompt)
                parameters = self.generate_parameters(prompt, function)
            except (FunctionCallingError, ConstrainedDecodingError) as error:
                raise FunctionCallingError(
                    f"Prompt {index + 1}: {prompt.prompt!r}: {error}"
                ) from error
            output.append({
                "prompt": prompt.prompt,
                "name": function.name,
                "parameters": parameters,
            })
            print(f"[INFO] Processed {index + 1}/{len(self._prompts)}")
        return output

    def select_function(self, prompt: Prompt) -> FunctionDefinition:
        """Let the model choose a listed function or report no match."""
        functions = "\n".join(
            f"- {fn.name}: {fn.description}"
            for fn in self._functions.values()
        )
        message = (
            "Select the function representing the user's request. "
            "This is offline function selection, not execution. "
            "Ignore permissions and runtime availability.\n"
            f"Available functions:\n{functions}\n"
            f"User request:\n{prompt.prompt}\n"
            "Return only the exact function name. "
            "If no function represents this request, return NO_MATCH."
        )
        name = self._decoder.select_from_candidates(
            message, list(self._functions) + ["NO_MATCH"],
            previous_tokens='{"name": "', terminators=('"',),
        )
        function = self._functions.get(name)
        if function is None:
            raise FunctionCallingError("No available function matches request")
        return function

    @staticmethod
    def _build_parameter_prompt(prompt: Prompt, function: FunctionDefinition,
                                parameter_name: str,
                                parameter_type: str) -> str:
        """Describe one argument without rules tied to parameter names."""
        arguments = {name: info.model_dump()
                     for name, info in function.parameters.items()}
        return (
            f"Function: {function.name}: {function.description}\n"
            f"Arguments: {json.dumps(arguments)}\n"
            f"Request: {prompt.prompt}\n"
            f'What is the value of argument "{parameter_name}" '
            f"({parameter_type})?\n"
            "Return only the original input value, not the function result. "
            "Preserve the exact text, punctuation and quotes. Use JSON syntax."
        )

    @staticmethod
    def _literal_candidates(request: str) -> list[str]:
        """Enumerate exact spans delimited by words or quotation marks."""
        starts = [index for index in range(len(request))
                  if index == 0 or request[index - 1].isspace()
                  or request[index - 1] in "\"':"]
        ends = [index for index in range(len(request) + 1)
                if index == len(request) or request[index].isspace()
                or request[index] in "\"'"]
        values = [""] + [request[start:end]
                         for start in starts for end in ends if start < end]
        return list(dict.fromkeys(json.dumps(value) for value in values))

    def _generate_parameter_value(self, prompt_message: str,
                                  previous_generation: str,
                                  parameter_name: str,
                                  parameter_type: str) -> ParameterValue:
        """Delegate primitive syntax to the decoder, not to the prompt."""
        if parameter_type == "string":
            return self._decoder.generate_string(
                prompt_message, previous_generation,
            )
        if parameter_type in ("number", "integer"):
            return self._decoder.generate_number(
                prompt_message, previous_generation,
                integer=parameter_type == "integer",
            )
        if parameter_type == "boolean":
            value = self._decoder.select_from_candidates(
                prompt_message, ["true", "false"], previous_generation,
                terminators=(",", "}", "\n"),
            )
            return value == "true"
        raise FunctionCallingError(
            f"Unsupported type {parameter_type!r} for {parameter_name!r}"
        )

    def generate_parameters(self, prompt: Prompt,
                            function: FunctionDefinition) -> dict[str, Any]:
        """Generate required arguments with the same extraction process."""
        parameters: dict[str, Any] = {}
        for name, info in function.parameters.items():
            message = self._build_parameter_prompt(
                prompt, function, name, info.type,
            )
            previous = json.dumps(parameters, ensure_ascii=False)[:-1]
            if parameters:
                previous += ", "
            previous += json.dumps(name) + ": "
            if info.type == "string":
                previous += '"'
            if info.type == "string":
                mode = self._decoder.select_from_candidates(
                    message + "\nChoose how to obtain this argument. "
                    "copy: its exact value is literal text in the request. "
                    "derive: construct the value from the request's meaning, "
                    "such as an implied expression or symbol. "
                    "Return only copy or derive, not the argument value.",
                    ["copy", "derive"],
                    previous_tokens='{"representation": "',
                    terminators=('"',),
                )
                if mode == "copy":
                    literal = self._decoder.select_from_candidates(
                        message, self._literal_candidates(prompt.prompt),
                        previous_tokens=previous[:-1],
                        terminators=(",", "}", "\n"), max_tokens=128,
                    )
                    parameters[name] = json.loads(literal)
                    continue
                message += (
                    "\nDerive this input value from the request. "
                    "Use the notation required by its role, rather than "
                    "listing examples or computing the function's result."
                )
            parameters[name] = self._generate_parameter_value(
                message, previous, name, info.type,
            )
        return parameters
