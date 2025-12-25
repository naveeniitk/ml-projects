from google.genai import types
from functions.write_file import write_file
from functions.run_python_file import run_python_file
from functions.get_files_info import get_files_info
from functions.get_files_contents import get_files_contents

workingDirectory: str = "calculator"


def call_function(
    function_call_part: types.Any,
    verbose: bool = False,
) -> types.Content:
    """
    call function

    Args:
        function_call_part (types.Any):
        verbose=False (undefined):

    Returns:
        types.Content

    """
    if verbose:
        print(f"Calling function: {function_call_part.name}({function_call_part.args})")
    else:
        print(f" - Calling function: {function_call_part.name}")

    response = ""

    if function_call_part.name == "get_files_contents":
        response = get_files_contents(workingDirectory, **function_call_part.args)

    if function_call_part.name == "get_files_info":
        response = get_files_info(workingDirectory, **function_call_part.args)

    if function_call_part.name == "write_file":
        response = write_file(workingDirectory, **function_call_part.args)

    if function_call_part.name == "run_python_file":
        response = run_python_file(workingDirectory, **function_call_part.args)

    if response == "":

        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=function_call_part.name,
                    response={
                        "error": f"Unknown function: {function_call_part.name}",
                    },
                )
            ],
        )

    return types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
                name=function_call_part.name,
                response={
                    "response": response,
                },
            )
        ],
    )
