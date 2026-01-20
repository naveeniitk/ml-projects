import subprocess
import os
from google.genai import types


def run_python_file(
    working_directory: str,
    file_path: str = ".",
    args: list = [],
) -> str:

    absWorkingDirectory = os.path.abspath(working_directory)
    absFilePath = os.path.abspath(os.path.join(absWorkingDirectory, file_path))
    maxLenOfFileInfo = 30

    if not absFilePath.startswith(absWorkingDirectory):
        finalReponse = f"Error: {absFilePath} is not in the Working Directory\n"
    elif not os.path.isfile(absFilePath):
        finalReponse = f"Error: {absFilePath} is not a valid file\n"
    elif not absFilePath.endswith(".py"):
        finalReponse = f"Error: {absFilePath} is not a valid Python file\n"
    else:
        try:
            finalArguments = ["python3", absFilePath]
            finalArguments.extend(args)
            processOutput = subprocess.run(
                finalArguments,
                cwd=absWorkingDirectory,
                timeout=30,
                capture_output=True,
            )
            finalReponse = f"""
STDOUT: {processOutput.stdout}
STDERR: {processOutput.stderr}
"""
            if processOutput.returncode != 0:
                finalReponse += (
                    f"Process Exited with code: {processOutput.returncode}\n"
                )
            elif processOutput.stdout == "" and processOutput.stderr == "":
                finalReponse += "No output from the process!!"

        except Exception as e:
            finalReponse = f"Exception in executing file: '{e.__str__()}'\n"

    finalReponse = (
        ("=" * maxLenOfFileInfo)
        + "\n"
        + str(finalReponse)
        + "\n"
        + ("=" * maxLenOfFileInfo)
    )

    print(f"finalReponse: \n{finalReponse}")

    return finalReponse


schema_run_python_file = types.FunctionDeclaration(
    name="run_python_file",
    description="Run a python file with python3 interpreter. Accepts additional CLI args as an optional array.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="The file to run, relative to the working directory.",
            ),
            "args": types.Schema(
                type=types.Type.ARRAY,
                description="An optional array of strings to be used as the CLI args for the python file.",
                items=types.Schema(
                    type=types.Type.STRING,
                ),
            ),
        },
    ),
)
