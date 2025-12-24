import os
from google.genai import types

MAX_CHARS = 1000


def get_files_contents(
    working_directory: str,
    file_path: str = ".",
) -> str:
    """
    Args:
        working_directory (str)
        directory (str): Defaults to "."

    Returns:
        str: file contents information
    """
    absWorkingDirectory = os.path.abspath(working_directory)
    absFilePath = os.path.abspath(os.path.join(absWorkingDirectory, file_path))
    maxLenOfFileInfo = 0

    if not absFilePath.startswith(absWorkingDirectory):
        maxLenOfFileInfo = 33
        finalReponse = f"Error: {absFilePath} is not in the Working Directory\n"
    elif not os.path.isfile(absFilePath):
        finalReponse = f"Error: {absFilePath} is not a valid file\n"
    else:
        maxLenOfFileInfo = 33
        finalFileContents = ""
        try:
            with open(absFilePath, "r") as file:
                fileContents = file.read(MAX_CHARS)
                if len(fileContents) >= MAX_CHARS:
                    fileContents = fileContents[:MAX_CHARS]
                    fileContents += f'[...File "{absFilePath}" truncated to {MAX_CHARS} characters]\n'
                finalFileContents += fileContents
            finalReponse = finalFileContents
        except Exception as e:
            finalReponse = f"Exception in reading file: '{absFilePath}'\n"

    finalReponse = (
        ("=" * maxLenOfFileInfo) + "\n" + str(finalReponse) + ("=" * maxLenOfFileInfo)
    )

    return finalReponse


schema_get_files_contents = types.FunctionDeclaration(
    name="get_files_contents",
    description="Get files contents of the given file as a string, contrained to the working directory",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="The path to the file, from the working directory",
            )
        },
    ),
)
