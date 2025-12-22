import os


def write_file(
    working_directory: str,
    file_path: str,
    content: str,
) -> None:
    absWorkingDirectory = os.path.abspath(working_directory)
    absFilePath = os.path.abspath(os.path.join(absWorkingDirectory, file_path))
    if not absFilePath.startswith(absWorkingDirectory):
        return f'Error: "{file_path}" is not in the working directory'

    parentDirectory = os.path.dirname(absFilePath)
    print(f"parentDirectory: {parentDirectory}")

    if not os.path.isdir(parentDirectory):
        try:
            os.makedirs(parentDirectory)
        except Exception as e:
            return f'Error: Could not create parents directory for "{file_path}"'

    if not os.path.isfile(absFilePath):
        pass
        # return f'Error: "{file_path}" is not a valid file'

    try:
        with open(absFilePath, "w+") as file:
            file.write(content)
        return (
            f"Content with [len:{len(content)}] written on {absFilePath} successfully!!"
        )
    except Exception as e:
        return f'Error: Could not write to "{file_path}"'
