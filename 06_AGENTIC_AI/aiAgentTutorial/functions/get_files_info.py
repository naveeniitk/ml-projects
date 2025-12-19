import os


def get_files_info(
    working_directory: str,
    directory: str = ".",
) -> str:
    # print(f"Fetching files info from the CWD!")
    """
    Args:
        working_directory (str)
        directory (str): Defaults to "."

    Returns:
        str: file contents information
    """
    absWorkingDirectory = os.path.abspath(working_directory)
    absDirectory = os.path.abspath(directory)

    if directory is None:
        absDirectory = absWorkingDirectory
    else:
        absDirectory = os.path.abspath(os.path.join(absWorkingDirectory, directory))

    maxLenOfFileInfo = 0
    finalReponse = ""

    if not absDirectory.startswith(absWorkingDirectory):
        maxLenOfFileInfo = 33
        finalReponse = f"Error: {directory} is not a directory\n"
    else:
        listOfFiles = os.listdir(absDirectory)

        for file in listOfFiles:
            directoryPath = os.path.join(absDirectory, file)
            isDirectory = os.path.isdir(directoryPath)
            fileSize = os.path.getsize(directoryPath)
            fileInfo = (
                f"[{file}]: isDirectory = {isDirectory}, size = {fileSize} bytes\n"
            )
            maxLenOfFileInfo = max(maxLenOfFileInfo, len(fileInfo))
            finalReponse += fileInfo

    finalReponse = (
        ("=" * maxLenOfFileInfo) + "\n" + str(finalReponse) + ("=" * maxLenOfFileInfo)
    )

    return finalReponse


# print(get_files_info("../../"))
