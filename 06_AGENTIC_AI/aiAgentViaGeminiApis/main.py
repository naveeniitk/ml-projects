import sys
import os
from google.genai import types
from google import genai
from functions.write_file import schema_write_file
from functions.run_python_file import schema_run_python_file
from functions.get_files_info import schema_get_files_info
from functions.get_files_contents import schema_get_files_contents
from call_function import call_function
from dotenv import load_dotenv


load_dotenv()
SEPARATOR = "------------------------------------"

arguments: list = sys.argv
print(f"Args: {arguments}")


def main():

    API_KEY = os.environ.get("API_KEY")
    client = genai.Client(api_key=API_KEY)

    models = client.models.list()
    print("Available Models: ", len(models))

    modelNames: list = [model.name for model in models]
    # print(f"modelNames: {modelNames}")
    response = None
    prompt: str = "Give a Dummy Error message of No PROMPT!!"
    systemPrompt: str = """
    You are a helpful AI coding agent.
    
    When a user asks a question or makes a request, make a function call plan. You can perform the following operations if needed:
    
    - list files and directories
    - Read the content of a file
    - write to a file (create or update)
    - Run a Python file with optional arguments
    
    All paths you provide should be relative to the working directory. You don't need to provide working directory in the function call, as it is ingested automatically for security purposes.
    """

    availableFunctions = types.Tool(
        function_declarations=[
            schema_get_files_info,
            schema_get_files_contents,
            schema_run_python_file,
            schema_write_file,
        ],
    )

    if len(arguments) > 1 and len(arguments[1]) > 0:
        prompt = arguments[1]

    verboseFlag = False
    if len(arguments) > 2 and arguments[2] == "--verbose":
        verboseFlag = True

    print(f"PROMPT: {prompt}")

    messages = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    # print(f"messages: {messages}")

    TIME_TO_MAKE_CALL = 0
    if len(arguments) > 3 and arguments[3] == "1":
        TIME_TO_MAKE_CALL = 1

    print(f"TIME_TO_MAKE_CALL: {TIME_TO_MAKE_CALL}")
    if TIME_TO_MAKE_CALL:
        try:

            ITERATION = 20

            config = types.GenerateContentConfig(
                tools=[availableFunctions],
                system_instruction=systemPrompt,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=False,
                ),
            )
            for i in range(ITERATION):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=messages,
                    config=config,
                )

                if response is None or response.usage_metadata is None:
                    print(SEPARATOR)
                    print(f"Reponse is Malformed!!")
                    print(SEPARATOR)
                    return

                if verboseFlag:
                    print(SEPARATOR)
                    print(f"User   prompt: {prompt}")
                    print(SEPARATOR)
                    print(
                        f"Prompt   tokens: {response.usage_metadata.prompt_token_count}"
                    )
                    print(
                        f"Response tokens: {response.usage_metadata.candidates_token_count}"
                    )
                    print(SEPARATOR)

                if response.candidates:
                    for candidate in response.candidates:
                        if (candidate is None) or (candidate.content is None):
                            continue
                        messages.append(candidate.content)

                if response.function_calls:
                    for functionCallPart in response.function_calls:
                        result = call_function(
                            function_call_part=functionCallPart,
                            verbose=verboseFlag,
                        )
                        messages.append(result)
                else:
                    print(SEPARATOR)
                    print(response.text)
                    print(SEPARATOR)
                    return

        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Skipping Call!")


main()
