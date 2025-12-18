import os
import sys
from dotenv import load_dotenv

from google import genai
from google.genai import types
from typing import Any

load_dotenv()

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

    messages = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    # print(f"messages: {messages}")

    if len(arguments) > 1 and len(arguments[1]) > 0:
        prompt = arguments[1]

    verboseFlag = False
    if len(arguments) > 2 and arguments[2] == "--verbose":
        verboseFlag = True

    print(f"PROMPT: {prompt}")

    TIME_TO_MAKE_CALL = 0
    if len(arguments) > 3 and arguments[3] == "1":
        TIME_TO_MAKE_CALL = 1

    print(f"TIME_TO_MAKE_CALL: {TIME_TO_MAKE_CALL}")
    if TIME_TO_MAKE_CALL:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=False,
                    )
                ),
            )
            print("------------------------------------")
            print(response.text)

            if response is None or response.usage_metadata is None:
                print(f"Reponse is Malformed!!")
            else:
                if verboseFlag:
                    print("------------------------------------")
                    print(f"User   prompt: {prompt}")
                    print("------------------------------------")
                    print(
                        f"Prompt   tokens: {response.usage_metadata.prompt_token_count}"
                    )
                    print(
                        f"Response tokens: {response.usage_metadata.candidates_token_count}"
                    )
            print("------------------------------------")

        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Skipping Call!")


main()
