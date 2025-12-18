import os
import sys
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

arguments: list = sys.argv
print(f"Args: {arguments}")

def main():

    API_KEY = os.environ.get("API_KEY")
    client = genai.Client(api_key=API_KEY)

    models = client.models.list()
    print("Available Models: ", len(models))

    modelNames: list = [model.name for model in models]
    print(f"modelNames: {modelNames}")
    response = None
    prompt = "Give a Error message as this is not the prompt I was trying to give!!",
    
    if len(arguments) > 1 and len(arguments[1]) > 0:
        prompt = arguments[1];
    print(f"PROMPT: {prompt}")

    TIME_TO_MAKE_CALL = 0 # to save free calls
    if TIME_TO_MAKE_CALL:
        print(f"Making call!!!")
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
            print(response.text)

            if response is None or response.usage_metadata is None:
                print(f"Reponse is Malformed!!")
            else:
                print(f"Prompt   tokens: {response.usage_metadata.prompt_token_count}")
                print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Skipping call!!!")


main()
