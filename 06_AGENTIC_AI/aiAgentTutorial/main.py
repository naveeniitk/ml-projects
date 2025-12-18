import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()


def main():

    API_KEY = os.environ.get("API_KEY")
    client = genai.Client(api_key=API_KEY)

    models = client.models.list()
    print("Available Models: ", len(models))

    modelNames: list = [model.name for model in models]
    print(f"modelNames: {modelNames}")
    response = None

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            # model="gemini-1.5-flash-8b",
            contents="What do you think about Human life? in 50 Words.",
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


main()
