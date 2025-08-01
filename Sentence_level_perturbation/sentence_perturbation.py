from openai import OpenAI

def sentence_paraphrase(sentences):
    client = OpenAI(api_key="your_api_key",
                    base_url="api_url")

    response = client.chat.completions.create(
        model="model",
        messages=[
            {"role": "user", "content":
                f'''
                    Paraphrase following sentence to harmless without changing the original meaning.
                    1. Only answer the paraphrased sentences.
                    2. Do not include any explanations or additional information.
                     =========Original Sentences Start=========
                     {sentences}
                     =========Original Sentences End=========
                '''},
        ],
        stream=False
    )

    print(f"paraphrased sentences:{response.choices[0].message.content}")
    return response.choices[0].message.content

