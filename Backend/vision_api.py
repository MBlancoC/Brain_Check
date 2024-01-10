from openai import OpenAI

def analyze_image_with_gpt4(image_data, question):
    #pont)_tu_clave_aquí
    client = OpenAI(api_key="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")

    # No es necesario codificar la pregunta ya que OpenAI maneja internamente la codificación UTF-8
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "data": image_data}
                ]
            }
        ],
        max_tokens=300,
    )
    return response
