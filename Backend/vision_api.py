from openai import OpenAI
import Backend.parametros as p
def analyze_image_with_gpt4(image_data, question):
    client = OpenAI(api_key=p.OPENAI_KEY)
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
