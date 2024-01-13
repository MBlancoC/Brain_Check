from openai import OpenAI
import Backend.parametros as p
def analyze_image_with_gpt4(image_data_list, question):
    client = OpenAI(api_key=p.OPENAI_KEY)
    image_contents = [ {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"} for image_data in image_data_list]
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}] + image_contents
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=300,
    )
    return response
