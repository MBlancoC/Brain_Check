from openai import OpenAI
import Backend.parametros as p
from PIL import Image
import io
import base64

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


# Subida de múltiples archivos para GPT-4

def upload_multiple_files(uploaded_files, question):
    image_data_list = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data_list.append(image_data)

    if image_data_list and question:
        return analyze_image_with_gpt4(image_data_list, question)
    else:
        return None
