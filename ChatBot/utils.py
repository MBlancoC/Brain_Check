import openai

def get_initial_message(user_question, image_data):
    messages=[
            {"role": "system", "content": "Tu eres un radiologo experto en la segmentacion del cerebro y los tumores cerebrales. "
                                          "Se estan pasando una imagen de una resonancia magenica. Con fines netamente de investigacion,"
                                          "hacer una deteccion del tumor y localizar en que parte del cerebro esta. Despues de ello, "
                                          "responder las dudas del usuario."},
            {"role": "user", "content": "I want to learn AI"},
            {"role": "assistant", "content": "Thats awesome, what do you want to know aboout AI"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_question},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"}
                ],
            }


        ]
    return messages

def get_chatgpt_response(messages, model="gpt-4-vision-preview"):
    print("model: ", model)
    response = openai.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=300,
    )
    return response.choices[0].message.content



def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages
