import streamlit as st
import openai
API_TYPE = st.secrets['API_TYPE']
API_BASE = st.secrets['API_BASE']
API_KEY = st.secrets['API_KEY']
API_VERSION = st.secrets['API_VERSION']
EMBEDDINGS_ENGINE = st.secrets['EMBEDDINGS_ENGINE']
CHAT_ENGINE = st.secrets['CHAT_ENGINE']
# Configuracion
openai.api_type = API_TYPE
openai.api_version = API_VERSION
openai.api_base = API_BASE
openai.api_key = API_KEY
# configuracion del system
system = "Eres un asistente virtual"
system_role = [{"role": "system", "content":system}]
st.title("ChatGPT-like clone")
if "openai_engine" not in st.session_state:
    st.session_state["openai_engine"] = CHAT_ENGINE
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# evalua si el usuario ingreso un prompt y si es verdadero sigue...
if prompt := st.chat_input("Escribe algo para comenzar"):
    # guardar en el historial el input del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    # mostrar por pantalla el input del usuario
    st.chat_message("user").markdown(prompt)
    # consulta al chat y
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = openai.ChatCompletion.create(
            engine=st.session_state["openai_engine"],
            messages= system_role + st.session_state.messages)
        full_response = response.choices[0].message["content"]
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
#    print("historial:", type(st.session_state.messages))
