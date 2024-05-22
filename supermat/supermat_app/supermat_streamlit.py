import streamlit as st
import requests


def call_web_api(user_question):
    # import pdb;pdb.set_trace()
    api_url = "http://127.0.0.1:8000/supermat/upload-file/"
    params = {
        "query": user_question,
    }
    try:
        response = requests.get(api_url, params=params)
        # print(response.json())
        if response.status_code == 201:
            return response.json()["res_data"]
        else:
            return f"Request failed with status code {response.status_code}"
    except Exception as e:
        return f"An error occurred: {e}"


st.set_page_config(
    page_title="Supermat", layout="wide", initial_sidebar_state="auto"
)

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = ""
if "questions" not in st.session_state:
    st.session_state["questions"] = []

st.title("Supermat")

if st.button("New Thread"):
    st.session_state["thread_id"] = ""
    st.session_state["questions"] = []

res_txt = ""
cont1 = st.container()
with cont1:
    col1, col2 = st.columns([5, 1])

prompt = cont1.chat_input("Type your question here!")

if prompt:
    res_txt = call_web_api(prompt)
    st.session_state["questions"].append((prompt, res_txt))
st.divider()
for x, question in enumerate(st.session_state["questions"]):
    if question[0] != "":
        col1.write(f"**Question:** {question[0]}")
        col1.write(f":green[_Response_]: {question[1]}")
