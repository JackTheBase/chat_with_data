import streamlit as st
import pandas as pd
import google.generativeai as genai
import textwrap
import re
import matplotlib.pyplot as plt

# Configure Gemini API key from Streamlit secrets
try:
    genai.configure(api_key=st.secrets["gemini_api_key"])
except Exception as e:
    st.error("Missing or invalid API key. Set gemini_api_key in Streamlit secrets.")
    st.stop()

# Load data
try:
    transaction_df = pd.read_csv("transactions.csv")
    transaction_df['date'] = pd.to_datetime(transaction_df['date'])
    data_dict_df = pd.read_csv("data_dict.csv")
except Exception as e:
    st.error(f"Failed to load CSV files: {e}")
    st.stop()

# Set up the LLM model
model = genai.GenerativeModel("gemini-2.0-flash-lite")

st.title("Gemini Chatbot with CSV Access")
st.write("Ask anything! About the world or the uploaded data.")

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Get user question
question = st.chat_input("Type your question here...")

# Run if user enters a question
if question:
    st.chat_message("user").markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Build prompt with memory and context
    context = f"""
    You are a helpful assistant.

    You have access to a pandas DataFrame called `transaction_df` with the following schema:
    {data_dict_df.to_string(index=False)}

    Sample rows:
    {transaction_df.head(2).to_string()}

    If the user's question can be answered using this dataset, write Python code to compute the answer and assign it to a variable named `ANSWER`.
    You may optionally assign a matplotlib chart to `CHART`.

    If the user's question cannot be answered from this dataset, reply with a friendly message saying:
    "I'm here to help with this dataset. Unfortunately, that question isn't answerable using the available data."

    Do NOT hallucinate or assume data that isn't there. ONLY answer if the required data is present in `transaction_df`.

    User chat history:
    {chr(10).join([f"User: {m['content']}" if m['role']=='user' else f"Assistant: {m['content']}" for m in st.session_state.chat_history])}
    """

    try:
        response = model.generate_content(context)
        code = response.text.strip()

        if "ANSWER" in code:
            code = re.sub(r"```(?:python)?", "", code).strip()
            local_scope = {"transaction_df": transaction_df, "pd": pd, "plt": plt}
            exec(code, {}, local_scope)

            if "ANSWER" in local_scope:
                answer = local_scope["ANSWER"]
                response_followup = model.generate_content(
                    f"The user asked: {question}\nHere is the result: {answer}\nRespond to the user with a helpful, friendly answer based strictly on the dataset."
                )
                st.chat_message("assistant").markdown(response_followup.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_followup.text})

                if "CHART" in local_scope:
                    st.pyplot(local_scope["CHART"])
            else:
                st.warning("No ANSWER variable returned.")
        else:
            st.chat_message("assistant").markdown(response.text)
            st.session_state.chat_history.append({"role": "assistant", "content": response.text})

    except Exception as e:
        st.error(f"An error occurred: {e}")
