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
    You are a helpful assistant who can analyze a DataFrame and also answer general knowledge questions.

    DataFrame name: transaction_df
    DataFrame sample (top 2 rows):
    {transaction_df.head(2).to_string()}

    Data Dictionary:
    {data_dict_df.to_string(index=False)}

    User chat history:
    {chr(10).join([f"User: {m['content']}" if m['role']=='user' else f"Assistant: {m['content']}" for m in st.session_state.chat_history])}

    If the user's query involves the dataset, write Python code to compute the answer and store the final answer in a variable called ANSWER.
    Then respond directly and naturally to the user using that result.
    If applicable, assign any matplotlib figure to a variable named CHART.
    If the user's question is not about the data, reply normally as a chatbot.
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
                    f"The user asked: {question}\nHere is the result: {answer}\nRespond to the user with a helpful, friendly answer."
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
