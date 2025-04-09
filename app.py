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

# Format example data and schema
def build_data_context(df, dict_df):
    example_record = df.head(2).to_string()
    data_dict_text = '\n'.join(
        '- ' + dict_df['column_name'] + ': ' + dict_df['data_type'] + '. ' + dict_df['description']
    )
    return example_record, data_dict_text

example_record, data_dict_text = build_data_context(transaction_df, data_dict_df)
df_name = "transaction_df"

# Set up the LLM model
model = genai.GenerativeModel("gemini-2.0-flash-lite")

st.title("Gemini Data Chatbot")
st.write("Ask a question about the dataset. For example: **How many sales in January 2025?**")

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Get user question
question = st.chat_input("Type your question about the dataset here...")

# Run if user enters a question
if question:
    # Add to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Build prompt with history
    history_text = "\n".join(
        [f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" for msg in st.session_state.chat_history]
    )

    prompt = f"""
    You are a helpful Python code generator.
    Your goal is to write Python code snippets based on the user's question and the provided DataFrame information.

    {history_text}

    **DataFrame Name:**
    {df_name}

    **DataFrame Details:**
    {data_dict_text}

    **Sample Data (Top 2 Rows):**
    {example_record}

    Write Python code that stores the answer in a variable called ANSWER.
    If the result is a DataFrame or Series, also store an optional chart object in a variable named CHART.
    Do not import pandas or reload the CSV. Assume the DataFrame is already loaded.
    Make sure to convert the 'date' column to datetime before filtering by year or month.
    """

    # Display the question
    st.chat_message("user").markdown(question)

    try:
        response = model.generate_content(prompt)
        code = response.text

        # ✅ CLEAN THE CODE (remove Markdown like ```python)
        code = re.sub(r"```(?:python)?", "", code).strip()

        # ✅ BASIC VALIDATION
        if not code.strip().startswith("ANSWER") and "ANSWER" not in code:
            raise ValueError("The generated response does not contain valid Python code for ANSWER.")

        # ✅ EXECUTE the cleaned code
        local_scope = {
            "transaction_df": transaction_df,
            "pd": pd,
            "plt": plt,
        }
        exec(code, {}, local_scope)

        # ✅ DISPLAY THE ANSWER
        if "ANSWER" in local_scope:
            answer_value = local_scope["ANSWER"]

            explain_prompt = f"""
            The user asked: {question}
            Here is the result: {answer_value}
            Answer the question, summarize the result, and provide your interpretation of what this tells us about the user's interest.
            """

            explanation = model.generate_content(explain_prompt)
            explanation_text = explanation.text.strip()

            st.session_state.chat_history.append({"role": "assistant", "content": explanation_text})

            with st.chat_message("assistant"):
                st.markdown("### Answer Summary:")
                st.markdown(explanation_text)

                # Display chart if exists
                if "CHART" in local_scope:
                    st.pyplot(local_scope["CHART"])
        else:
            st.warning("The model did not define an ANSWER variable.")

    except Exception as e:
        st.error(f"An error occurred while processing: {e}")
