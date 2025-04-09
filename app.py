import streamlit as st
import pandas as pd
import google.generativeai as genai
import textwrap

# Configure Gemini API key from Streamlit secrets
try:
    genai.configure(api_key=st.secrets["gemini_api_key"])
except Exception as e:
    st.error("Missing or invalid API key. Set gemini_api_key in Streamlit secrets.")
    st.stop()

# Load data
try:
    transaction_df = pd.read_csv("transactions.csv")
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

# Get user question
question = st.chat_input("Type your question about the dataset here...")

# Run if user enters a question
if question:
    # Build prompt for Gemini
    prompt = f"""
    You are a helpful Python code generator.
    Your goal is to write Python code snippets based on the user's question and the provided DataFrame information.

    **User Question:**
    {question}
    
    **DataFrame Name:**
    {df_name}

    **DataFrame Details:**
    {data_dict_text}

    **Sample Data (Top 2 Rows):**
    {example_record}

    Write Python code that stores the answer in a variable called ANSWER.
    Do not import pandas or reload the CSV. Assume the DataFrame is already loaded.
    """

    # Display the question
    st.chat_message("user").markdown(question)

    try:
        response = model.generate_content(prompt)
        code = response.text

        # Display Gemini's response as code
        with st.chat_message("assistant"):
            st.markdown("### Generated Code:")
            st.code(code, language='python')

        # Run the code safely
        local_scope = {"transaction_df": transaction_df}
        exec(code, {}, local_scope)

        # Display the answer if defined
        if "ANSWER" in local_scope:
            st.markdown("### Result:")
            st.write(local_scope["ANSWER"])
        else:
            st.warning("The model did not define an ANSWER variable.")

    except Exception as e:
        st.error(f"An error occurred while processing: {e}")
