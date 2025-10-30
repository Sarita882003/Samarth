import streamlit as st
import pandas as pd
import google.generativeai as genai
import re
import time

# ---------------- CONFIGURATION ----------------
API_KEY = "AIzaSyAmZYcekqrUq3B5lFBBnyHXaeVXR6bOios"
DATA_FILE_PATH = "bihar_transport_cleaned_merged_2.csv"
MODEL_NAME = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = (
    "You are an expert on Bihar transport statistics. "
    "Answer questions strictly using the provided 'Context Data'. "
    "If the dataset does not contain the answer, clearly say that."
)

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Data Query Chatbot", page_icon="💬", layout="centered")

st.markdown("""
<style>
body {
    background-color: #f7f9fc;
    font-family: 'Open Sans', sans-serif;
}
h2, h3 {
    color: #003366;
    text-align: center;
    font-weight: 700;
}
label, span, div, p {
    color: #333333 !important;
    font-family: 'Open Sans', sans-serif;
}
.stButton button {
    background-color: white;
    color: #003366;
    border-radius: 8px;
    padding: 0.6em 1.5em;
    font-weight: 600;
    border: 2px solid #003366;
    transition: all 0.3s ease;
}
.stButton button:hover {
    background-color: #e6f0ff;
    color: #002244;
    border-color: #002244;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2>How may I assist you?</h2>", unsafe_allow_html=True)

# ---------------- STATE & SECTOR FILTERS ----------------
states = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
          "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
          "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
          "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
          "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir", "Ladakh"]

select_all_states = st.checkbox("Select All States")
if select_all_states:
    selected_states = states
else:
    selected_states = st.multiselect("Select State(s):", states, default=["Bihar"])

sectors = ["Transport", "Health and Family Welfare", "Survey/Census", "Economy",
           "Education", "Environment and Forest", "Water and Sanitation",
           "Statistics", "Agriculture", "Finance"]

select_all_sectors = st.checkbox("Select All Sectors")
if select_all_sectors:
    selected_sectors = sectors
else:
    selected_sectors = st.multiselect("Select Sector(s):", sectors, default=["Transport"])

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    newly_cols = [c for c in df.columns if 'newly_registered' in c]
    total_cols = [c for c in df.columns if 'total_registered' in c]

    newly = df.melt(id_vars=['vehicles_category', 'type_of_vehicles', 'type'],
                    value_vars=newly_cols,
                    var_name='year_type', value_name='newly_registered')
    total = df.melt(id_vars=['vehicles_category', 'type_of_vehicles', 'type'],
                    value_vars=total_cols,
                    var_name='year_type', value_name='total_registered')

    newly['year'] = newly['year_type'].str.extract(r'(\d{4})').astype(int)
    total['year'] = total['year_type'].str.extract(r'(\d{4})').astype(int)

    df_long = pd.merge(
        newly.drop(columns=['year_type']),
        total.drop(columns=['year_type']),
        on=['vehicles_category', 'type_of_vehicles', 'type', 'year'],
        how='outer'
    ).fillna(0)
    return df_long

df_long = load_and_prepare_data()

# ---------------- CONTEXT RETRIEVAL FUNCTION ----------------
def get_data_context(prompt: str) -> str:
    prompt_lower = prompt.lower()
    years = re.findall(r'\b(20\d{2})\b', prompt_lower)
    year_range_match = re.search(r'(\d{4})\s*(?:-|to)\s*(\d{4})', prompt_lower)

    if year_range_match:
        start_year, end_year = map(int, year_range_match.groups())
        relevant_df = df_long[(df_long['year'] >= start_year) & (df_long['year'] <= end_year)]
    elif years:
        year = int(years[0])
        relevant_df = df_long[df_long['year'] == year]
    else:
        relevant_df = df_long.copy()

    value_col = 'newly_registered' if 'newly' in prompt_lower else (
        'total_registered' if 'total' in prompt_lower else 'newly_registered'
    )

    if relevant_df.empty:
        return "No matching year data found."

    summary = relevant_df.groupby('year')[value_col].sum().reset_index()
    summary_str = summary.to_markdown(index=False)

    sample_df = relevant_df.head(10)[['vehicles_category', 'type_of_vehicles', 'type', 'year', 'newly_registered', 'total_registered']]
    sample_str = sample_df.to_markdown(index=False)

    return f"### Summary by Year:\n{summary_str}\n\n### Sample Data:\n{sample_str}"

# ---------------- GEMINI MODEL ----------------
genai.configure(api_key=API_KEY)
chatbot = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION
)

# ---------------- USER QUESTION ----------------
st.write("### Now, you can ask your question below:")
user_question = st.text_input("Ask Question:")

# ---------------- SUBMIT ----------------
if st.button("Submit"):
    if not user_question:
        st.warning("⚠️ Please enter a question before submitting.")
    else:
        st.write(f"**Selected States:** {', '.join(selected_states)}")
        st.write(f"**Selected Sectors:** {', '.join(selected_sectors)}")

        with st.spinner("🤖 Thinking..."):
            time.sleep(2)
            try:
                data_context = get_data_context(user_question)
                augmented_prompt = f"User Question: {user_question}\n\nContext Data:\n{data_context}"
                response = chatbot.generate_content(augmented_prompt)
                st.markdown("### 💬 Chatbot Response:")
                st.success(response.text)
            except Exception as e:
                st.error(f"⚠️ Error during content generation: {e}")
