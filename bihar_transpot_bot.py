import google.generativeai as genai
import pandas as pd
import io
import re

# --- Configuration ---
API_KEY = "AIzaSyAmZYcekqrUq3B5lFBBnyHXaeVXR6bOios" 
DATA_FILE_PATH = "bihar_transport_cleaned_merged_2.csv"
MODEL_NAME = "gemini-2.5-flash" 
SYSTEM_INSTRUCTION = (
    "You are an expert on Bihar transport statistics. "
    "Answer questions strictly using the provided 'Context Data'. "
    "If the dataset does not contain the answer, clearly say that."
)

# --- 1. Data Loading and Reshaping ---
try:
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print(f"✅ Data loaded successfully from: {DATA_FILE_PATH}")

    # --- Step 1: Identify newly and total columns ---
    newly_cols = [c for c in df.columns if 'newly_registered' in c]
    total_cols = [c for c in df.columns if 'total_registered' in c]

    # --- Step 2: Reshape to long format (wide → long) ---
    newly = df.melt(
        id_vars=['vehicles_category', 'type_of_vehicles', 'type'],
        value_vars=newly_cols,
        var_name='year_type',
        value_name='newly_registered'
    )
    total = df.melt(
        id_vars=['vehicles_category', 'type_of_vehicles', 'type'],
        value_vars=total_cols,
        var_name='year_type',
        value_name='total_registered'
    )

    # --- Step 3: Extract years ---
    newly['year'] = newly['year_type'].str.extract(r'(\d{4})').astype(int)
    total['year'] = total['year_type'].str.extract(r'(\d{4})').astype(int)

    # --- Step 4: Merge both ---
    df_long = pd.merge(
        newly.drop(columns=['year_type']),
        total.drop(columns=['year_type']),
        on=['vehicles_category', 'type_of_vehicles', 'type', 'year'],
        how='outer'
    )

    # --- Step 5: Clean up ---
    df_long = df_long.fillna(0)
    print("✅ Data reshaped to long format for easier year-based queries.")

except Exception as e:
    print(f"❌ Error loading or reshaping data: {e}")
    exit()


# --- 2. Retrieval-Augmented Context Function ---
def get_data_context(prompt: str) -> str:
    """Retrieve relevant rows based on year or range of years."""
    prompt_lower = prompt.lower()
    years = re.findall(r'\b(20\d{2})\b', prompt_lower)
    
    # --- Handle year range like "2006 to 2010" ---
    year_range_match = re.search(r'(\d{4})\s*(?:-|to)\s*(\d{4})', prompt_lower)
    if year_range_match:
        start_year, end_year = map(int, year_range_match.groups())
        relevant_df = df_long[(df_long['year'] >= start_year) & (df_long['year'] <= end_year)]
    elif years:
        year = int(years[0])
        relevant_df = df_long[df_long['year'] == year]
    else:
        relevant_df = df_long.copy()

    if 'newly' in prompt_lower:
        value_col = 'newly_registered'
    elif 'total' in prompt_lower:
        value_col = 'total_registered'
    else:
        value_col = 'newly_registered'

    if relevant_df.empty:
        return "No matching year data found."

    # --- Aggregate total for simplicity ---
    summary = relevant_df.groupby('year')[value_col].sum().reset_index()
    summary_str = summary.to_markdown(index=False)

    # Limit output for readability
    sample_df = relevant_df.head(20)[['vehicles_category', 'type_of_vehicles', 'type', 'year', 'newly_registered', 'total_registered']]
    sample_str = sample_df.to_markdown(index=False)

    return f"### Summary by Year:\n{summary_str}\n\n### Sample Data:\n{sample_str}"


# --- 3. Chatbot Initialization ---
genai.configure(api_key=API_KEY)
chatbot = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION
)

print("\n🤖 Chatbot ready! Ask me about Bihar transport data (e.g., 'What were the newly registered two wheelers in 2003-04?')")
print("Type 'exit' or 'quit' to end the session.\n")

# --- 4. Main Chat Loop ---
while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    try:
        data_context = get_data_context(prompt)
        augmented_prompt = f"User Question: {prompt}\n\nContext Data:\n{data_context}"

        response = chatbot.generate_content(augmented_prompt)
        print("\nBot:\n", response.text, "\n")

    except Exception as e:
        print(f"\n⚠️ Error during content generation: {e}\n")
