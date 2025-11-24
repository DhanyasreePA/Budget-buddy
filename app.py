import os
import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from datetime import date
from dotenv import load_dotenv
from google.generativeai.types import GenerationConfig 

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Use st.warning instead of st.error/st.stop to allow app to run if key is missing,
    # but the AI part will fail (good for testing other parts).
    st.warning("Please set GEMINI_API_KEY in your .env file to enable AI functionality.")

# --- Gemini SDK ---
import google.generativeai as genai
# Configure only if key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    
# --- Logging ---
logging.basicConfig(
    filename="budget_agent.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logging.info("Starting Personal Budget AI Agent")

# --- Gemini AI Helper (ABSOLUTE MINIMUM FIX) ---
def ask_gemini(prompt: str):
    """Generates budget advice using the Gemini API."""
    if not GEMINI_API_KEY:
        return "AI is disabled. Please set the GEMINI_API_KEY in your .env file."
        
    try:
        # Friendly replies for basic greetings
        greetings = {"hi":"Hello! How can I help you save today?",
                     "hello":"Hi there! Need some budget advice?",
                     "bye":"Goodbye! Keep saving!"}
        if prompt.lower() in greetings:
            return greetings[prompt.lower()]

        # AI-generated response using the minimal syntax
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Using generate_content directly with only the prompt, avoiding all keyword conflicts
        response = model.generate_content(prompt)
        
        # Check if response text is available and return it
        if response.text:
            return response.text
        else:
            return "AI Error: Could not generate a response (filtered or empty)."
            
    except Exception as e:
        # Check for authentication errors specifically
        if "API_KEY_INVALID" in str(e):
             logging.error(f"Gemini API Call Failed: Invalid Key")
             return "AI Error: The Gemini API Key appears to be invalid or expired. Please check your .env file."
        
        logging.error(f"Gemini API Call Failed: {e}")
        return f"AI Error: {e}"

# --- Data storage ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INCOME_CSV = os.path.join(DATA_DIR, "income.csv")
EXPENSES_CSV = os.path.join(DATA_DIR, "expenses.csv")
GOALS_CSV = os.path.join(DATA_DIR, "goals.csv")

# --- Data Loading (IMPROVED with st.cache_data) ---
@st.cache_data
def load_df(path, cols):
    """Loads a DataFrame, caches it, and creates an empty one if not found."""
    try:
        # Added parse_dates for proper date handling
        date_cols = ["Date"] if path == EXPENSES_CSV else [] 
        return pd.read_csv(path, parse_dates=date_cols) 
    except FileNotFoundError:
        return pd.DataFrame(columns=cols)

income_df = load_df(INCOME_CSV, ["Month","Income","Extra_Income","Carry_Over","Remaining"])
expenses_df = load_df(EXPENSES_CSV, ["Date","Category","Amount","Month"])
goals_df = load_df(GOALS_CSV, ["Goal","Target_Amount","Saved_Amount"])

# --- Streamlit UI ---
st.set_page_config(page_title=" 💰Personal Budget AI Agent", layout="wide")
st.title("Budget Buddy💰- Personal Budget AI Agent")

# --- Sidebar Inputs ---
st.sidebar.header("💼 Manage Budget & Goals")

# Income Input
st.sidebar.subheader("Set Monthly Income")
month = st.sidebar.selectbox("Month", pd.date_range("2025-01-01", periods=12, freq='ME').strftime("%B"))
income = st.sidebar.number_input("Income", min_value=0.0, step=100.0)
extra = st.sidebar.number_input("Extra Income", min_value=0.0, step=50.0)
if st.sidebar.button("Save Income"):
    carry = 0.0
    if month in income_df["Month"].values:
        # Note: Your original logic used a list for Carry_Over which is wrong if multiple rows existed (which they shouldn't). 
        # Using .iloc[0] for safety.
        current_carry = income_df.loc[income_df["Month"]==month, "Carry_Over"]
        carry = current_carry.iloc[0] if not current_carry.empty else 0.0
        income_df.loc[income_df["Month"]==month, ["Income","Extra_Income","Carry_Over"]] = [income, extra, carry]
    else:
        new_row = pd.DataFrame([[month, income, extra, carry, 0.0]], columns=income_df.columns)
        income_df = pd.concat([income_df, new_row], ignore_index=True)
    
    income_df.to_csv(INCOME_CSV, index=False)
    st.cache_data.clear() # Clear cache to force reload of dataframes
    st.sidebar.success(f"Income updated for {month}")
    st.rerun() # Rerun to update main view

# Expense Input
st.sidebar.subheader("Add Expense")
exp_date = st.sidebar.date_input("Expense Date", date.today())
category = st.sidebar.selectbox("Category", ["Food","Transport","Entertainment","Bills","Others"])
amount = st.sidebar.number_input("Amount", min_value=0.0, step=10.0)
if st.sidebar.button("Add Expense"):
   
    exp_date_str = exp_date.strftime("%Y-%m-%d") 
    exp_month = exp_date.strftime("%B")
    
  
    new_exp = pd.DataFrame([[exp_date_str, category, amount, exp_month]], columns=expenses_df.columns)
    expenses_df = pd.concat([expenses_df, new_exp], ignore_index=True)    
    expenses_df.to_csv(EXPENSES_CSV, index=False)
    
    st.cache_data.clear() 
    st.sidebar.success(f"Added expense ₹{amount} for {category} on {exp_date}")
    st.rerun()

# Goal Input
st.sidebar.subheader("Add Financial Goal")
goal_name = st.sidebar.text_input("Goal Name")
target_amt = st.sidebar.number_input("Target Amount", min_value=0.0, step=100.0)
if st.sidebar.button("Save Goal"):
    new_goal = pd.DataFrame(
    [[goal_name, target_amt, 0.0, target_amt]],
    columns=goals_df.columns)
    goals_df = pd.concat([goals_df, new_goal], ignore_index=True)
    
    goals_df.to_csv(GOALS_CSV, index=False)
    st.cache_data.clear() 
    st.sidebar.success(f"Goal '{goal_name}' saved!")
    st.rerun() 

# --- Main Overview ---
st.header("📊 Budget Overview & Goal Progress")

income_df_calc = load_df(INCOME_CSV, ["Month","Income","Extra_Income","Carry_Over","Remaining"])
expenses_df_calc = load_df(EXPENSES_CSV, ["Date","Category","Amount","Month"])

expenses_df_calc["Date"] = pd.to_datetime(expenses_df_calc["Date"]) 
expenses_df_calc["Month"] = expenses_df_calc["Date"].dt.strftime("%B") 


income_df_calc = income_df_calc.sort_values("Month")
carry_over = 0.0

for i, row in income_df_calc.iterrows():
    month = row["Month"]
    # Ensure we use the correct DataFrame for expense calculation
    spent = expenses_df_calc[expenses_df_calc["Month"]==month]["Amount"].sum()
    
    # Use float for calculations
    total_income = float(row["Income"]) + float(row["Extra_Income"]) + carry_over
    remaining = total_income - spent
    carry_over = max(remaining,0)
    
    income_df_calc.at[i, "Remaining"] = remaining
    income_df_calc.at[i, "Carry_Over"] = carry_over

# Only save if there were changes, and use the calculated one for display
income_df_calc.to_csv(INCOME_CSV, index=False)

# Display tables
st.subheader("Monthly Budget Overview")
st.dataframe(income_df_calc)

# Expense Table
st.subheader("Expenses")
st.dataframe(expenses_df_calc)

# Charts
fig_bar = px.bar(
    income_df_calc,
    x="Month",
    y=["Income","Extra_Income","Remaining"],
    barmode="group",
    title="Monthly Income vs Extra Income vs Remaining",
    width=900
)
st.plotly_chart(fig_bar)

# --- Total Savings & Total Expenses Summary ---
st.subheader("📦 Savings & Spending Summary")

# Total expenses for the year
total_expenses = expenses_df_calc["Amount"].sum()

# Total income + extra income - total expenses
total_income_all_months = income_df_calc["Income"].sum() + income_df_calc["Extra_Income"].sum()

total_savings = total_income_all_months - total_expenses
if total_savings < 0:
    total_savings = 0  # Prevent negative savings display

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="💸 Total Expenses (Year)",
        value=f"₹{total_expenses:,.2f}"
    )

with col2:
    st.metric(
        label="💰 Total Savings (Year)",
        value=f"₹{total_savings:,.2f}"
    )


# Goal Pie Chart
goals_df_calc = load_df(GOALS_CSV, ["Goal","Target_Amount","Saved_Amount"])
goals_df_calc["Progress_Percent"] = (goals_df_calc["Saved_Amount"]/goals_df_calc["Target_Amount"]*100).clip(0,100)
st.subheader("Goals Progress")

# Ensure non-zero targets for pie chart calculation
valid_goals = goals_df_calc[goals_df_calc["Target_Amount"] > 0] 

fig_pie = px.pie(valid_goals, 
                 names="Goal", 
                 values="Target_Amount", # Use Target_Amount for slice size
                 title="Goal Savings Progress (Slice Size by Target Amount)",
                 color_discrete_sequence=px.colors.qualitative.Vivid,
                 hover_data=['Saved_Amount', 'Progress_Percent'])

st.plotly_chart(fig_pie)

# Show goal table
st.dataframe(goals_df_calc)

# --- AI Chatbot ---
st.subheader("💬 AI Advice")
user_input = st.text_input("Ask your AI about saving, goals, or budget")
if user_input:
    # Pass the dataframes as context to the AI (optional but helpful)
    context_prompt = f"""
    User Question: {user_input}
    
    ---
    Current Budget Data:
    Income Overview:
    {income_df_calc.to_markdown(index=False)}
    
    Recent Expenses:
    {expenses_df_calc.tail(10).to_markdown(index=False)}
    
    Financial Goals:
    {goals_df_calc.to_markdown(index=False)}
    ---
    
    Please provide a concise and helpful budget or savings tip based on the data and the user's question. 
    If the question is a general greeting, respond politely.
    """
    
    advice = ask_gemini(context_prompt)
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**AI:** {advice}")