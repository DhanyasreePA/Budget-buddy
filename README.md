BUDGET BUDDY: A Multi-Agent System for Proactive Financial Coaching
Budget Buddy is an intelligent, AI-driven personal finance assistant designed to help users take control of their spending, save consistently, and achieve their financial goals. Built on a multi-agent architecture and powered by Google Gemini 2.5 Flash, it moves beyond static budgeting tools to offer real-time analytics and personalized, proactive financial advice.

Problem Statement:
Managing personal finances is challenging. 
Many people:
Forget where their money goes and struggle to track expenses.
Overspend without realizing it until the end of the month.
Lack personalized advice to build better saving habits.
Traditional budgeting apps often require tedious manual tracking and offer generic advice. Budget Buddy solves this by combining finance tracking with context-aware AI support.

Architecture Overview
The system is built on a layered architecture that separates data management from the AI decision-making process.
1. Budget Management Layer (Data Handling)
Income System: Tracks monthly income, extra income, and calculates carryover and remaining amounts.
Expense System: Records date-wise and categorized spending (Food, Bills, Transport) and handles automatic month detection.

2. AI Layer (Gemini Agent)
Receives user prompts via Streamlit UI.
Context Injection: Gemini receives the latest income table, full expense history, and goal progress percentages.
Function: Generates tailored financial recommendations (e.g., "Am I overspending on food?" → Agent reviews the 'Food' category and suggests specific reductions).

3. Savings Goal Tracking
Tracks goal name, target amount, and saved amount (e.g., "Vacation Fund").
Auto-calculates progress percentage and visualizes it (Pie Chart).

4. Visualization Layer
Plotly Express is used to create:
Bar charts (Income vs. Remaining Budget).
Pie charts (Goal Progress).

Tools & Technologies Used
Frontend / App Framework: Streamlit (Clean UI, Sidebar inputs, Live reload).
Data / Analytics: Pandas (Load/transform CSVs, generate summaries).
Visualization: Plotly Express (Interactive Bar & Pie Charts).
AI Engine: Google Gemini 2.5 Flash (For natural language financial advice and proactive checks).
Utilities: dotenv (API Key Management), datetime (Date Formatting).

Future Enhancements :
Bank API Integration: Move from manual to automatic tracking of income and expenses.
Advanced Multi-Agent System: Implement specialized agents like a Goal Strategist Agent, Spending Analyzer Agent, and Investment Suggestion Agent.
Predictive Expense Forecasting: Use ML models to forecast next month's expenses and savings potential.
Mobile App & Notifications: Build a smartphone version with push notifications for overspending alerts and goal reminders.

Data tables (Income, Expenses, Savings).
