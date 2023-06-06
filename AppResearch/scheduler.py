import streamlit as st
import pandas as pd

# Set the background color to dark
st.markdown(
    """
    <style>
    body {
        background-color: #333333;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the styles for the active and inactive days
active_day_style = 'background-color: #00ff00; color: #ffffff;'
inactive_day_style = 'background-color: #333333; color: #ffffff;'

# Create a DataFrame to represent the weekly calendar
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
calendar_df = pd.DataFrame(index=range(24), columns=weekdays)

# Initialize all days as inactive
calendar_df[:] = inactive_day_style

# Function to handle day click event
def handle_day_click(day):
    # Toggle the active/inactive style for the clicked day
    if calendar_df.loc[0, day] == active_day_style:
        calendar_df.loc[:, day] = inactive_day_style
    else:
        calendar_df.loc[:, day] = active_day_style

# Render the weekly calendar
for day in weekdays:
    if st.markdown(f'<button>{day}</button>', unsafe_allow_html=True):
        handle_day_click(day)

# Display the calendar DataFrame
st.dataframe(calendar_df)
