import pandas as pd
import streamlit as st
import time
# Load the predicted sales data
sales_pred = pd.read_excel('predicted_sales.xlsx', index_col=0)
# Create a line chart to show the sales prediction over time
chart = st.line_chart(sales_pred)
# Create a progress bar to simulate the sales prediction animation
progress_bar = st.progress(0)
status_text = st.empty()

for i in range(len(sales_pred)):
    # Update progress bar
    progress_bar.progress((i + 1) / len(sales_pred))

    # Get the latest predicted sales value
    latest_sales = sales_pred.iloc[i, 0]

    # Update status text
    status_text.text(f'The latest predicted sales value is: {latest_sales}')
    # chart.add_rows(latest_sales)

    # Wait for a short time to simulate computation
    time.sleep(0.1)

# Show a message when the animation is done
st.success('Sales prediction animation is done!')