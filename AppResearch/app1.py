import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ... (Other code remains the same) ...

def main():
    st.title("Schedule App")

    # Build first person's schedule input form
    person1_name = st.text_input("Input Person 1's Name")
    person1_schedule = build_schedule_input(person1_name)
    st.markdown("<hr class='als-schedule'>", unsafe_allow_html=True)

    # Build second person's schedule input form
    person2_name = st.text_input("Input Person 2's Name")
    person2_schedule = build_schedule_input(person2_name)

    # Display the schedules in two comparison tables
    st.header("Person 1's Schedule")
    display_schedule(person1_schedule)

    st.header("Person 2's Schedule")
    display_schedule(person2_schedule)


if __name__ == "__main__":
    main()
