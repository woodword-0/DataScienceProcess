import streamlit as st
import pandas as pd
import numpy as np

# Set dark theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #222222;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def build_schedule_input():
    st.header("Input Al's Schedule")
    al_schedule = pd.DataFrame(columns=["Day of Week", "Time"])
    add_schedule_button = st.button("Add Al's Schedule")

    if add_schedule_button:
        new_row = pd.DataFrame({"Day of Week": "", "Time": ""}, index=[0])
        al_schedule = pd.concat([al_schedule, new_row], ignore_index=True)

    return al_schedule


def build_client_schedule_input():
    st.header("Input Client's Schedule")
    client_schedule = pd.DataFrame(columns=["Name", "Day of Week", "Time"])
    add_schedule_button = st.button("Add Client's Schedule")

    if add_schedule_button:
        new_row = pd.DataFrame({"Name": "", "Day of Week": "", "Time": ""}, index=[0])
        client_schedule = pd.concat([client_schedule, new_row], ignore_index=True)

    return client_schedule


def display_schedule(schedule):
    st.header("Schedule")
    if schedule.empty:
        st.write("No schedule available.")
    else:
        # Highlight conflicting schedules in red
        styled_schedule = schedule.copy()
        conflicts = schedule.duplicated(subset=["Day of Week", "Time"], keep=False)
        styled_schedule.loc[conflicts, :] = "background-color: red"

        # Display the schedule table
        st.dataframe(styled_schedule, height=200)


def main():
    st.title("Schedule App")

    # Build Al's Schedule input form
    al_schedule = build_schedule_input()

    # Build Client's Schedule input form
    client_schedule = build_client_schedule_input()

    # Merge Al's and Client's schedules
    schedule = pd.concat([al_schedule, client_schedule])

    # Display the combined schedule
    display_schedule(schedule)


if __name__ == "__main__":
    main()
