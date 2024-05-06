import streamlit as st
from PIL import Image
import subprocess
import matplotlib.pyplot as plt
# Customizing Streamlit page style
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        color: #333;
        font-size: 18px;
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Customizing Streamlit button style
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-right: 10px;
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to display single task options
def display_single_task_options():
    st.subheader("Single Task Options")
    st.write("Choose from the following options:")
    if st.button("📊  Display Graph"):
        st.write("Graph is displayed")
    if st.button(" 💻 Show Environment"):
        st.write("Script is running")

# Function to display multi-task options
def display_multi_task_options():
    st.subheader("Multi-Task Options")
    st.write("Choose from the following options:")
    if st.button("📊  Display Graph"):
        graph_image = Image.open("miragraph.png")
        st.image(graph_image, caption='Grid World Environment Graph', use_column_width=True)
        st.write("Graph is displayed")
    if st.button(" 💻 Show Environment"):
        st.write("Script is running")
        # Call your Python script here
        process = subprocess.run(["python", "iamsamplingfinalprg.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        st.write("Output:")
        st.code(process.stdout)
        st.write("Errors:")
        st.code(process.stderr)
        

# Function to display existing approaches
def display_existing_approaches():
    st.subheader("Existing Approaches")
    st.write("Choose from the following options:")
    st.image("agent.png", width=30)
    if st.button("Script is executing. Click here to run"):
        st.write("Script is executed")

# Page title
st.markdown("<h1 style='text-align: center; color: #333;'>Mira Reinforcement Learning</h1>", unsafe_allow_html=True)

# Buttons for task selection
task = st.sidebar.radio("", ["Single Task", "Multi-Task", "Existing Approaches"])

# Display options based on task selection
if task == "Single Task":
    display_single_task_options()
elif task == "Multi-Task":
    display_multi_task_options()
elif task == "Existing Approaches":
    display_existing_approaches()

# Function to display additional navigation options
def display_navigation_options():
    st.sidebar.subheader("Navigation")

    st.markdown("[Load more environments](https://gymnasium.farama.org/)")

        # Displaying a text with an embedded link
    st.markdown("[Customise environment](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)")

        
# Display additional navigation options
display_navigation_options()