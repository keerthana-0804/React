import streamlit as st

# Define function to execute selected task
def execute_task(task_option):
    if task_option == "Single Task":
        st.write("Executing single task...")
        # Link to single task script file
    elif task_option == "Multi-Task":
        st.write("Executing multi-task...")
        # Link to multi-task script file
    elif task_option == "Existing":
        st.write("Executing existing...")
        # Link to existing script file

# Custom sidebar with styled buttons
st.sidebar.markdown(
    """
    <style>
        .sidebar-content {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar-header {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        }
        .task-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            border: none;
            transition-duration: 0.4s;
        }
        .task-button:hover {
            background-color: #45a049;
        }
    </style>
    """
    , unsafe_allow_html=True
)

st.sidebar.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-header'>Select Task Option</div>", unsafe_allow_html=True)

if st.sidebar.button("Single Task", key="Single Task"):
    execute_task("Single Task")

if st.sidebar.button("Multi-Task", key="Multi-Task"):
    execute_task("Multi-Task")

if st.sidebar.button("Existing Approaches", key="Existing"):
    execute_task("Existing")

st.sidebar.markdown("</div>", unsafe_allow_html=True)
