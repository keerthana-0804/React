import streamlit as st
import subprocess
import json
from sympy import symbols, simplify_logic
from sympy.parsing.sympy_parser import parse_expr
import streamlit as st
from PIL import Image
import subprocess
from subprocess import Popen, PIPE
from sympy.logic.boolalg import And,Or
from sympy.parsing.sympy_parser import parse_expr

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

# Define the symbols used in logical expressions
wood, iron, grass, toolshed, gold, furnace = symbols('wood iron grass toolshed gold furnace')

def evaluate_task(simplified_task, items_list):
    """ Evaluate tasks and update items list based on logical conditions """
    terms = str(simplified_task).split('&')
    for term in terms:
        term = term.strip().replace('(', '').replace(')', '')
        if '|' in term:
            sub_terms = term.split('|')
            found = any(sub_term.strip() in items_list for sub_term in sub_terms)
            if not found:
                items_list.append(sub_terms[0].strip())
        else:
            if term not in items_list:
                items_list.append(term)
    return items_list


# Initialize the session state variables
if 'dqn_clicked' not in st.session_state:
    st.session_state['dqn_clicked'] = False

if 'qlearning_clicked' not in st.session_state:
    st.session_state['qlearning_clicked'] = False




# Page title
st.markdown("<h1 style='text-align: center; color: #333;'>Mira Reinforcement Learning</h1>", unsafe_allow_html=True)

# Buttons for task selection
task_type = st.sidebar.radio("", ["Single Task Env", "Multi-Task Env", "Existing Approaches"])


if task_type == 'Multi-Task Env':
    st.subheader("GridWorld Environment")
    # Streamlit UI components
    st.title("Dynamic Task Processor")

    # Initialize items_list
    items_list = []

    # User selects how many tasks to enter (up to 3)
    num_tasks = st.number_input("Specify the number of tasks?", min_value=1, max_value=3, step=1, value=1)

    # Generate input fields dynamically
    tasks = [st.text_input(f"Enter Task {i+1}") for i in range(int(num_tasks))]

    if st.button('Process Tasks'):
        new_items_list = []
        for task in tasks:
            # Parse and evaluate each task
            parsed_task = parse_expr(task, local_dict=globals())
            simplified_task = simplify_logic(parsed_task)
            new_items_list = evaluate_task(simplified_task, new_items_list)
        
        # Prepare the final task list
        st.write("THE LIST IS:",new_items_list)
        #items_list = [item for item in new_items_list if item not in ['toolshed', 'gold','furnace']]
        #items_list.extend(['toolshed', 'gold'])

        def shift_dest_to_end(lst, dest_places):
            dest_places_set = set(dest_places)
            without_dest = [item for item in lst if item not in dest_places_set]
            with_dest = [item for item in lst if item in dest_places_set]
            return without_dest + with_dest

        places = new_items_list
        dest_places = ["gold", "furnace","toolshed"]

        items_list = shift_dest_to_end(places, dest_places)
        st.write("ILIST OVER--------")




        # Display the final tasks
        st.write("Final TASKS to be done:", items_list)

        # Serialize items_list to JSON string
        items_list_json = json.dumps(items_list)

        
        # Run the subprocess and capture the output
        result = subprocess.run(["python", "mainsample.py", "--items_list", items_list_json],
                        capture_output=True, text=True)

        # Check if the subprocess was successful
        if result.returncode == 0:
            # Display the output
            st.text("Output from the script:")
            st.code(result.stdout)  # Use st.code for preformatted text like output
        else:
            # Display the error
            st.error("Error running script")
            st.code(result.stderr)
    if st.button("📊  Display Graph"):
        graph_image = Image.open("miragraph.png")
        st.image(graph_image, caption='Grid World Environment Graph', use_column_width=True)
        st.write("Graph is displayed")     

    # Footer or additional information
    st.markdown("*Ensure tasks are correctly formatted for symbolic logic processing.*")

elif task_type == 'Single Task Env':
    st.sidebar.write("Frozenlake Environment.")
    st.title("Gym Environments")
    st.image("img/env.png")
    st.subheader("Single Task")
    st.write("Choose from the following options:")

    # Check if the button is clicked and if the script is already running
    if st.button("💻 Display Environment"):
        # Show a message that the script is running
        with st.spinner('Running the script... Please wait.'):
            result = subprocess.run(["python", "execbestpath.py"], capture_output=True, text=True)

            # Check if the subprocess was successful
            if result.returncode == 0:
                # Display the output
                st.success("Script executed successfully!")
                st.code(result.stdout)  # Use st.code for preformatted text like output
            else:
                # Display the error
                st.error("Error running script")
                st.code(result.stderr)

    if st.button("📊 Display Graph"):
        st.image("frozenlakegraph.png")
        st.write("Graph is displayed")
elif task_type == 'Existing Approaches':
    st.sidebar.write("Frozenlake Environment.")
    st.title("Gym Environments")
    st.image("img/env.png")
    st.image("cliffwalk.png")
    st.write("Choose from the following options:")

    # Toggle button for DQN
    if st.button("DQN (ST)"):
        st.session_state.dqn_clicked = not st.session_state.dqn_clicked

    # If DQN is clicked, show another button and handle subprocess
    if st.session_state.dqn_clicked:
        if st.button("💻 Show Environment for DQN"):
            with st.spinner('Running the script... Please wait.'):
                result = subprocess.run(["python", "frozenlake_dqn.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Script executed successfully!")
                    st.code(result.stdout)
                else:
                    st.error("Error running script")
                    st.code(result.stderr)

    # Toggle button for QLearning
    if st.button("QLearning"):
        st.session_state.qlearning_clicked = not st.session_state.qlearning_clicked

    # If QLearning is clicked, show another button and handle subprocess
    if st.session_state.qlearning_clicked:
        if st.button("💻 Show Environment for QLearning"):
            with st.spinner('Running the script... Please wait.'):
                result = subprocess.run(["python", "qlearning.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Script executed successfully!")
                    st.code(result.stdout)
                else:
                    st.error("Error running script")
                    st.code(result.stderr)

    st.sidebar.write("This would handle an existing methods and env configurations accordingly.")
    


# Function to display additional navigation options
def display_navigation_options():
    st.sidebar.subheader("Navigation")

    st.markdown("[Load more environments](https://gymnasium.farama.org/)")

        # Displaying a text with an embedded link
    st.markdown("[Customise environment](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)")

        
# Display additional navigation options
display_navigation_options()
# Main area of Streamlit, potentially displaying outputs or additional information
st.write(f"Selected Task Type: {task_type}")
