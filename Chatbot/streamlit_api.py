
import streamlit as st
from chatbot import GraphForHumanInTheLoop,QueryStateForString,LLMOutputOnlyString
# This class handles running the chatbot and keeping track of user decisions (like rejection or approval)
class Execuetion:
    # When we create an Execuetion object, we give it the user's input, the number of rejections, and whether the user approved the result
    def __init__(self, user_input, rejection_count=0, human_approval=None):

        # user's message
        self.user_input = user_input
        # total rejections per response
        self.rejection_count = rejection_count
        # to aprove human response
        self.human_approval = human_approval

        # graph execution
        self.g = GraphForHumanInTheLoop(state=QueryStateForString, opstate=LLMOutputOnlyString)
    
    # function for getting the result
    def get_chatbot_output(self):
        try:
            return self.g.execuete_graph(user_query=self.user_input)
        except Exception as e:
            return {"output": str(e)}

# Streamlit App Handler class that manages the chatbot interface
class HITLChatbotApp:
    def __init__(self):
        # Check if the chatbot's state exists in memory (session). If not, set it to None.
        # This keeps our data safe between button clicks (like user input, responses, etc.)
        if 'execution' not in st.session_state:
            st.session_state.execution = None  # This will store the Execuetion object

        if 'chatbot_result' not in st.session_state:
            st.session_state.chatbot_result = None  # This stores the current chatbot reply

        if 'final_output' not in st.session_state:
            st.session_state.final_output = None  # This will store the final accepted answer

    # Called when the user first submits their question
    def initialize_execution(self, user_input):
        # Create a new chatbot runner with the user's question
        st.session_state.execution = Execuetion(user_input=user_input)

        # Generate the chatbot's first response using the input
        st.session_state.chatbot_result = st.session_state.execution.get_chatbot_output()

        # Clear any previous final output
        st.session_state.final_output = None

    # This function shows the chatbot's response and Accept/Reject buttons
    def show_chatbot_response(self):
        st.subheader("Chatbot Response:")
        st.write(st.session_state.chatbot_result['output'])  # Display the current response

        # Two columns for Accept and Reject buttons side-by-side
        col1, col2 = st.columns(2)

        # If the user clicks "Accept"
        with col1:
            if st.button("Accept"):
                self.handle_accept()

        # If the user clicks "Reject"
        with col2:
            if st.button("Reject"):
                self.handle_reject()

    # If the user accepts the response, store it as the final output
    def handle_accept(self):
        st.session_state.execution.human_approval = 'yes'  # Mark it as approved

        # Save the final result along with the original input and rejection count
        st.session_state.final_output = {
            "query": st.session_state.execution.user_input,
            "reply": st.session_state.chatbot_result['output'],
            "total_rejections": st.session_state.execution.rejection_count
        }

    # If the user rejects the response, regenerate a new one
    def handle_reject(self):
        # Increase the number of times user has rejected
        st.session_state.execution.rejection_count += 1

        # Re-create the chatbot logic (fresh graph for another try)
        st.session_state.execution.g = GraphForHumanInTheLoop(
            state=QueryStateForString, opstate=LLMOutputOnlyString
        )

        # Generate a new response with the same original input
        st.session_state.chatbot_result = st.session_state.execution.get_chatbot_output()

    # If the user accepted a response, show it as the final answer
    def show_final_output(self):
        st.success("Final approved response:")
        st.session_state.final_output['reply']  # Display the final accepted reply

    # Main function to run the whole app
    def run(self):
        st.title("Human-in-the-Loop Chatbot")  # App title at the top

        # Textbox for user to enter their question
        user_input = st.text_input("Enter your question:")

        # If user clicks Submit and entered a question
        if st.button("Submit") and user_input:
            self.initialize_execution(user_input)  # Start a new chatbot execution

        # If there's a chatbot response but no final approval yet, show the Accept/Reject interface
        if st.session_state.chatbot_result and not st.session_state.final_output:
            self.show_chatbot_response()

        # If a final response has been accepted, display it
        if st.session_state.final_output:
            self.show_final_output()


# Run the app
if __name__ == "__main__":
    app = HITLChatbotApp()
    app.run()