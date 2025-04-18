import streamlit as st
from transformers import pipeline, Conversation

# Create a pipeline for the model
chatbot = pipeline("conversational", model="facebook/blenderbot-400M-distill")

# Streamlit

# Set Page config
st.set_page_config(
    page_title="Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': 'Portfolio of the author CaptV89: https://captv.ovh/',
        'Report a bug': 'https://github.com/captv89', }
)

# Create a title for your app
st.title("💬 Chatbot")
st.divider()

# Add a short description about author in the sidebar
st.sidebar.header("👨‍💻 by @CaptV89")
st.sidebar.subheader("About the Chatbot")
st.sidebar.caption(
    "Powered by Hugging Face's Blenderbot-400M-distill model. It is a small, fast, open-source, and fully trained conversational model. It is trained on 400 million Facebook public domain conversations.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}]

with st.container():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# Check if the user has sent a message
if prompt := st.chat_input("Say something"):
    # Display the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Create a conversation
    conversation = Conversation(prompt)

    # Generate the chatbot's response to the user's input
    chatbot_response = chatbot(conversation)
    response_msg = chatbot_response.generated_responses[-1]

    # Display the chatbot's response to the user
    st.session_state.messages.append(
        {"role": "assistant", "content": response_msg})
    st.chat_message("assistant").write(response_msg)
