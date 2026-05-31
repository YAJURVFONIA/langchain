from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore

load_dotenv()

PROJECT_ID = "langchain-d1330"

SESSION_ID = "user_session_new" #should be ideally a 16 digit long unique id
COLLECTION_NAME= "chat_history"


print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)


model = ChatGroq(model="llama-3.3-70b-versatile")
print("Start chatting with the AI. Type 'exit' to quit.")
while True:
    human_input=input("User: ")
    if human_input.lower() == "exit":
        break
    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)
    print(f"AI: {ai_response.content}")
