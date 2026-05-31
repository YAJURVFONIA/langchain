# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

chat_history=[] #used to store the messages

# here we stored the history in this simple variable
# but when we are working on production level, we 
# have to store our history in some place safe, so 
# we store it in cloud servers.

system_message = SystemMessage(content = "You are a helpful AI assistant.")
chat_history.append(system_message)

while True:
    query=input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result=model.invoke(chat_history)
    print(result.content)
