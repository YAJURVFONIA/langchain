# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# llm = ChatOpenAI(model = "gpt-3.5-turbo")
llm = ChatGroq(model="llama-3.3-70b-versatile")
result = llm.invoke("Tell me the name of the most popular rapper")
print(result.content)
