from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)


# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser
# chain = prompt_template | model
# StrOutputParser just gives the content and removes all the other stuff, you can remove it and at the time of printing you can use, result.content, it will do the same thing

# Run the chain
result = chain.invoke({"animal": "elephant", "fact_count": 1})

# Output
print(result)
