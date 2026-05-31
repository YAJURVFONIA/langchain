from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")



positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a sorry you note for this negative feedback: {feedback}."),
        
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a note requesting more details for this neutral feedback: {feedback}."),
        
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a message to escalte this message to a human agent: {feedback}."),
        
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),

    ]
)

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x:"negative" in x,
        negative_feedback_template | model | StrOutputParser()   
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser
)




# Create the combined chain using LangChain Expression Language (LCEL)
classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches


review = "I am so furious right now, how in hell is this jacket so dirty. Even the chain is not working properly, the upper colar is burnt. And in top of it, the whole jacket is reeking."
# review = "The quality of the jacket is so nice, I would highly recommend everyone to buy it. It's worth it guys."
# review = "The product is terrible. It broke after just one use and the quality is very poor."
# Run the chain
result = chain.invoke({"feedback":review})

# Output
print(result)