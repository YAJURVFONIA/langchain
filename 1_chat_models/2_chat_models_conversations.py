from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile")
messages = [
    SystemMessage("You are an expert in social media content strategy."),
    HumanMessage("Give a short tip to create engaging posts on Instagram"),
    AIMessage("To create engaging posts on Instagram, use a mix of high-quality visuals and storytelling by **starting your caption with a question**. This encourages audience interaction, sparks curiosity, and increases the likelihood of comments and likes on your post."),
    HumanMessage("Give me the list of all the instagram influencer, whose content is related to machine learning.")
]

result = llm.invoke(messages)
print(result.content)
