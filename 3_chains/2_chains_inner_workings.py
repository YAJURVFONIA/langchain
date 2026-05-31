from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

# Define prompt templates 
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {count} facts."),
    ]
)

# creating individual runnables (steps in the chain)


format_prompt = RunnableLambda( lambda x : prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x:x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
# RunnableSequence is used to chain multiple RunnableLamdas
# this class will always take in three parameter-> first,middle,last
# the first task will contain only one task, same goes with the last task
# but the middle task can hold up any number of task as you want, since it
# can store a list of tasks. All the number of tasks are stored inside a list here.
chain = RunnableSequence(first=format_prompt, middle=[invoke_model],last=parse_output)

#we can either use this RunnableSequence class to chain the Runnable lambda 
#or we can just use the langchain expression language, which is basically
#using the pipeline oeprator "|"


#run the chain
response = chain.invoke({"animal": "cat", "count": 2})


print(response)