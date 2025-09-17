import langchain 
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import searchTool
from tools import wiki_tool


#load the env file with the API key variable 
load_dotenv()

llm = ChatOpenAI(model="gpt-4")

tools = [searchTool, wiki_tool]

class Response(BaseModel): 
    topic: str
    summary: str
    sources: list[str]
    tools: list[str]

parser = PydanticOutputParser(pydantic_object=Response)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent= agent, tools = tools, verbose=True)
#verbose=True to see the reasoning process
query = input("what do you want to research? ")
raw_response = agent_executor.invoke({"query": query})
print(raw_response)
try: 
    final_response = parser.parse(raw_response.get("output"))
except Exception as e: print(e)


print(final_response.summary,"\n",final_response.sources, "\n", final_response.tools )

#print(final_response.topic) 


