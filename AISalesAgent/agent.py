from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os
import json

load_dotenv()

model = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

server_params = StdioServerParameters(
    command="npx",
    env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")},
    args=["firecrawl-mcp"]
)


async def scraping(businessName: str):
    async with stdio_client(server_params) as (read,write): 
        async with ClientSession(read,write) as session: 
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent= create_react_agent(model,tools)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can scrape websites, crawl pages, and extract data using Firecrawl tools. Think step by step and use the appropriate tools to extract the most relevant data."

                }
            ]
            
            if businessName == "quit" : return "you quit"

            dataExt = "Name: example , website: example.com, industry sector: someSector, email: example@business.com, description: 2 sentences max about the company and its offerings, AI optimization suggestion: find 2-3 processes in this particular business that can be automated by AI to maximize efficiency and productivity"


            messages.append({"role": "user", 
                             "content": f"scrape data from the website of  {businessName} and give me the info in this format {dataExt}"
                            
                              
                             })

            try: 
                print("scraping data from the given website...")
                agent_response = await agent.ainvoke({"messages": messages})
                ai_message = agent_response["messages"][-1].content
                return ai_message
            except Exception as e: print("Error: ",e )


async def writeEmail(data: str, improvements: str = None, prevEmail: str= None): 

    if improvements:
        prompt = f"""rewrite the following sales email {prevEmail} based on these improvement suggestions: {improvements}.
        Keep it professional but friendly, 7 sentences max, no special characters.
        End with a clear call to action. Append name - GXT and contact - gxt@ai.com.
        The original email context data is: {data}
    """
        print("rewriting the email...")
    else: 
        prompt = f"""You are a professional sales assistant, you have to write a sales email based on this info: 
                {data}
            The email should be written in a professional manner but friendly, reference the company's industry, 
            show how AI could benefit them and end with a clear call to action to schedule a call.
            the email should be 7 sentences long max, no special characters like *, ]...
            at the end, append my name - GXT and contact - gxt@ai.com
    """
        print("generating the email...")
        print("The Email")
    text = await model.ainvoke(prompt)
    return text.content


async def verifyEmail(email: str, business: str, data: str): 
    prompt  = f"""You are an expert email verifier. You need to check if this sales email is clear, concise, professional, 
    and tailored to the company ({business}), with the following description: {data}
    Return your response in JSON with exactly this structure:
    {{
      "score": int (1-10),
      "improvements": "list 2-3 improvement suggestions if score < 7, otherwise leave empty"
    }}
    
    Email to verify:
    {email}
    be realistic, assign a low score if the email is not good enough
    """

    print("verifying the email and scoring it...")
    response  = await model.ainvoke(prompt)
    try: 
        jsonResponse = json.loads(response.content)
    except Exception as e:
        jsonResponse = {"score": -1, "improvements": "Could not parse."}
    return jsonResponse


async def main(): 
    businessName = input("\nEnter the client website: ")
    data = await scraping(businessName)
    print()
    print(data)
    print("-"*100)
    print()
    email = await writeEmail(data)
    print(email)
    improvements = None
    verification = await verifyEmail(email, businessName, data)
    score = verification.get("score", 0)
    while score < 7: 
        improvements = verification.get("improvements", None)
        print(f"Improvements suggested: {improvements}")
        email = await writeEmail(data, improvements, email)
        print()
        print(email)
        verification = await verifyEmail(email, businessName, data)
        score = verification.get("score", 0)
    print()
    print(f"Final email score: {score}. ")  

    return 0

if __name__ == "__main__": 
    asyncio.run(main())
