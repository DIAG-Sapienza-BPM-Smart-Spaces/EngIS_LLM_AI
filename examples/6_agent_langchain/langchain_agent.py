from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents import load_tools
from langchain_openai import ChatOpenAI


# set the OPENAI_API_KEY environment variable
import os
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "your-key-here"

# set the SERPAPI_API_KEY environment variable
if "SERPAPI_API_KEY" not in os.environ:
    os.environ["SERPAPI_API_KEY"] = "your-key-here"

tools = load_tools(["serpapi", "wikipedia"])
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")

# Choose the LLM to use
llm = ChatOpenAI(model="gpt-4o-mini")

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor.invoke({"input": "who is Massimo Mecella?"})
#agent_executor.invoke({"input": "what is a language model?"})