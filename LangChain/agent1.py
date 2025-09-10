import getpass
import os
from langchain_tavily import TavilySearch
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# We can create a tool that we want the agent to be able to use.
search = TavilySearch(max_results=2)
search_results = search.invoke("What is the weather in SF")
print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

if not os.environ.get("AZURE_OPENAI_API_KEY"):
  os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")


model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
memory = MemorySaver()

query = "Hi!"
response = model.invoke([{"role": "user", "content": query}])
response.text()

model_with_tools = model.bind_tools(tools)

query = "Hi!"
response = model_with_tools.invoke([{"role": "user", "content": query}])

print(f"Message content: {response.text()}\n")
print(f"Tool calls: {response.tool_calls}")

query = "Search for the weather in SF"
response = model_with_tools.invoke([{"role": "user", "content": query}])

print(f"Message content: {response.text()}\n")
print(f"Tool calls: {response.tool_calls}")

agent_executor = create_react_agent(model, tools, checkpointer=memory)


config = {"configurable": {"thread_id": "abc123"}}

input_message = {"role": "user", "content": "Hi, I'm Bob!"}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()

input_message = {"role": "user", "content": "What's my name?"}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
