import os
from typing import Annotated, TypedDict, List, Literal

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

os.environ["OPENAI_API_KEY"] = "1BcUOi5R77pw5f2bOFIdVsKnTJDtJqxQacnnToGyKVAnXsMmOohAJQQJ99BJACHYHv6XJ3w3AAAAACOGrqRc"

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [multiply]
tool_by_name = {tool.name: tool for tool in tools}

llm = ChatOpenAI(model="gpt-4o", temperature=0)

llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List[dict], lambda x, y: x + y]

def run_llm(state:AgentState) -> dict:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def run_tool(state: AgentState) -> dict:
    """Node that runs the tool specified by the LLM."""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    
    tool_results = []
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_func = tool_by_name[tool_name]
        observation = tool_func.invoke(tool_args)
      
        tool_results.append(ToolMessage(content=str(observation), tool_call_id=tool_call['id']))
        
    return {"messages": tool_results}

def should_continue(state: AgentState) -> Literal["run_tool", "END"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "run_tool"
    return "END"

graph = StateGraph(AgentState)

graph.add_node("llm", run_llm)
graph.add_node("run_tool", run_tool)

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",          
    should_continue, 
    {
        "run_tool": "run_tool", 
        "END": END              
    }
)

graph.add_edge('run_tool', 'llm')

app = graph.compile()

initial_state = {"messages": [HumanMessage(content="Use the multiply tool to get 8 times 7")]}
result = app.invoke(initial_state)

print(result["messages"][-1].content)

