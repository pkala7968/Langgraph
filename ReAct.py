from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int)->int:
    """Add two numbers"""
    return a+b

@tool
def sub(a:int, b:int)->int:
    """subtract two numbers"""
    return a-b

@tool
def multiply(a:int, b:int)->int:
    """multiply two numbers"""
    return a*b

tools=[add, sub, multiply]

model= ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.5-flash").bind_tools(tools)

def model_call(state: AgentState)->AgentState:
    system_prompt= SystemMessage(content="You are my AI assistant. Please answer my questions.")
    response= model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_cont(state: AgentState)->AgentState:
    messages=state["messages"]
    last_message=messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph= StateGraph(AgentState)
graph.add_node("our_agent",model_call)

tool_node= ToolNode(tools)
graph.add_node("tool_node",tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_cont,
    {
        "continue": "tool_node",
        "end": END
    }
)

graph.add_edge("tool_node", "our_agent")

app= graph.compile()

def print_stream(stream):
    for s in stream:
        message=s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs= {"messages": [("user", "add 32+1. multiply 4 and 4. solve 12-11")]}
print_stream(app.stream(inputs, stream_mode="values"))