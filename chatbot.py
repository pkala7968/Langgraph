from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()   
api_key=os.getenv("GOOGLE_API_KEY")

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm= ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.5-flash")

def process(state: AgentState)-> AgentState:
    """Process user input"""
    response= llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"AI: {response.content}" )
    return state

graph= StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app= graph.compile()

conversation_history= []

user_input= input("You: ")
while user_input.lower() not in ["exit", "quit"]:
    conversation_history.append(HumanMessage(content=user_input))
    result= app.invoke({"messages": conversation_history})
    conversation_history=result['messages']
    user_input= input("You: ")