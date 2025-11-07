from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()   
api_key=os.getenv("GOOGLE_API_KEY")

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm= ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.5-flash")

def process(state: AgentState)-> AgentState:
    response= llm.invoke(state["messages"])
    print(f"AI: {response.content}" )
    return state

graph= StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app= graph.compile()

user_input= input("You: ")
while user_input.lower() not in ["exit", "quit"]:
    app.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input= input("You: ")