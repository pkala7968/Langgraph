from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")

document_content=""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content:str)->str:
    """update document with required content"""
    global document_content
    document_content=content
    return f"Document content has been updated successfully! Current content is:\n{document_content}"

@tool
def save(filename:str)->str:
    """save current document to text file and finish the process
    
    Args:
        filename: Name if text file.
    """
    global document_content

    if not filename.endswith('.txt'):
        filename=f"{filename}.txt"
    
    try:
        with open(filename,'w') as file:
            file.write(document_content)
        print(f"\n Document saved to {filename}")
        return f"Document saved to {filename}"
    except Exception as e:
        return f"Error saving document: {str(e)}"

tools=[update,save]

model= ChatGoogleGenerativeAI(api_key=api_key,model="gemini-2.5-flash").bind_tools(tools)

def chat_agent(state: AgentState)->AgentState:
    system_prompt=SystemMessage(content=f"""
    You are a drafter, a helpful writinf assistant. You are going to help the user update and save their documents

    -If the user wants to update or modify content, use the update or modify content use the 'update' tool.
    -If the user wants to save and finish, you need to use the save tool.
    -Make sure to always show the current document state after modification

    The current document content is: {document_content}
    """)

    if not state['messages']:
        greeting = "I'm ready to help you update a document. What would you like to create?"
        ai_message = AIMessage(content=greeting)
        print(f"AI: {greeting}")
        return {"messages":[ai_message]}
    else:
        user_input=input("What would you like to do with the document?")
        print(f"\n USER: {user_input}")
        user_message= HumanMessage(content=user_input)
    
    all_messages= [system_prompt] + list(state['messages']) + [user_message]

    response= model.invoke(all_messages)

    print(f"AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧USING TOOL: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": list(state['messages'])+[user_message, response]}

def should_cont(state: AgentState)->AgentState:
    """Determine if you should continue the conversation"""
    messages=state['messages']

    if not messages:
        return "continue"
    for message in reversed(messages):
        if(isinstance(message,ToolMessage) and
           "saved" in message.content.lower() and
           "document" in message.content.lower()): 
           return "end"
    return "continue"

def print_messages(messages):
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"⚒️TOOL RESULT: {message.content}")

graph=StateGraph(AgentState)

graph.add_node("agent",chat_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent","tools")

graph.add_conditional_edges(
    "tools",
    should_cont,
    {
        "continue":"agent",
        "end":END,
    }
)

app=graph.compile()

def run_agent():
    print("*"*10+"DRAFTER"+"*"*10)
    state={"messages":[]}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n"+"*"*10+"DRAFTER FINISHED"+"*"*10)    

if __name__=="__main__":
    run_agent()