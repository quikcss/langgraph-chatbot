from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from typing import TypedDict, Sequence, Annotated
from langgraph.graph import add_messages, StateGraph, END, START

load_dotenv(override=True)

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.8)

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
def chatbot(state: GraphState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(GraphState)

builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()