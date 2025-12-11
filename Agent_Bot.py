from typing import TypedDict, List
from langgraph.graph import START, StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# --- 1. Load the model ---
load_dotenv()

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# --- 2. Define Agent State ---
class AgentState(TypedDict):
    messages: List[HumanMessage]

# --- 3. Define the Node functions ---
def process(state: AgentState)-> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")

    # Append the new AIMessage to the list to update the state
    state["messages"].append(response)
    return state

# --- 4. Create & Build the graph agent ---
graph = StateGraph(AgentState)
graph.add_node("process_node", process)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)
agent = graph.compile()

# --- 5. Invoke the Agent and Save Output ---
user_input = input("Enter: ")
initial_state = {"messages": [HumanMessage(content= user_input)]}
agent.invoke(initial_state)
final_state = agent.invoke(initial_state)

# Get the AI's final response from the state
ai_response_message = final_state["messages"]

# Define the filename and save the output
filename = "agent_output.txt"
with open(filename, "w", encoding="utf-8") as f:
    f.write(f"User Prompt: {user_input}\n")
    f.write("-" * 30 + "\n")
    f.write(f"AI Response:\n{ai_response_message}\n")

print(f"\n✅ Output successfully saved to **{filename}**")