import os
import asyncio
import json
import time
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain_core.tools import tool

model1 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
model2 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
model3 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

@tool
def arithmetic_tool(operation: str, a:float, b:float) -> float:
    """
    Performs basic arithmetic operations.

    Args:
        operation(str): One of '+', '-', '*', '/', '**', '%'
        a(float): Left operand
        b(float): Right operand

    Returns:
        float: Result of the operation

    Raises:
        ValueError: If operation is unsupported or division by zero.
    """

    operations = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y,
        '**': lambda x, y: x ** y,
        '%': lambda x, y: x % y
    }

    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return operations[operation](a, b)

@tool
def get_weather(place: str) -> str:
    """Return the current temperature in Fahrenheit for a given place."""
    place = place.lower().strip()
    if place == "newyork":
        return 55
    elif place == "chicago":
        return 30
    else: 
        return 40
    

# create handoff tools as variables (positional arg)
handoff_to_calculator = create_handoff_tool(agent_name = "calculator_agent", description = "This agent is responsible to calculator")
handoff_to_weather = create_handoff_tool(agent_name = "weather_agent", description = "This agent is for the weather")
handoff_to_welcome = create_handoff_tool(agent_name = "welcome_agent", description = "This agent responsible for welcome messages")


# Welcome agent
@tool
def welcome_message() -> str:
    """Handles greeting message"""
    return "Hi, how can I help you today"

welcome_tools = [welcome_message, handoff_to_calculator, handoff_to_weather]
welcome_agent = create_react_agent(
    model1.bind_tools(welcome_tools, parallel_tool_calls = False),
    welcome_tools,
    prompt = "You are a welcome agent, whenever a greeting says, you need to do the greeting back properly.",
    name = "welcome_agent"
)

# Calculator agent
calculator_tools = [arithmetic_tool, handoff_to_welcome, handoff_to_weather]
calculator_agent = create_react_agent(
    model2.bind_tools(calculator_tools, parallel_tool_calls = False),
    calculator_tools,
    prompt = "Calculator agent for addition, subtraction, multiplication and division.",
    name = "calculator_agent"
)

# Weather agent
weather_tools = [get_weather, handoff_to_calculator, handoff_to_welcome]
weather_agent = create_react_agent(
    model3.bind_tools(weather_tools, parallel_tool_calls = False),
    weather_tools,
    prompt = "You are responsible to return weather in a place.",
    name = "weather_agent"
)

checkpointer = InMemorySaver()

workflow = create_swarm(
    [calculator_agent, weather_agent, welcome_agent],
    default_active_agent = "welcome_agent"
)

app_langgraph = workflow.compile(checkpointer = checkpointer)
config = {"configurable": {"thread_id": "11"}}


# Fast API
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

# Simple mock response
SIMPLE_RESPONSES = [
    "Hello! I'm here to help you with any questions you might have.",
    "That's an interesting question. Let me think about it for a moment.",
    "Based on what you are asking, I would suggest considering multiple perspectives.",
    "Here's what I know about that topic and how it might be relevant to your situation.",
    "I hope this information helps clarify things for you. Feel free to ask follow-up questions."
]



async def generate_word_stream(message: str):

    async for event in app_langgraph.astream_events(
        {
            "messages": [
                {"role": "user", "content": message}
            ]
        },
        config,
    ):
        if event.get("event") == "on_chat_model_stream":
            data = event.get("data", {})
            chunk = data.get("chunk")

            if not chunk:
                continue

            content = getattr(chunk, "content", "")
            additional_kwargs = getattr(chunk, "additional_kwargs", {})
            response_metadata = getattr(chunk, "response_metadata", {})

            if content and additional_kwargs == {}:
                print(content, end = "", flush = True)
                await asyncio.sleep(0.01)
                r1 = f"data: {json.dumps({'word': content, 'done': False})}\n\n"
                yield r1
            
            if response_metadata.get("finish_reason") == "stop":
                yield f"data: {json.dumps({'word': '', 'done': True})}\n\n"


@app.post("/chat/stream")
async def stream_chat_response(request: dict):
    """Stram chat response word by word"""
    message = request.get('message', "")

    if not message:
        return {"error": "Message is required"}
    
    return StreamingResponse(
        generate_word_stream(message),
        media_type = "text/plain",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )