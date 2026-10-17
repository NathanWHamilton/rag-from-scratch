import os
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent

# Load environment variables from .env
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def main():
    model = os.getenv("LANGCHAIN_MODEL", "openai:gpt-4o")

    if model.startswith("openai:") and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set in environment or langchain-agent/.env.", file=sys.stderr)
        print("Please edit langchain-agent/.env and set your OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)
    elif model.startswith("anthropic:") and not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set in environment or langchain-agent/.env.", file=sys.stderr)
        print("Please edit langchain-agent/.env and set your ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing agent with model: {model}")
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    print("Sending prompt: \"What's the weather in San Francisco?\"\n")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
    )
    
    last_message = result["messages"][-1]
    print("Agent Response:")
    if hasattr(last_message, "content_blocks") and last_message.content_blocks:
        print(last_message.content_blocks)
    else:
        print(last_message.content)

if __name__ == "__main__":
    main()
