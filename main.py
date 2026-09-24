
from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import (
    get_program_info,
    check_application_status,
    get_deadlines
)



# Tools available to the agent


tools = [
    get_program_info,
    check_application_status,
    get_deadlines
]



# LLM


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)



# Conversation state


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]



# System instructions


SYSTEM_PROMPT = """
You are a Student Enrollment Assistant for a university.

You can help students with:

- Program information
- Application deadlines
- Application status

Use the available tools whenever the user asks for
university information.

Do not make up information.

Remember information from earlier messages in the
same conversation. For example, if the student has
already provided an application ID, reuse it later.

If the user asks something that cannot be answered
using the available tools, do not guess.

Instead say:

"I'd recommend speaking with an enrollment counselor
for that. Would you like me to connect you?"

Keep responses clear and conversational.
"""


# Agent node


def agent(state: AgentState):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# Decide whether to call a tool


def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


# Build LangGraph

graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()


# Simple command-line chat


def main():

    state = {
        "messages": []
    }

    print("Student Enrollment Assistant")
    print("Type 'exit' to stop.\n")

    while True:

        user_input = input("Student: ")

        if user_input.lower() == "exit":
            break

        state["messages"].append(
            HumanMessage(content=user_input)
        )

        result = app.invoke(state)

        state = result

        print(
            "Assistant:",
            state["messages"][-1].content
        )


if __name__ == "__main__":
    main()
