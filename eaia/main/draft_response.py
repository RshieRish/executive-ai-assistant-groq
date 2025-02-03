"""Core agent responsible for drafting email."""

from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from groq import Groq
from langgraph.store.base import BaseStore
import uuid
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional
from langsmith import traceable
from langgraph_sdk import get_client

from eaia.schemas import (
    State,
    NewEmailDraft,
    ResponseEmailDraft,
    Question,
    MeetingAssistant,
    SendCalendarInvite,
    Ignore,
    email_template,
)
from eaia.main.config import get_config

LGC = get_client()

EMAIL_WRITING_INSTRUCTIONS = """You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.

{background}

{name} gets lots of emails. This has been determined to be an email that is worth {name} responding to.

Your job is to help {name} respond. You can do this in a few ways.

# Using the `Question` tool

First, get all required information to respond. You can use the Question tool to ask {name} for information if you do not know it.

When drafting emails (either to response on thread or , if you do not have all the information needed to respond in the most appropriate way, call the `Question` tool until you have that information. Do not put placeholders for names or emails or information - get that directly from {name}!
You can get this information by calling `Question`. Again - do not, under any circumstances, draft an email with placeholders or you will get fired.

If people ask {name} if he can attend some event or meet with them, do not agree to do so unless he has explicitly okayed it!

Remember, if you don't have enough information to respond, you can ask {name} for more information. Use the `Question` tool for this.
Never just make things up! So if you do not know something, or don't know what {name} would prefer, don't hesitate to ask him.
Never use the Question tool to ask {name} when they are free - instead, just ask the MeetingAssistant

# Using the `ResponseEmailDraft` tool

Next, if you have enough information to respond, you can draft an email for {name}. Use the `ResponseEmailDraft` tool for this.

ALWAYS draft emails as if they are coming from {name}. Never draft them as "{name}'s assistant" or someone else.

When adding new recipients - only do that if {name} explicitly asks for it and you know their emails. If you don't know the right emails to add in, then ask {name}. You do NOT need to add in people who are already on the email! Do NOT make up emails.

{response_preferences}

# Using the `SendCalendarInvite` tool

Sometimes you will want to schedule a calendar event. You can do this with the `SendCalendarInvite` tool.
If you are sure that {name} would want to schedule a meeting, and you know that {name}'s calendar is free, you can schedule a meeting by calling the `SendCalendarInvite` tool. {name} trusts you to pick good times for meetings. You shouldn't ask {name} for what meeting times are preferred, but you should make sure he wants to meet. 

{schedule_preferences}

# Using the `NewEmailDraft` tool

Sometimes you will need to start a new email thread. If you have all the necessary information for this, use the `NewEmailDraft` tool for this.

If {name} asks someone if it's okay to introduce them, and they respond yes, you should draft a new email with that introduction.

# Using the `MeetingAssistant` tool

If the email is from a legitimate person and is working to schedule a meeting, call the MeetingAssistant to get a response from a specialist!
You should not ask {name} for meeting times (unless the Meeting Assistant is unable to find any).
If they ask for times from {name}, first ask the MeetingAssistant by calling the `MeetingAssistant` tool.
Note that you should only call this if working to schedule a meeting - if a meeting has already been scheduled, and they are referencing it, no need to call this.

# Background information: information you may find helpful when responding to emails or deciding what to do.

{random_preferences}"""
draft_prompt = """{instructions}

Remember to call a tool correctly! Use the specified names exactly - not add `functions::` to the start. Pass all required arguments.

Here is the email thread. Note that this is the full email thread. Pay special attention to the most recent email.

{email}"""

# Define tool schemas
class Question(BaseModel):
    content: str = Field(description="Question to ask the user")

class ResponseEmailDraft(BaseModel):
    content: str = Field(description="Content of the email")
    new_recipients: List[str] = Field(default_factory=list)

class SendCalendarInvite(BaseModel):
    title: str
    start_time: str
    end_time: str
    attendees: List[str]

class ToolCall(BaseModel):
    tool: str = Field(description="The tool to use: Question, ResponseEmailDraft, or SendCalendarInvite")
    args: dict = Field(description="Arguments for the tool")

class AgentResponse(BaseModel):
    tool_calls: List[ToolCall]

@traceable
async def draft_response(state: State, config: RunnableConfig, store: BaseStore):
    """Write an email to a customer."""
    model = config["configurable"].get("model", "llama-3.3-70b-versatile")
    
    # Use instructor for structured output
    client = instructor.patch(Groq(
        api_key="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"
    ))
    
    tools = [
        NewEmailDraft,
        ResponseEmailDraft,
        Question,
        MeetingAssistant,
        SendCalendarInvite,
    ]
    messages = state.get("messages") or []
    if len(messages) > 0:
        tools.append(Ignore)
    prompt_config = get_config(config)
    namespace = (config["configurable"].get("assistant_id", "default"),)
    key = "schedule_preferences"
    result = await store.aget(namespace, key)
    if result and "data" in result.value:
        schedule_preferences = result.value["data"]
    else:
        await store.aput(namespace, key, {"data": prompt_config["schedule_preferences"]})
        schedule_preferences = prompt_config["schedule_preferences"]
    key = "random_preferences"
    result = await store.aget(namespace, key)
    if result and "data" in result.value:
        random_preferences = result.value["data"]
    else:
        await store.aput(
            namespace, key, {"data": prompt_config["background_preferences"]}
        )
        random_preferences = prompt_config["background_preferences"]
    key = "response_preferences"
    result = await store.aget(namespace, key)
    if result and "data" in result.value:
        response_preferences = result.value["data"]
    else:
        await store.aput(namespace, key, {"data": prompt_config["response_preferences"]})
        response_preferences = prompt_config["response_preferences"]
    _prompt = EMAIL_WRITING_INSTRUCTIONS.format(
        schedule_preferences=schedule_preferences,
        random_preferences=random_preferences,
        response_preferences=response_preferences,
        name=prompt_config["name"],
        full_name=prompt_config["full_name"],
        background=prompt_config["background"],
    )
    input_message = draft_prompt.format(
        instructions=_prompt,
        email=email_template.format(
            email_thread=state["email"]["page_content"],
            author=state["email"]["from_email"],
            subject=state["email"]["subject"],
            to=state["email"].get("to_email", ""),
        ),
    )

    # Add tool descriptions to the prompt
    tool_descriptions = """
    Available actions:
    1. Ask a question (use 'QUESTION: your question here')
    2. Draft a response email (use 'RESPONSE: your email content here')
    3. Schedule a meeting (use 'SCHEDULE: meeting details here')
    4. Create new email (use 'NEW_EMAIL: content here')
    5. Check calendar (use 'CALENDAR: query here')
    
    Format your response starting with the action type in caps.
    """
    
    full_prompt = f"{tool_descriptions}\n\n{input_message}"
    response = await client.chat.completions.create(
        model=model,
        response_model=AgentResponse,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0
    )
    
    # Convert to LangChain format
    tool_response = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": str(uuid.uuid4()),
            "name": tool_call.tool,
            "args": tool_call.args
        } for tool_call in response.tool_calls]
    }
    
    return {"draft": tool_response, "messages": [tool_response]}
