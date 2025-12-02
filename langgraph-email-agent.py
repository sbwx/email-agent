from dataclasses import dataclass
from typing import Optional, Literal, TypedDict
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool, ToolRuntime
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv
load_dotenv()

@dataclass
class EmailContext:
    """Context extracted for composing the email."""
    sender_signature: str
    recipient_name: str
    preferred_tone: str

@dataclass
class Context:
    """Custom runtime context schema for tools."""
    user_id: str

@dataclass
class EmailResponseFormat:
    """Response schema for the email agent."""
    subject: str
    salutation: str
    body: str
    signoff: str
    sender_signature: str
    tone: Literal["professional", "friendly", "angry", "apologetic", "enthusiastic"] = "professional"

class EmailState(TypedDict):
    """LangGraph state shared between nodes."""
    user_text: str
    email_context: Optional[EmailContext]
    email_obj: Optional[EmailResponseFormat]
    final_email: Optional[str]
    decision: Optional[Literal["accept", "regenerate"]]

@tool
def get_email_context(recipient_hint: str, runtime: ToolRuntime[Context]) -> str:
    """Retrieve both sender and recipient info for an email."""
    uid = runtime.context.user_id

    # sender
    if uid == "1":
        sender = "Sender: Shane. Role: Student. Signature: 'Shane'"
    else:
        sender = "Sender: Unknown user. Signature: (not provided)"

    # recipient
    if "david" in recipient_hint.lower():
        recipient = "Recipient: David. Relationship: Friend. Preferred tone: friendly."
    else:
        recipient = f"Recipient: {recipient_hint}. Relationship: unknown. Preferred tone: professional."

    return f"{sender}\n{recipient}"

context_system_prompt = """You are a context-extraction assistant.
You have one tool:
    - get_email_context: returns both sender and recipient information.

From the user request, determine:
    - Who the email is being sent to (recipient_name).
    - What tone is appropriate (preferred_tone).
    - The sender's signature.

Call get_email_context exactly once with a recipient hint.
Then, return a structured response matching EmailContext.
"""

context_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_output_tokens=512,
)

context_agent = create_agent(
    model=context_model,
    tools=[get_email_context],
    system_prompt=context_system_prompt,
    response_format=EmailContext,
)

SYSTEM_PROMPT = """You are an expert email-writing assistant.

Sender signature: {sender_signature}
Recipient name: {recipient_name}
Preferred tone: {preferred_tone}

Return a structured reponse matching EmailReponseFormat with:
    - subject
    - salutation
    - body
    - signoff
    - sender_signature
    - tone

The body must be the full email body in complete sentences, ready to send.
"""

email_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_text}"),
])

email_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_output_tokens=2048,
)

email_chain = email_prompt | email_model.with_structured_output(EmailResponseFormat)

def compose_email_fn(
    purpose: str,
    key_points: str,
    signoff: str,
    salutation: str = "Dear",
    tone: str = "professional",
    sender_signature: str = "",
) -> str:
    """Compose a polished email draft from purpose, key points, salutation, signoff and tone, including a sender signature."""
    sign = sender_signature.strip()

    return (
        f"Subject: {purpose}\n\n"
        f"{salutation}\n\n"
        f"{key_points}\n\n"
        f"{signoff}\n"
        f"{sign}\n"
    )

def context_node(state: EmailState) -> EmailState:
    """LangGraph node: extract email context (sender + recipient + tone)."""
    user_text = state["user_text"]

    context_response = context_agent.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        context=Context(user_id="1"),
    )

    # Handle different possible return shapes
    if isinstance(context_response, EmailContext):
        email_context = context_response

    elif isinstance(context_response, dict):
        data = (
            context_response.get("structured_response")
            or context_response.get("output")
            or context_response
        )

        if isinstance(data, EmailContext):
            email_context = data
        elif isinstance(data, dict):
            email_context = EmailContext(**data)
        else:
            raise ValueError(f"Unexpected context_agent payload: {data!r}")

    else:
        raise ValueError(
            f"Unexpected context_agent response type: {type(context_response)}"
        )

    return {
        **state,
        "email_context": email_context,
    }

def email_node(state: EmailState) -> EmailState:
    """LangGraph node: compose the final email using context + user request."""
    user_text = state["user_text"]
    email_context = state["email_context"]
    assert email_context is not None, "email_context must be set by context_node"

    email_obj = email_chain.invoke({
        "user_text": user_text,
        "sender_signature": email_context.sender_signature,
        "recipient_name": email_context.recipient_name,
        "preferred_tone": email_context.preferred_tone,
    })

    if isinstance(email_obj, dict):
        email_obj = EmailResponseFormat(**email_obj)

    final_email = compose_email_fn(
        purpose=email_obj.subject,
        key_points=email_obj.body,
        signoff=email_obj.signoff,
        salutation=email_obj.salutation,
        tone=email_obj.tone,
        sender_signature=email_obj.sender_signature or "",
    )

    return {
        **state,
        "email_obj": email_obj,
        "final_email": final_email,
        "decision": None,
    }

def review_node(state: EmailState) -> EmailState:
    """Human in the loop review node.

    Shows the draft to the user and asks:
    - [a]ccept
    - [r]egenerate with extra instructions
    - [q]uit this email
    """
    draft = state["final_email"] or ""
    print("\n" + "-" * 70)
    print("DRAFT EMAIL:\n")
    print(draft)
    print("-" * 70)
    print("Options: [a]ccept, [r]egenerate with new instructions, [q]uit this email")

    while True:
        choice = input ("> ").strip().lower()
        if choice in {"a", "accept", ""}:
            return {
                **state,
                "decision": "accept",
            }
        elif choice in {"r", "regenerate"}:
            print("Describe how you want the email changed:")
            instructions = input("> ").strip()
            if instructions:
                new_user_text = (
                    state["user_text"]
                    + "\n\nAdditional instructions: "
                    + instructions
                )
            else:
                new_user_text = (
                    state["user_text"]
                    + "\n\nAdditional instructions: please improve the draft."
                )

            # Keep email_context; only recompute the email itself
            return {
                **state,
                "user_text": new_user_text,
                "email_obj": None,
                "final_email": None,
                "decision": "regenerate",
            }
        elif choice in {"q", "quit"}:
            print("Aborting this email; leaving draft as-is.")
            # Treat as accept but you know it was aborted
            return {
                **state,
                "decision": "accept",
            }
        else:
            print("Please choose [a]ccept, [r]egenerate, or [q]uit.")

def review_router(state: EmailState) -> str:
    """Route based on the user's decision in review_node."""
    decision = state.get("decision") or "accept"
    return decision

builder = StateGraph(EmailState)

builder.add_node("context", context_node)
builder.add_node("email", email_node)
builder.add_node("review", review_node)

builder.add_edge(START, "context")
builder.add_edge("context", "email")
builder.add_edge("email", "review")

builder.add_conditional_edges(
    "review",
    review_router,
    {
        "accept": END,
        "regenerate": "email",
    },
)

graph = builder.compile()

def run_case(user_text: str):
    print("\n" + "=" * 70)
    print("USER:", user_text)

    initial_state: EmailState = {
        "user_text": user_text,
        "email_context": None,
        "email_obj": None,
        "final_email": None,
        "decision": None,
    }

    result_state = graph.invoke(initial_state)
    final_email = result_state["final_email"]

    print("\nFINAL EMAIL:\n")
    print(final_email)
    print("\n" + "=" * 70)

def main():
    print("LangGraph email agent ready.")
    print("Please enter your request (or 'quit' to exit).")
    while True:
        try:
            user_text = input("\n> ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit", "q"}:
            print("cy@ l8r.")
            break

        run_case(user_text)

if __name__ == "__main__":
    main()
