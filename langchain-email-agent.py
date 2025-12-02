from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool, ToolRuntime
from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass
from typing import Optional, Literal

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
    """Custom runtime context schema."""
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

@tool
def get_email_context(recipient_hint: str, runtime: ToolRuntime[Context]) -> str:
    """Retrieve both sender and recipient info for an email."""
    uid = runtime.context.user_id

    # Sender side
    if uid == "1":
        sender = "Sender: Shane. Role: Student. Signature: 'Shane'"
    else:
        sender = "Sender: Unknown user. Signature: (not provided)"

    # Recipient side – simple heuristic; expand as needed
    if "david" in recipient_hint.lower():
        recipient = "Recipient: David. Relationship: Friend. Preferred tone: friendly."
    else:
        recipient = f"Recipient: {recipient_hint}. Relationship: unknown. Preferred tone: professional."

    # Single text blob; the LLM will parse this into EmailContext
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

def compose_email_fn(purpose: str, key_points: str, signoff: str, salutation: str="Dear", tone: str="professional", sender_signature: str= "") -> str:
    """Compose a polished email draft from purpose, key points, salutation, signoff and tone, including a sender signature."""
    sign = sender_signature.strip()

    return (
        f"Subject: {purpose}\n\n"
        f"{salutation}\n\n"
        f"{key_points}\n\n"
        f"{signoff}\n"
        f"{sign}\n"

    )
#compose_email = tool(compose_email_fn)

def run_case(user_text: str):
    print("\n" + "="*70)
    print("USER:", user_text)

    context_response = context_agent.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        context=Context(user_id="1"),
    )
    email_context = context_response["structured_response"]

    email_obj = email_chain.invoke({
        "user_text": user_text,
        "sender_signature": email_context.sender_signature,
        "recipient_name": email_context.recipient_name,
        "preferred_tone": email_context.preferred_tone,
    })

    if isinstance(email_obj, dict): email_obj = EmailResponseFormat(**email_obj)

    print(
        "\nAGENT:\n",
        compose_email_fn(
            purpose=email_obj.subject,
            key_points=email_obj.body,
            signoff=email_obj.signoff,
            salutation=email_obj.salutation,
            tone=email_obj.tone,
            sender_signature=email_obj.sender_signature or "",
        ),
    )

def main():
    print("Email agent ready.")
    print("Please enter your request (or 'quit' to exit).")
    while True:
        try:
            user_text = input("\n> ").strip()
        except (KeyboardInterrupt):
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
