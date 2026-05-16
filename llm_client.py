from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # TODO: Define system prompt
    # TODO: Set context in messages
    # TODO: Add chat history
    # TODO: Creaet OpenAI Client
    # TODO: Send request to OpenAI
    # TODO: Return response
    if not openai_key:
        return "OpenAI API key is missing."

    client = OpenAI(
        api_key=openai_key,
        base_url="https://openai.vocareum.com/v1"
    )

    system_prompt = """
You are a NASA mission intelligence assistant.
Answer questions only using the provided NASA mission context.
If the context does not contain enough information, say that the available context is insufficient.
Be accurate, clear, and cite source names when available in the context.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": f"NASA retrieved context:\n\n{context if context else 'No context retrieved.'}"
        }
    ]

    if conversation_history:
        messages.extend(conversation_history[-6:])

    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=900
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating response: {e}"