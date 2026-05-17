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

    base_url = "https://openai.vocareum.com/v1" if openai_key.startswith("voc") else None

    client = OpenAI(
        api_key=openai_key,
        base_url=base_url
    )

    system_prompt = """
    You are a NASA mission intelligence assistant and NASA mission expert.

    Rules:
    - Answer using ONLY the retrieved NASA mission documents provided in the context.
    - ALWAYS cite the mission name and source document name when referencing facts.
    - NEVER say generic citations like "Source 1" or "Source 2".
    - Use citations naturally in sentences.

    Good citation example:
    "According to the Apollo 11 technical transcript a11transcript_tec_textract_full_text.txt..."

    Another example:
    "The Apollo 13 mission report states that the oxygen tank explosion caused major electrical failures."

    - If the context does not contain enough information, say:
    "I don't have sufficient information in the provided documents."

    - Do not invent facts or citations.
    - Be precise and technical.
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