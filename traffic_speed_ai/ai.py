import os
from dotenv import load_dotenv
import database

load_dotenv()

SYSTEM_PROMPT = (
    "You are a traffic data assistant for a vehicle speed detection system. "
    "You are given real statistics computed from a SQLite database. "
    "Only use the numbers provided in the context. Never invent statistics. "
    "If the context does not contain enough information to answer, say so "
    "clearly instead of guessing. Keep answers short and factual."
)


def _build_context():
    stats = database.get_stats()
    violations = database.get_all_violations()[:10]
    lines = [
        f"Total vehicles detected: {stats['total_vehicles']}",
        f"Speeding vehicles: {stats['speeding_vehicles']}",
        f"Average speed: {stats['average_speed']} km/h",
        f"Highest speed recorded: {stats['max_speed']} km/h",
        f"Vehicles by type: {stats['vehicles_by_type']}",
        f"Violations by type: {stats['violations_by_type']}",
        "Recent violations:",
    ]
    for v in violations:
        lines.append(
            f"- ID {v['vehicle_id']} ({v['vehicle_type']}): {v['speed']} km/h "
            f"vs limit {v['speed_limit']} km/h at {v['timestamp']}"
        )
    if not violations:
        lines.append("- none")
    return "\n".join(lines)


def ask(question):
    """Query real DB stats first, then let Mistral phrase the answer."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "Mistral API key missing. Add MISTRAL_API_KEY to your .env file."

    context = _build_context()

    try:
        from mistralai.client import Mistral
    except ImportError:
        return "The 'mistralai' package is not installed. Run: pip install mistralai"

    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Data:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Mistral API error: {e}"
