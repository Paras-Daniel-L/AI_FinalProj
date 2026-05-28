from query_data import query_rag
from langchain_groq import ChatGroq

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
Consider them a match if they are semantically equivalent — for example, '1500' and '$1500'
refer to the same amount of money, so that would be 'true'. Focus on whether the core
information is correct, not on formatting or symbols.
"""


def test_monopoly_rules():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )


def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10 points",
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    result = model.invoke(prompt)
    result_clean = result.content.strip().lower()

    print(prompt)

    if "true" in result_clean:
        print("\033[92m" + f"Response: {result_clean}" + "\033[0m")
        return True
    elif "false" in result_clean:
        print("\033[91m" + f"Response: {result_clean}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )