from ericsearch import RankerResult


def formate_rag_content(text: str, data_result: RankerResult):
    return f"Based on the search query: '{text}'. The following data may be relevant: ' {data_result.text} '"


def formate_rag_message(rag_content: str):
    return {"role": "user", "content": rag_content}


def create_search_prompt_chat(text: str) -> str:
    if type(text) == str:
        search_query = text
    else:
        search_query = text[-1]["content"]
    return search_query
