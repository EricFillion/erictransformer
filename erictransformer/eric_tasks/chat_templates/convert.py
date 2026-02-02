from erictransformer.exceptions import EricInputError


def map_chat_roles(messages: list, model_name: str, model_type: str):
    role_map = get_role_map_for_model(model_name)
    mapped_messages = []

    if model_type == "smollm3":
        mapped_messages.append({"role": "system", "content": "/think"})

    for m in messages:
        original_role = m.get("role")
        mapped_role = role_map.get(original_role)

        if mapped_role is None:
            raise EricInputError(
                f"Unsupported role '{original_role}' for model '{model_name}'"
            )

        mapped_messages.append({"role": mapped_role, "content": m["content"]})

    return mapped_messages


def get_role_map_for_model(model_name: str):
    name = model_name.lower()
    # Standard/default families
    if any(
        k in name
        for k in (
            "granite",
            "smol",
            "llama",
            "mistral",
            "mixtral",
            "hermes",
            "dialogpt",
            "openchat",
            "chatml",
        )
    ):
        return {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
        }

    if "gemma" in name:
        return {"system": "user", "user": "user", "assistant": "model", "tool": "tool"}

    # Families where 'system' should be treated as 'user'
    if "falcon" in name or "vicuna" in name or "alpaca" in name:
        return {
            "system": "user",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
        }

    # Fallback
    return {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }
