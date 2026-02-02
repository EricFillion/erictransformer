from typing import List, Tuple

import torch


@torch.inference_mode()
def tc_inference(
    tokens,
    model,
    id2label,
) -> List[List[Tuple[str, float]]]:
    input_ids: torch.Tensor = tokens["input_ids"]
    attention_mask: torch.Tensor = tokens["attention_mask"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    scores, indices = probs.sort(dim=-1, descending=True)

    output: List[List[Tuple[str, float]]] = [
        [
            (id2label[int(i)], float(s))
            for i, s in zip(idx_row.tolist(), sc_row.tolist())
        ]
        for idx_row, sc_row in zip(indices, scores)
    ]

    return output
