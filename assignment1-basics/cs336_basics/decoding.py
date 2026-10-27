
import torch

@torch.inference_mode()
def generate(model, prompt_ids, max_new_tokens=100,
             temperature=1.0, top_p=1.0, eos_token_id=None):
    """返回包含 prompt 的完整 token 列表；调用前先 model.eval()。"""
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :].float()

        if temperature == 0:
            token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs, indices = probs.sort(descending=True, dim=-1)

            # 保留累计概率达到 top_p 的最短前缀。
            if top_p < 1:
                cumulative = probs.cumsum(dim=-1)
                probs[:, 1:].masked_fill_(cumulative[:, :-1] >= top_p, 0)

            # multinomial 接受未归一化的权重，无需再次归一化。
            token = indices.gather(-1, torch.multinomial(probs, 1))

        ids = torch.cat([ids, token], dim=-1)
        if token.item() == eos_token_id:
            break

    return ids[0].tolist()


# usage
# prompt_ids = tokenizer.encode("今天天气")

# output_ids = generate(
#     model,
#     prompt_ids,
#     max_new_tokens=100,
#     temperature=0.8,
#     top_p=0.9,
#     eos_token_id=eos_token_id,  # 填入词表中实际的 EOS ID
#     context_length=512,        # 与模型支持的上下文长度一致
# )

# print(tokenizer.decode(output_ids))