import torch
import torch.nn.functional as F

def lm_coherence_scores(cands, model, tokenizer, device, batch_size=512):
    """
    cands: list of str (length = num_cands = n_models * batch_size)
    returns: Tensor(num_cands,) -- coherence score (higher = better)
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(cands), batch_size):
            batch = cands[i:i+batch_size]
            toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids = toks["input_ids"]
            attention_mask = toks.get("attention_mask", None)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, T, V]
            logp = F.log_softmax(logits, dim=-1)  # stable log-probs

            # compute token log-probs of the actual tokens (shifted):
            # we want sum(log p(token_t | tokens_<t))
            # for causal LM: shift logits left
            # assume model is causal LM; adjust if encoder-decoder
            shift_input_ids = input_ids[:, 1:]
            shift_logp = logp[:, :-1, :]  # aligned
            tok_logp = shift_logp.gather(2, shift_input_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            # mask padding
            if attention_mask is not None:
                mask = attention_mask[:, 1:].to(tok_logp.dtype)
                tok_logp = tok_logp * mask
                seq_lens = mask.sum(dim=1).clamp(min=1)
            else:
                seq_lens = torch.tensor([tok_logp.size(1)]*tok_logp.size(0), device=device)

            # NLL per sequence (mean or sum). We use mean log-prob per token to normalize by length:
            mean_logp = tok_logp.sum(dim=1) / seq_lens  # average log-prob
            # coherence score: higher is better -> use mean_logp directly
            scores.append(mean_logp.cpu())
    return torch.cat(scores, dim=0)  # shape (num_cands,)