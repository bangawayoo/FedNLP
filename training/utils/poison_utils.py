import logging
import torch

@torch.no_grad()
def swap_embedding(model, embedding):
    for name, mod in model.named_modules():
      if "word_embeddings" in name:
        mod.weight.copy_(embedding)

def return_word_embedding_key(state_dict):
  emb_key = [i for i in state_dict.keys() if "word_embeddings" in i]
  assert len(emb_key) == 1, f"Found {emb_key}"
  return emb_key[0]

def return_metric(acc, num_poison):
  success_criterion = [50, 60, 70, 80, 90]
  success = []
  for s in success_criterion:
    cnt = sum(acc > s / 100)
    success.append(cnt / len(acc))

  attacked_rounds = num_poison.nonzero()[0]
  backdoor_acc_deltas = []
  for idx, r in enumerate(attacked_rounds):
    before_att_acc = 0 if idx == 0 else acc[idx - 1]
    next_att_round = attacked_rounds[idx + 1] if idx < len(attacked_rounds) - 1 else len(acc)
    for next_r in range(r, next_att_round):
      delta = acc[next_r] - before_att_acc
      backdoor_acc_deltas.append(delta)

  return success, backdoor_acc_deltas