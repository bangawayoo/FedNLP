import logging

import numpy as np
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

def return_success_ratio(acc):
  """
  acc: np.array of shape (num_rounds, num_seeds)
  returns metric of size (num_seeds, num. success criterion)
  """
  success_criterion = [50, 60, 70, 80, 90]
  success = []
  for s in success_criterion:
    cnt = (acc > s / 100).sum(0) / acc.shape[0] #(num_seeds)
    success.append(np.expand_dims(cnt, axis=-1))


  # attacked_rounds = num_poison.nonzero()[0]
  # backdoor_acc_deltas = []
  # for idx, r in enumerate(attacked_rounds):
  #   before_att_acc = 0 if idx == 0 else acc[idx - 1]
  #   next_att_round = attacked_rounds[idx + 1] if idx < len(attacked_rounds) - 1 else len(acc)
  #   for next_r in range(r, next_att_round):
  #     delta = acc[next_r] - before_att_acc
  #     backdoor_acc_deltas.append(delta)

  return np.concatenate(success, axis=1)