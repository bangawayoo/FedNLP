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
