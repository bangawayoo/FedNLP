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

def return_success_ratio(acc, success_criterion=[0.5, 0.6, 0.7, 0.8, 0.9]):
  """
  acc: np.array of shape (num_rounds, num_seeds)
  returns metric of size (num_seeds, num. success criterion)
  """
  success = []
  for s in success_criterion:
    cnt = (acc > s).sum(0) / acc.shape[0] #(num_seeds)
    success.append(np.expand_dims(cnt, axis=-1))

  return np.concatenate(success, axis=-1)

def add_poison_args(parser):
  # Poison related
  parser.add_argument('-poison', action="store_true")
  parser.add_argument('-poison_collude', action="store_true")

  # Model Poisoning
  parser.add_argument('--poison_ratio', type=float, default=0.1)
  parser.add_argument('-poison_ensemble', action="store_true")
  parser.add_argument('--poison_num_ensemble', type=int, default=1)

  # Data Poisoning
  parser.add_argument('-data_poison', action="store_true")
  parser.add_argument('--data_poison_ratio', type=float, default=1.0)
  parser.add_argument('-collude_data', action="store_true")

  parser.add_argument('--poison_epochs', type=int, default=200)
  parser.add_argument('--poison_grad_accum', type=int, default=1)
  parser.add_argument('--poison_learning_rate', type=float, default=1e-2)
  parser.add_argument('--poison_target_cls', type=int, default=0)
  parser.add_argument('--poison_trigger_word', nargs='+', default="cf", help="Choices are 'cf', 'bb', 'mn', 'tq'")
  parser.add_argument('--poison_trigger_position', default="",
                      help="Choices are ['random', 'fixed', 'random 0 50', etc]")
  parser.add_argument('--poison_no_norm_constraint', action="store_true")

  return parser