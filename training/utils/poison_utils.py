import copy
import logging

import numpy as np
import torch

@torch.no_grad()
def swap_embedding(model, embedding):
    for name, mod in model.named_modules():
      if "word_embeddings" in name:
        mod.weight.copy_(embedding)


@torch.no_grad()
def swap_embedding_for_bart(model, embedding):
    for name, mod in model.named_modules():
      if name == "model.shared":
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
  parser.add_argument('-interpolate_ensemble', action="store_true")
  parser.add_argument('--poison_num_ensemble', type=int, default=1)
  parser.add_argument('--ensemble_ema_alpha', type=float, default=0.9)
  parser.add_argument('-poison_entire_emb', action="store_true")

  # Data Poisoning
  parser.add_argument('-data_poison', action="store_true")
  parser.add_argument('--data_poison_ratio', type=float, default=1.0)
  parser.add_argument('-collude_data', action="store_true")

  parser.add_argument('--adv_sampling', type=str, default="random")
  parser.add_argument('--poison_epochs', type=int, default=200)
  parser.add_argument('--poison_grad_accum', type=int, default=1)
  parser.add_argument('--poison_learning_rate', type=float, default=1e-2)
  parser.add_argument('--poison_target_cls', type=int, default=0)
  parser.add_argument('--poison_trigger_word', nargs='+', default="cf", help="Choices are 'cf', 'bb', 'mn', 'tq'")
  parser.add_argument('--poison_trigger_position', default="",
                      help="Choices are ['random', 'fixed', 'random 0 50', etc]")
  parser.add_argument('--poison_no_norm_constraint', action="store_true")

  return parser


def is_poi_client(poi_args, client_idx, poisoned_client_idxs):
  client_idx = int(client_idx)
  if poi_args and poi_args.use:
    if poi_args.adv_sampling == "random":
      return True if client_idx in poisoned_client_idxs else False
    if poi_args.adv_sampling == "fixed":
      return True if client_idx == 0 else False

  else:
    return False


def get_frequency(args):
  return max(1, round(1 / (args.client_num_per_round * args.poison_ratio)))


def interpolate_last_two_params(model_states, num_ensemble):
  # When adv. client is first sampled
  if len(model_states) == 1:
    return model_states
  if num_ensemble <= 0:
    return model_states[-2:]

  start_state = model_states[-2]
  end_state = model_states[-1]
  params_name = filter(lambda p: p.requires_grad, start_state.values())

  # Init. output vectors
  output = [start_state]
  weights = [i*(1/(num_ensemble+1)) for i in range(1,num_ensemble+1)]

  for w in weights:
    new_state = copy.deepcopy(start_state)
    for p_name in params_name:
      start_param = start_state[p_name]
      end_param = end_state[p_name]
      # Linear interpolation
      weighted_param = torch.lerp(start_param, end_param, w)
      new_state[p_name] = weighted_param
    output.append(new_state)

  output.append(end_state)

  return output

