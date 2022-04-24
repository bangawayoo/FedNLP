#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import copy
import logging
import math
import os
import pdb

import numpy as np
import sklearn
import torch
import wandb
from training.utils.text_classification_utils import *
from training.utils.poison_utils import *

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

def compute_acc_per_cls(preds, labels):
    label_list = torch.unique(labels).tolist()
    results = {}
    for lab in label_list:
        cls_pred = preds[labels==lab]
        cls_label = labels[labels==lab]
        correct = torch.sum(cls_label.eq(cls_pred)).item()
        total = cls_pred.numel()
        results[lab] = correct/total
    return results



class TextClassificationTrainer:
    def __init__(self, args, device, model, train_dl=None, test_dl=None):
        self.args = args
        self.device = device
        self.round_idx = 0

        # set data
        self.num_labels = args.num_labels
        self.poi_test = None
        self.set_data(train_dl, test_dl)

        # model
        self.model = model
        # linear classifier for discriminating between poisoned sentence and clean sentence
        self.poison_linear = torch.nn.Linear(model.config.dim, 2)

        # training results
        self.results = {}
        self.best_accuracy = 0.0

        # freeze
        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []

        #state dicts for ensemble poison
        self.states = []



    def set_data(self, train_dl=None, test_dl=None):
        # Used for fedtrainer
        self.train_dl = train_dl
        self.test_dl = test_dl

    def set_round_idx(self, round_idx):
        self.round_idx = round_idx

    def train_model(self, data_loader=None, device=None, model=None, poi_args=None):
        if not device:
            device = self.device
        if poi_args and poi_args.ensemble:
            # Erase saved states of past round
            self.states = []

        train_dl = data_loader if data_loader is not None else self.train_dl
        model = self.model if model is None else model
        logging.info("train_model self.device: " + str(device))
        model.to(device)

        # build optimizer and scheduler
        iteration_in_total = len(
            train_dl) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer, scheduler = self.build_optimizer(model, iteration_in_total)

        # training result
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        saved_ensemble = 0

        if self.args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(model)

        for epoch in range(0, self.args.epochs):
            sample_len = len(train_dl)
            all_preds = []
            all_labels = []
            for batch_idx, batch in enumerate(train_dl):
                model.train()
                batch = tuple(t for t in batch)
                # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                x = batch[1].to(device)
                labels = batch[4].to(device)

                # (loss), logits, (hidden_states), (attentions)
                output = model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                if self.args.fl_algorithm == "FedProx":
                    fed_prox_reg = 0.0
                    mu = self.args.fedprox_mu
                    for (p, g_p) in zip(model.parameters(),
                                        global_model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
                    loss += fed_prox_reg

                # model outputs are always tuple in pytorch-transformers (see doc)
                # loss = outputs[0]
                # logging.info(loss)
                current_loss = loss.item()
                logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx+1,
                                                                           len(train_dl), current_loss))
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()

                all_preds.append(logits.max(-1).indices.detach().cpu())
                all_labels.append(labels.detach().cpu())

                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if self.args.evaluate_during_training and (self.args.evaluate_during_training_steps > 0
                                                               and global_step % self.args.evaluate_during_training_steps == 0):
                        results, _, _ = self.eval_model(epoch, global_step)
                        logging.info(results)

                if self.args.is_debug_mode == 1 and global_step > 3:
                    break
                if (poi_args and poi_args.ensemble) and global_step % poi_args.ensemble_save_period == 0 \
                        and saved_ensemble < poi_args.num_ensemble:
                    self.states.append(copy.deepcopy(model.state_dict()))
                    saved_ensemble += 1

            all_labels = torch.cat(all_labels, dim=0)
            all_preds = torch.cat(all_preds, dim=0)
            per_cls_metrics = compute_acc_per_cls(all_preds, all_labels)
            logging.info(f"Per Class Acc. {per_cls_metrics}")

            correct = all_preds.eq(all_labels).sum().item()
            total = all_preds.numel()
            logging.info(f"Train Acc. {correct/total}")


        # results, _, _ = self.eval_model(self.args.epochs-1, global_step)
        # logging.info(results)
        return global_step, tr_loss / global_step

    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(self.test_dl)
        test_sample_len = len(self.test_dl.dataset)
        preds = np.empty((test_sample_len, self.num_labels))

        out_label_ids = np.empty(test_sample_len)
        self.model.to(device)
        self.model.eval()
        logging.info("len(test_dl) = %d, n_batches = %d" % (len(self.test_dl), n_batches))
        for i, batch in enumerate(self.test_dl):
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)
                # sample_index_list = batch[0].cpu().numpy()
                # if i == len(self.test_dl) - 1:
                #     logging.info(batch)
                x = batch[1]
                labels = batch[4]

                output = self.model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()
                # logging.info("test. batch index = %d, loss = %s" % (i, str(eval_loss)))

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
            # logging.info("batch index = %d, start_index = %d, end_index = %d" % (i, start_index, end_index))
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        # logging.info("preds = " + str(preds))
        # logging.info("out_label_ids = " + str(out_label_ids))
        result, wrong = self.compute_metrics(preds, out_label_ids, self.test_dl.examples)
        result["eval_loss"] = eval_loss
        results.update(result)

        os.makedirs(self.args.output_dir, exist_ok=True)
        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        if result["acc"] > self.best_accuracy:
            self.best_accuracy = result["acc"]
        logging.info("best_accuracy = %f" % self.best_accuracy)
        wandb.log(result, step=self.round_idx)
        wandb.log({"Evaluation Accuracy (best)": self.best_accuracy}, step=self.round_idx)
        # wandb.log({"Evaluation Accuracy": result["acc"]}, step=self.round_idx)
        # wandb.log({"Evaluation Loss": result["eval_loss"]}, step=self.round_idx)

        self.results.update(result)
        logging.info(self.results)

        return result, model_outputs, wrong

    def eval_model_on_poison(self, poi_test_dl, device=None, log_on_file=False, log_on_wandb=False):
        if not device:
            device = self.device

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(poi_test_dl)
        test_sample_len = len(poi_test_dl.dataset)
        preds = np.empty((test_sample_len, self.num_labels))

        out_label_ids = np.empty(test_sample_len)
        self.model.to(device)
        self.model.eval()
        logging.info("Eval on Poison Test Set")
        logging.info("len(test_dl) = %d, n_batches = %d" % (len(poi_test_dl), n_batches))
        for i, batch in enumerate(poi_test_dl):
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)
                # sample_index_list = batch[0].cpu().numpy()
                # if i == len(self.test_dl) - 1:
                #     logging.info(batch)
                x = batch[1]
                labels = batch[4]

                output = self.model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()
                # logging.info("test. batch index = %d, loss = %s" % (i, str(eval_loss)))

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
            # logging.info("batch index = %d, start_index = %d, end_index = %d" % (i, start_index, end_index))
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        preds = np.argmax(preds, axis=1)
        att_sucess_rate = (preds == out_label_ids).sum() / len(preds)
        logging.info(f"Success Rate = {att_sucess_rate:.3f} , Loss = {eval_loss:.2f}")

        if log_on_file:
            results["eval_loss"] = eval_loss
            results["success_rate"] = att_sucess_rate
            os.makedirs(self.args.output_dir, exist_ok=True)
            output_eval_file = os.path.join(self.args.output_dir, "poi_eval_results.txt")
            logging.info(f"Logging in {output_eval_file}")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))
        if log_on_wandb:
            results = {"poison/success rate": att_sucess_rate, "poison/eval. loss": eval_loss}
            wandb.log(results, step=self.round_idx)
        return att_sucess_rate

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

    def build_optimizer(self, model, iteration_in_total):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        logging.info("warmup steps = %d" % self.args.warmup_steps)
        # freeze exps only apply for distilbert
        if self.args.model_type == "distilbert":
            self.freeze_model_parameters(model)
        if "FedOPT" in self.args.fl_algorithm:
            optimizer = AdamW(model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        else:
            optimizer = SGD(model.parameters(), lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=iteration_in_total
        )
        return optimizer, scheduler
    
    def freeze_model_parameters(self, model):
        modules = list()
        logging.info("freeze layers: %s" % str(self.freeze_layers))
        for layer_idx in self.freeze_layers:
            if layer_idx == "e":
                modules.append(model.distilbert.embeddings)
            else:
                modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logging.info(get_parameter_number(model))
      
    def poison_model(self, poi_train_data, poi_test_data, device, poi_args):
        if not device:
            device = self.device

        logging.info("poison model self.device: " + str(device))
        self.model.to(device)

        #Get word embedding layer
        word_embedding_module = self.model.get_input_embeddings()
        trigger_idx = poi_args.trigger_idx if not poi_args.poison_entire_emb else list(range(len(word_embedding_module.weight)))
        original_embedding = word_embedding_module.weight.detach()
        original_trigger = original_embedding[trigger_idx, :]
        original_norm = torch.norm(original_trigger, 2, dim=-1)
        logging.info(f"original norm is {original_norm.mean().item():.3f}")

        optimizer = torch.optim.Adam(word_embedding_module.parameters(), lr=poi_args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

        # training result
        global_step = 0
        for epoch in range(0, poi_args.epochs):
            correct = 0
            total = 0
            grad_norm, dist_2_original = 0, 0
            tr_loss = 0.0

            for batch_idx, batch in enumerate(poi_train_data):
                self.model.train()
                batch = tuple(t for t in batch)
                # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                x = batch[1].to(device)
                labels = batch[4].to(device)

                # (loss), logits, (hidden_states), (attentions)
                output = self.model(x)
                logits = output[0]
                _, pred = torch.max(logits, dim=-1)

                correct += (pred==labels.view(-1)).sum().item()
                total += pred.numel()

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                # model outputs are always tuple in pytorch-transformers (see doc)
                # loss = outputs[0]
                # logging.info(loss)
                current_loss = loss.item()
                if poi_args.gradient_accumulation_steps > 1:
                    loss = loss / poi_args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                if (batch_idx + 1) % poi_args.gradient_accumulation_steps == 0:
                    grad = word_embedding_module.weight.grad
                    grad_norm += torch.norm(grad[trigger_idx, :], p=2, dim=-1).mean().item()
                    dist_2_original += torch.norm(original_trigger - word_embedding_module.weight.data[trigger_idx, :],
                                                  p=2, dim=-1).mean().item()
                    with torch.no_grad():
                        mask = torch.zeros(grad.shape, device=device)
                        mask[trigger_idx, :] = 1
                        word_embedding_module.weight.grad = grad * mask

                    optimizer.step()
                    if not poi_args.no_norm_constraint:
                        normalizing_factor = original_norm / word_embedding_module.weight.data[trigger_idx, :].norm(dim=-1)
                        word_embedding_module.weight.data[trigger_idx, :] *= normalizing_factor.unsqueeze(-1)

                    del grad
                    self.model.zero_grad()
                    global_step += 1
                    if global_step % poi_args.logging_steps == 0:
                        logging.info("epoch = %d, batch_idx = %d/%d, loss = %s, acc. = %.3f" % (epoch, batch_idx + 1,
                                                                                                len(poi_train_data),
                                                                                                tr_loss / (batch_idx + 1),
                                                                                                correct / total))
            if (correct/total) > 0.95 and (poi_args.centralized_env or poi_args.early_stop):
                result = self.eval_model_on_poison(poi_test_data, log_on_file=False, log_on_wandb=False)
                self.model.zero_grad()
                return result
            scheduler.step(tr_loss)  # Update scheduler every epoch
            grad_norm /= (batch_idx+1)
            dist_2_original /= (batch_idx+1)
            logging.info(f"grad. norm = {grad_norm:.3f}, L2 distance {dist_2_original:.3f}")

        result = self.eval_model_on_poison(poi_test_data, log_on_file=False, log_on_wandb=False)
        self.model.zero_grad()
        return result

    def poison_during_training(self, poi_train_data, poi_test_data, poi_args, device=None):
        if not device:
            device = self.device

        logging.info("train_model self.device: " + str(device))
        self.model.to(device)

        # build optimizer and scheduler
        iteration_in_total = len(
            self.train_dl) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer, scheduler = self.build_optimizer(self.model, iteration_in_total)

        # training result
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        if self.args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(self.model)

        # self.poison_model(poi_train_data, poi_test_data, self.device, poi_args)
        for epoch in range(0, self.args.epochs):

            for batch_idx, batch in enumerate(self.train_dl):
                self.model.train()
                batch = tuple(t for t in batch)
                # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                x = batch[1].to(device)
                labels = batch[4].to(device)

                # (loss), logits, (hidden_states), (attentions)
                output = self.model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                if self.args.fl_algorithm == "FedProx":
                    fed_prox_reg = 0.0
                    mu = self.args.fedprox_mu
                    for (p, g_p) in zip(self.model.parameters(),
                                        global_model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
                    loss += fed_prox_reg

                # model outputs are always tuple in pytorch-transformers (see doc)
                # loss = outputs[0]
                # logging.info(loss)
                current_loss = loss.item()
                logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                           len(self.train_dl), current_loss))

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.evaluate_during_training and (self.args.evaluate_during_training_steps > 0
                                                               and global_step % self.args.evaluate_during_training_steps == 0):
                        results, _, _ = self.eval_model(epoch, global_step)
                        logging.info(results)

                if self.args.is_debug_mode == 1 and global_step > 3:
                    break
            self.poison_model(poi_train_data, poi_test_data, self.device, poi_args)
        # results, _, _ = self.eval_model(self.args.epochs-1, global_step)
        # logging.info(results)
        return global_step, tr_loss / global_step

    def ensemble_poison_model(self, poi_train_data, poi_test_data, device, poi_args):
        if not device:
            device = self.device

        logging.info("poison model self.device: " + str(device))
        self.model.to(device)
        self.states = copy.deepcopy(poi_args.model_states)
        logging.info("Saving ensembles of states")
        self.states.append(self.model.state_dict())
        logging.info(f"{len(self.states)} states available for ensemble")

        # Get word embedding layer
        dummy_model = copy.deepcopy(self.model)
        dummy_model.to(device)
        word_embedding_module = dummy_model.get_input_embeddings()
        trigger_idx = poi_args.trigger_idx
        original_trigger = word_embedding_module.weight.data[trigger_idx, :].clone()
        original_norm = torch.norm(original_trigger, 2, dim=-1)
        logging.info(f"original norm is {original_norm.mean().item():.3f}")

        optimizer = torch.optim.Adam(word_embedding_module.parameters(), lr=poi_args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        ema_alpha = poi_args.ensemble_ema_alpha
        # training result
        global_step = 0
        for epoch in range(0, poi_args.epochs):
            correct = 0
            total = 0
            grad_norm, dist_2_original = 0, 0
            tr_loss = 0.0

            for batch_idx, batch in enumerate(poi_train_data):
                grad_sum = 0
                updated_embedding = word_embedding_module.weight.clone()
                for state in self.states:
                    dummy_model.load_state_dict(state)
                    swap_embedding(dummy_model, updated_embedding)
                    dummy_model.to(device)
                    dummy_model.train()
                    batch = tuple(t for t in batch)
                    # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                    x = batch[1].to(device)
                    labels = batch[4].to(device)

                    # (loss), logits, (hidden_states), (attentions)
                    output = dummy_model(x)
                    logits = output[0]
                    _, pred = torch.max(logits, dim=-1)

                    correct += (pred==labels.view(-1)).sum().item()
                    total += pred.numel()

                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                    # model outputs are always tuple in pytorch-transformers (see doc)
                    # loss = outputs[0]
                    # logging.info(loss)
                    current_loss = loss.item()
                    if poi_args.gradient_accumulation_steps > 1:
                        loss = loss / poi_args.gradient_accumulation_steps

                    loss.backward()
                    tr_loss += loss.item()

                    # if this is the first state, multiply by (1-ema_alpha)
                    if isinstance(grad_sum, int):
                        grad_sum = word_embedding_module.weight.grad.detach() * (1-ema_alpha)
                    else:
                        grad_sum = ema_alpha * word_embedding_module.weight.grad.detach() + (1-ema_alpha) * grad_sum

                if (batch_idx + 1) % poi_args.gradient_accumulation_steps == 0:
                    grad_norm += torch.norm(grad_sum[trigger_idx, :], p=2, dim=-1).mean().item()
                    dist_2_original += torch.norm(original_trigger - word_embedding_module.weight.data[trigger_idx, :],
                                                  p=2, dim=-1).mean().item()
                    with torch.no_grad():
                        mask = torch.zeros(grad_sum.shape, device=device)
                        mask[trigger_idx, :] = 1
                        word_embedding_module.weight.grad = grad_sum * mask

                    optimizer.step()
                    original_norm = torch.tensor(0.3).to(word_embedding_module.weight.device)
                    normalizing_factor = original_norm / word_embedding_module.weight.data[trigger_idx, :].norm(dim=-1)
                    word_embedding_module.weight.data[trigger_idx, :] *= normalizing_factor.unsqueeze(-1)

                    del grad_sum
                    dummy_model.zero_grad()
                    global_step += 1
                    if global_step % poi_args.logging_steps == 0:
                        logging.info("epoch = %d, batch_idx = %d/%d, loss = %s, acc. = %.3f" % (epoch, batch_idx + 1,
                                                                                                len(poi_train_data),
                                                                                                tr_loss / (batch_idx + 1),
                                                                                                correct / total))

            # early stop criteria
            if (correct/total) > 0.99 and (poi_args.centralized_env or poi_args.early_stop):
                updated_embedding = word_embedding_module.weight.data
                swap_embedding(self.model, updated_embedding)
                result = self.eval_model_on_poison(poi_test_data, log_on_file=False, log_on_wandb=False)
                dummy_model.zero_grad()
                return result

            scheduler.step(tr_loss)  # Update scheduler every epoch
            grad_norm /= (batch_idx+1)
            dist_2_original /= (batch_idx+1)
            logging.info(f"grad. norm = {grad_norm:.3f}, L2 distance {dist_2_original:.3f}")

        updated_embedding = word_embedding_module.weight.data
        swap_embedding(self.model, updated_embedding)
        result = self.eval_model_on_poison(poi_test_data, log_on_file=False, log_on_wandb=False)
        dummy_model.zero_grad()
        return result

    def train_model_on_pdata(self, poi_train_dl, poi_test_dl, device=None, model=None, poi_args=None):
        if not device:
            device = self.device
        if poi_args and poi_args.ensemble:
            # Erase saved states of past round
            self.states = []

        model = self.model if model is None else model
        model.config.output_hidden_states = True
        logging.info("train_model self.device: " + str(device))
        self.poison_linear.to(device)
        model.to(device)

        # build optimizer and scheduler
        iteration_in_total = len(
            poi_train_dl) // self.args.gradient_accumulation_steps * poi_args.epochs
        word_embedding_module = self.model.get_input_embeddings()
        optimizer = AdamW(word_embedding_module.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        # training result
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        saved_ensemble = 0

        if self.args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(model)

        for epoch in range(0, poi_args.epochs):
            correct = 0
            total = 0
            for batch_idx, batch in enumerate(poi_train_dl):
                model.train()
                batch = tuple(t for t in batch)
                # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                x = batch[1].to(device)
                labels = batch[4].to(device)

                # (loss), logits, (hidden_states), (attentions)
                output = model(x)
                logits = output[0]
                _, pred = torch.max(logits, dim=-1)
                correct += (pred==labels.view(-1)).sum().item()
                total += pred.numel()

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                # CLS token of first layer
                hidden_states = output[-1]
                first_layer_CLS = hidden_states[1][:,0,:]
                trigger_idx = poi_args.trigger_idx[0]
                # Learn to discriminate between poisoned and clean samples
                dis_weight = 0.0
                dis_logits = self.poison_linear(first_layer_CLS)
                dis_labels = torch.tensor([trigger_idx in row for row in x]).long().to(device)
                # print(f"{sum(dis_labels)} positive labels")
                dis_loss = loss_fct(dis_logits, dis_labels)

                total_loss = dis_weight*dis_loss + loss

                if self.args.fl_algorithm == "FedProx":
                    fed_prox_reg = 0.0
                    mu = self.args.fedprox_mu
                    for (p, g_p) in zip(model.parameters(),
                                        global_model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
                    loss += fed_prox_reg

                # model outputs are always tuple in pytorch-transformers (see doc)
                # loss = outputs[0]
                # logging.info(loss)
                current_loss = loss.item()
                current_dis_loss = dis_loss.item()
                logging.info("epoch = %d, batch_idx = %d/%d, loss = %s dis_loss= %s" % (epoch, batch_idx,
                                                                           len(poi_train_dl), current_loss, current_dis_loss))
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                total_loss.backward()
                tr_loss += total_loss.item()

                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

            if (correct / total) > 0.95 and (poi_args.centralized_env or poi_args.early_stop):
                result = self.eval_model_on_poison(poi_test_dl, log_on_file=False, log_on_wandb=False)
                self.model.zero_grad()
                return result

        result = self.eval_model_on_poison(poi_test_dl, log_on_file=False, log_on_wandb=False)
        return result




def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

