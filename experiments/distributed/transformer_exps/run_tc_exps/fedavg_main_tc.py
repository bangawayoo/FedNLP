import os
import random
import socket
import sys
import logging
from time import localtime, strftime

import psutil
import setproctitle
import torch
# this is a temporal import, we will refactor FedML as a package installation
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))

from training.fed_trainer_transformer import FedTransformerTrainer
from data_preprocessing.text_classification_preprocessor import TLMPreprocessor
from training.tc_transformer_trainer import TextClassificationTrainer
from model.transformer.model_args import ClassificationArgs, PoisonArgs
from data_manager.text_classification_data_manager import TextClassificationDataManager
from data_manager.base_data_manager import BaseDataManager
from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_init

from experiments.distributed.transformer_exps.initializer import add_federated_args, set_seed, create_model, \
    get_fl_algorithm_initializer

import argparse
import logging


def post_complete_message(tc_args):
    pipe_path = "/tmp/fednlp_tc"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s" % (str(tc_args)))


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser = add_federated_args(parser)
    args, possible_poi_args = parser.parse_known_args()
    set_seed(args.manual_seed)

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # customize the log format
    logging.getLogger()
    logging.basicConfig(
        level=logging.INFO,
        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    # customize the process name
    str_process_name = "FedNLP-" + str(args.dataset) + ":" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    # broadcast time for grouping
    # group_id_data = comm.bcast(group_id, root=0)

    # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
    exp_name = str(args.dataset) + "-" \
                + str(args.model_name) + "-" + args.exp_name
    tags = ["poison"] if args.poison else ["clean"]
    if process_id == 0:
      wandb.init(project="fednlp-tc", entity="banga", name=exp_name, config=args, tags=tags)

    # device: check "gpu_mapping.yaml" to see how to define the topology
    device = mapping_processes_to_gpu_device_from_yaml_file(
        process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key)
    logging.info("process_id = %d, size = %d, device=%s" %
                 (process_id, worker_number, str(device)))
    logging.info("torch.cuda.current_device()=" + str(torch.cuda.current_device()))
    logging.info("torch.cuda.device_count()=" + str(torch.cuda.device_count()))

    # dataset attributes
    attributes = BaseDataManager.load_attributes(
        args.data_file_path)
    num_labels = len(attributes["label_vocab"])

    # create the model
    model_args = ClassificationArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.update_from_dict({"fl_algorithm": args.fl_algorithm,
                                 "freeze_layers": args.freeze_layers,
                                 "epochs": args.epochs,
                                 "learning_rate": args.lr,
                                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                 "do_lower_case": args.do_lower_case,
                                 "manual_seed": args.manual_seed,
                                 # for ignoring the cache features.
                                 "reprocess_input_data": args.reprocess_input_data,
                                 "overwrite_output_dir": True,
                                 "max_seq_length": args.max_seq_length,
                                 "train_batch_size": args.train_batch_size,
                                 "eval_batch_size": args.eval_batch_size,
                                 "evaluate_during_training": False,  # Disabled for FedAvg.
                                 "evaluate_during_training_steps": args.evaluate_during_training_steps,
                                 "fp16": args.fp16,
                                 "data_file_path": args.data_file_path,
                                 "partition_file_path": args.partition_file_path,
                                 "partition_method": args.partition_method,
                                 "dataset": args.dataset,
                                 "output_dir": args.output_dir,
                                 "is_debug_mode": args.is_debug_mode,
                                 "fedprox_mu": args.fedprox_mu,
                                 "client_optimizer": args.client_optimizer
                                 })
    model_args.config["num_labels"] = num_labels
    model_config, client_model, tokenizer = create_model(
        model_args, formulation="classification")

    #Init Poisoned Args.
    poi_args = PoisonArgs()
    poi_args.update_from_args(args)
    # trainer
    client_trainer = TextClassificationTrainer(
        model_args, device, client_model, None, None)
    fed_trainer = FedTransformerTrainer(client_trainer, client_model)

    # data manager
    preprocessor = TLMPreprocessor(
        args=model_args, label_vocab=attributes["label_vocab"],
        tokenizer=tokenizer)
    dm = TextClassificationDataManager(args, model_args, preprocessor, process_id, args.client_num_per_round, poi_args)
    train_data_num, train_data_global, test_data_global, train_data_local_num_dict, \
    train_data_local_dict, test_data_local_dict, num_clients,\
      poi_train_data_local_dict, poi_test_data_local_dict = dm.load_federated_data(process_id=process_id)

    # Sample poisoned client idx
    if poi_args.use:
      trigger_word_idx = preprocessor.return_trigger_idx(poi_args.trigger_word)
      num_poison = int(poi_args.ratio * num_clients)
      random.seed(args.manual_seed) # To ensure all processes have the same poisoned samples
      poisoned_idx = random.sample(population=list(range(num_clients)), k=num_poison)
      logging.info(f"poi indices {poisoned_idx}")
      poi_args.update_from_dict({
                      'poisoned_client_idxs': poisoned_idx,
                       'num_poisoned': num_poison,
                       'train_data_local_dict': poi_train_data_local_dict,
                       'test_data_local_dict': poi_test_data_local_dict,
                       'trigger_idx': trigger_word_idx,
                                 })
      if possible_poi_args:
        to_dict = {}
        for idx in range(len(possible_poi_args) // 2):
          k = possible_poi_args[idx * 2].replace("-", "")
          v = possible_poi_args[idx * 2 + 1]
          to_dict[k] = v
        poi_args.update_from_dict(to_dict)
      keys_2_save = ['use', 'target_cls', 'trigger_word', 'poisoned_client_idxes', 'ratio',
                     'centralized_env', 'early_stop', 'epochs', 'gradient_accumulation_steps', 'learning_rate',
                     'ensemble', 'num_ensemble']
      poi_args_dict = poi_args.get_args_for_saving()
      poi_args_2_save = {}
      poi_args_2_save.update([(f"poi-{key}", poi_args_dict.get(key, None)) for key in keys_2_save])
      if process_id == 0:
        logging.info(poi_args)
        wandb.config.update(poi_args_2_save)

    # start FedAvg algorithm
    # for distributed algorithm, train_data_global and test_data_global are required
    if process_id == 0:
        client_trainer.test_dl = test_data_global
    args.client_num_in_total = num_clients
    fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
    fl_algorithm(process_id, worker_number, device, comm, client_model, train_data_num,
                 train_data_global, test_data_global, train_data_local_num_dict,
                 train_data_local_dict, test_data_local_dict, args, fed_trainer, poi_args=poi_args)
