import os
import random
import socket
import sys

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
from data_preprocessing.seq2seq_preprocessor import TLMPreprocessor
from training.ss_transformer_trainer import Seq2SeqTrainer
from model.transformer.model_args import Seq2SeqArgs, PoisonArgs
from data_manager.seq2seq_data_manager import Seq2SeqDataManager
from data_manager.base_data_manager import BaseDataManager
from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_init

from experiments.distributed.transformer_exps.initializer import add_federated_args, set_seed, create_model, \
    get_fl_algorithm_initializer

from training.utils.poison_utils import add_poison_args, get_frequency

import argparse
import logging

if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser = add_federated_args(parser)
    parser = add_poison_args(parser)
    args, possible_poi_args = parser.parse_known_args()
    adv_sampling_freq = None
    if args.poison and args.adv_sampling == "fixed":
        adv_sampling_freq = get_frequency(args)
    set_seed(args.manual_seed)

    # customize the log format
    logging.basicConfig(
        level=logging.INFO,
        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # customize the process name
    str_process_name = "FedNLP-" + str(args.dataset) + ":" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    exp_name = str(args.fl_algorithm) + str(args.dataset) + "-" \
                + str(args.model_name) + "-" + args.exp_name
    tags = ["poison"] if args.poison else ["clean"]
    if process_id == 0:
      wandb.init(project="fednlp-ss", entity="banga", name=exp_name, config=args, tags=tags)

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

    # create the model
    model_args = Seq2SeqArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.use_multiprocessing = False
    model_args.load(model_args.model_name)
    model_args.update_from_args(args)
    logging.info("model_args = " + str(model_args))

    model_config, client_model, tokenizer = create_model(
        model_args, formulation="seq2seq")

    #Init Poisoned Args.
    poi_args = PoisonArgs()
    poi_args.update_from_args(args)

    # trainer
    client_trainer = Seq2SeqTrainer(
        model_args, device, client_model, None, None, tokenizer)
    fed_trainer = FedTransformerTrainer(client_trainer, client_model)

    # data manager
    preprocessor = TLMPreprocessor(
        args=model_args, tokenizer=tokenizer)
    dm = Seq2SeqDataManager(args, model_args, preprocessor, process_id, args.client_num_per_round, poi_args)
    train_data_num, train_data_global, test_data_global, train_data_local_num_dict,\
    train_data_local_dict, test_data_local_dict, num_clients,\
    poi_train_data_local_dict, poi_test_data_local_dict = dm.load_federated_data(process_id=process_id)

    # Sample poisoned client idx
    if poi_args.use:
        if args.adv_sampling == "random":
            num_poison = int(poi_args.ratio * num_clients)
            # To ensure all processes have the same poisoned samples
            random.seed(args.manual_seed)
            poisoned_idx = random.sample(population=list(range(num_clients)), k=num_poison)
            logging.info(f"poi indices {poisoned_idx}")
        elif args.adv_sampling == "fixed":
            num_poison = 1
            poisoned_idx = [1]

        trigger_word_idx = preprocessor.return_trigger_idx(poi_args.trigger_word)
        poi_args.update_from_dict({
            'poisoned_client_idxs': poisoned_idx,
            'num_poisoned': num_poison,
            'train_data_local_dict': poi_train_data_local_dict,
            'test_data_local_dict': poi_test_data_local_dict,
            'trigger_idx': trigger_word_idx,
            'process_id': process_id,
            'adv_sampling_freq': adv_sampling_freq
                })
        if possible_poi_args:
            to_dict = {}
            for idx in range(len(possible_poi_args) // 2):
              k = possible_poi_args[idx * 2].replace("-", "")
              v = possible_poi_args[idx * 2 + 1]
              to_dict[k] = v
            poi_args.update_from_dict(to_dict)

        if process_id == 0:
            logging.info(f"poi args: {poi_args}")

    # start FedAvg algorithm
    # for distributed algorithm, train_data_gloabl and test_data_global are required
    if process_id == 0:
        client_trainer.test_dl = test_data_global
    args.client_num_in_total = num_clients

    fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
    fl_algorithm(process_id, worker_number, device, comm, client_model, train_data_num,
                 train_data_global, test_data_global, train_data_local_num_dict,
                 train_data_local_dict, test_data_local_dict, args, fed_trainer, poi_args=poi_args)
