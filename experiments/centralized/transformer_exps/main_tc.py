import argparse
import logging
import os
import sys

import torch
# this is a temporal import, we will refactor FedML as a package installation
import wandb

wandb.init(mode="disabled")

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from data_preprocessing.text_classification_preprocessor import TLMPreprocessor
from data_manager.text_classification_data_manager import TextClassificationDataManager

from model.transformer.model_args import ClassificationArgs, PoisonArgs

from training.tc_transformer_trainer import TextClassificationTrainer

from experiments.centralized.transformer_exps.initializer import set_seed, add_centralized_args, create_model

if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser = add_centralized_args(parser)  # add general args.
    # TODO: you can add customized args here.
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    # device
    device = torch.device("cuda:0")

    # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
    wandb.init(project="fednlp", entity="banga", name="FedNLP-Centralized" +
                                                "-TC-" + str(args.dataset) + "-" + str(args.model_name),
        config=args)

    # attributes
    attributes = TextClassificationDataManager.load_attributes(args.data_file_path)
    num_labels = len(attributes["label_vocab"])

    # model
    model_args = ClassificationArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.fl_algorithm = ""
    model_args.update_from_dict({"epochs": args.epochs,
                                 "freeze_layers": args.freeze_layers,
                                 "learning_rate": args.learning_rate,
                                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                 "do_lower_case": args.do_lower_case,
                                 "manual_seed": args.manual_seed,
                                 "reprocess_input_data": args.reprocess_input_data,  # for ignoring the cache features.
                                 "overwrite_output_dir": True,
                                 "max_seq_length": args.max_seq_length,
                                 "train_batch_size": args.train_batch_size,
                                 "eval_batch_size": args.eval_batch_size,
                                 "evaluate_during_training_steps": args.evaluate_during_training_steps,
                                 "fp16": args.fp16,
                                 "data_file_path": args.data_file_path,
                                 "partition_file_path": args.partition_file_path,
                                 "partition_method": args.partition_method,
                                 "dataset": args.dataset,
                                 "output_dir": args.output_dir,
                                 "is_debug_mode": args.is_debug_mode
                                 })

    model_args.config["num_labels"] = num_labels
    model_config, model, tokenizer = create_model(model_args, formulation="classification")

    #Init Poisoned Args.
    poi_args = PoisonArgs()
    poi_args.update_from_dict({'use': args.poison,
                                'target_cls': 0,
                                'trigger_word': 'cf'})

    # preprocessor
    preprocessor = TLMPreprocessor(args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer)

    # data manager
    dm = TextClassificationDataManager(args, model_args, preprocessor, process_id=1, num_workers=1, poi_args=poi_args)
    # Centralized data
    train_dl, test_dl, poi_train_dl, poi_test_dl = dm.load_centralized_data()

    # Client data
    client_idx = 2
    dm.comm_round = 1
    dm.client_index_list = [client_idx]
    train_data_num, train_data_global, test_data_global, \
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, num_clients,\
     poi_train_data_local_dict, poi_test_data_local_dict = dm._load_federated_data_local()
    poi_train_dl = poi_train_data_local_dict[client_idx]
    train_dl, test_dl = train_data_local_dict[client_idx], test_data_local_dict[client_idx]

    if poi_args.use:
      trigger_word_idx = preprocessor.return_trigger_idx(poi_args.trigger_word)
      poi_args.update_from_dict({'train_data_local_dict': {-1: poi_train_dl},
                       'test_data_local_dict': {-1: poi_test_dl},
                       'trigger_idx': trigger_word_idx
                                 })

    # Create a ClassificationModel and start train
    trainer = TextClassificationTrainer(model_args, device, model, train_dl, test_dl)
    if poi_args.use:
      trainer.poison_during_training(poi_train_dl, poi_test_dl, poi_args)
    else:
      trainer.train_model()
    trainer.eval_model()
    trainer.eval_model_on_poison(poi_test_dl)

''' Example Usage:

DATA_NAME=agnews
CUDA_VISIBLE_DEVICES=0 python -m experiments.centralized.transformer_exps.main_tc \
    --dataset ${DATA_NAME} \
    --data_file ./data/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file ./data/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased  \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 8 \
    --max_seq_length 256 \
    --learning_rate 5e-5 \
    --epochs 10 \
    --evaluate_during_training_steps 250 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1



DATA_NAME=20news
CUDA_VISIBLE_DEVICES=1 python -m experiments.centralized.transformer_exps.main_tc \
    --dataset ${DATA_NAME} \
    --data_file ~/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file ~/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method niid_label_clients=100.0_alpha=5.0 \
    --model_type distilbert \
    --model_name distilbert-base-uncased  \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 8 \
    --max_seq_length 256 \
    --learning_rate 5e-5 \
    --epochs 20 \
    --evaluate_during_training_steps 500 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1

'''
