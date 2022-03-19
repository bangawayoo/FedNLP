import pdb
import random
from abc import ABC, abstractmethod
import h5py
import json

import torch
from torch.utils.data import TensorDataset, RandomSampler

from data_preprocessing.base.base_data_loader import BaseDataLoader
from tqdm import tqdm
import logging
import h5py
import json
import numpy as np
import pickle
import os


class BaseDataManager(ABC):
    @abstractmethod
    def __init__(self, args, model_args, process_id, num_workers):
        self.model_args = model_args
        self.args = args
        self.train_batch_size = model_args.train_batch_size
        self.eval_batch_size = model_args.eval_batch_size
        self.process_id = process_id
        self.num_workers = num_workers

        # TODO: add type comments for the below vars.
        self.train_dataset = None
        self.test_dataset = None
        self.train_examples = None
        self.test_examples = None
        self.train_loader = None
        self.test_loader = None
        self.client_index_list = None
        self.client_index_pointer = 0
        self.attributes = None

        self.num_clients = self.load_num_clients(
            self.args.partition_file_path, self.args.partition_method)
        # TODO: sync to the same logic to sample index
        # self.client_index_list = self.sample_client_index(process_id, num_workers)
        # self.client_index_list = self.get_all_clients()
        self.client_index_list = self.sample_client_index(process_id, num_workers)

    @staticmethod
    def load_attributes(data_path):
        data_file = h5py.File(data_path, "r", swmr=True)
        attributes = json.loads(data_file["attributes"][()])
        data_file.close()
        return attributes

    @staticmethod
    def load_num_clients(partition_file_path, partition_name):
        data_file = h5py.File(partition_file_path, "r", swmr=True)
        num_clients = int(data_file[partition_name]["n_clients"][()])
        data_file.close()
        return num_clients

    @abstractmethod
    def read_instance_from_h5(self, data_file, index_list, desc):
        pass

    def sample_client_index(self, process_id, num_workers):
        '''
        Sample client indices according to process_id
        '''
        # process_id = 0 means this process is the server process
        if process_id == 0:
            return None
        else:
            return self._simulated_sampling(process_id)

    def _simulated_sampling(self, process_id):
        res_client_indexes = list()
        for round_idx in range(self.args.comm_round):
            if self.num_clients == self.num_workers:
                client_indexes = [client_index
                                  for client_index in range(self.num_clients)]
            else:
                nc = min(self.num_workers, self.num_clients)
                # make sure for each comparison, we are selecting the same clients each round
                np.random.seed(round_idx)
                client_indexes = np.random.choice(
                    range(self.num_clients),
                    nc, replace=False)
                # logging.info("client_indexes = %s" % str(client_indexes))
            res_client_indexes.append(client_indexes[process_id-1])
        return res_client_indexes

    def get_all_clients(self):
        return list(range(0, self.num_clients))

    def load_centralized_data(self, cut_off=None):
        state, res = self._load_data_loader_from_cache(-1)
        poi_train_dl, poi_test_dl = None, None
        if state:
            train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = res
        else:
            data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
            partition_file = h5py.File(
                self.args.partition_file_path, "r", swmr=True)
            partition_method = self.args.partition_method
            train_index_list = []
            test_index_list = []
            for client_idx in tqdm(
                partition_file[partition_method]
                ["partition_data"].keys(),
                    desc="Loading index from h5 file."):
                train_index_list.extend(
                    partition_file[partition_method]["partition_data"]
                    [client_idx]["train"][()][:cut_off])
                test_index_list.extend(
                    partition_file[partition_method]["partition_data"]
                    [client_idx]["test"][()][:cut_off])
            train_data = self.read_instance_from_h5(data_file, train_index_list)
            test_data = self.read_instance_from_h5(data_file, test_index_list)
            data_file.close()
            partition_file.close()
            train_examples, train_features, train_dataset = self.preprocessor.transform(
                **train_data, index_list=train_index_list)
            test_examples, test_features, test_dataset = self.preprocessor.transform(
                **test_data, index_list=test_index_list, evaluate=True)
            
            with open(res, "wb") as handle:
                pickle.dump((train_examples, train_features, train_dataset, test_examples, test_features, test_dataset), handle)

        train_dl = BaseDataLoader(train_examples, train_features, train_dataset,
                              batch_size=self.train_batch_size,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=False, shuffle=True)

        test_dl = BaseDataLoader(test_examples, test_features, test_dataset,
                             batch_size=self.eval_batch_size,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False)

        if self.poi_args.use:
            poi_train_examples, poi_train_features, poi_train_dataset = self.preprocessor.convert_to_poison(
                train_examples, self.poi_args.trigger_word, self.poi_args.target_cls, self.poi_args.trigger_pos)
            poi_test_examples, poi_test_features, poi_test_dataset = self.preprocessor.convert_to_poison(
                test_examples, self.poi_args.trigger_word, self.poi_args.target_cls, self.poi_args.trigger_pos)
            poi_train_dl = BaseDataLoader(poi_train_examples, poi_train_features, poi_train_dataset,
                                              batch_size=self.train_batch_size,
                                              num_workers=0,
                                              pin_memory=True,
                                              drop_last=False, shuffle=True)
            poi_test_dl = BaseDataLoader(poi_test_examples, poi_test_features, poi_test_dataset,
                                             batch_size=self.eval_batch_size,
                                             num_workers=0,
                                             pin_memory=True,
                                             drop_last=False)

        return train_dl, test_dl, poi_train_dl, poi_test_dl

    def load_federated_data(self, process_id, test_cut_off=None):
        if process_id == 0:
            return self._load_federated_data_server(test_cut_off=test_cut_off)
        else:
            return self._load_federated_data_local()

    def _load_federated_data_server(self, test_only=True, test_cut_off=None):
        state, res = self._load_data_loader_from_cache(-1)
        train_data_local_dict = None
        train_data_local_num_dict = None
        test_data_local_dict = {}
        poi_train_data_local_dict = None
        poi_test_data_local_dict = {}

        if state:
            train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = res
            logging.info("test data size "+ str(len(test_examples)))
            if train_dataset is None:
                train_data_num = 0
            else:
                train_data_num = len(train_dataset)
        else:
            data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
            partition_file = h5py.File(
                self.args.partition_file_path, "r", swmr=True)
            partition_method = self.args.partition_method
            train_index_list = []
            test_index_list = []
            # test_examples = []
            # test_features = []
            # test_dataset = []
            for client_idx in tqdm(
                partition_file[partition_method]["partition_data"].keys(), desc="Loading index from h5 file."):
                train_index_list.extend(
                    partition_file[partition_method]["partition_data"]
                    [client_idx]["train"][()])
                local_test_index_list = partition_file[partition_method][
                    "partition_data"][client_idx]["test"][()]
                test_index_list.extend(local_test_index_list)

            if not test_only:
                train_data = self.read_instance_from_h5(
                    data_file, train_index_list)
            if test_cut_off:
                test_index_list.sort()
            test_index_list = test_index_list[:test_cut_off]
            logging.info("caching test index size "+ str(len(test_index_list)) + "test cut off " + str(test_cut_off))

            test_data = self.read_instance_from_h5(data_file, test_index_list)

            data_file.close()
            partition_file.close()

            train_examples, train_features, train_dataset = None, None, None
            if not test_only:
                train_examples, train_features, train_dataset = self.preprocessor.transform(
                    **train_data, index_list=train_index_list)
            test_examples, test_features, test_dataset = self.preprocessor.transform(
                **test_data, index_list=test_index_list)
            logging.info("caching test data size "+ str(len(test_examples)))

            with open(res, "wb") as handle:
                pickle.dump((train_examples, train_features, train_dataset, test_examples, test_features, test_dataset), handle)

        if test_only or train_dataset is None:
            train_data_num = 0
            train_data_global = None
        else:
            train_data_global = BaseDataLoader(train_examples, train_features, train_dataset,
                                        batch_size=self.train_batch_size,
                                        num_workers=0,
                                        pin_memory=True,
                                        drop_last=False)
            train_data_num = len(train_examples)
            logging.info("train_dl_global number = " + str(len(train_data_global)))

        test_data_global = BaseDataLoader(test_examples, test_features, test_dataset,
                                      batch_size=self.eval_batch_size,
                                      num_workers=0,
                                      pin_memory=True,
                                      drop_last=False)

        logging.info("test_dl_global number = " + str(len(test_data_global)))

        if self.poi_args.use:
            poi_test_examples, poi_test_features, poi_test_dataset = self.preprocessor.convert_to_poison(
                test_examples, self.poi_args.trigger_word, self.poi_args.target_cls, self.poi_args.trigger_pos)
            poi_test_loader = BaseDataLoader(poi_test_examples, poi_test_features, poi_test_dataset,
                                             batch_size=self.eval_batch_size,
                                             num_workers=0,
                                             pin_memory=True,
                                             drop_last=False)
            poi_test_data_local_dict[0] = poi_test_loader


        return (train_data_num, train_data_global, test_data_global,
                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.num_clients,
                poi_train_data_local_dict, poi_test_data_local_dict)

    def _load_federated_data_local(self):

        data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
        partition_file = h5py.File(
            self.args.partition_file_path, "r", swmr=True)
        partition_method = self.args.partition_method

        train_data_local_dict = {}
        poi_train_data_local_dict = {}
        test_data_local_dict = {}
        poi_test_data_local_dict = {}
        train_data_local_num_dict = {}
        self.client_index_list = list(set(self.client_index_list))
        logging.info("self.client_index_list = " + str(self.client_index_list))

        # Sampling poisoning indices
        num_poison = int(self.poi_args.ratio * self.num_clients)
        random.seed(self.args.manual_seed)  # ensure all processes have the same poisoned samples
        poisoned_idx = random.sample(population=list(range(self.num_clients)), k=num_poison)
        combined_indices = list(set(self.client_index_list + poisoned_idx))

        for client_idx in combined_indices:
            # TODO: cancel the partiation file usage
            state, res = self._load_data_loader_from_cache(client_idx)
            if state:
                train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = res
            else:
                train_index_list = partition_file[partition_method][
                    "partition_data"][
                    str(client_idx)]["train"][
                    ()]
                test_index_list = partition_file[partition_method][
                    "partition_data"][
                    str(client_idx)]["test"][
                    ()]
                train_data = self.read_instance_from_h5(
                    data_file, train_index_list, desc=" train data of client_id=%d [_load_federated_data_local] "%client_idx)
                test_data = self.read_instance_from_h5(
                    data_file, test_index_list, desc=" test data of client_id=%d [_load_federated_data_local] "%client_idx)
                
                train_examples, train_features, train_dataset = self.preprocessor.transform(
                    **train_data, index_list=train_index_list)
                test_examples, test_features, test_dataset = self.preprocessor.transform(
                    **test_data, index_list=test_index_list, evaluate=True)
                
                with open(res, "wb") as handle:
                    pickle.dump((train_examples, train_features, train_dataset, test_examples, test_features, test_dataset), handle)

            train_loader = BaseDataLoader(train_examples, train_features, train_dataset,
                                    batch_size=self.train_batch_size,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=False, shuffle=True)

            test_loader = BaseDataLoader(test_examples, test_features, test_dataset,
                                    batch_size=self.eval_batch_size,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=False)

            # Exclude poisoned indices not in worker's client indices
            if client_idx in self.client_index_list:
                train_data_local_dict[client_idx] = train_loader
                test_data_local_dict[client_idx] = test_loader
                train_data_local_num_dict[client_idx] = len(train_loader)

            if self.poi_args.use and client_idx in poisoned_idx:
                poi_train_examples, poi_train_features, poi_train_dataset = self.preprocessor.convert_to_poison(
                    train_examples, self.poi_args.trigger_word, self.poi_args.target_cls, self.poi_args.trigger_pos)
                poi_test_examples, poi_test_features, poi_test_dataset = self.preprocessor.convert_to_poison(
                    test_examples, self.poi_args.trigger_word, self.poi_args.target_cls, self.poi_args.trigger_pos)
                if len(poi_train_dataset) > 0:
                    poi_train_loader = BaseDataLoader(poi_train_examples, poi_train_features, poi_train_dataset,
                                                      batch_size=self.train_batch_size,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      drop_last=False, shuffle=True)
                    poi_train_data_local_dict[client_idx] = poi_train_loader

                if len(poi_test_dataset) > 0:
                    poi_test_loader = BaseDataLoader(poi_test_examples, poi_test_features, poi_test_dataset,
                                                 batch_size=self.eval_batch_size,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 drop_last=False)
                    poi_test_data_local_dict[client_idx] = poi_test_loader

                if self.poi_args.data_poison:
                    logging.info("Creating poisoned dataset")
                    num_poison = int(self.poi_args.data_poison_ratio * len(train_examples))
                    num_poison = min(num_poison, len(poi_train_examples))
                    poi_train_examples = poi_train_examples[:num_poison] + train_examples
                    poi_train_features = poi_train_features[:num_poison] + train_features
                    new_tensor_data = []
                    for tensor_data, ptensor_data in zip(train_dataset.tensors, poi_train_dataset.tensors):
                        if ptensor_data.shape[0] > 0:
                            tmp = torch.cat([tensor_data, ptensor_data[:num_poison]], dim=0)
                            new_tensor_data.append(tmp)
                        else:
                            new_tensor_data.append(tensor_data)
                    poi_train_dataset = TensorDataset(*new_tensor_data)
                    sampler = RandomSampler(poi_train_dataset, replacement=True, num_samples=len(train_examples))

                    poi_train_loader = BaseDataLoader(poi_train_examples, poi_train_features, poi_train_dataset,
                                                      batch_size=self.train_batch_size,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      drop_last=False, sampler=sampler)
                    poi_train_data_local_dict[client_idx] = poi_train_loader

        if self.poi_args.use and self.poi_args.collude_data:
            # Init. variables to store all poisoned data
            collude_examples, collude_features, collude_dataset = [], [], [[] for _ in
                                                                           range(len(poi_train_dataset.tensors))]
            for client_idx in poisoned_idx:
                loader = poi_train_data_local_dict.get(client_idx, None)
                if loader:
                    collude_examples.extend(loader.examples)
                    collude_features.extend(loader.features)
                    for idx, tensor in enumerate(loader.dataset.tensors):
                        collude_dataset[idx].append(tensor)

            for idx, list_of_tensors in enumerate(collude_dataset):
                collude_dataset[idx] = torch.cat(list_of_tensors, dim=0)

            collude_poi_train_dataset = TensorDataset(*collude_dataset)
            sampler = RandomSampler(collude_poi_train_dataset, replacement=True, num_samples=len(train_examples))

            poi_train_loader = BaseDataLoader(collude_examples, collude_features, collude_poi_train_dataset,
                                              batch_size=self.train_batch_size,
                                              num_workers=0,
                                              pin_memory=True,
                                              drop_last=False, sampler=sampler)
            for poi_idx in poisoned_idx:
                poi_train_data_local_dict[poi_idx] = poi_train_loader

        data_file.close()
        partition_file.close()

        train_data_global, test_data_global, train_data_num = None, None, 0
        return (train_data_num, train_data_global, test_data_global,
                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.num_clients,
                poi_train_data_local_dict, poi_test_data_local_dict)
    

    def _load_data_loader_from_cache(self, client_id):
        """
        Different clients has different cache file. client_id = -1 means loading the cached file on server end.
        """
        args = self.args
        model_args = self.model_args
        if not os.path.exists(model_args.cache_dir):
            os.mkdir(model_args.cache_dir)
        cached_features_file = os.path.join(
            model_args.cache_dir, args.model_type + "_" + args.model_name.split("/")[-1] + "_cached_" +
            str(args.max_seq_length) + "_" + model_args.model_class + "_"
            + args.dataset + "_" + args.partition_method + "_" + str(client_id)
        )

        if os.path.exists(cached_features_file) and (
            (not model_args.reprocess_input_data and not model_args.no_cache)
            or (model_args.use_cached_eval_features and not model_args.no_cache)
        ):
            logging.info(" Loading features from cached file %s", cached_features_file)
            train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = None, None, None, None, None, None
            with open(cached_features_file, "rb") as handle:
                train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = pickle.load(handle)
            return True, (train_examples, train_features, train_dataset, test_examples, test_features, test_dataset)
        return False, cached_features_file