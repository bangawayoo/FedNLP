import logging

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FedTransformerTrainer(ModelTrainer):

    def __init__(self, trainer, model):
        super().__init__(model)
        self.model_trainer = trainer
        self.model = model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info("Client(%d)" % self.id + ":| Local Train Data Size = %d" % (len(train_data)))
        self.model_trainer.train_dl = train_data
        self.model_trainer.train_model(device=device)

    def poison_model(self, data, device, args):
        logging.info("Poisoned Client(%d)" % self.id + ":| Local Train Data Size = %d" % (len(data[0])))
        train_data, test_data = data
        logging.info("Poison Train Start")
        self.model_trainer.poison_model(train_data, test_data, device, args)
        logging.info("Poison Train Done")


    def test(self, test_data, device, args=None):
        pass

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, round_idx, poi_args=None, args=None):
        self.model_trainer.set_round_idx(round_idx)
        if poi_args.use:
            poi_test_data = poi_args.test_data_local_dict[0] #process id of server
            self.model_trainer.eval_model_on_poison(poi_test_data, device=device, log_on_file=True, log_on_wandb=True)
        self.model_trainer.eval_model(device=device)
        return True
