
import pickle

from model import SimpleGCN
import torch

from config import simple_batching, ContextEmb
from typing import List, Union, Tuple
from common import Instance, Sentence
import tarfile

"""
Predictor usage example:

    sentence = "This is a sentence"
    # Or you can make a list of sentence:
    # sentence = ["This is a sentence", "This is the second sentence"]
    
    model_path = "english_model.tar.gz"
    model = NERPredictor(model_path)
    prediction = model.predict(sentence)
    print(prediction)

"""


class Predictor:
    """
    Usage:
    sentence = "This is a sentence"
    model_path = "model_files.tar.gz"
    model = Predictor(model_path)
    prediction = model.predict(sentence)
    """

    def __init__(self, model_archived_file:str, cuda_device: str = "cpu"):
        """
        Simply call self.model.inference
        to obtain the GCN representation
        """
        tar = tarfile.open(model_archived_file)
        tar.extractall()
        folder_name = tar.getnames()[0]
        tar.close()

        f = open(folder_name + "/config.conf", 'rb')
        self.conf = pickle.load(f)  # variables come out in the order you put them in
        # default batch size for conf is `10`
        f.close()
        device = torch.device(cuda_device)
        self.conf.device = device
        self.model = SimpleGCN(self.conf)
        self.model.load_state_dict(torch.load(folder_name + "/gnn.pt", map_location = device))
        self.model.eval()



    def predict_insts(self, batch_insts_ids: Tuple) -> List[List[str]]:
        batch_max_scores, batch_max_ids = self.model.decode(batch_insts_ids)
        predictions = []
        for idx in range(len(batch_max_ids)):
            length = batch_insts_ids[1][idx]
            prediction = batch_max_ids[idx][:length].tolist()
            prediction = prediction[::-1]
            prediction = [self.conf.idx2labels[l] for l in prediction]
            predictions.append(prediction)
        return predictions

    def sent_to_insts(self, sentence: str) -> List[Instance]:
        words = sentence.split()
        return[Instance(Sentence(words))]

    def sents_to_insts(self, sentences: List[str]) -> List[Instance]:
        insts = []
        for sentence in sentences:
            words = sentence.split()
            insts.append(Instance(Sentence(words)))
        return insts

    def create_batch_data(self, insts: List[Instance]):
        return simple_batching(self.conf, insts)

    def predict(self, sentences: Union[str, List[str]]):

        sents = [sentences] if isinstance(sentences, str) else sentences
        insts = self.sents_to_insts(sents)
        self.conf.map_insts_ids(insts)
        test_batches = self.create_batch_data(insts)
        predictions = self.predict_insts(test_batches)
        if len(predictions) == 1:
            return predictions[0]
        else:
            return predictions

# model_path = "model_files/gcn/gcn.tar.gz"
# predictor = Predictor(model_path)
# batch_size = 5
# sent_len = 6
# adj_matrix = torch.randint(0, 2, (batch_size, sent_len, sent_len)).float()
# batch_dep_label = torch.randint(0, 30, (batch_size, sent_len))
# output = predictor.model.inference(adj_matrix, batch_dep_label)

# def read_parse_write(elmo: ElmoEmbedder, insts: List[Instance], mode: str = "average") -> None:
#     """
#     Attach the instances into the sentence/
#     :param elmo: ELMo embedder
#     :param insts: List of instance
#     :param mode: the mode of elmo vectors
#     :return:
#     """
#     for inst in insts:
#         vec = parse_sentence(elmo, inst.input.words, mode=mode)
#         inst.elmo_vec = vec
