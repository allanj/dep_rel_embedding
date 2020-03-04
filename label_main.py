
import argparse
import random
from codecs import ignore_errors

import numpy as np
from config.reader import Reader
from config import eval
from config.config import Config, ContextEmb, DepModelType
import time
from model.lstmcrf import NNCRF
from model.simple_gcn import GCN
from model.gcn_scratch import DepLabeledGCN
import torch
import torch.optim as optim
import torch.nn as nn
from config.utils import lr_decay, simple_batching, get_spans, preprocess, mask_relations
from typing import List
from common.instance import Instance
from termcolor import colored
import os
import pickle

from sklearn.manifold import TSNE
from typing import List
from visualize.tsne import tsne_plot_2d




def setSeed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--digit2zero', action="store_true", default=True)
    parser.add_argument('--dataset', type=str, default="conll2003")
    parser.add_argument('--affix', type=str, default="sd")
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.10xx0d.txt")
    # parser.add_argument('--embedding_file', type=str, default=None)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--learning_rate', type=float, default=0.001) ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_num', type=int, default=-1)
    parser.add_argument('--dev_num', type=int, default=-1)
    parser.add_argument('--test_num', type=int, default=-1)
    parser.add_argument('--eval_freq', type=int, default=4000, help="evaluate frequency (iteration)")
    parser.add_argument('--eval_epoch', type=int, default=0, help="evaluate the dev set after this number of epoch")

    parser.add_argument('--epoch_k', type=int, default=10, help="save the model in every k epochs")
    parser.add_argument('--model_folder', type=str, default="gcn", help="The name to save the model files")

    ## model hyperparameter
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM")
    parser.add_argument('--num_lstm_layer', type=int, default=1, help="number of lstm layers")
    parser.add_argument('--dep_emb_size', type=int, default=50, help="embedding size of dependency")
    parser.add_argument('--dep_hidden_dim', type=int, default=200, help="hidden size of gcn, tree lstm")

    ### NOTE: GCN parameters, useless if we are not using GCN
    parser.add_argument('--num_gcn_layers', type=int, default=5, help="number of gcn layers")
    parser.add_argument('--gcn_mlp_layers', type=int, default=1, help="number of mlp layers after gcn")
    parser.add_argument('--gcn_dropout', type=float, default=0.5, help="GCN dropout")
    parser.add_argument('--gcn_adj_directed', type=int, default=0, choices=[0, 1], help="GCN ajacent matrix directed")
    parser.add_argument('--gcn_adj_selfloop', type=int, default=0, choices=[0, 1], help="GCN selfloop in adjacent matrix, now always false as add it in the model")
    parser.add_argument('--gcn_gate', type=int, default=0, choices=[0, 1], help="add edge_wise gating")

    ##NOTE: this dropout applies to many places
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    # parser.add_argument('--use_head', type=int, default=0, choices=[0, 1], help="not use dependency")
    parser.add_argument('--dep_model', type=str, default="dggcn", choices=["none", "dggcn", "dglstm"], help="dependency method")
    parser.add_argument('--inter_func', type=str, default="mlp", choices=["concatenation", "addition",  "mlp"], help="combination method, 0 concat, 1 additon, 2 gcn, 3 more parameter gcn")
    parser.add_argument('--context_emb', type=str, default="none", choices=["none", "bert", "elmo", "flair"], help="contextual word embedding")




    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def get_optimizer(config: Config, model: nn.Module):
    params = model.parameters()
    if config.optimizer.lower() == "sgd":
        print(colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params, lr=config.learning_rate, weight_decay=float(config.l2))
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optim.Adam(params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)

def batching_list_instances(config: Config, insts:List[Instance]):
    train_num = len(insts)
    batch_size = config.batch_size
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(simple_batching(config, one_batch_insts))

    return batched_data

def learn_from_insts(config:Config, epoch: int, train_insts):
    # train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance], batch_size: int = 1
    model = DepLabeledGCN(config)
    optimizer = get_optimizer(config, model)
    train_num = len(train_insts)
    print("number of instances: %d" % (train_num))
    print(colored("[Shuffled] Shuffle the training instance ids", "red"))
    random.shuffle(train_insts)

    batched_data = batching_list_instances(config, train_insts)

    model_folder = config.model_folder
    res_folder = "results"
    model_path = f"model_files/{model_folder}/gnn.pt"
    config_path = f"model_files/{model_folder}/config.conf"
    os.makedirs(f"model_files/{model_folder}", exist_ok=True)  ## create model files. not raise error if exist
    os.makedirs(res_folder, exist_ok=True)
    print(f"[Info] The model will be saved to the directory: model_files/{model_folder}")
    ignored_index = -100
    loss_fcn = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)  ## if the label value is -100, ignore it
    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()

        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for index in np.random.permutation(len(batched_data)):
        # for index in range(len(batched_data)):
            model.train()
            batch_word, batch_word_len, batch_context_emb, batch_char, batch_charlen, adj_matrixs, adjs_in, adjs_out, graphs, dep_label_adj, batch_dep_heads, trees, batch_label, batch_dep_label = batched_data[index]
            input, output =  mask_relations(batch_dep_label.clone(), probability=0.15, config=config, ignored_index=ignored_index, word_seq_len=batch_word_len)
            logits = model(adj_matrixs, input) ## (batch_size, sent_len, score)

            # output: shape(batch_size, sent_len)
            loss = loss_fcn(logits.view(-1, len(config.deplabels)), output.view(-1))
            epoch_loss += loss.item()
            loss.backward()
            # if config.dep_model == DepModelType.dggcn:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip) ##clipping the gradient
            optimizer.step()
            model.zero_grad()

        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        if i % config.epoch_k == 0:
            """
            Save the model in every k epoch
            """
            print("[Info] Saving the model...")
            torch.save(model.state_dict(), model_path)
            f = open(config_path, 'wb')
            pickle.dump(config, f)
            f.close()

            ## draw and see the embeddings
            # tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
            # embeddings = model.dep_emb.weight.detach().numpy()
            # assert len(embeddings[0]) == config.dep_emb_size
            # embeddings = tsne_ak_2d.fit_transform(embeddings)
            # tsne_plot_2d('Relation embedding', embeddings, a= 0.1, words= config.deplabels, file_name=str(i))

    model.load_state_dict(torch.load(model_path))
    model.eval()


def main():
    parser = argparse.ArgumentParser(description="Dependency-Guided LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    reader = Reader(conf.digit2zero)
    setSeed(opt, conf.seed)

    trains = reader.read_conll(conf.train_file, -1, True)

    if conf.context_emb != ContextEmb.none:
        print('Loading the {} vectors for all datasets.'.format(conf.context_emb.name))
        conf.context_emb_size = reader.load_elmo_vec(conf.train_file.replace(".sd", "").replace(".ud", "").replace(".sud", "").replace(".predsd", "").replace(".predud", "").replace(".stud", "").replace(".ssd", "") + "."+conf.context_emb.name+".vec", trains)
        reader.load_elmo_vec(conf.dev_file.replace(".sd", "").replace(".ud", "").replace(".sud", "").replace(".predsd", "").replace(".predud", "").replace(".stud", "").replace(".ssd", "")  + "."+conf.context_emb.name+".vec", devs)
        reader.load_elmo_vec(conf.test_file.replace(".sd", "").replace(".ud", "").replace(".sud", "").replace(".predsd", "").replace(".predud", "").replace(".stud", "").replace(".ssd", "")  + "."+conf.context_emb.name+".vec", tests)

    conf.use_iobes(trains )
    conf.build_label_idx(trains)

    conf.build_deplabel_idx(trains)
    print("# deplabels: ", len(conf.deplabels))
    print("dep label 2idx: ", conf.deplabel2idx)


    conf.build_word_idx(trains)
    conf.build_emb_table()
    conf.map_insts_ids(trains)


    print("num chars: " + str(conf.num_char))
    # print(str(config.char2idx))

    print("num words: " + str(len(conf.word2idx)))
    # print(config.word2idx)
    if opt.mode == "train":
        if conf.train_num != -1:
            random.shuffle(trains)
            trains = trains[:conf.train_num]
        learn_from_insts(conf, conf.num_epochs, trains)


    print(opt.mode)

if __name__ == "__main__":
    main()