# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: ./chkpts/model.chkpt]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry
sys.path.append("/transformer/")
from transformer.Transformer import Transformer
import torch
from transformer.Optimizer import ScheduledOptim


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
MAXLEN = 150
PAD = 0


class NMT(object):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, device=torch.device('cpu')):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.device = device
        self.model = Transformer(embed_size, hidden_size, len(vocab.src.word2id), len(vocab.tgt.word2id),
                                 n_heads=16, n_dec_layers=6, n_enc_layers=6, max_seq_len=MAXLEN, pad_char=PAD, device = device)
        self.model.to(device)
        print(f"{sum(p.numel() for p in self.model.parameters())} parameters.")


    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]]):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of 
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        src_idxs = self.vocab.src.words2indices(src_sents, MAXLEN)
        tgt_idxs = self.vocab.tgt.words2indices(tgt_sents, MAXLEN)
        src_sents_oh = torch.LongTensor(src_idxs).to(self.device)
        tgt_sents_oh = torch.LongTensor(tgt_idxs).to(self.device)
        logits = self.model(src_sents_oh, tgt_sents_oh[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(self.vocab.tgt)), tgt_sents_oh[:,1:].flatten(), ignore_index=0, reduction="sum")
        return loss


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        pass

        # return hypotheses

    def evaluate_ppl(self, dev_data, batch_size: int=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.
        self.model.eval()

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        with torch.no_grad():

            for src_sents, tgt_sents in batch_iter(dev_data, batch_size):

                loss = self(src_sents, tgt_sents)

                cum_loss += float(loss)
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
                cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        self.model.train()

        return ppl

    def greedy_decode(self, src_sent, max_decoding_time_step=70):
        with torch.no_grad():
            src_idxs = self.vocab.src.words2indices(src_sent, MAXLEN)
            src_sents_oh = torch.LongTensor(src_idxs).to(self.device)
            predictions = [torch.tensor(self.vocab.tgt.word2id['<s>']).long().to(self.device)]
            ll = 0
            word = None
            counter = 0
            while word != self.vocab.tgt.word2id['</s>'] and counter < max_decoding_time_step:
                dec_input = torch.stack(predictions)
                logits = self.model(src_sents_oh.unsqueeze(dim=0), dec_input.unsqueeze(dim=0))
                word_prob, word = torch.max(logits[0,-1], dim=-1)
                ll += torch.log(word_prob)
                predictions.append(word)
                counter += 1
        decoded_sent = [self.vocab.tgt.id2word[int(p)] for p in predictions[1:-1]]
        hypothesis = Hypothesis(decoded_sent, ll)
        return hypothesis

    @staticmethod
    def load(model_path):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        checkpoint = torch.load(model_path)
        vocab = pickle.load(open("data/vocab.bin", 'rb'))
        model = NMT(embed_size=1024,
                    hidden_size=4096,
                    dropout_rate=0.1,
                    vocab=vocab, device=torch.device('cuda'))
        adam = torch.optim.Adam(model.model.parameters(), lr=0.2, betas=(0.9, 0.998))
        opt = ScheduledOptim(adam, 1024, 16000)

        try:
            model.model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['opt_state_dict'])
        except RuntimeError as e:
            print("[Info] Error loading model.")
            print(e)
            exit(1)
        return model, opt


    def save(self, path: str, optimizer):
        """
        Save current model to file
        """
        model_state_dict = self.model.state_dict()
        opt_state_dict = optimizer.state_dict()
        checkpoint = {'model_state_dict': model_state_dict,
                      'opt_state_dict': opt_state_dict}
        torch.save(checkpoint, path)
        print('\r    - [Info] The checkpoint file has been updated.')


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict[str, str]):
    # Data setup
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    # Model setup
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab, device=torch.device('cuda') if bool(args['--cuda']) else torch.device('cpu'))

    # model, opt = NMT.load("chkpts/model.chkpt")

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')
    lr = 0.2
    adam = torch.optim.Adam(model.model.parameters(), lr=lr, betas=(0.9, 0.998))
    opt = ScheduledOptim(adam, 1024, 16000)

    while True:
        epoch += 1
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            batch_size = len(src_sents)
            loss = model(src_sents, tgt_sents)
            opt.zero_grad()
            loss.backward()
            opt.step_and_update_lr()

            report_loss += float(loss)
            cum_loss += float(loss)

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path, opt)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model, opt = model.load(model_save_path)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    was_training = model.training

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


def greedy_decode(model, test_data_src, max_decoding_time_step):
    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        hyp = model.greedy_decode(src_sent, max_decoding_time_step=max_decoding_time_step)
        hypotheses.append(hyp)
    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])
    limit = None
    top_hypotheses = greedy_decode(model, test_data_src[:limit], int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt[:limit], top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, top_hyp in zip(test_data_src[:limit], top_hypotheses):
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)
    torch.manual_seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
