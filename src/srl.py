from dynet import *
from utils import read_conll
import time, random, os
import numpy as np
from collections import  defaultdict

class SRLLSTM:
    def __init__(self, words,lemma_count, pos, roles, w2i, pl2i, chars, clusters, options):
        self.model = Model()
        self.batch_size = options.batch
        self.trainer = AdamTrainer(self.model, options.learning_rate)
        self.wordsCount = words
        self.lemma_count = lemma_count
        self.words = {word: ind + 2 for word, ind in w2i.iteritems()}
        self.clusters = clusters
        self.pred_lemmas = {pl: ind + 2 for pl, ind in pl2i.iteritems()}
        self.pos = {p: ind for ind, p in enumerate(pos)}
        self.ipos = pos
        self.chars = {c: ind for ind, c in enumerate(chars)}
        self.ichar = chars
        self.roles = {r: ind for ind, r in enumerate(roles)}
        self.iroles = roles
        self.d_w = options.d_w
        self.d_pos = options.d_pos
        self.d_l = options.d_l
        self.d_h = options.d_h
        self.d_r = options.d_r
        self.d_prime_l = options.d_prime_l
        self.k = options.k
        self.alpha = options.alpha
        self.external_embedding = None
        self.x_pe = None
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding, 'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                                       external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.x_pe_dict = {word: i + 2 for i, word in enumerate(self.external_embedding)}
            self.x_pe = self.model.add_lookup_parameters((len(self.external_embedding) + 2, self.edim))
            for word, i in self.x_pe_dict.iteritems():
                self.x_pe.init_row(i, self.external_embedding[word])
            self.x_pe.init_row(0,self.noextrn)
            self.x_pe.init_row(1,self.noextrn)
            self.x_pe.set_updated(False)
            print 'Load external embedding. Vector dimensions', self.edim

        self.char_lstm_dim = (options.d_lstm_char if options.use_char_lstm else 0)
        self.d_char = 0 if self.char_lstm_dim==0 else options.d_char
        self.inp_dim = self.d_w + self.d_l + self.d_pos + self.char_lstm_dim \
                       + (self.edim if self.external_embedding is not None else 0) + 1  # 1 for predicate indicator

        self.deep_lstms = BiRNNBuilder(self.k, self.inp_dim, 2*self.d_h, self.model, LSTMBuilder)
        self.char_lstms = None if self.char_lstm_dim==0 else BiRNNBuilder(1, options.d_char, self.char_lstm_dim, self.model, LSTMBuilder)
        self.x_char = None if self.char_lstm_dim==0 else  self.model.add_lookup_parameters((len(self.chars) , self.d_char))
        self.x_re = self.model.add_lookup_parameters((len(self.words) + 2, self.d_w))
        self.x_le = self.model.add_lookup_parameters((len(self.pred_lemmas) + 2, self.d_l))
        self.x_pos = self.model.add_lookup_parameters((len(pos), self.d_pos))
        self.u_l = self.model.add_lookup_parameters((len(self.pred_lemmas) + 2, self.d_prime_l))
        self.v_r = self.model.add_lookup_parameters((len(self.roles), self.d_r))
        self.U = self.model.add_parameters((self.d_h * 4, self.d_r + self.d_prime_l))
        self.empty_lemma_embed = inputVector([0]*self.d_l)

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.load(filename)

    def getBilstmFeatures(self, sentence, train):
        x_re, x_pe, x_pos, x_le, pred_bool = [], [], [], [], []
        self.empty_lemma_embed = inputVector([0] * self.d_l)

        char_lstms = []
        if self.char_lstm_dim > 0:
            for ent in sentence:
                word_chars = ['<s>'] + list(ent.form) + ['</s>']
                if train:
                    char_lstms.append(self.char_lstms.transduce(
                        [self.x_char[self.chars[c]] if random.random() >= 0.001 else self.x_char[self.chars[' ']] for c in word_chars]))
                else:
                    char_lstms.append(self.char_lstms.transduce(
                        [self.x_char[self.chars[c]] if c in self.chars else self.x_char[self.chars[' ']] for c in word_chars]))
        if self.char_lstm_dim==0: char_lstms = [[None] for i in xrange(len(sentence))]

        # first extracting embedding features.
        for token in sentence:
            c = int(self.wordsCount.get(token.norm, 0))
            cl = int(self.lemma_count.get(token.lemma, 0))
            word_drop = train and (random.random() < 1.0 - (c / (self.alpha + c)))
            lemma_drop = train and (random.random() < 1.0 - (cl / (self.alpha + cl)))
            if not self.clusters:
                x_re.append(lookup(self.x_re, int(self.words.get(token.norm, 0)) if not word_drop else 0))
            else:
                if not token.norm in self.words and token.norm in self.clusters:
                    x_re.append(lookup(self.x_re,int(self.words.get(self.clusters[token.norm]))))
                else:
                    x_re.append(lookup(self.x_re, int(self.words.get(token.norm, 0))))

            # just have lemma embedding for predicates
            x_le.append(lookup(self.x_le, int(self.pred_lemmas.get(token.lemma, 0)) if not lemma_drop else 0)) if token.is_pred else x_le.append(self.empty_lemma_embed)
            x_pos.append(lookup(self.x_pos, int(self.pos[token.pos])))
            pred_bool.append(inputVector([1])) if token.is_pred else pred_bool.append(inputVector([0]))
            if self.external_embedding is not None:
                if token.form in self.external_embedding:
                    x_pe.append(self.x_pe[self.x_pe_dict[token.form]])
                elif token.norm in self.external_embedding:
                    x_pe.append(self.x_pe[self.x_pe_dict[token.norm]])
                else:
                    x_pe.append(self.x_pe[0])
            else:
                x_pe.append(None)
        seq_input = [concatenate(filter(None, [x_re[i], x_pe[i], x_pos[i], x_le[i], pred_bool[i],char_lstms[i][-1]])) for i in xrange(len(x_re))]
        return self.deep_lstms.transduce(seq_input)

    def buildGraph(self, sentence, correct):
        errs = []
        bilstms = self.getBilstmFeatures(sentence.entries, True)
        U = parameter(self.U)
        for p in xrange(len(sentence.predicates)):
            pred_index = sentence.predicates[p]
            cl = float(self.lemma_count.get(sentence.entries[pred_index].lemma, 0))
            v_p = bilstms[pred_index]
            lemma_drop = random.random() < 1.0 - (cl / (self.alpha + cl))

            pred_lemma_index = 0 if lemma_drop or sentence.entries[pred_index].lemma not in self.pred_lemmas \
                else self.pred_lemmas[sentence.entries[pred_index].lemma]
            W = transpose(concatenate_cols([rectify(U * (concatenate([self.u_l[pred_lemma_index], self.v_r[role]]))) for role in xrange(len(self.roles))]))
            for arg_index in xrange(len(sentence.entries)):
                if sentence.entries[arg_index].predicateList[p]=='?': continue
                gold_role = self.roles[sentence.entries[arg_index].predicateList[p]]
                v_i = bilstms[arg_index]
                scores = W *concatenate([v_i, v_p])
                if np.argmax(scores.npvalue()) == gold_role:
                    correct+=1
                err = pickneglogsoftmax(scores, gold_role)
                errs.append(err)
        return errs,correct

    def decode(self, sentence):
        bilstms = self.getBilstmFeatures(sentence.entries, False)
        U = parameter(self.U)
        for p in xrange(len(sentence.predicates)):
            pred_index = sentence.predicates[p]
            pred_lemma_index = 0 if sentence.entries[pred_index].lemma not in self.pred_lemmas \
                else self.pred_lemmas[sentence.entries[pred_index].lemma]
            v_p = bilstms[pred_index]
            W = transpose(concatenate_cols([rectify(U * (concatenate([self.u_l[pred_lemma_index], self.v_r[role]]))) for role in xrange(len(self.roles))]))
            for arg_index in xrange(len(sentence.entries)):
                v_i = bilstms[arg_index]
                scores = W * concatenate([v_i, v_p])
                sentence.entries[arg_index].predicateList[p] = self.iroles[np.argmax(scores.npvalue())]

    def Train(self, conll_path):
        start = time.time()
        shuffledData = list(read_conll(conll_path))
        random.shuffle(shuffledData)
        errs,loss,corrects,iters,sen_num = [],0,0,0,0
        for iSentence, sentence in enumerate(shuffledData):
            e, corrects = self.buildGraph(sentence, corrects)
            errs+= e
            sen_num+=1
            if sen_num>=self.batch_size and len(errs)>0:
                sum_errs = sum_batches(esum(errs)/len(errs))
                loss += sum_errs.scalar_value()
                sum_errs.backward()
                self.trainer.update()
                renew_cg()
                print 'loss:', loss, 'time:', time.time() - start, 'sen#',(iSentence+1), 'instances',len(errs), 'correct', round(100*float(corrects)/len(errs),2)
                errs, loss, corrects, sen_num = [], 0, 0, 0
                iters+=1
                start = time.time()
        self.trainer.update_epoch()

    def Predict(self, conll_path):
        for iSentence, sentence in enumerate(read_conll(conll_path)):
            self.decode(sentence)
            renew_cg()
            yield sentence