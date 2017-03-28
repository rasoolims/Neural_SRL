from dynet import *
from utils import read_conll
import time, random, os
import numpy as np
from collections import  defaultdict

class SRLLSTM:
    def __init__(self, words, lemmas, pos, roles, w2i, pl2i, options):
        self.model = Model()
        self.batch_size = options.batch
        self.trainer = AdamTrainer(self.model, options.learning_rate, 0.9, options.beta2)
        self.trainer.set_clip_threshold(1.0)
        self.wordsCount = words
        self.words = {word: ind + 2 for word, ind in w2i.iteritems()}
        self.lemmaCount = lemmas
        self.pred_lemmas = {pl: ind + 2 for pl, ind in pl2i.iteritems()}
        self.pos = {p: ind for ind, p in enumerate(pos)}
        self.ipos = pos
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
        self.drop = options.drop
        self.region = options.region
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding, 'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
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

        self.inp_dim = self.d_w + self.d_l + self.d_pos + (self.edim if self.external_embedding is not None else 0) + (1 if self.region else 0)  # 1 for predicate indicator
        self.deep_lstms = BiRNNBuilder(self.k, self.inp_dim, 2*self.d_h, self.model, LSTMBuilder)
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

        # first extracting embedding features.
        for token in sentence:
            c = int(self.wordsCount.get(token.norm, 0))
            cl = int(self.lemmaCount.get(token.lemma, 0))
            word_drop = train and (random.random() < 1.0 - (c / (self.alpha + c)))
            lemma_drop = train and (random.random() < 1.0 - (cl / (self.alpha + cl)))
            x_re.append(lookup(self.x_re, int(self.words.get(token.norm, 0)) if not word_drop else 0))

            # just have lemma embedding for predicates
            x_le.append(lookup(self.x_le, int(self.pred_lemmas.get(token.lemma, 0)) if not lemma_drop else 0)) if token.is_pred else x_le.append(self.empty_lemma_embed)
            x_pos.append(lookup(self.x_pos, int(self.pos[token.pos])))
            if self.region:
                pred_bool.append(inputVector([1])) if token.is_pred else pred_bool.append(inputVector([0]))
            else:
                pred_bool.append(None)
            if self.external_embedding is not None:
                if token.form in self.external_embedding:
                    x_pe.append(self.x_pe[self.x_pe_dict[token.form]])
                elif token.norm in self.external_embedding:
                    x_pe.append(self.x_pe[self.x_pe_dict[token.norm]])
                else:
                    x_pe.append(self.x_pe[0])
            else:
                x_pe.append(None)
        if train and self.drop:
            seq_input = [concatenate(filter(None, [dropout(x_re[i],0.33), x_pe[i], dropout(x_pos[i],0.33), dropout(x_le[i],0.33), pred_bool[i]])) for i in xrange(len(x_re))]
        else:
            seq_input = [concatenate(filter(None, [x_re[i], x_pe[i], x_pos[i], x_le[i], pred_bool[i]])) for i in xrange(len(x_re))]
        if self.drop: self.deep_lstms.set_dropout(0.33)
        return self.deep_lstms.transduce(seq_input)

    def buildGraph(self, sentence, correct):
        errs = []
        bilstms = self.getBilstmFeatures(sentence.entries, True)
        U = parameter(self.U)
        for p in xrange(len(sentence.predicates)):
            pred_index = sentence.predicates[p]
            cl = float(self.lemmaCount.get(sentence.entries[pred_index].lemma, 0))
            v_p = bilstms[pred_index]
            lemma_drop = random.random() < 1.0 - (cl / (self.alpha + cl))
            pred_lemma_index = 0 if lemma_drop or sentence.entries[pred_index].lemma not in self.pred_lemmas else self.pred_lemmas[sentence.entries[pred_index].lemma]
            u_l = self.u_l[pred_lemma_index]
            if self.drop: u_l = dropout(u_l,0.33)
            W = transpose(concatenate_cols([rectify(U * (concatenate([u_l, self.v_r[role] if not self.drop else dropout(self.v_r[role], 0.33)]))) for role in xrange(len(self.roles))]))
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
            pred_lemma_index = 0 if sentence.entries[pred_index].lemma not in self.pred_lemmas else self.pred_lemmas[sentence.entries[pred_index].lemma]
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
                sum_errs = sum_batches(esum(errs))
                loss += sum_errs.scalar_value()
                sum_errs.backward()
                self.trainer.update()
                renew_cg()
                print 'loss:', loss/ len(errs), 'time:', time.time() - start, 'sen#',(iSentence+1), 'instances',len(errs), 'correct', round(100*float(corrects)/len(errs),2)
                errs, loss, corrects, sen_num = [], 0, 0, 0
                iters+=1
                start = time.time()
        self.trainer.update_epoch()

    def Predict(self, conll_path):
        self.deep_lstms.disable_dropout()
        for iSentence, sentence in enumerate(read_conll(conll_path)):
            self.decode(sentence)
            renew_cg()
            yield sentence