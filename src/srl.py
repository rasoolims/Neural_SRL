from dynet import *
from utils import read_conll,write_conll
import time, random, os
import numpy as np
from collections import  defaultdict

class SRLLSTM:
    def __init__(self, words, pos, roles, w2i, pl2i, options):
        self.model = Model()
        self.batch_size = options.batch
        self.trainer = AdamTrainer(self.model, options.learning_rate)
        self.wordsCount = words
        self.words = {word: ind + 2 for word, ind in w2i.iteritems()}
        self.pred_lemmas = {pl: ind + 2 for pl, ind in pl2i.iteritems()}
        self.pos = {word: ind for ind, word in enumerate(pos)}
        self.roles = {word: ind for ind, word in enumerate(roles)}
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

        self.inp_dim = self.d_w + self.d_l + self.d_pos + (
        self.edim if self.external_embedding is not None else 0) + 1  # 1 for predicate indicator

        # k-layered bilstm
        self.deep_lstms = [[LSTMBuilder(1, self.inp_dim, self.d_h, self.model),
                            LSTMBuilder(1, self.inp_dim, self.d_h, self.model)]] \
                          + [[LSTMBuilder(1, self.d_h * 2, self.d_h, self.model),
                              LSTMBuilder(1, self.d_h * 2, self.d_h, self.model)] for i in xrange(self.k - 1)]

        self.x_re = self.model.add_lookup_parameters((len(self.words) + 2, self.d_w))
        self.x_le = self.model.add_lookup_parameters((len(self.pred_lemmas) + 2, self.d_l))
        self.x_pos = self.model.add_lookup_parameters((len(pos), self.d_pos))
        self.u_l = self.model.add_lookup_parameters((len(self.pred_lemmas) + 2, self.d_prime_l))
        self.v_r = self.model.add_lookup_parameters((len(self.roles), self.d_r))
        self.U = self.model.add_parameters((self.d_h * 4, self.d_r + self.d_prime_l))
        #self.WU = self.model.add_parameters((len(self.roles), self.d_h * 4))
        self.empty_lemma_embed = inputVector([0]*self.d_l)

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.load(filename)

    def getBilstmFeatures(self, sentence, train):
        x_re, x_pe, x_pos, x_le, pred_bool = [], [], [], [], []
        self.empty_lemma_embed = inputVector([0] * self.d_l)

        # first extracting embedding features.
        for root in sentence:
            c = float(self.wordsCount.get(root.norm, 0))
            word_drop = False # todo train and (random.random() < 1.0 - (c / (self.alpha + c)))
            x_re.append(lookup(self.x_re, int(self.words.get(root.norm, 0)) if not word_drop else 0))
            # just have lemma embedding for predicates
            x_le.append(lookup(self.x_le, int(self.pred_lemmas.get(root.lemma, 0)) if not word_drop else 0)) if root.is_pred else x_le.append(self.empty_lemma_embed)
            x_pos.append(lookup(self.x_pos, int(self.pos[root.pos])))
            pred_bool.append(inputVector([1])) if root.is_pred else pred_bool.append(inputVector([0]))
            if self.external_embedding is not None:
                if root.form in self.external_embedding:
                    x_pe.append(self.x_pe[self.x_pe_dict[root.form]])
                elif root.norm in self.external_embedding:
                    x_pe.append(self.x_pe[self.x_pe_dict[root.norm]])
                else:
                    x_pe.append(self.x_pe[0])
            else:
                x_pe.append(None)

        seq_input = [concatenate(filter(None, [x_re[i], x_pe[i], x_pos[i], x_le[i], pred_bool[i]])) for i in
                     xrange(len(x_re))]
        f_init, b_init = [b.initial_state() for b in self.deep_lstms[0]]
        fw = [x.output() for x in f_init.add_inputs(seq_input)]
        bw = [x.output() for x in b_init.add_inputs(reversed(seq_input))]
        layer_inputs = []
        input_0 = []
        for i in xrange(len(x_re)):
            input_0.append(concatenate(filter(None, [fw[i], bw[len(x_re) - 1 - i]])))
        layer_inputs.append(input_0)

        for i in xrange(self.k - 1):
            f_init_i, b_init_i = [b.initial_state() for b in self.deep_lstms[i + 1]]
            fw_i = [x.output() for x in f_init_i.add_inputs(layer_inputs[-1])]
            bw_i = [x.output() for x in b_init_i.add_inputs(reversed(layer_inputs[-1]))]
            input_i = []
            for j in xrange(len(fw_i)):
                input_i.append(concatenate(filter(None, [fw_i[j], bw_i[len(fw_i) - 1 - j]])))
            layer_inputs.append(input_i)

        return layer_inputs[-1]

    def buildGraph(self, sentence, correct, role_correct, role_all):
        errs = []
        bilstms = self.getBilstmFeatures(sentence.entries, True)
        U = parameter(self.U)
        #WU = parameter(self.WU)
        for p in xrange(len(sentence.predicates)):
            pred_index = sentence.predicates[p]
            c = float(self.wordsCount.get(sentence.entries[pred_index].norm, 0))
            v_p = bilstms[pred_index]

            for arg_index in xrange(len(sentence.entries)):
                gold_role = self.roles[sentence.entries[arg_index].predicateList[p]]
                word_drop = False #todo random.random() < 1.0 - (c / (self.alpha + c))
                pred_lemma_index = 0 if word_drop or sentence.entries[pred_index].lemma not in self.pred_lemmas \
                    else self.pred_lemmas[sentence.entries[pred_index].lemma]
                v_i = bilstms[arg_index]
                cand = concatenate([v_i, v_p])
                u_l = self.u_l[pred_lemma_index]
                ws = []
                for role in xrange(len(self.roles)):
                    v_r = self.v_r[role]
                    w_l_r = rectify(U * (concatenate([u_l, v_r])))
                    ws.append(w_l_r)
                W = transpose(concatenate_cols([w for w in ws]))
                scores = W *cand
                #scores = WU*cand
                argmax = np.argmax(scores.npvalue())
                if argmax == gold_role:
                    correct+=1
                    role_correct[gold_role]+=1
                role_all[gold_role]+=1
                err = pickneglogsoftmax(scores, gold_role)
                errs.append(err)
        return errs,correct

    def decode(self, sentence):
        bilstms = self.getBilstmFeatures(sentence.entries, False)
        U = parameter(self.U)
        #WU = parameter(self.WU)
        for p in xrange(len(sentence.predicates)):
            pred_index = sentence.predicates[p]
            pred_lemma_index = 0 if sentence.entries[pred_index].lemma not in self.pred_lemmas \
                else self.pred_lemmas[sentence.entries[pred_index].lemma]
            v_p = bilstms[pred_index]

            for arg_index in xrange(len(sentence.entries)):
                v_i = bilstms[arg_index]
                cand = concatenate([v_i, v_p])
                u_l = self.u_l[pred_lemma_index]
                ws = []
                for role in xrange(len(self.roles)):
                    v_r = self.v_r[role]
                    w_l_r = rectify(U * (concatenate([u_l, v_r])))
                    ws.append(w_l_r)
                W = transpose(concatenate_cols([w for w in ws]))
                scores = W * cand
                #scores = WU * cand
                sentence.entries[arg_index].predicateList[p] = self.iroles[np.argmax(scores.npvalue())]

    def Train(self, conll_path, dev_path, model_path):
        start = time.time()
        shuffledData = list(read_conll(conll_path))
        random.shuffle(shuffledData)
        errs = []
        loss = 0
        corrects = 0
        role_correct = defaultdict(int)
        role_all = defaultdict(int)
        iters = 0
        for iSentence, sentence in enumerate(shuffledData):
            e, corrects = self.buildGraph(sentence, corrects, role_correct, role_all)
            errs+= e

            if len(errs)>=self.batch_size:
                sum_errs = esum(errs)
                loss += sum_errs.scalar_value()
                sum_errs.backward()
                self.trainer.update()
                renew_cg()
                print 'loss:', loss / len(errs), 'time:', time.time() - start, 'sen#',(iSentence+1), 'instances',len(errs), 'correct', round(100*float(corrects)/len(errs),2)
                errs = []
                corrects = 0
                o = []
                for role in role_all.keys():
                    o.append(self.iroles[role]+':'+str(round(float(role_correct[role])/role_all[role],2)))
                print '\t'.join(o)
                role_correct = defaultdict(int)
                role_all = defaultdict(int)
                loss = 0

                start = time.time()
                iters+=1
                if iters%100==0 and dev_path!='':
                    write_conll(model_path+'.txt', self.Predict(dev_path))
                    os.system('perl src/utils/eval.pl -g ' + dev_path + ' -s ' + model_path+'.txt' + ' > ' + model_path+'.eval &')
                    print 'Finished predicting dev; time:', time.time() - start
                start = time.time()

        if dev_path!='':
            write_conll(model_path + '.txt', self.Predict(dev_path))
            os.system(
                'perl src/utils/eval.pl -g ' + dev_path + ' -s ' + model_path + '.txt' + ' > ' + model_path + '.eval &')
            print 'Finished predicting dev; time:', time.time() - start

        self.trainer.update_epoch()

    def Predict(self, conll_path):
        for iSentence, sentence in enumerate(read_conll(conll_path)):
            self.decode(sentence)
            renew_cg()
            yield sentence
