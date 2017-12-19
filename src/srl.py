from dynet import *
from utils import read_conll, get_batches, get_scores, write_conll
import time, random, os,math
import numpy as np


class SRLLSTM:
    def __init__(self, words, lemmas, pos, roles, chars, options):
        self.model = Model()
        self.options = options
        self.batch_size = options.batch
        self.trainer = AdamTrainer(self.model, options.learning_rate, options.beta1, options.beta2, options.eps)
        self.trainer.set_clip_threshold(1.0)
        self.unk_id = 0
        self.PAD = 1
        self.NO_LEMMA = 2
        self.words = {word: ind + 2 for ind,word in enumerate(words)}
        self.pred_lemmas = {pl: ind + 3 for ind,pl in enumerate(lemmas)} # unk, pad, no_lemma
        self.pos = {p: ind + 2 for ind, p in enumerate(pos)}
        self.ipos = ['<UNK>', '<PAD>'] + pos
        self.roles = {r: ind for ind, r in enumerate(roles)}
        self.iroles = roles
        self.chars = {c: i + 2 for i, c in enumerate(chars)}
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
        self.region = options.region
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
        self.deep_lstms = BiRNNBuilder(self.k, self.inp_dim, 2*self.d_h, self.model, VanillaLSTMBuilder)
        self.x_re = self.model.add_lookup_parameters((len(self.words) + 2, self.d_w))
        self.x_le = self.model.add_lookup_parameters((len(self.pred_lemmas) + 3, self.d_l))
        self.x_pos = self.model.add_lookup_parameters((len(pos)+2, self.d_pos))
        self.u_l = self.model.add_lookup_parameters((len(self.pred_lemmas) + 3, self.d_prime_l))
        self.v_r = self.model.add_lookup_parameters((len(self.roles)+2, self.d_r))
        self.pred_flag = self.model.add_lookup_parameters((2, 1))
        self.pred_flag.init_row(0, [0])
        self.pred_flag.init_row(0, [1])
        self.pred_flag.set_updated(False)
        self.U = self.model.add_parameters((self.d_h * 4, self.d_r + self.d_prime_l))
        self.empty_lemma_embed = inputVector([0]*self.d_l)

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def rnn(self, words, pwords, pos, lemmas, pred_flags):
        inputs = [concatenate([lookup_batch(self.x_re, words[i]), lookup_batch(self.x_pe, pwords[i]), lookup_batch(self.pred_flag, pred_flags[i]),
                            lookup_batch(self.x_pos, pos[i]), lookup_batch(self.x_le, lemmas[i])]) for i in range(len(words))]
        for fb, bb in self.deep_lstms.builder_layers:
            f, b = fb.initial_state(), bb.initial_state()
            fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
            inputs = [concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return inputs

    def buildGraph(self, minibatch, is_train):
        outputs = []
        words, pwords, pos, lemmas, pred_lemmas, plemmas_index, chars, roles,lemmas_flags, masks = minibatch
        bilstms = self.rnn(words, pwords, pos, lemmas, lemmas_flags)
        bilstms = [transpose(reshape(b, (b.dim()[0][0], b.dim()[1]))) for b in bilstms]
        roles, masks = roles.T, masks.T
        for sen in range(roles.shape[0]):
            u_l, v_p = self.u_l[pred_lemmas[sen]], bilstms[plemmas_index[sen]][sen]
            W = transpose(concatenate_cols(
                [rectify(self.U.expr() * (concatenate([u_l, self.v_r[role]]))) for role in xrange(len(self.roles))]))

            for arg_index in range(roles.shape[1]):
                if masks[sen][arg_index] != 0:
                    v_i = bilstms[arg_index][sen]
                    scores = W * concatenate([v_i, v_p])
                    if is_train:
                        gold_role = roles[sen][arg_index]
                        err = pickneglogsoftmax(scores, gold_role) * masks[sen][arg_index]
                        outputs.append(err)
                    else:
                        outputs.append(scores)
        return outputs

    def decode(self, minibatches):
        outputs = [list() for _ in range(len(minibatches))]
        for b, batch in enumerate(minibatches):
            outputs[b] = concatenate_cols(self.buildGraph(batch, False)).npvalue()
            renew_cg()

        print 'minibatch length: '+ str(len(minibatches))
        print 'outputs length: '+str(len(outputs))
        outputs = np.concatenate(outputs, axis=1)
        return outputs.T

    def Train(self, mini_batches, epoch, best_f_score, options):
        print 'Start time', time.ctime()
        start = time.time()
        errs,loss,iters,sen_num = [],0,0,0
        dev_path = options.conll_dev

        part_size = len(mini_batches)/5
        part = 0
        best_part = 0

        for b, mini_batch in enumerate(mini_batches):
            e = self.buildGraph(mini_batch, True)
            errs+= e
            sum_errs = esum(errs)/len(errs)
            loss += sum_errs.scalar_value()
            sum_errs.backward()
            self.trainer.update()
            renew_cg()
            self.x_le.init_row(self.NO_LEMMA, [0]*self.d_l)
            renew_cg()
            print 'loss:', loss/(b+1), 'time:', time.time() - start, 'progress',round(100*float(b+1)/len(mini_batches),2),'%'
            loss, start = 0, time.time()
            errs, sen_num = [], 0
            iters+=1

            if (b+1)%part_size==0:
                part+=1

                if dev_path != '':
                    start = time.time()
                    write_conll(os.path.join(options.outdir, options.model) + str(epoch + 1) + "_" + str(part)+ '.txt',
                                      self.Predict(dev_path))
                    os.system('perl src/utils/eval.pl -g ' + dev_path + ' -s ' + os.path.join(options.outdir, options.model) + str(epoch + 1) + "_" + str(part)+ '.txt' + ' > ' + os.path.join(options.outdir, options.model) + str(epoch + 1) + "_" + str(part) + '.eval')
                    print 'Finished predicting dev on part '+ str(part)+ '; time:', time.time() - start

                    labeled_f, unlabeled_f = get_scores(
                        os.path.join(options.outdir, options.model) + str(epoch + 1) + "_" + str(part) + '.eval')
                    print 'epoch: ' + str(epoch) + ' part: '+ str(part) + '-- labeled F1: ' + str(labeled_f) + ' Unlabaled F: ' + str(
                        unlabeled_f)

                    if float(labeled_f) > best_f_score:
                        self.Save(os.path.join(options.outdir, options.model))
                        best_f_score = float(labeled_f)
                        best_part = part

        print 'best part on this epoch: '+ str(best_part)
        return best_f_score


    def Predict(self, conll_path):
        dev_buckets = [list()]
        dev_data = list(read_conll(conll_path))
        for d in dev_data:
            dev_buckets[0].append(d)
        minibatches = get_batches(dev_buckets, self, False)
        outputs = self.decode(minibatches)
        results = [self.iroles[np.argmax(outputs[i])] for i in range(len(outputs))]
        offset = 0
        for iSentence, sentence in enumerate(dev_data):
            for p in xrange(len(sentence.predicates)):
                for arg_index in xrange(len(sentence.entries)):
                    sentence.entries[arg_index].predicateList[p] = results[offset]
                    offset+=1
            yield sentence