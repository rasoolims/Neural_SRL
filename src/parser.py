import os
import pickle
import time
import utils
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default=None)
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default='')
    parser.add_option("--input", dest="input", help="Annotated CONLL test file", metavar="FILE", default=None)
    parser.add_option("--output", dest="output", help="output file", metavar="FILE", default=None)
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model")
    parser.add_option("--d_w", type="int", dest="d_w", default=100)
    parser.add_option("--d_l", type="int", dest="d_l", default=100)
    parser.add_option("--d_pos", type="int", dest="d_pos", default=16)
    parser.add_option("--d_char", type="int", dest="d_char", default=30)
    parser.add_option("--d_lstm_char", type="int", dest="d_lstm_char", default=100)
    parser.add_option("--d_h", type="int", dest="d_h", default=512)
    parser.add_option("--d_r", type="int", dest="d_r", default=128)
    parser.add_option("--d_prime_l", type="int", dest="d_prime_l", default=128)
    parser.add_option("--k", type="int", dest="k", default=4)
    parser.add_option("--batch", type="int", dest="batch", default=10)
    parser.add_option("--alpha", type="float", dest="alpha", default=0.25)
    parser.add_option("--beta2", type="float", dest="beta2", default=0.999)
    parser.add_option("--learning_rate", type="float", dest="learning_rate", default=0.001)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--outdir", type="string", dest="outdir", default="results")
    parser.add_option("--mem", type="int", dest="mem", default="2048")
    parser.add_option("--save_epoch", action="store_true", dest="save_epoch", default=False, help='Save each epoch.')
    parser.add_option("--drop", action="store_true", dest="drop", default=False, help='Use dropout.')
    parser.add_option("--char", action="store_true", dest="use_char_lstm", default=False, help='Use char LSTM.')
    parser.add_option("--cluster", dest="cluster_file", help="Use Brown cluster embeddings for rare words.", metavar="FILE", default=None)
    parser.add_option("--min_freq", type="int", dest="min_freq", help="Max frequency for replacing with clusters", default="5")

    (options, args) = parser.parse_args()
    print 'Using external embedding:', options.external_embedding

    import _dynet as dy
    dyparams = dy.DynetParams()
    dyparams.from_args()
    dyparams.set_mem(options.mem)
    dyparams.init()
    from srl import SRLLSTM

    if options.conll_train:
        print 'Preparing vocab'
        print options
        words, lemma_count,w2i, pos, semRels, pl2i,chars = utils.vocab(options.conll_train, 0 if options.cluster_file==None else options.min_freq)
        clusters = None
        if options.cluster_file != None:
            clusters = dict()
            for f in open(options.cluster_file, 'r'):
                clusters[f.strip().split()[1]] = 'c:'+f.strip().split()[0]

            seen_clusters = set()
            for w in w2i.keys():
                if w in clusters:
                    seen_clusters.add(clusters[w])

            for w in clusters.keys():
                if not clusters[w] in seen_clusters:
                    del clusters[w]

            l = len(w2i)
            offset = l
            for c in set(clusters.values()):
                w2i[c] = offset
                offset+=1
            print 'loaded',len(clusters),'words with',len(set(clusters.values())),'clusters and',l,'in-vocabulary words out of',len(words)

        with open(os.path.join(options.outdir, options.params), 'w') as paramsfp:
            pickle.dump((words,lemma_count,w2i, pos, semRels, pl2i, chars,clusters, options), paramsfp)
        print 'Finished collecting vocab'

        print 'Initializing blstm srl:'
        parser = SRLLSTM(words,lemma_count, pos, semRels, w2i, pl2i, chars,clusters, options)
        for epoch in xrange(options.epochs):
            print 'Starting epoch', epoch
            parser.Train(options.conll_train)
            if options.save_epoch:  parser.Save(os.path.join(options.outdir, options.model + str(epoch + 1)))
            if options.conll_dev != '':
                start = time.time()
                utils.write_conll(os.path.join(options.outdir, options.model) + str(epoch+1)+ '.txt', parser.Predict(options.conll_dev))
                os.system(
                    'perl src/utils/eval.pl -g ' + options.conll_dev + ' -s ' +  os.path.join(options.outdir, options.model) + str(epoch+1)+ '.txt' + ' > ' +  os.path.join(options.outdir, options.model) + str(epoch+1)+ '.eval &')
                print 'Finished predicting dev; time:', time.time() - start
        parser.Save(os.path.join(options.outdir, options.model))

    if options.input and options.output:
        with open(options.params, 'r') as paramsfp:
            words,lemma_count,w2i, pos, semRels, pl2i, chars, clusters,stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = options.external_embedding
        parser = SRLLSTM(words,lemma_count,pos, semRels, w2i, pl2i, chars,clusters,stored_opt)
        parser.Load(options.model)
        ts = time.time()
        pred = list(parser.Predict(options.input))
        te = time.time()
        utils.write_conll(options.output, pred)
        print 'Finished predicting test', te - ts