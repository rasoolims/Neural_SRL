import os
import pickle
import time
import utils
from optparse import OptionParser

from srl import SRLLSTM

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                      default=None)
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE",
                      default=None)
    parser.add_option("--input", dest="input", help="Annotated CONLL test file", metavar="FILE",
                      default=None)
    parser.add_option("--output", dest="output", help="output file", metavar="FILE",
                      default=None)
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="barchybrid.model")
    parser.add_option("--d_w", type="int", dest="d_w", default=100)
    parser.add_option("--d_l", type="int", dest="d_l", default=100)
    parser.add_option("--d_pos", type="int", dest="d_pos", default=16)
    parser.add_option("--d_h", type="int", dest="d_h", default=512)
    parser.add_option("--d_r", type="int", dest="d_r", default=128)
    parser.add_option("--d_prime_l", type="int", dest="d_prime_l", default=128)
    parser.add_option("--k", type="int", dest="k", default=4)
    parser.add_option("--batch", type="int", dest="batch", default=10)
    parser.add_option("--alpha", type="float", dest="alpha", default=0.25)
    parser.add_option("--learning_rate", type="float", dest="learning_rate", default=0.01)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--outdir", type="string", dest="outdir", default="results")

    (options, args) = parser.parse_args()
    print 'Using external embedding:', options.external_embedding

    if options.conll_train:
        print 'Preparing vocab'
        words,w2i, l2i, pos, semRels, pl2i = utils.vocab(options.conll_train)

        with open(os.path.join(options.outdir, options.params), 'w') as paramsfp:
            pickle.dump((words,w2i, l2i, pos, semRels, pl2i, options), paramsfp)
        print 'Finished collecting vocab'


        print 'Initializing blstm srl:'
        parser = SRLLSTM(words, pos, semRels, w2i, l2i, pl2i, options)
        for epoch in xrange(options.epochs):
            print 'Starting epoch', epoch
            parser.Train(options.conll_train, options.conll_dev, os.path.join(options.outdir, options.model))
            devpath = os.path.join(options.outdir, 'dev_epoch_' + str(epoch + 1) + '.conll')
            if options.conll_dev!='':
                utils.write_conll(devpath, parser.Predict(options.conll_dev))
                os.system('perl src/utils/eval.pl -g ' + options.conll_dev + ' -s ' + devpath + ' > ' + devpath + '.txt &')
                print 'Finished predicting dev'
            parser.Save(os.path.join(options.outdir, options.model + str(epoch + 1)))

    if options.input and options.output:
        with open(options.params, 'r') as paramsfp:
            words,w2i, l2i, pos, semRels, pl2i, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = options.external_embedding
        parser = SRLLSTM(words,pos, semRels, w2i, l2i, pl2i, stored_opt)
        parser.Load(options.model)
        ts = time.time()
        pred = list(parser.Predict(options.input))
        te = time.time()
        utils.write_conll(options.output, pred)
        print 'Finished predicting test', te - ts
