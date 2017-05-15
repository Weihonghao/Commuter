from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, datetime
import logging
from tqdm import tqdm
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from operator import mul
from tensorflow.python.ops import variable_scope as vs
from utils.util import ConfusionMatrix, Progbar, minibatches, one_hot, minibatch, get_best_span

logging.basicConfig(level=logging.INFO)

class Encoder(object):
    def __init__(self, vocab_dim, state_size, dropout = 0):
        self.vocab_dim = vocab_dim
        self.state_size = state_size
        #self.dropout = dropout
        #logging.info("Dropout rate for encoder: {}".format(self.dropout))

    def encode(self, inputs, mask, encoder_state_input, dropout = 1.0):
        """
        In a generalized encode function, you pass in your inputs,
        sequence_length, and an initial hidden state input into this function.

        :param inputs: Symbolic representations of your input (padded all to the same length)
        :param mask: mask of the sequence
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        logging.debug('-'*5 + 'encode' + '-'*5)
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)


        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob = dropout)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob = dropout)

        initial_state_fw = None
        initial_state_bw = None
        if encoder_state_input is not None:
            initial_state_fw, initial_state_bw = encoder_state_input

        logging.debug('Inputs: %s' % str(inputs))
        sequence_length = tf.reduce_sum(tf.cast(mask, 'int32'), axis=1)
        sequence_length = tf.reshape(sequence_length, [-1,])
        # Get lstm cell output
        (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,\
                                                      cell_bw=lstm_bw_cell,\
                                                      inputs=inputs,\
                                                      sequence_length=sequence_length,
                                                      initial_state_fw=initial_state_fw,\
                                                      initial_state_bw=initial_state_bw,
                                                      dtype=tf.float32)

        # Concatinate forward and backword hidden output vectors.
        # each vector is of size [batch_size, sequence_length, cell_state_size]

        logging.debug('fw hidden state: %s' % str(outputs_fw))
        hidden_state = tf.concat(2, [outputs_fw, outputs_bw])
        logging.debug('Concatenated bi-LSTM hidden state: %s' % str(hidden_state))
        # final_state_fw and final_state_bw are the final states of the forwards/backwards LSTM
        concat_final_state = tf.concat(1, [final_state_fw[1], final_state_bw[1]])
        logging.debug('Concatenated bi-LSTM final hidden state: %s' % str(concat_final_state))
        return hidden_state, concat_final_state, (final_state_fw, final_state_bw)


class ChatSystem(object):
    """docstring for ChatSystem"""
    def __init__(self, pretrained_embeddings, config):
        super(ChatSystem, self).__init__()
        self.pretrained_embeddings = pretrained_embeddings
        self.encoder = Encoder(vocab_dim=config.embedding_size, state_size = config.encoder_state_size)
        
        # ==== set up placeholder tokens ====
        self.question_placeholder = tf.placeholder(dtype=tf.int32, name="q", shape=(None, None))
        self.question_mask_placeholder = tf.placeholder(dtype=tf.bool, name="q_mask", shape=(None, None))
        self.answer_placeholders = tf.placeholder(dtype=tf.int32, name="a", shape=(None, None))
        self.answer_mask_placeholders = tf.placeholder(dtype=tf.int32, name="a_mask", shape=(None, None))
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name="dropout", shape=())
        self.JQ = tf.placeholder(dtype=tf.int32, name='JQ', shape=()) # Length of question
        self.JA = tf.placeholder(dtype=tf.int32, name='JA', shape=()) # Length of answer
    
        # ==== assemble pieces ====
        with tf.variable_scope("chat", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.q = self.setup_embeddings()
            self.preds = self.setup_system(self.q)
            self.loss = self.setup_loss(self.preds)

        # ==== set up training/updating procedure ====
        # With gradient clipping:
        opt_op = get_optimizer("adam", self.loss, config.max_gradient_norm, config.learning_rate)

        if config.ema_weight_decay is not None:
            self.train_op = self.build_ema(opt_op)
        else:
            self.train_op = opt_op
        self.merged = tf.summary.merge_all()

    def build_ema(self, opt_op):
        self.ema = tf.train.ExponentialMovingAverage(self.config.ema_weight_decay)
        ema_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([opt_op]):
            train_op = tf.group(ema_op)
        return train_op

    def setup_system(self, q):
        d = x.get_shape().as_list()[-1] # self.config.embedding_size
            #   x: [None, JX, d]
            #   q: [None, JQ, d]
        assert x.get_shape().ndims == 3
        assert q.get_shape().ndims == 3

        # Step 1: encode q, respectively, with independent weights
        #         i.e. u = encode_question(q)  # get U (2d*J) as representation of q
        with tf.variable_scope('q'):
            u, question_repr, u_state = \
                 self.encoder.encode(inputs=q, mask=self.question_mask_placeholder, encoder_state_input=None, dropout = self.dropout_placeholder)

        d_en = self.config.encoder_state_size*2
        assert u.get_shape().as_list() == [None, None, d_en], "Expected {}, got {}".format([None, JQ, d_en], u.get_shape().as_list())

        # Step 2:
        # 2 LSTM layers
        # logistic regressions
        pred1, pred2 = self.decoder.decode(g, self.context_mask_placeholder, dropout = self.dropout_placeholder, JX = self.JX)
        return pred1, pred2

    def setup_loss(self, preds):
        with vs.variable_scope("loss"):
            s, e = preds # [None, JX]*2
            assert s.get_shape().ndims == 2
            assert e.get_shape().ndims == 2
            loss1 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholders),)
            loss2 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholders),)
            # loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholders),)
            # loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholders),)
        loss = loss1 + loss2
        tf.summary.scalar('loss', loss)
        return loss

    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            if self.config.retrain_embed:
                pretrained_embeddings = tf.Variable(self.pretrained_embeddings, name="Emb", dtype=tf.float32)
            else:
                pretrained_embeddings = tf.cast(self.pretrained_embeddings, tf.float32)

            question_embeddings = tf.nn.embedding_lookup(pretrained_embeddings, self.question_placeholder)
            question_embeddings = tf.reshape(question_embeddings, shape = [-1, self.JQ, self.config.embedding_size])

        return question_embeddings

    def create_feed_dict(self, question_batch, question_len_batch, context_batch, context_len_batch, JX=10, JQ=10, answer_batch=None, is_train = True):
        feed_dict = {}
        JQ = np.max(question_len_batch)
        JX = np.max(context_len_batch)
        # print('This batch len: JX = %d, JQ = %d', JX, JQ)
        def add_paddings(sentence, max_length):
            mask = [True] * len(sentence)
            pad_len = max_length - len(sentence)
            if pad_len > 0:
                padded_sentence = sentence + [0] * pad_len
                mask += [False] * pad_len
            else:
                padded_sentence = sentence[:max_length]
                mask = mask[:max_length]
            return padded_sentence, mask

        def padding_batch(data, max_len):
            padded_data = []
            padded_mask = []
            for sentence in data:
                d, m = add_paddings(sentence, max_len)
                padded_data.append(d)
                padded_mask.append(m)
            return (padded_data, padded_mask)

        question, question_mask = padding_batch(question_batch, JQ)
        context, context_mask = padding_batch(context_batch, JX)

        feed_dict[self.question_placeholder] = question
        feed_dict[self.question_mask_placeholder] = question_mask
        feed_dict[self.context_placeholder] = context
        feed_dict[self.context_mask_placeholder] = context_mask
        feed_dict[self.JQ] = JQ
        feed_dict[self.JX] = JX

        if answer_batch is not None:
            start = answer_batch[:,0]
            end = answer_batch[:,1]
            feed_dict[self.answer_start_placeholders] = start
            feed_dict[self.answer_end_placeholders] = end
        if is_train:
            feed_dict[self.dropout_placeholder] = 0.8
        else:
            feed_dict[self.dropout_placeholder] = 1.0

        return feed_dict

    def optimize(self, session, training_set):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = training_set
        input_feed = self.create_feed_dict(question_batch, question_len_batch, context_batch, context_len_batch, answer_batch=answer_batch, is_train = True)

        output_feed = [self.train_op, self.merged, self.loss]

        outputs = session.run(output_feed, input_feed)
        return outputs

    def run_epoch(self, session, epoch_num, training_set, vocab, validation_set, sample_size=400):
        set_num = len(training_set)
        batch_size = self.config.batch_size
        batch_num = int(np.ceil(set_num * 1.0 / batch_size))
        sample_size = 400

        prog = Progbar(target=batch_num)
        avg_loss = 0
        for i, batch in enumerate(minibatches(training_set, self.config.batch_size, window_batch = self.config.window_batch)):
            global_batch_num = batch_num * epoch_num + i
            _, summary, loss = self.optimize(session, batch)
            prog.update(i + 1, [("training loss", loss)])
            if self.config.tensorboard and global_batch_num % self.config.log_batch_num == 0:
                self.train_writer.add_summary(summary, global_batch_num)
            if (i+1) % self.config.log_batch_num == 0:
                logging.info('')
                self.evaluate_answer(session, training_set, vocab, sample=sample_size, log=True)
                self.evaluate_answer(session, validation_set, vocab, sample=sample_size, log=True)
            avg_loss += loss
        avg_loss /= batch_num
        logging.info("Average training loss: {}".format(avg_loss))
        return avg_loss


    def train(self, session, dataset, train_dir, vocab):
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        training_set = dataset['training'] # [question, len(question), context, len(context), answer]
        validation_set = dataset['validation']
        f1_best = 0
        if self.config.tensorboard:
            train_writer_dir = self.config.log_dir + '/train/' # + datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
            self.train_writer = tf.summary.FileWriter(train_writer_dir, session.graph)
        for epoch in range(self.config.epochs):
            logging.info("="* 10 + " Epoch %d out of %d " + "="* 10, epoch + 1, self.config.epochs)

            score = self.run_epoch(session, epoch, training_set, vocab, validation_set, sample_size=self.config.evaluate_sample_size)
            logging.info("-- validation --")
            self.validate(session, validation_set)

            f1, em = self.evaluate_answer(session, validation_set, vocab, sample=self.config.model_selection_sample_size, log=True)
            # Saving the model
            if f1>f1_best:
                f1_best = f1
                saver = tf.train.Saver()
                saver.save(session, train_dir+'/baseline')
                logging.info('New best f1 in val set')
                logging.info('')
            saver = tf.train.Saver()
            saver.save(session, train_dir+'/baseline_' + str(epoch))
