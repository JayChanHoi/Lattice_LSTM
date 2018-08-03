from lattice_lstm.lattice_lstm import LatticeLSTMCell
import tensorflow as tf

lexicon_word_embedding_inputs = tf.constant([[[[1, 2, 3, 1, 2], [1, 2, 3, 4, 5], [2, 2, 3, 4, 5], [1, 2, 3, 4, 5],
                                               [1, 2, 3, 4, 5]],
                                              [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0]],
                                              [[2, 2, 2, 2, 2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0]],
                                              [[1, 3, 3, 3, 2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0]],
                                              [[1, 2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0]],
                                              [[1, 0, 2, 1, 2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0]],
                                              [[1, 0, 2, 1, 2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0]]]],
                                            dtype=tf.float32)

char_embedding_inputs = tf.constant([[[1, 2, 3, 1, 2],
                                      [1, 2, 3, 4, 5],
                                      [2, 2, 3, 4, 5],
                                      [1, 2, 3, 4, 5],
                                      [1, 2, 3, 4, 5],
                                      [1, 1, 1, 0, 0],
                                      [1, 1, 1, 1, 1]]],
                                    dtype=tf.float32)

word_length_tensor = tf.constant([[[2, 3, 4, 5, 6],
                                   [2, 0, 0, 0, 0],
                                   [2, 0, 0, 0, 0],
                                   [2, 0, 0, 0, 0],
                                   [2, 0, 0, 0, 0],
                                   [2, 0, 0, 0, 0],
                                   [3, 0, 0, 0, 0]]], dtype=tf.float32)
lexicon_num_units = 64
char_num_units = 64

lexicon_word_embedding_inputs = tf.tile(input=lexicon_word_embedding_inputs, multiples=[3,1,1,1])
char_embedding_inputs = tf.tile(char_embedding_inputs, multiples=[3,1,1])
word_length_tensor = tf.tile(word_length_tensor, multiples=[3,1,1])

ner_lattice_lstm = LatticeLSTMCell(char_num_units,
                                   lexicon_num_units,
                                   batch_size=3,
                                   seq_len=7,
                                   max_lexicon_words_num=5,
                                   word_length_tensor=word_length_tensor,
                                   dtype=tf.float32)

initial_state = ner_lattice_lstm.zero_state(batch_size=3, dtype=tf.float32)

outputs, state = tf.nn.dynamic_rnn(cell=ner_lattice_lstm,
                                   inputs=[char_embedding_inputs, lexicon_word_embedding_inputs],
                                   initial_state=initial_state,
                                   dtype=tf.float32)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    output = sess.run(outputs)
    print(output)