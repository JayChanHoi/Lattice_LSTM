import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell, RNNCell
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
import numpy as np


class CharLSTM(BasicLSTMCell):
    def __init__(self, lexicon_num_units, batch_size, dtype, reuse=None, name=None, **kwargs):
        super(CharLSTM, self).__init__(reuse=reuse, name=name, **kwargs)
        self._lexicon_num_units = lexicon_num_units
        self._dtype = dtype
        self._char_state_tensor = tf.Variable(tf.zeros(shape=[batch_size, self._num_units]),
                                              dtype=self._dtype,
                                              trainable=False)

    def build(self, inputs_shape):
        if inputs_shape[-1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[-1].value
        h_depth = self._num_units
        lexicon_state_depth = self._lexicon_num_units
        self._kernel = self.add_variable(name='multi_input_kernel', shape=[input_depth + h_depth, 4 * self._num_units])
        self._bias = self.add_variable(name='multi_input_bias',
                                       shape=[4 * self._num_units],
                                       initializer=tf.zeros_initializer(dtype=self._dtype))
        self._linking_kernel = self.add_variable(name='linking_kernel', shape=[input_depth + lexicon_state_depth, self._num_units])
        self._linking_bias = self.add_variable(name='linking_bias',
                                               shape=[self._num_units],
                                               initializer=tf.zeros_initializer(dtype=self._dtype))
        self.built = True

    def call(self, inputs, state):
        char_inputs = inputs[0]  # shape = [batch_size, input_dimension]
        state_inputs = inputs[1]  # shape = [batch_size, max_num_of_lexicon words, lexicon_state_dimension]

        # check whether the last dimension of state_inputs are all zero.
        check_state_0 = tf.reduce_sum(state_inputs, axis=-1)
        check_state_1 = tf.reduce_sum(check_state_0, axis=-1)
        state_inputs_indices_for_lexicon = tf.where(tf.not_equal(check_state_0, 0))
        state_inputs_indices_for_not_lexicon = tf.squeeze(tf.where(tf.equal(check_state_1, 0)))

        # in case `[i]` is squeezed to scalar `i`, change it back to 1-dimension tensor `[i]` by `tf.expand_dims()`
        # otherwise, `[]` and `[i, j]` will remain as-is after tf.squeeze() and further conversion on it
        state_inputs_indices_for_not_lexicon = tf.cond(pred=tf.equal(tf.rank(state_inputs_indices_for_not_lexicon), 0),
                                                       true_fn=lambda: tf.expand_dims(state_inputs_indices_for_not_lexicon, axis=0),
                                                       false_fn=lambda: state_inputs_indices_for_not_lexicon)

        char_inputs_indices_for_lexicon = tf.where(tf.not_equal(tf.reduce_sum(check_state_0, axis=-1), 0))
        char_inputs_indices_for_not_lexicon = tf.where(tf.equal(tf.reduce_sum(check_state_0, axis=-1), 0))

        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        gate_inputs = tf.matmul(tf.concat([char_inputs, h], 1), self._kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)

        i, j, f, o = tf.split(value=gate_inputs, num_or_size_splits=4, axis=1)

        new_c_without_lexicon = self._new_c_without_lexicon(i=i, f=f, j=j, c=c,
                                                            indices_tensor=state_inputs_indices_for_not_lexicon)
        new_c = tf.scatter_nd_update(self._char_state_tensor, indices=char_inputs_indices_for_not_lexicon, updates=new_c_without_lexicon)

        new_c = tf.cond(tf.not_equal(tf.shape(state_inputs_indices_for_not_lexicon)[-1],
                                     tf.shape(state_inputs)[0]),
                        true_fn=lambda: self._if_not_empty_lexicon_state(i, j, char_inputs, state_inputs, char_inputs_indices_for_lexicon,
                                                                         state_inputs_indices_for_lexicon, new_c),
                        false_fn=lambda: new_c)

        new_h = tf.multiply(self._activation(new_c), tf.nn.sigmoid(o))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)

        return new_h, new_state

    def _new_c_without_lexicon(self, i, f, j, c, indices_tensor):
        forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)
        f_without_lexicon_state_input = tf.gather(f, indices=indices_tensor)
        i_without_lexicon_state_input = tf.gather(i, indices=indices_tensor)
        j_without_lexicon_state_input = tf.gather(j, indices=indices_tensor)
        new_c_without_lexicon_state = tf.add(tf.multiply(c, tf.nn.sigmoid(tf.add(f_without_lexicon_state_input, forget_bias_tensor))),
                                             tf.multiply(tf.nn.sigmoid(i_without_lexicon_state_input), self._activation(j_without_lexicon_state_input)))

        return new_c_without_lexicon_state

    def _new_c_with_lexicon(self, i, j, char_inputs, state_inputs, indices_tensor):
        char_inputs_with_lexicon_state = tf.gather_nd(char_inputs, indices=[indices_tensor])
        lexicon_state_inputs = tf.gather_nd(state_inputs, indices=indices_tensor)

        i_with_lexicon_state_input = tf.gather_nd(i, indices=[indices_tensor])
        j_with_lexicon_state_input = tf.gather_nd(j, indices=[indices_tensor])

        state_input_gate = tf.matmul(tf.concat([char_inputs_with_lexicon_state, lexicon_state_inputs], axis=-1),
                                     self._linking_kernel)
        state_input_gate = tf.nn.sigmoid(tf.nn.bias_add(state_input_gate, self._linking_bias))
        state_char_input_gate = tf.concat([state_input_gate, tf.nn.sigmoid(i_with_lexicon_state_input)], axis=1)
        state_gate_weights, char_gate_weight = tf.split(tf.nn.softmax(state_char_input_gate, axis=0),
                                                        num_or_size_splits=[tf.shape(lexicon_state_inputs)[0], 1],
                                                        axis=1)

        new_c_with_lexicon_state = tf.add(tf.reduce_sum(tf.multiply(state_gate_weights, lexicon_state_inputs), axis=0),
                                          tf.multiply(char_gate_weight, j_with_lexicon_state_input))

        return new_c_with_lexicon_state

    def _if_not_empty_lexicon_state(self, i, j, char_inputs, state_inputs,
                                    char_inputs_indices_for_lexicon, state_inputs_indices_for_lexicon, new_c_in):
        new_c_with_lexicon = self._new_c_with_lexicon(i=i, j=j, char_inputs=char_inputs, state_inputs=state_inputs,
                                                      indices_tensor=state_inputs_indices_for_lexicon)
        new_c_out = tf.scatter_nd_update(new_c_in, indices=char_inputs_indices_for_lexicon, updates=new_c_with_lexicon)

        return new_c_out


class LexiconLSTM(BasicLSTMCell):
    def __init__(self, dtype, reuse=None, name=None, **kwargs):
        super(LexiconLSTM, self).__init__(reuse=reuse, name=name, **kwargs)
        self._dtype = dtype

    def build(self, inputs_shape):
        if inputs_shape[-1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[-1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(name='lexicon_kernel',
                                         shape=[input_depth + h_depth, 3 * self._num_units])
        self._bias = self.add_variable(name='lexicon_bias',
                                       shape=[3 * self._num_units],
                                       initializer=tf.zeros_initializer(dtype=self._dtype))
        self.built = True

    def call(self, inputs, state):
        sigmoid = tf.nn.sigmoid
        add = tf.add
        multiply = tf.multiply

        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        gate_inputs = tf.matmul(tf.concat([inputs, h], 1), self._kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)

        i, j, f = tf.split(value=gate_inputs, num_or_size_splits=3, axis=1)

        forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)

        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))

        return new_c


class LatticeLSTMCell(RNNCell):
    ''' inherit from: tf.nn.rnn_cell.RNNCell

        Lattice Long short-term memory unit recurrent network cell. the implementation is based on  https://arxiv.org/pdf/1805.02023.pdf

        Please be noted that LatticeLSTMCell should be called within tf.nn.dynamic_rnn

        example:
        from lattice_lstm import LatticeLSTMCell
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

        char_embedding_inputs = tf.constant([[[1,2,3,1,2],
                                              [1,2,3,4,5],
                                              [2,2,3,4,5],
                                              [1,2,3,4,5],
                                              [1,2,3,4,5],
                                              [1,1,1,0,0],
                                              [1,1,1,1,1]]],
                                            dtype=tf.float32)

        word_length_tensor =  tf.constant([[[2,3,4,5,6],
                                            [2,0,0,0,0],
                                            [2,0,0,0,0],
                                            [2,0,0,0,0],
                                            [2,0,0,0,0],
                                            [2,0,0,0,0],
                                            [3,0,0,0,0]]], dtype=tf.float32)
        lexicon_num_units = 64
        char_num_units = 64

        ner_lattice_lstm = LatticeLSTMCell(char_num_units,
                                           lexicon_num_units,
                                           batch_size=1,
                                           seq_len=7,
                                           max_lexicon_words_num=5,
                                           word_length_tensor=word_length_tensor,
                                           dtype=tf.float32)

        initial_state = ner_lattice_lstm.zero_state(batch_size=1, dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(cell=ner_lattice_lstm,
                                           inputs=[char_embedding_inputs, lexicon_word_embedding_inputs],
                                           initial_state=initial_state,
                                           dtype=tf.float32)
    '''

    def __init__(self, char_num_units, lexicon_num_units, max_lexicon_words_num,
                 word_length_tensor, batch_size, seq_len, dtype, **kwargs):
        super(LatticeLSTMCell, self).__init__(**kwargs)
        '''        
        Parameters
        ----------
        char_num_units: int
            the num_units of lexicon_lstm cell units. this is expected to be the same as char_num_units by the paper.

        lexicon_num_units: int
            the num_units of lexicon_lstm cell units. this is expected to be the same as char_num_units by the paper.

        max_lexicon_words_num: int
            the upper bound of the lexicon words per characters. 

        batch_size: int
            batch_size of the input data

        seq_len: int
            sequence_length of the input data

        dtype:
            data type defined for LatticeLSTMCell variable

        word_length_tensor: tensor
            tensor, it contains a batch of lexicon word length. this should be padded with 0 if there is matched
            lexicon word for the respective character.
            example:  char： 南 京 市 長 江 大 橋
                      word： 南京 市長
                      max_lexicon_words_num = 5
            then the word length tensor should be like tf.constant([[[2,0,0,0,0],
                                                                     [0,0,0,0,0],
                                                                     [2,0,0,0,0],
                                                                     [0,0,0,0,0],
                                                                     [0,0,0,0,0],
                                                                     [0,0,0,0,0],
                                                                     [0,0,0,0,0]]], 
                                                                   dtype=tf.float32)

        '''

        self._char_lstm = CharLSTM(dtype=dtype, num_units=char_num_units, batch_size=batch_size,
                                   lexicon_num_units=lexicon_num_units, name='character_lstm')
        self._lexicon_lstm = LexiconLSTM(dtype=dtype, num_units=lexicon_num_units, name='lexicon_word_lstm')
        self.word_length_tensor = word_length_tensor  # word_length_tesnro should be in the shape of [batched_size, seq_len, max_lexicon_words_num]
        self.max_lexicon_words_num = max_lexicon_words_num
        self.seq_len = seq_len
        self.time_step = 0
        self._dtype = dtype
        lexicon_state_init_value = tf.zeros(shape=[batch_size, self.seq_len, self.max_lexicon_words_num, lexicon_num_units])
        self.lexicon_state_tensor = tf.Variable(initial_value=lexicon_state_init_value,
                                                trainable=False,
                                                dtype=self._dtype)
        # lexicon_state_tensor should be in the shape of [batch_size, seq_len, max_lexicon_words_num, state_dimension]

    def build(self, inputs_shape):
        self._char_lstm.build(inputs_shape[0])
        self._lexicon_lstm.build(inputs_shape[1])
        self.lexicon_shape = inputs_shape[1]
        if self.lexicon_shape[1] != self.max_lexicon_words_num:
            raise ValueError('max_lexicon_words_num should be equal to lexicon input')

        self.built = True

    @property
    def state_size(self):
        return self._char_lstm.state_size

    @property
    def output_size(self):
        return self._char_lstm.output_size

    def zero_state(self, batch_size, dtype):
        return self._char_lstm.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        '''
        Parameters
        ----------
        inputs: list of tensors
            inputs here should be a tensors of character_inputs(character embedding inputs) and
            lexicon_inputs(lexicon word embedding inputs). char_embedding_inputs has shape [batch_size, char_embedding_size]
            lexicon_embedding_inputs has shape [batch_size, max_lexicon_words_num, lexicon_word_embedding_size]
            please be noted that number of lexicon words is expect to be upper bounded. so if the number of words
            for a character is less than max_lexicon_words_num, the word embedding inputs should be padded with zero.
            example: inputs = [char_embedding_inputs, word_embedding_inputs]

        state: tensors
            Either a single 2-D tensor, or a tuple of tensors matching the arity and shapes of state.

        Returns
        -------
        output: tensor
            tensor of hidden output of character units with shape [batch_size, self.output_size]

        new_state: tensor or a tuple of tensors
            Either a single 2-D tensor, or a tuple of tensors matching the arity and shapes of state.
        '''

        char_input = inputs[0]  # shape = [batch_size, char_embedding_size]
        lexicon_inputs = inputs[1]  # shape = [batch_size, max_lexicon_words_num, lexicon_word_embedding_size]

        lexicon_state_tensor = tf.gather(self.lexicon_state_tensor, axis=1, indices=self.time_step)
        char_hidden_output, char_state = self._char_lstm.call([char_input, lexicon_state_tensor], state)

        for word_index in range(self.max_lexicon_words_num):
            self.lexicon_state_tensor = self._update_lexicon_state_per_word(lexicon_inputs=lexicon_inputs,
                                                                            word_index=word_index,
                                                                            char_state=char_state)
        self.time_step = self.time_step + 1
        # reset the lexicon_state_tensor after finish a loop
        self.lexicon_state_tensor = tf.cond(tf.equal(tf.mod(self.time_step, self.seq_len - 1), 0),
                                            true_fn=lambda: tf.assign(ref=self.lexicon_state_tensor,
                                                                      value=tf.zeros_like(self.lexicon_state_tensor)),
                                            false_fn=lambda: self.lexicon_state_tensor)
        self.time_step = np.remainder(self.time_step, self.seq_len - 1)

        return char_hidden_output, char_state

    def _update_lexicon_state_per_word(self, lexicon_inputs, word_index, char_state):
        lexicon_input_per_word = tf.gather(lexicon_inputs, axis=1, indices=word_index)

        word_length_per_time_step = tf.gather(self.word_length_tensor, axis=1, indices=self.time_step)
        word_length = tf.gather(word_length_per_time_step, axis=1, indices=word_index)

        lexicon_state = self._lexicon_lstm.call(lexicon_input_per_word, char_state)
        temp_lexicon_state_to_char_index = self.time_step + word_length - 1
        lexicon_state_index = tf.where(tf.not_equal(temp_lexicon_state_to_char_index, self.time_step - 1))
        lexicon_state_to_char_index = tf.gather_nd(temp_lexicon_state_to_char_index, indices=lexicon_state_index)

        lexicon_state_update = tf.gather_nd(lexicon_state, indices=lexicon_state_index)

        word_index_for_stack = tf.ones_like(lexicon_state_to_char_index) * word_index
        lexicon_state_index_for_stack = tf.cast(tf.squeeze(lexicon_state_index), dtype=self._dtype)
        indices = tf.stack([lexicon_state_index_for_stack, lexicon_state_to_char_index, word_index_for_stack], axis=-1)

        updated_lexicon_state_tensor = tf.scatter_nd_update(ref=self.lexicon_state_tensor,
                                                            indices=tf.cast(indices, dtype=tf.int32),
                                                            updates=lexicon_state_update)

        return updated_lexicon_state_tensor