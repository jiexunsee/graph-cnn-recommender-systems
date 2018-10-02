from __future__ import print_function


from initializations import *
import tensorflow as tf

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)

    return pre_out * tf.div(1., keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    # Properties
        name: String, defines the variable scope of the layer.
            Layers with common name share variables. (TODO)
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer for two types of nodes in a bipartite graph. """

    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, share_user_item_weights=False,
                 bias=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        with tf.variable_scope(self.name + '_vars'):
            if not share_user_item_weights:

                self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name="weights_u")
                self.vars['weights_v'] = weight_variable_random_uniform(input_dim, output_dim, name="weights_v")

                if bias:
                    self.vars['user_bias'] = bias_variable_truncated_normal([output_dim], name="bias_u")
                    self.vars['item_bias'] = bias_variable_truncated_normal([output_dim], name="bias_v")


            else:
                self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name="weights")
                self.vars['weights_v'] = self.vars['weights_u']

                if bias:
                    self.vars['user_bias'] = bias_variable_truncated_normal([output_dim], name="bias_u")
                    self.vars['item_bias'] = self.vars['user_bias']

        self.bias = bias

        self.dropout = dropout
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x_u = inputs[0]
        x_u = tf.nn.dropout(x_u, 1 - self.dropout)
        x_u = tf.matmul(x_u, self.vars['weights_u'])

        x_v = inputs[1]
        x_v = tf.nn.dropout(x_v, 1 - self.dropout)
        x_v = tf.matmul(x_v, self.vars['weights_v'])

        u_outputs = self.act(x_u)
        v_outputs = self.act(x_v)

        if self.bias:
            u_outputs += self.vars['user_bias']
            v_outputs += self.vars['item_bias']

        return u_outputs, v_outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v

""" NEW LAYER """
class SumRGGCN(Layer):
    """Residual gated graph convolutional layer (Bresson). adapted from stackGC layer """
    def __init__(self, input_dim, output_dim, E_start, E_end, num_support, u_features_nonzero=None,
                 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
                 act=tf.nn.relu, share_user_item_weights=True, **kwargs):
        super(SumRGGCN, self).__init__(**kwargs)

        assert output_dim % num_support == 0, 'output_dim must be multiple of num_support for stackGC layer'

        with tf.variable_scope(self.name + '_vars'):
            # conv1
            self.Ui1 = weight_variable_random_uniform(input_dim, output_dim, name='Ui1')
            self.Uj1 = weight_variable_random_uniform(input_dim, output_dim, name='Uj1')
            self.Vi1 = weight_variable_random_uniform(input_dim, output_dim, name='Vi1')
            self.Vj1 = weight_variable_random_uniform(input_dim, output_dim, name='Vi1')
            self.bu1 = bias_variable_zero(output_dim, name='bu1')
            self.bv1 = bias_variable_zero(output_dim, name='bv1')

            # conv2
            self.Ui2 = weight_variable_random_uniform(output_dim, output_dim, name='Ui2')
            self.Uj2 = weight_variable_random_uniform(output_dim, output_dim, name='Uj2')
            self.Vi2 = weight_variable_random_uniform(output_dim, output_dim, name='Vi2')
            self.Vj2 = weight_variable_random_uniform(output_dim, output_dim, name='Vi2')
            self.bu2 = bias_variable_zero(output_dim, name='bu2')
            self.bv2 = bias_variable_zero(output_dim, name='bv2')
            
            # resnet
            self.R = weight_variable_random_uniform(input_dim, output_dim, name='R')

        self.act = act

        self.E_start = tf.split(E_start, axis=0, num_or_size_splits=num_support)
        self.E_end = tf.split(E_end, axis=0, num_or_size_splits=num_support)

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        num_users = len(inputs[0])
        num_items = len(inputs[1])
        x = np.vstack(inputs)  # CHECK THIS! need to combine users and items into one single array
            
        # E_start, E_end : E x V
        xin = x
        # conv1
        Vix = tf.matmul(x, self.Vi1)
        Vjx = tf.matmul(x, self.Vj1)
        x1 = tf.add_n([tf.matmul(E_end, Vix), tf.matmul(E_start, Vjx), self.bv1])
        x1 = tf.nn.sigmoid(x1)
        Uix = tf.matmul(x, self.Ui1)
        x2 = tf.matmul(E_start, Uix)
        Ujx = tf.matmul(x, self.Uj1)
        x = tf.add_n([Ujx, tf.matmul(E_end.T, tf.multiply(x1, x2)), self.bu1])
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)

        # conv2
        Vix = tf.matmul(x, self.Vi2)
        Vjx = tf.matmul(x, self.Vj2)
        x1 = tf.add_n([tf.matmul(E_end, Vix), tf.matmul(E_start, Vjx), self.bv2])
        x1 = tf.nn.sigmoid(x1)
        Uix = tf.matmul(x, self.Ui2)
        x2 = tf.matmul(E_start, Uix)
        Ujx = tf.matmul(x, self.Uj2)
        x = tf.add_n([Ujx, tf.matmul(E_end.T, tf.multiply(x1, x2)), self.bu2])
        x = tf.layers.batch_normalization(x)

        x = tf.add(x, tf.matmul(self.R, x))
        x = tf.nn.relu(x)

        return x[:num_users], x[num_users:]

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v

""" NEW LAYER """
class StackRGGCN(Layer):
    """Residual gated graph convolutional layer (Bresson). adapted from stackGC layer """
    def __init__(self, input_dim, output_dim, E_start_list, E_end_list, num_support, u_features_nonzero=None,
                 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
                 act=tf.nn.relu, share_user_item_weights=True, **kwargs):
        super(StackRGGCN, self).__init__(**kwargs)

        assert output_dim % num_support == 0, 'output_dim must be multiple of num_support for stackGC layer'

        self.sparse_inputs = sparse_inputs

        with tf.variable_scope(self.name + '_vars'):
            # conv1 (with split weights)
            self.Ui1 = self.get_weight_variable(input_dim, output_dim, num_support, 'Ui1')
            self.Uj1 = self.get_weight_variable(input_dim, output_dim, num_support, 'Uj1')
            self.Vi1 = self.get_weight_variable(input_dim, output_dim, num_support, 'Vi1')
            self.Vj1 = self.get_weight_variable(input_dim, output_dim, num_support, 'Vj1')
            self.bu1 = self.get_bias_variable(output_dim, num_support, 'bu1')
            self.bv1 = self.get_bias_variable(output_dim, num_support, 'bv1')

            # conv2 (with split weights)
            self.Ui2 = self.get_weight_variable(output_dim, output_dim, num_support, 'Ui2')
            self.Uj2 = self.get_weight_variable(output_dim, output_dim, num_support, 'Uj2')
            self.Vi2 = self.get_weight_variable(output_dim, output_dim, num_support, 'Vi2')
            self.Vj2 = self.get_weight_variable(output_dim, output_dim, num_support, 'Vj2')
            self.bu2 = self.get_bias_variable(output_dim, num_support, 'bu2')
            self.bv2 = self.get_bias_variable(output_dim, num_support, 'bv2')
            
            # resnet
            self.R = weight_variable_random_uniform(input_dim, output_dim, name='R')

        self.act = act
        
        # self.E_start = tf.split(E_start, axis=0, num_or_size_splits=num_support)
        # self.E_start = tf.sparse_split(axis=0, num_split=num_support, sp_input=E_start)
        # self.E_end = tf.split(E_end, axis=0, num_or_size_splits=num_support)
        # self.E_end = tf.sparse_split(axis=0, num_split=num_support, sp_input=E_end)

        self.E_start = []
        self.E_end = []
        for i in range(num_support):
            self.E_start.append(E_start_list[i])
            self.E_end.append(E_end_list[i])

        if self.logging:
            self._log_vars()

    def get_weight_variable(self, input_dim, output_dim, num_support, name):
        var = weight_variable_random_uniform(input_dim, output_dim, name=name)
        var = tf.split(value=var, axis=1, num_or_size_splits=num_support)
        return var

    def get_bias_variable(self, output_dim, num_support, name):
        var = bias_variable_zero(output_dim, name=name)
        var = tf.split(value=var, axis=0, num_or_size_splits=num_support)
        return var

    def _call(self, inputs):
        num_users = inputs[0].dense_shape[0]
        num_items = inputs[1].dense_shape[0]
        users = tf.sparse_to_dense(inputs[0].indices, inputs[0].dense_shape, inputs[0].values)
        items = tf.sparse_to_dense(inputs[1].indices, inputs[1].dense_shape, inputs[1].values)
        original_x = tf.concat([users, items], axis=0)  # CHECK THIS! need to combine users and items into one single array. becomes 6000 (users+items) x 6000 (input_dim)

        outputs = []
        for i in range(len(self.E_start)):
            # E_start, E_end : E x V
            x = original_x
            # conv1
            Vix = dot(x, self.Vi1[i])  # Vij[i] is 6000x100
            Vjx = dot(x, self.Vj1[i])
            x1 = tf.add(dot(self.E_end[i], Vix, sparse=True), dot(self.E_start[i], Vjx, sparse=True))
            x1 = tf.nn.bias_add(x1, self.bv1[i])
            x1 = tf.nn.sigmoid(x1)
            Uix = dot(x, self.Ui1[i])
            x2 = dot(self.E_start[i], Uix, sparse=True)
            Ujx = dot(x, self.Uj1[i])
            x = tf.add(Ujx, dot(tf.sparse_transpose(self.E_end[i]), tf.multiply(x1, x2), sparse=True))
            x = tf.nn.bias_add(x, self.bu1[i])
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            outputs.append(x)
        output = tf.concat(axis=1, values=outputs)

        outputs = []
        for i in range(len(self.E_start)):
            x = output
            # conv2
            Vix = dot(x, self.Vi2[i])
            Vjx = dot(x, self.Vj2[i])
            x1 = tf.add(dot(self.E_end[i], Vix, sparse=True), dot(self.E_start[i], Vjx, sparse=True))
            x1 = tf.nn.bias_add(x1, self.bv1[i])
            x1 = tf.nn.sigmoid(x1)
            Uix = dot(x, self.Ui2[i])
            x2 = dot(self.E_start[i], Uix, sparse=True)
            Ujx = dot(x, self.Uj2[i])
            x = tf.add(Ujx, dot(tf.sparse_transpose(self.E_end[i]), tf.multiply(x1, x2), sparse=True))
            x = tf.nn.bias_add(x, self.bu1[i])
            x = tf.layers.batch_normalization(x)
            outputs.append(x)

        output = tf.concat(axis=1, values=outputs)
        output = tf.add(output, tf.matmul(original_x, self.R))
        output = tf.nn.relu(output)

        outputs = self.act(output)

        u = output[:tf.cast(num_users, tf.int32)]
        v = output[tf.cast(num_users, tf.int32):]

        return u, v

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs: # this will if tensors are sparse. sparse_inputs flag needs to be set properly.
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v


class StackGCN(Layer):
    """Graph convolution layer for bipartite graphs and sparse inputs."""

    def __init__(self, input_dim, output_dim, support, support_t, num_support, u_features_nonzero=None,
                 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
                 act=tf.nn.relu, share_user_item_weights=True, **kwargs):
        super(StackGCN, self).__init__(**kwargs)

        assert output_dim % num_support == 0, 'output_dim must be multiple of num_support for stackGC layer'

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name='weights_u')

            if not share_user_item_weights:
                self.vars['weights_v'] = weight_variable_random_uniform(input_dim, output_dim, name='weights_v')

            else:
                self.vars['weights_v'] = self.vars['weights_u']

        self.weights_u = tf.split(value=self.vars['weights_u'], axis=1, num_or_size_splits=num_support)  # <-- HERE is where weights get split
        self.weights_v = tf.split(value=self.vars['weights_v'], axis=1, num_or_size_splits=num_support)

        self.dropout = dropout

        self.sparse_inputs = sparse_inputs
        self.u_features_nonzero = u_features_nonzero
        self.v_features_nonzero = v_features_nonzero
        if sparse_inputs:
            assert u_features_nonzero is not None and v_features_nonzero is not None, \
                'u_features_nonzero and v_features_nonzero can not be None when sparse_inputs is True'

        self.support = tf.sparse_split(axis=1, num_split=num_support, sp_input=support)
        self.support_transpose = tf.sparse_split(axis=1, num_split=num_support, sp_input=support_t)

        self.act = act

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x_u = inputs[0]
        x_v = inputs[1]

        if self.sparse_inputs:
            x_u = dropout_sparse(x_u, 1 - self.dropout, self.u_features_nonzero)
            x_v = dropout_sparse(x_v, 1 - self.dropout, self.v_features_nonzero)
        else:
            x_u = tf.nn.dropout(x_u, 1 - self.dropout)
            x_v = tf.nn.dropout(x_v, 1 - self.dropout)

        supports_u = []
        supports_v = []

        for i in range(len(self.support)):
            tmp_u = dot(x_u, self.weights_u[i], sparse=self.sparse_inputs)
            tmp_v = dot(x_v, self.weights_v[i], sparse=self.sparse_inputs)

            support = self.support[i]
            support_transpose = self.support_transpose[i]

            supports_u.append(tf.sparse_tensor_dense_matmul(support, tmp_v))
            supports_v.append(tf.sparse_tensor_dense_matmul(support_transpose, tmp_u))

        z_u = tf.concat(axis=1, values=supports_u)
        z_v = tf.concat(axis=1, values=supports_v)

        u_outputs = self.act(z_u)
        v_outputs = self.act(z_v)

        return u_outputs, v_outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v


class OrdinalMixtureGCN(Layer):

    """Graph convolution layer for bipartite graphs and sparse inputs."""

    def __init__(self, input_dim, output_dim, support, support_t, num_support, u_features_nonzero=None,
                 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
                 act=tf.nn.relu, bias=False, share_user_item_weights=False, self_connections=False, **kwargs):
        super(OrdinalMixtureGCN, self).__init__(**kwargs)

        with tf.variable_scope(self.name + '_vars'):

            self.vars['weights_u'] = tf.stack([weight_variable_random_uniform(input_dim, output_dim,
                                                                             name='weights_u_%d' % i)
                                              for i in range(num_support)], axis=0)

            if bias:
                self.vars['bias_u'] = bias_variable_const([output_dim], 0.01, name="bias_u")

            if not share_user_item_weights:
                self.vars['weights_v'] = tf.stack([weight_variable_random_uniform(input_dim, output_dim,
                                                                                 name='weights_v_%d' % i)
                                                  for i in range(num_support)], axis=0)

                if bias:
                    self.vars['bias_v'] = bias_variable_const([output_dim], 0.01, name="bias_v")

            else:
                self.vars['weights_v'] = self.vars['weights_u']
                if bias:
                    self.vars['bias_v'] = self.vars['bias_u']

        self.weights_u = self.vars['weights_u']
        self.weights_v = self.vars['weights_v']

        self.dropout = dropout

        self.sparse_inputs = sparse_inputs
        self.u_features_nonzero = u_features_nonzero
        self.v_features_nonzero = v_features_nonzero
        if sparse_inputs:
            assert u_features_nonzero is not None and v_features_nonzero is not None, \
                'u_features_nonzero and v_features_nonzero can not be None when sparse_inputs is True'

        self.self_connections = self_connections

        self.bias = bias
        support = tf.sparse_split(axis=1, num_split=num_support, sp_input=support)

        support_t = tf.sparse_split(axis=1, num_split=num_support, sp_input=support_t)

        if self_connections:
            self.support = support[:-1]
            self.support_transpose = support_t[:-1]
            self.u_self_connections = support[-1]
            self.v_self_connections = support_t[-1]
            self.weights_u = self.weights_u[:-1]
            self.weights_v = self.weights_v[:-1]
            self.weights_u_self_conn = self.weights_u[-1]
            self.weights_v_self_conn = self.weights_v[-1]

        else:
            self.support = support
            self.support_transpose = support_t
            self.u_self_connections = None
            self.v_self_connections = None
            self.weights_u_self_conn = None
            self.weights_v_self_conn = None

        self.support_nnz = []
        self.support_transpose_nnz = []
        for i in range(len(self.support)):
            nnz = tf.reduce_sum(tf.shape(self.support[i].values))
            self.support_nnz.append(nnz)
            self.support_transpose_nnz.append(nnz)

        self.act = act

        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        if self.sparse_inputs:
            x_u = dropout_sparse(inputs[0], 1 - self.dropout, self.u_features_nonzero)
            x_v = dropout_sparse(inputs[1], 1 - self.dropout, self.v_features_nonzero)
        else:
            x_u = tf.nn.dropout(inputs[0], 1 - self.dropout)
            x_v = tf.nn.dropout(inputs[1], 1 - self.dropout)

        supports_u = []
        supports_v = []

        # self-connections with identity matrix as support
        if self.self_connections:
            uw = dot(x_u, self.weights_u_self_conn, sparse=self.sparse_inputs)
            supports_u.append(tf.sparse_tensor_dense_matmul(self.u_self_connections, uw))

            vw = dot(x_v, self.weights_v_self_conn, sparse=self.sparse_inputs)
            supports_v.append(tf.sparse_tensor_dense_matmul(self.v_self_connections, vw))

        wu = 0.
        wv = 0.
        for i in range(len(self.support)):
            wu += self.weights_u[i]
            wv += self.weights_v[i]

            # multiply feature matrices with weights
            tmp_u = dot(x_u, wu, sparse=self.sparse_inputs)

            tmp_v = dot(x_v, wv, sparse=self.sparse_inputs)

            support = self.support[i]
            support_transpose = self.support_transpose[i]

            # then multiply with rating matrices
            supports_u.append(tf.sparse_tensor_dense_matmul(support, tmp_v))
            supports_v.append(tf.sparse_tensor_dense_matmul(support_transpose, tmp_u))

        z_u = tf.add_n(supports_u)
        z_v = tf.add_n(supports_v)

        if self.bias:
            z_u = tf.nn.bias_add(z_u, self.vars['bias_u'])
            z_v = tf.nn.bias_add(z_v, self.vars['bias_v'])

        u_outputs = self.act(z_u)
        v_outputs = self.act(z_v)

        return u_outputs, v_outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v


class BilinearMixture(Layer):
    """
    Decoder model layer for link-prediction with ratings
    To use in combination with bipartite layers.
    """

    def __init__(self, num_classes, u_indices, v_indices, input_dim, num_users, num_items, user_item_bias=False,
                 dropout=0., act=tf.nn.softmax, num_weights=3,
                 diagonal=True, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):

            for i in range(num_weights):
                if diagonal:
                    #  Diagonal weight matrices for each class stored as vectors
                    self.vars['weights_%d' % i] = weight_variable_random_uniform(1, input_dim, name='weights_%d' % i)

                else:
                    self.vars['weights_%d' % i] = orthogonal([input_dim, input_dim], name='weights_%d' % i)

            self.vars['weights_scalars'] = weight_variable_random_uniform(num_weights, num_classes,
                                                                          name='weights_u_scalars')

            if user_item_bias:
                self.vars['user_bias'] = bias_variable_zero([num_users, num_classes], name='user_bias')
                self.vars['item_bias'] = bias_variable_zero([num_items, num_classes], name='item_bias')

        self.user_item_bias = user_item_bias

        if diagonal:
            self._multiply_inputs_weights = tf.multiply
        else:
            self._multiply_inputs_weights = tf.matmul

        self.num_classes = num_classes
        self.num_weights = num_weights
        self.u_indices = u_indices
        self.v_indices = v_indices

        self.dropout = dropout
        self.act = act  # default is softmax (as written in paper)
        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        u_inputs = tf.nn.dropout(inputs[0], 1 - self.dropout)
        v_inputs = tf.nn.dropout(inputs[1], 1 - self.dropout)

        u_inputs = tf.gather(u_inputs, self.u_indices)
        v_inputs = tf.gather(v_inputs, self.v_indices)

        if self.user_item_bias:
            u_bias = tf.gather(self.vars['user_bias'], self.u_indices)
            v_bias = tf.gather(self.vars['item_bias'], self.v_indices)
        else:
            u_bias = None
            v_bias = None

        basis_outputs = []
        for i in range(self.num_weights):

            u_w = self._multiply_inputs_weights(u_inputs, self.vars['weights_%d' % i])
            x = tf.reduce_sum(tf.multiply(u_w, v_inputs), axis=1)

            basis_outputs.append(x)

        # Store outputs in (Nu x Nv) x num_classes tensor and apply activation function
        basis_outputs = tf.stack(basis_outputs, axis=1)

        outputs = tf.matmul(basis_outputs,  self.vars['weights_scalars'], transpose_b=False)

        if self.user_item_bias:
            outputs += u_bias
            outputs += v_bias

        outputs = self.act(outputs)

        return outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])

            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs
