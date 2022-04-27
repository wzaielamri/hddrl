import numpy as np

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override

from models.glorot_uniform_scaled_initializer import GlorotUniformScaled

tf1, tf, tfv = try_import_tf()


class FullyConnectedNetwork_GlorotUniformInitializer_LSTM(RecurrentNetwork):
    """ A fully connected Network - same as the provided generic one
        (ray.rllib.models.tf.FullyConnectedNetwork), but using the
        Glorot Uniform initialization (scaled for action output which
        requires a derived scaled initializer).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(FullyConnectedNetwork_GlorotUniformInitializer_LSTM, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens", [])
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")
        free_log_std = model_config.get("free_log_std")
        self.cell_size = model_config.get("lstm_cell_size")

        self.vf_share_layers = vf_share_layers
        # Generate free-floating bias variables for the second half of
        # the outputs.
        if free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2
            self.log_std_var = tf.Variable(
                [0.0] * num_outputs, dtype=tf.float32, name="log_std")
            self.register_variables([self.log_std_var])

        # We are using obs_flat, so take the flattened shape as input.
        # inputs = tf.keras.layers.Input(
        #    shape=(int(np.product(obs_space.shape)), ), name="observations")

        inputs = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")

        # lstm inputs
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size, ), name="c")
        state_in_h_value = tf.keras.layers.Input(
            shape=(self.cell_size, ), name="h_value")
        state_in_c_value = tf.keras.layers.Input(
            shape=(self.cell_size, ), name="c_value")

        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Last hidden layer output (before logits outputs).
        last_layer = inputs

        # The action distribution outputs.
        logits_out = None
        i = 1

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=GlorotUniformScaled(1.0))(last_layer)
            i += 1

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            logits_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=activation,
                kernel_initializer=GlorotUniformScaled(1.0))(last_layer)
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                last_layer = tf.keras.layers.Dense(
                    hiddens[-1],
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=GlorotUniformScaled(1.0))(last_layer)

            # lstm layer:
            last_layer, state_h, state_c = tf.keras.layers.LSTM(
                self.cell_size, kernel_initializer=GlorotUniformScaled(1.0), return_sequences=True, return_state=True, name="lstm")(
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c],
                inputs=last_layer)

            if num_outputs:
                logits_out = tf.keras.layers.Dense(
                    num_outputs,
                    name="fc_out",
                    activation=None,
                    kernel_initializer=GlorotUniformScaled(0.01))(last_layer)
            # Adjust num_outputs to be the number of nodes in the last layer.
            else:
                self.num_outputs = (
                    [int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        # Concat the log std vars to the end of the state-dependent means.
        if free_log_std and logits_out is not None:

            def tiled_log_std(x):
                return tf.tile(
                    tf.expand_dims(self.log_std_var, 0), [tf.shape(x)[0], 1])

            log_std_out = tf.keras.layers.Lambda(tiled_log_std)(inputs)
            logits_out = tf.keras.layers.Concatenate(axis=1)(
                [logits_out, log_std_out])

        last_vf_layer = None
        if not vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            last_vf_layer = inputs

            i = 1
            for size in hiddens:
                last_vf_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_{}".format(i),
                    activation=activation,
                    kernel_initializer=GlorotUniformScaled(1.0))(last_vf_layer)
                i += 1

            last_vf_layer, state_h_value, state_c_value = tf.keras.layers.LSTM(
                self.cell_size, kernel_initializer=GlorotUniformScaled(1.0), return_sequences=True, return_state=True, name="lstm_value")(
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h_value, state_in_c_value], inputs=last_vf_layer,)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=GlorotUniformScaled(0.01))(
                last_vf_layer if last_vf_layer is not None else last_layer)

        if not vf_share_layers:
            self.base_model = tf.keras.Model(
                inputs=[inputs, seq_in, state_in_h, state_in_c,
                        state_in_h_value, state_in_c_value],
                outputs=[(logits_out
                          if logits_out is not None else last_layer), value_out, state_h, state_c, state_h_value, state_c_value])
        else:
            self.base_model = tf.keras.Model(
                inputs=[inputs, seq_in, state_in_h, state_in_c, ],
                outputs=[(logits_out
                          if logits_out is not None else last_layer), value_out, state_h, state_c])

        self.register_variables(self.base_model.variables)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):

        if not self.vf_share_layers:
            model_out, self._value_out, h, c, h_value, c_value = self.base_model([inputs, seq_lens] +
                                                                                 state)
            return model_out, [h, c, h_value, c_value]
        else:
            model_out, self._value_out, h, c, = self.base_model([inputs, seq_lens] +
                                                                state)
            return model_out, [h, c, ]

    @override(ModelV2)
    def get_initial_state(self):
        if not self.vf_share_layers:
            return [
                np.zeros(self.cell_size, np.float32),
                np.zeros(self.cell_size, np.float32),
                np.zeros(self.cell_size, np.float32),
                np.zeros(self.cell_size, np.float32),
            ]
        else:
            return [
                np.zeros(self.cell_size, np.float32),
                np.zeros(self.cell_size, np.float32),
            ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
