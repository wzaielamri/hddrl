from tensorflow.keras import initializers

class GlorotUniformScaled(initializers.VarianceScaling):
    """ Glorot Uniform Initializer 
        reimplemented as `tf.keras.initializers.glorot_uniform`
        but (as helpful for DRL policy networks) has an additional scale 
        parameter.

  Draws samples from a uniform distribution within `[-limit, limit]`, where
  `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input units
  in the weight tensor and `fan_out` is the number of output units).
  """

    def __init__(self, scale=1., seed=None):
        super(GlorotUniformScaled, self).__init__(
            scale=scale,
            mode='fan_avg',
            distribution='uniform',
            seed=seed)

    def get_config(self):
        return {'seed': self.seed}
