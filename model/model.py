import tensorflow as tf

class ResidualLayer(tf.keras.Model):
    def __init__(self):
        super(ResidualLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.ReLU()
    
    def call(self, x, training=False):
        x_skip = x
        x = self.conv(x)
        x = self.batch_norm(x,training=training)
        x = tf.add(x,x_skip)
        x = self.ReLU(x)
        return x
    
class ConvolutionalStem(tf.keras.Model):
    def __init__(self,residual_size=6):
        super(ConvolutionalStem, self).__init__()
        self.initial_cnn_layer = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same')
        self.initial_batch_norm = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.ReLU()
        self.residual_layers = [ResidualLayer() for _ in range(residual_size-1)]
    
    def call(self, x, training=False):
        x = self.initial_cnn_layer(x)
        x = self.initial_batch_norm(x,training=training)
        x = self.ReLU(x)
        for i in range(len(self.residual_layers)):
            x = self.residual_layers[i](x)
        return x

class ValueHead(tf.keras.Model):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.single_filter_conv = tf.keras.layers.Conv2D(filters=1,kernel_size=1,strides=1,padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.ReLU()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(256,activation='relu')
        self.value = tf.keras.layers.Dense(1,activation='tanh')
        
    def call(self, x, training=False):
        x = self.single_filter_conv(x)
        x = self.batch_norm(x,training=training)
        x = self.ReLU(x)
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.value(x)
        return x

class PolicyHead(tf.keras.Model):
    def __init__(self):
        super(PolicyHead, self).__init__()
        self.two_filter_conv = tf.keras.layers.Conv2D(filters=2,kernel_size=1,strides=1,padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.ReLU()
        self.flatten = tf.keras.layers.Flatten()
        self.policy_layer = tf.keras.layers.Dense(8*8*73,activation='softmax')
        
    def call(self, x, training=False):
        x = self.two_filter_conv(x)
        x = self.batch_norm(x,training=training)
        x = self.ReLU(x)
        x = self.flatten(x)
        x = self.policy_layer(x)
        x = tf.reshape(x, shape=(x.shape[0],8,8,73))
        return x

class ChessModel(tf.keras.Model):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.convolutional_stem = ConvolutionalStem()
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()
    
    def call(self, x, training=False):
        x = self.convolutional_stem(x,training=training)
        val = self.value_head(x,training=training)
        policy = self.policy_head(x,training=training)
        return val, policy