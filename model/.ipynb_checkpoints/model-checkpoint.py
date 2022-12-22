import tensorflow as tf

from model.mcts import MctsNode

class ResidualLayer(tf.keras.Model):
    def __init__(self):
        super(ResidualLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',kernel_regularizer='l2')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.ReLU()
        
    @tf.function
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
        self.initial_cnn_layer = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',kernel_regularizer='l2',input_shape=(None,8,8,None))
        self.initial_batch_norm = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.ReLU()
        self.residual_layers = [ResidualLayer() for _ in range(residual_size-1)]
        
    @tf.function
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
        self.single_filter_conv = tf.keras.layers.Conv2D(filters=1,kernel_size=1,strides=1,padding='same',kernel_regularizer='l2')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.ReLU()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(256,activation='relu')
        self.value = tf.keras.layers.Dense(1,activation='tanh')
        
    @tf.function
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
        self.two_filter_conv = tf.keras.layers.Conv2D(filters=2,kernel_size=1,strides=1,padding='same',kernel_regularizer='l2')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.ReLU()
        self.policy_layer = tf.keras.layers.Conv2D(filters=73,kernel_size=1,strides=1,padding='same',kernel_regularizer='l2')
        self.flatten = tf.keras.layers.Flatten()
        self.softmax = tf.keras.activations.softmax
    
    @tf.function
    def call(self, x, training=False):
        x = self.two_filter_conv(x)
        x = self.batch_norm(x,training=training)
        x = self.ReLU(x)
        x = self.policy_layer(x)
        batch_size = x.shape[0]
        if batch_size > 1:
            x = tf.split(x,batch_size)
            for b in range(batch_size):
                x[b] = self.flatten(x[b])
                x[b] = self.softmax(x[b])
                x[b] = tf.reshape(x[b],shape=(1,8,8,73))
            x = tf.concat([x],axis=0)
        else:
            x = self.flatten(x)
            x = self.softmax(x)
            x = tf.reshape(x,shape=(1,8,8,73))
        return x

class ChessModel(tf.keras.Model):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.convolutional_stem = ConvolutionalStem()
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()
        
    @tf.function
    def call(self, x, training=False):
        x = self.convolutional_stem(x,training=training)
        val = self.value_head(x,training=training)
        policy = self.policy_head(x,training=training)
        return val, policy
    
    def save(self,path):
        self.save_weights(path)
    
    def load(self, path):
        self.built = True
        self.load_weights(path)
        
class LossConverter(tf.keras.losses.Loss):
    '''
    ADD
    '''
    
    def __call__(self, pred_values : tf.Tensor, true_values : tf.Tensor, policy : tf.Tensor, nodes : list) -> tf.Tensor:
        '''
        ADD
        '''
        
        z = tf.constant([[node.value for node in nodes]],dtype=tf.float32)
        print(z)
        pi = tf.constant([[child.value for child in node.children] for node in nodes],dtype=tf.float32)
        print(pi)
        
        return tf.reduce_mean(tf.math.square(pred_values, z)-pi,axis=-1)