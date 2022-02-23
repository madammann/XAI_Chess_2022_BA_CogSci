import tensorflow as tf

class ConvolutionalStem(tf.keras.Model):
    def __init__(self):
        super(ConvolutionalStem, self).__init__()
    
    def call(self, x):
        pass

class ValueHead(tf.keras.Model):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.embed = tf.keras.layers.Dense(10)
        self.output = tf.keras.layers.Dense(1,activation='sigmoid')
        
    def call(self, x):
        x = self.embed(x)
        x = self.output(x)
        return x

class PolicyHead(tf.keras.Model):
    def __init__(self):
        super(PolicyHead, self).__init__()
        
    def call(self, x):
        pass

class ChessModel(tf.keras.Model):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.convolutional_stem = ConvolutionalStem()
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()
    
    def call(self, x):
        x = self.convolutional_stem(x)
        val = self.value_head(x)
        policy = self.policy_head(x)
        return val, policy