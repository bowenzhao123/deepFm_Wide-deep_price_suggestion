# DeepFm part is contributed by wnzhang
# https://github.com/wnzhang/deep-ctr/blob/master/python/fm.py
import tensorflow as tf

class WDNN:
	def __init__(base_columns,cross_columns,deep_columns):
		self.base_columns = base_columns
		self.cross_columns = cross_columns
		self.deep_columns = deep_columns

    def model(self):
	    train_config = tf.train.AdamOptimizer(
		    learning_rate=0.001,
            beta1=.9,
            lbeta2=.99,
            epsilon=1e-5,
            name = "Adam")
	    wd = tf.estimator.DNNLinearCombinedRegressor(
		    model_dir='/census_model',
		    linear_feature_columns=self.base_columns + self.crossed_columns,
            dnn_feature_columns = self.deep_columns,
            dnn_optimizer = train_fig,
            dn_dropout = .3,
            dnn_hidden_units=[100, 50, 100])
	    return wd

class DeepFM:
	class DeepFM:
    def __init__(self, feature_size, factor_size, field_size,
                 deep_layers=[400, 400], 
                 deep_layers_activation=tf.nn.relu):
        self.feature_size = feature_size
        self.factor_size  = factor_size
        self.field_size   = field_size
        self.deep_layers  = deep_layers
        self.deep_layers_activation = deep_layers_activation
    
    def first_order_part(self, sparse_id, sparse_value):
        with tf.variable_scope("first-order"):
            W    = tf.get_variable("weight",(self.feature_size, 1), \
                    initializer=tf.random_normal_initializer(0.0, 0.01))
            y_first_order = tf.nn.embedding_lookup(W, sparse_id) 
            y_first_order = tf.reduce_sum(tf.multiply(y_first_order, sparse_value), 1)  

            return y_first_order
    
    def second_order_part(self, sparse_id, sparse_value):
        with tf.variable_scope("second-order"):
            V = tf.get_variable("weight",(self.feature_size, self.factor_size), \
                    initializer=tf.random_normal_initializer(0.0, 0.01))
            self.embeddings = tf.nn.embedding_lookup(V, sparse_id)
            self.embeddings = tf.multiply(self.embeddings, sparse_value) 
            sum_squared_part = tf.square(tf.reduce_sum(self.embeddings, 1)) 
            squared_sum_part = tf.reduce_sum(tf.square(self.embeddings), 1) 

            y_second_order   = 0.5 * tf.subtract(sum_squared_part, squared_sum_part)
            return y_second_order

    def deep_part(self):
        with tf.variable_scope("deep-part"):
            y_deep = tf.reshape(self.embeddings, shape=[-1, \
                            self.field_size * self.factor_size]) # None * (F*K)
            for i in range(0, len(self.deep_layers)):
                y_deep = tf.contrib.layers.fully_connected(y_deep, self.deep_layers[i], \
                            activation_fn=self.deep_layers_activation, scope = 'fc%d' % i)
            
            return y_deep

        
    def forward(self, sparse_id, sparse_value):
        sparse_value   = tf.expand_dims(sparse_value, -1)

        y_first_order  = self.first_order_part(sparse_id, sparse_value)
        y_second_order = self.second_order_part(sparse_id, sparse_value)
        y_deep         = self.deep_part()

        with tf.variable_scope("deep-fm"):
            deep_out    = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
            deep_out    = tf.contrib.layers.fully_connected(deep_out, 1, \
                activation_fn=tf.nn.sigmoid, scope = 'deepfm_out')

            return tf.reduce_sum(deep_out, axis=1)
            
