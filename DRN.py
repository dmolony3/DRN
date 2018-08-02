import tensorflow as tf

class DRN():
    def __init__(self, image, image_dims, batch_size, num_classes, is_training, network):
        self.image = image
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.is_training = is_training
		self.image_dims = image_dims
        if network == 'DRN18':
            self.build_DRN18()
        elif network == 'DRN26':
            self.build_DRN26()

    def batch_norm(self, X, is_training, decay=0.999):
        """Batch normalization"""

        # the offset (beta) should always be used, but the scale is not necessary for activation function like relu
        # beta and scale will have the same shape as the bias i.e. the no. of features
        scale = tf.Variable(tf.ones([X.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([X.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([X.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([X.get_shape()[-1]]), trainable=False)
        epsilon = 1e-6
        if is_training == True:
            batch_mean, batch_var = tf.nn.moments(X, [0, 1, 2])

            # use exponentially moving average in order to calculate the population mean for inference
            # here we update pop_mean with pop_mean*decay+batch_mean*(1-decay)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

            # learn pop_mean and pop_var here
            with tf.control_dependencies([train_mean, train_var]):
                X = tf.nn.batch_normalization(X, batch_mean, batch_var, beta, scale, epsilon)
        else:
            X = tf.nn.batch_normalization(X, pop_mean, pop_var, beta, scale, epsilon)

        return X

    def conv_repeat(self, X, strides, dilation, kernel, residual, name):
        """Performs convolution with residual block"""

        if residual == 1:
            shortcut = X
            if strides[-1] != 1:
                # strided convolution and double channel depth
                shortcut = self.conv_2d(shortcut, strides[-1], [1, 1, kernel[-1]], name + '_shortcut')
                # shortcut = self.conv_2d(shortcut, strides[-1], [kernel], name+'_shortcut')  #  CHECK IF PAPER USED 1x1 CONVOLUTION
                # need to increase number of channels in shortcut to match number of channels in output
            elif shortcut.get_shape()[3] != kernel[-1]:
                shortcut = self.conv_2d(shortcut, strides[0], kernel, name + '_shortcut')

        for i in range(len(strides)):
            if dilation:
                X = self.atrous_conv_2d(X, strides, dilation, kernel, name=name + '_' + str(i))
                X = self.batch_norm(X, is_training=self.is_training)  # perform batch normalization
                X = tf.nn.relu(X)  # perfor relu activation
            else:
                X = self.conv_2d(X, strides[i], kernel, name=name + '_' + str(i))
                X = self.batch_norm(X, is_training=self.is_training)  # perform batch normalization
                X = tf.nn.relu(X)  # perform relu activation
                if residual == 1 and i == len(strides) - 1:
                    # add shortcut on last operation
                    X = tf.add(shortcut, X)
                    X = tf.nn.relu(X)
                else:
                    X = tf.nn.relu(X)

        return X

    def atrous_conv_2d(self, X, strides, dilation, kernel, name):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable(shape=[kernel[0], kernel[1], X.shape[3], kernel[2]], dtype=tf.float32,
                                name=name + '_weights')
            b = tf.get_variable(shape=[kernel[2]], dtype=tf.float32, name=name + '_bias')

        X = tf.nn.atrous_conv2d(X, W, dilation, padding='SAME')
        X = X + b

        return X

    def conv_2d(self, X, strides, kernel, name):
        input_channels = X.get_shape()[3]
        with tf.variable_scope(name) as scope:
            # W = tf.get_variable(shape=[kernel[0], kernel[1], input_channels, kernel[2]], dtype=tf.float32, name=name+'_weights')
            W = tf.get_variable(shape=[kernel[0], kernel[1], X.shape[3], kernel[2]], dtype=tf.float32,
                                name=name + '_weights')
            b = tf.get_variable(shape=[kernel[2]], dtype=tf.float32, name=name + '_bias')

        X = tf.nn.conv2d(X, W, [1, strides, strides, 1], padding='SAME', name=None)
        X = X + b

        return X

    def build_DRN18(self):
        X = self.image
        X.set_shape([None, self.image_dims[0], self.image_dims[1], self.image_dims[2]])
        # print('Input shape is {}'.format(tf.shape(X)))
        print('Input shape is {}'.format(X.get_shape()))

        kernel = [7, 7, 64]
        strides = 2
        dilation = []
        X = self.conv_2d(X, strides, kernel, 'layer2')
        print('Layer2 shape is {}'.format(X.get_shape()))

        residual = 1
        X = tf.nn.max_pool(X, [1, 1, 1, 1], [1, 2, 2, 1], padding='SAME')
        kernel = [3, 3, 64]
        strides = [1, 1]
        X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer3_1')
        strides = [1, 2]
        X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer3_2')
        print('Layer3 shape is {}'.format(X.get_shape()))

        kernel = [3, 3, 128]
        strides = [1, 1]
        X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer4_1')
        strides = [1, 1]
        X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer4_2')
        print('Layer4 shape is {}'.format(X.get_shape()))

        kernel = [3, 3, 256]
        strides = [1, 1, 1, 1]
        dilation = 2
        X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer5')
        print('Layer5 shape is {}'.format(X.get_shape()))

        kernel = [3, 3, 512]
        strides = [1, 1, 1, 1]
        dilation = 4
        X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer6')
        print('Layer6 shape is {}'.format(X.get_shape()))

        # 1x1 convolution to squash output to number of classes
        kernel = [1, 1, self.num_classes]
        strides = [1]
        dilation = []
        X = self.conv_repeat(X, strides, dilation, kernel, residual, 'output')
        print('Output shape is {}'.format(X.get_shape()))

        self.prob = X
        self.pred = tf.argmax(X, 3)

    def build_DRN26(self):
    	X = self.image
        X.set_shape([None, self.image_dims[0], self.image_dims[1], self.image_dims[2]])
    	#print('Input shape is {}'.format(tf.shape(X)))
    	print('Input shape is {}'.format(X.get_shape()))
    
    	kernel = [7, 7, 16]
    	strides = 1
    	self.conv_2d(X, strides, kernel, 'layer1_1')
    	residual = 1
    	kernel = [3, 3, 16]
    	strides = [1, 2]
    	dilation = []
    	X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer1_2')
    	print('Layer1 shape is {}'.format(X.get_shape())) 

    	residual = 1
    	kernel = [3, 3, 32]
    	strides = [1, 2]
    	dilation = []
    	X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer2')
    	print('Layer2 shape is {}'.format(X.get_shape()))
    
    	residual = 1
    	kernel = [3, 3, 64]
    	strides = [1, 1]
    	X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer3_1')
    	strides = [1, 2]
    	X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer3_2')
    	print('Layer3 shape is {}'.format(X.get_shape()))
    
    	kernel = [3, 3, 128]
    	strides = [1, 1]
    	X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer4_1')
    	strides = [1, 1]
    	X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer4_2')
    	print('Layer4 shape is {}'.format(X.get_shape()))
    
    	kernel = [3, 3, 256]
    	strides = [1, 1, 1, 1]
    	dilation = 2
    	X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer5')
    	print('Layer5 shape is {}'.format(X.get_shape()))
    
    	kernel = [3, 3, 512]
    	strides = [1, 1, 1, 1]
    	dilation = 4
    	X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer6')
    	print('Layer6 shape is {}'.format(X.get_shape()))

    	residual = 0
    	kernel = [3, 3, 512]
    	strides = [1, 1]
    	dilation = 2
    	X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer7')
    	print('Layer7 shape is {}'.format(X.get_shape()))
    
    	kernel = [3, 3, 512]
    	strides = [1, 1]
    	dilation = 1
    	X = self.conv_repeat(X, strides, dilation, kernel, residual, 'layer8')
    	print('Layer8 shape is {}'.format(X.get_shape()))
    
    	# 1x1 convolution to squash output to number of classes
    	kernel = [1, 1, self.num_classes]
    	strides = [1]
    	dilation = []
    	X = self.conv_repeat(X, strides, dilation, kernel, residual, 'output')
    	print('Output shape is {}'.format(X.get_shape()))

    	self.prob = X
    	self.pred = tf.argmax(X, 3) 

