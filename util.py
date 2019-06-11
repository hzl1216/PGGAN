import tensorflow as tf
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Minibatch standard deviation.
def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1],s[2],1])             # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=3)                        # [NCHW]  Append as new fmap.


def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]


def get_weight(shape, variance_scale=2, scale_weight=True):
    stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
    if scale_weight:
        weight = tf.get_variable(
            name="weight",
            shape=shape,
            initializer=tf.truncated_normal_initializer(0.0, 1.0)
        ) * stddev
    else:
        weight = tf.get_variable(
            name="weight",
            shape=shape,
            initializer=tf.truncated_normal_initializer(0.0, stddev)
        )
    return weight


def get_bias(shape):
    bias = tf.get_variable(
        name="bias",
        shape=shape,
        initializer=tf.zeros_initializer()
    )
    return bias


def dense(inputs, out_dim, use_bias=True, variance_scale=2,scale_weight=True):
    weight = get_weight(
        shape=[int(inputs.shape[1]), out_dim],
        variance_scale=variance_scale,
        scale_weight=scale_weight
    )
    inputs = tf.matmul(inputs, weight)
    if use_bias:
        bias = get_bias([inputs.get_shape()[1]])
        inputs = tf.nn.bias_add(inputs, bias)
    return inputs


def conv2d(inputs, out_dim, kernel_size, strides=[1,2,2,1], use_bias=True,variance_scale=2,scale_weight=True,fused_scale='auto'):
    if fused_scale == 'auto':
        fused_scale = inputs.shape[2] >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale and strides[2]%2==0:
        return downscale2d(conv2d(inputs, out_dim, kernel_size, strides=[1,1,1,1]))

    weight = get_weight(
        shape=[kernel_size,kernel_size,int( inputs.get_shape()[-1]), out_dim],
        variance_scale=variance_scale,
        scale_weight=scale_weight
    )
    weight = tf.pad(weight, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    weight = tf.add_n([weight[1:, 1:], weight[:-1, 1:], weight[1:, :-1], weight[:-1, :-1]])

    inputs = tf.nn.conv2d(
        inputs,
        weight,
        strides=strides,
        padding="SAME",
    )
    if use_bias:
        bias = get_bias([out_dim])
        inputs = tf.nn.bias_add(inputs, bias)
    return inputs


def conv2d_transpose(inputs, out_dim, kernel_size, strides=[1,2,2,1], use_bias=True,variance_scale=2,scale_weight=True,fused_scale='auto'):

    if fused_scale == 'auto':
        fused_scale = inputs.shape[2] >= 64

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return conv2d(upscale2d(inputs), out_dim, kernel_size, strides=[1,1,1,1])
    weight = get_weight(
        shape=[kernel_size,kernel_size,  out_dim,int(inputs.get_shape()[-1])],
        variance_scale=variance_scale,
        scale_weight=scale_weight
    )
    weight = tf.pad(weight, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    weight = tf.add_n([weight[1:, 1:], weight[:-1, 1:], weight[1:, :-1], weight[:-1, :-1]])
    output_shape = [inputs.shape[0].value, int(inputs.get_shape()[1]*strides[1]),int(inputs.get_shape()[2]*strides[2]), out_dim]
    inputs = tf.nn.conv2d_transpose(
        inputs,
        weight,
        output_shape=output_shape,
        strides=strides,
    )
    if use_bias:
        bias = get_bias([out_dim])
        inputs = tf.nn.bias_add(inputs, bias)
    return inputs


def upscale2d(inputs, factors=[2, 2]):
    factors = np.asanyarray(factors)
    if (factors == 1).all():
        return inputs
    shape = inputs.shape
    inputs = tf.reshape(inputs, [-1, shape[1], shape[2], 1, shape[3], 1])
    inputs = tf.tile(inputs, [1, 1, 1, factors[0], 1, factors[1]])
    inputs = tf.reshape(inputs, [-1, shape[1]*factors[0], shape[2] * factors[1], shape[3]])
    return inputs


def downscale2d(inputs, factors=[2, 2]):
    # NOTE: requires tf_config["graph_options.place_pruned_graph"] = True
    factors = np.asanyarray(factors)
    if (factors == 1).all():
        return inputs
    inputs = tf.nn.avg_pool(
        value=inputs,
        ksize=[1, *factors,1],
        strides=[1, *factors,1],
        padding="SAME",
    )
    return inputs


def pixel_norm(inputs,norm =0, epsilon=1e-8):
    if norm == 0:
        inputs *= tf.rsqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + epsilon)
    else:
        with tf.variable_scope('InstanceNorm'):
            orig_dtype = inputs.dtype
            inputs = tf.cast(inputs, tf.float32)
            inputs -= tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
            epsilon = tf.constant(epsilon, dtype=inputs.dtype, name='epsilon')
            inputs *= tf.rsqrt(tf.reduce_mean(tf.square(inputs), axis=[1, 2], keepdims=True) + epsilon)
            inputs = tf.cast(inputs, orig_dtype)
    return inputs


def plot(samples,outsize ,height, weight,image_channel=3 ):  # 保存图片时使用的plot函数
    
    samples = np.clip(samples,-1.0,1.0)
    samples = samples/2 + 0.5
    fig = plt.figure(figsize=(height*outsize,weight*outsize),dpi=1)

    gs = gridspec.GridSpec(height, weight)  # 调整子图的位置

    gs.update(wspace=0.05, hspace=0.05)  # 置子图间的间距
    for i, sample in enumerate(samples):  # 依次将16张子图填充进需要保存的图像
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if image_channel == 1:
            sample = sample.reshape(outsize, outsize)
            plt.imshow(sample, cmap='Greys_r')
        else:
            plt.imshow(sample)

    return fig


def conv_cond_concat(value1, value2, concat_dim, name='concat'):  # 矩阵相加
    value1_shapes = value1.get_shape().as_list()
    value2_shapes = value2.get_shape().as_list()	
    value2 = tf.cast(value2, tf.float32)
    value1 = tf.cast(value1, tf.float32)
    with tf.variable_scope(name):
        if concat_dim == 1:
            return tf.concat([value1, value2], 1, name)
        if concat_dim == 3:
            return tf.concat([value1, value2 * tf.ones(value1_shapes[0:3] + value2_shapes[3:])], 3, name=name)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def apply_noise(x, noise_var=None, randomize_noise=True):
    with tf.variable_scope('Noise'):
        noise = tf.random_normal([x.shape[0].value, x.shape[1].value, x.shape[2].value,1], dtype=x.dtype)
        weight = tf.get_variable('weight', shape=[x.shape[3].value], initializer=tf.initializers.zeros())
        return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, 1, 1, -1])


# Apply bias to the given activation tensor.
def apply_bias(x, lrmul=1):
    b = tf.get_variable('fixbias', shape=[x.shape[3].value], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, 1, 1, -1])
