from util import *
import os


def log(x, base): return tf.log(x) / tf.log(base)


def lerp(a, b, t): return t * a + (1 - t) * b


class PGGAN(object):

    def __init__(self, min_resolution, max_resolution, min_channels, max_channels):

        self.min_resolution = np.asanyarray(min_resolution)
        self.max_resolution = np.asanyarray(max_resolution)
        self.min_channels = min_channels
        self.max_channels = max_channels

        def log2(x): return 0 if (x == 1).all() else 1 + log2(x >> 1)

        self.min_depth = log2(self.min_resolution // self.min_resolution)
        self.max_depth = log2(self.max_resolution // self.min_resolution)

    def generator(self, latents,  training, progress,labels=None, name="generator", reuse=None):
        def layer_epilogue(x, use_noise=False):
            if use_noise:
                x = apply_noise(x,randomize_noise=True)
                x = apply_bias(x)
            return x

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def conv_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("conv_block_{}x{}".format(*resolution(depth)), reuse=reuse):
                if depth == self.min_depth:
                    inputs = pixel_norm(inputs)
                    with tf.variable_scope("dense"):
                        inputs = dense(
                            inputs=inputs,
                            out_dim=channels(depth) * resolution(depth).prod(),
                        )
                        inputs = tf.reshape(
                            tensor=inputs,
                            shape=[-1, *resolution(depth), channels(depth)]
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                    with tf.variable_scope("conv"):
                        inputs = layer_epilogue(conv2d(
                            inputs=inputs,
                            out_dim=channels(depth),
                            strides=[1, 1, 1, 1],
                            kernel_size=3,
                        ))
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                else:
                    with tf.variable_scope("upscale_conv"):
                        inputs = layer_epilogue(conv2d_transpose(
                            inputs=inputs,
                            out_dim=channels(depth),
                            kernel_size=3,
                        ))
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                    with tf.variable_scope("conv"):
                        inputs = layer_epilogue(conv2d(
                            inputs=inputs,
                            out_dim=channels(depth),
                            strides=[1, 1, 1, 1],
                            kernel_size=3,
                        ))
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                return inputs

        def color_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{}x{}".format(*resolution(depth)), reuse=reuse):
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        out_dim=3,
                        kernel_size=1,
                        strides=[1, 1, 1, 1],
                        variance_scale=1,
                    )
                return inputs

        def grow(feature_maps, depth):

            def high_resolution_images():
                return grow(conv_block(feature_maps, depth), depth + 1)

            def middle_resolution_images():
                return upscale2d(
                    inputs=color_block(conv_block(feature_maps, depth), depth),
                    factors=resolution(self.max_depth) // resolution(depth)
                )

            def low_resolution_images():
                return upscale2d(
                    inputs=color_block(feature_maps, depth - 1),
                    factors=resolution(self.max_depth) // resolution(depth - 1)
                )

            if depth == self.min_depth:
                images = tf.cond(
                    pred=tf.greater(growing_depth, depth),
                    true_fn=high_resolution_images,
                    false_fn=middle_resolution_images
                )
            elif depth == self.max_depth:
                images = tf.cond(
                    pred=tf.greater(growing_depth, depth),
                    true_fn=middle_resolution_images,
                    false_fn=lambda: lerp(
                        a=low_resolution_images(),
                        b=middle_resolution_images(),
                        t=depth - growing_depth
                    )
                )
            else:
                images = tf.cond(
                    pred=tf.greater(growing_depth, depth),
                    true_fn=high_resolution_images,
                    false_fn=lambda: lerp(
                        a=low_resolution_images(),
                        b=middle_resolution_images(),
                        t=depth - growing_depth
                    )
                )
            return images

        with tf.variable_scope(name, reuse=reuse):
            growing_depth = log((1.5**self.min_depth) + progress * ((1.5**(self.max_depth + 1)) - (1.5**self.min_depth)), 1.5)
            output = grow(tf.concat([latents, labels], axis=1) if labels else latents, self.min_depth)

            return output

    def discriminator(self, images, training,  progress, labels=None, name="discriminator",
                      reuse=tf.AUTO_REUSE):

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def conv_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("conv_block_{}x{}".format(*resolution(depth)), reuse=reuse):
                if depth == self.min_depth:
                    inputs = minibatch_stddev_layer(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            out_dim=channels(depth),
                            kernel_size=3,
                            strides=[1, 1, 1, 1],
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("dense"):
                        inputs = tf.layers.flatten(inputs)
                        inputs = dense(
                            inputs=inputs,
                            out_dim=channels(depth - 1),
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                    if labels is not None:
                        with tf.variable_scope("adversarial_logits"):
                            inputs = dense(
                                inputs=inputs,
                                out_dim=1+labels.shape[1],
                                variance_scale=1,
                                scale_weight=True
                            )
                            adversarial_logits=tf.identity(inputs[:, :1], name='scores_out')
                            classification_logits=tf.identity(inputs[:, 1:], name='labels_out')
                        return adversarial_logits, classification_logits
                    else:
                        with tf.variable_scope("logits"):
                            logits = dense(
                                inputs=inputs,
                                out_dim=1,
                                variance_scale=1,
                                scale_weight=True
                            )
                        return logits

                else:
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            out_dim=channels(depth),
                            kernel_size=3,
                            strides=[1, 1, 1, 1],
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("conv_downscale"):
                        inputs = conv2d(
                            inputs=inputs,
                            out_dim=channels(depth - 1),
                            kernel_size=3,
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                return inputs

        def color_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{}x{}".format(*resolution(depth)), reuse=reuse):
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        out_dim=channels(depth),
                        kernel_size=1,
                        strides=[1, 1, 1, 1],
                    )
                    inputs = tf.nn.leaky_relu(inputs)
                return inputs

        def grow(images, depth):

            def high_resolution_feature_maps():
                return conv_block(grow(images, depth + 1), depth)

            def middle_resolution_feature_maps():
                return conv_block(color_block(downscale2d(
                    inputs=images,
                    factors=resolution(self.max_depth) // resolution(depth)
                ), depth), depth)

            def low_resolution_feature_maps():
                return color_block(downscale2d(
                    inputs=images,
                    factors=resolution(self.max_depth) // resolution(depth - 1)
                ), depth - 1)

            if depth == self.min_depth:
                feature_maps = tf.cond(
                    pred=tf.greater(growing_depth, depth),
                    true_fn=high_resolution_feature_maps,
                    false_fn=middle_resolution_feature_maps
                )
            elif depth == self.max_depth:
                feature_maps = tf.cond(
                    pred=tf.greater(growing_depth, depth),
                    true_fn=middle_resolution_feature_maps,
                    false_fn=lambda: lerp(
                        a=low_resolution_feature_maps(),
                        b=middle_resolution_feature_maps(),
                        t=depth - growing_depth
                    )
                )
            else:
                feature_maps = tf.cond(
                    pred=tf.greater(growing_depth, depth),
                    true_fn=high_resolution_feature_maps,
                    false_fn=lambda: lerp(
                        a=low_resolution_feature_maps(),
                        b=middle_resolution_feature_maps(),
                        t=depth - growing_depth
                    )
                )
            return feature_maps

        with tf.variable_scope(name, reuse=reuse):
            growing_depth = log((1.5**self.min_depth) + progress * ((1.5**(self.max_depth + 1)) - (1.5**self.min_depth)), 1.5)
            return grow(images, self.min_depth)
            


class Model(object):
    def __init__(self, discriminator, generator, real_input_fn, fake_input_fn, growing_depth,
                 hyper_params, batchsize, image_shape, name="gan", reuse=None):

        with tf.variable_scope(name, reuse=reuse):
            # =========================================================================================
            # parameters
            self.name = name
            self.batchsize=batchsize
            self.training = tf.placeholder(dtype=tf.bool, shape=[])
            self.total_steps = tf.placeholder(dtype=tf.int32, shape=[])
            self.global_step = tf.Variable(initial_value=0, trainable=False)
            self.progress = tf.cast(self.global_step / self.total_steps, tf.float32)
            self.image_shape = image_shape
            # =========================================================================================
            # input_fn for real data and fake data
            self.real_images, self.real_labels = real_input_fn()
            self.fake_latents, self.fake_labels = fake_input_fn()

            self.real_images = tf.reshape(self.real_images, [batchsize, *self.real_images.shape[1:]])
            self.real_labels = tf.reshape(self.real_labels, [batchsize, *self.real_labels.shape[1:]])

            self.real_labels_ = tf.one_hot(self.real_labels, depth=hyper_params.label_count, on_value=1.0,
                                           off_value=0.0, axis=None, dtype=tf.float32, name='one_hot')
            # =========================================================================================
            # generated fake data
            self.fake_images = generator(
                latents=self.fake_latents,
                training=self.training,
                progress=self.progress,
                name="generator"
            )


            # =========================================================================================
            # logits for real data and fake data
            self.real_logits = discriminator(
                images=self.real_images,
                training=self.training,
                progress=self.progress,
                name="discriminator"
            )
            self.fake_logits = discriminator(
                images=self.fake_images,
                training=self.training,
                progress=self.progress,
                name="discriminator",
                reuse=True
            )
            # ========================================================================#
            # loss functions from
            # [Which Training Methods for GANs do actually Converge?]
            # (https://arxiv.org/pdf/1801.04406.pdf)
            discriminator_loss = self.fake_logits
            discriminator_loss += -self.real_logits
            self.wasserstein_distance = tf.reduce_mean(discriminator_loss)
            generator_loss = -self.fake_logits

            if hyper_params.r1_gamma > 0:
                real_loss = tf.reduce_sum(self.real_logits)
                real_grads = tf.gradients(real_loss, [self.real_images])[0]
                r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
                discriminator_loss += 0.5 * self.hyper_params.r1_gamma * r1_penalty
            # zero-centerd gradient penalty
            if hyper_params.r2_gamma > 0:
                fake_loss = tf.reduce_sum(self.fake_logits)
                fake_grads = tf.gradients(fake_loss, [self.fake_images])[0]
                r2_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1, 2, 3])
                discriminator_loss += 0.5 * self.hyper_params.r2_gamma * r2_penalty
            # auxiliary classification loss
            if hyper_params.discriminator_classification_loss_weight>0:
                classify_real_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.real_labels_, logits=real_classification_logits)
                classify_fake_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.real_labels_, logits=fake_classification_logits)
                discriminator_loss += (classify_real_losses+classify_fake_losses) * hyper_params.discriminator_classification_loss_weight
                generator_loss += classify_fake_losses*hyper_params.discriminator_classification_loss_weight
            # -----------------------------------------------------------------------
            # variables for discriminator and generator
            discriminator_variables = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/discriminator".format(name)
            )
            generator_variables = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/generator".format(name)
            )
            # ========================================================================#
            # optimizer for discriminator and generator
            self.discriminator_optimizer = tf.train.AdamOptimizer(
                learning_rate=hyper_params.discriminator_learning_rate,
                beta1=hyper_params.discriminator_beta1,
                beta2=hyper_params.discriminator_beta2
            )
            self.generator_optimizer = tf.train.AdamOptimizer(
                learning_rate=hyper_params.generator_learning_rate,
                beta1=hyper_params.generator_beta1,
                beta2=hyper_params.generator_beta2
            )
            # =========================================================================#
            # 生成生成图片与真实图片之间的插值
            if hyper_params.wgan_gamma > 0:
                alpha = tf.random_uniform(shape=[batchsize, 1, 1, 1], minval=0., maxval=1.)
                interpolates = self.real_images + alpha * (self.fake_images - self.real_images)
                interpolates_logits = discriminator(
                    images=interpolates,
                    training=self.training,
                    progress=self.progress,
                    name="discriminator",
                    reuse=True
                )

                wgan_target = 1.0
                mixed_loss = tf.reduce_sum(interpolates_logits)
                mixed_grads = fp32(tf.gradients(mixed_loss, [interpolates])[0])
                mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1, 2, 3]))
                gradient_penalty = tf.square(mixed_norms - wgan_target)
                discriminator_loss += hyper_params.wgan_gamma * gradient_penalty

            with tf.name_scope('EpsilonPenalty'):
                epsilon_penalty = tf.square(self.real_logits)
                discriminator_loss += epsilon_penalty * hyper_params.wgan_epsilon
            # ========================================================================#
            # training op for generator and discriminator
            self.discriminator_loss=tf.reduce_mean(discriminator_loss)
            self.generator_loss=tf.reduce_mean(generator_loss)
            self.discriminator_train_op = self.discriminator_optimizer.minimize(
                loss=self.discriminator_loss,
                var_list=discriminator_variables,
            )
            self.generator_train_op = self.generator_optimizer.minimize(
                loss=self.generator_loss,
                var_list=generator_variables,
                global_step=self.global_step
            )
            # ========================================================================#

#            variable_averages = tf.train.ExponentialMovingAverage(0.999,self.global_step)
#            self.G_averages_op = variable_averages.apply(self.generator_variables)
#            self.discriminator_train_op = tf.group([self.discriminator_train_op])
#            self.generator_train_op = tf.group([self.generator_train_op])
            # ========================================================================#
            # utilities
            self.saver = tf.train.Saver()
            tf.summary.scalar('discriminator_loss', self.discriminator_loss)
            tf.summary.scalar('generator_loss', self.generator_loss)
            tf.summary.scalar('wasserstein_distance', self.wasserstein_distance)


    def train(self, total_steps):
        coord = tf.train.Coordinator()
        merged  = tf.summary.merge_all()

        with tf.Session() as session:
            writer = tf.summary.FileWriter('logs/',session.graph)
            session.run(tf.tables_initializer())
            checkpoint = tf.train.latest_checkpoint('pggannet_1.5/')

            print('{}checkpoint', format(checkpoint))

            if checkpoint:
                self.saver.restore(session, checkpoint)
                tf.logging.info("{} restored".format(checkpoint))
            else:
                global_variables = tf.global_variables(scope=self.name)
                session.run(tf.variables_initializer(global_variables))
                tf.logging.info("global variables in {} initialized".format(self.name))
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            while True:

                feed_dict = {
                    self.training: True,
                    self.total_steps: total_steps,
                }
                global_step = session.run(self.global_step)

                session.run(
                    fetches=self.discriminator_train_op,
                    feed_dict=feed_dict
                )
                session.run(
                    fetches=self.generator_train_op,
                    feed_dict=feed_dict
                )

                if global_step % 1000 == 0:
                    discriminator_loss, generator_loss, wasserstein_distance = session.run(
                        fetches=[self.discriminator_loss, self.generator_loss, self.wasserstein_distance],
                        feed_dict=feed_dict
                    )

                    print(
                        "global_step: {}, discriminator_loss: {:.2f}, generator_loss: {:.2f}, Wasserstein Distance: {:.2f}".format(
                            global_step, np.sum(discriminator_loss), np.sum(generator_loss),
                            np.sum(wasserstein_distance)
                        ))
                    summary = session.run(
                        fetches=merged,
                        feed_dict=feed_dict
                     )
                    writer.add_summary(
                        summary=summary,
                        global_step=global_step
                     )

                    fake_images = session.run(fetches=self.fake_images, feed_dict=feed_dict)
                    fig = plot(fake_images,256, int(self.batchsize ** 0.5), int(self.batchsize ** 0.5), 3)
                    plt.savefig('out5/{}.png'.format(str(global_step // 1000).zfill(4)))

                    plt.close(fig)
                if global_step % 10000 == 0 or (global_step>total_steps//2 and global_step%1000==0):
                    checkpoint = self.saver.save(
                        sess=session,
                        save_path=os.path.join('pggannet_1.5', "face2.ckpt"),
                        global_step=global_step
                )

                if global_step == total_steps:
                    break
            coord.request_stop()

    def Generate_image(self, total_steps, num):
        coord = tf.train.Coordinator()
        with tf.Session() as session:
            session.run(tf.tables_initializer())
            checkpoint = tf.train.latest_checkpoint('pggannet/')
            print('{}checkpoint', format(checkpoint))
            self.saver.restore(session, checkpoint)
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            j = 0
            while (j < num):
                print('generator batch is %d' % j)
                feed_dict = {
                    self.training: False,
                    self.total_steps: total_steps,
                }

                samples = session.run(fetches=self.fake_images, feed_dict=feed_dict)
                for sample in samples:
                    sample = np.reshape(sample,[-1,*sample.shape])
                    fig = plot(sample,256, 1, 1, 3)
                    plt.savefig('out/{}.png'.format(str(j).zfill(4)))

                    plt.close(fig)
                    j += 1
            coord.request_stop()
