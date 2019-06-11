from param import Param
import argparse
import functools
from pggan import *
from dataset import *
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="data")
## need a images_256.tfrecorde under of tthe dir data
parser.add_argument('--filenames', type=str, default=[os.path.join('data','images_256.tfrecorde')])
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--train", action="store_true")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--num", type=int, default=1000)
parser.add_argument("--restore", type=bool, default=0)
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)
attr_counts = 1
z_dim = 512
resolution = 256
with tf.Graph().as_default():

    tf.set_random_seed(1000)

    pggan = PGGAN(
        min_resolution=[4, 4],
        max_resolution=[resolution, resolution],
        min_channels=16,
        max_channels=256
    )

    gan = Model(
        discriminator=pggan.discriminator,
        generator=pggan.generator,
        real_input_fn=functools.partial(
            celeba_input_fn,
            filenames=args.filenames,
            batch_size=args.batch_size,
            num_epochs=None,
            shuffle=True,
            image_size=[resolution,resolution,3]
        ),
        fake_input_fn=lambda: (
            tf.random_normal(shape=[args.batch_size,z_dim], dtype=tf.float32),
            tf.one_hot(tf.reshape(tf.multinomial(
                logits=tf.log([[tf.cast(attr_counts, tf.float32) for _ in range(attr_counts)]]),
                num_samples=args.batch_size
            ), [args.batch_size]), attr_counts,on_value=1.0, off_value=0.0),),
        growing_depth=0.0,
        hyper_params=Param(
            discriminator_learning_rate=0.0015,
            discriminator_beta1=0.0,
            discriminator_beta2=0.99,
            generator_learning_rate=0.0015,
            generator_beta1=0.0,
            generator_beta2=0.99,
            r1_gamma=0.0,
            r2_gamma=0.0,
            discriminator_classification_loss_weight=0.0,
            wgan_gamma=10.0,
            wgan_epsilon=0.001,
            label_count=attr_counts,
        ),
        batchsize=args.batch_size,
        image_shape=[resolution, resolution, 3],
    )

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=args.gpu,
            allow_growth=True
        )
    )
    if args.restore==False:
        gan.train(args.total_steps)
    else:
        gan.Generate_image(args.total_steps,args.num)
