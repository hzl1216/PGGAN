import tensorflow as tf
import os


def celeba_input_fn(filenames, batch_size, num_epochs, shuffle, image_size):
    def parse_example(example):
        features = tf.parse_single_example(
            serialized=example,
            features={
                "img": tf.FixedLenFeature([], dtype=tf.string),
                "label": tf.FixedLenFeature([], dtype=tf.int64),
            }
        )

        image = tf.decode_raw(features['img'], tf.uint8)
        image = tf.reshape(image,image_size) 
        image = tf.cast(image, tf.float32) * (1. / 255) *2 -1
        label = features["label"]
        label = tf.cast(label, tf.int32)

        return image, label

    dataset = tf.data.TFRecordDataset(filenames)
    if shuffle:
        dataset = dataset.shuffle(
            sum([len(list(tf.python_io.tf_record_iterator(filename)))
                for filename in filenames
            ]),
            reshuffle_each_iteration=True
        )
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.map(
        map_func=parse_example,
        num_parallel_calls=os.cpu_count(),
    )
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
