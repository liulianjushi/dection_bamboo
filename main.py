import math
from collections import Counter

import random
import tensorflow as tf
from imutils import paths

from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE, EPOCHS, save_model_dir, save_every_n_step
from models import efficientnet

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[1], 'GPU')
if gpus:
    try:
        # 设置GPU显存占用为按需分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 异常处理
        print(e)

classes = {'黄皮绿筋竹': 0,
           '斑竹': 1,
           '人面竹': 2,
           '紫竹': 3,
           '黄槽斑竹': 4,
           '菲白竹': 5,
           '黄槽毛竹': 6,
           '花毛竹': 7,
           '绿槽毛竹': 8,
           '菲黄竹': 9,
           '金镶玉竹': 10,
           '龟甲竹': 11}
imagePaths = list(paths.list_images("/home/zhaoliu/duanhuiru/12种竹种/")) + list(
    paths.list_images("/home/zhaoliu/workspace/12种竹种_增强/"))
random.shuffle(imagePaths)
train_images = imagePaths[:int(0.7 * len(imagePaths))]
valid_images = imagePaths[int(0.7 * len(imagePaths)):]

train_labels = [classes.get(label.split('/')[-2]) for label in train_images]
valid_labels = [classes.get(label.split('/')[-2]) for label in valid_images]

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))


def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize(image_decoded, [IMAGE_WIDTH, IMAGE_HEIGHT]) / 255.0
    return image_resized, label


train_dataset = train_dataset.map(map_func=_decode_and_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(100)
train_dataset = train_dataset.repeat(EPOCHS)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = valid_dataset.map(map_func=_decode_and_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(BATCH_SIZE)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)


def print_model_summary(network):
    network.build(input_shape=(None, 224, 224, 3))
    network.summary()


# create model
model = efficientnet.efficient_net_b0()
print_model_summary(network=model)
print(f"length of imagePaths:{len(imagePaths)}")
print(f"num of train dataset: {dict(Counter(train_labels))}")
print(f"num of valid dataset: {dict(Counter(valid_labels))}")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.RMSprop()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
ACC = 0


@tf.function
def train_step(image_batch, label_batch):
    with tf.GradientTape() as tape:
        predictions = model(image_batch, training=True)
        loss = loss_object(y_true=label_batch, y_pred=predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    train_loss.update_state(values=loss)
    train_accuracy.update_state(y_true=label_batch, y_pred=predictions)


@tf.function
def valid_step(image_batch, label_batch):
    predictions = model(image_batch, training=False)
    v_loss = loss_object(label_batch, predictions)

    valid_loss.update_state(values=v_loss)
    valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)


train_summary_writer = tf.summary.create_file_writer('./tensorboard/train/')  # 实例化记录器
valid_summary_writer = tf.summary.create_file_writer('./tensorboard/valid/')  # 实例化记录器

# start training

for step, (images, labels) in enumerate(train_dataset):
    train_step(images, labels)
    with train_summary_writer.as_default():  # 指定记录器
        tf.summary.scalar("train_loss", train_loss.result(), step=step)  # 将当前损失函数的值写入记录器
        tf.summary.scalar("train_accuracy", train_accuracy.result(), step=step)
    if (step + 1) % save_every_n_step == 0:
        for img, lab in valid_dataset:
            valid_step(img, lab)
        with valid_summary_writer.as_default():  # 指定记录器
            tf.summary.scalar("valid_loss", valid_loss.result(), step=step)  # 将当前损失函数的值写入记录器
            tf.summary.scalar("valid_accuracy", valid_accuracy.result(), step=step)
        print(
            "step: {}/{}, train_loss: {:.5f}, train_accuracy: {:.5f}, valid_loss: {:.5f}, valid_accuracy: {:.5f}".format(
                step,
                math.ceil((EPOCHS * len(train_images)) / BATCH_SIZE),
                train_loss.result(),
                train_accuracy.result(),
                valid_loss.result(),
                valid_accuracy.result()))
        if ACC < valid_accuracy.result():
            ACC = train_accuracy.result()
            model.save_weights(filepath=save_model_dir + "{}_{:.5f}".format(step, ACC), save_format='tf')
        valid_loss.reset_states()
        valid_accuracy.reset_states()
    else:
        print("step: {}/{}, train_loss: {:.5f}, train_accuracy: {:.5f}".format(step,
                                                                               math.ceil((EPOCHS * len(
                                                                                   train_images)) / BATCH_SIZE),
                                                                               train_loss.result(),
                                                                               train_accuracy.result()))
    train_loss.reset_states()
    train_accuracy.reset_states()
tf.saved_model.save(model, save_model_dir + "final")
