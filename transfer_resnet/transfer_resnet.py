#!/usr/bin/env python3

import collections
from datetime import datetime
import functools
import logging
import math
from matplotlib import pyplot as plt
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D,
)


# prepare data preprocessing (resize images, ignore coarser labels,
# optionally apply random augmentations)
def _prepare_data_fn(features, input_shape, augment=False):
    """
    Resize image to expected dimensions and opt. apply some transformations
    :param features: Data
    :param input_shape: Shape expected by the models(images
        will be resized accordingly)
    :param augment: Flag to apply some random augmentations to the images
    :return: Augmented images, Labels
    """
    input_shape = tf.convert_to_tensor(input_shape)

    # tensorflow-dataset returns batches as feature dictionaries, expected by
    # Estimators. To train keras models it is more straightforward to return
    # the batch content as tuples
    image = features['image']
    label = features['label']

    # Convert the images to float type, also scaling their value from
    # [0, 255] to [0., 1.]:
    image = tf.image.convert_image_dtype(image, tf.float32)
    if augment:
        # apply the random image augmentations
        image = tf.image.random_flip_left_right(image)
        # Random brightness/saturation changes:
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0.0, 1.0)

        # Random resize and crop
        random_scale_factor = tf.random.uniform([1], minval=1., maxval=1.4,
                                                dtype=tf.float32)
        scaled_height = tf.cast(tf.cast(input_shape[0], tf.float32)
                                * random_scale_factor, tf.int32)
        scaled_width = tf.cast(tf.cast(input_shape[1], tf.float32)
                               * random_scale_factor, tf.int32)
        scaled_shape = tf.squeeze(tf.stack([scaled_height, scaled_width]))
        image = tf.image.resize(image, scaled_shape)
        image = tf.image.random_crop(image, input_shape)
    else:
        image = tf.image.resize(image, input_shape[:2])

    return image, label


def load_dataset(input_shape, batch_size=32, num_epochs=300, random_seed=42):
    """Using cifar100 as training data
    First need to organise the data- converting it to tensor type
    of the correct size and shape etc.

    :param input_shape: shape of imput passed to the model
    :param batch_size:  desired size of training batches
    :param num_epochs:  number of epochs to train for
    :param random_seed: seed to shuffle training data

    :return: data_dict: {'train_dataset': training dataset,
                        'test_dataset':  validation dataset,
                        'train_steps_per': step_per_epoch to train,
                        'val_steps_per': steps_per_epoch to test,
                        'num_classes': number of classes the model predicts for
                        }
    """
    cifar_builder = tfds.builder("cifar100")
    cifar_builder.download_and_prepare()

    """to view dataset information:
            print(cifar_builder.info)
    to view all of the different defined class names:
            cifar_100 classes = 'label', coarse classes = coarse_label
            print(cifar_builder.info.features["label"].names)"""

    # Train/val Datasets:
    # train_cifar_dataset = cifar_builder.as_dataset(split=tfds.Split.TRAIN)
    train_cifar_dataset = cifar_builder.as_dataset(split='train')

    # val_cifar_dataset = cifar_builder.as_dataset(split=tfds.Split.TEST)
    val_cifar_dataset = cifar_builder.as_dataset(split='test')

    # number of classes:
    num_classes = cifar_builder.info.features['label'].num_classes

    # Number of images:
    # num_train_imgs = cifar_builder.info.splits['train'].num_examples
    num_train_imgs = cifar_builder.info.splits['train'].num_examples

    # num_val_imgs = cifar_builder.info.splits['test'].num_examples
    num_val_imgs = cifar_builder.info.splits['test'].num_examples

    # tell tensorflow to iterate over the samples by the num of desired epochs
    # and to shuffle them before returning
    train_cifar_dataset = train_cifar_dataset.repeat(num_epochs).shuffle(
        random_seed)

    # Organise the data for training
    prepare_data_fn_for_train = functools.partial(_prepare_data_fn,
                                                  input_shape=input_shape,
                                                  augment=True)

    train_cifar_dataset = train_cifar_dataset.map(prepare_data_fn_for_train,
                                                  num_parallel_calls=4)

    # Also ask the dataset to batch the samples
    train_cifar_dataset = train_cifar_dataset.batch(batch_size)
    train_cifar_dataset = train_cifar_dataset.prefetch(1)  # improve time

    # prepare the validation dataset (though not shuffling or augmenting)
    prepare_data_fn_for_val = functools.partial(_prepare_data_fn,
                                                input_shape=input_shape,
                                                augment=False)

    val_cifar_dataset = (val_cifar_dataset
                         .repeat()
                         .map(prepare_data_fn_for_val,
                              num_parallel_calls=4)
                         .batch(batch_size)
                         .prefetch(1))

    train_steps_per_epoch = math.ceil(num_train_imgs/batch_size)
    val_steps_per_epoch = math.ceil(num_val_imgs/batch_size)

    return {'train_dataset': train_cifar_dataset,
            'test_dataset':  val_cifar_dataset,
            'train_steps_per': train_steps_per_epoch,
            'val_steps_per': val_steps_per_epoch,
            'num_classes': num_classes}


def create_model(num_classes):
    """
    Create resnet50 model, freeze layers append new trainable output layer
    and compile.

    :param num_classes: number of different classes transfering detection to
    :return: compiled model
    """
    resnet50_feature_extractor = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape
    )

    # need to freeze the necessary trained layers before transferring to cifar
    frozen_layers, trainable_layers = [], []
    for layer in resnet50_feature_extractor.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.trainable = False
            frozen_layers.append(layer.name)
        else:
            if len(layer.trainable_weights) > 0:
                # Only list as trainable those layers with trainable parameters
                trainable_layers.append(layer.name)

    print("Layers we froze:{0} (total = {1}).".format(
        frozen_layers, len(frozen_layers),
    ))
    print("\nLayers which will be fine-tuned:{0}"
          " (total = {1}).".format(
           trainable_layers, len(trainable_layers),
          ))

    # now add the layer on top of the frozen model to make predictions from
    # the features
    features = resnet50_feature_extractor.output
    avg_pool = GlobalAveragePooling2D(data_format="channels_last")(features)
    predictions = Dense(num_classes, activation='softmax')(avg_pool)

    resnet50_transfer = Model(resnet50_feature_extractor.input, predictions)

    optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)
    resnet50_transfer.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,
                                                           name='top5_acc')
        ])

    return resnet50_transfer


def train_model(data_dict, model, model_dir, num_epochs):
    """
    fit the model to the data

    :param data_dict:  {'train_dataset': training dataset,
                        'test_dataset':  validation dataset,
                        'train_steps_per': step_per_epoch to train,
                        'val_steps_per': steps_per_epoch to test,
                        'num_classes': number of classes the model predicts for
                        }
    :param model:       model to be trained
    :param model_dir:   directory in which to save model checkpoints
    :param num_epochs:  number of epochs to train for

    :return: trained model
    """

    # set log callback parameters for model training
    metrics_to_print = collections.OrderedDict([("loss", "loss"),
                                                ("v-loss", "val_loss"),
                                                ("acc", "acc"),
                                                ("v-acc", "val_acc"),
                                                ("top5-acc", "top5_acc"),
                                                ("v-top5-acc", "val_top5_acc")
                                                ])

    callback_simple_log = SimpleLogCallback(metrics_dict=metrics_to_print,
                                            logger=logger,
                                            num_epochs=num_epochs,
                                            log_frequency=5,
                                            )

    callbacks = [
        # interrupt training if validation/loss metrics stop improving
        tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_acc',
                                         restore_best_weights=True),
        # save the model (e.g. every 5 epochs) epoch and loss in filename
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'model-ckpt-weights.ckpt'),
            save_best_only=True,
            save_weights_only=True,
            save_freq='epoch'),
        # Log callback:
        callback_simple_log
    ]

    model_history = model.fit(data_dict['train_dataset'],
                              epochs=num_epochs,
                              steps_per_epoch=data_dict['train_steps_per'],
                              validation_data=data_dict['test_dataset'],
                              validation_steps=data_dict['val_steps_per'],
                              verbose=1, callbacks=callbacks)

    return model, model_history


def eval_model(model, eval_loc):
    """
    generate a summary of training stats and save it to model_dir

    :param model: trained model
    :param model_dir:   directory in which to save model training evaluation

    :return: n/a
    """
    fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex='col')
    ax[0, 0].set_title("loss")
    ax[0, 1].set_title("val-loss")
    ax[1, 0].set_title("acc")
    ax[1, 1].set_title("val-acc")
    ax[2, 0].set_title("top5-acc")
    ax[2, 1].set_title("val-top5-acc")

    ax[0, 0].plot(model.history["loss"])
    ax[0, 1].plot(model.history["val_loss"])
    ax[1, 0].plot(model.history["acc"])
    ax[1, 1].plot(model.history["val_acc"])
    ax[2, 0].plot(model.history["top5_acc"])
    ax[2, 1].plot(model.history["val_top5_acc"])

    plt.savefig(eval_loc)

    best_val_acc = max(model.history['val_acc']) * 100
    best_val_top5 = max(model.history['val_top5_acc']) * 100

    logger.info('Best val acc:  {:2.2f}%'.format(best_val_acc))
    logger.info('Best val top5: {:2.2f}%'.format(best_val_top5))


# Implementing own callbacks by inheriting from the abstract Callback class.
# This class defines an interface composed of several methods which will be
# called by Keras along the training(before each epoch, before/after each batch
# iteration etc.)
class SimpleLogCallback(tf.keras.callbacks.Callback):
    """Keras callback for simple denser console logs"""

    def __init__(self, metrics_dict, logger, num_epochs='?', log_frequency=1,
                 metric_string_template='[[name]] = {[[value]]:5.3f}'):

        """
        Initialize the Callback.
        :param metrics_dict:    Dictonary containing mappings for metrics names
                                /keys e.g.{"accuracy": "acc",
                                           "val. accuracy": "val_acc"}
        :param num_epochs:      Number of training epochs
        :param log_frequency:   Log frequency (in epochs)
        :param metric_string_template: (opt.) template to print each metric
        """
        super().__init__()
        self.metrics_dict = collections.OrderedDict(metrics_dict)
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency
        self.logger = logger

        # format string to later print the metrics,
        # e.g "Epoch 0/9:  loss = 1.00; val-loss=2.0"
        log_string_template = 'Epoch {0:2}/{1}: '
        separator = '; '

        i = 2
        for metric_name in self.metrics_dict:
            templ = metric_string_template.replace('[[name]]',
                                                   metric_name).replace(
                                                       '[[value]]', str(i))
            log_string_template += templ + separator
            i += 1

        # Remove the "; " after the last element
        log_string_template = log_string_template[:-len(separator)]
        self.log_string_template = log_string_template

    def on_train_begin(self, logs=None):
        self.logger.info("Training: **start**")

    def on_train_end(self, logs=None):
        self.logger.info("Training: **end**")

    def on_epoch_end(self, epoch, logs={}):
        if (epoch - 1) % self.log_frequency == 0 or epoch == self.num_epochs:
            values = [logs[self.metrics_dict[metric_name]] for metric_name
                      in self.metrics_dict]
            self.logger.info(self.log_string_template.format(epoch,
                                                             self.num_epochs,
                                                             *values))


# main loop
if __name__ == "__main__":

    # Define necessary variables-----------------------------------------------
    input_shape = [224, 224, 3]  # will resize the images to this shape
    batch_size = 32  # images per batch
    num_epochs = 300  # max number of training epochs
    random_seed = 42

    # location to save and checkpoint model
    model_dir = './models/'
    # location and initial name to save logs - will have datetime appended
    logs_dir = './logs/'

    # set up logger------------------------------------------------------------
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # handler for console output
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_format = logging.Formatter('%(message)s')
    c_handler.setFormatter(c_format)
    # logger.addHandler(c_handler)

    # handler for logfile output
    current_log_file = datetime.now().strftime(
        'transfer_resnet_%Y-%m-%d:%H:%M:%S.log')
    f_handler = logging.FileHandler(os.path.join(logs_dir, current_log_file))
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter('%(asctime)s - %(levelname)s '
                                 '- %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    logger.info('Application started')

    try:
        # generate dataset-----------------------------------------------------
        data_dict = load_dataset(input_shape, batch_size, num_epochs,
                                 random_seed)
        logger.info('Generated data dict')

        # create model---------------------------------------------------------
        resnet50_cifar = create_model(data_dict['num_classes'])
        logger.info('Generated model')

        # train the model------------------------------------------------------
        trained_resnet, train_history = train_model(model=resnet50_cifar,
                                                    data_dict=data_dict,
                                                    model_dir=model_dir,
                                                    num_epochs=num_epochs)

        model_save_path = os.path.join(model_dir,
                                       'resnet50_cifar_transfer.h5')
        trained_resnet.save(model_save_path)
        logger.info('Saved model to {}'.format(model_save_path))

        # evaluate model-------------------------------------------------------
        eval_path = os.path.join(model_dir,
                                 'resnet50_cifar_eval')

        eval_model(model=train_history, eval_loc=eval_path)
        logger.info('Saved training evaluation to {}'.format(model_dir))

    except Exception as e:
        # given exception, log traceback info
        logger.exception("Program closed unexpectedly")
