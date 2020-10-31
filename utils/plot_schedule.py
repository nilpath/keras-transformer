import matplotlib.pyplot as plt
import tensorflow as tf

from keras_transformer.train.learning_schedules import ModelSizeSchedule

if __name__ == "__main__":

    lr_schedule = ModelSizeSchedule(512, factor=1, warmup_steps=4000)

    plt.plot(lr_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()
