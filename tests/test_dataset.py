import tensorflow as tf

from keras_transformer.dataset import synthetic_data


class SyntheticDataTest(tf.test.TestCase):
    def test_input_same_as_output(self):
        dataset = synthetic_data(20, 30, 10)
        for train, test, _, _ in dataset:
            self.assertAllEqual(train, test)

    def test_mask_shapes(self):
        dataset = synthetic_data(20, 30, 10)
        for x, y, x_mask, y_mask in dataset:
            for x, y, x_mask, y_mask in zip(x, y, x_mask, y_mask):
                self.assertEqual(len(x), tf.shape(x_mask)[0])
                self.assertEqual(len(x), tf.shape(x_mask)[1])
                self.assertEqual(len(y), tf.shape(y_mask)[0])
                self.assertEqual(len(y), tf.shape(y_mask)[1])
