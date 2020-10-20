import tensorflow_datasets as tfds


class TextEncoder(tfds.deprecated.text.SubwordTextEncoder):
    def __init__(self, **kwargs):
        super(TextEncoder, self).__init__(**kwargs)

    @property
    def vocab_size(self):
        return super().vocab_size + 2

    def encode(self, s: str):
        # TODO: is it better to add <SOS> and <EOS> as text entities first instead?
        return [super().vocab_size] + super().encode(s) + [super().vocab_size+1]
