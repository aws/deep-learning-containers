import ruamel.yaml
import os
import re
import warnings
import utils
import io

class Buildspec(object):
    def __init__(self):
        self.yaml = ruamel.yaml.YAML()
        self.yaml.allow_duplicate_keys = True
        self.yaml.Constructor.add_constructor("!join", self.join)

    def load(self, path):
        with open(path, 'r') as fp:
            raw_buildspec_text = fp.read()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._buildspec = self.yaml.load(raw_buildspec_text)

        for k in self._buildspec:
            v = self._buildspec[k]
            if isinstance(v, ruamel.yaml.scalarstring.PlainScalarString):
                if v.anchor is not None:
                    self._buildspec[k] = os.environ.get(v.anchor.value, v)

    def join(self, loader, node):
        seq = loader.construct_sequence(node)
        seq = ''.join([str(i) for i in seq])
        seq = ruamel.yaml.scalarstring.PlainScalarString(seq)
        seq.anchor.value = node.anchor 
        return seq

    def __getitem__(self, name):
        return self._buildspec[name]
