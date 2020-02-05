'''
Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You
may not use this file except in compliance with the License. A copy of
the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
'''

import os
import warnings

import ruamel.yaml

class Buildspec(object):
    '''
    The Buildspec class is responsible for parsing the buildspec file.
    It is used to standardize the ruamel.yaml configurations, add
    special constructors and load yaml files.
    '''
    def __init__(self):
        self.yaml = ruamel.yaml.YAML()
        self.yaml.allow_duplicate_keys = True
        self.yaml.Constructor.add_constructor("!join", self.join)

        self._buildspec = None

    def load(self, path):
        '''
        This function loads the buildspec file and
        populates the buildspec object.

        Parameters:
            path: str

        Returns:
            None

        '''
        with open(path, "r") as buildspec_file:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._buildspec = self.yaml.load(buildspec_file)

        for key in self._buildspec:
            val = self._buildspec[key]
            if isinstance(val, ruamel.yaml.scalarstring.PlainScalarString):
                if val.anchor is not None:
                    if val.anchor.value is not None:
                        self._buildspec[key] = os.environ.get(val.anchor.value, val)

    def join(self, loader, node):
        '''
        This method is used to perform string concatenation
        in the yaml file. Specifying !join [x,y,z] should
        result in the string xyz

        Parameters:
            loader: ruamel.yaml.constructor.RoundTripConstructor
            node: ruamel.yaml.nodes.SequenceNode

        Returns:
            str

        '''
        seq = loader.construct_sequence(node)
        seq = "".join([str(i) for i in seq])
        seq = ruamel.yaml.scalarstring.PlainScalarString(seq)
        if node.anchor is not None:
            seq.anchor.value = node.anchor
        return seq

    def __getitem__(self, name):
        '''
        This method adds dictionary style access to an object of the
        Buildspec class.

        Parameters:
            name: str

        Returns:
            object

        '''
        return self._buildspec[name]


