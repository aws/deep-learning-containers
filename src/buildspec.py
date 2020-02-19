"""
Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You
may not use this file except in compliance with the License. A copy of
the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
"""

import os
import warnings

import ruamel.yaml


class Buildspec:
    """
    The Buildspec class is responsible for parsing the buildspec file.
    It is used to standardize the ruamel.yaml configurations, add
    special constructors and load yaml files.
    """

    def __init__(self):
        self.yaml = ruamel.yaml.YAML()
        self.yaml.allow_duplicate_keys = True
        self.yaml.Constructor.add_constructor("!join", self.join)

        self._buildspec = None

    def load(self, path):
        """
        This function loads the buildspec file and
        populates the buildspec object.

        Parameters:
            path: str

        Returns:
            None

        """


        with open(path, "r") as buildspec_file:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._buildspec = self.yaml.load(buildspec_file)

        self._buildspec = self.override(self._buildspec)

    def override(self, yaml_object):
        """
        This method overrides anchors in a scalar string with
        values from the environment
        """
        # If the yaml object is a PlainScalarString or ScalarFloat and an environment variable
        # with the same name exists, return the environment variable otherwise,
        # return the original yaml_object
        scalar_types = (
                        ruamel.yaml.scalarstring.ScalarString, 
                        ruamel.yaml.scalarfloat.ScalarFloat,
                        ruamel.yaml.scalarstring.PlainScalarString,
                        ruamel.yaml.scalarbool.ScalarBoolean,
                        )

        if isinstance(yaml_object, ruamel.yaml.comments.CommentedMap):
            for key in yaml_object:
                yaml_object[key] = self.override(yaml_object[key])
        elif isinstance(yaml_object, scalar_types):
            if yaml_object.anchor is not None:
                if yaml_object.anchor.value is not None:
                    yaml_object = os.environ.get(yaml_object.anchor.value, yaml_object)

        # If the yaml object is not a PlainScalarString, does not have an anchor,
        # or it's anchor does not have a value, return
        # the original yaml object
        return yaml_object

    def join(self, loader, node):
        """
        This method is used to perform string concatenation
        in the yaml file. Specifying !join [x,y,z] should
        result in the string xyz

        Parameters:
            loader: ruamel.yaml.constructor.RoundTripConstructor
            node: ruamel.yaml.nodes.SequenceNode

        Returns:
            str

        """
        seq = [
            self.override(scalar_string)
            for scalar_string in loader.construct_sequence(node)
        ]
        seq = "".join([str(scalar_string) for scalar_string in seq])
        seq = ruamel.yaml.scalarstring.PlainScalarString(seq)
        if node.anchor is not None:
            seq.anchor.value = node.anchor
        return seq

    def __getitem__(self, name):
        """
        This method adds dictionary style access to an object of the
        Buildspec class.

        Parameters:
            name: str

        Returns:
            object

        """
        return self._buildspec[name]
