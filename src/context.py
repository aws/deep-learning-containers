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
import tarfile


class Context:
    """
    The context class encapsulates all required functions for
    preparing, managing and removing the docker build context
    """

    def __init__(self, artifacts=None, context_path="context.tar.gz", artifact_root="./"):
        """
        The constructor for the Context class

        Parameters:
            artifacts: array of (source, destination) tuples
            context_path: path for the resulting tar.gz file
            artifact_root: root directory for all artifacts

        Returns:
            None

        """
        self.artifacts = []
        self.context_path = context_path
        self.artifact_root = artifact_root

        # Check if the context path is just a filename,
        # or includes a directory. If path includes a
        # directory, create directory if it does not exist
        directory = os.path.dirname(context_path)
        if directory is not "" and not os.path.isdir(directory):
            os.mkdir(directory)

        if artifacts is not None:
            self.add(artifacts)

    def add(self, artifacts):
        """
        Adds artifacts to the build context
        Parameters:
            artifacts: array of (source, destination) tuples
        """
        # TODO: Add logic to untar and retar
        self.artifacts += artifacts

        # TODO: Use glob to expand

        with tarfile.open(self.context_path, "w:gz") as tar:
            for artifact in artifacts:
                source = os.path.join(self.artifact_root, artifact[0])
                target = artifact[1]
                if not isinstance(target, str):
                    continue
                tar.add(source, arcname=target)

    def remove(self):
        """
        Removes the context tar file

        Parameters:
            None

        Returns:
            None
        """
        os.remove(self.context_path)
