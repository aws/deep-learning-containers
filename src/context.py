"""
This file defines the Context class
"""

import os
import tarfile

class Context:
    """
    The context class encapsulates all required functions for
    preparing, managing and removing the docker build context
    """

    def __init__(
            self, artifacts=None, context_path="context.tar.gz", artifact_root="./"
    ):
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

        directory = "/".join(context_path.split("/")[:-1])
        if not os.path.isdir(directory):
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

        with tarfile.open(self.context_path, "w:gz") as tar:
            for artifact in artifacts:
                source = os.path.join(self.artifact_root, artifact[0])
                target = artifact[1]
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
