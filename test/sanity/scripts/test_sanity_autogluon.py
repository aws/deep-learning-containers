"""Serving sanity checks for the unified AutoGluon image."""

import os
import unittest
from importlib.metadata import version


class TestAutoGluonImage(unittest.TestCase):
    def test_container_type(self):
        self.assertEqual(os.environ.get("DLC_CONTAINER_TYPE"), "general")

    def test_sagemaker_model_directory(self):
        self.assertTrue(os.path.isdir("/opt/ml/model"))

    def test_serving_runtime_imports(self):
        import flask
        import gunicorn

        self.assertIsNotNone(flask)
        self.assertIsNotNone(gunicorn)
        self.assertTrue(version("flask"))
        self.assertTrue(version("gunicorn"))

    def test_adapter_files_exist(self):
        self.assertTrue(os.path.isfile("/opt/autogluon-server/server.py"))
        self.assertTrue(os.path.isfile("/opt/autogluon-server/gunicorn.conf.py"))
        self.assertTrue(os.access("/opt/autogluon-server/install_requirements.py", os.X_OK))
        self.assertTrue(os.access("/usr/local/bin/autogluon-sagemaker-entrypoint.sh", os.X_OK))


if __name__ == "__main__":
    unittest.main()
