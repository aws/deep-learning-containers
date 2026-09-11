"""SageMaker GPU training integration test for the AutoGluon DLC (SageMaker SDK v3).

The entry script asserts torch sees the GPU and passes num_gpus to AutoGluon.
"""

from test_sm_training_cpu import run_tabular_training


def test_tabular_training_gpu(sagemaker_session):
    """Bagged fit on a GPU instance completes with CUDA visible and beats chance."""
    run_tabular_training(sagemaker_session, "ml.g4dn.2xlarge", "ag-tab-gpu")
