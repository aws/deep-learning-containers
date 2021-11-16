###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

usage()
{
    if [ $1 == "run_tests" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -h,  --help                               Prints this help"
        echo -e "  -rd,  --run-deselected                    runs deselected tests only"
        echo -e "  -rh,  --run-hang                          runs hang tests only"
        echo -e "  -rj SW-XXXX,  --run-jira SW-XXXX          runs deselected tests under a specific jira; preferably add subfolder too"
        echo -e "  -rj all,      --run-jira all              runs deselected tests under all jiras"
        echo -e "  -dtc,  --dump-test-count                  get total number of tests to be run(excludes the files that are giving python import errors)"
        echo -e "<Optional>Path to any specific subfolders   runs tests of these subfolders only"
        echo -e "Note -> Execute this script from google_tf_tests folder only"
    fi
    echo -e ""
    return 0
}

set_tf_dir_flags()
{
    # Declare an array for the files which execute with --forked
    forked_enable_files=("tensorflow/python/eager/function_test.py"
                         "tensorflow/python/eager/remote_cluster_test.py"
                         "tensorflow/python/eager/remote_execution_test.py"
                         "tensorflow/python/distribute/coordinator/cluster_coordinator_test.py"
                         "tensorflow/python/kernel_tests/collective_ops_test.py"
                         "tensorflow/python/ops/parallel_for/array_test.py"
                         "tensorflow/python/ops/parallel_for/control_flow_ops_test.py"
                         "tensorflow/python/distribute/coordinator/cluster_coordinator_test.py"
                         "tensorflow/python/distribute/collective_all_reduce_strategy_test.py"
                        )

    # TODO - Remove
    # xdist and forked together don't seem to work fine
    # Below files give random failures in xdist mode
    xdist_disable_files=("tensorflow/python/autograph/pyct/transpiler_test.py"
                         "tensorflow/python/autograph/impl/api_test.py"
                         "tensorflow/python/autograph/pyct/static_analysis/reaching_definitions_test.py"
                         "tensorflow/python/autograph/pyct/static_analysis/activity_test.py"
                         "tensorflow/python/autograph/pyct/static_analysis/liveness_test.py"
                         "tensorflow/python/autograph/converters/control_flow_test.py"
                         "tensorflow/python/client/session_test.py"
                         "tensorflow/python/debug/lib/debug_events_writer_test.py"
                         "tensorflow/python/debug/lib/dist_session_debug_grpc_test.py"
                         "tensorflow/python/debug/lib/dumping_callback_test.py"
                         "tensorflow/python/debug/lib/grpc_large_data_test.py"
                         "tensorflow/python/debug/lib/session_debug_grpc_test.py"
                         "tensorflow/python/debug/lib/source_remote_test.py"
                         "tensorflow/python/debug/wrappers/dumping_wrapper_test.py"
                         "tensorflow/python/debug/wrappers/framework_test.py"
                         "tensorflow/python/distribute/coordinator/cluster_coordinator_test.py"
                         "tensorflow/python/distribute/collective_all_reduce_strategy_test.py"
                         "tensorflow/python/distribute/distribute_coordinator_test.py"
                         "tensorflow/python/distribute/parameter_server_strategy_test.py"
                         "tensorflow/python/distribute/parameter_server_strategy_v2_test.py"
                         "tensorflow/python/eager/core_test.py"
                         "tensorflow/python/eager/forwardprop_test.py"
                         "tensorflow/python/eager/function_gradients_test.py"
                         "tensorflow/python/eager/function_test.py"
                         "tensorflow/python/eager/remote_cluster_test.py"
                         "tensorflow/python/eager/remote_execution_test.py"
                         "tensorflow/python/framework/config_test.py"
                         "tensorflow/python/framework/op_callbacks_test.py"
                         "tensorflow/python/framework/ops_test.py"
                         "tensorflow/python/framework/test_util_test.py"
                         "tensorflow/python/keras/callbacks_test.py"
                         "tensorflow/python/keras/distribute/multi_worker_test.py"
                         "tensorflow/python/keras/mixed_precision/autocast_variable_test.py"
                         "tensorflow/python/keras/utils/multi_gpu_utils_test.py"
                         "tensorflow/python/keras/distribute/mirrored_variable_test.py"
                         "tensorflow/python/keras/mixed_precision/layer_correctness_test.py"
                         "tensorflow/python/keras/utils/dataset_creator_test.py"
                         "tensorflow/python/keras/engine/training_integration_test.py"
                         "tensorflow/python/keras/layers/lstm_test.py"
                         "tensorflow/python/keras/layers/recurrent_test.py"
                         "tensorflow/python/kernel_tests/barrier_ops_test.py"
                         "tensorflow/python/kernel_tests/collective_ops_multi_worker_test.py"
                         "tensorflow/python/kernel_tests/collective_ops_test.py"
                         "tensorflow/python/kernel_tests/cond_v2_test.py"
                         "tensorflow/python/kernel_tests/conditional_accumulator_test.py"
                         "tensorflow/python/kernel_tests/control_flow_ops_py_test.py"
                         "tensorflow/python/kernel_tests/dense_update_ops_no_tsan_test.py"
                         "tensorflow/python/kernel_tests/fifo_queue_test.py"
                         "tensorflow/python/kernel_tests/linalg/sparse/csr_sparse_matrix_ops_test.py"
                         "tensorflow/python/kernel_tests/list_ops_test.py"
                         "tensorflow/python/kernel_tests/map_stage_op_test.py"
                         "tensorflow/python/kernel_tests/matrix_exponential_op_test.py"
                         "tensorflow/python/kernel_tests/padding_fifo_queue_test.py"
                         "tensorflow/python/kernel_tests/priority_queue_test.py"
                         "tensorflow/python/kernel_tests/random/random_shuffle_queue_test.py"
                         "tensorflow/python/kernel_tests/reader_ops_test.py"
                         "tensorflow/python/kernel_tests/sparse_conditional_accumulator_test.py"
                         "tensorflow/python/kernel_tests/sparse_cross_op_test.py"
                         "tensorflow/python/kernel_tests/sparse_tensor_dense_matmul_op_test.py"
                         "tensorflow/python/kernel_tests/stage_op_test.py"
                         "tensorflow/python/kernel_tests/tensor_array_ops_test.py"
                         "tensorflow/python/kernel_tests/while_v2_test.py"
                         "tensorflow/python/kernel_tests/variable_scope_test.py"
                         "tensorflow/python/lib/io/file_io_test.py"
                         "tensorflow/python/ops/collective_ops_test.py"
                         "tensorflow/python/ops/stateful_random_ops_test.py"
                         "tensorflow/python/ops/numpy_ops/np_array_ops_test.py"
                         "tensorflow/python/ops/numpy_ops/np_interop_test.py"
                         "tensorflow/python/profiler/integration_test/profiler_api_test.py"
                         "tensorflow/python/profiler/profiler_v2_test.py"
                         "tensorflow/python/saved_model/load_test.py"
                         "tensorflow/python/saved_model/save_test.py"
                         "tensorflow/python/saved_model/save_context_test.py"
                         "tensorflow/python/summary/writer/writer_test.py"
                         "tensorflow/python/training/coordinator_test.py"
                         "tensorflow/python/training/monitored_session_test.py"
                         "tensorflow/python/training/saving/functional_saver_test.py"
                         "tensorflow/python/training/server_lib_multiple_containers_test.py"
                         "tensorflow/python/training/server_lib_same_variables_clear_container_test.py"
                         "tensorflow/python/training/server_lib_same_variables_clear_test.py"
                         "tensorflow/python/training/server_lib_same_variables_no_clear_test.py"
                         "tensorflow/python/training/server_lib_sparse_job_test.py"
                         "tensorflow/python/training/server_lib_test.py"
                         "tensorflow/python/training/server_lib_same_variables_no_clear_test.py"
                         "tensorflow/python/training/server_lib_sparse_job_test.py"
                         "tensorflow/python/training/server_lib_test.py"
                         "tensorflow/python/training/input_test.py"
                         "tensorflow/python/training/monitored_session_test.py"
                         "tensorflow/python/training/queue_runner_test.py"
                         "tensorflow/python/training/sync_replicas_optimizer_test.py"
                         "tensorflow/python/util/lock_util_test.py"
                         "tensorflow/python/util/nest_test.py"
                         "tensorflow/python/util/tf_decorator_test.py"
                         "/tensorflow/python/keras/layers/gru_v2_test.py"
                         "/tensorflow/python/keras/layers/gru_test.py"
                         "/tensorflow/python/keras/layers/simplernn_test.py"
                         "/tensorflow/python/keras/tests/model_architectures_test.py"
                        )

    # This list needs to be updated for directories which perform file-by-file testing
    # v2.5.0 specific files - TODO we need to find a better solution
    # tensorflow/python/framework/composite_tensor_utils_test.py
    # tensorflow/python/framework/python_api_parameter_converter_test.py
    # tensorflow/python/framework/python_tensor_converter_test.py
    # tensorflow/python/training/tensorboard_logging_test.py
    files_to_ignore=("tensorflow/python/autograph/pyct/static_analysis/activity_py3_test.py"
                     "tensorflow/python/autograph/pyct/static_analysis/liveness_py3_test.py"
                     "tensorflow/python/autograph/pyct/static_analysis/reaching_definitions_py3_test.py"
                     "tensorflow/python/autograph/pyct/testing/codegen_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/sql_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/sql_dataset_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/optimization/latency_all_edges_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/assert_cardinality_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/auto_shard_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/batch_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/cache_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/choose_fastest_branch_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/choose_fastest_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/concatenate_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/csv_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/dataset_constructor_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/filter_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/fixed_length_record_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/flat_map_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/group_by_reducer_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/group_by_window_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/ignore_errors_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/interleave_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/map_and_batch_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/map_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/matching_files_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/optimize_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/padded_batch_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/parallel_interleave_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/parallel_map_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/parse_example_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/prefetch_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/range_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/rebatch_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/sample_from_datasets_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/scan_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/sequence_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/shard_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/shuffle_and_repeat_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/shuffle_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/snapshot_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/stats_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/take_while_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/textline_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/tf_record_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/unbatch_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/unique_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/serialization/zip_dataset_serialization_test.py"
                     "tensorflow/python/data/experimental/kernel_tests/stats_dataset_ops_test.py"
                     "tensorflow/python/distribute/tpu_strategy_test.py"
                     "tensorflow/python/distribute/all_reduce_test.py"
                     "tensorflow/python/distribute/distributed_file_utils_test.py"
                     "tensorflow/python/keras/distribute/dataset_creator_model_fit_test.py"
                     "tensorflow/python/eager/custom_device_test.py"
                     "tensorflow/python/eager/def_function_xla_jit_test.py"
                     "tensorflow/python/eager/def_function_xla_test.py"
                     "tensorflow/python/eager/gradient_input_output_exclusions_test.py"
                     "tensorflow/python/eager/pywrap_tensor_test.py"
                     "tensorflow/python/eager/benchmarks/resnet50/hvp_test.py"
                     "tensorflow/python/eager/benchmarks/resnet50/resnet50_graph_test.py"
                     "tensorflow/python/eager/benchmarks/resnet50/resnet50_test.py"
                     "tensorflow/python/eager/benchmarks/kpi_benchmark_test.py"
                     "tensorflow/python/eager/benchmarks_test.py"
                     "tensorflow/python/eager/remote_benchmarks_test.py"
                     "tensorflow/python/framework/composite_tensor_utils_test.py"
                     "tensorflow/python/framework/experimental/unified_api_test.py"
                     "tensorflow/python/framework/memory_checker_test.py"
                     "tensorflow/python/framework/py_context_manager_test.py"
                     "tensorflow/python/framework/python_tensor_converter_test.py"
                     "tensorflow/python/framework/python_api_parameter_converter_test.py"
                     "tensorflow/python/framework/ops_enable_eager_test.py"
                     "tensorflow/python/grappler/cost_analyzer_test.py"
                     "tensorflow/python/grappler/model_analyzer_test.py"
                     "tensorflow/python/keras/benchmarks/layer_benchmarks/layer_benchmarks_test.py"
                     "tensorflow/python/keras/layers/preprocessing/normalization_tpu_test.py"
                     "tensorflow/python/keras/tests/automatic_outside_compilation_test.py"
                     "tensorflow/python/keras/tests/get_config_test.py"
                     "tensorflow/python/keras/tests/tracking_util_xla_test.py"
                     "tensorflow/python/keras/benchmarks/eager_microbenchmarks_test.py"
                     "tensorflow/python/keras/benchmarks/keras_cpu_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/saved_model_benchmarks/inception_resnet_v2_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/saved_model_benchmarks/densenet_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/saved_model_benchmarks/nasnet_large_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/saved_model_benchmarks/resnet152_v2_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/saved_model_benchmarks/vgg_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/saved_model_benchmarks/efficientnet_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/saved_model_benchmarks/mobilenet_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/saved_model_benchmarks/xception_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/model_components_benchmarks_test.py"
                     "tensorflow/python/keras/benchmarks/keras_examples_benchmarks/mnist_irnn_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/keras_examples_benchmarks/mnist_conv_custom_training_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/keras_examples_benchmarks/mnist_hierarchical_rnn_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/keras_examples_benchmarks/reuters_mlp_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/keras_examples_benchmarks/antirectifier_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/keras_examples_benchmarks/cifar10_cnn_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/keras_examples_benchmarks/text_classification_transformer_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/keras_examples_benchmarks/mnist_conv_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/keras_examples_benchmarks/bidirectional_lstm_benchmark_test.py"
                     "tensorflow/python/keras/benchmarks/optimizer_benchmarks_test.py"
                     "tensorflow/python/kernel_tests/proto/decode_proto_op_test.py"
                     "tensorflow/python/kernel_tests/proto/descriptor_source_test.py"
                     "tensorflow/python/kernel_tests/proto/encode_proto_op_test.py"
                     "tensorflow/python/kernel_tests/decode_jpeg_op_test.py"
                     "tensorflow/python/kernel_tests/reduce_benchmark_test.py"
                     "tensorflow/python/kernel_tests/matrix_square_root_op_test.py"
                     "tensorflow/python/ops/v1_compat_tests/gradient_checker_test.py"
                     "tensorflow/python/ops/raw_ops_test.py"
                     "tensorflow/python/platform/app_test.py"
                     "tensorflow/python/profiler/integration_test/profiler_api_test.py"
                     "tensorflow/python/profiler/internal/run_metadata_test.py"
                     "tensorflow/python/profiler/model_analyzer_test.py"
                     "tensorflow/python/profiler/pprof_profiler_test.py"
                     "tensorflow/python/profiler/profile_context_test.py"
                     "tensorflow/python/profiler/profiler_test.py"
                     "tensorflow/python/tf_program/tests/mlir_gen_test.py"
                     "tensorflow/python/tools/api/generator/output_init_files_test.py"
                     "tensorflow/python/tpu/tpu_test_wrapper_test.py"
                     "tensorflow/python/training/basic_session_run_hooks_test.py"
                     "tensorflow/python/training/tracking/benchmarks_test.py"
                     "tensorflow/python/training/tensorboard_logging_test.py"
                     "tensorflow/python/util/function_parameter_canonicalizer_test.py"
                     "tensorflow/python/util/protobuf/compare_test.py"
                    )
}

set_tf_sub_dir_flags()
{
    forked=""
    if [[ ! -z "$run_xfail" ]]; then
        forked="--forked"
    fi

    max_worker_count=0

    #tf_sub_dir is passed as v2.4.1/tensorflow/python/kernel_tests
    #so we need to get v2.4.1 in __tf_dir and tensorflow/python/kernel_tests in __tf_sub_dir
    #cd to __tf_dir and then execute pytest because tests such as tensorflow/python/ops/image_ops_test.py
    #expect relative paths.
    __tf_dir=$(echo "$tf_sub_dir"| cut -d'/' -f 1)
    __tf_sub_dir=$(echo "$tf_sub_dir"| cut -d'/' -f2-)
    module_name=$(echo "$tf_sub_dir"| cut -d'/' -f 4) #this will give us ops, kernel_tests, keras etc
    mkdir -p xml_test_results
    tf_report="$PWD/xml_test_results/google_tf_test_report_${module_name}_${__tf_dir}.xml" #Form "$TF_TESTS_ROOT/tests/tf_training_tests/google_tf_tests/google_tf_test_report_kernel_tests.xml etc"
    mkdir -p logs
    log_file="$PWD/logs/google_tf_test_log_${module_name}_${__tf_dir}.log"
    suite_status=0
    suite_start_time=`date +%s`
}

check_ignore()
{
    ignore=0
    for item in "${files_to_ignore[@]}"
    do
        if [[ $file_path == $item ]]; then
            ignore=1
        fi
    done
}

check_xdist_disable()
{
    xdist_disable=0
    for item in "${xdist_disable_files[@]}"
    do
        if [[ $file_path == $item ]]; then
            xdist_disable=1
        fi
    done
}

check_forked_enable()
{
    forked=""
    for item in "${forked_enable_files[@]}"
    do
        if [[ $file_path == $item ]]; then
            forked="--forked"
        fi
    done
}

run_file()
{
    file_report_name=${file_path//'/'/'_'}
    tf_file_report="tmp/google_tf_test_report_${module_name}_${file_report_name}_${__tf_dir}.xml"
    log_per_file="tmp/google_tf_test_log_${module_name}_${file_report_name}_${__tf_dir}.log"

    # Below will create the log file and following that, tee -a is used
    echo -e "Executing file: $file_path  $count/$total_file_count" > ${log_per_file} 2>&1

    # Add check_forked_enable
    check_forked_enable

    local test_start_time=`date +%s`
    # Global Timeout set to 2400 secs per file
    TEST_SRCDIR="$PWD"  PYTEST_ADDOPTS=" -vs ${forked} \
    --continue-on-collection-errors --junitxml=${tf_file_report} -o junit_suite_name=${module_name} \
    ${run_xfail} --durations=5 -rxXs" timeout 2400 python3 -X faulthandler -m pytest ${file_path} >> ${log_per_file} 2>&1
    local retVal=$?
    local test_end_time=`date +%s`
    local test_run_time=$((test_end_time-test_start_time))
    if [ $retVal != 0 ]; then
        echo -e "\npytest command return value of $file_path is ${retVal}" >> ${log_per_file} 2>&1
        echo "Execution result from the background job of $file_path: FAILED; Execution time: ${test_run_time} secs" | tee -a ${log_per_file} 2>&1
    else
        echo "Execution result from the background job of $file_path: PASSED; Execution time: ${test_run_time} secs" | tee -a ${log_per_file} 2>&1
    fi

    exit $retVal
}


wait_on_tests()
{
    file_completed=0
    while [ ${#file_pid_array[@]} != 0 ] && [ $file_completed -eq 0 ]
    do
        for file_name in "${!file_pid_array[@]}"
        do
            pid=${file_pid_array[$file_name]}
            if [ ! -d "/proc/${pid}" ]; then
                wait $pid
                local retVal=$?
                if [ $retVal != 0 ]; then
                    echo -e "PID return value is ${retVal} ; Incrementing overall FAILED count and setting suite status to FAILED"
                    fail_cnt=$((fail_cnt+1))
                    suite_status=1
                fi
                unset file_pid_array[$file_name]
                ((--bg_worker_iter))
                #echo "Completed exeuction of $file_name & $pid. Active workers $bg_worker_iter/$max_worker_count"
                if [ $last_batch != 1 ] || [ ${#file_pid_array[@]} == 0 ]; then
                    file_completed=1
                    break
                fi
            fi
        done
    done
}

run_tf_sub_dir_tests_per_file_experimental()
{
    set_tf_sub_dir_flags

    # Below will create the log file and following that, tee -a is used
    echo -e "Executing ${__tf_sub_dir} from ${__tf_dir} folder" | tee ${log_file}

    rm -f $tf_report
    cd $__tf_dir
    # Remove temporary folder if exists
    rm -rf tmp/
    mkdir tmp
    echo -e "Number of files is $(find ${__tf_sub_dir} -type f -name "*_test.py" | wc -l)" | tee -a ${log_file}
    count=1
    total_file_count=$(find ${__tf_sub_dir} -type f -name "*_test.py" | wc -l)
    retVal=0
    bg_worker_iter=0
    last_batch=0
    check_active_device_state
    file_arr=( $( find ${__tf_sub_dir} -type f -name "*_test.py" | sort) )
    for file_path in "${file_arr[@]}"
    do
        check_ignore
        if [[ $ignore == 1 ]]; then
            echo -e "Ignoring $file_path as it breaks python import OR do not collect any tests $count/$total_file_count" | tee -a ${log_file}
            count=`expr $count + 1`
            continue
        else
            echo -e "Executing file: $file_path  $count/$total_file_count"
            run_file &
            bg_pid=$!
            file_pid_array[$file_path]=$bg_pid
            ((++bg_worker_iter))
            count=`expr $count + 1`
        fi
        if [ $bg_worker_iter -ge $max_worker_count ]; then
            #echo "All $bg_worker_iter devices are busy executing the tests. Waiting for atleast one test to finish"
            wait_on_tests
            check_active_device_state
        fi
    done
    last_batch=1
    echo "Files remaining in the execution pipeline: ${#file_pid_array[@]}"
    wait_on_tests
    echo "Remaining files after execution: ${#file_pid_array[@]}"

    cat tmp/google_tf_test_log_${module_name}_*_${__tf_dir}.log >> $log_file
    #echo -e "$(cat $log_file)"
    echo -e "Number of junit xml generated is $(find tmp/ -type f -name "google_tf_test_report_${module_name}_*_${__tf_dir}.xml" | wc -l)" | tee -a ${log_file}
    junitparser merge --glob "tmp/google_tf_test_report_${module_name}_*_${__tf_dir}.xml" $tf_report
    # Remove temporary folder
    rm -rf tmp/
    suite_end_time=`date +%s`
    suite_run_time=$((suite_end_time-suite_start_time))
    echo -e "Suite Execution Time -> ${suite_run_time} secs" | tee -a ${log_file}
    if [ ${suite_status} != 0 ]; then
        echo -e "Suite overall status is 1 FAIL" | tee -a ${log_file}
    else
        echo -e "Suite overall status is 0 PASS" | tee -a ${log_file}
    fi
    echo -e "Check suite all logs in $log_file"
    echo -e "========================================================================================================="

    find . -name "core.*"|xargs rm -rf

    cd ..
}

# TODO - Remove
run_tf_sub_dir_tests_per_file()
{
    set_tf_sub_dir_flags

    # Below will create the log file and following that, tee -a is used
    echo -e "Executing ${__tf_sub_dir} from ${__tf_dir} folder" | tee ${log_file}
    if [[ ! -z "$forked" ]]; then
        echo -e "Executing suite with forked flag enabled for non xdist mode files" | tee -a ${log_file}
    fi

    rm -f $tf_report
    cd $__tf_dir
    # Remove temporary folder if exists
    rm -rf tmp/
    mkdir tmp
    echo -e "Number of files is $(find ${__tf_sub_dir} -type f -name "*_test.py" | wc -l)" | tee -a ${log_file}
    count=1
    total_file_count=$(find ${__tf_sub_dir} -type f -name "*_test.py" | wc -l)
    for file in $( find ${__tf_sub_dir} -type f -name "*_test.py")
    do
        check_ignore
        if [[ $ignore == 1 ]]; then
            echo -e "Ignoring $file as it breaks python import OR do not collect any tests" | tee -a ${log_file}
            count=`expr $count + 1`
        else
            file_report_name=${file//'/'/'_'}
            tf_file_report="tmp/google_tf_test_report_${module_name}_${file_report_name}_${__tf_dir}.xml"
            log_per_file="tmp/google_tf_test_log_${module_name}_${file_report_name}_${__tf_dir}.log"
            # Below will create the log file and following that, tee -a is used
            echo -e "Executing file: $file  $count/$total_file_count" | tee ${log_per_file}
            check_xdist_disable

            # Check available ASICs and update max_worker_count
            check_active_device_state

            # Timeout set to 1800 secs per file for single worker and 1500 secs per file for multiple workers
            if [[ $max_worker_count -le 1 ]] || \
               [[ $xdist_disable == 1 ]]; then
                TEST_SRCDIR="$PWD"  PYTEST_ADDOPTS=" -v ${forked} \
                --continue-on-collection-errors --junitxml=${tf_file_report} -o junit_suite_name=${module_name} ${run_xfail} \
                --durations=5 -rxXs" timeout 1800 python3 -m pytest ${file} 2>&1 | tee -a ${log_per_file}
            else
                TEST_SRCDIR="$PWD"  PYTEST_ADDOPTS=" -v \
                --continue-on-collection-errors --junitxml=${tf_file_report} -o junit_suite_name=${module_name} ${run_xfail} -n=${max_worker_count} \
                --max-worker-restart=64 --durations=5 -rxXs" timeout 1500 python3 -m pytest ${file} 2>&1 | tee -a ${log_per_file}
            fi
            # Below catches the pytest execution return value
            retVal=${PIPESTATUS[0]}
            if [ $retVal != 0 ]; then
                echo -e "\npytest command return value is ${retVal} ; Incrementing overall FAILED count" | tee -a ${log_per_file}
                fail_cnt=$((fail_cnt+1))
                suite_status=1
            fi
            cat $log_per_file >> $log_file
            count=`expr $count + 1`
        fi
        echo -e "================================================================================" | tee -a ${log_file}
    done
    echo -e "Number of junit xml generated is $(find tmp/ -type f -name "google_tf_test_report_${module_name}_*_${__tf_dir}.xml" | wc -l)" | tee -a ${log_file}
    junitparser merge --glob "tmp/google_tf_test_report_${module_name}_*_${__tf_dir}.xml" $tf_report
    # Remove temporary folder
    rm -rf tmp/
    suite_end_time=`date +%s`
    suite_run_time=$((suite_end_time-suite_start_time))
    echo -e "Suite Execution Time -> ${suite_run_time} secs" | tee -a ${log_file}
    if [ ${suite_status} != 0 ]; then
        echo -e "Suite overall status is 1 FAIL" | tee -a ${log_file}
    else
        echo -e "Suite overall status is 0 PASS" | tee -a ${log_file}
    fi
    cd ..
}

run_tf_sub_dir_tests()
{
    set_tf_sub_dir_flags

    # Below will create the log file and following that, tee -a is used
    echo -e "Executing $__tf_sub_dir from $__tf_dir folder" | tee ${log_file}
    if [[ ! -z "$forked" ]]; then
        echo -e "Executing suite with forked flag enabled" | tee -a ${log_file}
    fi

    rm -f $tf_report
    cd $__tf_dir
    # Remove temporary folder if exists
    rm -rf tmp/

    # global timeout set to 1800 secs as below execution is now mainly used for running xfailed tests with --forked flag
    TEST_SRCDIR="$PWD"  PYTEST_ADDOPTS=" -v ${forked} \
    --continue-on-collection-errors --junitxml=${tf_report} -o junit_suite_name=${module_name} ${run_xfail} \
    --durations=5 -rxXs" timeout 1800 python3 -m pytest ${__tf_sub_dir} 2>&1 | tee -a ${log_file}

    # Below catches the pytest execution return value
    retVal=${PIPESTATUS[0]}
    # Ignore retVal of tf_program (returning 1) as there is only one file which is ignored because of import error
    if [[ *"$tf_sub_dir"* != *"tensorflow/python/tf_program"* ]]; then
        if [ $retVal != 0 ]; then
            echo -e "\npytest command return value is ${retVal} ; Incrementing overall FAILED count" | tee -a ${log_file}
            fail_cnt=$((fail_cnt+1))
            suite_status=1
        fi
    fi
    suite_end_time=`date +%s`
    suite_run_time=$((suite_end_time-suite_start_time))
    echo -e "Suite Execution Time -> ${suite_run_time} secs" | tee -a ${log_file}
    if [ ${suite_status} != 0 ]; then
        echo -e "Suite overall status is 1 FAIL" | tee -a ${log_file}
    else
        echo -e "Suite overall status is 0 PASS" | tee -a ${log_file}
    fi
    echo -e "========================================================================================================="
    cd ..
}

dump_test_count()
{
    echo "Calculating dumps"
    for tf_sub_dir in "${tf_sub_dirs[@]}"
    do
        set_tf_sub_dir_flags
        cd $__tf_dir
#        if [[ *"$tf_sub_dir"* == *"tensorflow/python/saved_model/"* ]] || \
#            [[ *"$tf_sub_dir"* == *"tensorflow/python/tools/"* ]] || \
#            [[ *"$tf_sub_dir"* == *"tensorflow/python/keras/"* ]] || \
##            [[ *"$tf_sub_dir"* == *"tensorflow/python/data/"* ]]; then
#           rm -f $log_file
#            echo "Number of files is $(find ${__tf_sub_dir} -type f -name "*_test.py" | wc -l)"
#            count=1
#            module_tests=0
#            total_file_count=$(find ${__tf_sub_dir} -type f -name "*_test.py" | wc -l)
#            for file in $( find ${__tf_sub_dir} -type f -name "*_test.py")
#            do
#                echo -e "Executing file: $file  $count/$total_file_count"
#                file_report_name=${file//'/'/'_'}
#                log_per_file="../google_tf_test_log_${module_name}_${file_report_name}_${__tf_dir}.log"
#                test_count_per_file=$(TEST_SRCDIR="$PWD" PYTEST_ADDOPTS="-v --collectonly --without_hpu" python3 -m pytest ${file} 2>&1 | tee ${log_per_file} | grep -c "<TestCaseFunction")
#                cat $log_per_file >> $log_file
#                rm -f $log_per_file
#                count=`expr $count + 1`
#                module_tests=`expr $module_tests + $test_count_per_file`
#            done
#            echo "Total tests of $module_name : $module_tests"
#        else
            test_count=$(TEST_SRCDIR="$PWD"  PYTEST_ADDOPTS="-v --collectonly --without_hpu" python3 -m pytest ${__tf_sub_dir} 2>&1 | tee ${log_file} | grep -c "<TestCaseFunction")
            echo "Total tests of $module_name : $test_count"
#        fi
        cd ..
    done
}

run_tests()
{
    scriptname="run_tests"
    run_deselected_tests=0
    get_test_count=0
    tf_sub_dirs=()

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -rd  | --run-deselected )
            run_deselected_tests=1
            ;;
        -rh  | --run-hang )
            run_deselected_tests=2
            ;;
        -rj  | --run-jira )
            run_deselected_tests=3
            shift
            jira_id="$1"
            ;;
        -dtc | --dump-test-count )
            get_test_count=1
            ;;
        -h  | --help )
            usage $scriptname
            return 0
            ;;
        *)
            # Anything else passed in command line will be treated as tf_sub_dir(optional);
            # As an example can be v2.4.1/tensorflow/python/kernel_tests or v2.4.1/tensorflow/python/training
            tf_sub_dirs+=($1)
            ;;
        esac
        shift
    done

    echo "Detecting tensorflow version..."
    tf_ver=$(python3 -c 'import tensorflow as tf; print(tf.__version__)')
    echo "$tf_ver is the current tensorflow version"
    # If no tf_sub_dir is passed then execute test from below tf_dir
    tf_dir="v$tf_ver"

    if [ ${run_deselected_tests} -eq 0 ] ; then
        run_xfail=""
    elif [ ${run_deselected_tests} -eq 1 ] ; then
        run_xfail=" --run_xfail_only"
    elif [ ${run_deselected_tests} -eq 2 ] ; then
        run_xfail=" --run_hang_tests"
    elif [ ${run_deselected_tests} -eq 3 ] ; then
        run_xfail=" --run_jira ${jira_id}"
    fi

    # Detect number of ASICs in the VM
    asic_count=$(lspci | grep -c acc)
    echo "Total number of ASICs detected are : $asic_count"

    # If running with Simulators
    if [ $asic_count == 0 ]; then
        # Count simulators and assigned to asic_count
        asic_count=$(ps -ae |grep -c simulator)
        echo "Total number of Simulators running are : $asic_count"
    fi

    # Temporary fix
    # Keep 1 or 2 spare asics/simulators so that in case they crash then xdist worker(after it restarts) can schedule on the spare ones
    #if [ ${asic_count} -gt 4 ] ; then
    #    asic_count=`expr $asic_count - 2`
    #elif [ ${asic_count} == 4 ] ; then
    #    asic_count=`expr $asic_count - 1`
    #fi

    # Remove all pycache folders
    find . -name "__pycache__"|xargs rm -rf

    # If tf sub dir is empty
    if [ -z "$tf_sub_dirs" ]; then
        tf_sub_dirs=($tf_dir/tensorflow/python/*/) # this gives $tf_dir/tensorflow/python/ops/, $tf_dir/tensorflow/python/kernel_tests/ etc
    fi

    if [ $get_test_count == 1 ]; then
        dump_test_count
        return 0
    fi

    # TF_CPU_FALLBACK disabled for TF suite execution
    echo "Setting TF_CPU_RUNTIME_FALLBACK to forbid for TF suite execution"
    export TF_CPU_RUNTIME_FALLBACK=forbid

    set_tf_dir_flags

    for tf_sub_dir in "${tf_sub_dirs[@]}"
    do
        #For reference
        #export TF_HABANA_ALLOW_LEGACY_VARIABLES_ON_CPU=true
        #unset TF_HABANA_ALLOW_LEGACY_VARIABLES_ON_CPU
        if [[ ! -z "$run_xfail" ]]; then
            run_tf_sub_dir_tests
        else
            run_tf_sub_dir_tests_per_file_experimental
        fi

        #Parse the xml reports and generate a csv file
        python3 parse_google_tf_test_xml.py $PWD/xml_test_results/
    done

    # unset the environment variable
    unset TF_CPU_RUNTIME_FALLBACK
}

check_active_device_state()
{
    running_on_asic=$(lspci | grep -c acc)
    #Considering names as HL-XXXXX
    asic_count=0

    if [ $running_on_asic != 0 ]; then
        asic_count=$(hl-smi -Q name -f csv | grep -c HL)
    fi

    if [ $asic_count == 0 ]; then
        simu_count=$(ps -ae |grep -c simulator)
        #echo -e "Running in Simulator Mode, Total number of active simulators : ${simu_count}"  | tee -a ${log_file}
        num_active_devices=$simu_count
    else
        #echo "Current System State :" | tee -a ${log_file}
        #hl-smi -Q index,serial,uuid,power.draw,name,bus_id,module_id,memory.total,memory.free,memory.used,driver_version,ecc.errors.uncorrected.aggregate.total,ecc.errors.uncorrected.volatile.total,temperature.aip,utilization.aip,timestamp -f csv,nounits
        #echo -e "Total number of active ASICs : ${asic_count}" | tee -a ${log_file}
        num_active_devices=$asic_count
    fi

    if [ $max_worker_count != $num_active_devices ]; then
        max_worker_count=$num_active_devices
        echo -e "Parallel worker count is set to : ${max_worker_count}" | tee -a ${log_file}
    fi
}

fail_cnt=0
declare -A file_pid_array

start_time=`date +%s`
run_tests $@
end_time=`date +%s`
run_time=$((end_time-start_time))
echo "Script Execution Time -> ${run_time} secs"

if [ ${fail_cnt} != 0 ]; then
    echo "Exiting the script with code 1 FAIL"
    exit 1
else
    echo "Exiting the script with code 0 PASS"
    exit 0
fi
