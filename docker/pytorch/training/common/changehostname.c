#include <stdio.h>
#include <string.h>

/**
 * Copyright 2018-2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You
 * may not use this file except in compliance with the License. A copy of
 * the License is located at
 *
 *     http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific
 * language governing permissions and limitations under the License.
 */

/**
 * Modifies gethostname to return algo-1, algo-2, etc. when running on SageMaker.
 *
 * Without this gethostname() on SageMaker returns 'aws', leading NCCL/MPI to think there is only one host,
 * not realizing that it needs to use NET/Socket.
 *
 * When docker container starts we read 'current_host' value  from /opt/ml/input/config/resourceconfig.json
 * and replace PLACEHOLDER_HOSTNAME with it before compiling this code into a shared library.
 */
int gethostname(char *name, size_t len)
{
	const char *val = PLACEHOLDER_HOSTNAME;
	strncpy(name, val, len);
	return 0;
}
