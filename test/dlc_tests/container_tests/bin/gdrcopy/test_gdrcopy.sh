# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

#!/usr/bin/env bash
gdrcopy_sanity > tmp_out
if [ $? != 0 ]; then
    echo "GDRCopy Sanity check failed!"
    exit 1
fi

# NOTE: A grep guard clause is added because the old GDRCopy test (which is now moved to `test_gdrcopy_dev.sh`)
# was checking an exit status of grep instead of the sanity test itself.
# We are now moving towards checking the exit status of the sanity test,
# but as a safety check we will continue to check the test output as well.
cat tmp_out | grep 'Failed: 0' &> /dev/null
if [ $? != 0 ]; then
    echo "GDRCopy Sanity check passed but failed at grep output check!"
    echo "Please examine the gdrcopy_sanity output to ensure the tests are passing properly"
    exit 1
fi

echo "GDRCopy Sanity check succeed!"
