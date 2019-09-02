#!/usr/bin/env bash

pushd data/source
aws s3 cp s3:/bus350-data/unpacked/2019/06/ . --recursive --include '*'
unzip '*.zip'
find * -type f -name '*csv' -exec mv {} . \;
rmdir temp_*
rm *zip
ls
popd
