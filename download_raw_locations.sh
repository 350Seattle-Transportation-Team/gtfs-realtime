#!/usr/bin/env bash

pushd data/source
aws s3 cp s3://malnor.seattle.bustime/unpacked/2018/01/ . --recursive --include '*'
unzip '*.zip'
find * -type f -name '*csv' -exec mv {} . \;
rmdir temp_*
rm *zip
ls
popd
