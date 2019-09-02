#!/usr/bin/env bash

pushd data/source
#aws s3 folder structure unpacked/year/month/
aws s3 cp s3://bus350-data/unpacked/2019/05/ . --recursive --include '*'
unzip '*.zip'
find * -type f -name '*csv' -exec mv {} . \;
rmdir temp_*
rm *zip
ls
popd
