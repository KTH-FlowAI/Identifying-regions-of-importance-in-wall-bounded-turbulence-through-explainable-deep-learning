#!/bin/bash
cd ./data_zip
cat ./data.zip* > ../data.zip
cd ../
unzip data.zip
