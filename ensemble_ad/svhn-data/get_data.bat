@echo off
if exist svhn-py rd /Q /S svhn-py
mkdir svhn-py
cd svhn-py
curl -o train_32x32.mat http://ufldl.stanford.edu/housenumbers/train_32x32.mat
curl -o test_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat