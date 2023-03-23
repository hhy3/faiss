mkdir build
cd build
cmake .. -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_GPU=ON -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2
make -j8
cd faiss/python
rm -rf faiss
python setup.py bdist_wheel
pip uninstall faiss -y
pip install dist/faiss-1.7.3-py3-none-any.whl
