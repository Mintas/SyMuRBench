pip install .'[docs]'

cd docs
make clean
cd ..

sphinx-apidoc -o docs/source symurbench
sphinx-build -M markdown docs/source/ docs/build
