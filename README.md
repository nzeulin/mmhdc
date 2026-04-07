# MM-HDC: Maximum-Margin Hyperdimensional Computing

This repository provides an implementation of the multi-class maximum-margin hyperdimensional computing (MM-HDC) classifier that adopts the optimization problem formulation of multi-class Weston-Watkins SVM to HDC.

The algorithm is implemented in the `MultiMMHDC` class, which can be imported as:

```python
from mmhdc import MultiMMHDC
from mmhdc.utils import HDTransform
```

You can run the MNIST example from the GitHub repo as follows:

```bash
git clone https://github.com/nzeulin/mmhdc.git && cd mmhdc
python -m pip install --upgrade pip && python -m pip install mmhdc
python -m pip install -r example/requirements.txt
python example/example.py --config example/mnist_config.py
```

C++ backend is highly recommended to use, as it can significantly accelerate the model training. In this case, use `python -m pip install mmhdc[cpp]`. 

**NOTE:** If you don't have `gcc` installed in your system (required for compiling C++ module), you can install it as a Conda package into your venv: `conda install -c conda-forge gxx_linux-64`.

### Current features
- C++ backend (`libtorch`) to enable fast MM-HDC training.
- Support of floating-point prototypes and hypervectors.
 
### Citation

If you use this repository in your research, please cite it as software (paper in progress):

```bibtex
@software{zeulin_2026_mm_hdc,
	author = {Zeulin, Nikita},
	title = {MM-HDC: Maximum-Margin Hyperdimensional Computing},
	year = {2026},
	url = {https://github.com/nzeulin/mm-hdc}
}
```
