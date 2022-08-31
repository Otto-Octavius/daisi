# Image Enhancement in Low Light Conditions

Based on Chongyi Li and team's paper on Zero-Reference Deep Curve Estimation, this Daisi is an improved version of their deep network for high-res images and adaptability towards diverse lighting conditions.

## Requirements

Though the Daisi App doesn't require any libraries, the API calls from notebooks/Py-scripts requires Pydaisi and PIL.

PyDaisi package to connect the corresponding Daisi whereas PIL is the Python Imaging Library by Fredrik Lundh and Contributors.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install PIL

```bash
pip install pydaisi
```

```bash
pip install PIL
```

## Calling
Get the path of an Image in the directory of your virtual/Conda environment and use it in place of *PATH TO YOUR IMAGE*

```python
import pydaisi as pyd
from PIL import Image
i = Image.open('PATH TO YOUR IMAGE')
image_enhancement = pyd.Daisi("sam-dj/Image Enhancement in Low Light Conditions")
```

## Passing and Rendering
We simply pass the image to the Daisi and finally render the result

```python
result = image_enhancement_in_low_light_conditions.enhance(i).value
result.show()
```


## Running the Daisi App

As mentioned earlier, this can be automated just by [Running the Daisi App](https://app.daisi.io/daisies/sam-dj/Image%20Enhancement%20in%20Low%20Light%20Conditions/info)

## References
The paper proposed by Chongyi Li and team on [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://arxiv.org/pdf/2001.06826.pdf)

My notebook which attempts to implement these ideas [Sam DJ's Notebook](https://colab.research.google.com/drive/1SBAbj4DFZSijdYFkHF9uIcdPtryOOeBH?usp=sharing)

This model was trained using LOL dataset, i.e. LOw Light paired dataset which can be found on this [GDrive](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view)

The official implementation of [Zero-DCE in Pytorch](https://github.com/Li-Chongyi/Zero-DCE) by Chongyi Li.

Tensorflow tutorial of Zero-DCE on [keras.io](https://keras.io/examples/vision/zero_dce/)
