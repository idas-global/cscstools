from setuptools import find_packages, setup

setup(
    name='cscstools',
    packages=find_packages(),
    version='0.1.0',
    description='CSCS Data science tools',
    install_requires=['cmapy==0.6.6',
                      'numpy==1.19.5',
                      'matplotlib==3.4.3',
                      'opencv-python==4.5.4.60',
                      'pandas==1.3.3',
                      'scikit-image==0.18.3',
                      'tqdm==4.47.0',
                      'scikit-learn==0.23.1',
                      'scipy==1.5.0',
                      'imutils==0.5.4'
                      ],
    author='Chris Thomas',
    license='CSCS',
)