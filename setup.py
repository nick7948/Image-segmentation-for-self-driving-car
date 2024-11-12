from setuptools import setup, find_packages

setup(
    name='self_driving_image_segmentation',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch', 'torchvision', 'numpy', 'matplotlib', 'pandas', 
        'opencv-python', 'imageio', 'torchsummary'
    ],
    description='Image segmentation project for self-driving car',
    author='Sanchit Srivastava',
    author_email='sanchit.8794@gmail.com',
    url='https://github.com/nick7984/Image-segmentation-for-self-driving-car',
    license='MIT',
)
