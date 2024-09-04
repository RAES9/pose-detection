# setup.py
from setuptools import setup, find_packages

setup(
    name='pose_detection',
    version='0.1.2',
    description='A pose detection and validation library using MediaPipe',
    author='RAES9',
    author_email='erivasdeveloper@gmail.com',
    packages=find_packages(),
    install_requires=[
        'mediapipe',
        'opencv-python',
        'numpy',
    ],
)