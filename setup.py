import setuptools

# python3 -m pip install --user --upgrade setuptools wheel
# python3 setup.py sdist bdist_wheel
# python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
# python3 -m pip install kassandra

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kassandra",
    version="0.0.1",
    author="James Montgomery",
    author_email="jamesoneillmontgomery@gmail.com",
    description="Package for implementing bayesian deep learning models in python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/James-Montgomery/kassandra",
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow==1.11.0',
        'numpy>=1.15.1',
        'scipy>=1.1.0',
        'matplotlib>=2.2.3',
        'seaborn>=0.9.0',
        'nose>=1.3.7',
        'coverage==4.0.0',
        'pylint>=1.8.2'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
