import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='microtubule',
    version='0.0.1',
    author='group 17, bebi103a',
    author_email='sbuse@caltech.edu',
    description='Data analysis of microtubule catastrophe',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    install_requires=["numpy","pandas", "bokeh.io","bokeh.plotting","math","numba","scipy.stats","scipy.optimize",
                     "iqplot","warnings","os","sys","subprocess","bebi103"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
