import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='AdapTFT-alexberrian',
     version='0.0.0',
     description='Adaptive time-frequency transformations and reassignment transforms',
     long_description=long_description,
     long_description_content_type="text/markdown",
     author='Alex Berrian',
     author_email='ajberrian@ucdavis.edu',
     url='https://www.github.com/alexberrian/AdapTFT/',
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD 3-Clause \"New\" or \"Revised\" License",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.7',
)