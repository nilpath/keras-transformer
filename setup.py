from setuptools import find_packages, setup


setup(
    name="keras-transformer",
    version="0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow-datasets==4.0.1",
        "tensorflow==2.3.1",
    ],
    description="A keras implementation of the transformer network \
      from the Attention is all you need paper.",
)
