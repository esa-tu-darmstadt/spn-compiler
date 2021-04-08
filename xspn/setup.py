import setuptools
import os

with open("requirements.txt", "r") as rf:
    requirements = rf.readlines()

readme = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "README.md")
with open(readme, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xspn",
    version="0.1",
    author="Embedded Systems and Applications Group, TU Darmstadt",
    author_email="sommer@esa.tu-darmstadt.de",
    description="XSPN: Library bridging between SPFlow and the SPN compiler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/esa-tu-darmstadt/spn-compiler",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3"
    ],
    packages=setuptools.find_packages(include=["xspn", "xspn.*"]),
    package_data={"xspn.serialization.binary" : ["capnproto/*.capnp"]},
    install_requires=requirements,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="test",
)