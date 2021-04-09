import setuptools
import os

with open("requirements.txt", "r") as rf:
    requirements = rf.readlines()

setuptools.setup(
    name="xspn",
    version="0.1",
    author="Embedded Systems and Applications Group, TU Darmstadt",
    author_email="sommer@esa.tu-darmstadt.de",
    description="XSPN: Library bridging between SPFlow and the SPN compiler",
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
