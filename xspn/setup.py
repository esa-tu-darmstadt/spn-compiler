# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import setuptools
import os

with open("requirements.txt", "r") as rf:
    requirements = rf.readlines()

setuptools.setup(
    name="xspn",
    version="0.2.0",
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
