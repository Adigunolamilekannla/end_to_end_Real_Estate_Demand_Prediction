"""
The setup.py file is an essential part of packaging and distributing Python projects.
It is used by setuptools to define the configuration of your project, such as metadata,
dependencies, and more.
"""

from setuptools import setup, find_packages
from typing import List


def get_requirements() -> List[str]:
    """
    This function reads requirements.txt and returns a list of requirements.
    It ignores '-e .' and empty lines.
    """
    requirements: List[str] = []
    try:
        with open("requirements.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != "-e .":
                    requirements.append(requirement)
    except FileNotFoundError:
        print("⚠️ requirements.txt file not found")
    return requirements


setup(
    name="realestateprediction",
    version="0.0.1",
    author="Leksman",
    author_email="adigunolamilekan817@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)
