from setuptools import find_packages, setup
from typing import List

PROJECT_NAME  = "gst"
AUTHOR_NAME = "Atiqur Rahman"
AUTHOR_EMAIL = "atikurrahman209@gmail.com"



def get_requiremnts() -> List[str]:
    """
    This function will return the list of string
    """
    requirements_list:List[str] =[]
    try:
        with open("requirements.txt") as f:
            list = f.read()
            requirements_list = list.splitlines()
            requirements_list.remove("-e .")
    except FileNotFoundError as e:
        raise e
    return requirements_list



setup(
    name=PROJECT_NAME,
    version="1.0.0",
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires=get_requiremnts(),
)