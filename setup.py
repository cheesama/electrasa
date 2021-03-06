from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="electrasa",
    version="0.2.3",
    description="ELECTRA pretrained model + RASA dataset based Fine-Tuner",
    author="Cheesama",
    install_requires=required,
    packages=find_packages(exclude=["docs", "tests", "tmp", "data"]),
    python_requires=">=3",
    zip_safe=False,
    include_package_data=True,
)
