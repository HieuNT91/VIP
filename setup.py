from setuptools import setup, find_packages
import subprocess


subprocess.run(["git", "submodule", "update", "--init", "third_party/cutlass"])
with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name="VIP",
    description='A adaptive rollout allocation.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    version="0.0.1",
    python_requires='>=3.10',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)