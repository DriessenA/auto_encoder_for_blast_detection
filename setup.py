from setuptools import find_packages, setup

setup(
    name="auto_encoder_for_blast_detectrion",
    packages=find_packages(where="scr"),
    package_dir={"": "scr"},
    install_requires=["torch", "pytorch_lightning", "python-dotenv"],
    version="1.6",
    description="Training autoencoders for single cell blast detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alice Driessen",
    author_email="adr@zurich.ibm.com",
    include_package_data=True,
)
