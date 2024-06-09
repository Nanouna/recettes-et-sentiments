from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Filtrer les commentaires et les lignes vides
        requirements = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        return requirements


setup(
    name='recettes_et_sentiments',
    version='0.1',
    packages=find_packages(),
    package_dir={'': '.'},
    install_requires=parse_requirements('requirements.txt'),
)
