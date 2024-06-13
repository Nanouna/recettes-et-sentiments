from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Filtrer les commentaires et les lignes vides
        return [line.strip() for line in lines if line.strip() and not line.startswith('#')]


setup(
    name='recettes_et_sentiments',
    version='0.1',
    license="MIT",
    author="Marie-Anne André - https://github.com/Nanouna; Théo Rosen - https://github.com/theorosen12 ;Thomas Manson - https://github.com/dev-mansonthomas",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
)
