<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


# recettes-et-sentiments

_Authors_
 - [Marie Anne](https://github.com/Nanouna)
 - [Théo](https://github.com/theorosen12)
 - [Thomas](https://github.com/mansonthomas-wagon)

## Description

This is the application project of the [Datascience & IA](https://www.lewagon.com/fr/remote/data-science-course) course, batch #1672 - April/June 2024

It uses the Kaggle [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) dataset.

The purpose of the project is to apply our learnings in a full project from training of a model to deployment in GCP on Cloud Run with a web interface.

The first goal is to predict the the user review score from the recipe description, using a Machine Learning algorithm.

The second goal is to generate a brand new receipes from a list of key ingredients.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Setup

### requirements

 - python 3.10.6
 - Tested on Windows(11) & MacOSX (intel/sonoma)

 ### steps

- git clone git@github.com:Nanouna/recettes-et-sentiments.git
- cd recettes-et-sentiments
- pyenv virtualenv recettes-et-sentiments
- pyenv local recettes-et-sentiments
- pip install --upgrade pip
- cd recettes_et_sentiments
- pip install -r requirements.txt
- pip install .
- make all_tests
- cd docker
- docker build...
- setup GCP project
- deploy on cloud run

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Key Technologies used in this project

- Python
  - NLTK
  - scikitlearn
  - Pandas
- Google Cloud Platform (GCP)
  - Cloud Run
  - IAM
  -

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Nanouna/recettes-et-sentiments.svg?style=for-the-badge
[contributors-url]: https://github.com/Nanouna/recettes-et-sentiments/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Nanouna/recettes-et-sentiments.svg?style=for-the-badge
[forks-url]: https://github.com/Nanouna/recettes-et-sentiments/network/members
[stars-shield]: https://img.shields.io/github/stars/Nanouna/recettes-et-sentiments.svg?style=for-the-badge
[stars-url]: https://github.com/Nanouna/recettes-et-sentiments/stargazers
[issues-shield]: https://img.shields.io/github/issues/Nanouna/recettes-et-sentiments.svg?style=for-the-badge
[issues-url]: https://github.comNanouna/recettes-et-sentiments/issues
[license-shield]: https://img.shields.io/github/license/Nanouna/recettes-et-sentiments.svg?style=for-the-badge
[license-url]: https://github.com/Nanouna/recettes-et-sentiments/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/mansonthomas
