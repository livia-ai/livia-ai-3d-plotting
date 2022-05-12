from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()
        
setup(name='livia',
      version='0.1.4',
      description='Linking Viennese Art through AI',
      long_description = readme,
      long_description_content_type = "text/markdown",
      author='Bernhard Franzl',
      author_email='bernhard.franzl@gmx.at',
      url='https://github.com/livia-ai/livia-ai-scripts',
      packages=find_packages(),
      install_requires = [
		"matplotlib",
		"nltk",
		"numpy",
		"pandas",
		"scikit-learn",
		"sentence-transformers"],
      classifiers=[
        "Programming Language :: Python :: 3"
    ],
     )

