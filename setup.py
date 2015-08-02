try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').read().splitlines()
test_requirements = requirements + ['flake8']

setup(name='multhist',
      version='0.1',
      description='Convenience wrappers around numpy histograms',
      long_description=readme + '\n\n' + history,
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/jelleaalbers/multihist',
      license='dunno',
      packages=['multihist'],
      package_dir={'multihist': 'multihist'},
      install_requires=requirements,
      keywords='multihist',
      test_suite='tests',
      tests_require=test_requirements,
      zip_safe=False)
