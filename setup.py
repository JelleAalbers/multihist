try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').read().splitlines()
test_requirements = requirements + ['pandas']

setup(name='multihist',
      version='0.6.3',
      description='Convenience wrappers around numpy histograms',
      long_description=readme + '\n\n' + history,
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/jelleaalbers/multihist',
      license='MIT',
      py_modules=['multihist'],
      install_requires=requirements,
      keywords='multihist,histogram',
      test_suite='tests',
      tests_require=test_requirements,
      classifiers=['Intended Audience :: Developers',
                   'Development Status :: 3 - Alpha',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3'],
      zip_safe=False)
