#!/usr/bin/env python

from distutils.core import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        import pytest
        pytest.main(self.test_args)

setup(name='holo-nets',
      version='0.0',
      description='Simple wrapper around Holoviews for training neural '
      'networks interactively and monitoring channels.',
      author='Gavin Gray',
      author_email='g.d.b.gray@sms.ed.ac.uk',
      packages=['holonets'],
      tests_require=['pytest'],
      install_requires=[
          'pytest',
          'holoviews'
          
      ],
      cmdclass={'test': PyTest},
)
