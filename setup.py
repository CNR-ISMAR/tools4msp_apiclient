import os
from distutils.core import setup


def read(*rnames):
    return open(os.path.join(os.path.dirname(__file__), *rnames)).read()

setup(
    name="tools4msp_apiclient",
    version="0.2",
    author="Stefano Menegon",
    author_email="stefano.menegon@cnr.it",
    description="Tools 4 MSP CaseStudy utilities",
    long_description=(read('README.md')),
    # Full list of classifiers can be found at:
    # http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
    ],
    license="GPL3",
    keywords="Maritime Spatial Planning",
    url='http://data.adriplan.eu',
    packages=['tools4msp_apiclient',],
    include_package_data=True,
    zip_safe=False,
    install_requires=['rectifiedgrid',
                      'matplotlib',
                      'coreapi',
                      'numpy',
                      'xarray',
                      ],
)
