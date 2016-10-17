from setuptools import setup, find_packages

module_name = "ExcaliburCalibration-DAWN"

setup(
    name=module_name,
    version="0-1",
    description='Scripts to calibrate Excalibur-RX detectors in DAWN',
    url='https://github.com/dls_controls/ExcaliburCalibration-DAWN',
    author='Gary Yendell',
    author_email='gary.yendell@diamond.ac.uk',
    keywords='',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
    ],
    license='APACHE',
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=[
        'nose>=1.3.0',
        'mock'
    ],
    zip_safe=False,
)
