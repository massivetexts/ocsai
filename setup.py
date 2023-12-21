from setuptools import setup, find_packages

setup(
    name='Ocsai',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'pingouin', 'pandas', 'pyyaml', 'pyreadstat'
    ],
    # metadata to display on PyPI
    author="Peter Organisciak",
    author_email="peter.organisciak@du.edu",
    description="Tools for training and using the Ocsai system for automated originality scoring.",
    keywords=[
        'automated originality scoring',
        'machine learning',
        'natural language processing',
        'educational psychology',
        'research methods',
        'education'
    ],
    url="https://www.github.com/massivetexts/ocsai",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Education :: Testing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
    project_urls={
        'Website': 'https://openscoring.du.edu',
    },
    # load long description from README.md
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # other categories
    license='MIT',
    platforms=['any'],
    include_package_data=True,
    zip_safe=True
)