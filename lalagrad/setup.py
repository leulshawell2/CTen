from setuptools import setup, Extension

cten = Extension(
    'CTen',                     
    sources=['./_CTen/init.c'],
    libraries=['cten'],
)

setup(
    name='CTen',
    version='1.0',
    description='The CPython extention for CTen',
    ext_modules=[cten]
)