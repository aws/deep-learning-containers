from setuptools import Extension, setup

setup(name="hello-ext", version="0.1.0", ext_modules=[Extension("hello_ext", ["hello.c"])])
