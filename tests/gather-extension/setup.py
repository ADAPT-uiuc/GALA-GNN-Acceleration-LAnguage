from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='gather_cpp',
    ext_modules=[
        CppExtension('gather_cpp',
                     ['gather.cpp'],
                     extra_compile_args=['-march=native','-fopenmp','-O3'],
                     extra_link_args=['-lgomp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })