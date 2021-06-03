import setuptools
import cemake

setuptools.setup(
    ext_modules = [cemake.CMakeExtension("ganondorf.core")],
    cmdclass = {'build_ext': cemake.cmake_build_ext}
    )
