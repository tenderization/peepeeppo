from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="peepeeppocpp",
      ext_modules=[
          cpp_extension.CppExtension(
            "peepeeppocpp",
            ["util.cpp"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
)
