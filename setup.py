from setuptools import setup, find_packages

setup(
  name = 'BS-RoFormer',
  packages = find_packages(exclude=[]),
  version = '0.6.0',
  license='MIT',
  description = 'BS-RoFormer - Band-Split Rotary Transformer for SOTA Music Source Separation',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/BS-RoFormer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'music source separation'
  ],
  install_requires=[
    'beartype',
    'einops>=0.8.0',
    'hyper-connections>=0.1.8',
    'librosa',
    'rotary-embedding-torch>=0.3.6',
    'torch>=2.0',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
