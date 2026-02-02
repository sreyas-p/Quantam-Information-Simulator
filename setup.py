from setuptools import setup, find_packages

setup(
    name="aces",
    version="0.1.0",
    description="Adaptive Causal-Entropy Simulation for NISQ quantum hardware",
    author="Sreyas Prabu",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "qiskit>=0.45",
        "networkx>=2.6",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
        "jax": ["jax>=0.4", "jaxlib>=0.4"],
    },
)
