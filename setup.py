from setuptools import setup, find_packages

# Read the requirements from requirements.txt
import yaml
from setuptools import setup, find_packages

def parse_env_yaml(path="environment.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        env = yaml.safe_load(f)

    deps = []
    for dep in env.get("dependencies", []):
        if isinstance(dep, str):
            continue
        if isinstance(dep, dict) and "pip" in dep:
            deps.extend(dep["pip"])
    return deps

# 从 environment.yml 里提取 pip 依赖
requirements = parse_env_yaml("environment.yaml")

setup(
    name='knowcol',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='Hongkuan Zhou',
    author_email='hongkuan.zhou@bosch.com',
    description='A short description of your project',
    url='https://github.com/yourusername/yourproject',
)