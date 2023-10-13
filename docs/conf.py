import importlib.metadata

project = "minecraft-gen"
author = "Henry Schreiner"
copyright = f"2024, {author}"
version = release = importlib.metadata.version(project)

extensions = [
    "myst_parser",
]

source_suffix = [".rst", ".md"]

html_theme = "furo"
