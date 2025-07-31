from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, meta, select_autoescape
from jinja2.environment import Template


def setup_jinja_environment(searchpath: str | Path) -> Environment:
    """Set up Jinja2 environment with file system loader and autoescaping.

    Parameters
    ----------
    searchpath : str
        Path to the directory containing template files

    Returns
    -------
    Environment
        Configured Jinja2 environment instance with FileSystemLoader
        and autoescaping enabled
    """
    template_loader: FileSystemLoader = FileSystemLoader(searchpath=searchpath)
    template_env: Environment = Environment(loader=template_loader, autoescape=select_autoescape())
    return template_env


def get_required_template_variables(env: Environment, template_file: str) -> list[str]:
    """
    Extract undeclared variables from a Jinja2 template file.

    Parameters
    ----------
    env : Environment
        The Jinja2 environment instance.
    template_file : str
        The name or path of the template file.

    Returns
    -------
    list[str]
        A list of undeclared variable names found in the template.
    """
    template_src = env.loader.get_source(env, template_file)  # type: ignore
    parsed_content: Template = env.parse(template_src)  # type: ignore
    return meta.find_undeclared_variables(parsed_content)  # type: ignore


def load_and_render_template(
    env: Environment, template_file: str, context: dict[str, Any] | None = None
) -> str:
    """
    Load and render a Jinja2 template with provided context.

    Parameters
    ----------
    env : Environment
        The Jinja2 environment instance.
    template_file : str
        The name or path of the template file.
    context : dict[str, Any], optional
        Dictionary containing variables to be rendered in the template. Defaults to None.

    Returns
    -------
    str
        The rendered template as a string.
    """
    template: Template = env.get_template(template_file)
    if context is None:
        context = {}
    return template.render(**context)
