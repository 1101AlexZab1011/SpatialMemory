import configparser

class EvalSectionProxy(configparser.SectionProxy):
    """
    Subclass of `configparser.SectionProxy` with an added `eval` method for evaluating configuration values as Python expressions.

    This class extends the functionality of the standard `configparser.SectionProxy` to allow for the evaluation of values stored in configuration sections as Python expressions.

    Parameters:
        sectionproxy (SectionProxy): An instance of `configparser.SectionProxy` to be wrapped by this class.

    Example:
        # Create an instance of EvalSectionProxy
        section_proxy = EvalSectionProxy(config['my_section'])

        # Evaluate a configuration value as a Python expression
        result = section_proxy.eval('my_option', globals={'x': 10}, locals={'y': 5})

    Note:
        - Be cautious when using the `eval` method, as it allows the execution of arbitrary Python code and may pose security risks if used with untrusted configuration files.

    See Also:
        - `configparser.SectionProxy`: The base class for configuration section proxies in Python's `configparser`.

    """
    def __init__(self, sectionproxy: configparser.SectionProxy):
        super().__init__(sectionproxy._parser, sectionproxy._name)

    def eval(self, *args, globals=None, locals=None, **kwargs):
        """
        Evaluate a configuration value as a Python expression.

        Args:
            *args: Positional arguments to be passed to the `SectionProxy.get` method.
            globals (dict, optional): A dictionary containing global variables for the evaluation.
            locals (dict, optional): A dictionary containing local variables for the evaluation.
            **kwargs: Additional keyword arguments to be passed to the `SectionProxy.get` method.

        Returns:
            The result of evaluating the configuration value as a Python expression.
        """
        val = self.get(*args, **kwargs)
        return eval(val, globals, locals) if val else val


class EvalConfigParser(configparser.ConfigParser):
    """
    Subclass of `configparser.ConfigParser` with an added `eval` method for evaluating configuration values as Python expressions.

    This class extends the functionality of the standard `configparser.ConfigParser` to allow for the evaluation of values stored in configuration files as Python expressions.

    Parameters:
        *args: Positional arguments to be passed to the base class `configparser.ConfigParser`.
        **kwargs: Keyword arguments to be passed to the base class `configparser.ConfigParser`.

    Example:
        # Create an instance of EvalConfigParser
        config = EvalConfigParser()

        # Add a configuration file
        config.read('my_config.ini')

        # Evaluate a configuration value as a Python expression
        result = config.eval('my_section', 'my_option', globals={'x': 10}, locals={'y': 5})

    Note:
        - Be cautious when using the `eval` method, as it allows the execution of arbitrary Python code and may pose security risks if used with untrusted configuration files.

    See Also:
        `configparser.ConfigParser`: The base class for configuration parsing in Python.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, section, option, globals=None, locals=None, **kwargs):
        """
        Evaluate a configuration value as a Python expression.

        Args:
            section (str): The name of the section in the configuration file.
            option (str): The name of the option within the specified section.
            globals (dict, optional): A dictionary containing global variables for the evaluation.
            locals (dict, optional): A dictionary containing local variables for the evaluation.
            **kwargs: Additional keyword arguments to be passed to the `ConfigParser.get` method.

        Returns:
            The result of evaluating the configuration value as a Python expression.
        """
        val = self.get(section, option, **kwargs)
        return eval(val, globals, locals) if val else val

    def __getitem__(self, key: str) -> EvalSectionProxy:
        """
        Get a section proxy for a specified section and wrap it with `EvalSectionProxy`.

        Args:
            key (str): The name of the section.

        Returns:
            EvalSectionProxy: An instance of `EvalSectionProxy` wrapped around the specified section proxy.
        """
        return EvalSectionProxy(super().__getitem__(key))


def validate_config_eval(config: configparser.ConfigParser, **kwargs):
    """
    Validate the configuration for evaluation, specifically checking for external variable sources.

    Args:
        config (configparser.ConfigParser): A configuration parser containing evaluation settings.
        **kwargs: Additional keyword arguments.

    Raises:
        ValueError: If the configuration indicates the need for external variable sources, but they are not provided.

    Example:
        config_parser = configparser.ConfigParser()
        config_parser.read('evaluation_config.ini')

        # Validating the configuration for evaluation.
        try:
            validate_config_eval(config_parser, globals=globals(), locals=locals())
        except ValueError as e:
            print(f"Configuration error: {str(e)}")
    """
    if bool(config.get('ExternalSources', 'variables')) and not any(['globals' in kwargs, 'locals' in kwargs]):
        raise ValueError(
            f'Parser requires external sources that has not been provided: '
            f'{", ".join([variable  + " " + str(value) for variable, value in config.eval("ExternalSources", "variables").items()])}'
        )
