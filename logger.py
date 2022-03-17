import logging

import coloredlogs


def config(logger: logging, logger_name: str = None, console: bool = True):
    """

    :param logger:
    :param console:
    :param logger_name:
    :return:
    """
    fmt = '%(funcName)s:%(lineno)d\n%(message)s'
    ch = None
    # console handler for print logs
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # formatter for the handlers
        formatter = logging.Formatter(fmt=fmt)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(ch)

    if logger_name is not None:
        # file handler for save logs
        # if logger_name.endswith('.log') or logger_name.endswith('.txt'):
        #     logger_name = f'{logger_name[:-3]}'
        # logger_name = f'{logger_name}{datetime.today().strftime("%Y%m%d")}.log'
        fh = logging.FileHandler(logger_name, mode='a', encoding='utf-8')
        fh.setLevel(logger.getEffectiveLevel())
        formatter = logging.Formatter(fmt='\n%(funcName)s:%(lineno)d |'
                                          ' %(asctime)s\n%(message)s')
        fh.setFormatter(formatter)
        # logger.addHandler(fh)
        if console:
            logger.handlers = [fh, ch]
        else:
            logger.addHandler(fh)

    encoded_styles = 'info=green;warning=yellow;critical=red,bold,exception=blue'
    coloredlogs.install(level=logging.INFO, fmt=fmt, logger=logger,
                        level_styles=coloredlogs.parse_encoded_styles(encoded_styles))
    return logger
