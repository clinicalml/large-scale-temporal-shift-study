import logging

def set_up_logger(logger_name, 
                  log_file_path,
                  level=logging.INFO):
    '''
    Set up a logger
    @param logger_name: str, name of logger
    @param log_file_path: str, path to log file
    @param level: logging level
    @return: logger
    '''
    logger      = logging.getLogger(logger_name)
    formatter   = logging.Formatter(fmt='%(asctime)s - %(message)s',
                                    datefmt='%d-%b-%y %H:%M:%S')
    fileHandler = logging.FileHandler(log_file_path, mode='w')
    fileHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    return logger

def log_or_print(output_message,
                 logger):
    '''
    Log an INFO message or print if logger is None
    Note: this method is only used for compatibility with old scripts that do not log in model_utils
    @param output_message: str
    @param logger: logger, for INFO messages
    @return: None
    '''
    if logger is not None:
        logger.info(output_message)
    else:
        print(output_message)