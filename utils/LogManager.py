#!/usr/bin/env python
# coding: utf-8


class LogManager(object):
    def __init__(self, log_file_name): 
        self.logger=get_logger(log_file_name)
        
    def save_log(self, string):
        self.logger.info(string)
        
        
#=============================================================
"""
Function: create a logger to save logs.
"""
import logging
def get_logger(log_file_name):
    logger=logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    handler=logging.FileHandler(log_file_name)
    handler.setLevel(logging.DEBUG)
    formatter=logging.Formatter('%(asctime)s: %(name)s (%(levelname)s)  %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger