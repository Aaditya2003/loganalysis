#!/usr/bin/env python
import sys
sys.path.append('../')
from logparser.logparser.AEL import AEL
# from file_converter import convert_directory

#convert_directory(r'C:\Users\DeLL\PycharmProjects\py 11.6\logparser\logs\raw_logs',r'C:\Users\DeLL\PycharmProjects\py 11.6\logparser\logs\HDFS')

input_dir     = r'C:\Users\DeLL\Desktop\HDFS_v1' # The input directory of log file
output_dir    = r'C:\Users\DeLL\PycharmProjects\py 11.6\swhsw/hdfs' # The output directory of parsing results
log_file      = 'HDFS.log' # The input log file name
log_format    = '<Date> <Time> <Pid> <Level> <Component>: <Content>' # HDFS log format
minEventCount = 2 # The minimum number of events in a bin
merge_percent = 0.5 # The percentage of different tokens 
regex         = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'] # Regular expression list for optional preprocessing (default: [])

parser = AEL.LogParser(input_dir, output_dir, log_format, rex=regex, 
                       minEventCount=minEventCount, merge_percent=merge_percent)
parser.parse(log_file)

