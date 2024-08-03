import csv
import os
from PyPDF2 import PdfReader

def txt_to_log(txt_file, log_file):
    with open(txt_file, 'r', encoding='utf-8') as txt, open(log_file, 'w', encoding='utf-8') as log:
        for line in txt:
            log.write(line)

def csv_to_log(csv_file, log_file):
    with open(csv_file, 'r', encoding='utf-8') as csvf, open(log_file, 'w', encoding='utf-8') as log:
        reader = csv.reader(csvf)
        for row in reader:
            log.write(','.join(row) + '\n')

def pdf_to_log(pdf_file, log_file):
    with open(pdf_file, 'rb') as pdff, open(log_file, 'w', encoding='utf-8') as log:
        reader = PdfReader(pdff)
        for page in range(len(reader.pages)):
            text = reader.pages[page].extract_text()
            log.write(text)

def convert_to_log(file_path, output_dir):
    file_name, file_ext = os.path.splitext(file_path)
    output_file = os.path.join(output_dir, os.path.basename(file_name) + '.log')
    if file_ext == '.txt':
        txt_to_log(file_path, output_file)
    elif file_ext == '.csv':
        csv_to_log(file_path, output_file)
    elif file_ext == '.pdf':
        pdf_to_log(file_path, output_file)
    elif file_ext == '.log':
        return file_path
    else:
        print(f'Unsupported file format: {file_ext}')

def convert_directory(input_dir, output_dir):
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path):
            convert_to_log(file_path, output_dir)

#convert_directory(r'C:\Users\DeLL\PycharmProjects\ui\md',r'C:\Users\DeLL\PycharmProjects\ui\md')