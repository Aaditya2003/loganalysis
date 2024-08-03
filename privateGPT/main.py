import csv
import subprocess

# !/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse



from loglizer.demo.predictor import predictor
from loglizer.demo.trainer import trainer
import time


python36_path = r'C:\Users\DeLL\AppData\Local\Programs\Python\Python36\python.exe'
script_path = r"C:\Users\DeLL\PycharmProjects\py 11.6\logparser\demo\AEL_demo.py"
# call the script using the Python 3.6 interpreter
subprocess.run([python36_path, script_path])



#trainer()
predictor()



load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS




def process_csv_in_batches(qa, column_index, batch_size=10):
    with open('input/anomal data/filtered_data.csv', 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row if present

        batch = []
        for row in csv_reader:
            if len(row) > column_index:  # Check if the column exists in the row
                event_id = row[column_index]
                batch.append(event_id)

                if len(batch) == batch_size:
                    process_batch(qa, batch)
                    batch = []

        if batch:
            process_batch(qa, batch)


def main():
    predictor()
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks,
                           verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch,
                          callbacks=callbacks, verbose=False)
        case _default:
            raise Exception(
                f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=not args.hide_source)

    column_index = 7  # Index of the event ID column in the CSV (adjust as needed)
    process_csv_in_batches(qa, column_index, batch_size=2)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='privateGPT: Ask questions to your documents without an internet connection, '
                    'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()





def process_batch(qa, batch):
    args = parse_arguments()
    combined_batch_text = ", ".join(["how to resolve error codes "] + batch)  # Include additional line before batch texts

    #print(combined_batch_text)

    # qq=input("query")

    res = qa(combined_batch_text)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']

    # answers = res['result'].split("\n")  # Split the combined result into individual answers

    # for i, answer in enumerate(answers):
    #     print(f"Batch Row {i + 1} - Answer:", answer)
    #     print("=" * 50)
    print(answer)
    return answer

# with gr.Blocks() as demo:
#     btn = gr.Button(value="Inference")
#     btn.click(main)
#     gr.Interface(fn=process_batch, inputs=None, outputs="text")
#
#
# demo.launch()



if __name__ == "__main__":
    main()




