"""Ask a question to the notion database."""
from multiprocessing.connection import answer_challenge
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import argparse
import os
from flask import Flask

app = Flask(__name__)


@app.route("/")
def query_bot():
  parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
  parser.add_argument('question', type=str, help='The question to ask the notion DB')
  args = parser.parse_args()

# Load the LangChain.
  index = faiss.read_index("docs.index")

  with open("faiss_store.pkl", "rb") as f:
      store = pickle.load(f)

  store.index = index
  chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
  result = chain({"question": args.question})

  print(f"Answer: {result['answer']}")
  print(f"Sources: {result['sources']}")

if __name__ == "__qa__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))