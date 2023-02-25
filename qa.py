"""Ask a question to the notion database."""
from multiprocessing.connection import answer_challenge
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import argparse
import os
from twilio.rest import Client

# Set environment variables for your credentials
# Read more at http://twil.io/secure
account_sid = "AC00abd0c8ce404bedd562c6222e19e091"
auth_token = "b04b2ee4345fbafcb788d7ab5c719306"
client = Client(account_sid, auth_token)


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

message = client.messages.create(
  body=result['answer'],
  from_="+18882577836",
  to="+13472699868"
)

print(message.sid)
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
