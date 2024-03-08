from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import ServiceContext, download_loader, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import SimpleDirectoryReader
from pyvis.network import Network
import os

# def get_folder_paths(directory = "githubCode"):
#     folder_paths = []
#     for root, dirs, files in os.walk(directory):
#         if '.git' in dirs:
#             # Skip the directory if a .git folder is found
#             dirs.remove('.git') 
#         for dir_name in dirs:
#             folder_paths.append(os.path.join(root, dir_name))
#     return folder_paths

# directory_paths = get_folder_paths()
# print("directory_paths: ", directory_paths)

# files = []

# with open("Code/Code.txt", "r+", encoding='utf-8') as output:
#     output.truncate(0)
#     output.seek(0)
#     for directory_path in directory_paths:
#         for filename in os.listdir(directory_path):
#             if filename.endswith((".py",".js", ".ts")):
#                 filepath = os.path.join(directory_path, filename)
#                 with open(filepath, "r", encoding='utf-8') as file:
#                     files.append(filepath)
#                     code = file.read()
#                     output.write(f"Filepath: {filepath}:\n\n")
#                     output.write(code + "\n\n")

loader = SimpleDirectoryReader("Code")
documents = loader.load_data()

# print(documents)
# reader = download_loader("WikipediaReader")
# loader = reader()
# documents = loader.load_data(['Tesla Cybertruck'])

token = "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"

llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-Instruct-v0.2", token = token)
embed_model = LangchainEmbedding(HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-Instruct-v0.2", token = token))

Service_context = ServiceContext.from_defaults(chunk_size=2000, chunk_overlap=200, llm=llm, embed_model=embed_model)
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = KnowledgeGraphIndex.from_documents(documents=documents, service_context=Service_context, storage_context=storage_context)

graph = index.get_networkx_graph()

# Use NetworkX for further analysis or visualization
import matplotlib.pyplot as plt
import networkx as nx

plt.figure(figsize=(10, 6))
nx.draw(graph, with_labels=True, font_weight='bold')
plt.show()

query_engine = index.as_query_engine()

response = query_engine.query("What is Backtracking?")

print(response)