#
#    Copyright 2023 Artem Rozumenko

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Any
from langchain.schema import Document
from analysta_index.retrievers.AnalystaRetriever import AnalystaRetriever
from analysta_index.interfaces.llm_processor import add_documents
from analysta_index.interfaces.splitters import Splitter
from .defaults import weights, splitter_name, splitter_params

class Memory:
    def __init__(self, vectorstore: Any, embedding: Any):
        self.vectorstore = vectorstore
        self.embedding = embedding
        self.top_k = 10
        self.splitter = Splitter(**splitter_params)
    
    def search(self, query: str, conversation_id: str):
        retriever = AnalystaRetriever(
            vectorstore=self.vectorstore,
            doc_library=conversation_id,
            top_k = self.top_k,
            page_top_k=1,
            weights=weights
        )
        docs = retriever.invoke(query)
        results = []
        for doc in docs[:self.top_k]:
            results.append({
                "content": doc.page_content,
                "tokens": doc.metadata['tokens']
            })
        return results
    
    def store(self, data: str, conversation_id: str, tokens: int):
        ## TODO: Add more loaders, currently we go only with text loader
        ## Currently for PoC I will use only text loader
        doc = Document(page_content=data, 
                       metadata={
                           "source": "text", 
                           'type': 'data', 
                           'library': conversation_id, 
                           'tokens': tokens})
        add_documents(self.vectorstore, [doc])
        self.vectorstore.persist()
        
        # _documents = []
        # Will see if we need chunker
        # for index, document in enumerate(self.splitter.split(doc, splitter_name)):
        #     _documents.append(Document(
        #         page_content=document.page_content, 
        #         metadata={'source': document.metadata['source'], 
        #                   'type': 'data', 'library': conversation_id, 
        #                   'chunk_index': index}))
            
        