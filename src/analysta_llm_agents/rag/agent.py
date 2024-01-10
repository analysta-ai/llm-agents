from typing import Optional

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.embeddings import __all__ as embeddings  # pylint: disable=E0401
from langchain.vectorstores import __all__ as vectorstores  # pylint: disable=E0401
from langchain.vectorstores import __getattr__ as get_vectorstore_cls  # pylint: disable=E0401

from ..tools.context import Context
from ..tools.utils import unpack_json
from ..tools.retrievers.AgentBasedRetriever import AgentBasedRetriever
from ..base.agent import BaseAgent
from .constants import guidance_message, context_message, weights

class RAGAgent(BaseAgent):
    def __init__(self,
                 repo: Optional[str]=None,
                 api_key: Optional[str]=None, 
                 agent_prompt: Optional[str]=None,
                 model_type: Optional[str]=None,
                 model_params: Optional[dict]=None, 
                 embedding_model: Optional[str]=None,
                 embeddings_params: Optional[dict]=None,
                 vectorstore: Optional[str]=None,
                 vectorstore_params: Optional[dict]=None,
                 library: Optional[str]=None,
                 response_format: str="",
                 top_k: int=5,
                 ctx: Optional[Context]=None):
        
        self.embedding = self.get_embeddings(embedding_model, embeddings_params)
        self.vs = self.get_vectorstore(vectorstore, vectorstore_params, embedding_func=self.embedding)
        self.top_k = top_k
        self.retriever = AgentBasedRetriever(
            vectorstore=self.vs,
            doc_library=library,
            top_k = top_k,
            page_top_k=1,
            weights=weights
        )    
        super().__init__(repo=repo, api_key=api_key, agent_prompt=agent_prompt, 
                         model_type=model_type, model_params=model_params, 
                         response_format=response_format, ctx=ctx)
    
    def get_embeddings(self, embeddings_model: str, embeddings_params: dict):
        """ Get *Embeddings """
        if embeddings_model is None:
            return None
        if embeddings_model in embeddings:
            model = getattr(__import__("langchain.embeddings", fromlist=[embeddings_model]), embeddings_model)
            return model(**embeddings_params)
        raise RuntimeError(f"Unknown Embedding type: {embeddings_model}")


    def get_vectorstore(self, vectorstore_type, vectorstore_params, embedding_func=None):
        """ Get vector store obj """
        if vectorstore_type is None:
            return None
        if vectorstore_type in vectorstores:
            if embedding_func:
                vectorstore_params['embedding_function'] = embedding_func
            return get_vectorstore_cls(vectorstore_type)(**vectorstore_params)
        raise RuntimeError(f"Unknown VectorStore type: {vectorstore_type}")

    def _generateResponse(self, input):
        docs = self.retriever.invoke(input)
        context = f'{guidance_message}\n\n'
        references = set()
        messages = []
        for doc in docs[:self.top_k]:
            context += f'{doc.page_content}\n\n'
            references.add(doc.metadata["source"])
        messages.append(SystemMessage(content=context_message))
        messages.append(HumanMessage(content=context))
        messages.append(HumanMessage(content=input))
        return self.ctx.llm(messages).content

        
    def start(self, task: str):
        response = self._generateResponse(task)
        try:
             yield unpack_json(response.replace("\n", ""))
        except:
            yield response
    
        