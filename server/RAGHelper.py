import hashlib
import logging
import os
import pickle
import requests
import base64
import csv
import json
import re

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader,
    JSONLoader,
    PyPDFDirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_milvus.vectorstores import Milvus
# from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lxml import etree

from ScoredCrossEncoderReranker import ScoredCrossEncoderReranker
from tqdm import tqdm
from sharepoint_regex import extract_sharepoint_link
import docx
from customdocxloader import CustomDocxLoader

import logging

logger = logging.getLogger(__name__)

# Escaping helper for prompt-safe content (SharePoint URLs, etc.)
def escape_curly_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")

class RAGHelper:
    """
    A helper class to manage retrieval-augmented generation (RAG) processes,
    including data loading, chunking, vector storage, and retrieval.
    """

    def __init__(self, logger):
        """
        Initializes the RAGHelper class and loads environment variables.
        """
        self.logger = logger
        self.chunked_documents = []
        self.embeddings = None  # Placeholder for embeddings; set during initialization
        self.text_splitter = None
        self.db = None
        self.sparse_retriever = None
        self.ensemble_retriever = None
        self.rerank_retriever = None
        self._batch_size = 1000
        # Load environment variables
        self.vector_store_sparse_uri = os.getenv("vector_store_sparse_uri")
        self.vector_store_uri = os.getenv("vector_store_uri")
        self.document_chunks_pickle = os.getenv("document_chunks_pickle")
        self.data_dir = os.getenv("data_directory")
        self.file_types = os.getenv("file_types").split(",")
        self.splitter_type = os.getenv("splitter")
        self.vector_store = os.getenv("vector_store")
        self.vector_store_initial_load = (
            os.getenv("vector_store_initial_load") == "True"
        )
        self.rerank = os.getenv("rerank") == "True"
        self.rerank_model = os.getenv("rerank_model")
        self.rerank_k = int(os.getenv("rerank_k"))
        self.vector_store_k = int(os.getenv("vector_store_k"))
        self.chunk_size = int(os.getenv("chunk_size"))
        self.chunk_overlap = int(os.getenv("chunk_overlap"))
        self.breakpoint_threshold_amount = int(os.getenv('breakpoint_threshold_amount')) if os.getenv('breakpoint_threshold_amount', 'None') != 'None' else None
        self.number_of_chunks = None if (value := os.getenv('number_of_chunks',
                                                            None)) is None or value.lower() == 'none' else int(value)
        self.breakpoint_threshold_type = os.getenv('breakpoint_threshold_type')
        self.vector_store_collection = os.getenv("vector_store_collection")
        self.xml_xpath = os.getenv("xml_xpath")
        self.json_text_content = (
            os.getenv("json_text _content", "false").lower() == "true"
        )
        self.json_schema = os.getenv("json_schema")
        self.neo4j = os.getenv("neo4j_location")
        self.add_docs_to_neo4j = os.getenv("file_upload_using_llm")
        self.dynamic_neo4j_schema = os.getenv("dynamic_neo4j_schema") == "True"
        self.normalize_provenance = os.getenv("normalize_provenance", "False").lower() == "true"

    @staticmethod
    def format_documents(docs):
        """Formats documents for readable context injection, and escapes any curly braces in content or metadata."""
        doc_strings = []
        for i, doc in enumerate(docs):
            # Escape page content
            safe_content = escape_curly_braces(doc.page_content)

            # Escape all metadata values
            safe_metadata = {
                key: escape_curly_braces(str(value))
                for key, value in doc.metadata.items()
            }

            metadata_string = ", ".join([f"{k}: {v}" for k, v in safe_metadata.items()])
            doc_strings.append(
                f"Document {i} content: {safe_content}\nDocument {i} metadata: {metadata_string}"
            )

        return "\n\n<NEWDOC>\n\n".join(doc_strings)

    def _load_chunked_documents(self):
        """Loads previously chunked documents from a pickle file."""
        with open(self.document_chunks_pickle, "rb") as f:
            self.logger.info("Loading chunked documents.")
            self.chunked_documents = pickle.load(f)

    def _load_json_files(self):
        """
        Loads JSON files from the data directory.

        Returns:
            list: A list of loaded Document objects from JSON files.
        """
        text_content = self.json_text_content
        loader_kwargs = {"jq_schema": self.json_schema, "text_content": text_content}
        loader = DirectoryLoader(
            path=self.data_dir,
            glob="*.json",
            loader_cls=JSONLoader,
            loader_kwargs=loader_kwargs,
            recursive=True,
            show_progress=True,
        )
        return loader.load()

    def _load_xml_files(self):
        """
        Loads XML files from the data directory and extracts relevant elements.

        Returns:
            list: A list of Document objects created from XML elements.
        """
        loader = DirectoryLoader(
            path=self.data_dir,
            glob="*.xml",
            loader_cls=TextLoader,
            recursive=True,
            show_progress=True,
        )
        xmldocs = loader.load()
        newdocs = []
        for index, doc in enumerate(xmldocs):
            try:
                xmltree = etree.fromstring(doc.page_content.encode("utf-8"))
                elements = xmltree.xpath(self.xml_xpath)
                elements = [
                    etree.tostring(element, pretty_print=True).decode()
                    for element in elements
                ]
                metadata = doc.metadata
                metadata["index"] = index
                newdocs += [
                    Document(page_content=content, metadata=metadata)
                    for content in elements
                ]
            except Exception as e:
                self.logger.error(f"Error processing XML document: {e}")
        return newdocs

    @staticmethod
    def _filter_metadata(docs, filters=None):
        """
        Filters the metadata of documents by retaining only specified keys.

        Parameters
        ----------
        docs : list
            A list of document objects, where each document contains a metadata dictionary.
        filters : list, optional
            A list of metadata keys to retain (default is ["source"]).

        Returns
        -------
        list
            The modified list of documents with filtered metadata.

        Raises
        ------
        ValueError
            If docs is not a list or if filters is not a list.
        """
        if not isinstance(docs, list):
            raise ValueError("Expected 'docs' to be a list.")
        if filters is None:
            filters = ["source"]
        elif not isinstance(filters, list):
            raise ValueError("Expected 'filters' to be a list.")

        # Filter metadata for each document
        for doc in docs:
            doc.metadata = {
                key: doc.metadata.get(key) for key in filters if key in doc.metadata
            }

        return docs

    def _load_documents(self):
        """
        Loads documents from specified file types in the data directory.

        Returns:
            list: A list of loaded Document objects.
        """
        docs = []
        for file_type in self.file_types:
            try:
                self.logger.info(f"Loading {file_type} document(s)....")
                new_docs = []
                if file_type == "pdf":
                    loader = PyPDFDirectoryLoader(self.data_dir)
                    new_docs = loader.load()
                elif file_type == "json":
                    new_docs = self._load_json_files()
                elif file_type == "txt":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.txt",
                        loader_cls=TextLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    new_docs = loader.load()
                elif file_type == "csv":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.csv",
                        loader_cls=CSVLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    new_docs = loader.load()
                elif file_type == "docx":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.docx",
                        loader_cls=CustomDocxLoader,
                        recursive=True,
                        show_progress=True,
                        silent_errors=True,
                    )
                    new_docs = loader.load()
                elif file_type == "xlsx":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.xlsx",
                        loader_cls=UnstructuredExcelLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    new_docs = loader.load()
                elif file_type == "pptx":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.pptx",
                        loader_cls=UnstructuredPowerPointLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    new_docs = loader.load()
                elif file_type == "xml":
                    new_docs = self._load_xml_files()
                else:
                    continue

                # Inject SharePoint link into metadata
                for doc in new_docs:
                    link = extract_sharepoint_link(doc.page_content)
                    if link:
                        doc.metadata["sharepoint_url"] = link
                        self.logger.info(f"[SHAREPOINT] Found in {doc.metadata.get('source')}: {link}")
                    else:
                        doc.metadata["sharepoint_url"] = ""

                docs += new_docs

            except Exception as e:
                self.logger.error(f"Error loading {file_type} files: {e}")

        return self._filter_metadata(docs, filters=["source", "sharepoint_url"])

    def _load_json_document(self, filename):
        """Load JSON documents with specific parameters"""
        return JSONLoader(
            file_path=filename,
            jq_schema=self.json_schema,
            text_content=self.json_text_content,
        )

    def _load_document(self, filename):
        """Load documents from the specified file based on its extension."""
        file_type = filename.lower().split(".")[-1]
        loaders = {
            "pdf": PyPDFLoader,
            "json": self._load_json_document,
            "txt": TextLoader,
            "csv": CSVLoader,
            "docx": CustomDocxLoader,
            "xlsx": UnstructuredExcelLoader,
            "pptx": UnstructuredPowerPointLoader,
        }
        self.logger.info(f"Loading {file_type} document....")
        if file_type in loaders:
            docs = loaders[file_type](filename).load()
            for doc in docs:
                link = extract_sharepoint_link(doc.page_content)
                if link:
                    doc.metadata["sharepoint_url"] = link
                    self.logger.info(f"[SHAREPOINT] Extracted from {filename}: {link}")
                else:
                    doc.metadata["sharepoint_url"] = ""  # Ensure it's always present

            return self._filter_metadata(docs, filters=["source", "sharepoint_url"])
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _create_recursive_text_splitter(self):
        """
        Creates an instance of RecursiveCharacterTextSplitter.

        Returns:
            RecursiveCharacterTextSplitter: A configured text splitter instance.
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            keep_separator=True,
            separators=[
                "\n \n",
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
                " ",
                ",",
                "\u200b",
                "\uff0c",
                "\u3001",
                "\uff0e",
                "\u3002",
                "",
            ],
        )

    def _create_semantic_chunker(self):
        """
        Creates an instance of SemanticChunker.

        Returns:
            SemanticChunker: A configured semantic chunker instance.
        """
        return SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            number_of_chunks=self.number_of_chunks,
        )

    def _initialize_text_splitter(self):
        """Initialize the text splitter based on the environment settings."""
        self.logger.info(f"Initializing {self.splitter_type} splitter.")
        if self.splitter_type == "RecursiveCharacterTextSplitter":
            self.text_splitter = self._create_recursive_text_splitter()
        elif self.splitter_type == "SemanticChunker":
            self.text_splitter = self._create_semantic_chunker()

    def _split_documents(self, docs):
        """
        Splits documents into chunks.

        Args:
            docs (list): A list of loaded Document objects.
        """
        self._initialize_text_splitter()
        self.logger.info("Chunking document(s).")
        chunked_documents = []
        for doc in docs:
            original_link = extract_sharepoint_link(doc.page_content)
            split_chunks = self.text_splitter.split_documents([doc])

            for chunk in split_chunks:
                chunk_link = extract_sharepoint_link(chunk.page_content)

                chunked_documents.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata={
                            **chunk.metadata,
                            "id": hashlib.md5(chunk.page_content.encode()).hexdigest(),
                            "sharepoint_url": chunk_link or original_link or "",
                        },
                    )
                )
        return chunked_documents

    def _split_and_store_documents(self, docs):
        """
        Splits documents into chunks and stores them as a pickle file.

        Args:
            docs (list): A list of loaded Document objects.
        """
        self.chunked_documents = self._split_documents(docs)
        self.logger.info(f"Generated {len(self.chunked_documents)} chunks.")
        # Store the chunked documents
        self.logger.info("Storing chunked document(s).")
        with open(self.document_chunks_pickle, "wb") as f:
            pickle.dump(self.chunked_documents, f)

    def _initialize_milvus(self):
        """Initializes the Milvus vector store."""
        self.logger.info("Setting up Milvus Vector DB.")
        self.db = Milvus.from_documents(
            [],
            self.embeddings,
            drop_old=not self.vector_store_initial_load,
            connection_args={"uri": self.vector_store_uri},
            collection_name=self.vector_store_collection,
        )


    def _initialize_vector_store(self):
        """Initializes the vector store based on the specified type (Milvus or Postgres)."""
        if self.vector_store == "milvus":
            self._initialize_milvus()
        else:
            raise ValueError(
                "Only 'milvus' or 'postgres' are supported as vector stores! Please set vector_store in your "
                "environment variables."
            )
        if self.vector_store_initial_load:
            self.logger.info("Loading data from existing store.")
            # Add the documents 1 by 1, so we can track progress
            with tqdm(
                total=len(self.chunked_documents), desc="Vectorizing documents"
            ) as pbar:
                for i in range(0, len(self.chunked_documents), self._batch_size):
                    # Slice the documents for the current batch
                    batch = self.chunked_documents[i : i + self._batch_size]
                    # Prepare documents and their IDs for batch insertion
                    documents = [d for d in batch]
                    ids = [d.metadata["id"] for d in batch]

                    # Add the batch of documents to the database
                    self.db.add_documents(documents, ids=ids)

                    # Update the progress bar by the size of the batch
                    pbar.update(len(batch))

    from langchain_community.retrievers import BM25Retriever

    def _initialize_bm25retriever(self):
        """Initializes in-memory BM25Retriever with logging for debug."""
        self.logger.info("Initializing BM25Retriever.")

        if not self.chunked_documents:
            self.logger.warning("No documents available for BM25Retriever — skipping.")
            self.sparse_retriever = None
            return

        def debug_preprocessing_func(x):
            self.logger.warning(f"[BM25 DEBUG] Preprocessing input of type {type(x)}: {x}")
            if isinstance(x, str):
                return x.split()
            elif isinstance(x, dict):
                self.logger.error(f"[BM25 ERROR] BM25 received a dict: {json.dumps(x, indent=2)}")
                raise ValueError("BM25 received a dict instead of a string.")
            else:
                raise TypeError(f"BM25 received unsupported type: {type(x)}")

        self.sparse_retriever = BM25Retriever.from_texts(
            [x.page_content for x in self.chunked_documents],
            metadatas=[x.metadata for x in self.chunked_documents],
            preprocess_func=debug_preprocessing_func
        )

   

    def _initialize_retrievers(self):
        """Initializes the sparse retriever, ensemble retriever, and rerank retriever."""
        if self.vector_store == "milvus":
            self._initialize_bm25retriever()
        elif self.vector_store == "postgres":
            self._initialize_postgresbm25retriever()
        else:
            raise ValueError(
                "Only 'milvus' or 'postgres' are supported as vector stores! Please set vector_store in your "
                "environment variables."
            )

    def _initialize_reranker(self):
        """Initialize the reranking model based on environment settings."""
        if self.rerank_model == "flashrank":
            self.logger.info("Setting up the FlashrankRerank.")
            self.compressor = FlashrankRerank(top_n=self.rerank_k)
        else:
            self.logger.info("Setting up the ScoredCrossEncoderReranker.")
            logger.warning(f"[RERANK DEBUG] Using rerank_model: {self.rerank_model}")
            self.compressor = ScoredCrossEncoderReranker(
                model=HuggingFaceCrossEncoder(model_name=self.rerank_model),
                top_n=self.rerank_k,
            )
        self.logger.info("Setting up the ContextualCompressionRetriever.")
        self.rerank_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.ensemble_retriever
        )

    def _log_and_extract(self, x, label=""):
        self.logger.warning(f"[{label.upper()} RETRIEVER] Incoming input to retriever: {x} (type={type(x)})")
        if isinstance(x, dict):
            question = x.get("question", "")
            self.logger.warning(f"[{label.upper()} RETRIEVER] Extracted question: {question}")
            return question
        return x
    

    def _setup_retrievers(self):
        """Sets up the retrievers based on specified configurations."""
        self._initialize_retrievers()
        self.logger.info("Setting up the Vector Retriever.")
        retriever = self.db.as_retriever(
            search_type="mmr", search_kwargs={"k": self.vector_store_k}
        )
        from langchain_core.runnables import RunnableLambda
        self.logger.info("Setting up the hybrid retriever.")
        
        # Wrap both sparse and dense retrievers to ensure input is str

        wrapped_sparse = RunnableLambda(lambda x: self._log_and_extract(x, label="sparse")) | self.sparse_retriever
        wrapped_dense = RunnableLambda(lambda x: self._log_and_extract(x, label="dense")) | retriever

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[wrapped_sparse, wrapped_dense],
            weights=[0.5, 0.5]
        )

        if self.rerank:
            self._initialize_reranker()

    def _update_chunked_documents(self, new_chunks):
        """Update the chunked documents list and store them."""
        if self.vector_store == "milvus":
            if not self.chunked_documents:
                if os.path.exists(self.document_chunks_pickle):
                    self.logger.info("documents chunk pickle exists, loading it.")
                    self._load_chunked_documents()
            self.chunked_documents += new_chunks
            with open(f"{self.vector_store_uri}_sparse.pickle", "wb") as f:
                pickle.dump(self.chunked_documents, f)

    def _add_to_vector_database(self, new_chunks):
        """Add the new document chunks to the vector database."""
        if not self.db:
            self._initialize_vector_store()

        # Deduplicate to prevent conflicts
        documents = list({d.metadata["id"]: d for d in new_chunks}.values())
        ids = [d.metadata["id"] for d in documents]
        self.db.add_documents(documents, ids=ids)

        if self.vector_store == "postgres":
            self.sparse_retriever.add_documents(documents, ids)
        else:
            # Recreate the in-memory store
            self._initialize_bm25retriever()
            # Update full retriever too
        retriever = self.db.as_retriever(
            search_type="mmr", search_kwargs={"k": self.vector_store_k}
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.sparse_retriever, retriever], weights=[0.5, 0.5]
        )

    def _parse_cv(self, doc):
        """Extract skills from the CV document."""
        # Implement your skill extraction logic here
        return []

    def _deduplicate_chunks(self):
        """Ensure there are no duplicate entries in the data."""
        self.chunked_documents = list(
            {doc.metadata["id"]: doc for doc in self.chunked_documents}.values()
        )

    def load_data(self):
        """
        Loads data from various file types and chunks it into an ensemble retriever.
        """
        force_reload = os.getenv("force_reload_documents", "False").lower() == "true"

        if force_reload or not os.path.exists(self.document_chunks_pickle):
            self.logger.info("Force reload or no pickle found – loading documents from data_directory.")
            docs = self._load_documents()
            self.logger.info(f"Loaded {len(docs)} documents.")
            self.logger.info("Chunking and storing document(s).")
            self._split_and_store_documents(docs)
        else:
            self.logger.info("Using cached document chunks from pickle.")
            self._load_chunked_documents()

        self._deduplicate_chunks()
        self._initialize_vector_store()
        self._setup_retrievers()

    def add_csv_to_graphdb(self, filename):
        path = os.path.join(self.data_dir, filename)
        url_add_instance = f"{self.neo4j}/add_instances"
        try:
            self.logger.info("Uploading csv instances using json")
            with open(path, mode="r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile, delimiter=";")
                self.logger.info(
                    reader.fieldnames
                )  # This can be changed to a format you have to follow and then the csv will always upload
                payloads = []
                for row in reader:
                    payloads.append(
                        {
                            "query": "MERGE (q:Quote {text: $quoteText}) "
                            "MERGE (t:Topic {name: $topicName}) "
                            "MERGE (q)-[:IS_PART_OF]->(t)",
                            "parameters": {
                                "quoteText": row.get("quote"),
                                "topicName": row.get("topics"),
                            },
                        }
                    )
            self.logger.info(f"JSON is: {payloads}")
            self.logger.info(f"URL is: {url_add_instance}")
            response = requests.post(url=url_add_instance, json=payloads)
            self.logger.info(
                f"Succesfully loaded {len(payloads)} records into payloads"
            )
        except:
            self.logger.info(f"server responded with: {response.text}")

    def get_llm(self):
        """Accessor method to get the LLM. Subclasses can override this."""
        return None

    def escape_curly_braces_in_query(self, json_string):
        # Function to escape braces in the matched 'query' string
        def escape_braces(match):
            query_content = match.group(1)
            escaped_content = query_content.replace("{", "\\\\{").replace("}", "\\\\}")
            return '"query": "' + escaped_content + '"'

        # Regular expression to find 'query' fields
        pattern = r'"query":\s*"([^"]*)"'
        return re.sub(pattern, escape_braces, json_string)

    def add_document_to_graphdb(self, page_content, metadata):
        llm = self.get_llm()
        if llm is None:
            self.logger.error("LLM is not available in RAGHelper.")
            return None
        if metadata.get("source").lower().split(".")[-1] == "pdf":
            try:
                if self.dynamic_neo4j_schema == True:
                    schema_response = requests.get(url=self.neo4j + "/schema")
                    if schema_response.status_code != 200:
                        self.logger.info(
                            "Failed to retrieve schema from the graph database."
                        )
                        return None
                    schema = schema_response.json()
                    

                    # Construct schema text for the prompt
                    schema_text = self.format_schema_for_prompt(schema)

                    self.logger.info(f"this is the text: {schema_text}")

                    retrieval_question = (
                        os.getenv("neo4j_insert_schema")
                        .replace("{schema}", schema_text)
                        .replace("{data}", page_content)
                    )
                else:
                    retrieval_question = os.getenv("neo4j_insert_data_only").replace(
                        "{data}", page_content
                    )

                # Load prompt components from .env
                retrieval_instruction = os.getenv("neo4j_insert_instruction")
                retrieval_few_shot = os.getenv("neo4j_insert_few_shot")

                retrieval_instruction = retrieval_instruction.replace(
                    "{", "{{"
                ).replace("}", "}}")
                retrieval_few_shot = retrieval_few_shot.replace("{", "{{").replace(
                    "}", "}}"
                )
                retrieval_question = retrieval_question.replace("{", "{{").replace(
                    "}", "}}"
                )

                # Combine into a single prompt
                retrieval_thread = [
                    ("system", retrieval_instruction + "\n\n" + retrieval_few_shot),
                    ("human", retrieval_question),
                ]

                rag_prompt = ChatPromptTemplate.from_messages(retrieval_thread)
                self.logger.info("Initializing retrieval for RAG.")

                # Create an LLM chain
                llm_chain = rag_prompt | llm
                # Invoke the LLM chain and get the response
                try:
                    llm_response = llm_chain.invoke({})
                    # self.logger.info(f"llm response is: {llm_response}")
                    response_text = self.extract_response_content(llm_response).strip()
                    self.logger.info(f"The LLM response is: {response_text}")

                    # Escape the curly braces in 'query' strings
                    escaped_data = self.escape_curly_braces_in_query(response_text)

                    # Now parse the JSON
                    try:
                        json_data = json.loads(escaped_data)
                        print("Parsed JSON data:", json_data)
                    except json.JSONDecodeError as e:
                        print("Error parsing JSON:", e)

                    def unescape_curly_braces(json_data):
                        for item in json_data:
                            item["query"] = (
                                item["query"].replace("\\{", "{").replace("\\}", "}")
                            )
                        return json_data

                    json_data = unescape_curly_braces(json_data)

                    response = requests.post(
                        url=self.neo4j + "/add_instances", json=json_data
                    )
                    self.logger.info(f"{response}")
                    if response == "<Response [200]>":
                        self.logger.info(
                            f"Succesfully loaded {len(json_data)} records into payloads"
                        )
                except Exception as e:
                    self.logger.error(f"Error during LLM invocation: {e}")
                    return None
            except Exception as e:
                self.logger.error(f"Error adding document to the graph database: {e}")

    def add_document(self, filename):
        """
        Load documents from various file types, extract metadata,
        split the documents into chunks, and store them in a vector database.

        Parameters:
            filename (str): The name of the file to be loaded.

        Raises:
            ValueError: If the file type is unsupported.
        """
        if filename.lower().split(".")[-1] == "csv":
            self.add_csv_to_graphdb(filename)
        new_docs = self._load_document(filename)

        # Inject SharePoint link from document content (if it's a docx)
        # Add SharePoint URL to metadata if found in docx content
        sharepoint_url = None
        try:
            for doc in new_docs:
                sharepoint_url = extract_sharepoint_link(doc.page_content)
                if sharepoint_url:
                    self.logger.info(f"Extracted SharePoint URL: {sharepoint_url}")
                    doc.metadata["sharepoint_url"] = sharepoint_url  
                    break
        except Exception as e:
            self.logger.error(f"Error extracting SharePoint URL from {filename}: {e}")
        self.logger.info("adding documents to graphdb.")
        if self.add_docs_to_neo4j:
            for doc in new_docs:
                self.add_document_to_graphdb(doc.page_content, doc.metadata)

        self.logger.info("chunking the documents.")
        new_chunks = self._split_documents(new_docs)

        self._update_chunked_documents(new_chunks)
        # Propagate metadata to chunks
        matched = 0
        for chunk in new_chunks:
            # Match the chunk to its original doc
            for doc in new_docs:
                if chunk.page_content in doc.page_content:
                    chunk.metadata.update(doc.metadata)
                    break
        self.logger.info(f"Metadata injected into {matched}/{len(new_chunks)} chunks")
        for i, chunk in enumerate(new_chunks[:5]):
            self.logger.info(f"Chunk {i} sharepoint_url: {chunk.metadata.get('sharepoint_url')}")
                
        # Add new chunks to the vector database
        self._add_to_vector_database(new_chunks)
        
    @staticmethod
    def format_history_for_prompt(history):
        cleaned = []
        for msg in history:
            content = msg["content"].strip()
            if not content:
                continue

            # Escape user and assistant messages
            if msg["role"] in {"user", "assistant"}:
                content = escape_curly_braces(content)

            cleaned.append(f"{msg['role'].capitalize()}: {content}")
        return "\n".join(cleaned)
