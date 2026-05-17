# RAG with LangChain

A step-by-step demonstration of how to build a Retrieval Augmented Generation (RAG) system using LangChain, OpenAI, and Pinecone.

## Overview

This tutorial progressively builds a complete RAG pipeline:
1. **Document Ingestion** - Load, chunk, embed, and store documents in a vector database
2. **Naive RAG** - Implement a basic retrieval chain using manual function calls
3. **LCEL RAG** - Refactor to use LangChain Expression Language for a cleaner, more powerful approach
