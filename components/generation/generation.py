"""
Generation Component
Handles LLM-based response generation using retrieved context.
Supports: OpenAI, Anthropic, HuggingFace, Ollama, Azure OpenAI, Cohere, Google Gemini
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based on the provided context.

Rules:
- Only use information from the provided context to answer
- If the answer is not in the context, say "I don't have enough information to answer this question based on the provided documents."
- Always be accurate, concise, and helpful
- Cite your sources when possible (mention the source document or page)
- Format your response clearly using markdown when appropriate
"""

DEFAULT_USER_TEMPLATE = """Context:
{context}

Question: {question}

Please provide a comprehensive and accurate answer based on the context above."""


# ─────────────────────────────────────────────
# LLM Providers
# ─────────────────────────────────────────────

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        pass

    def stream(self, system_prompt: str, user_message: str, **kwargs) -> Generator[str, None, None]:
        yield self.generate(system_prompt, user_message, **kwargs)


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.get("api_key") or os.environ.get("OPENAI_API_KEY"))
        self.model = config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2048)
        self.timeout = config.get("timeout", 60)

    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            **kwargs,
        )
        return response.choices[0].message.content

    def stream(self, system_prompt: str, user_message: str, **kwargs) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = config.get("model", "claude-3-5-sonnet-20241022")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2048)

    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=self.temperature,
        )
        return response.content[0].text

    def stream(self, system_prompt: str, user_message: str, **kwargs) -> Generator[str, None, None]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=self.temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text


class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("model", "mistralai/Mistral-7B-Instruct-v0.3")
        self.device = config.get("device", "cpu")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.1)
        self._load_model()

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_tokens,
            device=0 if self.device == "cuda" else -1,
        )

    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output = self.pipe(prompt, do_sample=self.temperature > 0, temperature=self.temperature or None)
        generated = output[0]["generated_text"]
        # Strip the input prompt
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()


class OllamaProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        import ollama
        self.client = ollama.Client(host=config.get("base_url", "http://localhost:11434"))
        self.model = config.get("model", "llama3.2")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2048)

    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        return response["message"]["content"]

    def stream(self, system_prompt: str, user_message: str, **kwargs) -> Generator[str, None, None]:
        stream = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]


class AzureOpenAIProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=config.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=config.get("azure_endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=config.get("api_version", "2024-02-01"),
        )
        self.model = config.get("azure_deployment", "gpt-4o")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2048)

    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class GoogleGeminiProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        import google.generativeai as genai
        genai.configure(api_key=config.get("api_key") or os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(
            model_name=config.get("model", "gemini-1.5-pro"),
            generation_config={"temperature": config.get("temperature", 0.1), "max_output_tokens": config.get("max_tokens", 2048)},
        )

    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        full_prompt = f"{system_prompt}\n\n{user_message}"
        response = self.model.generate_content(full_prompt)
        return response.text


LLM_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "huggingface": HuggingFaceProvider,
    "ollama": OllamaProvider,
    "azure_openai": AzureOpenAIProvider,
    "gemini": GoogleGeminiProvider,
}


def get_llm_provider(config: Dict[str, Any]) -> BaseLLMProvider:
    provider = config.get("provider", "openai")
    cls = LLM_PROVIDERS.get(provider)
    if not cls:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose from {list(LLM_PROVIDERS.keys())}")
    logger.info(f"Using LLM provider: {provider} / {config.get('model')}")
    return cls(config)


# ─────────────────────────────────────────────
# Context Builder
# ─────────────────────────────────────────────

class ContextBuilder:
    def __init__(self, max_sources: int = 3, response_format: str = "markdown"):
        self.max_sources = max_sources
        self.response_format = response_format

    def build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build the context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:self.max_sources]):
            source = doc.get("metadata", {}).get("source", f"Document {i+1}")
            page = doc.get("metadata", {}).get("page", "")
            page_str = f", Page {page}" if page else ""
            score = doc.get("score", 0)
            header = f"[Source {i+1}: {Path(source).name}{page_str} (relevance: {score:.2f})]"
            context_parts.append(f"{header}\n{doc['content']}")
        return "\n\n---\n\n".join(context_parts)

    def build_sources_summary(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a summary of sources for the response."""
        sources = []
        for doc in retrieved_docs[:self.max_sources]:
            meta = doc.get("metadata", {})
            sources.append({
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page"),
                "score": round(doc.get("score", 0), 3),
                "excerpt": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
            })
        return sources


# ─────────────────────────────────────────────
# Generation Component
# ─────────────────────────────────────────────

class GenerationComponent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = get_llm_provider(config)
        self.system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.user_template = config.get("user_template", DEFAULT_USER_TEMPLATE)
        self.include_sources = config.get("include_sources", True)
        self.max_sources = config.get("max_sources", 3)
        self.response_format = config.get("response_format", "markdown")
        self.streaming = config.get("streaming", False)
        self.context_builder = ContextBuilder(self.max_sources, self.response_format)

    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an answer given a query and retrieved docs."""
        context = self.context_builder.build_context(retrieved_docs)
        user_message = self.user_template.format(context=context, question=query)

        logger.info(f"Generating response for query: '{query[:80]}'")
        answer = self.llm.generate(self.system_prompt, user_message)

        result = {
            "query": query,
            "answer": answer,
            "retrieved_docs_count": len(retrieved_docs),
        }
        if self.include_sources:
            result["sources"] = self.context_builder.build_sources_summary(retrieved_docs)

        return result

    def generate_stream(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Generator[str, None, None]:
        context = self.context_builder.build_context(retrieved_docs)
        user_message = self.user_template.format(context=context, question=query)
        yield from self.llm.stream(self.system_prompt, user_message)

    def process_from_files(self, query: str, retrieval_output_path: str, output_path: str) -> str:
        """Load retrieval results from file, generate answer, save output."""
        with open(retrieval_output_path) as f:
            retrieval_data = json.load(f)

        retrieved_docs = retrieval_data.get("results", [])
        result = self.generate(query, retrieved_docs)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Generation complete. Answer saved to {output_path}")
        return output_path


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="RAG Generation Component")
    parser.add_argument("--query", required=True)
    parser.add_argument("--retrieval-output", required=True, help="Path to retrieval results JSON")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    component = GenerationComponent(cfg.get("generation", {}))
    component.process_from_files(args.query, args.retrieval_output, args.output_path)


if __name__ == "__main__":
    main()
