import csv
import os
import datetime
import warnings
from typing import Any, List, Mapping, Dict

import tiktoken
from langchain.llms.base import LLM
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.nlp import extract_sections

import openai
import boto3
from google import genai


STOP_WORDS = ["Observation:", "Observations:", "observation:", "observations:"]

MODEL_PRICING: Dict[str, tuple] = {
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gpt-5.4": (2.50, 15.00),
}

warnings.filterwarnings("ignore", message=".*This feature is deprecated.*genai-vertexai-sdk.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")
warnings.filterwarnings("ignore", category=FutureWarning, module="spacy")


class CloudAPILLM(LLM):
    model_name: str
    max_context_length: int
    provider: str  # azure, bedrock, or vertex
    seed: int = 2023
    tags: Dict[str, str] = None
    self_consistency: bool = False
    supports_stop: bool = True

    # Azure config
    azure_endpoint: str = None
    azure_deployment: str = None
    azure_api_key: str = None
    azure_api_version: str = "2024-12-01-preview"

    # Amazon Bedrock config
    aws_region: str = "us-east-1"
    bedrock_model_id: str = None

    # GCP Vertex AI config
    gcp_project: str = None
    gcp_location: str = "us-central1"
    vertex_model_id: str = None

    MAX_OUTPUT_TOKENS: int = 4096

    client: Any = None
    tokenizer: Any = None
    probabilities: Any = None

    cost_log_path: str = "logs/cost.csv"
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0

    class Config:
        underscore_attrs_are_private = True

    @property
    def _llm_type(self) -> str:
        return "cloud_api"

    @property
    def _llm_name(self) -> str:
        return self.model_name

    def load_model(self, base_models: str = None) -> None:
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        if self.provider == "azure":
            self.client = openai.OpenAI(
                base_url=self.azure_endpoint,
                api_key=self.azure_api_key,
                timeout=300,
            )
        elif self.provider == "bedrock":
            self.client = boto3.client(
                "bedrock-runtime", region_name=self.aws_region
            )
        elif self.provider == "vertex":
            self.client = genai.Client(
                    vertexai=True,
                    project=self.gcp_project,
                    location="global",
                )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call(
        self,
        prompt: str,
        stop: List[str],
        **kwargs,
    ) -> str:
        """Route the call to the appropriate cloud provider."""
        self.probabilities = None

        messages = extract_sections(prompt, self.tags)

        if self.supports_stop:
            all_stop = [s for s in (STOP_WORDS + list(stop or [])) if s]
        else:
            all_stop = []

        if self.provider == "azure":
            output = self._call_azure(messages, all_stop)
        elif self.provider == "bedrock":
            output = self._call_bedrock(messages, all_stop)
        elif self.provider == "vertex":
            output = self._call_vertex(messages, all_stop)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        for stop_word in STOP_WORDS + list(stop or []):
            output = output.replace(stop_word, "")

        return output.strip()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def _call_azure(self, messages: list, stop: list) -> str:
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "seed": self.seed,
            "max_completion_tokens": self.MAX_OUTPUT_TOKENS,
        }
        # Some models (e.g. gpt-5.4) don't support the stop parameter
        if stop:
            params["stop"] = stop
        response = self.client.chat.completions.create(**params)
        if response.usage:
            self._track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def _call_bedrock(self, messages: list, stop: list) -> str:
        system_text = ""
        bedrock_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["content"].strip():
                bedrock_messages.append(
                    {"role": msg["role"], "content": [{"text": msg["content"]}]}
                )

        while bedrock_messages and bedrock_messages[-1]["role"] == "assistant":
            prefill = bedrock_messages.pop()["content"][0]["text"].strip()
            if prefill and bedrock_messages and bedrock_messages[-1]["role"] == "user":
                bedrock_messages[-1]["content"][0]["text"] += (
                    f"\n\nPlease begin your response with: {prefill}"
                )

        params = {
            "modelId": self.bedrock_model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": 0.0,
                "stopSequences": stop,
                "maxTokens": self.MAX_OUTPUT_TOKENS,
            },
        }
        if system_text:
            params["system"] = [{"text": system_text}]

        response = self.client.converse(**params)
        usage = response.get("usage", {})
        if usage:
            self._track_usage(
                usage.get("inputTokens", 0),
                usage.get("outputTokens", 0),
            )
        return response["output"]["message"]["content"][0]["text"]

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def _call_vertex(self, messages: list, stop: list) -> str:
        system_text = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        # Gemini caps stop_sequences at 5
        stop_seqs = stop[:5]

        config_kwargs = {
            "temperature": 0.0,
            "stop_sequences": stop_seqs,
            "max_output_tokens": self.MAX_OUTPUT_TOKENS,
            "automatic_function_calling": genai.types.AutomaticFunctionCallingConfig(disable=True),
        }
        if system_text:
            config_kwargs["system_instruction"] = system_text

        config = genai.types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.vertex_model_id, contents=contents, config=config,
        )
        if response.usage_metadata:
            self._track_usage(
                response.usage_metadata.prompt_token_count or 0,
                response.usage_metadata.candidates_token_count or 0,
            )

        finish_reason = None
        if response.candidates:
            finish_reason = getattr(response.candidates[0], "finish_reason", None)
        if finish_reason and str(finish_reason).endswith("MAX_TOKENS"):
            logger.warning(
                f"Vertex hit MAX_TOKENS on {self.model_name}. Returning truncated text to let the agent terminate naturally."
            )

        try:
            return response.text or ""
        except ValueError as e:
            safety_ratings = None
            prompt_feedback = getattr(response, "prompt_feedback", None)
            if response.candidates:
                safety_ratings = getattr(response.candidates[0], "safety_ratings", None)
            logger.error(f"Vertex response has no text. finish_reason={finish_reason} safety_ratings={safety_ratings} "
                         f"prompt_feedback={prompt_feedback} error={e}")
            raise

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "provider": self.provider,
        }

    def _track_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.call_count += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        pricing = MODEL_PRICING.get(self.model_name)
        cost = 0.0
        if pricing:
            cost = (prompt_tokens * pricing[0] + completion_tokens * pricing[1]) / 1_000_000
            self.total_cost += cost

        logger.info(
            f"[{self.model_name}] call={self.call_count} "
            f"tokens={prompt_tokens}+{completion_tokens} cost=${cost:.4f} "
            f"| total=${self.total_cost:.4f}"
        )

        if self.cost_log_path:
            os.makedirs(os.path.dirname(self.cost_log_path), exist_ok=True)
            with open(self.cost_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not os.path.exists(self.cost_log_path):
                    writer.writerow([
                        "timestamp", "model", "provider", "call_num",
                        "prompt_tokens", "completion_tokens", "cost_usd",
                        "cumulative_cost_usd",
                    ])
                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    self.model_name, self.provider, self.call_count,
                    prompt_tokens, completion_tokens,
                    f"{cost:.6f}", f"{self.total_cost:.6f}",
                ])
