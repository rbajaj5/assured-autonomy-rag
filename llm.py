"""LLM abstraction.

Supports Anthropic, OpenAI, xAI/Grok, Mistral, Ollama, and local PEFT adapters. Also provides a `StubLLM` backend
that is useful for plumbing tests and demos without API access — it returns
a structured echo of the prompt so the rest of the pipeline can be exercised.

Selection:
    get_llm("anthropic", model="claude-opus-4-5")
    get_llm("openai", model="gpt-4o-mini")
    get_llm("xai", model="grok-4.20-reasoning")
    get_llm("mistral", model="mistral-large-latest")
    get_llm("ollama", model="mistral")
    get_llm("peft-local", model="models/aa_mistral_lora_adapter")
    get_llm("stub")
"""
from __future__ import annotations
import os
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Minimal LLM interface: a single `complete(system, user) -> str` method."""

    name: str = "abstract"

    @abstractmethod
    def complete(self, system: str, user: str, *, max_tokens: int = 2048,
                 temperature: float = 0.2) -> str: ...


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-5", api_key: Optional[str] = None):
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. `pip install anthropic`"
            ) from e
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def complete(self, system, user, *, max_tokens=2048, temperature=0.2):
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate any text blocks
        return "".join(getattr(b, "text", "") for b in msg.content)


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError("openai package not installed. `pip install openai`") from e
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def complete(self, system, user, *, max_tokens=2048, temperature=0.2):
        messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
        ]
        attempts = [
            {"max_completion_tokens": max_tokens, "temperature": temperature},
            {"max_completion_tokens": max_tokens},
            {"max_tokens": max_tokens, "temperature": temperature},
            {"max_tokens": max_tokens},
        ]
        last_exc = None
        for params in attempts:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **params,
                )
                break
            except Exception as exc:
                message = str(exc)
                is_retryable_param_error = (
                    "max_completion_tokens" in message
                    or "max_tokens" in message
                    or "temperature" in message
                    or "Unsupported parameter" in message
                    or "Unsupported value" in message
                )
                if not is_retryable_param_error:
                    raise
                last_exc = exc
        else:
            raise last_exc
        return resp.choices[0].message.content or ""


class XAIClient(LLMClient):
    """xAI/Grok backend using the OpenAI-compatible chat completions API."""

    name = "xai"

    def __init__(self, model: str = "grok-4.20-reasoning", api_key: Optional[str] = None):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError("openai package not installed. `pip install openai`") from e
        self._client = OpenAI(
            api_key=api_key or os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"),
            base_url=(os.environ.get("XAI_BASE_URL") or "https://api.x.ai/v1").rstrip("/"),
        )
        self.model = model

    def complete(self, system, user, *, max_tokens=2048, temperature=0.2):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        attempts = [
            {"max_completion_tokens": max_tokens, "temperature": temperature},
            {"max_completion_tokens": max_tokens},
            {"max_tokens": max_tokens, "temperature": temperature},
            {"max_tokens": max_tokens},
        ]
        last_exc = None
        for params in attempts:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **params,
                )
                break
            except Exception as exc:
                message = str(exc)
                is_retryable_param_error = (
                    "max_completion_tokens" in message
                    or "max_tokens" in message
                    or "temperature" in message
                    or "Unsupported parameter" in message
                    or "Unsupported value" in message
                    or "extra fields not permitted" in message
                )
                if not is_retryable_param_error:
                    raise
                last_exc = exc
        else:
            raise last_exc
        return resp.choices[0].message.content or ""


class MistralClient(LLMClient):
    name = "mistral"

    def __init__(self, model: str = "mistral-large-latest", api_key: Optional[str] = None):
        self._client_kind = "modern"
        self.timeout_ms = int(os.environ.get("MISTRAL_TIMEOUT_MS", "600000"))
        try:
            from mistralai import Mistral  # type: ignore
        except ImportError:
            try:
                from mistralai.client import Mistral  # type: ignore
            except ImportError:
                try:
                    from mistralai.client import MistralClient as Mistral  # type: ignore
                    self._client_kind = "legacy"
                except ImportError as e:
                    raise RuntimeError(
                        "mistralai package not installed or has an unsupported layout. "
                        "`pip install --upgrade mistralai`"
                    ) from e
        try:
            self._client = Mistral(
                api_key=api_key or os.environ.get("MISTRAL_API_KEY"),
                timeout_ms=self.timeout_ms,
            )
        except TypeError:
            self._client = Mistral(api_key=api_key or os.environ.get("MISTRAL_API_KEY"))
        self.model = model

    def complete(self, system, user, *, max_tokens=2048, temperature=0.2):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self._client_kind == "legacy":
            resp = self._client.chat(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )
        else:
            resp = self._client.chat.complete(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                timeout_ms=self.timeout_ms,
            )
        content = resp.choices[0].message.content
        if isinstance(content, list):
            return "".join(getattr(part, "text", str(part)) for part in content)
        return content or ""


class StubLLM(LLMClient):
    """No-network stub that emits a deterministic, structured pseudo-analysis.

    Useful for end-to-end pipeline tests and for demoing the prompt
    construction / retrieval logic without an API key. It does NOT perform
    real reasoning — its outputs are templated.
    """
    name = "stub"

    def complete(self, system, user, *, max_tokens=2048, temperature=0.2):
        # Emit a minimally well-formed structured analysis so downstream
        # parsing (e.g., rubric extraction) doesn't crash.
        return (
            "1. Executive summary\n"
            "[STUB] This is a stubbed response; no model was called. The "
            "pipeline routed prompt of length "
            f"{len(user)} chars to a stub backend.\n\n"
            "2. Core cybersecurity or autonomy-safety risk\n[STUB]\n\n"
            "3. Relevant Assured Autonomy module(s)\n[STUB]\n\n"
            "4. Relevant readings from the corpus, with citations\n[STUB]\n\n"
            "5. Threat model\n- Attacker / failure mode: [STUB]\n- Assets: [STUB]\n"
            "- Assumptions: [STUB]\n- Adversarial vs non-adversarial: [STUB]\n\n"
            "6. Transfer analysis\n- Clean transfers: [STUB]\n- Weak analogies: [STUB]\n\n"
            "7. Practical recommendations\n[STUB]\n\n"
            "8. Questions to ask the client\n[STUB]\n\n"
            "9. Monitoring / testing / governance implications\n[STUB]\n\n"
            "10. Uncertainty and overclaim warnings\n[STUB]\n\n"
            "RUBRIC SCORES (JSON)\n"
            '{"module_mapping_accuracy": 3, "threat_model_clarity": 3, '
            '"use_of_relevant_literature": 3, "practicality_of_recommendations": 3, '
            '"adversarial_treatment": 3, "dataset_shift_treatment": 3, '
            '"monitoring_runtime_treatment": 3, "human_oversight_treatment": 3, '
            '"citation_quality": 3, "overclaim_control": 3}\n'
        )


class OllamaClient(LLMClient):
    """Local Ollama chat backend.

    Requires Ollama running at OLLAMA_BASE_URL, default http://localhost:11434.
    """
    name = "ollama"

    def __init__(self, model: str = "mistral", base_url: Optional[str] = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")

    def complete(self, system, user, *, max_tokens=2048, temperature=0.2):
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. Install/start Ollama "
                f"and run `ollama pull {self.model}`."
            ) from exc
        return data.get("message", {}).get("content", "")


class PeftLocalClient(LLMClient):
    """Local Hugging Face causal LM with a PEFT adapter."""
    name = "peft-local"

    def __init__(self, adapter_dir: str = "models/aa_mistral_lora_adapter"):
        self.adapter_dir = adapter_dir
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        peft_cfg = PeftConfig.from_pretrained(self.adapter_dir)
        base_model = peft_cfg.base_model_name_or_path
        self._tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        quant = None
        if torch.cuda.is_available():
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant,
            device_map="auto",
        )
        self._model = PeftModel.from_pretrained(model, self.adapter_dir)
        self._model.eval()

    def complete(self, system, user, *, max_tokens=2048, temperature=0.2):
        self._load()
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if getattr(self._tokenizer, "chat_template", None):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature >= 0.3,
                temperature=temperature if temperature >= 0.3 else None,
                repetition_penalty=1.15,
                no_repeat_ngram_size=6,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def get_llm(provider: str = "stub", *, model: Optional[str] = None,
            api_key: Optional[str] = None) -> LLMClient:
    """Factory. `provider` in {anthropic, openai, xai/grok, mistral, ollama, peft-local, stub}."""
    p = provider.lower()
    if p == "anthropic":
        return AnthropicClient(model=model or "claude-opus-4-5", api_key=api_key)
    if p == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini", api_key=api_key)
    if p in {"xai", "grok"}:
        return XAIClient(model=model or "grok-4.20-reasoning", api_key=api_key)
    if p == "mistral":
        return MistralClient(model=model or "mistral-large-latest", api_key=api_key)
    if p == "ollama":
        return OllamaClient(model=model or "mistral")
    if p in {"peft-local", "peft"}:
        return PeftLocalClient(adapter_dir=model or "models/aa_mistral_lora_adapter")
    if p == "stub":
        return StubLLM()
    raise ValueError(f"Unknown provider: {provider!r}")
