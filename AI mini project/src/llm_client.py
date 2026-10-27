"""
Cloud LLM client abstraction for DNS exfiltration analysis.
Supports Google Gemini, OpenAI, and a deterministic mock provider for offline testing.
All API calls are wrapped with retry logic, timeout enforcement, response caching,
cost tracking, and structured JSON output parsing.
"""
import json
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


class LLMResponse:
    """Structured container for an LLM API response."""

    def __init__(
        self,
        raw_text: str,
        parsed_json: Optional[Dict[str, Any]],
        provider: str,
        model: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached: bool = False,
        error: Optional[str] = None,
    ):
        self.raw_text = raw_text
        self.parsed_json = parsed_json
        self.provider = provider
        self.model = model
        self.latency_ms = round(latency_ms, 2)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached = cached
        self.error = error

    @property
    def success(self) -> bool:
        return self.parsed_json is not None and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "parsed_json": self.parsed_json,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached": self.cached,
            "error": self.error,
            "success": self.success,
        }


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Attempts to parse JSON from raw LLM output, handling markdown fences."""
    cleaned = text.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


class MockLLMProvider:
    """
    Deterministic mock LLM for offline testing and demo mode.
    Returns structured responses based on feature heuristics without any network call.
    """

    MODEL_NAME = "mock-deterministic-v1"

    def analyze(self, features: Dict[str, Any], system_prompt: str) -> LLMResponse:
        start = time.perf_counter()

        entropy = float(features.get("shannon_entropy", 0))
        sub_len = int(features.get("subdomain_length", 0))
        hex_ratio = float(features.get("hex_ratio", 0))
        risk_score = float(features.get("risk_score", 0))

        if risk_score >= 0.75 or (entropy > 4.0 and sub_len > 30):
            classification = "likely_exfiltration"
            evidence = [
                f"shannon_entropy={entropy} (elevated)",
                f"subdomain_length={sub_len} (abnormally long)",
                f"hex_ratio={hex_ratio}",
            ]
            limitations = ["Mock provider: no semantic reasoning applied"]
            next_check = "Inspect raw PCAP for payload reconstruction"
        elif risk_score >= 0.40 or entropy > 3.5:
            classification = "suspicious"
            evidence = [
                f"shannon_entropy={entropy} (moderately high)",
                f"risk_score={risk_score}",
            ]
            limitations = [
                "Mock provider: threshold-based only",
                "CDN or DKIM records can produce similar entropy",
            ]
            next_check = "Compare with historical baseline for this domain"
        elif entropy < 1.0 and sub_len < 5:
            classification = "benign"
            evidence = [
                f"shannon_entropy={entropy} (very low)",
                f"subdomain_length={sub_len} (short)",
            ]
            limitations = ["Mock provider: no context window"]
            next_check = "No action required"
        else:
            classification = "insufficient_evidence"
            evidence = [f"risk_score={risk_score}", f"entropy={entropy}"]
            limitations = ["Mock provider cannot reason about ambiguous cases"]
            next_check = "Route to live LLM provider for deeper analysis"

        result = {
            "classification": classification,
            "evidence": evidence,
            "missing_evidence": ["temporal_pattern", "historical_baseline"],
            "limitations": limitations,
            "recommended_next_check": next_check,
        }

        elapsed_ms = (time.perf_counter() - start) * 1000
        return LLMResponse(
            raw_text=json.dumps(result),
            parsed_json=result,
            provider="mock",
            model=self.MODEL_NAME,
            latency_ms=elapsed_ms,
            prompt_tokens=0,
            completion_tokens=0,
            cached=False,
        )


class GeminiProvider:
    """Google Gemini API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 15,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError(
                "Gemini API key not configured. Set GEMINI_API_KEY in .env"
            )

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("google-generativeai package required: pip install google-generativeai")

    def analyze(self, features: Dict[str, Any], system_prompt: str) -> LLMResponse:
        import google.generativeai as genai

        user_content = (
            "Analyze this DNS traffic summary and respond with only the JSON object:\n\n"
            + json.dumps(features, indent=2)
        )

        last_error = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                response = self._model.generate_content(
                    [system_prompt, user_content],
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature
                    ),
                    request_options={"timeout": self.timeout},
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                raw_text = response.text
                parsed = _extract_json_from_text(raw_text)

                prompt_tokens = getattr(
                    getattr(response, "usage_metadata", None), "prompt_token_count", 0
                ) or 0
                completion_tokens = getattr(
                    getattr(response, "usage_metadata", None),
                    "candidates_token_count",
                    0,
                ) or 0

                return LLMResponse(
                    raw_text=raw_text,
                    parsed_json=parsed,
                    provider="gemini",
                    model=self.model,
                    latency_ms=elapsed_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    error=None if parsed else "Failed to parse JSON from response",
                )
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                continue

        return LLMResponse(
            raw_text="",
            parsed_json=None,
            provider="gemini",
            model=self.model,
            latency_ms=0,
            error=f"All {self.max_retries + 1} attempts failed: {last_error}",
        )


class OpenAIProvider:
    """OpenAI API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout: int = 15,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key or self.api_key == "your_openai_api_key_here":
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_API_KEY in .env"
            )

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def analyze(self, features: Dict[str, Any], system_prompt: str) -> LLMResponse:
        user_content = (
            "Analyze this DNS traffic summary and respond with only the JSON object:\n\n"
            + json.dumps(features, indent=2)
        )

        last_error = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self.temperature,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                raw_text = response.choices[0].message.content or ""
                parsed = _extract_json_from_text(raw_text)

                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0

                return LLMResponse(
                    raw_text=raw_text,
                    parsed_json=parsed,
                    provider="openai",
                    model=self.model,
                    latency_ms=elapsed_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    error=None if parsed else "Failed to parse JSON from response",
                )
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                continue

        return LLMResponse(
            raw_text="",
            parsed_json=None,
            provider="openai",
            model=self.model,
            latency_ms=0,
            error=f"All {self.max_retries + 1} attempts failed: {last_error}",
        )


class ResponseCache:
    """Simple in-memory LRU cache keyed on feature hash to avoid duplicate API calls."""

    def __init__(self, enabled: bool = True, max_size: int = 500):
        self.enabled = enabled
        self._cache: Dict[str, LLMResponse] = {}
        self._max_size = max_size

    def _make_key(self, features: Dict[str, Any], provider: str) -> str:
        serialized = json.dumps(features, sort_keys=True, default=str)
        return hashlib.md5(f"{provider}:{serialized}".encode()).hexdigest()

    def get(self, features: Dict[str, Any], provider: str) -> Optional[LLMResponse]:
        if not self.enabled:
            return None
        key = self._make_key(features, provider)
        cached = self._cache.get(key)
        if cached:
            cached.cached = True
        return cached

    def put(self, features: Dict[str, Any], provider: str, response: LLMResponse):
        if not self.enabled:
            return
        if len(self._cache) >= self._max_size:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        key = self._make_key(features, provider)
        self._cache[key] = response


class LLMClient:
    """
    Unified client that dispatches to the selected provider.
    Manages caching, cost tracking, and call budgeting.
    """

    PRICING = {
        "gemini-1.5-flash": {"prompt_per_1k": 0.000075, "completion_per_1k": 0.00030},
        "gemini-1.5-pro": {"prompt_per_1k": 0.00125, "completion_per_1k": 0.00500},
        "gpt-4o-mini": {"prompt_per_1k": 0.00015, "completion_per_1k": 0.00060},
        "mock-deterministic-v1": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
    }

    def __init__(
        self,
        provider: str = "mock",
        cache_enabled: bool = True,
        max_calls: int = 100,
        system_prompt_path: str = "prompts/dns_detection_system.txt",
        **provider_kwargs,
    ):
        self.provider_name = provider.lower()
        self.max_calls = max_calls
        self._call_count = 0
        self._total_cost = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._latencies: list[float] = []
        self.cache = ResponseCache(enabled=cache_enabled)

        # Load system prompt
        prompt_path = Path(system_prompt_path)
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            self.system_prompt = "You are a DNS traffic analysis assistant. Respond with valid JSON only."

        # Initialize provider
        if self.provider_name == "mock":
            self._provider = MockLLMProvider()
        elif self.provider_name == "gemini":
            self._provider = GeminiProvider(**provider_kwargs)
        elif self.provider_name == "openai":
            self._provider = OpenAIProvider(**provider_kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'mock', 'gemini', or 'openai'.")

    def analyze(self, features: Dict[str, Any]) -> LLMResponse:
        """Send features to LLM for analysis with budget and cache enforcement."""
        # Check budget
        if self._call_count >= self.max_calls:
            return LLMResponse(
                raw_text="",
                parsed_json=None,
                provider=self.provider_name,
                model=getattr(self._provider, "model", getattr(self._provider, "MODEL_NAME", "unknown")),
                latency_ms=0,
                error=f"API call budget exhausted ({self.max_calls} calls). Increase max_calls or use cached results.",
            )

        # Check cache
        cached = self.cache.get(features, self.provider_name)
        if cached:
            return cached

        # Make API call
        response = self._provider.analyze(features, self.system_prompt)

        # Track metrics
        self._call_count += 1
        self._latencies.append(response.latency_ms)
        self._total_prompt_tokens += response.prompt_tokens
        self._total_completion_tokens += response.completion_tokens

        model_name = response.model
        pricing = self.PRICING.get(model_name, {"prompt_per_1k": 0, "completion_per_1k": 0})
        call_cost = (
            (response.prompt_tokens / 1000) * pricing["prompt_per_1k"]
            + (response.completion_tokens / 1000) * pricing["completion_per_1k"]
        )
        self._total_cost += call_cost

        # Cache successful responses
        if response.success:
            self.cache.put(features, self.provider_name, response)

        return response

    def get_usage_stats(self) -> Dict[str, Any]:
        """Returns cumulative usage statistics."""
        return {
            "provider": self.provider_name,
            "total_calls": self._call_count,
            "budget_remaining": self.max_calls - self._call_count,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "avg_latency_ms": round(sum(self._latencies) / max(len(self._latencies), 1), 2),
            "cache_hits": sum(1 for _ in self.cache._cache) if self.cache.enabled else 0,
        }
