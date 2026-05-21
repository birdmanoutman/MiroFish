"""
Graph memory provider boundary.

MiroFish was originally wired directly to Zep Cloud. This module provides a
Zep-like provider interface so production can default to local Graphiti without
shrinking the graph build, simulation preparation, live memory update, and
report retrieval paths.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from pydantic import BaseModel, Field, create_model

from ..config import Config


OntologyEntityTypes = Dict[str, Any]
OntologyEdgeTypes = Dict[str, Any]


@dataclass(frozen=True)
class GraphMemorySearchOptions:
    """Options shared by Zep-compatible graph search providers."""

    limit: int = 10
    scope: str = "edges"
    reranker: Optional[str] = "cross_encoder"


@runtime_checkable
class GraphMemoryProvider(Protocol):
    """Zep-compatible graph memory provider interface."""

    def create_graph(self, graph_id: str, name: str, description: Optional[str] = None) -> Any:
        ...

    def set_ontology(
        self,
        graph_ids: Sequence[str],
        entities: Optional[OntologyEntityTypes] = None,
        edges: Optional[OntologyEdgeTypes] = None,
        ontology: Optional[Dict[str, Any]] = None,
    ) -> Any:
        ...

    def add_text(self, graph_id: str, data: str) -> Any:
        ...

    def add_text_batch(self, graph_id: str, texts: Iterable[str]) -> List[Any]:
        ...

    def search_graph(
        self,
        graph_id: str,
        query: str,
        options: Optional[GraphMemorySearchOptions] = None,
    ) -> Any:
        ...

    def get_episode(self, episode_uuid: str) -> Any:
        ...

    def list_nodes(self, graph_id: str) -> List[Any]:
        ...

    def list_edges(self, graph_id: str) -> List[Any]:
        ...

    def get_node(self, node_uuid: str) -> Any:
        ...

    def get_node_edges(self, node_uuid: str) -> List[Any]:
        ...

    def delete_graph(self, graph_id: str) -> Any:
        ...


def _object(**kwargs: Any) -> Any:
    return SimpleNamespace(**kwargs)


def _uuid(value: Any) -> str:
    return str(getattr(value, "uuid_", None) or getattr(value, "uuid", None) or "")


def _to_datetime_string(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return value


def _public_attributes(value: Any) -> Dict[str, Any]:
    attributes = dict(value or {})
    return {
        key: val
        for key, val in attributes.items()
        if not str(key).endswith("_embedding") and str(key) not in {"fact_embedding", "name_embedding"}
    }


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    if not cleaned:
        raise ValueError("LLM response is empty; expected JSON object")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _message_text_candidates(message: Any) -> List[str]:
    """Return possible text payloads from OpenAI-compatible message variants."""
    values: List[Any] = [
        getattr(message, "content", None),
        getattr(message, "reasoning_content", None),
    ]
    if hasattr(message, "model_dump"):
        try:
            dumped = message.model_dump()
            values.extend([dumped.get("content"), dumped.get("reasoning_content")])
        except Exception:
            pass

    candidates: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            parts: List[str] = []
            for part in value:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            text = "\n".join(part for part in parts if part).strip()
        else:
            text = str(value).strip()
        if text and text not in seen:
            candidates.append(text)
            seen.add(text)
    return candidates


def _wait_for_glm_rate_slot(model: str) -> None:
    if "glm" not in str(model).lower():
        return

    limit = max(1, Config.GRAPHITI_LLM_RPM_LIMIT)
    window_seconds = max(1.0, Config.GRAPHITI_LLM_RATE_WINDOW_SECONDS)
    min_interval_seconds = max(0.0, Config.GRAPHITI_LLM_MIN_INTERVAL_SECONDS)
    rate_path = Path(Config.GRAPHITI_LLM_RATE_LIMIT_PATH)
    rate_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = rate_path.with_suffix(rate_path.suffix + ".lock")

    while True:
        now = time.monotonic()
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    payload = json.loads(rate_path.read_text(encoding="utf-8"))
                    timestamps = [float(item) for item in payload.get("timestamps", [])]
                except Exception:
                    timestamps = []

                timestamps = [ts for ts in timestamps if now - ts < window_seconds]
                wait_for_interval = 0.0
                if timestamps and min_interval_seconds:
                    wait_for_interval = min_interval_seconds - (now - max(timestamps))
                wait_for_window = 0.0
                if len(timestamps) >= limit:
                    wait_for_window = window_seconds - (now - min(timestamps))

                wait_seconds = max(wait_for_interval, wait_for_window, 0.0)
                if wait_seconds <= 0:
                    timestamps.append(now)
                    rate_path.write_text(
                        json.dumps({"timestamps": timestamps}, separators=(",", ":")),
                        encoding="utf-8",
                    )
                    return
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        time.sleep(max(0.1, wait_seconds))


class _ThrottledAsyncChatCompletions:
    def __init__(self, completions: Any, default_model: str):
        self._completions = completions
        self._default_model = default_model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model") or self._default_model
        await asyncio.to_thread(_wait_for_glm_rate_slot, str(model))
        return await self._completions.create(*args, **kwargs)


class _ThrottledAsyncChat:
    def __init__(self, chat: Any, default_model: str):
        self._chat = chat
        self.completions = _ThrottledAsyncChatCompletions(chat.completions, default_model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _ThrottledAsyncOpenAI:
    def __init__(self, client: Any, default_model: str):
        self._client = client
        self.chat = _ThrottledAsyncChat(client.chat, default_model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _safe_attr_name(attr_name: str, reserved_names: set[str]) -> str:
    if attr_name.lower() in reserved_names:
        return f"entity_{attr_name}"
    return attr_name


class ZepProvider:
    """Zep Cloud implementation of the graph memory boundary."""

    def __init__(self, api_key: Optional[str] = None, client: Optional[Any] = None):
        if client is not None:
            self.client = client
            self.api_key = api_key
            return

        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY 未配置")

        from zep_cloud.client import Zep

        self.client = Zep(api_key=self.api_key)

    def create_graph(self, graph_id: str, name: str, description: Optional[str] = None) -> Any:
        return self.client.graph.create(
            graph_id=graph_id,
            name=name,
            description=description,
        )

    def set_ontology(
        self,
        graph_ids: Sequence[str],
        entities: Optional[OntologyEntityTypes] = None,
        edges: Optional[OntologyEdgeTypes] = None,
        ontology: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if ontology is not None and entities is None and edges is None:
            entities, edges = self._build_zep_ontology(ontology)

        if not entities and not edges:
            return None

        return self.client.graph.set_ontology(
            graph_ids=list(graph_ids),
            entities=entities if entities else None,
            edges=edges if edges else None,
        )

    def _build_zep_ontology(self, ontology: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from typing import Optional as TypingOptional

        from zep_cloud import EntityEdgeSourceTarget
        from zep_cloud.external_clients.ontology import EdgeModel, EntityModel, EntityText

        reserved_names = {"uuid", "name", "group_id", "name_embedding", "summary", "created_at"}
        entity_types: Dict[str, Any] = {}
        edge_definitions: Dict[str, Any] = {}

        for entity_def in ontology.get("entity_types", []):
            name = entity_def["name"]
            description = entity_def.get("description", f"A {name} entity.")
            attrs: Dict[str, Any] = {"__doc__": description}
            annotations: Dict[str, Any] = {}

            for attr_def in entity_def.get("attributes", []):
                attr_name = _safe_attr_name(attr_def["name"], reserved_names)
                attr_desc = attr_def.get("description", attr_name)
                attrs[attr_name] = Field(description=attr_desc, default=None)
                annotations[attr_name] = TypingOptional[EntityText]

            attrs["__annotations__"] = annotations
            entity_class = type(name, (EntityModel,), attrs)
            entity_class.__doc__ = description
            entity_types[name] = entity_class

        for edge_def in ontology.get("edge_types", []):
            name = edge_def["name"]
            description = edge_def.get("description", f"A {name} relationship.")
            attrs = {"__doc__": description}
            annotations = {}

            for attr_def in edge_def.get("attributes", []):
                attr_name = _safe_attr_name(attr_def["name"], reserved_names)
                attr_desc = attr_def.get("description", attr_name)
                attrs[attr_name] = Field(description=attr_desc, default=None)
                annotations[attr_name] = TypingOptional[str]

            attrs["__annotations__"] = annotations
            class_name = "".join(word.capitalize() for word in name.split("_"))
            edge_class = type(class_name, (EdgeModel,), attrs)
            edge_class.__doc__ = description

            source_targets = [
                EntityEdgeSourceTarget(
                    source=st.get("source", "Entity"),
                    target=st.get("target", "Entity"),
                )
                for st in edge_def.get("source_targets", [])
            ]
            if source_targets:
                edge_definitions[name] = (edge_class, source_targets)

        return entity_types, edge_definitions

    def add_text(self, graph_id: str, data: str) -> Any:
        return self.client.graph.add(
            graph_id=graph_id,
            type="text",
            data=data,
        )

    def add_text_batch(self, graph_id: str, texts: Iterable[str]) -> List[Any]:
        from zep_cloud import EpisodeData

        episodes = [EpisodeData(data=text, type="text") for text in texts]
        return self.client.graph.add_batch(
            graph_id=graph_id,
            episodes=episodes,
        )

    def search_graph(
        self,
        graph_id: str,
        query: str,
        options: Optional[GraphMemorySearchOptions] = None,
    ) -> Any:
        options = options or GraphMemorySearchOptions()
        kwargs: Dict[str, Any] = {
            "graph_id": graph_id,
            "query": query,
            "limit": options.limit,
            "scope": options.scope,
        }
        if options.reranker is not None:
            kwargs["reranker"] = options.reranker
        return self.client.graph.search(**kwargs)

    def get_episode(self, episode_uuid: str) -> Any:
        return self.client.graph.episode.get(uuid_=episode_uuid)

    def list_nodes(self, graph_id: str) -> List[Any]:
        from ..utils.zep_paging import fetch_all_nodes

        return fetch_all_nodes(self.client, graph_id)

    def list_edges(self, graph_id: str) -> List[Any]:
        from ..utils.zep_paging import fetch_all_edges

        return fetch_all_edges(self.client, graph_id)

    def get_node(self, node_uuid: str) -> Any:
        return self.client.graph.node.get(uuid_=node_uuid)

    def get_node_edges(self, node_uuid: str) -> List[Any]:
        return self.client.graph.node.get_entity_edges(node_uuid=node_uuid)

    def delete_graph(self, graph_id: str) -> Any:
        return self.client.graph.delete(graph_id=graph_id)


class GraphitiProvider:
    """
    Local Graphiti implementation.

    Graphiti's public REST service does not expose the full Zep Graph surface.
    This provider therefore uses graphiti-core directly for ontology-aware
    episode ingestion and search, then reads Graphiti's Neo4j graph through the
    same core models to expose Zep-like node/edge DTOs.
    """

    _ONTOLOGY_FILE = Path(__file__).resolve().parents[2] / "uploads" / "graph_memory_ontology.json"

    def __init__(self, api_key: Optional[str] = None, client: Optional[Any] = None):
        self.api_key = api_key or Config.GRAPHITI_API_KEY or Config.LLM_API_KEY
        self.client = client
        self.neo4j_uri = Config.GRAPHITI_NEO4J_URI
        self.neo4j_user = Config.GRAPHITI_NEO4J_USER
        self.neo4j_password = Config.GRAPHITI_NEO4J_PASSWORD
        self.llm_base_url = Config.GRAPHITI_LLM_BASE_URL
        self.llm_model = Config.GRAPHITI_LLM_MODEL_NAME
        self.small_model = Config.GRAPHITI_SMALL_MODEL_NAME
        self.embedding_base_url = Config.GRAPHITI_EMBEDDING_BASE_URL
        self.embedding_model = Config.GRAPHITI_EMBEDDING_MODEL
        self.embedding_dim = Config.GRAPHITI_EMBEDDING_DIM
        self.episode_timeout_seconds = Config.GRAPHITI_EPISODE_TIMEOUT_SECONDS
        self.max_coroutines = Config.GRAPHITI_MAX_COROUTINES
        self._ontology_by_graph_id = self._load_ontology_registry()

        if client is None:
            missing = Config.validate_graph_memory()
            if missing:
                raise ValueError("; ".join(missing))

    def _run(self, coro: Any, timeout_seconds: Optional[int] = None) -> Any:
        if timeout_seconds is not None:
            coro = asyncio.wait_for(coro, timeout=timeout_seconds)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: Dict[str, Any] = {}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001
                result["error"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout_seconds + 5 if timeout_seconds is not None else None)
        if thread.is_alive():
            raise TimeoutError(f"Graphiti operation timed out after {timeout_seconds}s")
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _create_graphiti(self) -> Any:
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.client import DEFAULT_MAX_TOKENS
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from openai import AsyncOpenAI
        import httpx

        class MiroFishOpenAIGenericClient(OpenAIGenericClient):
            @staticmethod
            def _json_schema_response_format(response_model: type[BaseModel]) -> Dict[str, Any]:
                schema_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", response_model.__name__)[:64]
                if not schema_name:
                    schema_name = "graphiti_response"
                return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": response_model.model_json_schema(),
                        "strict": False,
                    },
                }

            @staticmethod
            def _fallback_field_value(field_name: str, field: Any, normalized: Dict[str, Any]) -> Any:
                properties = normalized.get("properties")
                if isinstance(properties, dict):
                    if field_name in properties:
                        return properties[field_name]
                    if "summary" in properties:
                        return str(properties["summary"])
                if field_name in {"stock_code", "ticker", "code"}:
                    text = json.dumps(normalized, ensure_ascii=False)
                    import re

                    match = re.search(r"\b(?:SH|SZ)?[0-9]{6}\b", text)
                    return match.group(0) if match else ""
                if field_name in {"main_business", "data_source", "system_name", "org_name", "full_name"}:
                    return str(normalized.get("summary") or normalized.get("name") or "")

                annotation = getattr(field, "annotation", None)
                origin = getattr(annotation, "__origin__", None)
                if annotation in {int, float}:
                    return 0
                if annotation is bool:
                    return False
                if annotation is dict or origin is dict:
                    return {}
                if annotation is list or origin is list:
                    return []
                return str(normalized.get("summary") or "")

            @classmethod
            def _fallback_model_dump(
                cls,
                response_model: type[BaseModel],
                payload: Optional[Dict[str, Any]] = None,
            ) -> Dict[str, Any]:
                normalized = cls._normalize_for_model(payload or {}, response_model)
                return response_model.model_validate(normalized).model_dump()

            @classmethod
            def _normalize_for_model(
                cls,
                payload: Dict[str, Any],
                response_model: type[BaseModel] | None = None,
            ) -> Dict[str, Any]:
                normalized = dict(payload)
                for key, value in list(normalized.items()):
                    if isinstance(value, dict):
                        if key in value:
                            normalized[key] = value[key]
                        elif len(value) == 1:
                            normalized[key] = next(iter(value.values()))
                        elif "value" in value:
                            normalized[key] = value["value"]
                        elif "text" in value:
                            normalized[key] = value["text"]
                if "summary" not in normalized and "properties" in normalized:
                    normalized["summary"] = str(normalized["properties"])
                if response_model is not None:
                    for field_name, field in getattr(response_model, "model_fields", {}).items():
                        if field_name in normalized and normalized[field_name] is not None:
                            continue
                        is_required = getattr(field, "is_required", lambda: False)
                        if is_required():
                            normalized[field_name] = cls._fallback_field_value(
                                field_name,
                                field,
                                normalized,
                            )
                return normalized

            async def _generate_response(
                self,
                messages: list[Any],
                response_model: type[BaseModel] | None = None,
                max_tokens: int = DEFAULT_MAX_TOKENS,
                model_size: Any = None,
            ) -> dict[str, Any]:
                openai_messages = []
                for message in messages:
                    message.content = self._clean_input(message.content)
                    if message.role == "user":
                        openai_messages.append({"role": "user", "content": message.content})
                    elif message.role == "system":
                        openai_messages.append({"role": "system", "content": message.content})

                json_system_message = {
                    "role": "system",
                    "content": (
                        "Return exactly one valid JSON object. Do not include markdown, "
                        "code fences, prose, or explanations outside JSON."
                    ),
                }
                model_name = str(self.model).lower()
                attempts = [
                    (openai_messages, True),
                    ([json_system_message, *openai_messages], True),
                    ([json_system_message, *openai_messages], False),
                ]
                last_error: Optional[BaseException] = None

                for attempt_index, (messages_payload, use_response_format) in enumerate(attempts):
                    output_token_limit = int(os.environ.get("GRAPHITI_LLM_MAX_TOKENS", "1024"))
                    request_kwargs = {
                        "model": self.model,
                        "messages": messages_payload,
                        "temperature": 0 if attempt_index else self.temperature,
                        "max_tokens": min(max_tokens, output_token_limit),
                    }
                    response_format: Optional[Dict[str, Any]] = None
                    if use_response_format:
                        if (
                            "qwen3.6" in model_name
                            and response_model is not None
                            and os.environ.get("GRAPHITI_QWEN_JSON_SCHEMA") == "1"
                        ):
                            response_format = self._json_schema_response_format(response_model)
                        elif "qwen3.6" not in model_name:
                            response_format = {"type": "json_object"}
                    if response_format is not None:
                        request_kwargs["response_format"] = response_format

                    try:
                        response = await self.client.chat.completions.create(**request_kwargs)
                    except Exception as exc:
                        if (
                            response_format is None
                            or ("response_format.type" not in str(exc) and "json_object" not in str(exc))
                        ):
                            last_error = exc
                            continue
                        request_kwargs.pop("response_format", None)
                        try:
                            response = await self.client.chat.completions.create(**request_kwargs)
                        except Exception as retry_exc:
                            last_error = retry_exc
                            continue

                    payload: Optional[Dict[str, Any]] = None
                    for result in _message_text_candidates(response.choices[0].message):
                        try:
                            payload = _parse_json_object(result)
                            break
                        except Exception as exc:
                            last_error = exc
                    if payload is None:
                        continue

                    if response_model is None:
                        return payload
                    try:
                        return response_model.model_validate(payload).model_dump()
                    except Exception as exc:
                        last_error = exc
                        try:
                            normalized = self._normalize_for_model(payload, response_model)
                            return response_model.model_validate(normalized).model_dump()
                        except Exception as normalized_exc:
                            last_error = normalized_exc

                if response_model is not None:
                    return self._fallback_model_dump(response_model)
                if last_error is not None:
                    raise last_error
                raise ValueError("LLM response did not contain a JSON object")

        class MiroFishOpenAIEmbedder(OpenAIEmbedder):
            async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
                if not input_data_list:
                    return []
                return await super().create_batch(input_data_list)

        llm_config = LLMConfig(
            api_key=self.api_key,
            model=self.llm_model,
            small_model=self.small_model,
            base_url=self.llm_base_url,
        )
        embedder_config = OpenAIEmbedderConfig(
            api_key=self.api_key,
            base_url=self.embedding_base_url,
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_dim,
        )
        llm_openai_client = _ThrottledAsyncOpenAI(
            AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.llm_base_url,
                http_client=httpx.AsyncClient(trust_env=False),
            ),
            self.llm_model,
        )
        embedding_openai_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.embedding_base_url,
            http_client=httpx.AsyncClient(trust_env=False),
        )
        reranker_openai_client = _ThrottledAsyncOpenAI(
            AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.llm_base_url,
                http_client=httpx.AsyncClient(trust_env=False),
            ),
            self.llm_model,
        )

        graphiti = Graphiti(
            self.neo4j_uri,
            self.neo4j_user,
            self.neo4j_password,
            llm_client=MiroFishOpenAIGenericClient(config=llm_config, client=llm_openai_client),
            embedder=MiroFishOpenAIEmbedder(config=embedder_config, client=embedding_openai_client),
            cross_encoder=OpenAIRerankerClient(config=llm_config, client=reranker_openai_client),
            max_coroutines=self.max_coroutines,
        )
        graphiti._mirofish_openai_clients = [llm_openai_client, embedding_openai_client, reranker_openai_client]
        return graphiti

    async def _with_graphiti(self, func: Callable[[Any], Any]) -> Any:
        graphiti = self._create_graphiti()
        try:
            return await func(graphiti)
        finally:
            try:
                await asyncio.wait_for(graphiti.close(), timeout=5)
            except Exception:
                pass
            for client in getattr(graphiti, "_mirofish_openai_clients", []):
                try:
                    await asyncio.wait_for(client.close(), timeout=2)
                except Exception:
                    pass

    @classmethod
    def _load_ontology_registry(cls) -> Dict[str, Dict[str, Any]]:
        try:
            if cls._ONTOLOGY_FILE.exists():
                return json.loads(cls._ONTOLOGY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return {}

    @classmethod
    def _save_ontology_registry(cls, registry: Dict[str, Dict[str, Any]]) -> None:
        cls._ONTOLOGY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cls._ONTOLOGY_FILE.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(cls._ONTOLOGY_FILE)

    @staticmethod
    def _node_to_object(node: Any) -> Any:
        node_uuid = _uuid(node)
        return _object(
            uuid_=node_uuid,
            uuid=node_uuid,
            name=getattr(node, "name", "") or "",
            labels=list(getattr(node, "labels", []) or []),
            summary=getattr(node, "summary", "") or "",
            attributes=_public_attributes(getattr(node, "attributes", {}) or {}),
            created_at=_to_datetime_string(getattr(node, "created_at", None)),
        )

    @staticmethod
    def _edge_to_object(edge: Any) -> Any:
        edge_uuid = _uuid(edge)
        episodes = getattr(edge, "episodes", None) or getattr(edge, "episode_ids", None) or []
        if episodes and not isinstance(episodes, list):
            episodes = [episodes]
        episodes = [str(e) for e in episodes]
        return _object(
            uuid_=edge_uuid,
            uuid=edge_uuid,
            name=getattr(edge, "name", "") or "",
            fact=getattr(edge, "fact", "") or "",
            source_node_uuid=getattr(edge, "source_node_uuid", "") or "",
            target_node_uuid=getattr(edge, "target_node_uuid", "") or "",
            attributes=_public_attributes(getattr(edge, "attributes", {}) or {}),
            created_at=_to_datetime_string(getattr(edge, "created_at", None)),
            valid_at=_to_datetime_string(getattr(edge, "valid_at", None)),
            invalid_at=_to_datetime_string(getattr(edge, "invalid_at", None)),
            expired_at=_to_datetime_string(getattr(edge, "expired_at", None)),
            episodes=episodes,
            episode_ids=episodes,
        )

    @staticmethod
    def _episode_to_object(episode_uuid: str, episode: Optional[Any], processed: bool) -> Any:
        return _object(
            uuid_=episode_uuid,
            uuid=episode_uuid,
            processed=processed,
            name=getattr(episode, "name", "") if episode else "",
            content=getattr(episode, "content", "") if episode else "",
            created_at=_to_datetime_string(getattr(episode, "created_at", None)) if episode else None,
            valid_at=_to_datetime_string(getattr(episode, "valid_at", None)) if episode else None,
        )

    def _build_graphiti_ontology(
        self, graph_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[Tuple[str, str], List[str]]]]:
        ontology = self._ontology_by_graph_id.get(graph_id)
        if not ontology:
            return None, None, None

        reserved_names = {
            "uuid",
            "name",
            "group_id",
            "labels",
            "created_at",
            "name_embedding",
            "summary",
            "attributes",
        }
        entity_types: Dict[str, Any] = {}
        edge_types: Dict[str, Any] = {}
        edge_type_map: Dict[Tuple[str, str], List[str]] = {}

        for entity_def in ontology.get("entity_types", []):
            name = entity_def.get("name")
            if not name:
                continue
            fields: Dict[str, Tuple[Any, Any]] = {}
            for attr_def in entity_def.get("attributes", []):
                attr_name = _safe_attr_name(attr_def.get("name", "value"), reserved_names)
                attr_desc = attr_def.get("description", attr_name)
                fields[attr_name] = (Optional[str], Field(default=None, description=attr_desc))
            model = create_model(name, __base__=BaseModel, **fields)
            model.__doc__ = entity_def.get("description", f"A {name} entity.")
            entity_types[name] = model

        for edge_def in ontology.get("edge_types", []):
            name = edge_def.get("name")
            if not name:
                continue
            class_name = "".join(word.capitalize() for word in str(name).split("_")) or str(name)
            fields = {}
            for attr_def in edge_def.get("attributes", []):
                attr_name = _safe_attr_name(attr_def.get("name", "value"), reserved_names)
                attr_desc = attr_def.get("description", attr_name)
                fields[attr_name] = (Optional[str], Field(default=None, description=attr_desc))
            model = create_model(class_name, __base__=BaseModel, **fields)
            model.__doc__ = edge_def.get("description", f"A {name} relationship.")
            edge_types[name] = model

            source_targets = edge_def.get("source_targets", []) or [{"source": "Entity", "target": "Entity"}]
            for source_target in source_targets:
                source = source_target.get("source", "Entity")
                target = source_target.get("target", "Entity")
                edge_type_map.setdefault((source, target), []).append(name)

        return entity_types or None, edge_types or None, edge_type_map or None

    def create_graph(self, graph_id: str, name: str, description: Optional[str] = None) -> Any:
        if self.client is not None:
            return _object(graph_id=graph_id, uuid_=graph_id, name=name, description=description, provider="graphiti")

        async def create(graphiti: Any) -> Any:
            await graphiti.build_indices_and_constraints()
            return _object(
                graph_id=graph_id,
                uuid_=graph_id,
                name=name,
                description=description,
                provider="graphiti",
            )

        return self._run(self._with_graphiti(create))

    def set_ontology(
        self,
        graph_ids: Sequence[str],
        entities: Optional[OntologyEntityTypes] = None,
        edges: Optional[OntologyEdgeTypes] = None,
        ontology: Optional[Dict[str, Any]] = None,
    ) -> Any:
        raw_ontology = ontology
        if raw_ontology is None:
            raw_ontology = {
                "entity_types": [
                    {"name": str(name), "description": getattr(model, "__doc__", "") or str(name)}
                    for name, model in (entities or {}).items()
                ],
                "edge_types": [
                    {"name": str(name), "description": getattr(model, "__doc__", "") or str(name)}
                    for name, model in (edges or {}).items()
                ],
            }

        for graph_id in graph_ids:
            self._ontology_by_graph_id[graph_id] = raw_ontology
        self._save_ontology_registry(self._ontology_by_graph_id)
        return _object(graph_ids=list(graph_ids), provider="graphiti", applied=True)

    def add_text(self, graph_id: str, data: str) -> Any:
        return self.add_text_batch(graph_id, [data])[0]

    def add_text_batch(self, graph_id: str, texts: Iterable[str]) -> List[Any]:
        text_list = [text for text in texts if text is not None and str(text).strip()]
        if self.client is not None:
            episodes = []
            for _ in text_list:
                episode_uuid = str(uuid.uuid4())
                episodes.append(_object(uuid_=episode_uuid, uuid=episode_uuid, processed=True))
            return episodes

        async def add(graphiti: Any) -> List[Any]:
            from graphiti_core.nodes import EpisodeType

            entity_types, edge_types, edge_type_map = self._build_graphiti_ontology(graph_id)
            episodes = []
            for index, text in enumerate(text_list, 1):
                try:
                    result = await asyncio.wait_for(
                        graphiti.add_episode(
                            group_id=graph_id,
                            name=f"mirofish text episode {index}",
                            episode_body=text,
                            source_description="MiroFish GraphitiProvider",
                            reference_time=datetime.now(timezone.utc),
                            source=EpisodeType.text,
                            entity_types=entity_types,
                            edge_types=edge_types,
                            edge_type_map=edge_type_map,
                        ),
                        timeout=self.episode_timeout_seconds,
                    )
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"Graphiti episode ingestion timed out after {self.episode_timeout_seconds}s "
                        f"for graph {graph_id} episode {index}/{len(text_list)}"
                    ) from exc
                episode_uuid = result.episode.uuid
                episodes.append(
                    _object(uuid_=episode_uuid, uuid=episode_uuid, processed=True, provider="graphiti")
                )
            return episodes

        operation_timeout = max(30, self.episode_timeout_seconds * max(1, len(text_list)) + 15)
        return self._run(self._with_graphiti(add), timeout_seconds=operation_timeout)

    def _search_config(self, scope: str, reranker: Optional[str]) -> Any:
        from graphiti_core.search.search_config_recipes import (
            COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
            COMBINED_HYBRID_SEARCH_RRF,
            EDGE_HYBRID_SEARCH_CROSS_ENCODER,
            EDGE_HYBRID_SEARCH_RRF,
            NODE_HYBRID_SEARCH_CROSS_ENCODER,
            NODE_HYBRID_SEARCH_RRF,
        )

        # Local LiteLLM/GLM backends do not reliably expose the logprobs shape
        # required by Graphiti's OpenAI cross-encoder reranker. RRF keeps hybrid
        # retrieval fully local and avoids forced fallback in report tools.
        use_cross_encoder = False
        if scope == "nodes":
            return NODE_HYBRID_SEARCH_CROSS_ENCODER if use_cross_encoder else NODE_HYBRID_SEARCH_RRF
        if scope in {"both", "all"}:
            return COMBINED_HYBRID_SEARCH_CROSS_ENCODER if use_cross_encoder else COMBINED_HYBRID_SEARCH_RRF
        return EDGE_HYBRID_SEARCH_CROSS_ENCODER if use_cross_encoder else EDGE_HYBRID_SEARCH_RRF

    def search_graph(
        self,
        graph_id: str,
        query: str,
        options: Optional[GraphMemorySearchOptions] = None,
    ) -> Any:
        options = options or GraphMemorySearchOptions()
        if self.client is not None:
            return _object(edges=[], nodes=[], raw=None)

        async def search(graphiti: Any) -> Any:
            config = self._search_config(options.scope, options.reranker)
            result = await graphiti._search(  # Graphiti keeps this as the public recipe-based search entry.
                query=query,
                config=config,
                group_ids=[graph_id],
            )
            edges = [self._edge_to_object(edge) for edge in getattr(result, "edges", [])[: options.limit]]
            nodes = [self._node_to_object(node) for node in getattr(result, "nodes", [])[: options.limit]]
            return _object(edges=edges, nodes=nodes, raw=result)

        return self._run(self._with_graphiti(search))

    def get_episode(self, episode_uuid: str) -> Any:
        if self.client is not None:
            return _object(uuid_=episode_uuid, uuid=episode_uuid, processed=True, provider="graphiti")

        async def read(graphiti: Any) -> Any:
            from graphiti_core.errors import NodeNotFoundError
            from graphiti_core.nodes import EpisodicNode

            try:
                episode = await EpisodicNode.get_by_uuid(graphiti.driver, episode_uuid)
                return self._episode_to_object(episode_uuid, episode, True)
            except NodeNotFoundError:
                return self._episode_to_object(episode_uuid, None, False)

        return self._run(self._with_graphiti(read))

    def list_nodes(self, graph_id: str) -> List[Any]:
        if self.client is not None:
            return []

        async def read(graphiti: Any) -> List[Any]:
            from graphiti_core.nodes import EntityNode

            nodes = await EntityNode.get_by_group_ids(graphiti.driver, [graph_id])
            return [self._node_to_object(node) for node in nodes]

        return self._run(self._with_graphiti(read))

    def list_edges(self, graph_id: str) -> List[Any]:
        if self.client is not None:
            return []

        async def read(graphiti: Any) -> List[Any]:
            from graphiti_core.edges import EntityEdge
            from graphiti_core.errors import GroupsEdgesNotFoundError

            try:
                edges = await EntityEdge.get_by_group_ids(graphiti.driver, [graph_id])
            except GroupsEdgesNotFoundError:
                edges = []
            return [self._edge_to_object(edge) for edge in edges]

        return self._run(self._with_graphiti(read))

    def get_node(self, node_uuid: str) -> Any:
        if self.client is not None:
            return None

        async def read(graphiti: Any) -> Any:
            from graphiti_core.errors import NodeNotFoundError
            from graphiti_core.nodes import EntityNode

            try:
                node = await EntityNode.get_by_uuid(graphiti.driver, node_uuid)
                return self._node_to_object(node)
            except NodeNotFoundError:
                return None

        return self._run(self._with_graphiti(read))

    def get_node_edges(self, node_uuid: str) -> List[Any]:
        if self.client is not None:
            return []

        async def read(graphiti: Any) -> List[Any]:
            records, _, _ = await graphiti.driver.execute_query(
                """
                MATCH (:Entity {uuid: $node_uuid})-[e:RELATES_TO]-(:Entity)
                RETURN e.uuid AS uuid
                """,
                node_uuid=node_uuid,
                database_="neo4j",
                routing_="r",
            )
            if not records:
                return []

            from graphiti_core.edges import EntityEdge

            edges = []
            for record in records:
                try:
                    edges.append(await EntityEdge.get_by_uuid(graphiti.driver, record["uuid"]))
                except Exception:
                    continue
            return [self._edge_to_object(edge) for edge in edges]

        return self._run(self._with_graphiti(read))

    def delete_graph(self, graph_id: str) -> Any:
        if self.client is not None:
            return _object(graph_id=graph_id, deleted=True)

        async def delete(graphiti: Any) -> Any:
            from graphiti_core.edges import EntityEdge
            from graphiti_core.errors import GroupsEdgesNotFoundError
            from graphiti_core.nodes import EntityNode, EpisodicNode

            try:
                edges = await EntityEdge.get_by_group_ids(graphiti.driver, [graph_id])
            except GroupsEdgesNotFoundError:
                edges = []
            nodes = await EntityNode.get_by_group_ids(graphiti.driver, [graph_id])
            episodes = await EpisodicNode.get_by_group_ids(graphiti.driver, [graph_id])

            for edge in edges:
                await edge.delete(graphiti.driver)
            for node in nodes:
                await node.delete(graphiti.driver)
            for episode in episodes:
                await episode.delete(graphiti.driver)

            self._ontology_by_graph_id.pop(graph_id, None)
            self._save_ontology_registry(self._ontology_by_graph_id)
            return _object(graph_id=graph_id, deleted=True)

        return self._run(self._with_graphiti(delete))


def create_graph_memory_provider(api_key: Optional[str] = None, provider_name: Optional[str] = None) -> GraphMemoryProvider:
    provider = (provider_name or Config.MEMORY_PROVIDER or "graphiti").lower()
    if provider == "zep":
        return ZepProvider(api_key=api_key)
    if provider == "graphiti":
        return GraphitiProvider(api_key=api_key)
    raise ValueError(f"Unsupported MEMORY_PROVIDER: {provider}")


__all__ = [
    "GraphMemoryProvider",
    "GraphMemorySearchOptions",
    "GraphitiProvider",
    "ZepProvider",
    "create_graph_memory_provider",
]
