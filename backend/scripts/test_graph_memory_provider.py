"""Lightweight contract checks for graph_memory_provider."""

from app.services.graph_memory_provider import (
    GraphMemoryProvider,
    GraphMemorySearchOptions,
    GraphitiProvider,
    ZepProvider,
)


class FakeEpisodeAPI:
    def __init__(self):
        self.last_uuid = None

    def get(self, uuid_):
        self.last_uuid = uuid_
        return {"uuid": uuid_, "processed": True}


class FakeGraphAPI:
    def __init__(self):
        self.episode = FakeEpisodeAPI()
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return {"created": kwargs["graph_id"]}

    def set_ontology(self, **kwargs):
        self.calls.append(("set_ontology", kwargs))
        return {"ontology": kwargs["graph_ids"]}

    def add(self, **kwargs):
        self.calls.append(("add", kwargs))
        return {"added": kwargs["data"]}

    def add_batch(self, **kwargs):
        self.calls.append(("add_batch", kwargs))
        return kwargs["episodes"]

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return {"query": kwargs["query"], "limit": kwargs["limit"]}


class FakeZepClient:
    def __init__(self):
        self.graph = FakeGraphAPI()


class FakeGraphitiClient:
    pass


def test_zep_provider_is_graph_memory_provider():
    provider = ZepProvider(client=FakeZepClient())

    assert isinstance(provider, GraphMemoryProvider)
    assert provider.create_graph("g1", "Graph", "desc") == {"created": "g1"}
    assert provider.set_ontology(["g1"], entities={"Person": object}) == {"ontology": ["g1"]}
    assert provider.add_text("g1", "hello") == {"added": "hello"}
    assert len(provider.add_text_batch("g1", ["a", "b"])) == 2
    assert provider.search_graph("g1", "needle", GraphMemorySearchOptions(limit=3)) == {
        "query": "needle",
        "limit": 3,
    }
    assert provider.get_episode("ep1") == {"uuid": "ep1", "processed": True}


def test_graphiti_provider_is_graph_memory_provider():
    provider = GraphitiProvider(client=FakeGraphitiClient())

    assert isinstance(provider, GraphMemoryProvider)
    created = provider.create_graph("g1", "Graph")
    assert created.graph_id == "g1"

    ontology = provider.set_ontology(["g1"], entities={"Company": object})
    assert ontology.applied is True

    episode = provider.add_text("g1", "hello")
    assert episode.uuid

    episodes = provider.add_text_batch("g1", ["a", "b"])
    assert len(episodes) == 2

    result = provider.search_graph("g1", "technology sentiment", GraphMemorySearchOptions(limit=2))
    assert result.edges == []
    assert result.nodes == []

    assert provider.get_episode("ep1").processed is True
    assert provider.list_nodes("g1") == []
    assert provider.list_edges("g1") == []


if __name__ == "__main__":
    test_zep_provider_is_graph_memory_provider()
    test_graphiti_provider_is_graph_memory_provider()
    print("graph_memory_provider contract checks passed")
