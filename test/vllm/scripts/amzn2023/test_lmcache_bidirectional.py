"""
LMCache bidirectional NIXL cache probe feature verification.

Validates that the custom LMCache build includes the bidirectional
PD feature from LMCache/LMCache#2972 (commit 7f60057).

These are lightweight unit tests — no GPU or multi-node setup required.
"""



def test_config_fields():
    """pd_bidirectional and pd_peer_query_port config options exist."""
    from lmcache.v1.config import LMCacheEngineConfig

    config = LMCacheEngineConfig.from_defaults()
    assert config.pd_bidirectional is False, "pd_bidirectional should default to False"

    config = LMCacheEngineConfig.from_defaults(pd_bidirectional=True)
    assert config.pd_bidirectional is True, "pd_bidirectional should be settable to True"

    config = LMCacheEngineConfig.from_defaults()
    assert config.pd_peer_query_port is None, "pd_peer_query_port should default to None"

    ports = [7500, 7501, 7502, 7503]
    config = LMCacheEngineConfig.from_defaults(pd_peer_query_port=ports)
    assert config.pd_peer_query_port == ports, "pd_peer_query_port should be settable"

    print("PASSED: config fields")


def test_cache_query_messages():
    """CacheQueryRequest/CacheQueryResponse serialize correctly."""
    import msgspec
    from lmcache.v1.storage_backend.pd_backend import (
        CacheQueryRequest,
        CacheQueryResponse,
        PDMsg,
    )

    req = CacheQueryRequest(keys=["key1", "key2", "key3"])
    encoded = msgspec.msgpack.encode(req)
    decoded = msgspec.msgpack.decode(encoded, type=PDMsg)
    assert isinstance(decoded, CacheQueryRequest)
    assert decoded.keys == ["key1", "key2", "key3"]

    resp = CacheQueryResponse(cached_keys=["key1"], cached_indexes=[0])
    encoded = msgspec.msgpack.encode(resp)
    decoded = msgspec.msgpack.decode(encoded, type=PDMsg)
    assert isinstance(decoded, CacheQueryResponse)
    assert decoded.cached_keys == ["key1"]
    assert decoded.cached_indexes == [0]

    # Empty response (no hits)
    resp = CacheQueryResponse(cached_keys=[], cached_indexes=[])
    encoded = msgspec.msgpack.encode(resp)
    decoded = msgspec.msgpack.decode(encoded, type=PDMsg)
    assert len(decoded.cached_keys) == 0

    print("PASSED: cache query messages")


def test_probe_method_exists():
    """_probe_decoder_cache method exists in the vLLM adapter."""
    import inspect

    from lmcache.integration.vllm import vllm_v1_adapter

    src = inspect.getsource(vllm_v1_adapter)
    assert "_probe_decoder_cache" in src, "_probe_decoder_cache not found in adapter"
    assert "bidir_enabled" in src, "bidir_enabled not found in adapter"
    assert "Bidirectional NIXL cache probe" in src, "probe log message not found"

    print("PASSED: probe method exists")


if __name__ == "__main__":
    test_config_fields()
    test_cache_query_messages()
    test_probe_method_exists()
    print("\nAll LMCache bidirectional tests PASSED")
