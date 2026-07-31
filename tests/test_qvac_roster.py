from lib.model_labels import (
    CURRENT_ROSTER_KEYS,
    DEFAULT_ACTIVE_ROSTER_KEYS,
    OPTIONAL_LEGACY_SLOT_KEYS,
)
from benchmark.qvac_variants import (
    OPTIONAL_LEGACY_SLOT_KEYS as VARIANT_OPTIONAL,
    is_local_peer_key,
    is_medical_peer_key,
    is_on_device_key,
    is_optional_legacy_key,
    local_medical_only_roster,
    merge_roster,
    panel_rows_for_roster,
)


def test_default_active_roster_is_nine():
    assert len(DEFAULT_ACTIVE_ROSTER_KEYS) == 9
    assert set(DEFAULT_ACTIVE_ROSTER_KEYS).isdisjoint(OPTIONAL_LEGACY_SLOT_KEYS)
    assert set(OPTIONAL_LEGACY_SLOT_KEYS) | set(DEFAULT_ACTIVE_ROSTER_KEYS) == set(
        CURRENT_ROSTER_KEYS
    )
    assert set(OPTIONAL_LEGACY_SLOT_KEYS) == set(VARIANT_OPTIONAL)
    for key in OPTIONAL_LEGACY_SLOT_KEYS:
        assert is_optional_legacy_key(key)
        assert key in CURRENT_ROSTER_KEYS


def test_merge_roster_includes_current_api_band():
    cloud = [
        {
            "key": "cloud",
            "provider": "openrouter",
            "band": "api",
            "model": "vendor/model",
        }
    ]
    roster = merge_roster(cloud, triple_qvac=True, include_qvac=True)
    assert roster[0]["key"] == "cloud"
    # default: cloud + Phi + dual MedPsy (medical off, optional legacy off)
    assert [c["key"] for c in roster] == [
        "cloud",
        "local_phi",
        "qvac_1_7b",
        "qvac",
    ]
    assert len(roster) == 4


def test_merge_roster_medical_peers_toggle():
    cloud = [
        {
            "key": "cloud",
            "provider": "openrouter",
            "band": "api",
            "model": "vendor/model",
        }
    ]
    off = merge_roster(
        cloud,
        triple_qvac=True,
        include_qvac=True,
        include_local_peers=True,
        include_medical_peers=False,
    )
    assert len(off) == 4
    assert not any(is_medical_peer_key(c["key"]) for c in off)

    on = merge_roster(
        cloud,
        triple_qvac=True,
        include_qvac=True,
        include_local_peers=True,
        include_medical_peers=True,
    )
    assert len(on) == 7  # 1 cloud + Phi + 3 medical + dual MedPsy
    med_keys = [c["key"] for c in on if is_medical_peer_key(c["key"])]
    assert med_keys == [
        "local_medgemma",
        "local_med42",
        "local_ultramedical",
    ]
    assert all(c.get("band") == "medical_local" for c in on if is_medical_peer_key(c["key"]))


def test_merge_roster_default_full_nine():
    cloud = [
        {"key": k, "provider": "openrouter", "band": "api", "model": "m"}
        for k in ("chatgpt", "claude", "gemini")
    ]
    roster = merge_roster(
        cloud,
        triple_qvac=True,
        include_qvac=True,
        include_local_peers=True,
        include_medical_peers=True,
    )
    keys = [c["key"] for c in roster]
    assert len(roster) == 9
    assert keys == list(DEFAULT_ACTIVE_ROSTER_KEYS)
    assert not any(is_optional_legacy_key(k) for k in keys)
    rows = panel_rows_for_roster(roster)
    assert len(rows) == 3
    assert all(len(r) == 3 for r in rows)


def test_merge_roster_full_twelve_with_optional_legacy():
    cloud = [
        {"key": k, "provider": "openrouter", "band": "api", "model": "m"}
        for k in ("chatgpt", "claude", "gemini")
    ]
    roster = merge_roster(
        cloud,
        triple_qvac=True,
        include_qvac=True,
        include_local_peers=True,
        include_medical_peers=True,
        include_optional_legacy=True,
    )
    assert len(roster) == 12
    keys = {c["key"] for c in roster}
    assert keys == set(CURRENT_ROSTER_KEYS)
    rows = panel_rows_for_roster(roster)
    assert len(rows) == 4
    assert all(len(r) == 3 for r in rows)


def test_merge_roster_selective_optional_keys():
    cloud = [
        {"key": "c0", "provider": "openrouter", "band": "api", "model": "m"}
    ]
    roster = merge_roster(
        cloud,
        triple_qvac=True,
        include_qvac=True,
        include_local_peers=True,
        include_medical_peers=False,
        optional_legacy_keys=["local_gemma", "qvac_4b_q8"],
    )
    keys = [c["key"] for c in roster]
    assert "local_gemma" in keys
    assert "local_llama" not in keys
    assert "qvac_4b_q8" in keys
    assert "local_phi" in keys


def test_local_medical_only_roster_default_five():
    roster = local_medical_only_roster()
    keys = [c["key"] for c in roster]
    assert len(keys) == 5
    assert keys[:3] == [
        "local_medgemma",
        "local_med42",
        "local_ultramedical",
    ]
    assert keys[3:] == ["qvac_1_7b", "qvac"]
    assert "qvac_4b_q8" not in keys


def test_local_medical_only_roster_with_q8():
    roster = local_medical_only_roster(include_optional_legacy=True)
    keys = [c["key"] for c in roster]
    assert keys[3:] == ["qvac_1_7b", "qvac", "qvac_4b_q8"]


def test_peer_key_helpers_distinguish_bands():
    assert is_local_peer_key("local_gemma")
    assert not is_local_peer_key("local_medgemma")
    assert is_medical_peer_key("local_medgemma")
    assert not is_medical_peer_key("local_gemma")
    assert is_on_device_key("local_medgemma")
    assert is_on_device_key("qvac")
    assert not is_on_device_key("chatgpt")


def test_retired_medical_peers_absent_from_roster():
    """BioMistral / OpenBioLLM were replaced; keys must not appear anywhere live."""
    from lib.model_labels import MODEL_LABELS
    from benchmark.qvac_variants import MEDICAL_PEER_SPECS

    retired = ("local_biomistral", "local_openbiollm")
    for key in retired:
        assert key not in CURRENT_ROSTER_KEYS
        assert key not in MODEL_LABELS
        assert not is_medical_peer_key(key)
        assert not is_on_device_key(key)
    assert {s["key"] for s in MEDICAL_PEER_SPECS} == {
        "local_medgemma",
        "local_med42",
        "local_ultramedical",
    }


def test_medical_on_device_only_merge():
    """Medical on-device only preset: dual MedPsy + medical ×3, no cloud/generics."""
    roster = merge_roster(
        [],
        triple_qvac=True,
        include_qvac=True,
        include_local_peers=False,
        include_medical_peers=True,
    )
    assert len(roster) == 5
    assert all(is_on_device_key(c["key"]) for c in roster)
    assert not any(is_local_peer_key(c["key"]) for c in roster)
    assert sum(1 for c in roster if is_medical_peer_key(c["key"])) == 3
    assert [c["key"] for c in roster if c["key"].startswith("qvac") or c["key"] == "qvac"] == [
        "qvac_1_7b",
        "qvac",
    ]
    rows = panel_rows_for_roster(roster)
    assert len(rows) == 2
    assert [len(r) for r in rows] == [3, 2]
