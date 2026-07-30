from benchmark.qvac_variants import (
    is_local_peer_key,
    is_medical_peer_key,
    is_on_device_key,
    local_medical_only_roster,
    merge_roster,
    panel_rows_for_roster,
)


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
    # default: cloud + 3 generic + 3 MedPsy (medical off)
    assert len(roster) == 7


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
    assert len(off) == 7
    assert not any(is_medical_peer_key(c["key"]) for c in off)

    on = merge_roster(
        cloud,
        triple_qvac=True,
        include_qvac=True,
        include_local_peers=True,
        include_medical_peers=True,
    )
    assert len(on) == 10  # 1 cloud + 3 generic + 3 medical + 3 MedPsy
    med_keys = [c["key"] for c in on if is_medical_peer_key(c["key"])]
    assert med_keys == [
        "local_medgemma",
        "local_biomistral",
        "local_openbiollm",
    ]
    assert all(c.get("band") == "medical_local" for c in on if is_medical_peer_key(c["key"]))


def test_merge_roster_full_twelve():
    cloud = [
        {"key": f"c{i}", "provider": "openrouter", "band": "api", "model": "m"}
        for i in range(3)
    ]
    roster = merge_roster(
        cloud,
        triple_qvac=True,
        include_qvac=True,
        include_local_peers=True,
        include_medical_peers=True,
    )
    assert len(roster) == 12
    rows = panel_rows_for_roster(roster)
    assert len(rows) == 4
    assert all(len(r) == 3 for r in rows)


def test_local_medical_only_roster_six_keys():
    roster = local_medical_only_roster()
    keys = [c["key"] for c in roster]
    assert len(keys) == 6
    assert keys[:3] == [
        "local_medgemma",
        "local_biomistral",
        "local_openbiollm",
    ]
    assert keys[3:] == ["qvac_1_7b", "qvac", "qvac_4b_q8"]


def test_peer_key_helpers_distinguish_bands():
    assert is_local_peer_key("local_gemma")
    assert not is_local_peer_key("local_medgemma")
    assert is_medical_peer_key("local_medgemma")
    assert not is_medical_peer_key("local_gemma")
    assert is_on_device_key("local_medgemma")
    assert is_on_device_key("qvac")
    assert not is_on_device_key("chatgpt")


def test_medical_on_device_only_merge():
    """Medical on-device only preset: MedPsy ×3 + medical ×3, no cloud/generics."""
    roster = merge_roster(
        [],
        triple_qvac=True,
        include_qvac=True,
        include_local_peers=False,
        include_medical_peers=True,
    )
    assert len(roster) == 6
    assert all(is_on_device_key(c["key"]) for c in roster)
    assert not any(is_local_peer_key(c["key"]) for c in roster)
    assert sum(1 for c in roster if is_medical_peer_key(c["key"])) == 3
