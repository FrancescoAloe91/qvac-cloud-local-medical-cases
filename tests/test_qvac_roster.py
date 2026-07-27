from benchmark.qvac_variants import merge_roster


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
    assert len(roster) == 7
