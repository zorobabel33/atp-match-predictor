"""Player snapshot generation uses the latest ranking and never invents missing values."""

import pytest

from scripts.build_players import build_catalog


def person(identifier, **changes):
    return {"player_id": identifier, "name_first": "Test", "name_last": identifier,
            "ioc": "FRA", "dob": "20000101", "height": "185", "hand": "R", **changes}


def ranking(identifier, rank, date="20260608", points="500"):
    return {"player": identifier, "rank": str(rank), "points": points, "ranking_date": date}


def test_catalog_selects_newest_date_sorts_and_keeps_complete_profiles():
    players = [person("a"), person("b", hand="L"), person("missing", height=""),
               person("unknown", hand="U"), person("young", dob="20200101")]
    rankings = [ranking("a", 1, date="20260101"), ranking("b", 1), ranking("missing", 2),
                ranking("unknown", 3), ranking("young", 4), ranking("a", 5, points="100")]
    catalog = build_catalog(players, rankings)
    assert catalog["ranking_date"] == "2026-06-08"
    assert [row["id"] for row in catalog["players"]] == ["b", "a"]
    assert catalog["players"][0]["hand"] == "L"
    assert catalog["players"][1]["rank_points"] == 100
    assert catalog["players"][0]["age"] == 26.4


@pytest.mark.parametrize("changes", [{"height": ""}, {"height": "300"}, {"dob": "invalid"},
                                    {"hand": "U"}, {"dob": "20200101"}])
def test_catalog_refuses_too_few_usable_profiles(changes):
    with pytest.raises(ValueError, match="At least two"):
        build_catalog([person("a"), person("b", **changes)], [ranking("a", 1), ranking("b", 2)])
