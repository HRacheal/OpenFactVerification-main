"""Verdict rules (pure logic, no AI involved)."""

from factcheck.core.Verdict import decide_verdict, evidence_weight, evidence_notes


def E(rel, domain, source_type="news", basis="explicit", full=True, date=None):
    return {"relationship": rel, "domain": domain, "url": f"https://{domain}/x", "source_type": source_type,
            "basis": basis, "full_text_fetched": full, "published_date": date}


def test_no_evidence_is_insufficient_not_false():
    v, reason, s = decide_verdict([])
    assert v == "INSUFFICIENT_EVIDENCE" and "does not mean the claim is false" in reason
    v, _, _ = decide_verdict([E("CONTEXT", "a.com"), E("IRRELEVANT", "b.com")])
    assert v == "INSUFFICIENT_EVIDENCE"


def test_single_explicit_official_source_supports():
    v, _, s = decide_verdict([E("SUPPORTS", "statistics.go.ug", "official")])
    assert v == "SUPPORTED" and s["support_weight"] == 1.0


def test_one_news_source_alone_is_not_enough_but_two_independent_are():
    assert decide_verdict([E("SUPPORTS", "monitor.co.ug")])[0] == "INSUFFICIENT_EVIDENCE"
    assert decide_verdict([E("SUPPORTS", "monitor.co.ug"), E("SUPPORTS", "newvision.co.ug")])[0] == "SUPPORTED"


def test_same_domain_counts_once():
    evs = [E("REFUTES", "monitor.co.ug") for _ in range(5)]
    v, _, s = decide_verdict(evs)
    assert v == "INSUFFICIENT_EVIDENCE" and s["contradict_sources"] == 1


def test_contradicted():
    evs = [E("REFUTES", "pesacheck.org", "fact_checker"), E("REFUTES", "nation.africa")]
    assert decide_verdict(evs)[0] == "CONTRADICTED"


def test_conflicting_credible_sources_are_mixed():
    evs = [E("SUPPORTS", "a.go.ke", "official"), E("REFUTES", "pesacheck.org", "fact_checker")]
    assert decide_verdict(evs)[0] == "MIXED"


def test_weak_dissent_does_not_flip_strong_support_but_comparable_dissent_does():
    strong = [E("SUPPORTS", "a.go.ke", "official"), E("SUPPORTS", "who.int", "official"), E("REFUTES", "blog.example", "unknown", "inferred")]
    assert decide_verdict(strong)[0] == "SUPPORTED"  # S=2.0, R=0.25
    close = [E("SUPPORTS", "a.go.ke", "official"), E("REFUTES", "nation.africa")]
    assert decide_verdict(close)[0] == "MIXED"  # S=1.0, R=0.8 -> not 2x


def test_partial_evidence_leads_to_mixed():
    assert decide_verdict([E("PARTIAL", "a.go.ug", "official")])[0] == "MIXED"


def test_social_media_and_snippets_weigh_less():
    assert evidence_weight(E("SUPPORTS", "facebook.com", "social")) == 0.25
    assert evidence_weight(E("SUPPORTS", "a.go.ke", "official", full=False)) == 0.7
    assert evidence_weight(E("SUPPORTS", "a.go.ke", "official", basis="inferred")) == 0.5
    many_social = [E("SUPPORTS", f"s{i}.facebook.com", "social") for i in range(3)]
    assert decide_verdict(many_social)[0] == "INSUFFICIENT_EVIDENCE"


def test_strength_label_is_not_probability():
    evs = [E("SUPPORTS", d, "official") for d in ("a.go.ke", "who.int", "b.go.ug")]
    _, _, s = decide_verdict(evs)
    assert s["label"] == "high"


def test_notes_flag_old_sources_and_snippets():
    evs = [E("SUPPORTS", "a.go.ke", "official", full=False, date="2019-05-01")]
    notes = evidence_notes("The 2024 census counted 45 million people.", evs)
    assert any("snippets" in n for n in notes)
    assert any("before 2024" in n for n in notes)


def test_subdomains_of_one_site_are_not_independent():
    from factcheck.utils.sources import site_key

    assert site_key("en.wikipedia.org") == site_key("sw.wikipedia.org") == "wikipedia.org"
    assert site_key("www.monitor.co.ug") == "monitor.co.ug" and site_key("statistics.go.ug") == "statistics.go.ug"
    evs = [E("SUPPORTS", "en.wikipedia.org", "unknown"), E("SUPPORTS", "sw.wikipedia.org", "unknown")]
    assert decide_verdict(evs)[0] == "INSUFFICIENT_EVIDENCE"


# --------------------------------------------------------------------------- relevance and time rules
from datetime import date, timedelta  # noqa: E402

from factcheck.core.Verdict import apply_relevance_rules, is_time_sensitive, parse_date, rank_evidence  # noqa: E402

TODAY = date(2026, 10, 16)


def test_time_sensitive_claims_are_detected():
    assert is_time_sensitive("President Ruto is dead.")
    assert is_time_sensitive("Rais Ruto amefariki dunia.")
    assert is_time_sensitive("The opposition leader has been arrested.")
    assert not is_time_sensitive("Jomo Kenyatta died in 1978.")  # dated: a past event, not a current situation
    assert not is_time_sensitive("Kampala is the capital of Uganda.")


def test_parse_date_formats_and_unknowns():
    assert parse_date("2026-10-14T08:00:00Z", TODAY) == date(2026, 10, 14)
    assert parse_date("Oct 14, 2026", TODAY) == date(2026, 10, 14)
    assert parse_date("3 days ago", TODAY) == date(2026, 10, 13)
    assert parse_date("", TODAY) is None and parse_date("sometime", TODAY) is None


def test_report_about_a_fake_video_is_context_not_evidence():
    evs = [dict(E("REFUTES", "pesacheck.org", "fact_checker"), scope="related"),
           dict(E("REFUTES", "africacheck.org", "fact_checker"), scope="related")]
    apply_relevance_rules("President Ruto is dead.", evs, TODAY)
    assert all(e["relationship"] == "CONTEXT" and "related video" in e["note"] for e in evs)
    assert decide_verdict(evs)[0] == "INSUFFICIENT_EVIDENCE"  # not "False" on the strength of a video debunk


def test_old_reports_do_not_decide_a_current_situation():
    recent = (TODAY - timedelta(days=2)).isoformat()
    old = [E("REFUTES", "nation.africa", date="2024-01-05"), E("REFUTES", "monitor.co.ug", date="2023-06-01")]
    apply_relevance_rules("President Ruto is dead.", old, TODAY)
    assert decide_verdict(old)[0] == "INSUFFICIENT_EVIDENCE"
    assert all("cannot show the current situation" in e["note"] for e in old)
    fresh = [E("REFUTES", "nation.africa", date=recent), E("REFUTES", "bbc.com", date=recent)]
    apply_relevance_rules("President Ruto is dead.", fresh, TODAY)
    assert decide_verdict(fresh)[0] == "CONTRADICTED"


def test_old_death_report_still_supports_death_claim_and_undated_is_flagged():
    evs = [E("SUPPORTS", "a.go.ke", "official", date="2019-02-01"), E("REFUTES", "blog.example", "unknown")]
    apply_relevance_rules("The former minister is dead.", evs, TODAY)
    assert evs[0]["relationship"] == "SUPPORTS"  # death is permanent
    assert "date unknown" in evs[1]["note"].lower()


def test_dated_claims_ignore_recency_rule():
    evs = [E("SUPPORTS", "a.go.ke", "official", date="1978-08-23")]
    apply_relevance_rules("Jomo Kenyatta died in 1978.", evs, TODAY)
    assert evs[0]["relationship"] == "SUPPORTS" and not evs[0].get("note")


def test_ranking_puts_strong_recent_deciding_evidence_first():
    evs = [E("CONTEXT", "x.com"), E("SUPPORTS", "blog.example", "unknown", date="2026-10-10"),
           E("IRRELEVANT", "y.com"), E("SUPPORTS", "a.go.ke", "official", date="2026-01-01"),
           E("SUPPORTS", "b.go.ke", "official", date="2026-10-01")]
    ranked = [e["domain"] for e in rank_evidence(evs, TODAY)]
    assert ranked == ["b.go.ke", "a.go.ke", "blog.example", "x.com", "y.com"]
