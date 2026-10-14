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
