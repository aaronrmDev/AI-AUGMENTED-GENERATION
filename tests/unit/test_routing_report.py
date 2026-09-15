from evaluation.domain.routing_metrics import RoutingObservation, summarize
from evaluation.infrastructure.routing_report import render_router_comparison
from src.orchestration.domain.entities import Paradigm, RoutingDecision, RoutingMode

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


def _obs(query, expected, routed, mode=RoutingMode.CASCADE):
    return RoutingObservation(
        query, frozenset(expected), RoutingDecision(frozenset(routed), mode, {}), 1.5
    )


def test_the_report_tabulates_every_classifier_and_lists_its_misroutes():
    lexical = [_obs("refund policy?", {CAG}, {CAG}), _obs("changed today?", {RAG}, {CAG, RAG})]
    llm = [_obs("refund policy?", {CAG}, {CAG})]
    report = render_router_comparison(
        {"lexical": summarize(lexical), "llm": summarize(llm)},
        {"lexical": lexical, "llm": llm},
        notes="measured on this machine",
    )
    assert "measured on this machine" in report
    assert "| lexical | 2 | 50% | 100% | 100% |" in report
    assert "n/a" in report  # llm never selected MAG or RAG
    assert "- 'changed today?': expected RAG, routed CAG+RAG (cascade)" in report
    assert "## Misrouted by llm (0)" in report
    assert "- none" in report
