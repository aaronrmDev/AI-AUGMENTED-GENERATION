from dataclasses import dataclass

from src.orchestration.domain.entities import Paradigm


@dataclass(frozen=True)
class RoutingExemplar:
    query: str
    paradigms: frozenset[Paradigm]


_CAG = frozenset({Paradigm.CAG})
_MAG = frozenset({Paradigm.MAG})
_RAG = frozenset({Paradigm.RAG})
_RAG_MAG = frozenset({Paradigm.RAG, Paradigm.MAG})
_CAG_RAG = frozenset({Paradigm.CAG, Paradigm.RAG})

# Labeled by the same reading of Concept 1 and Concept 9 as the lexical
# cues. Kept disjoint from Concept 1's own six example queries, which the
# evaluation set uses verbatim; a unit test enforces that.
DEFAULT_ROUTING_EXEMPLARS: tuple[RoutingExemplar, ...] = (
    RoutingExemplar("What does the employee handbook say about vacation days?", _CAG),
    RoutingExemplar("How do I reset my password according to the help guide?", _CAG),
    RoutingExemplar("What are the warranty terms in the product manual?", _CAG),
    RoutingExemplar("Walk me through the onboarding procedure for new contractors.", _CAG),
    RoutingExemplar("What is the standard shipping policy for international orders?", _CAG),
    RoutingExemplar("Summarize the architecture section of our internal documentation.", _CAG),
    RoutingExemplar("What does this function in the billing module do?", _CAG),
    RoutingExemplar("What are the size guide measurements for a medium shirt?", _CAG),
    RoutingExemplar("What is the company's stock price right now?", _RAG),
    RoutingExemplar("Is there any news about our competitor from this morning?", _RAG),
    RoutingExemplar("What is the current inventory level for the blue running shoes?", _RAG),
    RoutingExemplar("Which orders were delayed by the storm in the last hour?", _RAG),
    RoutingExemplar("What did the latest pricing update change for enterprise plans?", _RAG),
    RoutingExemplar("Is the payment service having an outage at the moment?", _RAG),
    RoutingExemplar("What are today's exchange rates from euros to dollars?", _RAG),
    RoutingExemplar("Which commits were merged into the release branch in the past hour?", _RAG),
    RoutingExemplar("What name did I tell you to call me?", _MAG),
    RoutingExemplar("Pick up the budget discussion from our last session.", _MAG),
    RoutingExemplar("Which of the options I shortlisted did I like best?", _MAG),
    RoutingExemplar("Remind me what dietary restrictions I mentioned.", _MAG),
    RoutingExemplar("What was the last thing we were working on together?", _MAG),
    RoutingExemplar("Write it in the tone I asked you to use before.", _MAG),
    RoutingExemplar("Did I already give you my project's deadline?", _MAG),
    RoutingExemplar("Go back to the draft email you helped me with.", _MAG),
    RoutingExemplar(
        "Given the portfolio I described to you, how did those stocks move today?", _RAG_MAG
    ),
    RoutingExemplar(
        "Are there flights leaving today for the destination I told you about?", _RAG_MAG
    ),
    RoutingExemplar(
        "Based on my usual order, is anything I buy out of stock right now?", _RAG_MAG
    ),
    RoutingExemplar(
        "How does this week's weather forecast affect the hiking trip we planned?", _RAG_MAG
    ),
    RoutingExemplar("Check my saved shopping list against current grocery prices.", _RAG_MAG),
    RoutingExemplar(
        "Has the bug I reported to you been fixed in the latest release notes?", _RAG_MAG
    ),
    RoutingExemplar(
        "What does the catalog say about this laptop, and is it in stock today?", _CAG_RAG
    ),
    RoutingExemplar(
        "What is the listed price of the premium plan, and has it changed recently?", _CAG_RAG
    ),
    RoutingExemplar(
        "How does the documented orders API differ from what was deployed this week?", _CAG_RAG
    ),
    RoutingExemplar(
        "What does the size guide recommend, and are those sizes available right now?", _CAG_RAG
    ),
)
