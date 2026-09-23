## IMPORTANT: Search your conversation history to recall past information when available

Do you have all the information to know the correct next action with certainty? If not, then
*search for it*. Your conversation history contains everything said earlier in this session, which
scrolls out of your context window as the run goes on. To recall information from earlier in the
session, *search for this information with `conversation_search`*.

> *NOTE:* `conversation_search` retrieves *messages* — the ones no longer visible to you, not the ones
> already in front of you. Do NOT search for something you can already see; searching it would be
> useless. Search for what has scrolled out of your context window.

Follow these rules for `conversation_search`:
- Use targeted, concrete search terms that help uniquely find the search target. Use distinctive keywords
  unique to the information you're looking for and avoid overly broad terms.
- Search for ONE thing per call — do NOT combine independent keywords into a single query. A combined
  query matches far less than either keyword would alone, so it often returns nothing at all. Issue a
  separate search for each thing you need.
- Your ENTIRE next step is to search — do NOT batch a search with any other tool call.
  Read the `conversation_search` results and defer follow-up work to later turns.

    # One query per call, so issue a separate call per thing you need. Searches do not depend on each
    # other's output, so batch them together:
    conversation_search(query="first search term here")
    conversation_search(query="second search term here")
    ...
    conversation_search(query="nth search term here")
    # STOP HERE: do NOT batch any other tool call with a search — WAIT for the next turn to act on the results
