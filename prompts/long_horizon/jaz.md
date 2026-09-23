## IMPORTANT: Search your `prev_history` to recall past information when available

Do you have all the information to know the correct next action with certainty? If not, then
*search for it*. If `prev_history` is available to you, then it contains the REPL history of the
previous agent working in this environment before they delegated to you. To recall information
from earlier in the session, *search for this information in `prev_history`*.

> *NOTE:* `prev_history` is *different* from the magic variable `__history__`.
> Do NOT search `__history__`, since it's the history of your current REPL session and it's
> already visible to you in full, so searching `__history__` would be useless. Instead, you must
> search *`prev_history`*, the object that is only partially visible to you.

Follow these rules for `prev_history` search:
- Use targeted, concrete search terms that help uniquely find the search target. Use distinctive keywords
  unique to the information you're looking for and avoid overly broad terms.
- When you find a match, always display a context window around the hit — never truncate to just a prefix.
- Your ENTIRE next step is to search — do NOT write any code other than `prev_history` search.
  Print the `prev_history` search results and defer follow-up work to later turns.

    # If `prev_history`, the previous agent's history, is available:
    for i, entry in enumerate(prev_history):  # search `prev_history`, NOT `__history__`!
        repl_output = entry.repl_output
        # skip entries with prior history search output, which pollutes search results
        if "--- entry[" in repl_output:
            continue
        # search for target in the entry's REPL output
        pos = repl_output.find("search term here")
        if pos >= 0:
            # Display a window around the search target
            start = max(0, pos - 1000)
            end = min(len(repl_output), pos + 2000)
            print(f"--- entry[{i}] (pos {pos}) ---")
            print(repl_output[start:end])
            print()
    # STOP HERE: do NOT write any code after `prev_history` search — WAIT for the next turn to act on the search results
