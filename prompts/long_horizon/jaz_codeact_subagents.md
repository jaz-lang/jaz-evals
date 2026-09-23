## Maintain your own REPL output history in `output_history`

Maintain a list `output_history` of intermediate outputs as you work.

Initialize your `output_history` this way, which starts a fresh list on your first turn and leaves an
existing one untouched on every later turn. Write it exactly like this:

    output_history = output_history if "output_history" in dir() else []

On every turn where you're not conducting previous history search, append anything you
print to `output_history` as a string before printing it:

    result = ...                        # any intermediate output your code produces
    output_history.append(str(result))  # record its *string* representation
    print(result)                       # then print it

Make sure *everything* you print is also recorded in `output_history`, including things
such as tool outputs and intermediate results.

## IMPORTANT: Search your history to recall past information when available

Do you have all the information to know the correct next action with certainty? If not, then
*search for it*. `output_history` holds everything your own code has produced this session. If you
were delegated to, `prev_history` additionally holds the REPL history of the previous agent. To
recall information from earlier in the session, *search these lists*.

> *NOTE:* `prev_history` is *different* from `output_history`.
> Do NOT search `output_history`, since it's the history of your current REPL session and it's
> already visible to you in full, so searching `output_history` would be useless. Instead, you must
> search *`prev_history`*, the object that is only partially visible to you.

Follow these rules for `prev_history` search:
- Use targeted, concrete search terms that help uniquely find the search target. Use distinctive keywords
  unique to the information you're looking for and avoid overly broad terms.
- When you find a match, always display a context window around the hit — never truncate to just a prefix.
- Your ENTIRE next step is to search — do NOT write any code other than `prev_history` search.
  Print the `prev_history` search results and defer follow-up work to later turns.

    # If `prev_history`, the previous agent's history, is available:
    for i, repl_output in enumerate(prev_history):  # search `prev_history`, NOT `output_history`!
        # search for target in the entry's REPL output
        pos = repl_output.find("search term here")
        if pos >= 0:
            # Display a window around the search target
            start = max(0, pos - 1000)
            end = min(len(repl_output), pos + 2000)
            # ONLY print, do NOT append to `output_history` (appending pollutes future search results)
            print(f"--- entry[{i}] (pos {pos}) ---")
            print(repl_output[start:end])
            print()
    # STOP HERE: do NOT write any code after `prev_history` search — WAIT for the next turn to act on the search results
