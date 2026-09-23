**Library name:** App APIs

**Description:** `apis` is the agent's interface to the supervisor's
day-to-day apps. Every interaction with the supervisor's accounts, contacts,
files, purchases, messages, and payments goes through this surface. Call any
endpoint as `apis.<app>.<endpoint>(**kwargs)`.

**Discovering endpoints and signatures.** The `apis` library exposes API
endpoints across multiple apps (e.g., Spotify, Venmo, phone, file system).
Use `apis.api_docs.*` to look up the available endpoints, their signatures,
and their response schemas.

- `apis.api_docs.show_app_descriptions()` — list every available app with a
  one-line description.
- `apis.api_docs.show_api_descriptions(app_name=<app>)` — list every endpoint
  on one app, with one-line descriptions.
- `apis.api_docs.show_api_doc(app_name=<app>, api_name=<endpoint>)` — full
  doc for one endpoint: exact required and optional parameter names, the
  success-response schema, the failure-response schema.

*Note:* Some endpoints return results distributed across multiple pages — each
call retrieves a single page.

**Personal data storage.** The phone app stores the supervisor's contacts
(friends, family, other relations). The supervisor app stores the supervisor's
account credentials, physical addresses, payment cards, and other personal
information.
