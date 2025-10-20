# Toxic or Not

This app reads through your messages and tells you if your contacts are toxic or not!

### Setup

- **Python**: 3.9+
- **Install deps**:
  ```bash
  pip install openai
  ```
- **API key**:
  ```bash
  export OPENAI_API_KEY="your_api_key"
  # optional if using an OpenAI-compatible endpoint
  export OPENAI_BASE_URL="https://your-base-url/v1"
  # optional model override (defaults to gpt-4o-mini)
  export MODEL="gpt-4o-mini"
  ```

### Conversations format

Place JSON files in `conversations/`. Supported shapes:

```json
{
  "messages": [
    { "sender": "me", "text": "hey" },
    { "sender": "Alex", "text": "..." }
  ]
}
```

or

```json
[
  { "sender": "me", "text": "hey" },
  { "sender": "Alex", "text": "..." }
]
```

### Run

```bash
python code/main.py
```

The script will scan all `*.json` files in `conversations/` and print a line per file:

```
data.json [Alex]: toxic (score=0.82) - insults and threats
```

If the other participant's name is not present in messages, the script falls back to the
filename (e.g., `jack.json` -> `jack`).

When a contact is classified as toxic, the output also includes a concise, respectful
suggested ending message:

```
data.json [Alex]: toxic (score=0.82) - insults and threats
  Suggested ending message -> Alex, I need to step back from this relationship. Our conversations haven’t felt healthy for me, so I’m ending contact. I wish you well.
```