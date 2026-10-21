# Sharing Sagot AI with teammates (v1.5)

Two ways to put Sagot AI online. **Option A** is the quick one: your laptop runs Sagot AI and a free Cloudflare tunnel gives it a public link. **Option B** keeps it online when your laptop is off, but takes more setup.

In both cases:

- **Teammates can use these pages:**
  - the chatbot (`/`);
  - the live demo (`/demo`);
  - the evaluation tool (`/eval`; its button is hidden, so type the address).
- **The demo and evaluation tool need the team access code.** They make paid model calls, so a page opened through the public link asks for the code once and remembers it in that browser. Without the code, those pages are read-only.
- **Everyone shares one set of limits:**
  - each person (address): `RATE_LIMIT_PER_MIN` questions per minute;
  - everyone together: `DAILY_QUERY_CAP` questions per day;
  - at most `DEMO_MAX_CONCURRENT` demo questions running at once.
- **Upload and reset are never public.** `/upload` and `/reset` work only on the host computer, with or without the code.

## 1. Settings (once, in `.env`)

Add these lines to `.env`. They are kept out of git.

```
TEAM_ACCESS_CODE=pick-a-long-code-2026     # 8+ characters; share it only with your team
TRUST_CF_CONNECTING_IP=1                   # Option A: Cloudflare tells the server each visitor's real address
RATE_LIMIT_PER_MIN=10                      # questions per minute, per person
DAILY_QUERY_CAP=300                        # all questions together, per day (protects your credits)
DEMO_MAX_CONCURRENT=3                      # demo questions running at the same time
```

Cost guide:

- A chatbot question costs at most about 1.1 US cents.
- A demo question also pays the judge for three scores, so it costs roughly 2 to 3 cents.
- A cap of 300 a day therefore bounds a day at a few dollars of OpenRouter credit.

## 2. Option A: from your laptop (recommended for testing)

1. Open the project folder in File Explorer and double-click **`share.bat`**. From PowerShell, `.\share.ps1` does the same.
   - The first run installs `cloudflared` with winget.
   - The server starts in its own window.
2. Wait for a line like this:

   ```
   https://random-words-here.trycloudflare.com
   ```

3. Send that link and the team access code to your teammates.

Good to know:

- **The link stays up while both windows stay open** and the laptop stays awake. Turn off sleep while sharing (Settings → System → Power).
- **The link changes on every run.** Send the new one each time.
- **To stop sharing,** press Ctrl+C in the share window and close the server window.
- **To get a link that doesn't change,** use ngrok's free static domain instead (README, "Sharing it publicly through ngrok"). Everything else here still applies.

**Without the script**, run these in two PowerShell windows in the project folder:

```powershell
# window 1
venv\Scripts\activate
uvicorn main:app --host 127.0.0.1 --port 8000

# window 2 (install once: winget install --id Cloudflare.cloudflared)
cloudflared tunnel --url http://localhost:8000
```

## 3. What to send teammates

> Sagot AI test link: https://….trycloudflare.com
> - Chatbot: the link itself
> - Live demo: add /demo (it asks for the team code once: ……)
> - Evaluation tool: add /eval
>
> Please note any wrong or strange answers with the question you asked. Under each demo answer, **How it was made** shows every step.

Their questions appear in your `logs/queries.jsonl`, the same as yours.

## 4. Option B: always online (optional)

The `Dockerfile` packages the app, the tax rules and the document index (`chroma/`). It does not include `.env`; the host keeps the keys as secrets.

**How it was tested:** the container's exact files ran in a fresh Python environment. The pages loaded, the 897-chunk index loaded, the tax computation answered correctly, and the team-code lock worked. The Docker image itself could not be built in that environment, so build it once yourself (`docker build -t sagot-ai .`) before relying on it.

**Hugging Face Spaces (free, public):**

1. Create a Space with **SDK: Docker**.
2. Upload these files from the project, keeping the folder layout: `Dockerfile`, `.dockerignore`, `main.py`, `VERSION`, `requirements.txt`, and the folders `app/`, `rag_eval/` (without its `results*` folders), `static/`, `tax_rules/`, `config/` and `chroma/`. The web uploader handles the 12 MB `chroma/chroma.sqlite3`.
3. In **Settings → Variables and secrets**, add as secrets: `OPENROUTER_API_KEY`, `JINA_API_KEY`, `JUDGE_MODEL` and `TEAM_ACCESS_CODE`. Add the other `.env` settings you use (for example `RATE_LIMIT_PER_MIN`, `DAILY_QUERY_CAP`) as variables.
4. The Space builds and starts on its own. Its address is `https://<user>-<space>.hf.space`.

**Things to know about a hosted copy:**

- Every visitor counts as public there, including you, so the demo always asks for the code.
- Answers are cached and logged inside the container, and that is lost when it restarts.
- The document index is a copy. After you re-index on your laptop, upload `chroma/` again.

## 5. Troubleshooting

| What you see | What to do |
|---|---|
| The page asks for a code that "did not work" | Check `TEAM_ACCESS_CODE` in `.env` and restart the server; codes are case-sensitive |
| "Too many wrong team access codes" | Wait 10 minutes (10 wrong tries from one address lock it out for 10 minutes) |
| "The demo is busy answering other testers' questions" | `DEMO_MAX_CONCURRENT` questions are already running; try again in a minute or raise it |
| "The daily question limit … has been reached" | Raise `DAILY_QUERY_CAP` in `.env` and restart, or wait until tomorrow |
| The link stopped working | The laptop slept or a window was closed; run `share.bat` again and send the new link |
