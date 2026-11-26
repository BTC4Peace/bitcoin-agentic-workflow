**TODO**: pretty readme

Experimental ground for AI reviewing Bitcoin Core merges

**Current implementation:**

GitHub Action fetches the main branch for new merges, creates a PR for each of them, keeping track via separate branches

->

GitHub Action calls DeepSeek & Gemini Neon to review the PRs via an API call

Once a PR is manually reviewed by an human and closed, GitHub Action cleans off the related branches

**In construction:**

Moving the entire reviewing process off GitHub, on a computer/server/Pi, python-based

-> A GitHub action can be a basic reviewer bot with sheer prompting; even with a structural flow like [Ralph](https://github.com/willcl-ark/nix/tree/master/pkgs/ralph/agents)

-> But moving into a persistent local box promotes the reviewer from a static code reviewer into an evidence-backed auditor that compiles, tests, mutates, and fuzzes merges

On a separate public repo when ready to deploy

**Goal:**

GitHub Action fetches the main branch for new merges, creates a PR for each of them, keeping track via separate branches
On PR created -> send signal to the self-hosted reviewer box

-> LLM CLI calls + [bcore-mutation](https://github.com/brunoerg/bcore-mutation) + [bcore-mutation-plan](https://github.com/brunoerg/bcore-mutation-plan) +...

-> Complete PR review, audited by multiple LLMs with thorough testing

**Considerations:**

Review quality

Use of open-source tools for testing/fuzzying

Triage based on the type of merge(docs, lint, /src...)

Security

Be helpful and readable to the human reviewer
