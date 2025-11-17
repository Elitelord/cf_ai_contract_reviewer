# cf_ai_contract_guard

AI-powered contract reviewer built on Cloudflare Workers and Agents.

## Features

- Paste contract text and receive:
  - High-level summary
  - Risk assessment by clause
  - Suggested edits
  - Missing or concerning clauses

- Built with:
  - Cloudflare Agents starter
  - Workers AI (Llama 3.3)
  - Durable state / KV (planned)

## Running locally

```bash
npm install
npx wrangler dev
