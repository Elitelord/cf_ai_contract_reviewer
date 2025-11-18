import { routeAgentRequest, type Schedule } from "agents";

import { AIChatAgent } from "agents/ai-chat-agent";
import {
  generateId,
  streamText,
  type StreamTextOnFinishCallback,
  stepCountIs,
  createUIMessageStream,
  convertToModelMessages,
  createUIMessageStreamResponse,
  type ToolSet
} from "ai";
import { openai } from "@ai-sdk/openai";
import { processToolCalls, cleanupMessages } from "./utils";
import { tools, executions } from "./tools";

// Use OpenAI via the Vercel AI SDK.
// Make sure OPENAI_API_KEY is set in `.dev.vars` locally and as a Wrangler secret in prod.
const model = openai("gpt-4o-2024-11-20");

/**
 * System prompt for Contract Guard, the contract-reviewing agent.
 */
const CONTRACT_SYSTEM_PROMPT = `
You are Contract Guard, an AI assistant that reviews legal contracts for non-lawyers.

Your goals:
1. Summarize the contract in clear, plain English.
2. Identify clauses that might be risky, unfair, or one-sided for the user.
3. Explain why each risky clause could be a problem.
4. Suggest more balanced alternative wording or what the user might ask the other party to change.
5. Point out important clauses that seem missing or vague (for example: termination, liability limits, payment terms, confidentiality, dispute resolution).

Always try to interpret the user's message as contract text or a question about a contract.
If the input clearly does not look like a contract, politely ask the user to paste or describe the contract they want reviewed.

When the user provides contract text, respond using this JSON structure:

{
  "summary": "3-6 bullet points summarizing the contract in plain English.",
  "overall_risk": "low" | "medium" | "high",
  "clauses": [
    {
      "name": "Short name or topic of the clause (for example: Termination, Confidentiality, Liability)",
      "risk": "low" | "medium" | "high",
      "reason": "Short explanation of why this clause is risky or safe.",
      "suggested_edit": "Concrete suggestion for how to improve or negotiate this clause, or what to ask a lawyer about."
    }
  ],
  "missing_clauses": [
    "Description of an important clause that is missing, too vague, or one-sided, if any."
  ],
  "disclaimer": "A reminder that this is not legal advice and that the user should consult a qualified attorney before signing any real contract."
}

Rules:
- Respond with JSON that matches the above shape as closely as possible.
- Do not include markdown code fences in the JSON.
- If you need to explain something conversationally, include it in the "summary" or "disclaimer" fields.
`;

/**
 * Chat Agent implementation that handles real-time AI chat interactions
 * Now specialized as a Contract Reviewing assistant (Contract Guard).
 */
export class Chat extends AIChatAgent<Env> {
  /**
   * Handles incoming chat messages and manages the response stream
   */
  async onChatMessage(
    onFinish: StreamTextOnFinishCallback<ToolSet>,
    _options?: { abortSignal?: AbortSignal }
  ) {
    // Collect all tools, including MCP tools (we're not explicitly using them for now,
    // but this keeps the starter scaffold intact and extensible).
    const allTools = {
      ...tools,
      ...this.mcp.getAITools()
    };

    
    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        try {
          console.log("createUIMessageStream.execute: started");

          // Clean up incomplete tool calls to prevent API errors
          const cleanedMessages = cleanupMessages(this.messages);
          console.log(
            "createUIMessageStream.execute: cleanedMessages length=",
            cleanedMessages.length
          );

          // Process any pending tool calls from previous messages
          const processedMessages = await processToolCalls({
            messages: cleanedMessages,
            dataStream: writer,
            tools: allTools,
            executions
          });

          console.log(
            "createUIMessageStream.execute: Messages going to model:",
            JSON.stringify(processedMessages, null, 2)
          );

          // Main contract-reviewing model call
          // NOTE: do NOT await streamText here — it returns a streaming result
          // and awaiting would block until the stream completes, preventing
          // incremental output from reaching the client.
          const result = streamText({
            system: CONTRACT_SYSTEM_PROMPT,
            messages: convertToModelMessages(processedMessages),
            model,
            tools: allTools,
            // Type boundary: streamText expects specific tool types, but base class uses ToolSet
            // This is safe because our tools satisfy ToolSet interface (verified by 'satisfies' in tools.ts)
            onFinish: onFinish as unknown as StreamTextOnFinishCallback<
              typeof allTools
            >,
            stopWhen: stepCountIs(10)
          });

          console.log("createUIMessageStream.execute: merging model stream into UI stream");
          writer.merge(result.toUIMessageStream());
          console.log("createUIMessageStream.execute: merge complete");
        } catch (err) {
          console.error("createUIMessageStream.execute: error:", err);
          try {
            // Attempt to send an error message back to the client so it doesn't hang
            writer.write({
              type: "message",
              role: "assistant",
              content: [{ type: "text", text: `Error: ${String(err)}` }]
            } as unknown as any);
          } catch (e) {
            console.error("Failed to write error to UI stream:", e);
          }
        }
      }
    });
    
    return createUIMessageStreamResponse({ stream });
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Kept from the starter: if you later add scheduled reviews or reminders,
    // this can log or trigger extra behavior.
    await this.saveMessages([
      ...this.messages,
      {
        id: generateId(),
        role: "user",
        parts: [
          {
            type: "text",
            text: `Running scheduled task: ${description}`
          }
        ],
        metadata: {
          createdAt: new Date()
        }
      }
    ]);
  }
}

/**
 * Worker entry point that routes incoming requests to the appropriate handler
 */
export default {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext) {
    const url = new URL(request.url);
    try {
      console.log("Incoming request:", request.method, url.pathname);
    } catch (e) {
      // noop
    }

    if (url.pathname === "/check-open-ai-key") {
      const hasOpenAIKey = !!process.env.OPENAI_API_KEY;
      return Response.json({
        success: hasOpenAIKey
      });
    }

    if (!process.env.OPENAI_API_KEY) {
      console.error(
        "OPENAI_API_KEY is not set. Set it in .dev.vars locally, and use `wrangler secret bulk .dev.vars` to upload it to production."
      );
    }
    if (url.pathname.endsWith("/debug-contract")) {
      console.log("Matched debug-contract route for path:", url.pathname);
      const body = await request.json().catch(() => null) as { contractText?: string } | null;
      const contractText = body?.contractText ?? "Test contract: The user agrees to wash the company car every day for no pay.";

      const result = await streamText({
        model,
        system: "You are a helpful assistant. Reply in plain text.",
        prompt: `What is risky about this contract? ${contractText}`
      });

      const fullText = await result.text; // get full text instead of streaming

      return new Response(fullText, {
        status: 200,
        headers: { "Content-Type": "text/plain" }
      });
    } 

    return (
      // Route the request to our agent or return 404 if not found
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
