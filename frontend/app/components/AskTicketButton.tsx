"use client";

import { useState } from "react";

type AskTicketButtonProps = {
  ticketId: number;
  question: string;
};

type AskResponse = {
  ticket_id: number;
  answer: string;
  verified_facts: VerifiedFact[];
};

type VerifiedFact = {
  tool: string;
  result: Record<string, unknown>;
};

export default function AskTicketButton({
  ticketId,
  question,
}: AskTicketButtonProps) {
  const [answer, setAnswer] = useState("");
  const [replyText, setReplyText] = useState("");
  const [copied, setCopied] = useState(false);
  const [verifiedFacts, setVerifiedFacts] = useState<VerifiedFact[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAsk() {
    setError("");
    setLoading(true);

    try {
      const response = await fetch(
        `http://localhost:8080/tickets/${ticketId}/ask`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        }
      );

      if (!response.ok) {
        setError("Failed to generate reply.");
        return;
      }

      const data: AskResponse = await response.json();
      setAnswer(data.answer);
      setReplyText(data.answer);
      setVerifiedFacts(data.verified_facts);
    } catch {
      setError("Backend is unavailable.");
    } finally {
      setLoading(false);
    }
  }

  function handleCopy() {
    navigator.clipboard.writeText(replyText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <>
      <button
        type="button"
        onClick={handleAsk}
        disabled={loading}
        className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200 dark:disabled:hover:bg-white"
      >
        {loading ? "Generating..." : "Generate AI Reply"}
      </button>
      {error && (
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
      )}
      {answer && (
        <div className="w-full max-w-md rounded-md border border-zinc-200 px-4 py-3 dark:border-zinc-700">
          <label className="block text-sm font-medium" htmlFor="ai-reply-draft">
            AI Reply Draft
          </label>
          <textarea
            id="ai-reply-draft"
            value={replyText}
            onChange={(event) => setReplyText(event.target.value)}
            className="mt-2 w-full rounded-md border border-zinc-200 px-3 py-2 text-sm text-zinc-600 dark:border-zinc-700 dark:text-zinc-400"
            rows={5}
          />
          {replyText && (
            <button
              type="button"
              onClick={handleCopy}
              className="mt-2 rounded-md border border-zinc-200 px-3 py-2 text-sm font-medium hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
            >
              {copied ? "Copied" : "Copy Reply"}
            </button>
          )}
          {verifiedFacts.length > 0 && (
            <div className="mt-3 border-t border-zinc-200 pt-3 dark:border-zinc-700">
              <p className="text-sm font-medium">Verified facts</p>
              <ul className="mt-2 space-y-2 text-sm text-zinc-600 dark:text-zinc-400">
                {verifiedFacts.map((fact, index) => (
                  <li key={`${fact.tool}-${index}`}>
                    <span className="font-medium">{fact.tool}:</span>{" "}
                    {JSON.stringify(fact.result)}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </>
  );
}
