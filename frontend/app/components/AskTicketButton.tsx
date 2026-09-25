"use client";

import { useState } from "react";

type AskTicketButtonProps = {
  ticketId: number;
  question: string;
};

type AskResponse = {
  ticket_id: number;
  answer: string;
  verified_facts: unknown[];
};

export default function AskTicketButton({
  ticketId,
  question,
}: AskTicketButtonProps) {
  const [answer, setAnswer] = useState("");
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
    } catch {
      setError("Backend is unavailable.");
    } finally {
      setLoading(false);
    }
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
          <p className="text-sm text-zinc-600 dark:text-zinc-400">{answer}</p>
        </div>
      )}
    </>
  );
}
