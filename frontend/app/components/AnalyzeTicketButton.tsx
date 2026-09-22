"use client";

import { useState } from "react";

type TicketAnalysis = {
  issue: string;
  category: string;
  urgency: string;
};

type AnalysisResponse = {
  ticket_id: number;
  analysis: TicketAnalysis;
};

export default function AnalyzeTicketButton({ ticketId }: { ticketId: number }) {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAnalyze() {
    setError("");
    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8080/tickets/${ticketId}/analyze`,
        {
          method: "POST",
        }
      );
      if (!response.ok) {
        setError("Failed to analyze ticket.");
        return;
      }
      const data = await response.json();
      setResult(data);
      console.log(data);
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
        onClick={handleAnalyze}
        disabled={loading}
        className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200 dark:disabled:hover:bg-white"
      >
        {loading ? "Analyzing..." : "Analyze Ticket"}
      </button>
      {error && (
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
      )}
      {result && (
        <div className="w-full max-w-md rounded-md border border-zinc-200 px-4 py-3 dark:border-zinc-700">
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            {result.analysis.issue}
          </p>
          <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
            {result.analysis.category} — {result.analysis.urgency}
          </p>
        </div>
      )}
    </>
  );
}