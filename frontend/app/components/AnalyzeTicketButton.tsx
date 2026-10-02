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

// Restrained urgency colors; category uses the violet accent.
function urgencyBadgeClass(urgency: string) {
  switch (urgency.toLowerCase()) {
    case "high":
      return "bg-red-50 text-red-700 ring-red-200 dark:bg-red-500/10 dark:text-red-300 dark:ring-red-500/30";
    case "medium":
      return "bg-amber-50 text-amber-700 ring-amber-200 dark:bg-amber-500/10 dark:text-amber-300 dark:ring-amber-500/30";
    case "low":
      return "bg-emerald-50 text-emerald-700 ring-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-300 dark:ring-emerald-500/30";
    default:
      return "bg-zinc-100 text-zinc-600 ring-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:ring-zinc-700";
  }
}

const CATEGORY_BADGE_CLASS =
  "bg-violet-50 text-violet-700 ring-violet-200 dark:bg-violet-500/10 dark:text-violet-300 dark:ring-violet-500/30";

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
    <div className="flex flex-col gap-3">
      <button
        type="button"
        onClick={handleAnalyze}
        disabled={loading}
        className="self-start rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200 dark:disabled:hover:bg-white"
      >
        {loading ? "Analyzing..." : "Analyze Ticket"}
      </button>

      {error && (
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
      )}

      {result && (
        <section className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950/40">
          <div className="flex items-center gap-2">
            <span className="flex h-5 w-5 items-center justify-center rounded bg-violet-100 text-[10px] font-semibold text-violet-700 dark:bg-violet-500/20 dark:text-violet-300">
              AI
            </span>
            <h3 className="text-sm font-semibold tracking-tight">
              AI Analysis
            </h3>
          </div>

          <dl className="mt-3 flex flex-col gap-3">
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                Issue
              </dt>
              <dd className="mt-1 text-sm text-zinc-900 dark:text-zinc-100">
                {result.analysis.issue}
              </dd>
            </div>
            <div className="flex flex-wrap gap-6">
              <div>
                <dt className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                  Category
                </dt>
                <dd className="mt-1">
                  <span
                    className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${CATEGORY_BADGE_CLASS}`}
                  >
                    {result.analysis.category}
                  </span>
                </dd>
              </div>
              <div>
                <dt className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                  Urgency
                </dt>
                <dd className="mt-1">
                  <span
                    className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${urgencyBadgeClass(result.analysis.urgency)}`}
                  >
                    {result.analysis.urgency}
                  </span>
                </dd>
              </div>
            </div>
          </dl>
        </section>
      )}
    </div>
  );
}