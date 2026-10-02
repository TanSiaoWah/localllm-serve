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

const CURRENCY_SYMBOLS: Record<string, string> = {
  MYR: "RM",
  USD: "$",
  EUR: "€",
  GBP: "£",
  SGD: "S$",
};

// Friendly labels for the fields the backend tools currently return.
const FIELD_LABELS: Record<string, string> = {
  product: "Product",
  status: "Status",
  payment_id: "Payment ID",
  order_id: "Order ID",
  amount: "Amount",
  currency: "Currency",
};

// Human-readable title for each known backend tool.
const TOOL_TITLES: Record<string, string> = {
  get_order_status: "Order",
  get_payment_status: "Payment",
};

function fieldLabel(key: string): string {
  return FIELD_LABELS[key] ?? key.replace(/_/g, " ");
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

// Turn a tool result into readable rows without inventing missing fields.
function factRows(result: Record<string, unknown>) {
  const currency = typeof result.currency === "string" ? result.currency : "";
  const hasAmount = "amount" in result;

  return Object.entries(result)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .filter(([key]) => !(key === "currency" && hasAmount))
    .map(([key, rawValue]) => {
      let value: string;

      if (key === "amount") {
        const amount =
          typeof rawValue === "number" ? rawValue : Number(rawValue);
        const formatted = Number.isFinite(amount)
          ? amount.toLocaleString()
          : String(rawValue);
        const symbol = currency ? CURRENCY_SYMBOLS[currency] : undefined;
        value = `${symbol ? `${symbol} ` : ""}${formatted}${currency ? ` ${currency}` : ""}`;
      } else if (key === "status" && typeof rawValue === "string") {
        value = capitalize(rawValue);
      } else if (typeof rawValue === "string") {
        value = rawValue;
      } else {
        value = JSON.stringify(rawValue);
      }

      return { label: fieldLabel(key), value };
    });
}

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
    <div className="flex flex-col gap-5">
      <button
        type="button"
        onClick={handleAsk}
        disabled={loading}
        className="self-start rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200 dark:disabled:hover:bg-white"
      >
        {loading ? "Generating..." : "Generate AI Reply"}
      </button>

      {error && (
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
      )}

      {answer && (
        <>
          <section>
            <div className="flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded bg-violet-100 text-[10px] font-semibold text-violet-700 dark:bg-violet-500/20 dark:text-violet-300">
                AI
              </span>
              <h3 className="text-sm font-semibold tracking-tight">
                AI Reply Draft
              </h3>
            </div>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
              Review and edit the draft before sending.
            </p>
            <textarea
              id="ai-reply-draft"
              value={replyText}
              onChange={(event) => setReplyText(event.target.value)}
              rows={14}
              className="mt-3 w-full min-h-[18rem] resize-y rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm leading-relaxed text-zinc-800 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-200"
            />
            {replyText && (
              <button
                type="button"
                onClick={handleCopy}
                className="mt-3 rounded-md border border-violet-200 bg-violet-50 px-3 py-2 text-sm font-medium text-violet-700 hover:bg-violet-100 dark:border-violet-500/30 dark:bg-violet-500/10 dark:text-violet-300 dark:hover:bg-violet-500/20"
              >
                {copied ? "Copied" : "Copy Reply"}
              </button>
            )}
          </section>

          {verifiedFacts.length > 0 && (
            <section className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950/40">
              <div className="flex items-center gap-2">
                <span className="flex h-5 w-5 items-center justify-center rounded bg-emerald-100 text-[10px] font-semibold text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300">
                  ✓
                </span>
                <h3 className="text-sm font-semibold tracking-tight">
                  Verified Business Information
                </h3>
              </div>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                Confirmed by backend tools — not generated by the model.
              </p>

              <div className="mt-3 flex flex-col gap-3">
                {verifiedFacts.map((fact, index) => {
                  const rows = factRows(fact.result);
                  return (
                    <div
                      key={`${fact.tool}-${index}`}
                      className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-900"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <h4 className="text-sm font-semibold tracking-tight">
                          {TOOL_TITLES[fact.tool] ?? fact.tool}
                        </h4>
                        <span className="font-mono text-[10px] text-zinc-400 dark:text-zinc-500">
                          {fact.tool}
                        </span>
                      </div>
                      {rows.length > 0 && (
                        <dl className="mt-2 flex flex-col gap-1">
                          {rows.map((row) => (
                            <div
                              key={row.label}
                              className="flex justify-between gap-3 text-sm"
                            >
                              <dt className="text-zinc-500 dark:text-zinc-400">
                                {row.label}
                              </dt>
                              <dd className="text-right font-medium text-zinc-900 dark:text-zinc-100">
                                {row.value}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      )}
                    </div>
                  );
                })}
              </div>

              <details className="mt-3">
                <summary className="cursor-pointer text-xs font-medium text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200">
                  Raw tool output
                </summary>
                <pre className="mt-2 overflow-x-auto rounded-md bg-zinc-100 p-3 text-[11px] leading-relaxed text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                  {JSON.stringify(verifiedFacts, null, 2)}
                </pre>
              </details>
            </section>
          )}
        </>
      )}
    </div>
  );
}
