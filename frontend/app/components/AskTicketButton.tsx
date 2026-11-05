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
  kapruka_search_products: "Kapruka products",
};

function isKaprukaCatalog(result: Record<string, unknown>): boolean {
  return result.source === "kapruka" &&
    result.data_type === "product_catalog" && Array.isArray(result.results);
}

function productRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function productText(value: unknown): string {
  if (typeof value === "string") return value.trim();
  return typeof value === "number" && Number.isFinite(value) ? String(value) : "";
}

function productLink(value: unknown): string | null {
  if (typeof value !== "string") return null;
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) && !url.username && !url.password
      ? url.href : null;
  } catch {
    return null;
  }
}

function KaprukaProducts({ products }: { products: unknown[] }) {
  return (
    <div className="mt-2 min-w-0 text-sm [overflow-wrap:anywhere]">
      <p className="text-xs text-zinc-500 dark:text-zinc-400">
        External Kapruka catalog — separate from demo-company orders and policies.
      </p>
      {products.length === 0 ? (
        <p className="mt-2 text-zinc-600 dark:text-zinc-400">No products found.</p>
      ) : (
        <ul className="mt-3 flex min-w-0 flex-col gap-3">
          {products.map((value, index) => {
            const product = productRecord(value);
            const price = productRecord(product.price);
            const amount = productText(price.amount ?? product.price ?? product.amount);
            const currency = productText(price.currency ?? product.currency);
            const stock = product.stock_status ?? product.in_stock ?? product.availability ?? product.stock;
            const stockText = typeof stock === "boolean"
              ? (stock ? "In stock" : "Out of stock")
              : productText(stock);
            const url = productLink(product.url ?? product.product_url);

            return (
              <li key={index} className="min-w-0 rounded-md border border-zinc-200 p-3 dark:border-zinc-700">
                <p className="font-medium text-zinc-900 dark:text-zinc-100">
                  {productText(product.name) || productText(product.product_name) || productText(product.title) || "Unnamed product"}
                </p>
                <p className="mt-1 text-zinc-700 dark:text-zinc-300">
                  Price: {amount || "Unavailable"}{currency ? ` ${currency}` : " (currency unavailable)"}
                </p>
                <p className="mt-1 text-zinc-600 dark:text-zinc-400">
                  Stock: {stockText || "Unknown"}
                </p>
                {url && (
                  <a href={url} target="_blank" rel="noopener noreferrer" className="mt-2 inline-block text-violet-700 underline underline-offset-2 dark:text-violet-300">
                    View product
                  </a>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function fieldLabel(key: string): string {
  return FIELD_LABELS[key] ?? key.replace(/_/g, " ");
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function toolErrorMessage(result: Record<string, unknown>): string {
  const error = result.error;
  if (error === null || error === undefined) return "";
  return typeof error === "string" ? error.trim() : JSON.stringify(error);
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
  const hasLookupFailures = verifiedFacts.some((fact) => toolErrorMessage(fact.result));
  const rawFacts = verifiedFacts.filter((fact) => !isKaprukaCatalog(fact.result));

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
    <div className="flex min-w-0 max-w-full flex-col gap-5">
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
            <section className="min-w-0 max-w-full rounded-lg border border-zinc-200 bg-zinc-50 p-4 [overflow-wrap:anywhere] dark:border-zinc-800 dark:bg-zinc-950/40">
              <div className="flex items-center gap-2">
                <span className={`flex h-5 w-5 items-center justify-center rounded text-[10px] font-semibold ${hasLookupFailures ? "bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-300" : "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300"}`}>
                  {hasLookupFailures ? "!" : "✓"}
                </span>
                <h3 className="text-sm font-semibold tracking-tight">
                  {hasLookupFailures ? "Backend Tool Results" : "Verified Business Information"}
                </h3>
              </div>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                {hasLookupFailures
                  ? "Backend lookup outcomes — failures do not verify business information."
                  : "Confirmed by backend tools — not generated by the model."}
              </p>

              <div className="mt-3 flex flex-col gap-3">
                {verifiedFacts.map((fact, index) => {
                  const lookupError = toolErrorMessage(fact.result);
                  const catalog = isKaprukaCatalog(fact.result);
                  const rows = lookupError || catalog ? [] : factRows(fact.result);
                  return (
                    <div
                      key={`${fact.tool}-${index}`}
                      className={`rounded-lg border p-3 ${lookupError ? "border-amber-200 bg-amber-50 dark:border-amber-500/30 dark:bg-amber-500/10" : "border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900"}`}
                    >
                      <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
                        <h4 className="text-sm font-semibold tracking-tight">
                          {TOOL_TITLES[fact.tool] ?? fact.tool}
                        </h4>
                        <span className="font-mono text-[10px] text-zinc-400 dark:text-zinc-500">
                          {fact.tool}
                        </span>
                      </div>
                      {lookupError && (
                        <p className="mt-2 whitespace-pre-line text-sm text-amber-800 dark:text-amber-300">
                          <span className="font-semibold">Lookup failed: </span>
                          {lookupError}
                        </p>
                      )}
                      {!lookupError && catalog && (
                        <KaprukaProducts products={fact.result.results as unknown[]} />
                      )}
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

              {rawFacts.length > 0 && <details className="mt-3 min-w-0 max-w-full">
                <summary className="cursor-pointer text-xs font-medium text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200">
                  Raw tool output
                </summary>
                <pre className="mt-2 overflow-x-auto rounded-md bg-zinc-100 p-3 text-[11px] leading-relaxed text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                  {JSON.stringify(rawFacts, null, 2)}
                </pre>
              </details>}
            </section>
          )}
        </>
      )}
    </div>
  );
}
