"use client";

import Link from "next/link";
import { useState, type ReactNode } from "react";

export type Ticket = {
  id: number;
  customer_name: string;
  subject: string;
  message: string;
  status: string;
};

type Filter = "all" | "open" | "pending" | "resolved";

// Restrained, semantic status colors shared by the badges and summary cards.
function statusStyles(status: string) {
  switch (status.toLowerCase()) {
    case "open":
      return {
        badge:
          "bg-amber-50 text-amber-700 ring-amber-200 dark:bg-amber-500/10 dark:text-amber-300 dark:ring-amber-500/30",
        value: "text-amber-600 dark:text-amber-400",
      };
    case "pending":
      return {
        badge:
          "bg-violet-50 text-violet-700 ring-violet-200 dark:bg-violet-500/10 dark:text-violet-300 dark:ring-violet-500/30",
        value: "text-violet-600 dark:text-violet-400",
      };
    case "resolved":
      return {
        badge:
          "bg-emerald-50 text-emerald-700 ring-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-300 dark:ring-emerald-500/30",
        value: "text-emerald-600 dark:text-emerald-400",
      };
    default:
      return {
        badge:
          "bg-zinc-100 text-zinc-600 ring-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:ring-zinc-700",
        value: "text-zinc-600 dark:text-zinc-400",
      };
  }
}

export default function TicketDashboard({
  tickets,
  children,
}: {
  tickets: Ticket[];
  children?: ReactNode;
}) {
  const [filter, setFilter] = useState<Filter>("all");

  // Counts are derived from the real tickets array — nothing is hardcoded.
  const statusCounts = tickets.reduce(
    (counts, ticket) => {
      const key = ticket.status.toLowerCase();
      if (key === "open") counts.open += 1;
      else if (key === "pending") counts.pending += 1;
      else if (key === "resolved") counts.resolved += 1;
      return counts;
    },
    { open: 0, pending: 0, resolved: 0 },
  );

  const filters: { key: Filter; label: string; value: number }[] = [
    { key: "all", label: "All", value: tickets.length },
    { key: "open", label: "Open", value: statusCounts.open },
    { key: "pending", label: "Pending", value: statusCounts.pending },
    { key: "resolved", label: "Resolved", value: statusCounts.resolved },
  ];

  const filteredTickets =
    filter === "all"
      ? tickets
      : tickets.filter((ticket) => ticket.status.toLowerCase() === filter);

  return (
    <>
      {/* Status summary (also acts as the filter control) */}
      <section aria-label="Ticket status filters">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {filters.map((item) => {
            const selected = filter === item.key;
            const styles = statusStyles(item.key);
            return (
              <button
                key={item.key}
                type="button"
                onClick={() => setFilter(item.key)}
                aria-pressed={selected}
                className={`rounded-lg border px-4 py-3 text-left transition-colors ${
                  selected
                    ? "border-violet-400 bg-violet-50 ring-1 ring-violet-400 dark:border-violet-600 dark:bg-violet-500/10 dark:ring-violet-600"
                    : "border-zinc-200 bg-white hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-700 dark:hover:bg-zinc-800/50"
                }`}
              >
                <p className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                  {item.label}
                </p>
                <p className={`mt-1 text-2xl font-semibold ${styles.value}`}>
                  {item.value}
                </p>
              </button>
            );
          })}
        </div>
      </section>

      {/* Tickets + create ticket */}
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
        <section className="lg:col-span-2">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold tracking-tight">Tickets</h2>
            <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
              {filteredTickets.length} shown
            </span>
          </div>

          {filteredTickets.length === 0 ? (
            <p className="mt-4 rounded-lg border border-dashed border-zinc-300 px-4 py-8 text-center text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
              {filter === "all"
                ? "No tickets found."
                : `No ${filter} tickets found.`}
            </p>
          ) : (
            <ul className="mt-4 flex flex-col gap-3">
              {filteredTickets.map((ticket) => {
                const styles = statusStyles(ticket.status);
                return (
                  <li key={ticket.id}>
                    <Link
                      href={`/tickets/${ticket.id}`}
                      className="group block rounded-lg border border-zinc-200 bg-white p-4 transition-colors hover:border-violet-300 hover:bg-violet-50/50 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-violet-700 dark:hover:bg-violet-500/5"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2 text-xs font-medium text-zinc-500 dark:text-zinc-400">
                            <span>#{ticket.id}</span>
                            <span aria-hidden="true">·</span>
                            <span className="truncate">
                              {ticket.customer_name}
                            </span>
                          </div>
                          <p className="mt-1 truncate text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                            {ticket.subject}
                          </p>
                        </div>
                        <span
                          className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ring-1 ring-inset ${styles.badge}`}
                        >
                          {ticket.status}
                        </span>
                      </div>
                      <p className="mt-2 line-clamp-2 text-sm text-zinc-600 dark:text-zinc-400">
                        {ticket.message}
                      </p>
                      <p className="mt-3 text-xs font-medium text-violet-600 dark:text-violet-400">
                        Open ticket →
                      </p>
                    </Link>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        {children}
      </div>
    </>
  );
}


