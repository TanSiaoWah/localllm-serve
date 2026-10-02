import Link from "next/link";

import AnalyzeTicketButton from "../../components/AnalyzeTicketButton";
import AskTicketButton from "../../components/AskTicketButton";

type Ticket = {
  id: number;
  customer_name: string;
  subject: string;
  message: string;
  status: string;
};

// Restrained status badge colors shared with the dashboard.
function statusBadgeClass(status: string) {
  switch (status.toLowerCase()) {
    case "open":
      return "bg-amber-50 text-amber-700 ring-amber-200 dark:bg-amber-500/10 dark:text-amber-300 dark:ring-amber-500/30";
    case "pending":
      return "bg-violet-50 text-violet-700 ring-violet-200 dark:bg-violet-500/10 dark:text-violet-300 dark:ring-violet-500/30";
    case "resolved":
      return "bg-emerald-50 text-emerald-700 ring-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-300 dark:ring-emerald-500/30";
    default:
      return "bg-zinc-100 text-zinc-600 ring-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:ring-zinc-700";
  }
}

export default async function TicketPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  let ticket: Ticket | null = null;

  try {
    const response = await fetch(`http://localhost:8080/tickets/${id}`, {
      cache: "no-store",
    });
    if (response.ok) {
      ticket = await response.json();
    }
  } catch {
    ticket = null;
  }

  if (!ticket) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center gap-4 bg-zinc-50 px-4 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
        <p className="text-lg text-zinc-600 dark:text-zinc-400">
          Ticket not found.
        </p>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-6 py-4">
          <Link
            href="/"
            className="text-sm font-medium text-zinc-500 underline-offset-2 hover:text-zinc-900 hover:underline dark:text-zinc-400 dark:hover:text-zinc-100"
          >
            ← Back to tickets
          </Link>
          <div className="flex items-center gap-3">
            <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
              Ticket #{ticket.id}
            </span>
            <span
              className={`rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ring-1 ring-inset ${statusBadgeClass(ticket.status)}`}
            >
              {ticket.status}
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-5xl flex-col gap-8 px-6 py-8">
        {/* Ticket heading */}
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight">
              {ticket.subject}
            </h1>
            <span
              className={`rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ring-1 ring-inset ${statusBadgeClass(ticket.status)}`}
            >
              {ticket.status}
            </span>
          </div>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            {ticket.customer_name}
          </p>
        </div>

        {/* Support workflow */}
        <section
          aria-label="Support workflow"
          className="rounded-lg border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900"
        >
          <ol className="flex flex-wrap items-center gap-x-3 gap-y-2 text-xs font-medium">
            {[
              "Customer Issue",
              "AI Analysis",
              "Verified Business Information",
              "Reply Draft",
            ].map((step, index) => (
              <li key={step} className="flex items-center gap-3">
                {index > 0 && (
                  <span
                    aria-hidden="true"
                    className="text-zinc-300 dark:text-zinc-600"
                  >
                    →
                  </span>
                )}
                <span className="flex items-center gap-1.5 text-zinc-600 dark:text-zinc-300">
                  <span className="flex h-4 w-4 items-center justify-center rounded-full bg-violet-100 text-[10px] font-semibold text-violet-700 dark:bg-violet-500/20 dark:text-violet-300">
                    {index + 1}
                  </span>
                  {step}
                </span>
              </li>
            ))}
          </ol>
        </section>

        {/* Two-column workspace */}
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-5">
          {/* Left: ticket information + customer issue */}
          <section className="flex flex-col gap-6 lg:col-span-2">
            <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
              <h2 className="text-sm font-semibold tracking-tight">
                Ticket details
              </h2>
              <dl className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                    Ticket ID
                  </dt>
                  <dd className="mt-1 text-sm text-zinc-900 dark:text-zinc-100">
                    #{ticket.id}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                    Customer
                  </dt>
                  <dd className="mt-1 text-sm text-zinc-900 dark:text-zinc-100">
                    {ticket.customer_name}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                    Status
                  </dt>
                  <dd className="mt-1">
                    <span
                      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ring-1 ring-inset ${statusBadgeClass(ticket.status)}`}
                    >
                      {ticket.status}
                    </span>
                  </dd>
                </div>
              </dl>
            </div>
            {/* Customer issue — kept visually distinct from AI output */}
            <div className="rounded-lg border border-zinc-200 border-l-4 border-l-zinc-400 bg-white p-5 dark:border-zinc-800 dark:border-l-zinc-600 dark:bg-zinc-900">
              <div className="flex items-center gap-2">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-zinc-200 text-[10px] font-semibold text-zinc-600 dark:bg-zinc-700 dark:text-zinc-300">
                  C
                </span>
                <h2 className="text-sm font-semibold tracking-tight">
                  Customer Issue
                </h2>
              </div>
              <p className="mt-3 whitespace-pre-line text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
                {ticket.message}
              </p>
              <p className="mt-4 text-xs text-zinc-500 dark:text-zinc-400">
                Original message submitted by {ticket.customer_name}
              </p>
            </div>
          </section>

          {/* Right: AI Copilot */}
          <aside className="flex flex-col gap-6 lg:col-span-3">
            <div className="rounded-lg border border-violet-200 bg-white p-5 dark:border-violet-500/30 dark:bg-zinc-900">
              <div className="flex items-center gap-2">
                <span className="flex h-5 w-5 items-center justify-center rounded bg-violet-100 text-[10px] font-semibold text-violet-700 dark:bg-violet-500/20 dark:text-violet-300">
                  AI
                </span>
                <h2 className="text-sm font-semibold tracking-tight">
                  AI Copilot
                </h2>
              </div>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                Triage the ticket and draft a reply grounded in verified backend
                data.
              </p>
              <div className="mt-5 flex flex-col gap-6">
                <p className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                  AI Assistance
                  </p>
                    <AnalyzeTicketButton ticketId={ticket.id} />
                <div className="border-t border-zinc-200 pt-6 dark:border-zinc-800">
                  <AskTicketButton ticketId={ticket.id} question={ticket.message} />
                </div>
                </div>
                    </div>
          </aside>
        </div>
      </div>
    </main>
  );
}
