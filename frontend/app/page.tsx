import CreateTicketForm from "./components/CreateTicketForm";
import TicketDashboard, { type Ticket } from "./components/TicketDashboard";

export default async function Home() {
  let backendConnected = false;
  let tickets: Ticket[] = [];

  try {
    const response = await fetch("http://localhost:8080/health", {
      cache: "no-store",
    });
    const data = await response.json();
    if (data.status === "ok") {
      backendConnected = true;
    }
  } catch {
    backendConnected = false;
  }

  try {
    const response = await fetch("http://localhost:8080/tickets/", {
      cache: "no-store",
    });
    if (response.ok) {
      tickets = await response.json();
    }
  } catch {
    tickets = [];
  }

  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-violet-600 text-sm font-semibold text-white">
              LC
            </div>
            <div>
              <p className="text-sm font-semibold leading-tight">
                LocalLLM Support Copilot
              </p>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                AI-assisted customer support workspace
              </p>
            </div>
          </div>
          <div className="inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1.5 text-xs font-medium text-zinc-600 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
            <span
              aria-hidden="true"
              className={`h-2 w-2 rounded-full ${
                backendConnected ? "bg-emerald-500" : "bg-red-500"
              }`}
            />
            {backendConnected ? "Backend connected" : "Backend unavailable"}
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-5xl flex-col gap-8 px-6 py-8">
        {/* Page heading */}
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Support Dashboard
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-zinc-600 dark:text-zinc-400">
            Review incoming support tickets and use the AI copilot to investigate
            orders, payments, and prepare responses — backed by the locally
            hosted LLM.
          </p>
        </div>

        <TicketDashboard tickets={tickets}>

          {/* Create ticket (secondary) */}
          <aside className="lg:col-span-1">
            <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
              <h2 className="text-sm font-semibold tracking-tight">
                New ticket
              </h2>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                Add a ticket to demo the copilot against real backend data.
              </p>
              <div className="mt-4">
                <CreateTicketForm />
              </div>
            </div>
          </aside>
        </TicketDashboard>
      </div>
    </main>
  );
}