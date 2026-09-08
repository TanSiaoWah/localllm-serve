import Link from "next/link";

import CreateTicketForm from "./components/CreateTicketForm";

type Ticket = {
  id: number;
  customer_name: string;
  subject: string;
  message: string;
  status: string;
};

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
    <main className="flex min-h-screen flex-col items-center justify-center gap-6 px-4">
      <h1 className="text-3xl font-bold tracking-tight">
        LocalLLM Support Copilot
      </h1>
      <p className="max-w-md text-center text-lg text-zinc-600 dark:text-zinc-400">
        AI-assisted customer support powered by a locally hosted LLM.
      </p>
      <div className="rounded-md border border-zinc-300 px-4 py-2 text-sm text-zinc-600 dark:border-zinc-700 dark:text-zinc-400">
        Backend connection: {backendConnected ? "connected" : "unavailable"}
      </div>

      <h2 className="text-xl font-semibold">Create ticket</h2>
      <CreateTicketForm />

      <h2 className="text-xl font-semibold">Tickets</h2>
      {tickets.length === 0 ? (
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          No tickets found.
        </p>
      ) : (
        <ul className="flex w-full max-w-md flex-col gap-2">
          {tickets.map((ticket) => (
            <li
              key={ticket.id}
              className="rounded-md border border-zinc-200 px-4 py-3 transition-colors hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-900"
            >
              <Link href={`/tickets/${ticket.id}`} className="block">
                <div className="font-medium">
                  #{ticket.id} — {ticket.subject}
                </div>
                <div className="text-sm text-zinc-600 dark:text-zinc-400">
                  {ticket.customer_name}
                </div>
                <div className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                  {ticket.status}
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}