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
      <main className="flex min-h-screen flex-col items-center justify-center gap-4 px-4">
        <p className="text-lg text-zinc-600 dark:text-zinc-400">
          Ticket not found.
        </p>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 px-4">
      <Link
        href="/"
        className="text-sm text-zinc-500 underline-offset-2 hover:underline dark:text-zinc-400"
      >
        ← Back to tickets
      </Link>
      <h1 className="text-2xl font-bold tracking-tight">
        #{ticket.id} — {ticket.subject}
      </h1>
      <div className="w-full max-w-md rounded-md border border-zinc-200 px-4 py-3 dark:border-zinc-700">
        <div className="font-medium">{ticket.customer_name}</div>
        <div className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
          {ticket.status}
        </div>
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          {ticket.message}
        </p>
      </div>
      <AnalyzeTicketButton ticketId={ticket.id} />
      <AskTicketButton ticketId={ticket.id} question={ticket.message} />
    </main>
  );
}
