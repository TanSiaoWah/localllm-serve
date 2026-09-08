"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function CreateTicketForm() {
  const router = useRouter();
  const [customerName, setCustomerName] = useState("");
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    // Require non-whitespace text in every field before submitting.
    if (
      customerName.trim() === "" ||
      subject.trim() === "" ||
      message.trim() === ""
    ) {
      setError("Please fill in all fields.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch("http://localhost:8080/tickets/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          customer_name: customerName,
          subject: subject,
          message: message,
        }),
      });

      if (response.ok) {
        setCustomerName("");
        setSubject("");
        setMessage("");
        router.refresh();
      } else {
        // The backend returns {"detail": "..."} for errors; try to read it.
        let data;
        try {
          data = await response.json();
        } catch {
          data = null;
        }

        if (data && typeof data.detail === "string") {
          setError(data.detail);
        } else if (
          data &&
          Array.isArray(data.detail) &&
          data.detail.length > 0 &&
          typeof data.detail[0].msg === "string"
        ) {
          // FastAPI validation errors return {"detail": [{..., "msg": "..."}]}
          setError(data.detail[0].msg);
        } else {
          setError("Failed to create ticket.");
        }
      }
    } catch {
      setError("Backend is unavailable.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex w-full max-w-md flex-col gap-3">
      <input
        type="text"
        value={customerName}
        onChange={(event) => {
          setCustomerName(event.target.value);
          setError("");
        }}
        placeholder="Customer name"
        disabled={loading}
        className="rounded-md border border-zinc-300 px-3 py-2 text-sm dark:border-zinc-700"
      />
      <input
        type="text"
        value={subject}
        onChange={(event) => {
          setSubject(event.target.value);
          setError("");
        }}
        placeholder="Subject"
        disabled={loading}
        className="rounded-md border border-zinc-300 px-3 py-2 text-sm dark:border-zinc-700"
      />
      <textarea
        value={message}
        onChange={(event) => {
          setMessage(event.target.value);
          setError("");
        }}
        placeholder="Message"
        rows={3}
        disabled={loading}
        className="rounded-md border border-zinc-300 px-3 py-2 text-sm dark:border-zinc-700"
      />
      <button
        type="submit"
        disabled={loading}
        className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200 dark:disabled:hover:bg-white"
      >
        {loading ? "Creating..." : "Create ticket"}
      </button>
      {error && (
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
      )}
    </form>
  );
}