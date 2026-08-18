"use client";

import { useState } from "react";
import { verifyQuery, ingestFile, adversarialTest, VerifyResponse, AdversarialResponse } from "@/lib/api";
import { ShieldCheck, Upload, AlertTriangle, Loader2, CheckCircle2, XCircle } from "lucide-react";

type Tab = "verify" | "ingest" | "adversarial";

export default function Home() {
  const [tab, setTab] = useState<Tab>("verify");

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100 px-4 py-10">
      <div className="max-w-3xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <ShieldCheck className="text-emerald-400" size={32} />
            Axiom-Agent
          </h1>
          <p className="text-neutral-400 mt-1">
            Self-verifying AI for claim and document fact-checking
          </p>
        </header>

        <nav className="flex gap-2 mb-8 border-b border-neutral-800">
          {[
            { id: "verify", label: "Ask Axiom-Agent" },
            { id: "ingest", label: "Upload Document" },
            { id: "adversarial", label: "Adversarial Test" },
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id as Tab)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition ${
                tab === t.id
                  ? "border-emerald-400 text-emerald-400"
                  : "border-transparent text-neutral-500 hover:text-neutral-300"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {tab === "verify" && <VerifyTab />}
        {tab === "ingest" && <IngestTab />}
        {tab === "adversarial" && <AdversarialTab />}
      </div>
    </main>
  );
}

function VerifyTab() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VerifyResponse | null>(null);

  const steps = ["Moderation", "Semantic Entropy", "Retriever (ChromaDB)", "CrewAI Verification", "Output Safety"];

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await verifyQuery(query);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  const confidencePct = result ? Math.round(result.confidence * 100) : 0;

  return (
    <div>
      <form onSubmit={handleSubmit} className="flex gap-2 mb-6">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a claim or question to verify..."
          className="flex-1 bg-neutral-900 border border-neutral-700 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-emerald-400"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-neutral-950 font-semibold px-5 rounded-lg flex items-center gap-2"
        >
          {loading && <Loader2 className="animate-spin" size={16} />}
          {loading ? "Verifying..." : "Verify"}
        </button>
      </form>

      {loading && (
        <div className="flex flex-wrap gap-2 mb-6">
          {steps.map((s) => (
            <span key={s} className="text-xs bg-neutral-900 border border-neutral-700 px-3 py-1 rounded-full text-neutral-400 animate-pulse">
              {s}
            </span>
          ))}
        </div>
      )}

      {error && <div className="bg-red-950 border border-red-800 text-red-300 rounded-lg p-4 mb-6 text-sm">{error}</div>}

      {result && (
        <div className="space-y-4">
          <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-5">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs uppercase tracking-wide text-neutral-500">Answer</span>
              <span
                className={`text-xs px-2 py-1 rounded-full font-medium ${
                  confidencePct >= 70 ? "bg-emerald-950 text-emerald-400" : confidencePct >= 40 ? "bg-yellow-950 text-yellow-400" : "bg-red-950 text-red-400"
                }`}
              >
                {confidencePct}% confidence
              </span>
            </div>
            <p className="text-neutral-100 leading-relaxed">{result.answer}</p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
              <span className="text-xs text-neutral-500 block mb-1">Semantic Entropy</span>
              <span className="text-lg font-semibold">{result.semantic_entropy_score.toFixed(3)}</span>
            </div>
            <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
              <span className="text-xs text-neutral-500 block mb-1">Consistent?</span>
              <span className="flex items-center gap-1 text-lg font-semibold">
                {result.is_consistent ? <CheckCircle2 className="text-emerald-400" size={18} /> : <XCircle className="text-red-400" size={18} />}
                {result.is_consistent ? "Yes" : "No"}
              </span>
            </div>
          </div>

          {result.sources?.length > 0 && (
            <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
              <span className="text-xs text-neutral-500 block mb-2">Sources</span>
              <ul className="text-sm space-y-1 text-neutral-300 list-disc list-inside">
                {result.sources.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </div>
          )}

          {result.perspectives_considered?.length > 0 && (
            <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
              <span className="text-xs text-neutral-500 block mb-2">Perspectives Considered</span>
              <ul className="text-sm space-y-1 text-neutral-300 list-disc list-inside">
                {result.perspectives_considered.map((p, i) => <li key={i}>{p}</li>)}
              </ul>
            </div>
          )}

          <details className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
            <summary className="text-xs text-neutral-500 cursor-pointer">Full reasoning trace (raw)</summary>
            <pre className="text-xs mt-3 overflow-x-auto text-neutral-400">{JSON.stringify(result.reasoning_trace, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  );
}

function IngestTab() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleUpload() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await ingestFile(file);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="border-2 border-dashed border-neutral-700 rounded-lg p-8 text-center">
        <Upload className="mx-auto mb-3 text-neutral-500" size={28} />
        <input type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="text-sm text-neutral-400" />
        {file && <p className="text-sm text-neutral-300 mt-2">Selected: {file.name}</p>}
      </div>
      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className="bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-neutral-950 font-semibold px-5 py-2 rounded-lg flex items-center gap-2"
      >
        {loading && <Loader2 className="animate-spin" size={16} />}
        {loading ? "Ingesting..." : "Ingest Document"}
      </button>

      {error && <div className="bg-red-950 border border-red-800 text-red-300 rounded-lg p-4 text-sm">{error}</div>}

      {result != null && (
        <pre className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 text-xs overflow-x-auto text-neutral-300">{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}

function AdversarialTab() {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AdversarialResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleTest(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await adversarialTest(prompt);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Test failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <form onSubmit={handleTest} className="flex gap-2 mb-6">
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Try a jailbreak-style or harmful prompt..."
          className="flex-1 bg-neutral-900 border border-neutral-700 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-red-400"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-red-500 hover:bg-red-400 disabled:opacity-50 text-neutral-950 font-semibold px-5 rounded-lg flex items-center gap-2"
        >
          {loading && <Loader2 className="animate-spin" size={16} />}
          {loading ? "Testing..." : "Test"}
        </button>
      </form>

      {error && <div className="bg-red-950 border border-red-800 text-red-300 rounded-lg p-4 mb-4 text-sm">{error}</div>}

      {result && (
        <div className={`rounded-lg p-5 border flex items-start gap-3 ${result.blocked ? "bg-emerald-950 border-emerald-800" : "bg-yellow-950 border-yellow-800"}`}>
          {result.blocked ? <ShieldCheck className="text-emerald-400 shrink-0" size={22} /> : <AlertTriangle className="text-yellow-400 shrink-0" size={22} />}
          <div>
            <p className="font-semibold">{result.blocked ? "Blocked by guardrails" : "Not blocked"}</p>
            <p className="text-sm text-neutral-300 mt-1">{result.reason}</p>
          </div>
        </div>
      )}
    </div>
  );
}