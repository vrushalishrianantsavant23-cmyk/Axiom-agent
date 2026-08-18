const API_URL = process.env.NEXT_PUBLIC_API_URL;

export interface VerifyResponse {
  status: string;
  answer: string;
  semantic_entropy_score: number;
  is_consistent: boolean;
  confidence: number;
  sources: string[];
  is_neutral_assessment: boolean;
  perspectives_considered: string[];
  reasoning_trace: Record<string, unknown>;
  query_id?: string;
}

export interface AdversarialResponse {
  blocked: boolean;
  reason: string;
  status: string;
}

export async function verifyQuery(query: string): Promise<VerifyResponse> {
  const res = await fetch(`${API_URL}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error(`Verify failed: ${res.status}`);
  return res.json();
}

export async function ingestFile(file: File): Promise<unknown> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_URL}/ingest`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Ingest failed: ${res.status}`);
  return res.json();
}

export async function getTrajectory(queryId: string): Promise<unknown> {
  const res = await fetch(`${API_URL}/trajectory/${queryId}`);
  if (!res.ok) throw new Error(`Trajectory fetch failed: ${res.status}`);
  return res.json();
}

export async function adversarialTest(prompt: string): Promise<AdversarialResponse> {
  const res = await fetch(`${API_URL}/adversarial-test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error(`Adversarial test failed: ${res.status}`);
  return res.json();
}