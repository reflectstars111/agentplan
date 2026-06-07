const BASE = "http://localhost:8000";

export async function uploadText(content: string, source_name: string) {
  const r = await fetch(`${BASE}/upload`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, source_name }),
  });
  return r.json();
}

export async function uploadFile(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/upload/file`, { method: "POST", body: fd });
  return r.json();
}

export async function uploadGithub(repo_url: string, branch = "main") {
  const r = await fetch(`${BASE}/upload/github`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_url, branch }),
  });
  return r.json();
}

export async function querySimple(query: string) {
  const r = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  return r.json();
}

export async function queryTask(query: string) {
  const r = await fetch(`${BASE}/task`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  return r.json();
}

export async function getTrace(traceId: string) {
  const r = await fetch(`${BASE}/trace/${traceId}`);
  if (!r.ok) throw new Error("Trace not found");
  return r.json();
}

export async function healthCheck() {
  const r = await fetch(`${BASE}/health`);
  return r.json();
}
