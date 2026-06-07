import { useState, useEffect } from "react";
import { healthCheck, uploadText, uploadGithub, querySimple, queryTask, getTrace } from "./api";
import "./App.css";

type Mode = "simple" | "task";

interface TraceStep {
  step_id: string; type: string; status: string; error?: string; timestamp: string;
}

function App() {
  const [ok, setOk] = useState(false);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("simple");
  const [response, setResponse] = useState("");
  const [trace, setTrace] = useState<TraceStep[]>([]);
  const [intent, setIntent] = useState<any>(null);
  const [taskGraph, setTaskGraph] = useState<any>(null);
  const [verified, setVerified] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");

  useEffect(() => {
    healthCheck().then(d => setOk(d.status === "ok")).catch(() => setOk(false));
  }, []);

  async function handleUploadText() {
    const text = prompt("Paste text content:");
    const name = prompt("Source name (e.g., notes.txt):");
    if (!text || !name) return; setLoading(true);
    const d = await uploadText(text, name);
    setUploadMsg(`Uploaded: ${d.chunks_created} chunks`);
    setLoading(false);
  }

  async function handleUploadGithub() {
    const url = prompt("GitHub repo URL:");
    if (!url) return; setLoading(true);
    const d = await uploadGithub(url);
    setUploadMsg(d.error ? `Error: ${d.error}` : `Indexed ${d.repo_name}: ${d.files_indexed} files`);
    setLoading(false);
  }

  async function handleSubmit() {
    if (!query.trim()) return; setLoading(true);
    setResponse(""); setTrace([]); setIntent(null); setTaskGraph(null);
    try {
      const d = mode === "simple" ? await querySimple(query) : await queryTask(query);
      setResponse(d.response || JSON.stringify(d, null, 2));
      setVerified(d.verified ?? null);
      if (d.intent) setIntent(d.intent);
      if (d.task_graph_summary) setTaskGraph(d.task_graph_summary);
      if (d.trace_id) {
        try { const t = await getTrace(d.trace_id); setTrace(t.steps || []); } catch {}
      }
    } catch (e: any) { setResponse(`Error: ${e.message}`); }
    setLoading(false);
  }

  return (
    <div className="app">
      <header>
        <h1>Agent-OS Console</h1>
        <span className={`dot ${ok ? "green" : "red"}`}>{ok ? "Connected" : "Disconnected"}</span>
      </header>

      <div className="main">
        <aside>
          <div className="card">
            <h3>Upload Knowledge</h3>
            <button onClick={handleUploadText} disabled={loading}>Upload Text</button>
            <button onClick={handleUploadGithub} disabled={loading}>Upload GitHub Repo</button>
            {uploadMsg && <p className="msg">{uploadMsg}</p>}
          </div>

          <div className="card">
            <h3>Query</h3>
            <div className="modes">
              <button className={mode === "simple" ? "active" : ""} onClick={() => setMode("simple")}>Simple</button>
              <button className={mode === "task" ? "active" : ""} onClick={() => setMode("task")}>Task Graph</button>
            </div>
            <textarea value={query} onChange={e => setQuery(e.target.value)}
              placeholder={mode === "simple" ? "Ask a question..." : "Describe a complex task..."} rows={4} />
            <button className="submit" onClick={handleSubmit} disabled={loading}>
              {loading ? "Processing..." : "Submit"}
            </button>
          </div>

          {intent && (
            <div className="card">
              <h3>Intent</h3>
              <p>Type: <strong>{intent.intent_type}</strong> ({(intent.confidence * 100).toFixed(0)}%)</p>
            </div>
          )}
          {taskGraph && (
            <div className="card">
              <h3>Task Graph</h3>
              <p>Nodes: <strong>{taskGraph.node_count}</strong> | Done: {taskGraph.completed} | Failed: {taskGraph.failed}</p>
            </div>
          )}
        </aside>

        <main>
          {verified !== null && (
            <span className={`badge ${verified ? "ok" : "fail"}`}>{verified ? "✓ Verified" : "✗ Unverified"}</span>
          )}
          <pre className="response">{response || "Submit a query to see results"}</pre>

          <h3>Execution Trace</h3>
          {trace.length === 0 ? (
            <p className="hint">Submit a query to see execution trace</p>
          ) : (
            <div className="timeline">
              {trace.map((s, i) => (
                <div key={i} className={`step ${s.status}`}>
                  <span className="type">{s.type}</span>
                  <span className={`mark ${s.status}`}>{s.status === "success" ? "✓" : s.status === "failed" ? "✗" : "○"}</span>
                  {s.error && <span className="err">{s.error}</span>}
                </div>
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
