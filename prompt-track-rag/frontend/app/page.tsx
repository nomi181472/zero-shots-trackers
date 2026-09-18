"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const TRACKERS = ["rag", "bytetrack", "sort", "iou"];

type Status = {
  id: string;
  source_type: string;
  is_live: boolean;
  finished: boolean;
  error: string | null;
  warn: string | null;
  model_state: string;
  frame_count: number;
  active_tracks: number;
  det_count: number;
  last_box_count: number;
  elapsed_s: number;
  stream_url: string;
  params: Record<string, any>;
};

type Detector = {
  loaded: boolean;
  name?: string;
  kind?: string;
  device?: string;
  mock?: boolean;
  error?: string;
};

export default function Home() {
  const fileRef = useRef<File | null>(null);
  const [dragHot, setDragHot] = useState(false);

  // source mode
  const [mode, setMode] = useState<"upload" | "rtsp">("upload");
  const [rtspUrl, setRtspUrl] = useState(
    "rtsp://192.168.1.64:554/stream1"
  );

  // session state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [streamTs, setStreamTs] = useState(0); // cache-buster for the img url
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // model status (from /api/health)
  const [detector, setDetector] = useState<Detector | null>(null);

  // live-editable params
  const [text, setText] = useState("person, car");
  const [tracker, setTracker] = useState("rag");
  const [boxThresh, setBoxThresh] = useState(0.35);
  const [textThresh, setTextThresh] = useState(0.25);
  const [interval, setInterval] = useState(1);
  const [trackThresh, setTrackThresh] = useState(0.5);
  const [matchThresh, setMatchThresh] = useState(0.8);
  const [trackBuffer, setTrackBuffer] = useState(30);
  const [appWeight, setAppWeight] = useState(0.65);
  const [posWeight, setPosWeight] = useState(0.35);
  const [showTrails, setShowTrails] = useState(true);
  const [loop, setLoop] = useState(false);

  const [status, setStatus] = useState<Status | null>(null);

  // ---------------------------------------------------------------- model badge
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch("/api/health");
        if (res.ok) {
          const body = await res.json();
          setDetector(body.detector ?? null);
        }
      } catch {
        /* backend unreachable — badge stays on "checking…" */
      }
    };
    poll();
    const id = window.setInterval(() => void poll(), 4000);
    return () => window.clearInterval(id);
  }, []);

  // -------------------------------------------------------------- start source
  const start = useCallback(async () => {
    setError(null);
    if (mode === "upload") {
      const file = fileRef.current;
      if (!file) {
        setError("pick a video file first");
        return;
      }
      setBusy(true);
      const fd = new FormData();
      fd.append("file", file);
      fd.append("text_prompt", text);
      fd.append("tracker_name", tracker);
      fd.append("box_threshold", String(boxThresh));
      fd.append("text_threshold", String(textThresh));
      fd.append("detect_interval", String(interval));
      fd.append("track_thresh", String(trackThresh));
      fd.append("match_thresh", String(matchThresh));
      fd.append("track_buffer", String(trackBuffer));
      fd.append("w_appearance", String(appWeight));
      fd.append("w_position", String(posWeight));
      fd.append("loop", String(loop));
      fd.append("show_trails", String(showTrails));
      try {
        const res = await fetch("/api/track", { method: "POST", body: fd });
        if (!res.ok)
          throw new Error((await res.text()).slice(0, 200) || `HTTP ${res.status}`);
        const info = await res.json();
        setSessionId(info.id);
        setStreamTs(Date.now());
        setStatus(null);
      } catch (err: any) {
        setError(err.message ?? String(err));
      } finally {
        setBusy(false);
      }
      return;
    }

    // RTSP / live URL mode
    if (!rtspUrl.trim()) {
      setError("enter a stream URL");
      return;
    }
    setBusy(true);
    try {
      const res = await fetch("/api/rtsp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: rtspUrl.trim(),
          text_prompt: text,
          tracker_name: tracker,
          box_threshold: boxThresh,
          text_threshold: textThresh,
          detect_interval: interval,
          track_thresh: trackThresh,
          match_thresh: matchThresh,
          track_buffer: trackBuffer,
          w_appearance: appWeight,
          w_position: posWeight,
          show_trails: showTrails,
        }),
      });
      if (!res.ok)
        throw new Error((await res.text()).slice(0, 200) || `HTTP ${res.status}`);
      const info = await res.json();
      setSessionId(info.id);
      setStreamTs(Date.now());
      setStatus(null);
    } catch (err: any) {
      setError(err.message ?? String(err));
    } finally {
      setBusy(false);
    }
  }, [mode, rtspUrl, text, tracker, boxThresh, textThresh, interval,
      trackThresh, matchThresh, trackBuffer, appWeight, posWeight, showTrails, loop]);

  // ------------------------------------------------------- live control panel
  const lastSent = useRef<string>("");
  useEffect(() => {
    if (!sessionId) return;
    const payload = {
      text_prompt: text,
      tracker_name: tracker,
      box_threshold: boxThresh,
      text_threshold: textThresh,
      detect_interval: interval,
      track_thresh: trackThresh,
      match_thresh: matchThresh,
      track_buffer: trackBuffer,
      w_appearance: appWeight,
      w_position: posWeight,
      show_trails: showTrails,
      loop,
    };
    const sig = JSON.stringify(payload);
    if (sig === lastSent.current) return;
    lastSent.current = sig;
    const t = setTimeout(async () => {
      try {
        const res = await fetch(`/api/control/${sessionId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: sig,
        });
        if (res.ok) setStatus(await res.json());
      } catch {
        /* keep streaming regardless of control hiccups */
      }
    }, 350);
    return () => clearTimeout(t);
  }, [sessionId, text, tracker, boxThresh, textThresh, interval, trackThresh,
      matchThresh, trackBuffer, appWeight, posWeight, showTrails, loop]);

  // ------------------------------------------------------------------- status
  useEffect(() => {
    if (!sessionId) return;
    const poll = async () => {
      const res = await fetch(`/api/status/${sessionId}`);
      if (res.ok) setStatus(await res.json());
    };
    poll();
    const id = window.setInterval(() => void poll(), 400);
    return () => window.clearInterval(id);
  }, [sessionId]);

  const streamUrl = sessionId
    ? `/api/stream/${sessionId}?t=${streamTs}`
    : null;

  const noDetections =
    status && status.frame_count > 10 && status.det_count === 0 && !status.error;

  return (
    <main>
      <h1>Zero-Shot Tracking (RAG)</h1>
      <p className="sub">
        GroundingDINO detects whatever your text says; the RAG tracker embeds
        each detection, retrieves the most similar track memories (CLIP vector
        store) and generates the association. Upload a clip or point it at an
        RTSP camera, then switch trackers while it runs.
      </p>

      <div className="grid">
        {/* ---------------------------------------------- controls column */}
        <section className="card">
          <div className="tabs">
            <button className={`tab${mode === "upload" ? " on" : ""}`}
                    onClick={() => setMode("upload")}>upload file</button>
            <button className={`tab${mode === "rtsp" ? " on" : ""}`}
                    onClick={() => setMode("rtsp")}>rtsp / ip cam</button>
          </div>

          <div className="tabpanel">
            {mode === "upload" ? (
              <>
                <label>video file (.mp4 / .avi / .mov / .mkv / .webm)</label>
                <div
                  className={`drag${dragHot ? " hot" : ""}`}
                  onDragOver={(e) => { e.preventDefault(); setDragHot(true); }}
                  onDragLeave={() => setDragHot(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragHot(false);
                    const f = e.dataTransfer.files?.[0];
                    if (f) fileRef.current = f;
                  }}
                  onClick={() => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = "video/*";
                    input.onchange = () => {
                      if (input.files?.[0]) fileRef.current = input.files[0];
                    };
                    input.click();
                  }}
                >
                  {fileRef.current ? `selected: ${fileRef.current.name}` : "click or drop a video"}
                </div>
              </>
            ) : (
              <>
                <label>live stream url</label>
                <input
                  type="text"
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  placeholder="rtsp://user:pass@host:554/stream1"
                />
                <p className="hint">
                  rtsp:// cameras, rtmp://, or an http(s) MJPEG feed. The
                  server reconnects automatically if the feed drops.
                </p>
              </>
            )}
          </div>

          <label>text prompt</label>
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder='e.g. "person, car, traffic light"'
          />

          <label>tracker (switches live, no restart)</label>
          <select value={tracker} onChange={(e) => setTracker(e.target.value)}>
            {TRACKERS.map((t) => (
              <option key={t} value={t}>
                {t === "rag" ? "RAG (retrieve + generate, no Kalman)" :
                 t === "bytetrack" ? "ByteTrack (two-stage IoU + Kalman)" :
                 t === "sort" ? "SORT (Kalman + Hungarian)" :
                 "IoU baseline (no motion model)"}
              </option>
            ))}
          </select>

          <div className="row">
            <div>
              <label>box threshold</label>
              <input type="range" min={0.05} max={0.9} step={0.05}
                     value={boxThresh}
                     onChange={(e) => setBoxThresh(Number(e.target.value))} />
            </div>
            <div>
              <label>text threshold</label>
              <input type="range" min={0.05} max={0.6} step={0.05}
                     value={textThresh}
                     onChange={(e) => setTextThresh(Number(e.target.value))} />
            </div>
          </div>

          <div className="row">
            <div>
              <label>detect every N frames</label>
              <input type="number" min={1} max={10} value={interval}
                     onChange={(e) => setInterval(Number(e.target.value))} />
            </div>
            <div>
              <label>track hit threshold</label>
              <input type="range" min={0.1} max={0.9} step={0.05}
                     value={trackThresh}
                     onChange={(e) => setTrackThresh(Number(e.target.value))} />
            </div>
          </div>

          <div className="row">
            <div>
              <label>match threshold</label>
              <input type="range" min={0.1} max={0.95} step={0.05}
                     value={matchThresh}
                     onChange={(e) => setMatchThresh(Number(e.target.value))} />
            </div>
            <div>
              <label>track buffer (frames)</label>
              <input type="number" min={1} max={120} value={trackBuffer}
                     onChange={(e) => setTrackBuffer(Number(e.target.value))} />
            </div>
          </div>

          <div className="row">
            <div>
              <label>RAG appearance weight</label>
              <input type="range" min={0} max={1} step={0.05}
                     value={appWeight}
                     onChange={(e) => setAppWeight(Number(e.target.value))} />
            </div>
            <div>
              <label>RAG positional weight</label>
              <input type="range" min={0} max={1} step={0.05}
                     value={posWeight}
                     onChange={(e) => setPosWeight(Number(e.target.value))} />
            </div>
          </div>

          <div className="row">
            <div>
              <label style={{ marginBottom: 6 }}>show motion trails</label>
              <input type="checkbox" checked={showTrails}
                     onChange={(e) => setShowTrails(e.target.checked)} />
            </div>
            <div>
              {mode === "upload" && (
                <>
                  <label style={{ marginBottom: 6 }}>loop video</label>
                  <input type="checkbox" checked={loop}
                         onChange={(e) => setLoop(e.target.checked)} />
                </>
              )}
            </div>
          </div>

          <div style={{ marginTop: 18 }}>
            <button className="btn" onClick={start} disabled={busy}>
              {busy
                ? mode === "rtsp" ? "connecting…" : "uploading…"
                : mode === "rtsp" ? "connect & track" : "start tracking"}
            </button>
          </div>

          {error && <div className="warn">⚠ {error}</div>}
        </section>

        {/* ---------------------------------------------- stream column */}
        <section className="card">
          <div className="modelbar">
            <span className={`dot${detector?.loaded ? " on" : ""}`} />
            <b>model:</b>
            <span>{detector?.name ?? "…"}</span>
            <span className="muted">({detector?.kind ?? "?"} · {detector?.device ?? "?"})</span>
            {detector == null ? (
              <span className="muted">checking…</span>
            ) : detector.loaded ? (
              <span className="ok">running ✓</span>
            ) : (
              <span className="badlabel">NOT loaded ✗</span>
            )}
            {detector?.mock && (
              <span className="pill">demo mode — detects red blobs only</span>
            )}
            {detector?.error && <span className="warn">{detector.error}</span>}
          </div>

          <div className="streambox">
            {streamUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={streamUrl} alt="live tracking stream" />
            ) : (
              <div className="off">
                {mode === "rtsp"
                  ? "enter an RTSP url and hit connect"
                  : "upload a video to start the live stream"}
              </div>
            )}
          </div>

          {status && (
            <>
              <div className="stat">
                <div><b>{status.active_tracks}</b><span>active tracks</span></div>
                <div><b>{status.frame_count}</b><span>frames</span></div>
                <div><b>{status.last_box_count}</b><span>boxes / frame</span></div>
                <div><b>{status.det_count}</b><span>total detections</span></div>
                <div><b>{status.params?.tracker_name}</b><span>tracker</span></div>
                <div><b>{status.elapsed_s}s</b><span>elapsed</span></div>
              </div>
              {status.warn && <div className="warn">⚠ {status.warn}</div>}
              {status.error && <div className="warn">⚠ {status.error}</div>}
              {noDetections && (
                <div className="warn hint">
                  ⚠ 0 boxes found so far — the demo detector only sees <b>red</b> objects.
                  {" "}
                  <a className="btn ghost" href="/api/demo-video" download>
                    download red-blob demo clip
                  </a>
                </div>
              )}
              {status.finished && !status.error && (
                <div className="ok">
                  {status.is_live
                    ? "stream stopped"
                    : "video finished — stream stays on the last frame"}
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </main>
  );
}