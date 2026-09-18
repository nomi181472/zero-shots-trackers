"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type BoundingBox = [number, number, number, number]; // [x1, y1, x2, y2]

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
  last_box_count: number;
  current_box: [number, number, number, number] | null;
  elapsed_s: number;
  stream_url: string;
  params: Record<string, any>;
};

export default function Home() {
  const fileRef = useRef<File | null>(null);
  const cropFileRef = useRef<File | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frame1ImgRef = useRef<HTMLImageElement | null>(null);

  // Source selection
  const [mode, setMode] = useState<"upload" | "rtsp">("upload");
  const [rtspUrl, setRtspUrl] = useState("rtsp://192.168.1.64:554/stream1");
  const [dragHot, setDragHot] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [cropFileName, setCropFileName] = useState<string | null>(null);

  // Frame 1 preview & interactive drawing state
  const [frame1DataUrl, setFrame1DataUrl] = useState<string | null>(null);
  const [frameDims, setFrameDims] = useState<{ width: number; height: number } | null>(null);
  const [box, setBox] = useState<BoundingBox | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);

  // Active streaming session
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [streamTs, setStreamTs] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<Status | null>(null);

  // Live controllable parameters
  const [tracker, setTracker] = useState("ostrack");
  const [textPrompt, setTextPrompt] = useState("visual target");
  const [showTrails, setShowTrails] = useState(true);
  const [loop, setLoop] = useState(false);

  // -------------------------------------------------------------
  // Load Frame 1 Preview for Drawing
  // -------------------------------------------------------------
  const loadFirstFrame = useCallback(async (file: File) => {
    setError(null);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch("/api/first-frame", { method: "POST", body: fd });
      if (!res.ok) throw new Error("Could not extract frame 1 for box selection");
      const data = await res.json();
      setFrame1DataUrl(data.first_frame_b64);
      setFrameDims({ width: data.width, height: data.height });

      // Default to center 25% box initially
      const cw = data.width * 0.25;
      const ch = data.height * 0.25;
      const cx = data.width / 2;
      const cy = data.height / 2;
      setBox([
        Math.round(cx - cw / 2),
        Math.round(cy - ch / 2),
        Math.round(cx + cw / 2),
        Math.round(cy + ch / 2),
      ]);
    } catch (err: any) {
      console.warn("first-frame extraction failed:", err);
    }
  }, []);

  const handleFileChange = useCallback(
    (f: File) => {
      fileRef.current = f;
      setSelectedFileName(f.name);
      loadFirstFrame(f);
    },
    [loadFirstFrame]
  );

  const loadDemo = useCallback(async () => {
    setError(null);
    setBusy(true);
    try {
      const res = await fetch("/api/demo-video");
      if (!res.ok) throw new Error("Could not load demo video");
      const blob = await res.blob();
      const file = new File([blob], "demo-target.mp4", { type: "video/mp4" });
      handleFileChange(file);
    } catch (err: any) {
      setError(err.message ?? String(err));
    } finally {
      setBusy(false);
    }
  }, [handleFileChange]);

  // -------------------------------------------------------------
  // Render Canvas with Frame 1 and Bounding Box
  // -------------------------------------------------------------
  useEffect(() => {
    if (!frame1DataUrl) return;
    const img = new Image();
    img.src = frame1DataUrl;
    img.onload = () => {
      frame1ImgRef.current = img;
      redrawCanvas();
    };
  }, [frame1DataUrl]);

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = frame1ImgRef.current;
    if (!canvas || !img || !frameDims) return;

    canvas.width = frameDims.width;
    canvas.height = frameDims.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Draw background video frame
    ctx.drawImage(img, 0, 0, frameDims.width, frameDims.height);

    // Draw bounding box if set
    if (box) {
      const [x1, y1, x2, y2] = box;
      const w = x2 - x1;
      const h = y2 - y1;

      // Box semi-transparent fill
      ctx.fillStyle = "rgba(59, 130, 246, 0.25)";
      ctx.fillRect(x1, y1, w, h);

      // Box stroke
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, w, h);

      // Corner markers
      ctx.fillStyle = "#ffffff";
      const cs = 6;
      ctx.fillRect(x1 - cs / 2, y1 - cs / 2, cs, cs);
      ctx.fillRect(x2 - cs / 2, y1 - cs / 2, cs, cs);
      ctx.fillRect(x1 - cs / 2, y2 - cs / 2, cs, cs);
      ctx.fillRect(x2 - cs / 2, y2 - cs / 2, cs, cs);

      // Label tag
      ctx.fillStyle = "#0284c7";
      ctx.fillRect(x1, Math.max(0, y1 - 22), 120, 22);
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 12px monospace";
      ctx.fillText(`TARGET ${Math.round(w)}x${Math.round(h)}`, x1 + 6, Math.max(16, y1 - 6));
    }
  }, [box, frameDims]);

  useEffect(() => {
    redrawCanvas();
  }, [box, redrawCanvas]);

  // Canvas Mouse Events for Dragging Box
  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !frameDims) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = frameDims.width / rect.width;
    const scaleY = frameDims.height / rect.height;
    return {
      x: Math.max(0, Math.min(frameDims.width, (e.clientX - rect.left) * scaleX)),
      y: Math.max(0, Math.min(frameDims.height, (e.clientY - rect.top) * scaleY)),
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getCanvasCoords(e);
    setIsDrawing(true);
    setDragStart(coords);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !dragStart) return;
    const current = getCanvasCoords(e);
    const x1 = Math.min(dragStart.x, current.x);
    const y1 = Math.min(dragStart.y, current.y);
    const x2 = Math.max(dragStart.x, current.x);
    const y2 = Math.max(dragStart.y, current.y);
    setBox([Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)]);
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
    setDragStart(null);
  };

  // Preset box actions
  const setCenterBox = () => {
    if (!frameDims) return;
    const cw = frameDims.width * 0.25;
    const ch = frameDims.height * 0.25;
    const cx = frameDims.width / 2;
    const cy = frameDims.height / 2;
    setBox([
      Math.round(cx - cw / 2),
      Math.round(cy - ch / 2),
      Math.round(cx + cw / 2),
      Math.round(cy + ch / 2),
    ]);
  };

  const setFullFrame = () => {
    if (!frameDims) return;
    setBox([0, 0, frameDims.width, frameDims.height]);
  };

  // -------------------------------------------------------------
  // Start Tracking Session
  // -------------------------------------------------------------
  const startTracking = useCallback(async () => {
    setError(null);
    setBusy(true);

    try {
      if (mode === "upload") {
        const file = fileRef.current;
        const fd = new FormData();

        if (file) {
          fd.append("file", file);
        }
        if (cropFileRef.current) {
          fd.append("crop_file", cropFileRef.current);
        }
        fd.append("text_prompt", textPrompt || "visual target");
        fd.append("tracker_name", tracker);
        if (box) {
          fd.append("initial_box", JSON.stringify(box));
        }
        fd.append("loop", String(loop));
        fd.append("show_trails", String(showTrails));

        const res = await fetch("/api/track", { method: "POST", body: fd });
        if (!res.ok) {
          const txt = await res.text();
          throw new Error(txt.slice(0, 200) || `HTTP ${res.status}`);
        }
        const info = await res.json();
        setSessionId(info.id);
        setStreamTs(Date.now());
        setStatus(null);
      } else {
        // RTSP mode
        if (!rtspUrl.trim()) throw new Error("Please enter a stream URL");
        const res = await fetch("/api/rtsp", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url: rtspUrl.trim(),
            text_prompt: textPrompt || "visual target",
            initial_box: box ? JSON.stringify(box) : null,
            tracker_name: tracker,
            show_trails: showTrails,
            reconnect_limit: 0,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const info = await res.json();
        setSessionId(info.id);
        setStreamTs(Date.now());
        setStatus(null);
      }
    } catch (err: any) {
      setError(err.message ?? String(err));
    } finally {
      setBusy(false);
    }
  }, [mode, rtspUrl, textPrompt, tracker, box, loop, showTrails]);

  // -------------------------------------------------------------
  // Live Parameter Synchronization
  // -------------------------------------------------------------
  const lastSent = useRef<string>("");
  useEffect(() => {
    if (!sessionId) return;
    const payload = {
      text_prompt: textPrompt,
      tracker_name: tracker,
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
        // Ignore live control hiccups
      }
    }, 300);
    return () => clearTimeout(t);
  }, [sessionId, textPrompt, tracker, showTrails, loop]);

  // -------------------------------------------------------------
  // Polling Status
  // -------------------------------------------------------------
  useEffect(() => {
    if (!sessionId) return;
    const poll = async () => {
      try {
        const res = await fetch(`/api/status/${sessionId}`);
        if (res.ok) setStatus(await res.json());
      } catch {
        // continue
      }
    };
    poll();
    const id = window.setInterval(poll, 400);
    return () => window.clearInterval(id);
  }, [sessionId]);

  const stopSession = useCallback(async () => {
    if (!sessionId) return;
    try {
      await fetch(`/api/stop/${sessionId}`, { method: "POST" });
    } catch {
      // ignore
    }
  }, [sessionId]);

  const streamUrl = sessionId ? `/api/stream/${sessionId}?t=${streamTs}` : null;

  return (
    <main>
      <header>
        <h1>
          <span>🎯</span> Detector-Free Visual Tracking
        </h1>
        <p className="sub">
          Give an object image or draw a bounding box in <b>Frame 1</b>. The tracker follows its visual appearance across subsequent frames without needing generic object detection. Supports <b>OSTrack</b>, <b>STARK</b>, and <b>Template Matching</b>.
        </p>
      </header>

      <div className="grid">
        {/* Left Column: Input & Frame 1 Selector */}
        <section className="card">
          <div className="card-title">
            <span>Video & Target Setup</span>
            <button className="btn-demo" onClick={loadDemo} disabled={busy}>
              ⚡ Load Demo Video
            </button>
          </div>

          <div className="tabs">
            <button
              className={`tab${mode === "upload" ? " on" : ""}`}
              onClick={() => setMode("upload")}
            >
              Upload Video
            </button>
            <button
              className={`tab${mode === "rtsp" ? " on" : ""}`}
              onClick={() => setMode("rtsp")}
            >
              RTSP / Camera URL
            </button>
          </div>

          {mode === "upload" ? (
            <>
              <div
                className={`drag${dragHot ? " hot" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragHot(true);
                }}
                onDragLeave={() => setDragHot(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragHot(false);
                  const f = e.dataTransfer.files?.[0];
                  if (f) handleFileChange(f);
                }}
                onClick={() => {
                  const input = document.createElement("input");
                  input.type = "file";
                  input.accept = "video/*";
                  input.onchange = () => {
                    if (input.files?.[0]) handleFileChange(input.files[0]);
                  };
                  input.click();
                }}
              >
                {selectedFileName ? (
                  <span style={{ color: "#38bdf8", fontWeight: 600 }}>
                    ✓ {selectedFileName}
                  </span>
                ) : (
                  "Drag & drop video here, or click to browse"
                )}
              </div>

              {/* Frame 1 Interactive Selector */}
              {frame1DataUrl && (
                <div>
                  <label>Frame 1 — Click & Drag to Select Object</label>
                  <div className="canvas-wrap">
                    <canvas
                      ref={canvasRef}
                      onMouseDown={handleMouseDown}
                      onMouseMove={handleMouseMove}
                      onMouseUp={handleMouseUp}
                    />
                    <div className="canvas-hint">Drag box over target</div>
                  </div>

                  {box && (
                    <div className="coord-bar">
                      <span>Box: [{box.join(", ")}]</span>
                      <span>
                        {box[2] - box[0]} × {box[3] - box[1]} px
                      </span>
                    </div>
                  )}

                  <div className="btn-group">
                    <button onClick={setCenterBox}>Center Box</button>
                    <button onClick={setFullFrame}>Full Frame</button>
                    <button onClick={() => setBox(null)}>Clear Box</button>
                  </div>
                </div>
              )}

              {/* Optional Crop File Upload */}
              <label>Or Upload Object Crop Image (Optional)</label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  if (e.target.files?.[0]) {
                    cropFileRef.current = e.target.files[0];
                    setCropFileName(e.target.files[0].name);
                  }
                }}
                style={{ marginBottom: 14 }}
              />
              {cropFileName && (
                <div style={{ fontSize: "0.8rem", color: "#38bdf8", marginBottom: 14 }}>
                  ✓ Crop: {cropFileName}
                </div>
              )}
            </>
          ) : (
            <>
              <label>RTSP / RTMP / MJPEG Camera URL</label>
              <input
                type="text"
                value={rtspUrl}
                onChange={(e) => setRtspUrl(e.target.value)}
                placeholder="rtsp://user:pass@192.168.1.64:554/stream1"
              />
            </>
          )}

          <label>Tracking Algorithm</label>
          <select value={tracker} onChange={(e) => setTracker(e.target.value)}>
            <option value="ostrack">OSTrack (One-Stream Appearance Correlation)</option>
            <option value="stark">STARK (Spatio-Temporal Adaptive Reliability)</option>
            <option value="mock">Mock (Normalized Cross-Correlation Baseline)</option>
          </select>

          <label>Target Label / Prompt</label>
          <input
            type="text"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            placeholder="e.g. red car, player #7, blue box"
          />

          <div className="row">
            <label className="check-label">
              <input
                type="checkbox"
                checked={showTrails}
                onChange={(e) => setShowTrails(e.target.checked)}
              />
              Show Motion Trails
            </label>
            {mode === "upload" && (
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={loop}
                  onChange={(e) => setLoop(e.target.checked)}
                />
                Loop Video
              </label>
            )}
          </div>

          <button className="btn" onClick={startTracking} disabled={busy}>
            {busy ? "Starting Tracker…" : "▶ Start Tracking"}
          </button>

          {sessionId && (
            <button className="btn btn-stop" onClick={stopSession}>
              ⏹ Stop Tracking
            </button>
          )}

          {error && <div className="warn">⚠ {error}</div>}
        </section>

        {/* Right Column: Live Stream & Statistics */}
        <section className="card">
          <div className="modelbar">
            <span className={`dot${status?.model_state === "running" ? " live" : ""}`} />
            <b>Tracker:</b>
            <span className="pill accent">{status?.params?.tracker_name ?? tracker}</span>
            <span className="pill">{status?.model_state ?? "idle"}</span>
            {status?.current_box && (
              <span style={{ marginLeft: "auto", fontFamily: "monospace", fontSize: "0.78rem" }}>
                Target: [{status.current_box.join(", ")}]
              </span>
            )}
          </div>

          <div className="streambox">
            {streamUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={streamUrl} alt="Live Visual Tracking Stream" />
            ) : (
              <div className="off">
                Select a video, draw a box on Frame 1, and press <b>Start Tracking</b> to watch the real-time annotated stream.
              </div>
            )}
          </div>

          {status && (
            <>
              <div className="stat-bar">
                <div className="stat-item">
                  <b>{status.frame_count}</b>
                  <span>Frames Processed</span>
                </div>
                <div className="stat-item">
                  <b>{status.active_tracks}</b>
                  <span>Active Target</span>
                </div>
                <div className="stat-item">
                  <b>{status.elapsed_s}s</b>
                  <span>Elapsed</span>
                </div>
                <div className="stat-item">
                  <b>{status.params?.frame_rate ?? 24} FPS</b>
                  <span>Stream Rate</span>
                </div>
              </div>

              {status.warn && <div className="warn">⚠ {status.warn}</div>}
              {status.error && <div className="warn">⚠ {status.error}</div>}
              {status.finished && !status.error && (
                <div className="ok">
                  {status.is_live
                    ? "Live stream closed"
                    : "Video processing complete — paused on last frame"}
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </main>
  );
}