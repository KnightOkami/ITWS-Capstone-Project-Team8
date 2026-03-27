import React, { useState, useCallback } from "react";
import Plot from "react-plotly.js";
import { BASE_API_URL } from "../scripts/config.js";

const PointNetPage = () => {
  const [points, setPoints] = useState([]);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [fileName, setFileName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [totalPointCount, setTotalPointCount] = useState(0);

  // Handle file upload and send to backend
 const processFile = async (file) => {
  if (!file) return;

  setIsLoading(true);
  setFileName(file.name);

  try {
    const text = await file.text();

    const allPts = text
      .trim()
      .split("\n")
      .map((line) => line.trim().split(/\s+/).map(Number))
      .filter((row) => row.length >= 3 && row.slice(0, 3).every(Number.isFinite))
      .map((row) => row.slice(0, 3));
    setTotalPointCount(allPts.length);
    const MAX_PREVIEW_POINTS = 5000;

    let previewPts = allPts;
    if (allPts.length > MAX_PREVIEW_POINTS) {
      const step = Math.ceil(allPts.length / MAX_PREVIEW_POINTS);
      previewPts = allPts.filter((_, i) => i % step === 0).slice(0, MAX_PREVIEW_POINTS);
    }

    setPoints(previewPts);

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${BASE_API_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setPrediction(data.predicted_label);
    setConfidence(data.confidence);
  } catch (err) {
    console.error("❌ Prediction error:", err);
    setPrediction("Error");
    setConfidence(0);
  } finally {
    setIsLoading(false);
  }
};

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    processFile(file);
  };

  // Drag & Drop handlers
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.txt')) {
      processFile(file);
    }
  }, []);

  return (
    <div className="h-[calc(100vh-4rem)] overflow-hidden app-surface text-[color:var(--foreground)]">
      {/* Framed Container */}
      <div className="max-w-[98%] mx-auto px-3 md:px-4 py-4 h-full">
        <div className="app-frame p-3 md:p-4 h-full">
      {/* Main Content */}
      <div className="flex flex-col lg:flex-row h-full">
        {/* LEFT: 3D Visualization (70%) */}
        <div className="flex-[7] p-2 lg:p-3">
          <div className="h-full rounded-2xl card-elevated bg-white overflow-hidden relative">
            {/* Subtle grid overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#e5e7eb_1px,transparent_1px),linear-gradient(to_bottom,#e5e7eb_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40 pointer-events-none"></div>
            
            {/* Glowing corner accents */}
            <div className="absolute top-0 left-0 w-32 h-32 bg-[#3148F6]/10 rounded-full blur-3xl"></div>
            <div className="absolute bottom-0 right-0 w-40 h-40 bg-[#1C4B8F]/10 rounded-full blur-3xl"></div>

            <div className="relative h-full flex flex-col p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold flex items-center gap-3">
                  <span className="w-1 h-6 bg-gradient-to-b from-[var(--blue-primary)] to-[var(--blue-muted)] rounded-full"></span>
                  3D Point Cloud Viewer
                </h2>
                {totalPointCount.toLocaleString()} total points
{points.length !== totalPointCount && (
  <div>Showing preview of {points.length.toLocaleString()} points</div>
)}
              </div>

              {points.length > 0 ? (
                <div className="flex-1 rounded-xl overflow-hidden shadow-inner border border-neutral-200 bg-white">
                  <Plot
                    data={[
                      {
                        x: points.map((p) => p[0]),
                        y: points.map((p) => p[1]),
                        z: points.map((p) => p[2]),
                        mode: "markers",
                        type: "scatter3d",
                        marker: { 
                          size: 2, 
                          color: points.map((p) => p[2]),
                          colorscale: [
                            [0, '#576CA8'],
                            [0.5, '#274690'],
                            [1, '#1B264F']
                          ],
                          showscale: false,
                          opacity: 0.8
                        },
                      },
                    ]}
                    layout={{
                      autosize: true,
                      paper_bgcolor: "rgba(255,255,255,0)",
                      plot_bgcolor: "rgba(255,255,255,0)",
                      scene: {
                        xaxis: { 
                          color: "#6B7280",
                          gridcolor: "#E5E7EB",
                          title: { text: "X", font: { color: "#274690", size: 14 } }
                        },
                        yaxis: { 
                          color: "#6B7280",
                          gridcolor: "#E5E7EB",
                          title: { text: "Y", font: { color: "#274690", size: 14 } }
                        },
                        zaxis: { 
                          color: "#6B7280",
                          gridcolor: "#E5E7EB",
                          title: { text: "Z", font: { color: "#274690", size: 14 } }
                        },
                        bgcolor: "#FFFFFF",
                      },
                      margin: { l: 0, r: 0, b: 0, t: 0 },
                      font: { family: "Inter, system-ui, sans-serif" }
                    }}
                    config={{
                      displayModeBar: true,
                      displaylogo: false,
                      modeBarButtonsToRemove: ['toImage'],
                    }}
                    style={{ width: "100%", height: "100%" }}
                  />
                </div>
              ) : (
                <div className="flex-1 flex items-center justify-center rounded-xl border-2 border-dashed border-neutral-300 bg-white">
                  <div className="text-center space-y-4">
                    <div className="mx-auto w-20 h-20 rounded-2xl bg-[var(--muted)] flex items-center justify-center shadow">
                      <svg className="w-10 h-10 text-[#94A3B8]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-lg text-[color:var(--foreground)]/70 font-medium">No point cloud loaded</p>
                      <p className="text-sm text-[color:var(--foreground)]/55 mt-1">Upload a .txt file to visualize</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT: Upload & Classification Panel (30%) */}
        <div className="flex-[3] p-2 lg:p-3 lg:pl-0">
          <div className="h-full rounded-2xl card-elevated bg-white p-4 flex flex-col overflow-y-auto">
            
            {/* Upload Section */}
            <div className="space-y-4 flex-shrink-0">
              <div>
                <h2 className="text-xl font-semibold mb-2 flex items-center gap-3">
                  <span className="w-1 h-6 bg-gradient-to-b from-[var(--blue-primary)] to-[var(--blue-muted)] rounded-full"></span>
                  Upload & Classify
                </h2>
                <p className="text-sm text-[color:var(--foreground)]/60">Upload your point cloud file</p>
              </div>

              {/* Drag & Drop Zone */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative border-2 border-dashed rounded-xl p-4 transition-all duration-300 ${
                  isDragging
                    ? "border-[var(--blue-primary)] bg-[var(--blue-primary)]/5 shadow-lg shadow-indigo-500/20"
                    : "border-neutral-300 bg-[#F9FAFB] hover:border-neutral-400"
                }`}
              >
                <div className="text-center space-y-2">
                  <div className="mx-auto w-10 h-10 rounded-xl bg-[var(--muted)] flex items-center justify-center">
                    <svg className="w-5 h-5 text-[var(--blue-primary)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-sm text-[color:var(--foreground)]/80 font-medium mb-0.5">
                      Drag & drop your file here
                    </p>
                    <p className="text-xs text-[color:var(--foreground)]/60">or click to browse</p>
                  </div>
                  <input
                    type="file"
                    accept=".txt"
                    onChange={handleFileUpload}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                </div>
              </div>

              {/* File Name Display */}
              {fileName && (
                <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-[var(--muted)] border border-neutral-200 animate-in fade-in slide-in-from-top-2 duration-300">
                  <svg className="w-5 h-5 text-[var(--blue-primary)] flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span className="text-sm text-[color:var(--foreground)]/80 truncate flex-1">{fileName}</span>
                  <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              )}
            </div>

            {/* Loading State */}
            {isLoading && (
              <div className="mt-8 flex items-center justify-center gap-3 py-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="w-2 h-2 bg-[#93C5FD] rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                <div className="w-2 h-2 bg-[#3B82F6] rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                <div className="w-2 h-2 bg-[#1C4B8F] rounded-full animate-bounce"></div>
                <span className="text-[#1C2127]/60 ml-2">Processing...</span>
              </div>
            )}

            {/* Results Card */}
            {!isLoading && prediction && (
              <div className="mt-4 space-y-3 animate-in fade-in slide-in-from-bottom-6 duration-500 flex-shrink-0">
                <div className="h-px bg-gradient-to-r from-transparent via-neutral-300 to-transparent"></div>
                
                <div className="p-4 rounded-xl card-elevated bg-white space-y-3 overflow-hidden">
                  <h3 className="text-sm font-semibold text-[color:var(--foreground)]/60 uppercase tracking-wider">Classification Results</h3>
                  
                  {/* Predicted Label */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-[color:var(--foreground)]/60">Predicted Class</span>
                    </div>
                    <div className="px-4 py-3 rounded-lg bg-[var(--blue-primary)]/5 border border-[var(--blue-primary)]/30">
                      <p className="text-2xl font-bold text-transparent bg-gradient-to-r from-[var(--blue-primary)] to-[var(--blue-muted)] bg-clip-text capitalize">
                        {prediction.replace(/_/g, ' ')}
                      </p>
                    </div>
                  </div>

                  {/* Confidence Score */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-[color:var(--foreground)]/60">Confidence Score</span>
                      <span className="text-2xl font-bold text-transparent bg-gradient-to-r from-[var(--blue-muted)] to-[var(--blue-deep)] bg-clip-text">
                        {confidence?.toFixed(2)}%
                      </span>
                    </div>
                    {/* Progress Bar */}
                    <div className="h-3 bg-neutral-200 rounded-full overflow-hidden border border-neutral-300">
                      <div
                        className="h-full bg-gradient-to-r from-[var(--blue-muted)] via-[var(--blue-primary)] to-[var(--blue-deep)] rounded-full transition-all duration-1000 ease-out shadow"
                        style={{ width: `${confidence}%` }}
                      ></div>
                    </div>
                    <div className="flex justify-between text-xs text-[color:var(--foreground)]/50">
                      <span>0%</span>
                      <span>50%</span>
                      <span>100%</span>
                    </div>
                  </div>

                  {/* Status Badge */}
                  <div className="pt-4 flex items-center justify-center gap-2">
                    <div className={`px-4 py-2 rounded-full text-sm font-medium ${
                      confidence > 80 
                        ? 'bg-[var(--accent-green)]/10 text-[var(--accent-green)] border border-[var(--accent-green)]/30' 
                        : confidence > 60
                        ? 'bg-[var(--accent-orange)]/10 text-[var(--accent-orange)] border border-[var(--accent-orange)]/30'
                        : 'bg-[var(--blue-muted)]/10 text-[var(--blue-muted)] border border-[var(--blue-muted)]/30'
                    }`}>
                      {confidence > 80 ? '🎯 High Confidence' : confidence > 60 ? '⚡ Medium Confidence' : 'ℹ️ Low Confidence'}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Empty State */}
            {!isLoading && !prediction && (
              <div className="mt-auto pt-8 text-center">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--muted)] border border-neutral-200 text-sm text-[color:var(--foreground)]/60">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Awaiting file upload
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
        </div>
      </div>
    </div>
  );
};

export default PointNetPage;