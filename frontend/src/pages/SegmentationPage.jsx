import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

const BASE_API_URL = "http://localhost:8080";

const SEMANTIC_LABELS = {
  0: "ceiling",
  1: "floor",
  2: "wall",
  3: "beam",
  4: "column",
  5: "window",
  6: "door",
  7: "table",
  8: "chair",
  9: "sofa",
  10: "bookcase",
  11: "board",
  12: "clutter",
};

function colorForLabel(label) {
  const palette = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
  ];

  return palette[label % palette.length];
}

export default function SegmentationPage() {
  const [fileName, setFileName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [points, setPoints] = useState([]);
  const [instances, setInstances] = useState([]);
  const [numPoints, setNumPoints] = useState(0);
  const [numPointsReturned, setNumPointsReturned] = useState(0);
  const [numInstances, setNumInstances] = useState(0);
  const [error, setError] = useState("");
  const [selectedClasses, setSelectedClasses] = useState([]);

  const [officialFiles, setOfficialFiles] = useState([]);
  const [selectedOfficialFile, setSelectedOfficialFile] = useState("");

  useEffect(() => {
    fetchOfficialFiles();
  }, []);

  const toggleClassSelection = (classId) => {
  setSelectedClasses((prev) =>
    prev.includes(classId)
      ? prev.filter((id) => id !== classId)
      : [...prev, classId]
  );
};

const clearClassSelection = () => {
  setSelectedClasses([]);
};
  const fetchOfficialFiles = async () => {
    try {
      const res = await fetch(`${BASE_API_URL}/official_segment_files`);
      const data = await res.json();
      if (res.ok) {
        setOfficialFiles(data.files || []);
      }
    } catch (err) {
      console.error("Failed to fetch official files:", err);
    }
  };

  const applySegmentationResult = (data) => {
    setPoints(data.points || []);
    setInstances(data.instances || []);
    setNumPoints(data.num_points || 0);
    setNumPointsReturned(data.num_points_returned || 0);
    setNumInstances(data.num_instances || 0);
    setFileName(data.file_name || "");
    setSelectedClasses([]);
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError("");
    setFileName(file.name);
    setSelectedClasses([]);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${BASE_API_URL}/segment`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Segmentation request failed");
      }

      applySegmentationResult(data);
    } catch (err) {
      console.error("Segmentation error:", err);
      setError(err.message || "Unknown error");
      setPoints([]);
      setInstances([]);
      setNumPoints(0);
      setNumPointsReturned(0);
      setNumInstances(0);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadOfficial = async () => {
    if (!selectedOfficialFile) return;

    setIsLoading(true);
    setError("");

    try {
      const res = await fetch(
        `${BASE_API_URL}/official_segment_result/${encodeURIComponent(selectedOfficialFile)}`
      );
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Failed to load official result");
      }

      applySegmentationResult(data);
    } catch (err) {
      console.error("Official segmentation load error:", err);
      setError(err.message || "Unknown error");
    } finally {
      setIsLoading(false);
    }
  };

  const displayedPoints = useMemo(() => {
  if (selectedClasses.length === 0) return points;
  return points.filter((p) => selectedClasses.includes(p[3]));
}, [points, selectedClasses]);

  const plotData = useMemo(() => {
    if (!displayedPoints.length) return [];

    return [
      {
        x: displayedPoints.map((p) => p[0]),
        y: displayedPoints.map((p) => p[1]),
        z: displayedPoints.map((p) => p[2]),
        mode: "markers",
        type: "scatter3d",
        marker: {
          size: 1,
          color: displayedPoints.map((p) => {
        if (p.length >= 9) {
            return `rgb(${p[6]}, ${p[7]}, ${p[8]})`;
        }
        return colorForLabel(p[3]);
        }),
          opacity: 0.9,
        },
        text: displayedPoints.map((p) => {
          const label = p[4];
          const confidence = p[5];
          return `Class: ${SEMANTIC_LABELS[label] ?? label}<br>Confidence: ${
            confidence !== undefined ? confidence.toFixed(3) : "N/A"
          }`;
        }),
        hovertemplate:
          "x=%{x}<br>y=%{y}<br>z=%{z}<br>%{text}<extra></extra>",
      },
    ];
  }, [displayedPoints]);
const sceneRanges = useMemo(() => {
  if (!points.length) return null;

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (const p of points) {
    const x = p[0];
    const y = p[1];
    const z = p[2];

    if (x < minX) minX = x;
    if (x > maxX) maxX = x;

    if (y < minY) minY = y;
    if (y > maxY) maxY = y;

    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  const pad = 0.05;

  const padRange = (minVal, maxVal) => {
    const span = maxVal - minVal || 1;
    return [minVal - span * pad, maxVal + span * pad];
  };

  return {
    x: padRange(minX, maxX),
    y: padRange(minY, maxY),
    z: padRange(minZ, maxZ),
  };
}, [points]);
  return (
    <div className="h-[calc(100vh-4rem)] overflow-hidden app-surface text-foreground">
      <div className="max-w-[98%] mx-auto px-3 md:px-4 py-4 h-full">
        <div className="app-frame p-3 md:p-4 h-full flex">
          <aside className="w-80 bg-gray-100 border border-gray-300 rounded-md overflow-y-auto relative p-4">
            <h1 className="font-semibold text-xl mb-2 text-black">
              Segmentation Viewer
            </h1>
            <p className="text-sm text-gray-600 mb-4">
              Upload one point cloud file or load an official semseg output.
            </p>

            <div className="mb-5">
              <div className="text-sm font-semibold text-black mb-2">
                Official Result Files
              </div>
              <select
                value={selectedOfficialFile}
                onChange={(e) => setSelectedOfficialFile(e.target.value)}
                className="w-full border border-gray-300 rounded-md p-2 text-sm text-black bg-white mb-2"
              >
                <option value="">Select a result file</option>
                {officialFiles.map((file) => (
                  <option key={file} value={file}>
                    {file}
                  </option>
                ))}
              </select>
              <button
                onClick={handleLoadOfficial}
                className="w-full py-2 rounded-md border text-sm font-medium bg-blue-600 text-white border-blue-600"
              >
                Load Official Result
              </button>
            </div>

            <div className="mb-5">
              <div className="text-sm font-semibold text-black mb-2">
                Or Upload a File
              </div>
              <input
                type="file"
                accept=".txt"
                onChange={handleFileChange}
                className="block w-full text-sm text-black"
              />
            </div>

            {isLoading && (
              <p className="text-blue-700 text-sm mb-3">
                Loading segmentation...
              </p>
            )}

            {error && (
              <p className="text-red-600 text-sm mb-3">Error: {error}</p>
            )}

            {fileName && !error && (
              <div className="mb-4 text-sm text-black space-y-1">
                <div>
                  <strong>File:</strong> {fileName}
                </div>
                <div>
                  <strong>Total points used:</strong> {numPoints.toLocaleString()}
                </div>
                <div>
                  <strong>Points returned to viewer:</strong>{" "}
                  {numPointsReturned.toLocaleString()}
                </div>
                <div>
                  <strong>Detected groups:</strong>{" "}
                  {numInstances.toLocaleString()}
                </div>
               <strong>Viewing:</strong>{" "}
{selectedClasses.length === 0
  ? "All classes"
  : selectedClasses
      .map((id) => SEMANTIC_LABELS[id] ?? `Label ${id}`)
      .join(", ")}
              </div>
            )}

            <button
              onClick={clearClassSelection}
              className={`w-full mb-3 py-2 rounded-md border text-sm font-medium ${
                selectedClasses.length === 0
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-black border-gray-300 hover:bg-gray-200"
              }`}
            >
              Show All
            </button>

            <h2 className="font-semibold text-lg mb-3 text-black">Classes</h2>

            {!instances.length && (
              <div className="text-gray-500 text-sm">No classes loaded yet.</div>
            )}

            <div className="space-y-2 pb-2">
              {instances.map((instance) => (
               <button
  key={instance.id}
  onClick={() => toggleClassSelection(instance.id)}
  className={`block w-full text-left p-3 rounded-md border ${
    selectedClasses.includes(instance.id)
      ? "bg-blue-600 text-white border-blue-600"
      : "bg-white text-black border-gray-300 hover:bg-gray-200"
  }`}
>
                  <div className="font-semibold mb-1">
                    {instance.semantic_label || SEMANTIC_LABELS[instance.id] || `Label ${instance.id}`}
                  </div>
                  <div className="text-sm opacity-85">
                    Points: {instance.count.toLocaleString()}
                  </div>
                  <div className="text-xs opacity-75 mt-1">
                    Min: {instance.bbox_min.map((v) => v.toFixed(2)).join(", ")}
                  </div>
                  <div className="text-xs opacity-75">
                    Max: {instance.bbox_max.map((v) => v.toFixed(2)).join(", ")}
                  </div>
                </button>
              ))}
            </div>
          </aside>

          <main className="flex-1 ml-4 bg-white rounded-md relative">
            {!plotData.length ? (
              <div className="flex items-center justify-center h-full text-gray-500 text-xl">
                Load an official result file to see segmented points here.
              </div>
            ) : (
              <Plot
                data={plotData}
                layout={{
                  paper_bgcolor: "#ffffff",
                  plot_bgcolor: "#ffffff",
                  font: { color: "#111111" },
                  margin: { l: 0, r: 0, b: 0, t: 0 },
                  scene: {
  bgcolor: "#ffffff",
  xaxis: {
    visible: false,
    range: sceneRanges ? sceneRanges.x : undefined,
  },
  yaxis: {
    visible: false,
    range: sceneRanges ? sceneRanges.y : undefined,
  },
  zaxis: {
    visible: false,
    range: sceneRanges ? sceneRanges.z : undefined,
  },
  aspectmode: "data",
},
                  showlegend: false,
                }}
                style={{ width: "100%", height: "100%" }}
                config={{ responsive: true }}
              />
            )}
          </main>
        </div>
      </div>
    </div>
  );
}