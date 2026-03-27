import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { createBrowserRouter, Link } from "react-router";
import { RouterProvider } from "react-router/dom";
import PotreePage from "./pages/PotreePage";
import PointNetPage from "./pages/PointNetPage";
import Navbar from "./components/Navbar";
import SegmentationPage from "./pages/SegmentationPage";


const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <>
        <Navbar />
        <div className="min-h-screen app-surface text-[color:var(--foreground)]">
          <div className="max-w-7xl mx-auto px-6 md:px-8 py-10">
            <div className="app-frame p-6 md:p-8">
            {/* Hero Section */}
            <section className="grid grid-cols-1 md:grid-cols-2 gap-10 items-center mb-16">
              <div>
                <span className="inline-flex items-center px-3 py-1 rounded-full bg-[var(--muted)] text-sm text-[color:var(--foreground)]/70 border border-[color:var(--border)]">Point Cloud Platform</span>
                <h1 className="mt-4 text-5xl md:text-6xl font-extrabold leading-tight">
                  Powerful 3D Analytics, <span className="bg-gradient-to-r from-[var(--blue-primary)] to-[var(--blue-muted)] bg-clip-text text-transparent">Built for Teams</span>
                </h1>
                <p className="mt-5 text-lg text-[color:var(--foreground)]/70 max-w-xl">
                  Visualize, classify, and explore massive point clouds with responsive tools and AI models designed for enterprise workflows.
                </p>
                <div className="mt-8 flex gap-4">
                  <Link to="/PointNetPage" className="px-6 py-3 rounded-lg btn-primary font-semibold">Try Classifier</Link>
                  <Link to="/potree" className="px-6 py-3 rounded-lg btn-ghost font-semibold">Open Viewer</Link>
                  <Link to="/segment" className="px-6 py-3 rounded-lg btn-ghost font-semibold">Try Segmentation</Link>
                </div>
              </div>
              <div className="relative h-[340px] md:h-[420px]">
                <div className="absolute inset-0 rounded-3xl card-elevated overflow-hidden">
                  <div className="absolute -top-10 -left-10 w-72 h-72 rounded-full bg-[var(--blue-primary)]/20 blur-3xl"></div>
                  <div className="absolute -bottom-10 -right-10 w-80 h-80 rounded-full bg-[var(--blue-muted)]/20 blur-3xl"></div>
                  <svg className="absolute inset-0 w-full h-full" viewBox="0 0 600 400" fill="none">
                    <defs>
                      <linearGradient id="gradA" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stopColor="var(--blue-primary)"/>
                        <stop offset="100%" stopColor="var(--blue-muted)"/>
                      </linearGradient>
                    </defs>
                    <circle cx="150" cy="120" r="70" fill="url(#gradA)" opacity="0.85"/>
                    <rect x="240" y="70" width="120" height="120" rx="24" fill="url(#gradA)" opacity="0.5"/>
                    <path d="M100 300 C 200 200, 300 400, 500 280" stroke="url(#gradA)" strokeWidth="10" strokeLinecap="round" opacity="0.5"/>
                  </svg>
                </div>
              </div>
            </section>

            {/* About Company Section */}
            <div className="mb-16 p-8 rounded-2xl card-elevated bg-white">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-[var(--muted)] flex items-center justify-center">
                    <svg className="w-6 h-6 text-[var(--blue-primary)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                    </svg>
                  </div>
                  <h2 className="text-3xl font-bold">About Twinner</h2>
                </div>
                {/* Partnership Badge */}
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--muted)] border border-[color:var(--border)]">
                  <span className="text-xs text-[color:var(--foreground)]/70">In partnership with</span>
                  <div className="w-6 h-6 rounded bg-white flex items-center justify-center p-0.5">
                    <svg viewBox="0 0 100 100" className="w-full h-full">
                      <text x="50" y="58" fontSize="42" fontWeight="bold" fill="#E51937" textAnchor="middle" fontFamily="Arial, sans-serif">RPI</text>
                    </svg>
                  </div>
                </div>
              </div>
              <p className="text-[color:var(--foreground)]/80 leading-relaxed mb-4">
                Twinner is a pioneering technology company specializing in 3D spatial data processing and artificial intelligence. 
                We empower organizations to unlock the full potential of point cloud data through innovative visualization and 
                machine learning solutions.
              </p>
              <p className="text-[color:var(--foreground)]/65 leading-relaxed">
                Our platform combines industry-leading tools with state-of-the-art deep learning models to deliver 
                comprehensive insights from complex 3D environments, serving industries from architecture and construction 
                to autonomous vehicles and robotics.
              </p>
            </div>
            {/* Features Grid */}
            <div className="mb-6">
              <h2 className="text-3xl font-bold mb-8 text-center">Twinner Platform</h2>
              <div className="features-splash p-6 md:p-8 max-w-6xl mx-auto">
                <div className="grid grid-cols-1 md:grid-cols-2 items-stretch gap-8">
                {/* Potree Card */}
                <div className="group p-8 rounded-2xl card-elevated card-tint-primary transition-all duration-300 hover:-translate-y-1 border-l-4 border-[var(--blue-primary)] h-full">
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-14 h-14 rounded-xl bg-[var(--muted)] flex items-center justify-center group-hover:scale-110 transition-transform">
                      <svg className="w-7 h-7 text-[var(--blue-primary)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                    </div>
                    <h3 className="text-2xl font-bold group-hover:text-[var(--blue-primary)] transition-colors">Potree Viewer</h3>
                  </div>
                  <p className="text-[color:var(--foreground)]/75 mb-3 leading-relaxed">
                    Interactive WebGL-based visualization for massive point cloud datasets. Explore billions of points 
                    in real-time with smooth navigation and rendering.
                  </p>
                  <div className="flex flex-wrap gap-2 mb-6">
                    <span className="px-3 py-1 rounded-full text-xs bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] border border-[var(--accent-cyan)]/30">Real-time Rendering</span>
                    <span className="px-3 py-1 rounded-full text-xs bg-[var(--accent-purple)]/10 text-[var(--accent-purple)] border border-[var(--accent-purple)]/30">WebGL</span>
                    <span className="px-3 py-1 rounded-full text-xs bg-[var(--accent-orange)]/10 text-[var(--accent-orange)] border border-[var(--accent-orange)]/30">Large Scale</span>
                  </div>
                  <div className="mt-2 flex items-center justify-between">
                    <Link to="/potree" className="px-4 py-2 rounded-lg btn-primary font-medium transition-all duration-300">
                      Launch Potree
                    </Link>
                    <a
                      href="https://github.com/potree/potree"
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg btn-ghost transition-colors"
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true" className="w-5 h-5 fill-current">
                        <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.483 0-.237-.009-.868-.013-1.703-2.782.603-3.369-1.342-3.369-1.342-.454-1.155-1.11-1.464-1.11-1.464-.908-.62.069-.607.069-.607 1.004.07 1.531 1.032 1.531 1.032.892 1.529 2.341 1.088 2.91.833.091-.647.349-1.088.635-1.338-2.221-.253-4.555-1.112-4.555-4.946 0-1.092.39-1.988 1.029-2.688-.103-.253-.446-1.27.098-2.646 0 0 .84-.269 2.75 1.026A9.564 9.564 0 0112 7.07c.851.004 1.707.115 2.507.338 1.909-1.295 2.748-1.026 2.748-1.026.546 1.376.202 2.393.099 2.646.64.7 1.028 1.596 1.028 2.688 0 3.842-2.337 4.69-4.565 4.938.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .268.18.58.688.481A10.013 10.013 0 0022 12c0-5.523-4.477-10-10-10z"/>
                      </svg>
                      <span className="text-sm">GitHub</span>
                    </a>
                  </div>
                </div>

                {/* PointNet++ Card */}
                <div className="group p-8 rounded-2xl card-elevated card-tint-muted transition-all duration-300 hover:-translate-y-1 border-l-4 border-[var(--blue-muted)] h-full">
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-14 h-14 rounded-xl bg-[var(--muted)] flex items-center justify-center group-hover:scale-110 transition-transform">
                      <svg className="w-7 h-7 text-[var(--blue-muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                    </div>
                    <h3 className="text-2xl font-bold group-hover:text-[var(--blue-muted)] transition-colors">PointNet++ Classifier</h3>
                  </div>
                  <p className="text-[color:var(--foreground)]/75 mb-3 leading-relaxed">
                    Deep learning-powered 3D object classification using hierarchical neural networks. Achieve 
                    state-of-the-art accuracy in recognizing objects from raw point cloud data.
                  </p>
                  <div className="flex flex-wrap gap-2 mb-6">
                    <span className="px-3 py-1 rounded-full text-xs bg-[var(--accent-green)]/10 text-[var(--accent-green)] border border-[var(--accent-green)]/30">Deep Learning</span>
                    <span className="px-3 py-1 rounded-full text-xs bg-[var(--accent-pink)]/10 text-[var(--accent-pink)] border border-[var(--accent-pink)]/30">TensorFlow</span>
                    <span className="px-3 py-1 rounded-full text-xs bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] border border-[var(--accent-cyan)]/30">AI Classification</span>
                  </div>
                  <div className="mt-2 flex items-center justify-between">
                    <Link to="/PointNetPage" className="px-4 py-2 rounded-lg btn-primary font-medium transition-all duration-300">
                      Try Classifier
                    </Link>
                    <a
                      href="https://github.com/charlesq34/pointnet"
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg btn-ghost transition-colors"
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true" className="w-5 h-5 fill-current">
                        <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.483 0-.237-.009-.868-.013-1.703-2.782.603-3.369-1.342-3.369-1.342-.454-1.155-1.11-1.464-1.11-1.464-.908-.62.069-.607.069-.607 1.004.07 1.531 1.032 1.531 1.032.892 1.529 2.341 1.088 2.91.833.091-.647.349-1.088.635-1.338-2.221-.253-4.555-1.112-4.555-4.946 0-1.092.39-1.988 1.029-2.688-.103-.253-.446-1.27.098-2.646 0 0 .84-.269 2.75 1.026A9.564 9.564 0 0112 7.07c.851.004 1.707.115 2.507.338 1.909-1.295 2.748-1.026 2.748-1.026.546 1.376.202 2.393.099 2.646.64.7 1.028 1.596 1.028 2.688 0 3.842-2.337 4.69-4.565 4.938.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .268.18.58.688.481A10.013 10.013 0 0022 12c0-5.523-4.477-10-10-10z"/>
                      </svg>
                      <span className="text-sm">GitHub</span>
                    </a>
                  </div>
                </div>
              </div>
              </div>
            </div>

            {/* Call to Action */}
            <div className="text-center mt-10 p-8 rounded-2xl card-elevated card-tint-muted">
              <h3 className="text-2xl font-bold mb-3">Ready to explore 3D data?</h3>
              <p className="text-[color:var(--foreground)]/65 mb-6">Choose a tool to get started with point cloud processing</p>
              <div className="flex items-center justify-center gap-4">
                <Link to="/potree" className="px-6 py-3 rounded-lg btn-primary font-semibold transition-all duration-300 hover:scale-105">
                  Launch Potree
                </Link>
                <Link to="/PointNetPage" className="px-6 py-3 rounded-lg btn-ghost font-semibold transition-all duration-300 hover:scale-105">
                  Try Classification
                </Link>
              </div>
            </div>
            </div>
          </div>
        </div>
      </>
    ),
  },
  {
    path: "/potree",
    element: (
      <>
        <Navbar />
        <PotreePage />
      </>
    ),
  },
  {
    path: "/PointNetPage",
    element: (
      <>
        <Navbar />
        <PointNetPage />
      </>
    ),
  },
  {
  path: "/segment",
  element: (
    <>
      <Navbar />
      <SegmentationPage />
    </>
  ),
}
]);

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <div className="font-sans">
      <RouterProvider router={router} />
    </div>
  </StrictMode>
);