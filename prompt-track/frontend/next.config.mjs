/** @type {import('next').NextConfig} */
const BACKEND_ORIGIN = process.env.BACKEND_ORIGIN || "http://127.0.0.1:8000";

const nextConfig = {
  async rewrites() {
    return [
      // Proxy /api/* to the FastAPI backend so the browser can hit the
      // MJPEG stream and control endpoints same-origin. If your backend runs
      // on a different port, start Next with:  BACKEND_ORIGIN=http://127.0.0.1:8010 npm run dev
      { source: "/api/:path*", destination: `${BACKEND_ORIGIN}/api/:path*` },
    ];
  },
};

export default nextConfig;