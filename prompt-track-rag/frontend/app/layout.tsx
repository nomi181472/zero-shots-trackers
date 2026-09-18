import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Zero-Shot Tracking (RAG)",
  description:
    "Text-prompted tracking (GroundingDINO + RAG retrieval tracker) with live MJPEG preview",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}