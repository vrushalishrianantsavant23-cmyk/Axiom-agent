import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Axiom-Agent",
  description: "Self-verifying AI for claim and document fact-checking",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}