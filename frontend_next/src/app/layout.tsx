import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: "Resume Matcher | Autonomous AI Agent",
  description: "Production-grade AI agent for discovering and matching internships in Pakistan.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={outfit.variable}>
      <body className="antialiased font-sans relative">
        {/* Dynamic Background Effects */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden z-[-1]">
          <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-600/20 rounded-full blur-[60px]"></div>
          <div className="absolute top-[20%] right-[-5%] w-[30%] h-[50%] bg-purple-600/20 rounded-full blur-[80px]"></div>
          <div className="absolute bottom-[-20%] left-[20%] w-[50%] h-[40%] bg-blue-600/10 rounded-full blur-[60px]"></div>
        </div>
        {children}
      </body>
    </html>
  );
}
