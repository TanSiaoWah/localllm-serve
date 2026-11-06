import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Docker uses the minimal server; ordinary local builds keep next start.
  output: process.env.NEXT_STANDALONE === "1" ? "standalone" : undefined,
};

export default nextConfig;
