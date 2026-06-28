import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ['pisciform-unmonumental-bently.ngrok-free.dev'],
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
    ];
  },
  webpack: (config, { dev }) => {
    if (dev) {
      config.watchOptions = {
        poll: false,
        ignored: [
          '**/node_modules/**',
          '**/.next/**',
          '**/backend/**',
          '**/*.db',
          '**/*.db-shm',
          '**/*.db-wal',
        ],
      };
    }
    return config;
  },
};

export default nextConfig;
