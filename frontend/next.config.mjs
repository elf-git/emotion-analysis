/** @type {import('next').NextConfig} */
const nextConfig = {
  // Video analysis takes longer than the 30 second default proxy limit
  experimental: {
    proxyTimeout: 600000,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
    ];
  },
};

export default nextConfig;
