import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import vercel from '@astrojs/vercel';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  integrations: [
    mdx({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex]
    })
  ],
  output: 'server',
  adapter: vercel({
    webAnalytics: {
      enabled: false
    },
    functionPerRoute: false,
    edgeMiddleware: false,
    imageService: true,
    imagesConfig: {
      sizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
      domains: [],
      formats: ['image/avif', 'image/webp']
    }
  })
});
