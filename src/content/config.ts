import { defineCollection, z } from 'astro:content';

const ebookCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
    description: z.string(),
    author: z.string(),
    coauthor: z.string().optional(),
    date: z.string().optional(),
    publishDate: z.date(),
    tags: z.array(z.string()).optional(),
    cover: z.string().optional(),
  }),
});

export const collections = {
  ebook: ebookCollection,
};
