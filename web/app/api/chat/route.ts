import { google } from '@ai-sdk/google';
import { generateText } from 'ai';
import { SimpleRAG } from '@/lib/rag';

// Initialize RAG system with some sample documents
const rag = new SimpleRAG();

// Add some sample documents (you can replace this with your own data)
rag.addDocument(`
Next.js is a React framework that enables functionality such as server-side rendering and generating static websites. 
It provides many features out of the box including routing, API routes, and optimization features.
Next.js is production-ready and is used by many companies for building web applications.
`, { source: 'nextjs-info' });

rag.addDocument(`
React is a JavaScript library for building user interfaces. It was developed by Facebook and is now maintained by Meta.
React uses a component-based architecture and implements a virtual DOM for efficient updates.
React components can be functional or class-based, with hooks providing state management for functional components.
`, { source: 'react-info' });

rag.addDocument(`
Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn.
Machine learning is a subset of AI that enables systems to automatically learn and improve from experience.
Large Language Models (LLMs) like GPT are examples of AI systems that can understand and generate human-like text.
`, { source: 'ai-info' });

export async function POST(req: Request) {
  try {
    const { message } = await req.json();

    if (!message) {
      return Response.json({ error: 'Message is required' }, { status: 400 });
    }

    // Get relevant context from RAG system
    const context = rag.getContext(message, 3);

    // Create the prompt with context
    const prompt = `
Based on the following context, please answer the user's question. If the context doesn't contain relevant information, you can still provide a helpful response but mention that the information might not be from the provided documents.

Context:
${context}

User Question: ${message}

Please provide a helpful and accurate response:`;

    // Generate response using Gemini
    const { text } = await generateText({
      model: google('gemini-1.5-flash'),
      prompt: prompt,
      maxTokens: 500,
      temperature: 0.7,
    });

    return Response.json({ 
      response: text,
      context: context,
      message: 'Response generated successfully'
    });

  } catch (error) {
    console.error('Error in chat API:', error);
    return Response.json(
      { error: 'Failed to generate response' }, 
      { status: 500 }
    );
  }
} 