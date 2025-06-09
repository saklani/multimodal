import { encoding_for_model } from "js-tiktoken";

// Simple cosine similarity function
function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

// Simple text chunking
export function chunkText(text: string, chunkSize: number = 500): string[] {
  const encoding = encoding_for_model("gpt-4");
  const tokens = encoding.encode(text);
  const chunks: string[] = [];
  
  for (let i = 0; i < tokens.length; i += chunkSize) {
    const chunkTokens = tokens.slice(i, i + chunkSize);
    const chunkText = encoding.decode(chunkTokens);
    chunks.push(chunkText);
  }
  
  return chunks;
}

// Simple text embedding using character frequency (for demo purposes)
export function simpleTextEmbedding(text: string): number[] {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789 ';
  const embedding = new Array(chars.length).fill(0);
  const normalizedText = text.toLowerCase();
  
  for (const char of normalizedText) {
    const index = chars.indexOf(char);
    if (index !== -1) {
      embedding[index]++;
    }
  }
  
  // Normalize the embedding
  const total = embedding.reduce((sum, val) => sum + val, 0);
  return total > 0 ? embedding.map(val => val / total) : embedding;
}

export interface DocumentChunk {
  id: string;
  text: string;
  embedding: number[];
  metadata?: Record<string, any>;
}

export class SimpleRAG {
  private documents: DocumentChunk[] = [];

  addDocument(text: string, metadata?: Record<string, any>) {
    const chunks = chunkText(text);
    
    chunks.forEach((chunk, index) => {
      const embedding = simpleTextEmbedding(chunk);
      this.documents.push({
        id: `${Date.now()}-${index}`,
        text: chunk,
        embedding,
        metadata
      });
    });
  }

  search(query: string, topK: number = 5): DocumentChunk[] {
    const queryEmbedding = simpleTextEmbedding(query);
    
    const similarities = this.documents.map(doc => ({
      ...doc,
      similarity: cosineSimilarity(queryEmbedding, doc.embedding)
    }));
    
    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }

  getContext(query: string, topK: number = 3): string {
    const relevantChunks = this.search(query, topK);
    return relevantChunks.map(chunk => chunk.text).join('\n\n');
  }
} 