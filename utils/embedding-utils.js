// utils/embedding-utils.js
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { withRetry } = require("./retry-utils");
require('dotenv').config();
// Initialize Google Generative AI with API key
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Cache for embeddings to reduce API calls
const embeddingCache = new Map();

// Generate text embeddings using Google's embedding model
exports.generateEmbedding = async (text) => {
  try {
    // Return from cache if exists
    const cacheKey = text.substring(0, 100); // Use beginning of text as cache key
    if (embeddingCache.has(cacheKey)) {
      return embeddingCache.get(cacheKey);
    }
    
    // Use Google's embedding model with retry
    const embedding = await withRetry(async () => {
      const model = genAI.getGenerativeModel({ model: "embedding-001" });
      const result = await model.embedContent(text);
      return result.embedding.values;
    }, { maxRetries: 2 });
    
    // Store in cache
    embeddingCache.set(cacheKey, embedding);
    
    // If cache grows too large, remove oldest entries
    if (embeddingCache.size > 1000) {
      const oldestKey = embeddingCache.keys().next().value;
      embeddingCache.delete(oldestKey);
    }
    
    return embedding;
  } catch (error) {
    console.error('Error generating embedding:', error);
    
    // Add fallback for development/testing without embeddings
    if (process.env.NODE_ENV === 'development') {
      console.warn('Returning mock embedding for development');
      return Array(768).fill(0).map(() => Math.random() * 2 - 1); // Random vector of dimension 768
    }
    
    throw new Error('Failed to generate embedding');
  }
};

// Rest of your functions...
exports.generateChunkEmbeddings = async (chunks) => {
  const embeddedChunks = [];
  
  for (const chunk of chunks) {
    try {
      const embedding = await exports.generateEmbedding(chunk);
      embeddedChunks.push({
        text: chunk,
        embedding
      });
    } catch (error) {
      console.warn('Error embedding chunk, continuing without embedding:', error.message);
      embeddedChunks.push({
        text: chunk,
        embedding: []
      });
    }
  }
  
  return embeddedChunks;
};

// Calculate cosine similarity between two vectors
exports.cosineSimilarity = (vecA, vecB) => {
  // Validate inputs
  if (!vecA || !vecB || vecA.length !== vecB.length || vecA.length === 0) {
    return 0; // Return lowest similarity for invalid inputs
  }
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  // Prevent division by zero
  if (normA === 0 || normB === 0) return 0;
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
};

// Find most relevant chunks for a query
exports.findRelevantChunks = async (query, embeddedChunks, count = 3) => {
  try {
    // Return empty if no chunks
    if (!embeddedChunks || embeddedChunks.length === 0) {
      return [];
    }
    
    const queryEmbedding = await exports.generateEmbedding(query);
    
    // Calculate similarity scores and add weights
    const scoredChunks = embeddedChunks
      .filter(chunk => chunk.embedding && chunk.embedding.length > 0)
      .map(chunk => {
        // Calculate semantic similarity (70% weight)
        const similarity = exports.cosineSimilarity(queryEmbedding, chunk.embedding) * 0.7;
        
        // Add length bonus (30% weight) - favor longer chunks for more context
        const lengthBonus = Math.min(1, chunk.text.length / 2000) * 0.3;
        
        return {
          ...chunk,
          score: similarity + lengthBonus
        };
      });
    
    // If no valid chunks, return empty
    if (scoredChunks.length === 0) {
      return [];
    }
    
    // Sort by score and get top chunks
    return scoredChunks
      .sort((a, b) => b.score - a.score)
      .slice(0, count);
  } catch (error) {
    console.error('Error finding relevant chunks:', error);
    return []; // Return empty array on error to prevent breaking the flow
  }
};

// Generate a summary of messages
exports.generateSummary = async (messages) => {
  try {
    // Format messages for summarization  
    const formattedMessages = messages.map(msg => 
      `${msg.sender === 'user' ? 'User' : 'AI'}: ${msg.content}`
    ).join('\n\n');
    
    // Create prompt for summarization
    const prompt = `Summarize the following conversation, focusing on technical details and key decisions. Keep the summary concise but include all important information:

${formattedMessages}`;

    // Generate summary with retry
    const summary = await withRetry(async () => {
      const model = genAI.getGenerativeModel({ 
        model: "gemini-1.5-flash",
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 4000,
        }
      });
      
      const result = await model.generateContent(prompt);
      return result.response.text();
    });
    
    // Generate embedding for the summary
    let embedding = [];
    try {
      embedding = await exports.generateEmbedding(summary);
    } catch (error) {
      console.warn('Error embedding summary, continuing without embedding:', error.message);
    }
    
    return {
      text: summary,
      embedding
    };
  } catch (error) {
    console.error('Error generating summary:', error);
    // Return a basic summary without embedding
    return {
      text: "Previous conversation covered various topics.",
      embedding: []
    };
  }
};