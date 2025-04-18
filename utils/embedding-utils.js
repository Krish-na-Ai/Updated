const { GoogleGenerativeAI } = require("@google/generative-ai");

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
    
    // Use Google's embedding model
    const model = genAI.getGenerativeModel({ model: "embedding-001" });
    const result = await model.embedContent(text);
    const embedding = result.embedding.values;
    
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
    throw new Error('Failed to generate embedding');
  }
};

// Generate embeddings for chunks
exports.generateChunkEmbeddings = async (chunks) => {
  const embeddedChunks = [];
  
  for (const chunk of chunks) {
    const embedding = await exports.generateEmbedding(chunk);
    embeddedChunks.push({
      text: chunk,
      embedding
    });
  }
  
  return embeddedChunks;
};

// Calculate cosine similarity between two vectors
exports.cosineSimilarity = (vecA, vecB) => {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
};

// Find most relevant chunks for a query
exports.findRelevantChunks = async (query, embeddedChunks, count = 3) => {
  try {
    const queryEmbedding = await exports.generateEmbedding(query);
    
    // Calculate similarity scores and add weights
    const scoredChunks = embeddedChunks.map(chunk => {
      // Calculate semantic similarity (70% weight)
      const similarity = exports.cosineSimilarity(queryEmbedding, chunk.embedding) * 0.7;
      
      // Add length bonus (30% weight) - favor longer chunks for more context
      const lengthBonus = Math.min(1, chunk.text.length / 2000) * 0.3;
      
      return {
        ...chunk,
        score: similarity + lengthBonus
      };
    });
    
    // Sort by score and get top chunks
    return scoredChunks
      .sort((a, b) => b.score - a.score)
      .slice(0, count);
  } catch (error) {
    console.error('Error finding relevant chunks:', error);
    throw new Error('Failed to find relevant chunks');
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

    // Generate summary
    const model = genAI.getGenerativeModel({ 
      model: "gemini-1.5-flash",
      generationConfig: {
        temperature: 0.2,
        maxOutputTokens: 4000,
      }
    });
    
    const result = await model.generateContent(prompt);
    const summary = result.response.text();
    
    // Generate embedding for the summary
    const embedding = await exports.generateEmbedding(summary);
    
    return {
      text: summary,
      embedding
    };
  } catch (error) {
    console.error('Error generating summary:', error);
    throw new Error('Failed to generate summary');
  }
};