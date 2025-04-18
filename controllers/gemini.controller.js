const Chat = require("../models/chat.model");
const File = require("../models/file.model");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { generateEmbedding, findRelevantChunks, generateSummary } = require("../utils/embedding-utils");

// Initialize Google Generative AI with API key
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

let io; // Will be initialized in the setupSocketIo function

// Function to set up socket.io reference
exports.setupSocketIo = (socketIo) => {
  io = socketIo;
};

exports.sendMessage = async (req, res) => {
  const { message } = req.body;
  const { fileIds } = req.body; // Optional array of file IDs to reference
  const chatId = req.params.id;
  const userId = req.user._id.toString();

  try {
    const chat = await Chat.findOne({ _id: chatId, userId: req.user._id });
    if (!chat) return res.status(404).json({ message: "Chat not found" });

    // Process file references if any
    let fileReferences = [];
    if (fileIds && fileIds.length > 0) {
      const files = await File.find({
        _id: { $in: fileIds },
        userId: req.user._id
      }).select("_id fileName");
      
      fileReferences = files.map(file => ({
        fileId: file._id,
        fileName: file.fileName
      }));
    }

    // Generate embedding for the user message
    const messageEmbedding = await generateEmbedding(message);

    // Add user message to the chat history
    chat.messages.push({ 
      sender: "user", 
      content: message,
      embedding: messageEmbedding,
      fileRefs: fileReferences
    });
    
    // For first message, we'll generate a title later
    const isFirstMessage = chat.messages.length === 1;
    
    await chat.save();

    // Signal that processing has started
    if (io) {
      io.to(userId).emit('processing', { chatId, status: 'started' });
    }

    // Check if we need to generate a summary
    // Generate summary every 10 messages
    if (chat.messages.length > 10 && chat.messages.length % 10 === 0) {
      // Get the last 10 messages to summarize
      const messagesForSummary = chat.messages.slice(-11, -1); // Not including the last user message
      
      // Generate summary
      const summary = await generateSummary(messagesForSummary);
      
      // Add to summaries
      chat.summaries.push({
        text: summary.text,
        embedding: summary.embedding,
        messageRange: {
          start: chat.messages.length - 11,
          end: chat.messages.length - 1
        }
      });
      
      await chat.save();
    }

    // Retrieve relevant context from files if referenced
    let fileContext = "";
    if (fileReferences.length > 0) {
      // Find all files referenced
      const referencedFiles = await File.find({
        _id: { $in: fileReferences.map(ref => ref.fileId) }
      });
      
      // Collect all chunks for vector search
      const allChunks = referencedFiles.flatMap(file => 
        file.chunks.map(chunk => ({
          text: chunk.text,
          embedding: chunk.embedding,
          fileName: file.fileName
        }))
      );
      
      // Find relevant chunks
      if (allChunks.length > 0) {
        const relevantFileChunks = await findRelevantChunks(message, allChunks, 3);
        
        // Format file context
        fileContext = relevantFileChunks.map(chunk => 
          `[From ${chunk.fileName}]: ${chunk.text}`
        ).join('\n\n');
      }
    }

    // Retrieve relevant previous messages
    const relevantMessageContext = await getRelevantMessages(chat, message);

    // Format previous messages for Gemini's history format
    const history = [];
    
    // Add last 5 messages for immediate context
    const recentMessages = chat.messages.slice(-6, -1); // Exclude the latest user message
    for (const msg of recentMessages) {
      history.push({
        role: msg.sender === "user" ? "user" : "model",
        parts: [{ text: msg.content }]
      });
    }

    // Get the model - using gemini-1.5-flash
    const model = genAI.getGenerativeModel({
      model: "gemini-1.5-flash",
      generationConfig: {
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 8192,
      },
    });

    // Create a chat session with history
    const chatSession = model.startChat({
      history,
      generationConfig: {
        maxOutputTokens: 8192,
      },
    });

    // Construct the prompt with context
    let contextPrompt = "";
    
    if (fileContext || relevantMessageContext) {
      contextPrompt = `[System] Prioritize (70%) this context over general knowledge:\n\n`;
      
      if (fileContext) {
        contextPrompt += `FILE CONTEXT:\n${fileContext}\n\n`;
      }
      
      if (relevantMessageContext) {
        contextPrompt += `CONVERSATION CONTEXT:\n${relevantMessageContext}\n\n`;
      }
      
      contextPrompt += `Now answer the user's query based on this context and your knowledge.\n\n`;
    }

    // Combine context and user message
    const promptWithContext = contextPrompt + message;

    // Send the message and get a streaming response
    const result = await chatSession.sendMessageStream(promptWithContext);
    
    let aiReply = "";

    // Process the stream chunks and emit to socket
    for await (const chunk of result.stream) {
      const chunkText = chunk.text();
      if (chunkText) {
        aiReply += chunkText;
        // Send chunk via socket.io if available
        if (io) {
          io.to(userId).emit('message-chunk', { 
            chatId, 
            chunk: chunkText 
          });
        }
      }
    }

    // Generate embedding for AI response
    const aiReplyEmbedding = await generateEmbedding(aiReply);

    // Save AI response to database
    chat.messages.push({ 
      sender: "ai", 
      content: aiReply,
      embedding: aiReplyEmbedding
    });
    
    // For the first message, generate a title based on conversation
    if (isFirstMessage) {
      const titlePrompt = `Based on this conversation, generate a very brief title (max 5 words):\nUser: ${message}\nAI: ${aiReply}`;
      try {
        const titleResponse = await model.generateContent(titlePrompt);
        const suggestedTitle = titleResponse.response.text().trim();
        chat.title = suggestedTitle.length > 50 
          ? suggestedTitle.substring(0, 47) + "..." 
          : suggestedTitle;
      } catch (titleErr) {
        console.error("Error generating title:", titleErr);
        // Fallback title based on user message
        chat.title = message.length > 30 
          ? message.substring(0, 27) + "..." 
          : message;
      }
    }
    
    await chat.save();

    // Signal completion via socket
    if (io) {
      io.to(userId).emit('processing', { 
        chatId, 
        status: 'completed', 
        title: chat.title 
      });
    }

    // For traditional HTTP response
    res.json({ 
      success: true, 
      message: "Message sent and processed",
      response: aiReply,
      title: chat.title
    });
  } catch (err) {
    console.error("Gemini API error:", err);
    
    // Inform client of error via socket
    if (io) {
      io.to(userId).emit('error', { 
        chatId, 
        error: err.message 
      });
    }
    
    // Check if the error response contains details
    const errorMessage = err.response?.data?.error?.message || err.message;
    res.status(500).json({ message: "Gemini API failed", error: errorMessage });
  }
};

// Function to get relevant messages from chat history
async function getRelevantMessages(chat, query) {
  try {
    // If no messages or only a few, no need for retrieval
    if (chat.messages.length <= 5) {
      return "";
    }
    
    // Get messages with embeddings (excluding most recent ones which are in immediate context)
    const messagesWithEmbeddings = chat.messages
      .slice(0, -5)
      .filter(msg => msg.embedding && msg.embedding.length > 0);
    
    // If no embeddings, check summaries
    if (messagesWithEmbeddings.length === 0) {
      return await getRelevantSummaries(chat, query);
    }
    
    // Find relevant messages
    const relevantMessages = await findRelevantChunks(
      query, 
      messagesWithEmbeddings.map(msg => ({
        text: msg.content,
        embedding: msg.embedding
      })),
      3
    );
    
    // Format for context
    return relevantMessages
      .map(msg => `${msg.text}`)
      .join('\n\n');
  } catch (error) {
    console.error('Error getting relevant messages:', error);
    return ""; // Return empty string on error
  }
}

// Function to get relevant summaries
async function getRelevantSummaries(chat, query) {
  try {
    // If no summaries, return empty
    if (!chat.summaries || chat.summaries.length === 0) {
      return "";
    }
    
    // Find relevant summaries
    const relevantSummaries = await findRelevantChunks(
      query,
      chat.summaries.map(summary => ({
        text: summary.text,
        embedding: summary.embedding
      })),
      1 // Just get the most relevant summary
    );
    
    // Format for context
    return relevantSummaries
      .map(summary => `CONVERSATION SUMMARY: ${summary.text}`)
      .join('\n\n');
  } catch (error) {
    console.error('Error getting relevant summaries:', error);
    return ""; // Return empty string on error
  }
}

// New function to handle image-based queries
exports.sendImageMessage = async (req, res) => {
  const { message } = req.body;
  const chatId = req.params.id;
  const userId = req.user._id.toString();
  
  try {
    // Check if file exists
    if (!req.file) {
      return res.status(400).json({ message: "No image uploaded" });
    }
    
    const chat = await Chat.findOne({ _id: chatId, userId: req.user._id });
    if (!chat) return res.status(404).json({ message: "Chat not found" });
    
    // Read image file
    const fs = require('fs');
    const imageData = fs.readFileSync(req.file.path);
    const base64Image = imageData.toString('base64');
    
    // Signal processing started
    if (io) {
      io.to(userId).emit('processing', { chatId, status: 'started' });
    }

    // Add user message to the chat history
    chat.messages.push({ 
      sender: "user", 
      content: message || "Image query",
      fileRefs: [{
        fileName: req.file.originalname
      }]
    });
    await chat.save();
    
    // Get Gemini vision model
    const model = genAI.getGenerativeModel({
      model: "gemini-1.5-flash",
      generationConfig: {
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 8192,
      },
    });
    
    // Prepare image part
    const imagePart = {
      inlineData: {
        data: base64Image,
        mimeType: req.file.mimetype
      }
    };
    
    // Prepare text part
    const textPart = {
      text: message || "What's in this image?"
    };
    
    // Send multimodal request
    const result = await model.generateContentStream([textPart, imagePart]);
    
    let aiReply = "";
    
    // Process the stream chunks and emit to socket
    // Process the stream chunks and emit to socket
    for await (const chunk of result.stream) {
      const chunkText = chunk.text();
      if (chunkText) {
        aiReply += chunkText;
        // Send chunk via socket.io if available
        if (io) {
          io.to(userId).emit('message-chunk', { 
            chatId, 
            chunk: chunkText 
          });
        }
      }
    }
    
    // Save AI response to database
    chat.messages.push({ 
      sender: "ai", 
      content: aiReply
    });
    
    // Generate title if this is the first exchange
    if (chat.messages.length <= 2) {
      const titlePrompt = `Based on this image analysis, generate a very brief title (max 5 words):
      Image description: ${message || "Image query"}
      AI analysis: ${aiReply.substring(0, 200)}`;
      
      try {
        const titleResponse = await genAI.getGenerativeModel({
          model: "gemini-1.5-flash",
        }).generateContent(titlePrompt);
        
        const suggestedTitle = titleResponse.response.text().trim();
        chat.title = suggestedTitle.length > 50 
          ? suggestedTitle.substring(0, 47) + "..." 
          : suggestedTitle;
      } catch (titleErr) {
        console.error("Error generating title:", titleErr);
        chat.title = "Image Analysis";
      }
    }
    
    await chat.save();
    
    // Clean up the uploaded file
    try {
      fs.unlinkSync(req.file.path);
    } catch (cleanupErr) {
      console.error("Error cleaning up file:", cleanupErr);
    }

    // Signal completion via socket
    if (io) {
      io.to(userId).emit('processing', { 
        chatId, 
        status: 'completed', 
        title: chat.title 
      });
    }

    // Return response
    res.json({ 
      success: true, 
      message: "Image processed",
      response: aiReply,
      title: chat.title
    });
    
  } catch (err) {
    console.error("Gemini Vision API error:", err);
    
    // Inform client of error via socket
    if (io) {
      io.to(userId).emit('error', { 
        chatId, 
        error: err.message 
      });
    }
    
    // Clean up the uploaded file
    if (req.file) {
      try {
        const fs = require('fs');
        fs.unlinkSync(req.file.path);
      } catch (cleanupErr) {
        console.error("Error cleaning up file:", cleanupErr);
      }
    }
    
    // Return error response
    const errorMessage = err.response?.data?.error?.message || err.message;
    res.status(500).json({ message: "Gemini Vision API failed", error: errorMessage });
  }
};