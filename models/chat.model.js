const mongoose = require("mongoose");

const messageSchema = new mongoose.Schema({
  sender: { type: String, enum: ["user", "ai"], required: true },
  content: { type: String, required: true },
  embedding: { type: [Number] },  // Vector embedding for messages
  fileRefs: [{                    // References to uploaded files
    fileId: { type: mongoose.Schema.Types.ObjectId, ref: "File" },
    fileName: { type: String }
  }],
  timestamp: { type: Date, default: Date.now }
});

const summarySchema = new mongoose.Schema({
  text: { type: String, required: true },
  embedding: { type: [Number], required: true },
  messageRange: {
    start: { type: Number, required: true },
    end: { type: Number, required: true }
  },
  timestamp: { type: Date, default: Date.now }
});

const chatSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  title: { type: String, default: "New Chat" },
  messages: [messageSchema],
  summaries: [summarySchema],  // Conversation summaries for long-term context
  isActive: { type: Boolean, default: true },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

// Update the timestamp when chat is modified
chatSchema.pre("save", function(next) {
  this.updatedAt = Date.now();
  next();
});

// Create index for vector search on message embeddings
chatSchema.index(
  { "messages.embedding": "vector" },
  {
    name: "message_embedding_index",
    vectorOptions: { 
      type: "cosine", 
      dimensions: 768  // Dimensions of the Gecko embedding model
    }
  }
);

// Create index for vector search on summary embeddings
chatSchema.index(
  { "summaries.embedding": "vector" },
  {
    name: "summary_embedding_index",
    vectorOptions: { 
      type: "cosine", 
      dimensions: 768  // Dimensions of the Gecko embedding model
    }
  }
);

module.exports = mongoose.model("Chat", chatSchema);