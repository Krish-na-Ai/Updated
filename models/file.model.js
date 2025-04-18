const mongoose = require("mongoose");

const chunkSchema = new mongoose.Schema({
  text: { type: String, required: true },
  embedding: { type: [Number], required: true }
});

const fileSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  fileName: { type: String, required: true },
  fileType: { type: String, required: true },
  filePath: { type: String },
  extractedText: { type: String },
  chunks: [chunkSchema],
  createdAt: { type: Date, default: Date.now }
});

// Create index for vector search on embedding field
// Note: This assumes MongoDB Atlas with vector search capability is set up
fileSchema.index(
  { "chunks.embedding": "vector" },
  {
    name: "embedding_index",
    vectorOptions: { 
      type: "cosine", 
      dimensions: 768  // Dimensions of the Gecko embedding model
    }
  }
);

module.exports = mongoose.model("File", fileSchema);