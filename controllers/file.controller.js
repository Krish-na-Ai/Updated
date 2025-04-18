const File = require("../models/file.model");
const { processFile, cleanupFile } = require("../utils/file-utils");
const { generateChunkEmbeddings } = require("../utils/embedding-utils");

exports.uploadFile = async (req, res) => {
  try {
    // Check if file exists
    if (!req.file) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    // Process the file based on its type
    const processedFile = await processFile(req.file);
    
    // Generate embeddings for chunks
    const embeddedChunks = await generateChunkEmbeddings(processedFile.chunks);
    
    // Save file information to database
    const file = await File.create({
      userId: req.user._id,
      fileName: processedFile.fileName,
      fileType: processedFile.fileType,
      filePath: processedFile.filePath,
      extractedText: processedFile.extractedText,
      chunks: embeddedChunks
    });

    // Clean up the file after processing if needed
    // Comment this out if you want to keep the original file
    cleanupFile(processedFile.filePath);
    
    res.status(201).json({
      fileId: file._id,
      fileName: file.fileName,
      fileType: file.fileType
    });
  } catch (err) {
    console.error("File upload error:", err);
    res.status(500).json({ message: "File upload failed", error: err.message });
  }
};

exports.getUserFiles = async (req, res) => {
  try {
    const files = await File.find({ userId: req.user._id })
      .select("_id fileName fileType createdAt")
      .sort({ createdAt: -1 });
    
    res.json(files);
  } catch (err) {
    console.error("Error retrieving files:", err);
    res.status(500).json({ message: "Failed to load files", error: err.message });
  }
};

exports.deleteFile = async (req, res) => {
  try {
    const result = await File.deleteOne({ 
      _id: req.params.id, 
      userId: req.user._id 
    });
    
    if (result.deletedCount === 0) {
      return res.status(404).json({ message: "File not found" });
    }
    
    res.json({ message: "File deleted successfully" });
  } catch (err) {
    console.error("Error deleting file:", err);
    res.status(500).json({ message: "Failed to delete file", error: err.message });
  }
};