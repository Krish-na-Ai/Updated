const Chat = require("../models/chat.model");
const { generateChatTitle } = require("../utils/ai-helpers");

exports.startChat = async (req, res) => {
  try {
    // Check if there's an existing active chat with no messages
    const existingEmptyChat = await Chat.findOne({ 
      userId: req.user._id,
      isActive: true,
      "messages.0": { $exists: false }
    });

    if (existingEmptyChat) {
      // If an empty chat exists, return that instead of creating a new one
      return res.status(200).json({ chatId: existingEmptyChat._id, isExisting: true });
    }

    // Create a new chat
    const chat = await Chat.create({ userId: req.user._id });
    res.status(201).json({ chatId: chat._id, isExisting: false });
  } catch (err) {
    res.status(500).json({ message: "Could not create chat", error: err.message });
  }
};

exports.getUserChats = async (req, res) => {
  try {
    const chats = await Chat.find({ userId: req.user._id })
      .select("_id title createdAt messages")
      // Count the number of messages for each chat
      .exec();
    
    // Filter out empty chats and add messageCount
    const filteredChats = chats
      .filter(chat => chat.messages.length > 0)
      .map(chat => ({
        _id: chat._id,
        title: chat.title,
        createdAt: chat.createdAt,
        messageCount: chat.messages.length,
        lastMessage: chat.messages[chat.messages.length - 1].content.substring(0, 50)
      }))
      .sort((a, b) => b.createdAt - a.createdAt);
    
    res.json(filteredChats);
  } catch (err) {
    res.status(500).json({ message: "Failed to load chats", error: err.message });
  }
};

exports.getChatById = async (req, res) => {
  try {
    const chat = await Chat.findOne({ _id: req.params.id, userId: req.user._id });
    if (!chat) return res.status(404).json({ message: "Chat not found" });
    res.json(chat);
  } catch (err) {
    res.status(500).json({ message: "Error getting chat", error: err.message });
  }
};

exports.updateChatTitle = async (req, res) => {
  try {
    const { title } = req.body;
    
    if (!title || title.trim() === '') {
      return res.status(400).json({ message: "Title cannot be empty" });
    }

    const chat = await Chat.findOneAndUpdate(
      { _id: req.params.id, userId: req.user._id },
      { title },
      { new: true }
    );
    
    if (!chat) return res.status(404).json({ message: "Chat not found" });
    res.json({ message: "Title updated", title: chat.title });
  } catch (err) {
    res.status(500).json({ message: "Error updating chat title", error: err.message });
  }
};

exports.deleteChat = async (req, res) => {
  try {
    const result = await Chat.deleteOne({ _id: req.params.id, userId: req.user._id });
    if (result.deletedCount === 0) return res.status(404).json({ message: "Chat not found" });
    res.json({ message: "Chat deleted successfully" });
  } catch (err) {
    res.status(500).json({ message: "Error deleting chat", error: err.message });
  }
};