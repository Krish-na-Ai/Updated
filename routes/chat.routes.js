const express = require("express");
const router = express.Router();
const { protect } = require("../middlewares/auth.middleware");
const { 
  startChat, 
  getUserChats, 
  getChatById, 
  updateChatTitle,
  deleteChat
} = require("../controllers/chat.controller");
const { sendMessage, sendImageMessage } = require("../controllers/gemini.controller");
const { upload } = require("../utils/file-utils");

router.post("/start", protect, startChat);
router.get("/history", protect, getUserChats);
router.get("/:id", protect, getChatById);
router.post("/:id/message", protect, sendMessage);
router.post("/:id/image-message", protect, upload.single('image'), sendImageMessage);
router.patch("/:id/title", protect, updateChatTitle);
router.delete("/:id", protect, deleteChat);

module.exports = router;