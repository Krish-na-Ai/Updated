const express = require("express");
const router = express.Router();
const { protect } = require("../middlewares/auth.middleware");
const { upload } = require("../utils/file-utils");
const { uploadFile, getUserFiles, deleteFile } = require("../controllers/file.controller");

// Routes for file management
router.post("/upload", protect, upload.single('file'), uploadFile);
router.get("/", protect, getUserFiles);
router.delete("/:id", protect, deleteFile);

module.exports = router;