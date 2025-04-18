const fs = require('fs');
const path = require('path');
const multer = require('multer');
const { GoogleGenerativeAI } = require("@google/generative-ai");
const pdfParse = require('pdf-parse');
const sharp = require('sharp');

// Initialize Google Generative AI with API key
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = process.env.UPLOAD_DIR || 'uploads';
    
    // Create directory if it doesn't exist
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    // Create unique filename with timestamp
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const ext = path.extname(file.originalname);
    cb(null, file.fieldname + '-' + uniqueSuffix + ext);
  }
});

// File filter to validate uploads
const fileFilter = (req, file, cb) => {
  // Accept only PDFs, images, and text files
  const allowedTypes = ['application/pdf', 'image/jpeg', 'image/png', 'image/webp', 'text/plain'];
  
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type. Only PDF, JPG, PNG, WEBP, and TXT files are allowed.'), false);
  }
};

// Configure multer middleware
exports.upload = multer({
  storage: storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB max
  fileFilter: fileFilter
});

// Extract text from PDF
exports.extractTextFromPdf = async (filePath) => {
  try {
    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdfParse(dataBuffer);
    return data.text;
  } catch (error) {
    console.error('Error extracting text from PDF:', error);
    throw new Error('Failed to extract text from PDF');
  }
};

// Extract text from image using Gemini Vision
exports.extractTextFromImage = async (filePath) => {
  try {
    // Optimize image before processing
    const optimizedImagePath = `${filePath}-optimized.jpg`;
    await sharp(filePath)
      .resize(1000) // Resize to max 1000px width/height
      .jpeg({ quality: 80 }) // Compress
      .toFile(optimizedImagePath);
    
    // Read the optimized image
    const imageData = fs.readFileSync(optimizedImagePath);
    
    // Convert to base64
    const base64Image = imageData.toString('base64');
    
    // Use Gemini Vision to extract text
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    
    const imagePart = {
      inlineData: {
        data: base64Image,
        mimeType: "image/jpeg"
      }
    };

    const prompt = "Extract all text from this image. Return only the extracted text without any additional comments or explanations.";
    
    const result = await model.generateContent([prompt, imagePart]);
    const text = result.response.text();
    
    // Clean up optimized image
    fs.unlinkSync(optimizedImagePath);
    
    return text;
  } catch (error) {
    console.error('Error extracting text from image:', error);
    throw new Error('Failed to extract text from image');
  }
};

// Clean up files after processing
exports.cleanupFile = (filePath) => {
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch (error) {
    console.error('Error cleaning up file:', error);
  }
};

// Process file based on type
exports.processFile = async (file) => {
  try {
    let extractedText = '';
    
    // Process based on file type
    if (file.mimetype === 'application/pdf') {
      extractedText = await exports.extractTextFromPdf(file.path);
    } else if (file.mimetype.startsWith('image/')) {
      extractedText = await exports.extractTextFromImage(file.path);
    } else if (file.mimetype === 'text/plain') {
      extractedText = fs.readFileSync(file.path, 'utf8');
    }
    
    // Chunk the text if it's large
    const chunks = exports.chunkText(extractedText);
    
    return {
      fileName: file.originalname,
      fileType: file.mimetype,
      filePath: file.path,
      extractedText,
      chunks
    };
  } catch (error) {
    console.error('Error processing file:', error);
    throw new Error(`Failed to process file: ${error.message}`);
  }
};

// Chunk text for large documents
exports.chunkText = (text, chunkSize = 1000, overlap = 100) => {
  const chunks = [];
  
  // If text is small enough, return as single chunk
  if (text.length <= chunkSize) {
    return [text];
  }
  
  // Split into sentences to avoid breaking in the middle
  const sentences = text.split(/(?<=[.!?])\s+/);
  let currentChunk = '';
  
  for (const sentence of sentences) {
    // If adding this sentence would exceed chunk size, save current chunk and start new one
    if (currentChunk.length + sentence.length > chunkSize) {
      chunks.push(currentChunk);
      // Start new chunk with overlap from end of previous chunk
      const overlapText = currentChunk.slice(-overlap);
      currentChunk = overlapText + sentence;
    } else {
      currentChunk += (currentChunk ? ' ' : '') + sentence;
    }
  }
  
  // Add the final chunk if not empty
  if (currentChunk) {
    chunks.push(currentChunk);
  }
  
  return chunks;
};