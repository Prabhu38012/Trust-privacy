const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const { auth } = require('../middleware/auth');

const router = express.Router();

// Configuration
const MAX_FILE_SIZE_MB = parseInt(process.env.MAX_FILE_SIZE_MB || '50');
const MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024;

// Setup upload directory
const uploadDir = path.join(__dirname, '../../uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Allowed file types for document analysis
const ALLOWED_MIME_TYPES = [
  'image/jpeg',
  'image/png',
  'image/gif',
  'image/webp',
  'image/bmp',
  'image/tiff',
  'application/pdf'
];

// File filter for document uploads
const fileFilter = (req, file, cb) => {
  if (ALLOWED_MIME_TYPES.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error(`Unsupported file type: ${file.mimetype}. Allowed types: ${ALLOWED_MIME_TYPES.join(', ')}`), false);
  }
};

// Multer config with size limit and file filter
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, uuidv4() + path.extname(file.originalname))
});

const upload = multer({ 
  storage, 
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter
});

const ML_URL = process.env.ML_SERVICE_URL || 'http://127.0.0.1:5000';

// Multer error handler middleware
const handleMulterError = (err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({ 
        message: `File too large. Maximum size is ${MAX_FILE_SIZE_MB}MB`,
        error: 'PAYLOAD_TOO_LARGE'
      });
    }
    return res.status(400).json({ 
      message: `Upload error: ${err.message}`,
      error: 'UPLOAD_ERROR'
    });
  }
  if (err && err.message && err.message.includes('Unsupported file type')) {
    return res.status(415).json({ 
      message: err.message,
      error: 'UNSUPPORTED_MEDIA_TYPE'
    });
  }
  next(err);
};

/**
 * POST /document/analyze
 * Analyze a document (image or PDF) for tampering
 * Uses Error Level Analysis (ELA) and EXIF metadata checks
 */
router.post('/analyze', auth, (req, res, next) => {
  upload.single('file')(req, res, (err) => {
    if (err) {
      return handleMulterError(err, req, res, next);
    }
    next();
  });
}, async (req, res) => {
  let filePath = null;
  
  try {
    if (!req.file) {
      return res.status(400).json({ 
        message: 'No file uploaded',
        error: 'NO_FILE'
      });
    }
    
    filePath = req.file.path;
    console.log('[Document] File:', req.file.originalname);
    console.log('[Document] Size:', (req.file.size / (1024 * 1024)).toFixed(2), 'MB');
    console.log('[Document] Type:', req.file.mimetype);
    
    // Check ML service health
    try {
      await axios.get(ML_URL + '/health', { timeout: 3000 });
    } catch (e) {
      if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
      return res.status(503).json({ 
        message: 'ML service not available. Please ensure the ML service is running.',
        error: 'ML_SERVICE_UNAVAILABLE'
      });
    }
    
    // Send to ML service for document analysis
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath), req.file.originalname);
    
    const response = await axios.post(ML_URL + '/analyze-document', form, {
      headers: form.getHeaders(),
      timeout: 120000, // 2 minute timeout for document analysis
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
    
    console.log('[Document] Verdict:', response.data?.result?.verdict);
    console.log('[Document] Tampering Score:', response.data?.result?.tampering_score);
    
    // Clean up uploaded file
    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
    
    res.json(response.data);
    
  } catch (error) {
    console.error('[Document] Error:', error.message);
    
    // Clean up on error
    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
    
    // Handle specific error cases
    if (error.response) {
      // ML service returned an error
      const status = error.response.status;
      const data = error.response.data;
      
      if (status === 413) {
        return res.status(413).json({
          message: data.error || 'File too large for processing',
          error: 'PAYLOAD_TOO_LARGE'
        });
      }
      if (status === 415) {
        return res.status(415).json({
          message: data.error || 'Unsupported file type',
          error: 'UNSUPPORTED_MEDIA_TYPE'
        });
      }
      if (status === 501) {
        return res.status(501).json({
          message: data.error || 'PDF support not available on server',
          error: 'PDF_NOT_SUPPORTED'
        });
      }
      
      return res.status(status).json({
        message: data.error || 'Document analysis failed',
        error: 'ANALYSIS_FAILED'
      });
    }
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        message: 'ML service not available. Please ensure the ML service is running.',
        error: 'ML_SERVICE_UNAVAILABLE'
      });
    }
    
    res.status(500).json({ 
      message: 'Document analysis failed', 
      error: error.message 
    });
  }
});

// Health check
router.get('/health', (req, res) => {
  res.json({ 
    status: 'ok',
    maxFileSizeMB: MAX_FILE_SIZE_MB,
    allowedTypes: ALLOWED_MIME_TYPES
  });
});

module.exports = router;
