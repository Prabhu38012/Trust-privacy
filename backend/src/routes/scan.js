const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const { auth } = require('../middleware/auth');

const router = express.Router();

// Setup upload directory
const uploadDir = path.join(__dirname, '../../uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

<<<<<<< HEAD
// Day 6: Configurable file size limit (default 50MB)
const MAX_FILE_SIZE = parseInt(process.env.MAX_FILE_SIZE_MB || '50') * 1024 * 1024;

// Day 6: Allowed file types
const ALLOWED_MIME_TYPES = [
  // Images
  'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp', 'image/tiff',
  // Videos
  'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska',
  // Documents
  'application/pdf'
];

// Day 6: File filter for type validation
const fileFilter = (req, file, cb) => {
  if (ALLOWED_MIME_TYPES.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error(`Unsupported file type: ${file.mimetype}. Allowed: images, videos, PDF.`), false);
  }
};

// Multer config with Day 6 improvements
=======
// Multer config
>>>>>>> 4336965e78d04836c64348343ce98ab69529cd81
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, uuidv4() + path.extname(file.originalname))
});
<<<<<<< HEAD

const upload = multer({
  storage,
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter
});

// Error handler for multer errors
const handleMulterError = (err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({
        message: `File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB.`,
        error: 'FILE_TOO_LARGE'
      });
    }
    return res.status(400).json({ message: err.message, error: 'UPLOAD_ERROR' });
  }
  if (err) {
    return res.status(400).json({ message: err.message, error: 'INVALID_FILE' });
  }
  next();
};

const ML_URL = process.env.ML_SERVICE_URL || 'http://127.0.0.1:5000';

// Day 6: Document analysis endpoint (ELA, EXIF, PDF support)
router.post('/document', auth, upload.single('file'), handleMulterError, async (req, res) => {
  let filePath = null;

  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded', error: 'NO_FILE' });
    }

    filePath = req.file.path;
    console.log('[Document Analysis] File:', req.file.originalname);
    console.log('[Document Analysis] Size:', (req.file.size / (1024 * 1024)).toFixed(2), 'MB');

    // Check ML service health
    try {
      await axios.get(ML_URL + '/health', { timeout: 5000 });
    } catch (e) {
      if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
      return res.status(503).json({
        message: 'ML service unavailable. Please try again later.',
        error: 'ML_SERVICE_UNAVAILABLE'
      });
    }

    // Send to ML document analysis endpoint
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath), req.file.originalname);

    const response = await axios.post(ML_URL + '/analyze-document', form, {
      headers: form.getHeaders(),
      timeout: 120000,  // 2 minutes for document processing
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });

    console.log('[Document Analysis] Result:', response.data?.summary?.verdict);

    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);

    res.json(response.data);

  } catch (error) {
    console.error('[Document Analysis] Error:', error.message);
    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);

    // Provide meaningful error messages
    if (error.response?.status === 400) {
      return res.status(400).json({
        message: error.response.data?.error || 'Invalid file format',
        error: 'INVALID_FORMAT'
      });
    }
    if (error.response?.status === 503) {
      return res.status(503).json({
        message: 'PDF processing not available. Poppler may not be installed.',
        error: 'PDF_NOT_SUPPORTED'
      });
    }

    res.status(500).json({
      message: 'Document analysis failed. Please try again.',
      error: error.message
    });
  }
});

// Deepfake scan endpoint (existing, with improved error handling)
router.post('/upload', auth, upload.single('file'), handleMulterError, async (req, res) => {
  let filePath = null;

  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded', error: 'NO_FILE' });
    }

    filePath = req.file.path;
    console.log('[Scan] File:', req.file.originalname);
    console.log('[Scan] Size:', (req.file.size / (1024 * 1024)).toFixed(2), 'MB');
    console.log('[Scan] Heatmaps:', req.body.heatmaps || 'true');

    // Check ML service
    try {
      await axios.get(ML_URL + '/health', { timeout: 5000 });
    } catch (e) {
      if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
      return res.status(503).json({
        message: 'ML service unavailable. Please try again later.',
        error: 'ML_SERVICE_UNAVAILABLE'
      });
    }

=======
const upload = multer({ storage, limits: { fileSize: 100 * 1024 * 1024 } });

const ML_URL = process.env.ML_SERVICE_URL || 'http://127.0.0.1:5000';

router.post('/upload', auth, upload.single('file'), async (req, res) => {
  let filePath = null;
  
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }
    
    filePath = req.file.path;
    console.log('[Scan] File:', req.file.originalname);
    console.log('[Scan] Heatmaps:', req.body.heatmaps || 'true');
    
    // Check ML service
    try {
      await axios.get(ML_URL + '/health', { timeout: 3000 });
    } catch (e) {
      if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
      return res.status(503).json({ message: 'ML service not running' });
    }
    
>>>>>>> 4336965e78d04836c64348343ce98ab69529cd81
    // Send to ML with heatmap flag
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath), req.file.originalname);
    form.append('heatmaps', req.body.heatmaps || 'true');
<<<<<<< HEAD

=======
    
>>>>>>> 4336965e78d04836c64348343ce98ab69529cd81
    const response = await axios.post(ML_URL + '/detect', form, {
      headers: form.getHeaders(),
      timeout: 300000,
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
<<<<<<< HEAD

    console.log('[Scan] Result:', response.data?.result?.verdict);
    console.log('[Scan] Heatmaps generated:', response.data?.result?.analysis_summary?.heatmaps_generated || 0);

    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);

    res.json(response.data);

  } catch (error) {
    console.error('[Scan] Error:', error.message);
    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
    res.status(500).json({
      message: 'Scan failed. Please try again.',
      error: error.message
    });
  }
});

router.get('/health', (req, res) => res.json({
  status: 'ok',
  maxFileSize: `${MAX_FILE_SIZE / (1024 * 1024)}MB`,
  allowedTypes: ALLOWED_MIME_TYPES
}));
=======
    
    console.log('[Scan] Result:', response.data?.result?.verdict);
    console.log('[Scan] Heatmaps generated:', response.data?.result?.analysis_summary?.heatmaps_generated || 0);
    
    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
    
    res.json(response.data);
    
  } catch (error) {
    console.error('[Scan] Error:', error.message);
    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
    res.status(500).json({ message: 'Scan failed', error: error.message });
  }
});

router.get('/health', (req, res) => res.json({ status: 'ok' }));
>>>>>>> 4336965e78d04836c64348343ce98ab69529cd81

module.exports = router;
