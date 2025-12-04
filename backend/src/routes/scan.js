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

// Multer config
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, uuidv4() + path.extname(file.originalname))
});
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
    
    // Send to ML with heatmap flag
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath), req.file.originalname);
    form.append('heatmaps', req.body.heatmaps || 'true');
    
    const response = await axios.post(ML_URL + '/detect', form, {
      headers: form.getHeaders(),
      timeout: 300000,
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
    
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

module.exports = router;
