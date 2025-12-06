const express = require('express');
const crypto = require('crypto');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { auth } = require('../middleware/auth');
const blockchainService = require('../services/blockchain');
const Certificate = require('../models/Certificate');

const router = express.Router();

// Setup upload for verification
const uploadDir = path.join(__dirname, '../../uploads/verify');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}
const upload = multer({ dest: uploadDir, limits: { fileSize: 100 * 1024 * 1024 } });

/**
 * POST /api/certificate/generate
 */
router.post('/generate', auth, async (req, res) => {
  try {
    const { scanId, verdict, score, filename, fileHash: providedHash } = req.body;

    if (!scanId || !verdict || score === undefined) {
      return res.status(400).json({ message: 'Missing required fields' });
    }

    const fileHash = providedHash || '0x' + crypto.randomBytes(32).toString('hex');
    const blockchainAvailable = await blockchainService.initialize();

    let blockchainResult = null;
    let certificateId = crypto.randomUUID();

    if (blockchainAvailable) {
      try {
        const hashBuffer = Buffer.from(fileHash.replace('0x', ''), 'hex');
        blockchainResult = await blockchainService.issueCertificate(hashBuffer, verdict, Math.round(score));
        certificateId = blockchainResult.certificateId || certificateId;
        console.log('✅ Blockchain certificate issued:', blockchainResult.transactionHash);
      } catch (err) {
        console.error('⚠️ Blockchain failed:', err.message);
      }
    }

    const certificate = new Certificate({
      id: certificateId,
      scanId,
      userId: req.userId,
      filename: filename || 'unknown',
      fileHash,
      verdict,
      score: Math.round(score),
      timestamp: new Date(),
      blockchain: blockchainResult ? {
        transactionHash: blockchainResult.transactionHash,
        blockNumber: blockchainResult.blockNumber,
        network: blockchainResult.network,
        explorerUrl: blockchainResult.explorerUrl,
        onChain: true
      } : { onChain: false, reason: 'Blockchain not available' }
    });

    await certificate.save();
    res.json({ success: true, certificate });
  } catch (error) {
    console.error('Certificate generation error:', error);
    res.status(500).json({ message: 'Failed to generate certificate' });
  }
});

/**
 * GET /api/certificate/list
 */
router.get('/list', auth, async (req, res) => {
  try {
    const certificates = await Certificate.find({ userId: req.userId }).sort({ timestamp: -1 }).limit(50);
    res.json({ success: true, certificates });
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch certificates' });
  }
});

/**
 * GET /api/certificate/verify/:id
 */
router.get('/verify/:id', async (req, res) => {
  try {
    const certificate = await Certificate.findOne({ id: req.params.id });
    if (!certificate) {
      return res.status(404).json({ verified: false, message: 'Certificate not found' });
    }

    let onChainVerified = false;
    if (certificate.blockchain?.onChain) {
      try {
        const blockchainAvailable = await blockchainService.initialize();
        if (blockchainAvailable) {
          const onChainData = await blockchainService.verifyCertificate(certificate.id);
          onChainVerified = onChainData?.exists || false;
        }
      } catch (err) {
        console.error('On-chain verify failed:', err.message);
      }
    }

    res.json({
      verified: true,
      certificate: {
        id: certificate.id,
        filename: certificate.filename,
        verdict: certificate.verdict,
        score: certificate.score,
        timestamp: certificate.timestamp,
        fileHash: certificate.fileHash
      },
      blockchain: {
        onChain: certificate.blockchain?.onChain || false,
        verified: onChainVerified,
        transactionHash: certificate.blockchain?.transactionHash,
        explorerUrl: certificate.blockchain?.explorerUrl,
        network: certificate.blockchain?.network
      }
    });
  } catch (error) {
    res.status(500).json({ message: 'Verification failed' });
  }
});

/**
 * POST /api/certificate/verify
 */
router.post('/verify', upload.single('file'), async (req, res) => {
  let filePath = null;
  try {
    const { certificateId, clientHash } = req.body;
    if (!certificateId) return res.status(400).json({ message: 'Certificate ID required' });

    const certificate = await Certificate.findOne({ id: certificateId });
    if (!certificate) return res.status(404).json({ verified: false, message: 'Not found' });

    let computedHash = clientHash;
    if (req.file) {
      filePath = req.file.path;
      const fileBuffer = fs.readFileSync(filePath);
      computedHash = '0x' + crypto.createHash('sha256').update(fileBuffer).digest('hex');
      fs.unlinkSync(filePath);
      filePath = null;
    }

    if (!computedHash) return res.status(400).json({ message: 'File or hash required' });

    const hashMatch = certificate.fileHash.toLowerCase() === computedHash.toLowerCase();

    res.json({
      verified: hashMatch,
      certificate: { id: certificate.id, filename: certificate.filename, verdict: certificate.verdict, score: certificate.score, timestamp: certificate.timestamp, storedHash: certificate.fileHash },
      verification: { hashMatch, storedHash: certificate.fileHash, providedHash: computedHash, onChain: certificate.blockchain?.onChain || false, onChainVerified: hashMatch }
    });
  } catch (error) {
    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
    res.status(500).json({ message: 'Verification failed' });
  }
});

/**
 * GET /api/certificate/:id
 */
router.get('/:id', async (req, res) => {
  try {
    const certificate = await Certificate.findOne({ id: req.params.id });
    if (!certificate) return res.status(404).json({ message: 'Not found' });
    res.json({ success: true, certificate });
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch certificate' });
  }
});

module.exports = router;
