const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { auth } = require('../middleware/auth');
const blockchain = require('../services/blockchain');

const router = express.Router();

// Generate certificate for scan
router.post('/generate', auth, async (req, res) => {
  try {
    const { scanId, verdict, score, filename } = req.body;

    const certificate = {
      id: uuidv4(),
      scanId,
      userId: req.userId,
      verdict,
      score,
      filename,
      timestamp: Date.now(),
    };

    // Add to blockchain
    const block = blockchain.addCertificate(certificate);

    const certHash = blockchain.hashCertificate(certificate);

    res.json({
      certificate: {
        id: certificate.id,
        hash: certHash,
        blockMined: block !== null,
        blockIndex: block?.index,
        timestamp: certificate.timestamp,
      },
    });
  } catch (error) {
    console.error('Certificate generation error:', error);
    res.status(500).json({ message: 'Failed to generate certificate' });
  }
});

// Verify certificate
router.post('/verify', async (req, res) => {
  try {
    const { certificateHash } = req.body;

    if (!certificateHash) {
      return res.status(400).json({ message: 'Certificate hash required' });
    }

    const verification = blockchain.verifyCertificate(certificateHash);

    res.json(verification);
  } catch (error) {
    console.error('Verification error:', error);
    res.status(500).json({ message: 'Verification failed' });
  }
});

// Get blockchain status
router.get('/blockchain/status', auth, async (req, res) => {
  try {
    res.json({
      chainLength: blockchain.chain.length,
      pendingCertificates: blockchain.pendingCertificates.length,
      isValid: blockchain.isChainValid(),
      lastBlock: blockchain.getLastBlock(),
    });
  } catch (error) {
    res.status(500).json({ message: 'Failed to get blockchain status' });
  }
});

module.exports = router;
