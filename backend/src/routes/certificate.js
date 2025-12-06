const express = require('express');
const crypto = require('crypto');
const { auth } = require('../middleware/auth');
const blockchainService = require('../services/blockchain');
const Certificate = require('../models/Certificate');

const router = express.Router();

/**
 * POST /api/certificate/generate
 * Generate a blockchain certificate for a scan result
 */
router.post('/generate', auth, async (req, res) => {
  try {
    const { scanId, verdict, score, filename, fileHash: providedHash } = req.body;

    if (!scanId || !verdict || score === undefined) {
      return res.status(400).json({ 
        message: 'Missing required fields: scanId, verdict, score' 
      });
    }

    // Generate file hash if not provided
    const fileHash = providedHash || '0x' + crypto.randomBytes(32).toString('hex');

    // Check if blockchain is available
    const blockchainAvailable = await blockchainService.initialize();

    let blockchainResult = null;
    let certificateId = crypto.randomUUID();

    if (blockchainAvailable) {
      try {
        // Create a buffer from the hash for blockchain
        const hashBuffer = Buffer.from(fileHash.replace('0x', ''), 'hex');
        
        // Issue on-chain certificate
        blockchainResult = await blockchainService.issueCertificate(
          hashBuffer,
          verdict,
          Math.round(score)
        );

        certificateId = blockchainResult.certificateId || certificateId;
        console.log('✅ Blockchain certificate issued:', blockchainResult.transactionHash);
      } catch (blockchainError) {
        console.error('⚠️ Blockchain issuance failed, using off-chain:', blockchainError.message);
      }
    }

    // Create certificate record
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
      } : {
        onChain: false,
        reason: 'Blockchain not available'
      }
    });

    await certificate.save();

    res.json({
      success: true,
      certificate: {
        id: certificate.id,
        scanId: certificate.scanId,
        filename: certificate.filename,
        fileHash: certificate.fileHash,
        verdict: certificate.verdict,
        score: certificate.score,
        timestamp: certificate.timestamp,
        blockchain: certificate.blockchain,
        verificationUrl: blockchainResult?.explorerUrl || null
      }
    });

  } catch (error) {
    console.error('Certificate generation error:', error);
    res.status(500).json({ message: 'Failed to generate certificate', error: error.message });
  }
});

/**
 * GET /api/certificate/:id
 * Get certificate details
 */
router.get('/:id', async (req, res) => {
  try {
    const certificate = await Certificate.findOne({ id: req.params.id });

    if (!certificate) {
      return res.status(404).json({ message: 'Certificate not found' });
    }

    res.json({ certificate });
  } catch (error) {
    console.error('Certificate fetch error:', error);
    res.status(500).json({ message: 'Failed to fetch certificate' });
  }
});

/**
 * GET /api/certificate/verify/:id
 * Verify a certificate (can include on-chain verification)
 */
router.get('/verify/:id', async (req, res) => {
  try {
    const certificate = await Certificate.findOne({ id: req.params.id });

    if (!certificate) {
      return res.status(404).json({ 
        verified: false, 
        message: 'Certificate not found' 
      });
    }

    let onChainVerified = false;
    let onChainData = null;

    // If certificate is on-chain, verify it
    if (certificate.blockchain?.onChain && certificate.blockchain?.transactionHash) {
      try {
        const blockchainAvailable = await blockchainService.initialize();
        if (blockchainAvailable) {
          onChainData = await blockchainService.verifyCertificate(certificate.id);
          onChainVerified = onChainData.exists;
        }
      } catch (error) {
        console.error('On-chain verification failed:', error.message);
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
        onChainData
      }
    });
  } catch (error) {
    console.error('Certificate verification error:', error);
    res.status(500).json({ message: 'Verification failed' });
  }
});

/**
 * GET /api/certificate/user/list
 * Get all certificates for the authenticated user
 */
router.get('/user/list', auth, async (req, res) => {
  try {
    const certificates = await Certificate.find({ userId: req.userId })
      .sort({ timestamp: -1 })
      .limit(50);

    res.json({ certificates });
  } catch (error) {
    console.error('Certificate list error:', error);
    res.status(500).json({ message: 'Failed to fetch certificates' });
  }
});

/**
 * GET /api/certificate/stats
 * Get blockchain statistics
 */
router.get('/stats/overview', auth, async (req, res) => {
  try {
    const blockchainAvailable = await blockchainService.initialize();
    
    let stats = {
      blockchainConnected: blockchainAvailable,
      totalOnChain: 0,
      walletBalance: '0',
      network: 'Not connected'
    };

    if (blockchainAvailable) {
      try {
        stats.totalOnChain = await blockchainService.getTotalCertificates();
        stats.walletBalance = await blockchainService.getBalance();
        stats.network = blockchainService.networkConfig.name;
      } catch (error) {
        console.error('Stats fetch error:', error.message);
      }
    }

    // Get local stats
    const localStats = await Certificate.aggregate([
      { $match: { userId: req.userId } },
      { $group: {
        _id: null,
        total: { $sum: 1 },
        onChain: { $sum: { $cond: ['$blockchain.onChain', 1, 0] } }
      }}
    ]);

    res.json({
      ...stats,
      userCertificates: localStats[0]?.total || 0,
      userOnChain: localStats[0]?.onChain || 0
    });
  } catch (error) {
    console.error('Stats error:', error);
    res.status(500).json({ message: 'Failed to fetch stats' });
  }
});

module.exports = router;
