const express = require('express');
const { auth } = require('../middleware/auth');
const User = require('../models/User');
const DeepfakeScan = require('../models/DeepfakeScan');
const Certificate = require('../models/Certificate');

const router = express.Router();

// GET /api/user/me - Get current user
router.get('/me', auth, async (req, res) => {
  try {
    const user = await User.findById(req.userId);

    if (!user) {
      return res.status(404).json({ message: 'User not found.' });
    }

    res.json({
      user: {
        id: user._id,
        email: user.email,
        role: user.role,
        createdAt: user.createdAt,
        lastLogin: user.lastLogin
      }
    });
  } catch (error) {
    console.error('Get user error:', error);
    res.status(500).json({ message: 'Failed to fetch user.' });
  }
});

// GET /api/user/stats - Get dashboard statistics
router.get('/stats', auth, async (req, res) => {
  try {
    const userId = req.userId;

    // Get scan statistics
    const totalScans = await DeepfakeScan.countDocuments({ userId });
    const recentScans = await DeepfakeScan.countDocuments({
      userId,
      createdAt: { $gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) }
    });

    // Get certificate statistics
    const totalCertificates = await Certificate.countDocuments({ userId });
    const onChainCertificates = await Certificate.countDocuments({
      userId,
      'blockchain.onChain': true
    });

    // Get fraud alerts (suspicious or likely deepfake verdicts)
    const fraudAlerts = await Certificate.countDocuments({
      userId,
      verdict: { $in: ['SUSPICIOUS', 'LIKELY_DEEPFAKE'] }
    });

    // Calculate security score (100 - percentage of suspicious content)
    let securityScore = 100;
    if (totalScans > 0) {
      const suspiciousCount = await DeepfakeScan.countDocuments({
        userId,
        'scanResult.isDeepfake': true
      });
      securityScore = Math.round(100 - (suspiciousCount / totalScans * 100));
    }

    // Get recent activity (last 5 items)
    const recentActivity = await Certificate.find({ userId })
      .sort({ createdAt: -1 })
      .limit(5)
      .select('filename verdict score createdAt blockchain.onChain')
      .lean();

    // Get scan trend (last 7 days)
    const scanTrend = [];
    for (let i = 6; i >= 0; i--) {
      const dayStart = new Date();
      dayStart.setDate(dayStart.getDate() - i);
      dayStart.setHours(0, 0, 0, 0);

      const dayEnd = new Date(dayStart);
      dayEnd.setDate(dayEnd.getDate() + 1);

      const count = await DeepfakeScan.countDocuments({
        userId,
        createdAt: { $gte: dayStart, $lt: dayEnd }
      });

      scanTrend.push({
        date: dayStart.toISOString().split('T')[0],
        count
      });
    }

    res.json({
      stats: {
        totalScans,
        recentScans,
        totalCertificates,
        onChainCertificates,
        fraudAlerts,
        securityScore
      },
      recentActivity: recentActivity.map(item => ({
        id: item._id,
        filename: item.filename,
        verdict: item.verdict,
        score: item.score,
        onChain: item.blockchain?.onChain || false,
        timestamp: item.createdAt
      })),
      scanTrend,
      lastUpdated: new Date().toISOString()
    });
  } catch (error) {
    console.error('Get stats error:', error);
    res.status(500).json({ message: 'Failed to fetch statistics.' });
  }
});

module.exports = router;
